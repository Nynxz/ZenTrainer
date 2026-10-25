// Simple mode's recipes: one card per thing people train ("Krea 2 · Character"), each a full
// run setup with nothing left to choose but the photos and a name. Settings come from runs we
// trained and checked (see docs/ARCHITECTURE.md, "Recipes"); the model files are the ones those
// runs used, with the family's recommended file as the fallback when a file isn't installed.

import type { TrainSpec } from '@/api/engine'

type DeepPartial<T> = { [K in keyof T]?: NonNullable<T[K]> extends unknown[] ? T[K] : NonNullable<T[K]> extends object ? DeepPartial<NonNullable<T[K]>> : T[K] }
type Folder = 'diffusion_models' | 'text_encoders' | 'vae'

export interface Recipe {
  id: string
  /** model family id (engine/zentrainer/models/families.py) */
  family: string
  model: string
  kind: 'character' | 'style' | 'video' | 'general'
  title: string
  blurb: string
  /** what to put in: shown beside the drop zone */
  photos: { min: number; ideal: string; tips: string[] }
  /** the card's accent and icon */
  cover: { from: string; to: string; icon: string }
  /** the card's picture, as ComfyUI's template thumbnails: one image, two that crossfade on hover
   *  ("dissolve"), or a before/after the pointer slides across ("slider"); paths in web/public */
  thumb?: { src: string; after?: string; mode?: 'slider' | 'dissolve'; before_label?: string; after_label?: string }
  /** preferred files per folder; the family's recommended file when not installed */
  files: Partial<Record<Folder, string[]>>
  /** steps from the number of photos: perPhoto × photos, clamped */
  steps: { perPhoto: number; min: number; max: number }
  /** what auto-caption is asked; `subject` names the trigger as the main subject (characters) */
  /** what auto-caption is asked (`style`: a built-in caption style instead, e.g. Fibo's structured JSON) */
  caption: { prompt: string; subject: boolean; style?: string }
  spec: DeepPartial<TrainSpec>
  /** sample prompts; [trigger] is the LoRA's word */
  samples: string[]
  /** a hint shown on the card */
  note?: string
  /** the recipe has been trained end to end and its samples checked (else it's a draft) */
  validated?: boolean
}

const NL = 'Describe this image for training an image model: the subject, their appearance, clothing and pose, the setting, lighting and photographic or artistic style. One detailed paragraph. Start directly with the description; no preamble.'
const STYLE =
  'Describe what this image shows (subjects, objects, setting, composition, colours) in one paragraph, without naming or describing its artistic style, medium or technique. Start directly; no preamble.'

const character = (who = 'person') => ({
  min: 8,
  ideal: '15–40 photos',
  tips: [`Only one ${who}, clearly visible`, 'Different outfits, places and lighting', 'A few close-ups and a few full-body shots', 'Sharp photos: skip blurry or tiny ones'],
})
const general = {
  min: 10,
  ideal: '15–50 images',
  tips: ['One subject or one look across them all', 'Varied backgrounds, angles and light', 'Sharp, well-lit pictures', 'No text or watermarks if you can help it'],
}
const style = {
  min: 10,
  ideal: '20–60 images',
  tips: ['All in the one style you want', 'Different subjects and scenes', 'No text or watermarks if you can help it'],
}

export const RECIPES: Recipe[] = [
  {
    id: 'krea2-character',
    family: 'krea-2',
    model: 'Krea 2',
    kind: 'character',
    title: 'Krea 2 · Character',
    blurb: 'A person or character, photoreal. Previews on Krea 2 Turbo in 8 steps.',
    photos: character(),
    cover: { from: '#f97316', to: '#7c2d12', icon: 'mdi mdi-account-star-outline' },
    // placeholder: the Krea 2 template's thumbnail, until a validated run gives us before/after
    thumb: { src: '/templates/image_krea2_turbo_t2i-1.webp' },
    files: {
      diffusion_models: ['krea2_raw_int8_convrot.safetensors', 'krea2_raw_bf16.safetensors'],
      text_encoders: ['qwen3vl_4b_int8_convrot.safetensors', 'qwen3vl_4b_fp8_scaled.safetensors'],
      vae: ['qwen_image_vae.safetensors'],
    },
    // validated: Automagic v3 at 1e-4 on 37 photos reached likeness 0.53 at 1500 steps and was
    // still climbing, so ~46 a photo (1700 for that set)
    steps: { perPhoto: 46, min: 1000, max: 2000 },
    caption: { prompt: NL, subject: true },
    spec: {
      // 768 for the first 60%, blending into 1024 by 80%: identity is learned early, detail last.
      // Not validated with the stage yet (the 0.53 likeness above was 1024 throughout)
      dataset: { resolution: 1024, stages: [{ resolution: 768, until: 0.6 }], stage_blend: 0.2 },
      network: { rank: 16 },
      optim: { optimizer: 'Automagic3', learning_rate: 1e-4, scheduler: 'constant' },
      objective: { timesteps: 'uniform', shift: 0 },
      samples: { steps: 8, cfg: 1, sampler: 'euler', scheduler: 'simple', model_name: 'krea2_turbo_int8_convrot.safetensors' },
    },
    samples: ['[trigger], a close up portrait photo, soft window light', '[trigger] walking down a city street at night, candid photo, neon signs'],
  },
  {
    id: 'krea2-style',
    family: 'krea-2',
    model: 'Krea 2',
    kind: 'style',
    title: 'Krea 2 · Style',
    blurb: 'A look: an art style, a film stock, a mood. Apply it to anything with the trigger word.',
    photos: style,
    cover: { from: '#a855f7', to: '#1e1b4b', icon: 'mdi mdi-palette-outline' },
    files: {
      diffusion_models: ['krea2_raw_int8_convrot.safetensors', 'krea2_raw_bf16.safetensors'],
      text_encoders: ['qwen3vl_4b_int8_convrot.safetensors', 'qwen3vl_4b_fp8_scaled.safetensors'],
      vae: ['qwen_image_vae.safetensors'],
    },
    steps: { perPhoto: 30, min: 1000, max: 2000 },
    caption: { prompt: STYLE, subject: false },
    spec: {
      dataset: { resolution: 1024 },
      network: { rank: 32 },
      optim: { optimizer: 'AdamW', learning_rate: 1e-4, scheduler: 'constant' },
      objective: { timesteps: 'uniform', shift: 0 },
      samples: { steps: 8, cfg: 1, sampler: 'euler', scheduler: 'simple', model_name: 'krea2_turbo_int8_convrot.safetensors' },
    },
    samples: ['[trigger], a lighthouse on a cliff at dusk', '[trigger], a portrait of an old fisherman', '[trigger], a bowl of fruit on a table'],
  },
  {
    id: 'moss-character',
    family: 'moss-image',
    model: 'MOSS Image',
    kind: 'character',
    title: 'MOSS Image · Character',
    blurb: 'A person or character on MOSS Image Base. Trains only where a likeness lives: over twice as fast.',
    photos: character(),
    cover: { from: '#22c55e', to: '#064e3b', icon: 'mdi mdi-account-heart-outline' },
    files: {
      diffusion_models: ['moss_image_base_int8_convrot.safetensors', 'moss_image_base_bf16.safetensors'],
      text_encoders: ['qwen3.5_9b_int8_convrot.safetensors'],
      vae: ['flux2-vae.safetensors'],
    },
    // validated: LoKr factor 4 on the MLPs of blocks 24-41 (36 of 176 layers, where a full run's
    // change concentrated) reached likeness 0.63 at 1750 steps on 37 photos, 87% of the set's own
    // 0.72, against ~0.67 for every layer. 0.2 s a step against 0.53, a 170 MB file against 620 MB
    steps: { perPhoto: 50, min: 1250, max: 2000 },
    caption: { prompt: NL, subject: true },
    spec: {
      dataset: { resolution: 1024, stages: [{ resolution: 768, until: 0.6 }], stage_blend: 0.2 },
      network: { algorithm: 'LoKr', rank: 4, targets: 'mlp', blocks: '24-41' },
      // Automagic v3 stalls on MOSS without a floor under its rate
      optim: { optimizer: 'Automagic3', learning_rate: 5e-4, scheduler: 'constant', ema: 0.98, lr_floor: 0.5 },
      objective: { timesteps: 'logit_normal', shift: 0, bias: 'high_noise', spread: true },
      memory: { checkpoint_skip: 4 },
      // previews on the trained Base model, as the likeness above was measured
      samples: { steps: 25, cfg: 2.5, sampler: 'er_sde', scheduler: 'simple', slg_mode: 'uncond', slg_layers: [22], slg_start: 0 },
    },
    samples: ['[trigger], a close up portrait photo, soft window light', '[trigger] sitting in a cafe, candid photo'],
    validated: true,
  },
  {
    id: 'fibo2-character',
    family: 'fibo-2-0',
    model: 'Fibo 2.0',
    kind: 'general',
    title: 'Fibo 2.0 · Any subject',
    blurb: 'A person, a pet, a product or a look, on Bria’s Fibo 2.0 Dev. It learns fast, at a low rank and a gentle learning rate.',
    photos: general,
    cover: { from: '#6366f1', to: '#1e1b4b', icon: 'mdi mdi-account-star-outline' },
    files: {
      diffusion_models: ['fibo2_dev_int8_convrot.safetensors'],
      text_encoders: ['qwen3vl_4b_bf16.safetensors', 'qwen3vl_4b_int8_convrot.safetensors', 'qwen3vl_4b_fp8_scaled.safetensors'],
      vae: ['flux2-vae.safetensors'],
    },
    // from a run that trained well: AdamW 3e-5, rank 16, 3000 steps on ~120 photos (~25 a photo).
    // Fibo moves fast: 1e-4 on single samples had rebuilt a 12-image test's whole look in 30 steps
    steps: { perPhoto: 25, min: 1500, max: 3000 },
    // Fibo is trained on, and prompted with, structured JSON captions (Bria's VGL format)
    caption: { prompt: NL, subject: true, style: 'fibo_json' },
    spec: {
      dataset: { resolution: 1024, stages: [{ resolution: 768, until: 0.6 }], stage_blend: 0.2 },
      network: { algorithm: 'LoRA', rank: 16, targets: 'attn_mlp' },
      optim: { optimizer: 'AdamW', learning_rate: 3e-5, scheduler: 'constant', warmup_steps: 50, weight_decay: 0, batch_size: 1, grad_accumulation_steps: 1 },
      objective: { timesteps: 'uniform', shift: 0 },
      memory: { int8_forward: true, w8a8_backward: true, compile_blocks: true },
      samples: { steps: 30, cfg: 5, sampler: 'euler', scheduler: 'simple', negative: '' },
    },
    samples: ['[trigger] in soft window light, a close up photo', '[trigger] outdoors on a sunny afternoon, candid photo, shallow depth of field'],
    note: 'Bria’s own recipe (rank 64, 8 accumulated samples) is the Fibo 2.0 profile in the full form',
  },
  {
    id: 'h3-character',
    family: 'minimax-h3',
    model: 'MiniMax H3',
    kind: 'character',
    title: 'MiniMax H3 · Character',
    blurb: 'A person for the H3 video model, trained from photos. Use it in image or video prompts.',
    photos: character(),
    cover: { from: '#0ea5e9', to: '#0c4a6e', icon: 'mdi mdi-movie-star-outline' },
    thumb: { src: '/templates/video_minimax_h3_t2v-1.webp' },
    files: {
      diffusion_models: ['minimax_h3_fl2va_pruned_int8_convrot.safetensors'],
      text_encoders: ['qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'],
      vae: ['minimax_h3_video_vae_fp16.safetensors', 'minimax_h3_video_vae_int8_convrot.safetensors'],
    },
    steps: { perPhoto: 50, min: 1200, max: 2000 },
    caption: { prompt: NL, subject: true },
    spec: {
      dataset: { resolution: 1024 },
      network: { rank: 16, blocks: '20-49', training_adapter: 'training_adapters/minimax_h3_image_training_adapter.safetensors' },
      optim: { optimizer: 'AdamW', learning_rate: 1e-4, scheduler: 'constant', batch_size: 1 },
      objective: { timesteps: 'logit_normal', shift: 3 },
      memory: { offloading: true },
      samples: { steps: 30, cfg: 1, sampler: 'res_multistep', scheduler: 'simple' },
    },
    samples: ['[trigger], a close up portrait photo, soft window light', '[trigger] on a city street at night, candid photo, neon signs'],
    note: 'Needs the circlestone image training adapter (downloaded for you)',
  },
]

export const recipe = (id: string) => RECIPES.find((r) => r.id === id) ?? null

/** steps for this many photos */
export const stepsFor = (r: Recipe, photos: number) => Math.max(r.steps.min, Math.min(r.steps.max, Math.round((photos * r.steps.perPhoto) / 50) * 50))

/** "about 35 min" */
export function eta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return ''
  if (seconds < 90) return 'about a minute'
  const m = Math.round(seconds / 60)
  if (m < 60) return `about ${m} min`
  const h = Math.floor(m / 60)
  const r = Math.round((m % 60) / 5) * 5
  return r ? `about ${h} h ${r} min` : `about ${h} h`
}

/** a trigger word from a name: "Grandma Rose" -> "grandmarose" */
export const triggerFrom = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, '').slice(0, 24) || 'mylora'

// ---- turning a recipe into a run ----------------------------------------------------------

export interface Plan {
  /** files to train with, by folder (the names ComfyUI lists them by, or to-download names) */
  model: Record<Folder, string | null>
  clipType: string | null
  /** files to fetch first: [folder, name, size] (catalog downloads) */
  downloads: { folder: Folder; name: string; size: number | null }[]
  /** training adapter to fetch first (adapters.py catalog id), when the recipe uses one not installed */
  adapter: { id: string; name: string; size: number | null } | null
  /** the sampling model the recipe previews on, if installed (else samples use the trained model) */
  sampleModel: string | null
  problems: string[]
}

/** Which files a recipe trains with on this install, and what must be downloaded first. */
export async function plan(r: Recipe): Promise<Plan> {
  const { engine } = await import('@/api/engine')
  const fam = await engine.family(r.family, true)
  const out: Plan = { model: { diffusion_models: null, text_encoders: null, vae: null }, clipType: fam.clip_type, downloads: [], adapter: null, sampleModel: null, problems: [] }
  for (const folder of ['diffusion_models', 'text_encoders', 'vae'] as Folder[]) {
    const files = fam.files[folder] ?? []
    const want = r.files[folder] ?? []
    const have = want.map((n) => files.find((f) => f.name === n && f.installed)).find(Boolean)
    if (have) {
      out.model[folder] = have.installed
      continue
    }
    const rec = fam.recommended.train[folder]
    const pick = files.find((f) => f.name === want[0]) ?? files.find((f) => f.name === rec)
    if (!pick) {
      out.problems.push(`no ${folder.replace('_', ' ')} file known for ${r.model}`)
      continue
    }
    out.model[folder] = pick.installed ?? pick.name
    if (!pick.installed) out.downloads.push({ folder, name: pick.name, size: pick.size })
  }
  const sm = r.spec.samples?.model_name
  if (sm) {
    const f = (fam.files.diffusion_models ?? []).find((x) => x.name === sm && x.installed)
    out.sampleModel = f?.installed ?? null
  }
  const ta = r.spec.network?.training_adapter
  if (ta) {
    const list = await engine.adapters(r.family)
    const a = list.find((x) => x.installed === ta || x.name === ta.split('/').pop())
    if (!a?.installed) out.adapter = a ? { id: a.id, name: a.label, size: a.size } : null
  }
  return out
}

function merge(target: Record<string, unknown>, src: Record<string, unknown>) {
  for (const [k, v] of Object.entries(src)) {
    if (v && typeof v === 'object' && !Array.isArray(v) && target[k] && typeof target[k] === 'object')
      merge(target[k] as Record<string, unknown>, v as Record<string, unknown>)
    else target[k] = structuredClone(v)
  }
}

/** The full run spec: defaults, the recipe's settings, these files, this dataset and name. */
export function buildSpec(
  r: Recipe,
  p: Plan,
  base: TrainSpec,
  o: { name: string; trigger: string; dataset: string; photos: number; captions?: string[]; masks?: number },
): TrainSpec {
  const spec = structuredClone(base)
  merge(spec as unknown as Record<string, unknown>, r.spec as Record<string, unknown>)
  spec.name = o.name
  spec.model = {
    ...spec.model,
    kind: 'split',
    ckpt_name: null,
    unet_name: p.model.diffusion_models,
    clip_name: p.model.text_encoders,
    clip_type: p.clipType,
    vae_name: p.model.vae,
  }
  spec.dataset.folder = o.dataset
  spec.dataset.trigger = o.trigger
  spec.dataset.trigger_prepend = true
  spec.dataset.caption_set = 'default'
  // a mask says "only this part": someone drew it (or cut a face out), so honour it
  spec.dataset.masked_loss = (o.masks ?? 0) > 0
  spec.optim.steps = stepsFor(r, o.photos)
  // about 8 checkpoints, each with samples, to pick the best from
  spec.save_every = Math.max(100, Math.round(spec.optim.steps / 8 / 50) * 50)
  spec.samples.every = spec.save_every
  // the recipe's first prompt (somewhere new), plus two of the dataset's own captions: the
  // closest check that it learned what's in the photos
  spec.samples.prompts = [r.samples[0]!, ...captionPrompts(o.captions ?? [], o.trigger, r.kind)].slice(0, 3)
  if (spec.samples.prompts.length < 2) spec.samples.prompts = r.samples
  spec.samples.baseline = true
  spec.samples.model_name = p.sampleModel
  if (!p.sampleModel && r.spec.samples?.model_name) {
    // previewing on the trained (raw) model instead: its own steps and guidance
    spec.samples.steps = 25
    spec.samples.cfg = 4
  }
  spec.validation.every = 0
  return spec
}

/** Two captions as sample prompts: shortish ones, spread through the set, with the trigger. */
export function captionPrompts(captions: string[], trigger: string, kind: Recipe['kind']): string[] {
  const usable = captions.map((c) => c.trim()).filter((c) => c.length > 20)
  if (!usable.length) return []
  const pick = [usable[Math.floor(usable.length / 3)]!, usable[Math.floor((usable.length * 2) / 3)]!]
  return [...new Set(pick)].map((c) => {
    const short = c.length > 300 ? `${c.slice(0, 300).replace(/[^.,;]*$/, '').replace(/[,;\s]+$/, '')}` : c
    const has = trigger && short.toLowerCase().includes(trigger.toLowerCase())
    void kind
    return has ? short : `[trigger], ${short}`
  })
}
