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
  /** the icon beside its name (a picture, when there is one, is `thumb`) */
  icon: string
  /** the card's picture, as ComfyUI's template thumbnails: one image, two that crossfade on hover
   *  ("dissolve"), or a before/after the pointer slides across ("slider"); paths in web/public */
  thumb?: { src: string; after?: string; mode?: 'slider' | 'dissolve'; before_label?: string; after_label?: string }
  /** preferred files per folder; the family's recommended file when not installed */
  files: Partial<Record<Folder, string[]>>
  /** how long it trains: the length it was checked at (a draft's best guess). One length whatever the
   *  dataset's size: with a checkpoint and samples every 250 steps, the best of them is picked
   *  rather than a length guessed from the photo count; Train more carries a short one on */
  steps: number
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
    icon: 'mdi mdi-account-star-outline',
    files: {
      diffusion_models: ['krea2_raw_int8_convrot.safetensors', 'krea2_raw_bf16.safetensors'],
      text_encoders: ['qwen3vl_4b_int8_convrot.safetensors', 'qwen3vl_4b_fp8_scaled.safetensors'],
      vae: ['qwen_image_vae.safetensors'],
    },
    // validated: Automagic v3 at 1e-4 on 37 photos reached likeness 0.53 at 1500 steps and was
    // still climbing, so ~46 a photo (1700 for that set)
    steps: 2000,
    caption: { prompt: NL, subject: true },
    spec: {
      // 768 for the first 60%, blending into 1024 by 80%: identity is learned early, detail last.
      // Not validated with the stage yet (the 0.53 likeness above was 1024 throughout)
      dataset: { resolution: 1024, stages: [{ resolution: 768, until: 0.6 }], stage_blend: 0.2 },
      network: { rank: 16 },
      optim: { optimizer: 'Automagic3', learning_rate: 1e-4, lr_scheduler: 'constant' },
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
    icon: 'mdi mdi-palette-outline',
    files: {
      diffusion_models: ['krea2_raw_int8_convrot.safetensors', 'krea2_raw_bf16.safetensors'],
      text_encoders: ['qwen3vl_4b_int8_convrot.safetensors', 'qwen3vl_4b_fp8_scaled.safetensors'],
      vae: ['qwen_image_vae.safetensors'],
    },
    steps: 2000,
    caption: { prompt: STYLE, subject: false },
    spec: {
      dataset: { resolution: 1024 },
      network: { rank: 32 },
      optim: { optimizer: 'AdamW', learning_rate: 1e-4, lr_scheduler: 'constant' },
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
    icon: 'mdi mdi-account-heart-outline',
    files: {
      diffusion_models: ['moss_image_base_int8_convrot.safetensors', 'moss_image_base_bf16.safetensors'],
      text_encoders: ['qwen3.5_9b_int8_convrot.safetensors'],
      vae: ['flux2-vae.safetensors'],
    },
    // validated: LoKr factor 4 on the MLPs of blocks 24-41 (36 of 176 layers, where a full run's
    // change concentrated) reached likeness 0.63 at 1750 steps on 37 photos, 87% of the set's own
    // 0.72, against ~0.67 for every layer. 0.2 s a step against 0.53, a 170 MB file against 620 MB
    steps: 1750,
    caption: { prompt: NL, subject: true },
    spec: {
      dataset: { resolution: 1024, stages: [{ resolution: 768, until: 0.6 }], stage_blend: 0.2 },
      network: { algorithm: 'LoKr', rank: 4, targets: 'mlp', blocks: '24-41' },
      // Automagic v3 stalls on MOSS without a floor under its rate
      optim: { optimizer: 'Automagic3', learning_rate: 5e-4, lr_scheduler: 'constant', ema: 0.98, lr_floor: 0.5 },
      objective: { timesteps: 'logit_normal', shift: 0, bias: 'high_noise', spread: true },
      // previews on the trained Base model, as the likeness above was measured
      samples: { steps: 25, cfg: 2.5, sampler: 'er_sde', scheduler: 'simple', slg_mode: 'uncond', slg_layers: [22], slg_start: 0 },
    },
    samples: ['[trigger], a close up portrait photo, soft window light', '[trigger] sitting in a cafe, candid photo'],
    validated: true,
  },
  {
    id: 'moss-character-finetunes',
    family: 'moss-image',
    model: 'MOSS Image',
    kind: 'character',
    title: 'MOSS Image · Character for finetunes',
    blurb: 'A person or character that keeps its likeness on MOSS finetunes such as Serenity, not only on Base.',
    photos: character(),
    icon: 'mdi mdi-account-switch-outline',
    files: {
      diffusion_models: ['moss_image_base_int8_convrot.safetensors', 'moss_image_base_bf16.safetensors'],
      text_encoders: ['qwen3.5_9b_int8_convrot.safetensors'],
      vae: ['flux2-vae.safetensors'],
    },
    // draft: a LoRA on every block's MLPs at 3e-4 reached likeness 0.59 on Base at 1500 steps on
    // 37 photos (768 px) and kept 78% of it on Serenity v6 (0.46). The LoKr on blocks 24-41 above
    // reached more on Base (0.66) but kept 61%; the same LoRA on blocks 24-41 only kept 51%, and at
    // 1e-4 every LoRA trailed (0.49-0.52 on Base)
    steps: 1500,
    caption: { prompt: NL, subject: true },
    spec: {
      dataset: { resolution: 768 },
      network: { algorithm: 'LoRA', rank: 16, alpha: 16, targets: 'mlp', blocks: '' },
      // Automagic v3 stalls on MOSS without a floor under its rate
      optim: { optimizer: 'Automagic3', learning_rate: 3e-4, lr_scheduler: 'constant', ema: 0.98, lr_floor: 0.5 },
      objective: { timesteps: 'logit_normal', shift: 0, bias: 'high_noise', spread: true },
      samples: { steps: 25, cfg: 2.5, sampler: 'er_sde', scheduler: 'simple', slg_mode: 'uncond', slg_layers: [22], slg_start: 0 },
    },
    samples: ['[trigger], a close up portrait photo, soft window light', '[trigger] sitting in a cafe, candid photo'],
    note: 'Slower than MOSS Image · Character (every block trains); use that one for Base alone',
  },
  {
    id: 'moss-style',
    family: 'moss-image',
    model: 'MOSS Image',
    kind: 'style',
    title: 'MOSS Image · Style',
    blurb: 'A look: an art style, a palette, a way of drawing. Apply it to anything with the trigger word.',
    photos: style,
    icon: 'mdi mdi-palette-outline',
    files: {
      diffusion_models: ['moss_image_base_int8_convrot.safetensors', 'moss_image_base_bf16.safetensors'],
      text_encoders: ['qwen3.5_9b_int8_convrot.safetensors'],
      vae: ['flux2-vae.safetensors'],
    },
    // draft, not trained yet: two MOSS style LoRAs (ai-toolkit, rank 32, attention and MLPs) cut to
    // block ranges and rendered on fixed seeds kept nothing of their style in blocks 34-41 (0-2% of
    // the full change), where a character's likeness lives, so those blocks stay the base's. The
    // optimizer is the character LoRAs' above: at 1e-4 every MOSS LoRA we trained trailed
    steps: 2000,
    caption: { prompt: STYLE, subject: false },
    spec: {
      dataset: { resolution: 1024 },
      network: { algorithm: 'LoRA', rank: 32, alpha: 32, targets: 'attn_mlp', blocks: '0-33' },
      // Automagic v3 stalls on MOSS without a floor under its rate
      optim: { optimizer: 'Automagic3', learning_rate: 3e-4, lr_scheduler: 'constant', ema: 0.98, lr_floor: 0.5 },
      objective: { timesteps: 'logit_normal', shift: 0 },
      samples: { steps: 25, cfg: 2.5, sampler: 'er_sde', scheduler: 'simple', slg_mode: 'uncond', slg_layers: [22], slg_start: 0 },
    },
    samples: ['[trigger], a lighthouse on a cliff at dusk', '[trigger], a portrait of an old fisherman', '[trigger], a bowl of fruit on a table'],
  },
  {
    id: 'fibo2-character',
    family: 'fibo-2-0',
    model: 'Fibo 2.0',
    kind: 'general',
    title: 'Fibo 2.0 · Any subject',
    blurb: 'A person, a pet, a product or a look, on Bria’s Fibo 2.0 Dev. It learns fast, at a low rank and a gentle learning rate.',
    photos: general,
    icon: 'mdi mdi-account-star-outline',
    files: {
      diffusion_models: ['fibo2_dev_int8_convrot.safetensors'],
      text_encoders: ['qwen3vl_4b_bf16.safetensors', 'qwen3vl_4b_int8_convrot.safetensors', 'qwen3vl_4b_fp8_scaled.safetensors'],
      vae: ['flux2-vae.safetensors'],
    },
    // from a run that trained well: AdamW 3e-5, rank 16, 3000 steps on ~120 photos (~25 a photo).
    // Fibo moves fast: 1e-4 on single samples had rebuilt a 12-image test's whole look in 30 steps
    steps: 2000,
    // Fibo is trained on, and prompted with, structured JSON captions (Bria's VGL format)
    caption: { prompt: NL, subject: true, style: 'fibo_json' },
    spec: {
      dataset: { resolution: 1024, stages: [{ resolution: 768, until: 0.6 }], stage_blend: 0.2 },
      network: { algorithm: 'LoRA', rank: 16, targets: 'attn_mlp' },
      optim: { optimizer: 'AdamW', learning_rate: 3e-5, lr_scheduler: 'constant', warmup_steps: 50, weight_decay: 0, batch_size: 1, grad_accumulation_steps: 1 },
      objective: { timesteps: 'uniform', shift: 0 },
      memory: { int8_forward: true, int8_backward: 'w8a8' },
      samples: { steps: 30, cfg: 5, sampler: 'euler', scheduler: 'simple', negative: '' },
    },
    samples: ['[trigger] in soft window light, a close up photo', '[trigger] outdoors on a sunny afternoon, candid photo, shallow depth of field'],
    note: 'Bria’s own recipe (rank 64, 8 accumulated samples) is the Fibo 2.0 profile in the full form',
  },
  {
    id: 'anima-general',
    family: 'anima',
    model: 'Anima',
    kind: 'general',
    title: 'Anima · Any subject',
    blurb: 'A character, a person or a look on Circlestone Labs’ Anima, an anime model. Its makers’ light touch: a rank 32 LoRA at a low learning rate.',
    photos: general,
    icon: 'mdi mdi-account-star-outline',
    files: {
      diffusion_models: ['anima-base-v1.0.safetensors'],
      text_encoders: ['qwen_3_06b_base.safetensors'],
      vae: ['qwen_image_vae.safetensors'],
    },
    // validated: the makers' advice (huggingface.co/circlestone-labs/Anima, "Finetuning Tips"), rank 32
    // at 2e-5 with the LLM adapter left alone (ZenTrainer never adapts it), on 37 photos at 1024:
    // likeness 0.41 at 500 steps, 0.60 at 1500 and still climbing, clean throughout. 1e-4 had the
    // likeness by 500 but drew rougher
    steps: 1500,
    caption: { prompt: NL, subject: true },
    spec: {
      dataset: { resolution: 1024 },
      network: { algorithm: 'LoRA', rank: 32, alpha: 32, targets: 'attn_mlp', blocks: '' },
      optim: { optimizer: 'AdamW', learning_rate: 2e-5, lr_scheduler: 'constant' },
      // ComfyUI's Anima template: at CFG 4 an empty negative prompt draws noise
      samples: { steps: 30, cfg: 4, sampler: 'euler', scheduler: 'simple',
                 negative: 'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia' },
    },
    samples: ['[trigger], 1girl, solo, close-up portrait, soft window light', '[trigger] sitting in a cafe, candid photo'],
    note: 'An anime model: prompts in its own tag style ("1girl, solo, ...") come out cleanest',
    validated: true,
  },
  {
    id: 'h3-character',
    family: 'minimax-h3',
    model: 'MiniMax H3',
    kind: 'character',
    title: 'MiniMax H3 · Character',
    blurb: 'A person for the H3 video model, trained from photos. Use it in image or video prompts.',
    photos: character(),
    icon: 'mdi mdi-movie-star-outline',
    files: {
      diffusion_models: ['minimax_h3_fl2va_pruned_int8_convrot.safetensors'],
      text_encoders: ['qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'],
      vae: ['minimax_h3_video_vae_fp16.safetensors', 'minimax_h3_video_vae_int8_convrot.safetensors'],
    },
    steps: 2000,
    caption: { prompt: NL, subject: true },
    spec: {
      dataset: { resolution: 1024 },
      network: { rank: 16, blocks: '20-49', training_adapter: 'training_adapters/minimax_h3_image_training_adapter.safetensors' },
      optim: { optimizer: 'AdamW', learning_rate: 1e-4, lr_scheduler: 'constant', batch_size: 1 },
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
/** a checkpoint, with samples, every this many steps: a round 250 that's easy to compare across
 *  runs (4 to 8 of them at a recipe's lengths), 100 for a short run so it still has a few */
export const checkpointEvery = (steps: number) => (steps < 1000 ? 100 : 250)

/** past this many pictures a recipe's length shows each one only a few times: worth saying */
export const BIG_DATASET = 150

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
  o: { name: string; trigger: string; dataset: string; captions?: string[]; masks?: number },
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
  spec.optim.steps = r.steps
  spec.output.save_every = spec.samples.every = checkpointEvery(spec.optim.steps)
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

/** A recipe over a run that's already set up (the run form's recipes): its settings, model files
 *  and length, as Simple mode would; the run's name, dataset, trigger and captions stay as they are. */
export function applyTo(r: Recipe, p: Plan, current: TrainSpec): TrainSpec {
  const spec = JSON.parse(JSON.stringify(current)) as TrainSpec // the form's: parts of it are proxies
  merge(spec as unknown as Record<string, unknown>, r.spec as Record<string, unknown>)
  const d = current.dataset
  Object.assign(spec.dataset, { folder: d.folder, trigger: d.trigger, trigger_prepend: d.trigger_prepend, caption_set: d.caption_set, caption_mix: d.caption_mix })
  spec.name = current.name
  spec.model = {
    ...spec.model,
    kind: 'split',
    ckpt_name: null,
    unet_name: p.model.diffusion_models,
    clip_name: p.model.text_encoders,
    clip_type: p.clipType,
    vae_name: p.model.vae,
  }
  spec.optim.steps = r.steps
  spec.output.save_every = spec.samples.every = checkpointEvery(spec.optim.steps)
  spec.samples.prompts = r.samples
  spec.samples.model_name = p.sampleModel
  if (!p.sampleModel && r.spec.samples?.model_name) {
    spec.samples.steps = 25
    spec.samples.cfg = 4
  }
  return spec
}

export interface Change {
  key: string
  from: unknown
  to: unknown
}

/** What differs between two specs, setting by setting ("optim.learning_rate"), in the spec's order. */
export function changes(a: TrainSpec, b: TrainSpec): Change[] {
  const out: Change[] = []
  const walk = (x: unknown, y: unknown, key: string) => {
    const obj = (v: unknown) => v && typeof v === 'object' && !Array.isArray(v)
    if (obj(x) || obj(y)) {
      const xo = (obj(x) ? x : {}) as Record<string, unknown>
      const yo = (obj(y) ? y : {}) as Record<string, unknown>
      for (const k of new Set([...Object.keys(xo), ...Object.keys(yo)])) walk(xo[k], yo[k], key ? `${key}.${k}` : k)
    } else if (JSON.stringify(x ?? null) !== JSON.stringify(y ?? null)) out.push({ key, from: x, to: y })
  }
  walk(a, b, '')
  return out
}
