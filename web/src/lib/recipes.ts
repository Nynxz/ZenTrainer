// Simple mode's recipes: one card per thing people train ("Krea 2 · Character"), each a full
// run setup with nothing left to choose but the photos and a name. Settings come from runs we
// trained and checked (see docs/ARCHITECTURE.md, "Recipes"); the model files are the ones those
// runs used, with the family's recommended file as the fallback when a file isn't installed.
// The recipes themselves are data, one JSON file each in engine/catalog/recipes/: the engine reads
// them (models/recipes.py) and this fetches them as the app starts, so a changed recipe shows on a
// reload, with no build.

import type { TrainSpec } from '@/api/engine'
import { call } from '@/api/engine/client'

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
  /** where its settings come from: the runs and measurements behind them */
  notes?: string[]
}

/** the recipes, filled by loadRecipes() before the app mounts (empty if the engine can't be reached) */
export const RECIPES: Recipe[] = []

/** the engine's recipes (GET /recipes, resolved and in order), once, as the app starts */
export async function loadRecipes(): Promise<void> {
  try {
    const { recipes } = await call<{ recipes: Recipe[] }>('/recipes')
    RECIPES.splice(0, RECIPES.length, ...recipes)
  } catch {
    // the engine unreachable: no recipes, the rest of the app still works
  }
}

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
  o: { name: string; trigger: string; dataset: string; masks?: number },
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
  // the recipe's own prompts: not the dataset's captions, which would go into every LoRA's metadata
  // with the spec
  spec.samples.prompts = r.samples
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
