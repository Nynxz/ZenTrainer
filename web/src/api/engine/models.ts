// Model families, downloads, training adapters, noise schedules, LoRA merging, inspecting
// .safetensors files (engine server/routes/models.py, loras.py).

import { call } from './client'
import type { TrainSpec } from './spec'

export type ModelFolder = 'diffusion_models' | 'text_encoders' | 'vae' | 'checkpoints'

export interface ModelFile {
  name: string
  folder: ModelFolder
  url: string
  size: number | null
  distilled: boolean
  from_template: boolean
  /** the name ComfyUI's loaders list it by; null when not installed */
  installed: string | null
}

export type TensorFolder = 'loras' | 'diffusion_models' | 'checkpoints' | 'vae' | 'text_encoders'

export interface TensorRow {
  name: string
  dtype: string
  shape: number[]
  params: number
}

/** a .safetensors file read from its header (engine/zentrainer/models/tensors.py) */
export interface TensorFileInfo {
  size: number
  params: number
  by_dtype: Record<string, { tensors: number; params: number; bytes: number }>
  metadata: Record<string, string>
  tensors: TensorRow[]
  kind: 'lora' | 'model'
  algorithm?: 'LoRA' | 'LoKr' | 'LoHa'
  layers?: number
  ranks?: number[]
  quantized: boolean
}

export interface TensorCopy {
  folder: TensorFolder
  name: string
  out_name: string
  /** replaces the file's metadata; leave out to keep it */
  metadata?: Record<string, string>
  dtype?: 'bf16' | 'fp16' | 'fp32' | null
  /** tensor names, or prefixes everything under is left out */
  drop?: string[]
  overwrite?: boolean
}

export interface CompactRequest {
  name: string
  /** same: keep the file's kind; lora: LoKr layers become LoRA */
  format?: 'same' | 'lora'
  base?: { folder: 'diffusion_models' | 'checkpoints'; name: string } | null
  max_rank?: number
  target?: { keep: number } | { params: number }
}

export interface CompactPlan {
  params: number
  kept: number
  layers: number
  layers_kept: number
  mean_rank: number
  max_rank: number
  ranks: number[]
}

export interface CompactAnalysis {
  /** precision: the file's float precision by bytes (null when it's something else) */
  original: { params: number; bytes: number; precision: 'bf16' | 'fp16' | 'fp32' | null }
  /** the offered shares, each with what it keeps and costs */
  levels: { keep: number; params: number; kept: number }[]
  plan: CompactPlan
  names: string[]
  kind: ('lora' | 'lokr' | 'lokr_lora')[]
  full_ranks: number[]
  unsupported: string[]
  base_matched: number | null
}

export interface SampleDefaults {
  model: string
  steps: number
  cfg: number
  sampler: string
  scheduler: string
}

export interface ModelFamily {
  id: string
  name: string
  kind: 'split' | 'checkpoint'
  clip_type: string | null
  repos: string[]
  templates: string[]
  files: Partial<Record<ModelFolder, ModelFile[]>>
  sample_defaults: SampleDefaults | Record<string, never>
  repo_listed: boolean
  /** templates: from ComfyUI's workflow templates; repo: a Comfy-Org or maker's ComfyUI repo with no template yet */
  source: 'templates' | 'repo'
  /** hand-picked settings for this model, when we have them */
  profile: ModelProfile | null
  /** ISO date: first template, or repo creation */
  added: string | null
  /** listed under "Other models": few people train it (engine/catalog/families.json) */
  other?: boolean
  /** trained end to end here (families.json's `tested`); false: listed as untested. An older
   *  engine sends neither */
  tested?: boolean
  /** the repo asks you to accept a license (and set HF_TOKEN) before downloading */
  gated: boolean
  recommended: { train: Partial<Record<ModelFolder, string | null>>; sample: SampleDefaults | null }
  missing: { folder: ModelFolder; name: string; size: number | null }[]
}

/** engine/zentrainer/models/noise.py profile(): σ from full noise (first) to clean (last), evenly spaced */
export interface NoiseProfile {
  known: boolean
  model?: string
  flow?: boolean
  shift?: number | null
  sigmas?: number[]
  /** each ComfyUI scheduler's σ at 200 steps, full noise first */
  schedules?: Record<string, number[]>
  folder?: 'diffusion_models' | 'checkpoints'
  name?: string
}

/** training: on while training only; base: on always (training, samples, use) */
export type AdapterKind = 'training' | 'base'

/** engine/zentrainer/models/adapters.py listing() */
export interface TrainingAdapter {
  id: string
  label: string
  /** local file name */
  name: string
  /** the name ComfyUI lists it by when installed (what the spec stores), else null */
  installed: string | null
  source: 'curated' | 'custom' | 'local'
  /** suits the family asked about */
  fits: boolean
  /** worth offering when the family is picked */
  suggest: boolean
  families: string[]
  note: string
  repo: string
  file: string
  size: number | null
  /** the repo page */
  url: string
}

export interface ModelProfile {
  id: string
  families: string[]
  /** recipe: the model makers' published settings; verified: we've trained with it and checked */
  status: 'recipe' | 'verified'
  summary: string
  sources: { label: string; url: string }[]
  train: Record<string, unknown>
  samples: Partial<TrainSpec['samples']>
  notes: string[]
}

export interface ModelDownload {
  id: string
  family: string
  folder: ModelFolder | 'loras'
  name: string
  target: string
  total: number | null
  done: number
  status: 'queued' | 'downloading' | 'done' | 'error' | 'cancelled'
  error: string | null
  speed: number
  created: number
}

export const modelsApi = {
  families: () => call<{ families: ModelFamily[] }>('/families').then((r) => r.families),
  /** re-list the model makers' Hugging Face repos now: models added since the cached listing */
  relistFamilies: () => call<{ families: number; new: string[] }>('/families/relist', { method: 'POST' }),
  family: (id: string, repo = false, refresh = false) =>
    call<ModelFamily>(`/families/${encodeURIComponent(id)}?repo=${repo ? 1 : 0}${refresh ? '&refresh=1' : ''}`),
  downloads: () =>
    call<{ downloads: ModelDownload[]; targets: Partial<Record<ModelFolder, string>>; hf_token: boolean }>('/downloads'),
  download: (family: string, folder: ModelFolder, name: string) =>
    call<ModelDownload>('/downloads', { method: 'POST', body: JSON.stringify({ family, folder, name }) }),
  cancelDownload: (id: string) => call<{ cancelled: boolean }>(`/downloads/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  clearDownloads: () => call<{ ok: true }>('/downloads/clear', { method: 'POST' }),
  mergeLoras: (body: { members: { name: string; strength: number }[]; out_name: string; max_rank?: number; energy?: number }) =>
    call<{ lora_name: string; mean_rank?: number; kept_energy?: number; bytes: number }>('/loras/merge', { method: 'POST', body: JSON.stringify(body) }),
  /** what compacting a LoRA to a target keeps, layer by layer (engine/zentrainer/loras/compact.py) */
  analyzeCompact: (body: CompactRequest) => call<CompactAnalysis>('/loras/compact/analyze', { method: 'POST', body: JSON.stringify(body) }),
  compactLora: (body: CompactRequest & { out_name: string; dtype?: 'bf16' | 'fp16' | 'fp32'; overwrite?: boolean }) =>
    call<Omit<CompactPlan, 'ranks'> & { lora_name: string; bytes: number }>('/loras/compact', { method: 'POST', body: JSON.stringify(body) }),
  tensorFiles: () => call<{ folders: Partial<Record<TensorFolder, string[]>> }>('/tensors/files').then((r) => r.folders),
  inspectTensors: (folder: TensorFolder, name: string) => call<TensorFileInfo>(`/tensors?${new URLSearchParams({ folder, name })}`),
  /** what the copy would hold, without writing it */
  planTensorCopy: (body: TensorCopy) =>
    call<{ tensors: number; bytes: number; converted: number; dropped: number }>('/tensors/copy', { method: 'POST', body: JSON.stringify({ ...body, dry_run: true }) }),
  copyTensors: (body: TensorCopy) =>
    call<{ folder: TensorFolder; name: string; bytes: number; tensors: number; converted: number; dropped: number }>('/tensors/copy', { method: 'POST', body: JSON.stringify(body) }),
  /** a model's own noise schedule from its file's header (engine/zentrainer/models/noise.py) */
  noise: (folder: 'diffusion_models' | 'checkpoints', name: string) =>
    call<NoiseProfile>(`/noise?${new URLSearchParams({ folder, name })}`),
  /** one ComfyUI scheduler's exact σ on a model, as training uses them */
  noiseSchedule: (folder: 'diffusion_models' | 'checkpoints', name: string, scheduler: string, steps: number) =>
    call<{ sigmas: number[] }>(`/noise/schedule?${new URLSearchParams({ folder, name, scheduler, steps: String(steps) })}`),
  adapters: (family?: string | null, kind: AdapterKind = 'training') =>
    call<{ adapters: TrainingAdapter[] }>(
      `/adapters?kind=${kind}${family ? `&family=${encodeURIComponent(family)}` : ''}`,
    ).then((r) => r.adapters),
  addAdapter: (link: string, family?: string | null, file?: string, kind: AdapterKind = 'training') =>
    call<{ adapter?: TrainingAdapter; choices?: { file: string; size: number | null }[]; repo?: string }>('/adapters', {
      method: 'POST',
      body: JSON.stringify({ link, family, file, kind }),
    }),
  downloadAdapter: (id: string) => call<ModelDownload>('/adapters/download', { method: 'POST', body: JSON.stringify({ id }) }),
  removeAdapter: (id: string) => call<{ removed: boolean }>(`/adapters?id=${encodeURIComponent(id)}`, { method: 'DELETE' }),
  loraPlaces: () =>
    call<{ loras: { path: string; free: number; total: number; default: boolean; inside_install: boolean }[] }>('/storage/places').then((r) => r.loras),
}
