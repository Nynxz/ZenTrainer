// Runs, the training queue and run control (engine server/routes/runs.py).

import { getClientId } from '../comfy'
import { call } from './client'
import type { Preset, PresetKind, TrainSpec } from './spec'

export type RunStatus =
  | 'queued'
  | 'loading'
  | 'training'
  | 'paused'
  | 'sampling' // a finished run's checkpoint being sampled (it goes back to its status after)
  | 'done'
  | 'stopped'
  | 'error'
  | 'interrupted'
  | 'rejected'

export interface ShareFile {
  format: 'zentrainer-run'
  left_out: string[]
  version: number
  zentrainer: string
  name: string
  spec: Partial<TrainSpec>
}

export interface ShareImport {
  spec: Partial<TrainSpec>
  name: string
  warnings: string[]
  /** what the file leaves to choose here (the dataset, where LoRAs are saved...) */
  local: string[]
  files: { field: string; folder: string; name: string; installed: boolean | null }[]
}

export const ACTIVE: RunStatus[] = ['queued', 'loading', 'training', 'paused']

export interface RunMeta {
  run_id: string
  name: string
  prompt_id: string | null
  status: RunStatus
  created: number
  updated: number
  final_step: number | null
  message?: string
  /** paused, and let go of ComfyUI for a workflow: Resume continues from its saved state */
  parked?: boolean
  resume_from?: string | null
  /** when this go was handed to ComfyUI (a resume starts a new go); unset on runs from before it was recorded */
  started?: number | null
  first_started?: number | null
  /** when it last stopped, finished or failed; cleared when it's queued again */
  ended?: number | null
}

export interface MetricRow {
  run_id: string
  step: number
  total: number
  loss: number | null
  lr: number | null
  grad_norm?: number | null
  epoch?: number
  elapsed: number
  train_time?: number
  step_time?: number
  it_s: number
  /** the step's peak VRAM (bytes, PyTorch allocations) */
  vram_peak?: number
  /** seconds the step spent in each phase */
  phases?: { data: number; forward: number; backward: number; optimizer: number }
  /** each sample's [percent of the schedule, σ, plain loss] */
  drawn?: [number, number, number][]
}

export interface CheckpointRow {
  run_id: string
  step: number
  path: string
  final: boolean
  lora_name: string
  /** bytes on disk; null if the file is gone */
  size?: number | null
  /** its resumable training state (optimizer, RNG...), if one was kept */
  state?: string | null
}

export interface SampleRow {
  run_id: string
  step: number
  index: number
  prompt: string
  filename: string
  subfolder: string
  type: string
  model?: string | null
  /** a "Try a prompt" render, apart from the run's samples (index counts them) */
  manual?: boolean
  /** a manual render without the LoRA, to compare with the one at the same index */
  base?: boolean
  /** a manual render's sampling (its own, or the run's) */
  sampling?: { sampler: string; scheduler: string; steps: number; cfg: number; sigmas?: number[] }
  seed?: number
  width?: number
  height?: number
  frames?: number
}

export interface RunDetail {
  run: RunMeta
  spec: TrainSpec
  metrics: MetricRow[]
  checkpoints: CheckpointRow[]
  samples: SampleRow[]
  validation: ValidationRow[]
  /** with faces on: each sample's face scored against the dataset's (cosine; same person ≳ 0.5) */
  likeness?: LikenessRow[]
  /** checkpoints output.keep removed, each with why */
  pruned?: (CheckpointRow & { reason: string; at_step: number })[]
  live: boolean
}

export interface LikenessRow {
  run_id: string
  step: number
  index: number
  score: number | null
  manual?: boolean
  base?: boolean
  /** how alike the dataset's own photos are (their median against the others): about the best a
   * score can be (newer runs only) */
  ceiling?: number | null
}

export interface QueueState {
  paused: boolean
  current: RunMeta | null
  items: RunMeta[]
}

export interface ValidationRow {
  run_id: string
  step: number
  val_loss: number
  by_noise: Record<string, number>
  /** measured on a held-out dataset (validation.dataset), not on training pictures */
  held_out?: boolean
}

export type Command = 'pause' | 'resume' | 'stop' | 'save' | 'sample' | 'set_lr'

export const runPath = (id: string) => `/runs/${encodeURIComponent(id)}`

/** a link that saves one of the run's LoRAs (by its lora_name; the final if left out) */
export const downloadUrl = (id: string, lora?: string) =>
  `/zentrainer/api${runPath(id)}/download${lora ? `?lora=${encodeURIComponent(lora)}` : ''}`

/** a link that saves the run's settings as a file to share (engine sharing.py) */
export const exportUrl = (id: string) => `/zentrainer/api${runPath(id)}/export`

export interface SampleProgress {
  run_id: string
  phase: 'requested' | 'sampling' | 'done'
  step?: number
  index?: number
  count?: number
  prompt?: string
  s_step?: number
  s_total?: number
  preview?: string
  /** a "Try a prompt" render (previewed in the Manual tab), its try number, and whether it's the no-LoRA one */
  manual?: boolean
  manual_index?: number
  base?: boolean
}

export const runsApi = {
  runs: () => call<{ runs: RunMeta[] }>('/runs').then((r) => r.runs),
  run: (id: string) => call<RunDetail>(runPath(id)),
  /** position "next" starts it now if idle (else right after the current run); "end" queues it last */
  /** change a run still waiting in the queue, or one that ended before training (it's requeued) */
  editRun: (id: string, spec: TrainSpec) =>
    call<{ run: RunMeta; requeued: boolean }>(`/runs/${encodeURIComponent(id)}/spec`, {
      method: 'PUT',
      body: JSON.stringify({ spec, client_id: getClientId() }),
    }),
  start: (spec: TrainSpec, position: 'next' | 'end' = 'next') =>
    call<{ run: RunMeta }>('/runs', {
      method: 'POST',
      body: JSON.stringify({ spec, position, client_id: getClientId() }),
    }).then((r) => r.run),
  queue: () => call<QueueState>('/queue'),
  editQueue: (body: { action: 'pause' | 'resume' } | { action: 'order'; order: string[] } | { action: 'top' | 'remove' | 'swap'; run_id: string }) =>
    call('/queue', { method: 'POST', body: JSON.stringify(body) }),
  presets: <T = unknown>(kind: PresetKind) => call<{ presets: Preset<T>[] }>(`/presets/${kind}`).then((r) => r.presets),
  savePreset: (kind: PresetKind, name: string, data: unknown, description = '') =>
    call<{ preset: Preset }>(`/presets/${kind}`, { method: 'POST', body: JSON.stringify({ name, data, description }) }),
  /** run settings as a file to share (engine sharing.py): a run's, or the form's */
  exportRun: (id: string) => call<ShareFile>(`${runPath(id)}/export`),
  exportSpec: (spec: TrainSpec) => call<ShareFile>('/share/export', { method: 'POST', body: JSON.stringify({ spec }) }),
  /** a shared file (or a bare spec.json) read for this install: settings to lay over the form */
  importSpec: (doc: unknown) => call<ShareImport>('/share/import', { method: 'POST', body: JSON.stringify({ doc }) }),
  deletePreset: (kind: PresetKind, id: string) => call(`/presets/${kind}/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  control: (id: string, command: Command, args?: Record<string, unknown>) =>
    call<{ ok: true }>(`${runPath(id)}/control`, { method: 'POST', body: JSON.stringify({ command, args }) }),
  /** sample a finished run's checkpoint (default the latest): the run's prompts and/or tries */
  sampleRun: (id: string, body: { checkpoint?: string; run_prompts?: boolean; tries?: Record<string, unknown>[] } = {}) =>
    call<{ run: RunMeta }>(`${runPath(id)}/sample`, { method: 'POST', body: JSON.stringify({ ...body, client_id: getClientId() }) }),
  resume: (id: string, checkpoint?: string) =>
    call<{ run: RunMeta }>(`${runPath(id)}/resume`, {
      method: 'POST',
      body: JSON.stringify({ checkpoint, client_id: getClientId() }),
    }).then((r) => r.run),
  /** train a finished run on to `steps` in all, from its final state (the old final stays as a checkpoint) */
  extend: (id: string, steps: number) =>
    call<{ run: RunMeta }>(`${runPath(id)}/extend`, {
      method: 'POST',
      body: JSON.stringify({ steps, client_id: getClientId() }),
    }).then((r) => r.run),

}
