// This engine's process and machine: health, memory, disk use, ComfyUI's version (engine server/routes/host.py).

import { call } from './client'
import type { RunStatus } from './runs'

export interface MemSample {
  t: number
  ram_total?: number
  ram_used?: number
  proc_rss?: number
  vram_total?: number
  vram_used?: number
  torch_allocated?: number
  torch_reserved?: number
  aimdo_used?: number
}

/** a model ComfyUI holds, and how much of it (per block) is in VRAM */
export interface LoadedModel {
  /** the file it was loaded from (several joined with " + ") */
  file?: string | null
  kind?: 'diffusion model' | 'text encoder' | 'vae'
  name: string
  dynamic?: boolean
  bytes: number
  vram: number
  groups?: { name: string; bytes: number; vram: number }[]
  error?: string
}

export interface StorageReport {
  disks: { labels: string[]; path: string; total: number; used: number; free: number }[]
  runs: { run_id: string; name: string; status: RunStatus; created: number; checkpoints: number; checkpoint_count: number; states: number; samples: number }[]
  datasets: { folder: string; count: number; images: number; cache: number }[]
  apps: number
}

export interface ComfyVersion {
  path: string
  version?: string
  commit?: string
  describe?: string
  dirty?: boolean
  pin?: { ref: string; describe?: string }
  matches_pin?: boolean
  upstream?: { commit: string; describe: string; behind: number }
  upstream_error?: string
}

/** ZenTrainer's own version (engine server/routes/common.py _zt_version) */
export interface ZtVersion {
  version: string
  /** the commit this process loaded; none outside a git checkout */
  commit?: string
  describe?: string
  /** null: not on a branch (detached at a commit or tag) */
  branch?: string | null
  /** ZenTrainer's tracked files have local changes: `zt update` skips the update */
  dirty?: boolean
  /** the checkout moved on since this process started: a restart loads it */
  on_disk?: { commit: string; describe: string }
  /** with ?check=1: the commits upstream has and this checkout lacks (newest first, at most 12) */
  upstream?: { ref: string; describe: string; behind: number; ahead: number; commits: string[] }
  upstream_error?: string
}

export interface OptimizerInfo {
  id: string
  label: string
  available: boolean
  pip: string | null
  lr_hint: string
  note: string
}

export interface Health {
  version: string
  /** the engine's code on disk is newer than what's running: restart to load it */
  stale?: boolean
  capabilities: {
    optimizers?: OptimizerInfo[]
    train_attention?: string[]
    infer_attention?: string[]
    device?: string
    device_name?: string
    vram_total?: number
    vram_free?: number
  }
  /** ComfyUI is older than ZenTrainer needs: the message says what to do */
  comfy_old?: string | null
  /** when web/dist was built (changes when a rebuild lands) */
  web_built?: number | null
  /** the web source on disk is newer than the build (a pull: dist isn't in git) */
  web_stale?: boolean
  /** pnpm is here, so the engine can rebuild it */
  web_buildable?: boolean
}

/** ZenTrainer's own settings (engine paths.py): saved ones, and the folders this process uses */
export interface EngineSettings {
  /** folder: the ZenTrainer folder ("" = ComfyUI's input/output); datasets: where datasets live ("" = the
   *  folder's datasets/ or ComfyUI's input; relative = inside input, e.g. "datasets"; or an absolute path);
   *  face_model: the face recognition model */
  settings: { folder: string; datasets: string; face_model: string }
  active: { folder: string; datasets: string; output: string; loras: string }
  /** the saved folders differ from the ones in use: they apply after a ComfyUI restart */
  restart_needed: boolean
  /** datasets the saved datasets folder won't show: still where they were, to move by hand */
  datasets_elsewhere: string[]
  /** the face recognition models on offer (engine analysis/faces.py MODELS) */
  face_models: { id: string; label: string; note: string; downloaded: boolean }[]
  message?: string
}

export const hostApi = {
  health: () => call<Health>('/health'),
  system: (since = 0) => call<{ interval: number; samples: MemSample[] }>(`/system?since=${since}`),
  storage: () => call<StorageReport>('/storage'),
  disks: () => call<Pick<StorageReport, 'disks'>>('/storage?disks=1').then((r) => r.disks),
  cleanStorage: (body: { kind: 'states'; run_id: string } | { kind: 'states_all' } | { kind: 'cache'; folder: string }) =>
    call<{ freed: number; runs?: number }>('/storage/clean', { method: 'POST', body: JSON.stringify(body) }),
  loadedModels: () => call<{ models: LoadedModel[] }>('/system/models').then((r) => r.models),
  version: (check = false) => call<ZtVersion>(`/version${check ? '?check=1' : ''}`),
  comfyui: (check = false) => call<ComfyVersion>(`/comfyui${check ? '?check=1' : ''}`),
  buildWeb: () => call<Pick<Health, 'web_built' | 'web_stale'>>('/web/build', { method: 'POST' }),
  settings: () => call<EngineSettings>('/settings'),
  putSettings: (body: Partial<EngineSettings['settings']>) => call<EngineSettings>('/settings', { method: 'PUT', body: JSON.stringify(body) }),
  restart: (force = false) => call<{ ok: true }>('/restart', { method: 'POST', body: JSON.stringify({ force }) }),
}
