// The fetch helper every area's calls go through, and what they share: errors, image URLs, websocket events.

export class EngineError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: unknown,
  ) {
    super(message)
  }
}

export async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/zentrainer/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new EngineError(body?.error ?? res.statusText, res.status, body)
  return body as T
}

/** A file by its ComfyUI /view address (filename, subfolder, input/output type), served by the engine:
 *  /view only knows ComfyUI's own folders, and datasets and results can live in the ZenTrainer folder. */
export function viewUrl(f: { filename: string; subfolder: string; type?: string }): string {
  const q = new URLSearchParams({ filename: f.filename, subfolder: f.subfolder, type: f.type ?? 'output' })
  return `/zentrainer/api/files/view?${q}`
}

/** Websocket event names; mirror engine/zentrainer/events.py. */
export const EVENTS = {
  metrics: 'zentrainer.metrics',
  checkpoint: 'zentrainer.checkpoint',
  status: 'zentrainer.status',
  sample: 'zentrainer.sample',
  sampleProgress: 'zentrainer.sample_progress',
  validation: 'zentrainer.validation',
  likeness: 'zentrainer.likeness',
  caption: 'zentrainer.caption',
  download: 'zentrainer.download',
  faces: 'zentrainer.faces',
  queue: 'zentrainer.queue',
  system: 'zentrainer.system',
  loaded: 'zentrainer.loaded',
} as const
