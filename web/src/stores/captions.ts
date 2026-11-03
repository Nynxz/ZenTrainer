// Caption jobs, kept live from the engine's websocket events: the sidebar shows what's
// running from any page, and a dataset page opened mid-job picks its job back up.

import { computed, reactive } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { EVENTS, engine, type CaptionJob } from '@/api/engine'

export const captionJobs = reactive<Record<string, CaptionJob>>({})

export const activeCaptions = computed(() =>
  Object.values(captionJobs)
    .filter((j) => j.status === 'queued' || j.status === 'running')
    .sort((a, b) => (a.created ?? 0) - (b.created ?? 0)),
)

function put(j: CaptionJob) {
  captionJobs[j.job_id] = { ...captionJobs[j.job_id], ...j }
}

let started = false
export async function startCaptionSync(): Promise<void> {
  if (started) return
  started = true
  onComfyMessage((m) => {
    if (m.type === EVENTS.caption) put(m.data as unknown as CaptionJob)
  })
  try {
    for (const j of await engine.captionActive()) put(j)
  } catch {
    // engine unreachable; events will fill it in
  }
}

// ---- the caption panel's styles, models and recent jobs -------------------------------------
export type CaptionInfo = Awaited<ReturnType<typeof engine.captionInfo>>
const infos = new Map<string, Promise<CaptionInfo>>()
/** Asked once per dataset (the dataset page asks as it opens) and kept, so the panel opens with
 * it instead of waiting on the server. */
export function captionInfo(dataset: string): Promise<CaptionInfo> {
  let p = infos.get(dataset)
  if (!p) {
    p = engine.captionInfo(dataset)
    infos.set(dataset, p)
    p.catch(() => infos.delete(dataset))
  }
  return p
}
/** Ask again (the panel does as it opens, to show what changed since) */
export function refreshCaptionInfo(dataset: string): Promise<CaptionInfo> {
  infos.delete(dataset)
  return captionInfo(dataset)
}
/** What was asked before, if anything */
export const keptCaptionInfo = (dataset: string) => infos.get(dataset)
