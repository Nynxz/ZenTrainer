// App-wide run state: the run list plus the latest metric per active run, kept
// fresh from ComfyUI's websocket. The sidebar and the runs list both read it.

import { computed, reactive, ref } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { noPushes } from './system'
import { ACTIVE, EVENTS, engine, type MetricRow, type QueueState, type RunMeta, type RunStatus } from '@/api/engine'

export const runs = ref<RunMeta[]>([])
export const latest = reactive<Record<string, MetricRow>>({})
export const engineError = ref<string | null>(null)
/** recent losses per active run, newest last (for sparklines) */
export const lossTrail = reactive<Record<string, number[]>>({})
const TRAIL = 80

function pushLoss(id: string, loss: number | null | undefined) {
  if (loss == null) return
  const t = (lossTrail[id] ??= [])
  t.push(loss)
  if (t.length > TRAIL) t.splice(0, t.length - TRAIL)
}

export const queue = ref<QueueState>({ paused: false, current: null, items: [] })
let queueRetry = false

export async function refreshQueue(): Promise<void> {
  try {
    queue.value = await engine.queue()
    queueRetry = false
  } catch {
    queueRetry = true // retry on the next tick even while system pushes stay healthy
  }
}

export const activeRuns = computed(() => runs.value.filter((r) => ACTIVE.includes(r.status)))

export async function refreshRuns(): Promise<void> {
  try {
    runs.value = await engine.runs()
    engineError.value = null
    // seed progress for active runs so nothing reads 0/0 until the next live step. Each on its own:
    // a run that's only just starting may not have its details yet, and that isn't the engine down
    await Promise.all(
      runs.value
        .filter((r) => ACTIVE.includes(r.status) && !latest[r.run_id])
        .map(async (r) => {
          const metrics = (await engine.run(r.run_id).catch(() => null))?.metrics ?? []
          const last = metrics.at(-1)
          if (last && !latest[r.run_id]) latest[r.run_id] = last
          if (!lossTrail[r.run_id]) for (const m of metrics.slice(-TRAIL)) pushLoss(r.run_id, m.loss)
        }),
    )
  } catch (e) {
    engineError.value = String(e)
  }
}

let started = false
export function startRunSync(): void {
  if (started) return
  started = true
  refreshRuns()
  refreshQueue()
  setInterval(() => (noPushes.value || queueRetry) && refreshQueue(), 4000) // an engine that doesn't say (system.ts) or a failed queue read
  onComfyMessage((m) => {
    // the engine says when its queue changes; ComfyUI's status comes when the socket (re)connects,
    // after a restart, say, when the queue may have changed unseen
    if (m.type === EVENTS.queue || m.type === 'status') return void refreshQueue()
    const id = m.data.run_id as string | undefined
    if (!id) return
    if (m.type === EVENTS.metrics) {
      latest[id] = m.data as unknown as MetricRow
      pushLoss(id, latest[id].loss)
    } else if (m.type === EVENTS.status) {
      const r = runs.value.find((x) => x.run_id === id)
      // a run's progress messages (caching latents 5 of 18...) come as status events too: the queue
      // moves only when its status does
      if (r?.status !== m.data.status) refreshQueue()
      if (r) {
        r.status = m.data.status as RunStatus
        r.message = (m.data.message as string | null) ?? undefined
      } else {
        refreshRuns()
      }
    }
  })
}
