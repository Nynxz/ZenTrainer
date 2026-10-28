// App-wide run state: the run list plus the latest metric per active run, kept
// fresh from ComfyUI's websocket. The sidebar and the runs list both read it.

import { computed, reactive, ref } from 'vue'
import { onComfyMessage } from '@/api/comfy'
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

export async function refreshQueue(): Promise<void> {
  try {
    queue.value = await engine.queue()
  } catch {
    // engine unreachable; runs list reports it
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
  setInterval(refreshQueue, 4000)
  onComfyMessage((m) => {
    const id = m.data.run_id as string | undefined
    if (!id) return
    if (m.type === EVENTS.metrics) {
      latest[id] = m.data as unknown as MetricRow
      pushLoss(id, latest[id].loss)
    } else if (m.type === EVENTS.status) {
      refreshQueue()
      const r = runs.value.find((x) => x.run_id === id)
      if (r) {
        r.status = m.data.status as RunStatus
        r.message = (m.data.message as string | null) ?? undefined
      } else {
        refreshRuns()
      }
    }
  })
}
