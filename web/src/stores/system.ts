// Memory history from the engine's 1 Hz sampler: what it has when the socket connects (its last
// 15 minutes, or what this tab missed while away), then each new sample as the engine pushes it.
// Shared by the sidebar meters and the System page. The loaded models come the same way (loaded.ts).

import { computed, ref } from 'vue'
import { getClientId, onComfyMessage } from '@/api/comfy'
import { EVENTS, engine, type MemSample, type StorageReport } from '@/api/engine'

export const samples = ref<MemSample[]>([])
export const memLatest = computed(() => samples.value.at(-1))
const MAX = 900

function add(fresh: MemSample[]) {
  if (!fresh.length) return
  const byT = new Map(samples.value.map((s) => [s.t, s]))
  let changed = false
  for (const s of fresh) {
    if (!byT.has(s.t)) {
      byT.set(s.t, s)
      changed = true
    }
  }
  if (changed) samples.value = [...byT.values()].sort((a, b) => a.t - b.t).slice(-MAX)
}

/** free space where ZenTrainer writes; checked once a minute */
export const disks = ref<StorageReport['disks']>([])
async function pollDisks() {
  disks.value = await engine.disks().catch(() => disks.value)
}

// what's new since the last sample, and the pushes from now on
function catchUp() {
  engine
    .system(samples.value.at(-1)?.t ?? 0, getClientId())
    .then((r) => add(r.samples))
    .catch(() => {}) // engine away: the next connect asks again
}

/** true while the engine isn't pushing (one updated but not restarted yet, or a proxy that drops
 *  the websocket): the queue and loaded models are polled then, as these samples are */
export const noPushes = ref(false)
let lastPush = Date.now()

let started = false
export function startSystemSync(): void {
  if (started) return
  started = true
  onComfyMessage((m) => {
    if (m.type === EVENTS.system) {
      lastPush = Date.now()
      noPushes.value = false
      add([m.data as unknown as MemSample])
    }
    // ComfyUI's status comes when the socket (re)connects: an engine restarted since forgot this tab
    else if (m.type === 'status') catchUp()
  })
  catchUp() // the socket may be up already, its status gone by
  // a sample a second is due: none for 5 seconds, and they're asked for (which also asks for the
  // pushes again, so they take over once the engine sends them)
  setInterval(() => {
    if (Date.now() - lastPush < 5000) return
    noPushes.value = true
    catchUp()
  }, 2000)
  pollDisks()
  setInterval(pollDisks, 60_000)
}

export const gb = (b?: number) => (b == null ? '–' : (b / 1024 ** 3).toFixed(1))
