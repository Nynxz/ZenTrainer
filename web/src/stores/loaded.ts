// The models ComfyUI holds right now (engine/zentrainer/models/residency.py), for everything that
// shows them: the sidebar, a run's page, the log drawer. The engine sends the list when it changes,
// to the tabs that take its memory stats (system.ts); read every few seconds when it doesn't.

import { computed, ref } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { EVENTS, engine, type LoadedModel } from '@/api/engine'
import { noPushes, startSystemSync } from './system'

export const loadedModels = ref<LoadedModel[]>([])

let started = false
let gen = 0
export function startLoadedSync(): void {
  if (started) return
  started = true
  const take = (models: LoadedModel[]) => {
    gen++
    loadedModels.value = models.filter((x) => x.bytes > 0)
  }
  onComfyMessage((m) => m.type === EVENTS.loaded && take(m.data.models as LoadedModel[]))
  startSystemSync()
  setInterval(() => {
    if (!noPushes.value) return
    const g = gen
    engine.loadedModels().then((models) => g === gen && take(models), () => {})
  }, 3000)
}

export const loadedVram = computed(() => loadedModels.value.reduce((n, m) => n + m.vram, 0))
