// The app's own housekeeping status for the sidebar: the engine's code changed on disk since
// it started (restart to load it), the web app's source is newer than its build (rebuild it: a
// pull doesn't bring web/dist), a rebuild landed since this tab loaded (reload), or ZenTrainer or
// ComfyUI has an update upstream. Checked at launch, the engine every minute; the updates once (they
// fetch from git), and again from Settings > Versions.

import { computed, ref } from 'vue'
import { engine, type ComfyVersion, type ZtVersion } from '@/api/engine'

export const engineStale = ref(false)
/** set when this ComfyUI is older than ZenTrainer needs (a node-pack install into an old one) */
export const comfyOld = ref<string | null>(null)
/** what's running and (once checked) what's upstream, for the sidebar and Settings > Versions */
export const ztVersion = ref<ZtVersion | null>(null)
export const comfyVersion = ref<ComfyVersion | null>(null)
export const ztBehind = computed(() => ztVersion.value?.upstream?.behind ?? 0)
export const comfyBehind = computed(() => comfyVersion.value?.upstream?.behind ?? 0)
export const checkingUpdates = ref(false)
export const restarting = ref(false)
export const webStale = ref(false)
export const webBuildable = ref(false)
/** web/dist was rebuilt after this tab loaded it: a reload shows the new page */
export const webNewer = ref(false)
let webLoaded: number | null | undefined

let started = false
export function startStatusSync(): void {
  if (started) return
  started = true
  const tick = async () => {
    const h = await engine.health().catch(() => null)
    if (h) {
      engineStale.value = !!h.stale
      comfyOld.value = h.comfy_old ?? null
      webStale.value = !!h.web_stale
      webBuildable.value = !!h.web_buildable
      if (webLoaded === undefined) webLoaded = h.web_built
      webNewer.value = h.web_built != null && h.web_built !== webLoaded
    }
  }
  tick()
  setInterval(tick, 60_000)
  void checkUpdates(false) // the version itself now (the sidebar shows it); what's new upstream later
  setTimeout(() => checkUpdates(), 15_000)
}

/** Versions, and with `check` what's new upstream (both fetch from git: a few seconds). Without it,
 *  only fills in what isn't known yet, so it never drops a check's result. */
export async function checkUpdates(check = true): Promise<void> {
  if (!check) {
    const [z, c] = await Promise.all([engine.version().catch(() => null), engine.comfyui().catch(() => null)])
    ztVersion.value ??= z
    comfyVersion.value ??= c
    return
  }
  checkingUpdates.value = true
  try {
    const [z, c] = await Promise.all([engine.version(true).catch(() => null), engine.comfyui(true).catch(() => null)])
    if (z) ztVersion.value = z
    if (c) comfyVersion.value = c
  } finally {
    checkingUpdates.value = false
  }
}

/** Load what's waiting: rebuild the web app if its source moved on, restart ComfyUI if the
 *  engine's code did, then reload the page (once ComfyUI answers again). */
export async function restartEngine(): Promise<void> {
  restarting.value = true
  try {
    if (webStale.value) await engine.buildWeb()
    if (!engineStale.value) return location.reload()
    await engine.restart()
  } catch (e) {
    restarting.value = false
    throw e
  }
  await new Promise((r) => setTimeout(r, 3000))
  for (let i = 0; i < 120; i++) {
    if (await engine.health().then(() => true).catch(() => false)) return location.reload()
    await new Promise((r) => setTimeout(r, 1000))
  }
  restarting.value = false
}
