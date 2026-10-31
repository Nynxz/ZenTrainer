// One dataset as its page and the page's parts see it: the pictures and their captions in the
// chosen caption set, what's shown (filter, search, sort), the selection, caption drafts, muting,
// faces, and the view settings that are remembered. DatasetView makes it; its parts take it with
// useDatasetContext().
import type { ContextMenuItem, LightboxItem, ZenContextMenu } from '@nynxz/zenkit-ui'
import { computed, inject, onMounted, onUnmounted, reactive, ref, watch, type InjectionKey } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { onComfyMessage } from '@/api/comfy'
import { EVENTS, engine, thumbUrl, viewUrl, type CaptionJob, type DatasetFaces, type DatasetItem } from '@/api/engine'
import type DatasetStats from '@/components/DatasetStats.vue'
import { activeDownloads, fmtBytes, startDownloadSync } from '@/stores/downloads'

export const isVideoPath = (p: string) => /\.(mp4|webm|mov|mkv)$/i.test(p)
export const aspect = (it: DatasetItem) => (it.width && it.height ? `${it.width} / ${it.height}` : '1')
export const isSimplePair = (it: DatasetItem, inputs: string[]) => inputs.length === 1 && it.kind !== 'video' && !isVideoPath(inputs[0]!)
export const secs = (d?: number | null) => (d == null ? '' : d < 60 ? `${d.toFixed(1)}s` : `${Math.floor(d / 60)}:${String(Math.round(d % 60)).padStart(2, '0')}`)
export const pct = (v?: number) => (v == null ? '' : `${Math.round(v * 100)}%`)
export const kb = (b: number) => (b > 1024 ** 2 ? `${(b / 1024 ** 2).toFixed(1)} MB` : `${Math.round(b / 1024)} KB`)
// a card's file details, behind its ⓘ instead of a line under every caption
export const fileInfo = (it: DatasetItem) =>
  [
    it.path,
    it.width ? `${it.width}×${it.height}` : '',
    it.bytes ? kb(it.bytes) : '',
    it.kind === 'video' ? `${it.frames ?? '?'} frames at ${it.fps ?? '?'} fps` : '',
  ]
    .filter(Boolean)
    .join(' · ')

export function useDataset(name: () => string) {
  const router = useRouter()
  const route = useRoute()

  // ---- persisted view prefs -------------------------------------------------------------
  const PREFS = 'zentrainer.datasetView'
  const saved = (() => {
    try {
      return JSON.parse(localStorage.getItem(PREFS) ?? '{}')
    } catch {
      return {}
    }
  })()
  // edit pairs is for working on the inputs, not a way to browse: a dataset always opens in the grid
  // or the list (the last of those used), and pairs is never the one remembered
  const base = ref<'grid' | 'list'>(saved.view === 'list' ? 'list' : 'grid')
  const view = ref<'grid' | 'list' | 'pairs'>(base.value)
  watch(view, (v) => {
    if (v !== 'pairs') base.value = v
  })
  const zoom = ref<number>(saved.zoom ?? 200)
  const fit = ref<'cover' | 'contain'>(saved.fit ?? 'cover')
  const showMasks = ref<boolean>(saved.masks ?? false) // dim what each mask leaves out, on the grid
  const sideOpen = ref<boolean>(saved.side ?? true) // the side panel: health, faces, auto-caption, bulk edit, display
  watch([view, zoom, fit, showMasks, sideOpen], () => {
    try {
      localStorage.setItem(PREFS, JSON.stringify({ view: base.value, zoom: zoom.value, fit: fit.value, masks: showMasks.value, side: sideOpen.value }))
    } catch {
      // storage blocked
    }
  })

  // ---- data -------------------------------------------------------------------------------
  const items = ref<DatasetItem[]>([])
  // caption sets: <img>.txt is "default", <img>.<set>.txt the others; everything below
  // (view, edit, filters, bulk) works on the selected set
  const sets = ref<string[]>(['default'])
  // ?set=<caption set>: opened from somewhere that names one (a run's Setup card)
  const activeSet = ref(typeof route.query.set === 'string' && route.query.set ? route.query.set : 'default')
  const cap = (it: DatasetItem) => it.captions?.[activeSet.value] ?? ''
  // control sets: edit training's input pictures, named layers beside the pictures; the pairs
  // view, the grid's insets and the lightbox show the selected one's
  const controlSets = ref<string[]>([])
  const activeControl = ref('')
  const inputs = (it: DatasetItem) => (activeControl.value && it.control_sets?.[activeControl.value]) || []
  const k = (it: DatasetItem) => `${activeSet.value}:${it.path}`
  const loading = ref(true)
  const error = ref<string | null>(null)
  const notice = ref<string | null>(null)
  const drafts = reactive<Record<string, string>>({})
  const saving = reactive<Record<string, boolean>>({})
  const bust = ref(Date.now())

  async function load() {
    try {
      const r = await engine.dataset(name())
      items.value = r.items
      const facesOn = !!r.settings?.faces
      captionSubject.value = r.settings?.caption_subject ?? ''
      if (facesOn !== facesSetting.value) {
        facesSetting.value = facesOn
        if (facesOn && !faces.value) checkFaces() // cached per picture: quick after the first time
      }
      controlSets.value = r.control_sets ?? []
      // none shown until one's picked (opening a dataset shows its pictures, not their inputs)
      if (activeControl.value && !controlSets.value.includes(activeControl.value)) activeControl.value = ''
      sets.value = [...new Set([...r.sets, ...sets.value])].sort((a, b) => (a === 'default' ? -1 : b === 'default' ? 1 : a.localeCompare(b)))
      if (!sets.value.includes(activeSet.value)) activeSet.value = 'default' // a ?set= that's since been removed
      error.value = null
    } catch (e) {
      error.value = String((e as Error).message ?? e)
    } finally {
      loading.value = false
    }
  }
  onMounted(load)

  // ---- filter / sort / select -------------------------------------------------------------------
  type Filter = 'all' | 'uncaptioned' | 'masked' | 'unmasked' | 'pairs' | 'faces' | 'images' | 'muted'
  const filter = ref<Filter>('all')
  const query = ref('')
  const sortKey = ref<'name' | 'size' | 'res' | 'caption'>('name')
  const sortDir = ref<1 | -1>(1)
  const selected = reactive(new Set<string>())
  const boxing = ref(false) // a drag-select is under way in the grid
  // Review on a caption job: only the images it wrote
  const reviewing = ref<{ job: CaptionJob; paths: Set<string> } | null>(null)
  function review(job: CaptionJob) {
    reviewing.value = { job, paths: new Set(job.written ?? []) }
    activeSet.value = job.set_name
    filter.value = 'all'
  }
  // ?review=<caption job id>: opened from that job's page
  onMounted(async () => {
    const id = route.query.review
    if (typeof id === 'string' && id) review(await engine.captionJob(id))
  })

  const counts = computed(() => ({
    uncaptioned: items.value.filter((i) => !cap(i)).length,
    masked: items.value.filter((i) => i.mask).length,
    mismatched: items.value.filter((i) => i.mask_mismatch).length,
    videos: items.value.filter((i) => i.kind === 'video').length,
    pairs: items.value.filter((i) => inputs(i).length).length,
    faceIssues: items.value.filter((i) => faceIssue(i)).length,
    imageIssues: items.value.filter((i) => imageFlags.value[i.path]?.length).length,
    muted: items.value.filter((i) => isMuted(i)).length,
  }))

  // ---- muting: kept, captions and all, but not trained on in the active caption set ----------------
  const isMuted = (it: DatasetItem) => !!it.muted?.includes(activeSet.value)
  async function setMuted(paths: string[], on: boolean) {
    if (!paths.length) return
    try {
      await engine.datasetMuted(name(), activeSet.value, paths, on)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return
    }
    const want = new Set(paths)
    for (const it of items.value) {
      if (!want.has(it.path)) continue
      const s = new Set(it.muted ?? [])
      if (on) s.add(activeSet.value)
      else s.delete(activeSet.value)
      it.muted = [...s]
    }
    statsPanel.value?.load()
  }
  // the stats panel's per-picture flags (dark, blurry, bars, small, near-duplicate...)
  const imageFlags = ref<Record<string, string[]>>({})
  // the side panel's health card, reloaded when muting or faces change its scores
  const statsPanel = ref<InstanceType<typeof DatasetStats> | null>(null)

  // ---- faces (when on): a face per picture, how much it looks like the rest -----------------
  const faces = ref<DatasetFaces | null>(null)
  const facesBusy = ref(false)
  const facesDone = ref<{ done: number; total: number } | null>(null) // the check's progress (zentrainer.faces)
  onUnmounted(
    onComfyMessage((m) => {
      if (m.type === EVENTS.faces && m.data.dataset === name() && facesBusy.value)
        facesDone.value = { done: Number(m.data.done), total: Number(m.data.total) }
    }),
  )
  /** what the check is doing: the face model's first download, then picture by picture */
  const facesStatus = computed(() => {
    const dl = activeDownloads.value.find((d) => d.family === 'faces')
    if (dl) return `Downloading the face model… ${fmtBytes(dl.done)}${dl.total ? ` of ${fmtBytes(dl.total)}` : ''}`
    const p = facesDone.value
    return p && p.total ? `Checking faces… ${p.done} of ${p.total}` : 'Checking faces…'
  })
  /** the same, short enough for the switch's row */
  const facesShort = computed(() => {
    const dl = activeDownloads.value.find((d) => d.family === 'faces')
    if (dl) return dl.total ? `${Math.round((dl.done / dl.total) * 100)}%` : fmtBytes(dl.done)
    const p = facesDone.value
    return p && p.total ? `${p.done}/${p.total}` : ''
  })
  const captionSubject = ref('') // this dataset's "caption_subject" setting: what auto-captions call the subject
  const facesSetting = ref(false) // this dataset's "faces" setting: on = checked on open, in the stats and runs
  async function toggleFaces() {
    const on = !facesSetting.value
    facesSetting.value = on
    try {
      await engine.datasetSettings(name(), { faces: on })
    } catch {
      facesSetting.value = !on
      return
    }
    if (on) await checkFaces()
    else {
      faces.value = null
      if (filter.value === 'faces') filter.value = 'all'
      statsPanel.value?.load() // back to the balance score
    }
  }
  async function checkFaces() {
    facesBusy.value = true
    facesDone.value = null
    startDownloadSync() // the face model downloads the first time: its progress shows here
    try {
      faces.value = await engine.datasetFaces(name())
      statsPanel.value?.load() // the faces score joins the stats
    } catch (e) {
      faces.value = { available: false, hint: e instanceof Error ? e.message : String(e) }
    } finally {
      facesBusy.value = false
    }
  }
  const faceOf = (it: DatasetItem) => faces.value?.items?.[it.path]
  /** why a picture's face needs a look: none found, more than one person, or not like the others */
  function faceIssue(it: DatasetItem): string | null {
    const f = faceOf(it)
    if (!f) return null
    if (f.faces === 0) return f.masked ? `No face in the trained area (${f.masked} masked out)` : 'No face found'
    if (f.outlier) return `Doesn't look like the others (likeness ${f.likeness?.toFixed(2)})`
    if (f.faces > 1) return `${f.faces} faces: another person can bleed into the likeness. Mask the extra ones out`
    return null
  }
  /** a picture's face, described: its size, likeness, and any faces its mask leaves out */
  const faceNote = (it: DatasetItem) => {
    const f = faceOf(it)!
    const out = f.masked ? ` · ${f.masked} masked out` : ''
    return `Face ${Math.round(f.share * 100)}% of the height; likeness to the others ${f.likeness?.toFixed(2)}${out}`
  }
  const shown = computed(() => {
    const q = query.value.trim().toLowerCase()
    const f = filter.value
    const only = reviewing.value?.paths
    const rows = items.value.filter(
      (it) =>
        (!only || only.has(it.path)) &&
        (f === 'all' ||
          (f === 'uncaptioned' && !cap(it)) ||
          (f === 'masked' && it.mask) ||
          (f === 'unmasked' && !it.mask) ||
          (f === 'pairs' && inputs(it).length) ||
          (f === 'faces' && faceIssue(it)) ||
          (f === 'images' && imageFlags.value[it.path]?.length) ||
          (f === 'muted' && isMuted(it))) &&
        (!q || cap(it).toLowerCase().includes(q) || it.path.toLowerCase().includes(q)),
    )
    const key = (it: DatasetItem): string | number =>
      sortKey.value === 'size'
        ? it.bytes
        : sortKey.value === 'res'
          ? (it.width ?? 0) * (it.height ?? 0)
          : sortKey.value === 'caption'
            ? cap(it).length
            : it.path.toLowerCase()
    return [...rows].sort((a, b) => (key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0) * sortDir.value)
  })
  // a big dataset renders a page at a time (thousands of cards stall the browser): more as the
  // end scrolls near (MoreRows); a new filter, search or sort starts from the top
  const PAGE = 300
  const limit = ref(PAGE)
  watch([filter, query, sortKey, sortDir, reviewing, items], () => (limit.value = PAGE))
  const visible = computed(() => shown.value.slice(0, limit.value))
  const more = () => {
    if (limit.value < shown.value.length) limit.value += PAGE
  }
  function sortBy(k: typeof sortKey.value) {
    if (sortKey.value === k) sortDir.value = sortDir.value === 1 ? -1 : 1
    else {
      sortKey.value = k
      sortDir.value = 1
    }
  }
  let lastClicked: number | null = null
  function toggleSelect(it: DatasetItem, i: number, e?: MouseEvent) {
    if (e?.shiftKey && lastClicked != null) {
      const [a, b] = [Math.min(lastClicked, i), Math.max(lastClicked, i)]
      for (const row of shown.value.slice(a, b + 1)) selected.add(row.path)
    } else if (selected.has(it.path)) selected.delete(it.path)
    else selected.add(it.path)
    lastClicked = i
  }
  // a plain click opens the picture; Ctrl/Cmd+click toggles it in the selection, Shift+click a range
  function onItemClick(it: DatasetItem, i: number, e: MouseEvent) {
    if (e.ctrlKey || e.metaKey || e.shiftKey) toggleSelect(it, i, e)
    else lightbox.value = i
  }
  // right-click: acts on the selection when the picture is in it, else on that picture alone
  const menu = ref<InstanceType<typeof ZenContextMenu> | null>(null)
  const menuItems = ref<ContextMenuItem[]>([])
  function onItemMenu(it: DatasetItem, i: number, e: MouseEvent) {
    const many = selected.has(it.path) && selected.size > 1
    const paths = many ? [...selected] : [it.path]
    const targets = items.value.filter((x) => paths.includes(x.path))
    const anyMuted = targets.some(isMuted)
    const allMuted = targets.every(isMuted)
    const file = it.path.split('/').at(-1)!
    menuItems.value = [
      { heading: many ? `${paths.length} pictures` : file },
      ...(many ? [] : [{ label: 'Open', icon: 'mdi-open-in-app', run: () => (lightbox.value = i) }]),
      selected.has(it.path) && !many
        ? { label: 'Deselect', icon: 'mdi-checkbox-blank-outline', hint: 'Ctrl+click', run: () => selected.delete(it.path) }
        : many
          ? { label: 'Clear selection', icon: 'mdi-close', hint: 'Esc', run: () => selected.clear() }
          : { label: 'Select', icon: 'mdi-checkbox-marked-outline', hint: 'Ctrl+click', run: () => toggleSelect(it, i) },
      '-',
      ...(!allMuted ? [{ label: `Mute in “${activeSet.value}”`, icon: 'mdi-eye-off-outline', run: () => setMuted(paths, true) }] : []),
      ...(anyMuted ? [{ label: `Unmute in “${activeSet.value}”`, icon: 'mdi-eye-outline', run: () => setMuted(paths, false) }] : []),
      ...(many
        ? []
        : [
            { label: 'Copy caption', icon: 'mdi-content-copy', disabled: !cap(it), run: () => navigator.clipboard?.writeText(cap(it)) },
            ...(it.kind !== 'video'
              ? [{ label: 'Edit mask', icon: 'mdi-draw', run: () => router.push({ path: `/datasets/${encodeURIComponent(name())}/mask`, query: { path: it.path } }) }]
              : []),
          ]),
      '-',
      { label: many ? `Delete ${paths.length}…` : 'Delete…', icon: 'mdi-trash-can-outline', danger: true, run: () => removeMany(paths) },
    ]
    menu.value?.show(e)
  }
  const allShownSelected = computed(() => shown.value.length > 0 && shown.value.every((r) => selected.has(r.path)))
  function toggleAll() {
    if (allShownSelected.value) for (const r of shown.value) selected.delete(r.path)
    else for (const r of shown.value) selected.add(r.path)
  }

  // ---- editing ------------------------------------------------------------------------------
  function flash(msg: string) {
    notice.value = msg
    setTimeout(() => notice.value === msg && (notice.value = null), 2500)
  }
  async function save(it: DatasetItem) {
    const key = k(it)
    const set = activeSet.value
    const text = drafts[key]
    if (text == null || text.trim() === cap(it)) {
      delete drafts[key]
      return
    }
    saving[key] = true
    try {
      await engine.setCaption(name(), it.path, text, set)
      it.captions = { ...it.captions, [set]: text.trim() }
      delete drafts[key]
    } catch (e) {
      error.value = String((e as Error).message ?? e)
    } finally {
      saving[key] = false
    }
  }
  async function removeMany(paths: string[]) {
    if (!paths.length || !confirm(`Delete ${paths.length} image${paths.length === 1 ? '' : 's'} with their captions and masks?`)) return
    for (const p of paths) await engine.deleteItem(name(), p)
    for (const p of paths) selected.delete(p)
    items.value = items.value.filter((x) => !paths.includes(x.path))
    if (lightbox.value != null && lightbox.value >= shown.value.length) lightbox.value = shown.value.length ? shown.value.length - 1 : null
    flash(`${paths.length} deleted`)
  }

  // ---- pictures -------------------------------------------------------------------------------
  // the remembered view, except Pairs on a dataset without any (it shows the grid there)
  const layout = computed(() => (view.value === 'pairs' && !activeControl.value ? base.value : view.value))
  // an input file sits beside its result: same folder, so /view takes the result's subfolder
  const ctrlView = (it: DatasetItem, p: string) =>
    `${viewUrl({ filename: p.split('/').at(-1)!, subfolder: it.subfolder, type: 'input' })}&v=${bust.value}`
  // what holding Shift over an item shows: the file itself (a video plays), not its thumbnail
  const full = (it: DatasetItem) => `${viewUrl({ ...it, type: 'input' })}&v=${bust.value}`
  const thumb = (it: Pick<DatasetItem, 'path'>, size = 384) => `${thumbUrl(name(), it.path, size)}&v=${bust.value}`
  const thumbSize = computed(() => (zoom.value > 260 ? 768 : 384))

  // ---- lightbox -----------------------------------------------------------------------------
  const lightbox = ref<number | null>(null)
  const lbItems = computed<LightboxItem[]>(() =>
    shown.value.map((it) => ({
      src: `${viewUrl({ ...it, type: 'input' })}&v=${bust.value}`,
      thumb: thumb(it, 256),
      label: it.path,
      meta: it.width ? `${it.width}×${it.height} · ${kb(it.bytes)}` : kb(it.bytes),
      overlay: it.mask ? thumb({ path: it.mask } as DatasetItem, 1024) : undefined,
      overlayMode: 'mask' as const, // the picture dims; what the mask trains shows at full strength
      kind: it.kind === 'video' ? ('video' as const) : undefined,
    })),
  )

  return {
    view, zoom, fit, showMasks, sideOpen,
    items, sets, activeSet, cap, k, controlSets, activeControl, inputs, loading, error, notice, drafts, saving, bust, load,
    filter, query, sortKey, sortDir, selected, boxing, reviewing, review, counts,
    isMuted, setMuted, imageFlags, statsPanel,
    faces, facesBusy, facesStatus, facesShort, captionSubject, facesSetting, toggleFaces, faceOf, faceIssue, faceNote,
    shown, visible, more, sortBy, toggleSelect, onItemClick, menu, menuItems, onItemMenu, allShownSelected, toggleAll,
    flash, save, removeMany,
    layout, ctrlView, full, thumb, thumbSize, lightbox, lbItems,
  }
}

export type DatasetContext = ReturnType<typeof useDataset>
export const DATASET: InjectionKey<DatasetContext> = Symbol('dataset')
export const useDatasetContext = () => inject(DATASET)!
