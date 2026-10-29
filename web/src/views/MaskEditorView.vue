<script setup lang="ts">
// The mask editor for one dataset image: find the subject with SAM3 (a word, and/or clicks:
// left keeps, right excludes), touch it up with a brush, then grow/shrink and feather it, and
// save it as the image's mask (<name>-masklabel.png; white trains at full weight).
// Grow/shrink and feather apply on top of the raw mask, so they can be changed freely.
// A box (R) takes a region out, e.g. a watermark; the list on the left selects images, and "Apply to
// selected" writes the mask to every selected image of this size (cut out of masks they already
// have), with a preview first and an undo after.
// Keys: [ / ] brush size, B brush, E eraser, P points, R box, ←/→ previous/next image, Ctrl+S save,
// Ctrl+→ save and next.
import { ZenButton, ZenInput, ZenModal, ZenSlider, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { EngineError, engine, thumbUrl, viewUrl, type DatasetItem } from '@/api/engine'
import { remembered } from '@/lib/remembered'

const props = defineProps<{ name: string }>()
const route = useRoute()
const router = useRouter()

const items = ref<DatasetItem[]>([])
const path = computed(() => (typeof route.query.path === 'string' ? route.query.path : ''))
const index = computed(() => items.value.findIndex((i) => i.path === path.value))
const item = computed(() => items.value[index.value] ?? null)

// original files by their /view address (type input: the subfolder starts with the dataset)
const fileUrl = (rel: string) => {
  const parts = `${props.name}/${rel}`.split('/')
  const filename = parts.pop()!
  return `${viewUrl({ filename, subfolder: parts.join('/'), type: 'input' })}&t=${Date.now()}`
}

// ---- canvases ------------------------------------------------------------------------------
const stage = ref<HTMLDivElement | null>(null)
const view = ref<HTMLCanvasElement | null>(null)
const img = new Image()
const work = document.createElement('canvas') // the raw mask (white = keep), image resolution
const W = ref(0)
const H = ref(0)
const loaded = ref(false)
const dirty = ref(false)
const hasMask = ref(false)

const tool = ref<'points' | 'brush' | 'erase' | 'rect'>('points')
const brush = ref(40)
const grow = ref(0) // px, negative shrinks
const feather = ref(0) // px
const show = ref<'overlay' | 'mask'>('overlay')
// what's left out: faded into the theme's background, or (High-vis) a magenta checker nothing in a
// picture looks like, for edges against dark or busy ones
const highVis = remembered('zentrainer.maskHighVis', false)
const text = ref('')
const threshold = ref(0.5)
const points = ref<{ x: number; y: number; keep: boolean }[]>([])
const busy = ref(false)
const status = ref('')
const error = ref<string | null>(null)

function wctx() {
  return work.getContext('2d', { willReadFrequently: true })!
}

async function load() {
  loaded.value = false
  dirty.value = false
  points.value = []
  error.value = null
  status.value = ''
  if (!item.value) return
  await new Promise<void>((res, rej) => {
    img.onload = () => res()
    img.onerror = () => rej(new Error('image failed to load'))
    img.src = fileUrl(item.value!.path)
  }).catch((e) => (error.value = String(e)))
  W.value = img.naturalWidth
  H.value = img.naturalHeight
  work.width = W.value
  work.height = H.value
  const c = wctx()
  // a new mask starts as no mask at all: everything trains (white), and what you paint, box or
  // erase is what's left out; a SAM selection still replaces it with just the subject
  c.fillStyle = '#fff'
  c.fillRect(0, 0, W.value, H.value)
  hasMask.value = !!item.value.mask
  if (item.value.mask) {
    const m = new Image()
    await new Promise<void>((res) => {
      m.onload = () => res()
      m.onerror = () => res()
      m.src = fileUrl(item.value!.mask!)
    })
    if (m.naturalWidth) c.drawImage(m, 0, 0, W.value, H.value)
  }
  loaded.value = true
  await nextTick()
  render()
}

// the final mask: raw, grown or shrunk, then feathered (white = keep)
function finalMask(): HTMLCanvasElement {
  const out = document.createElement('canvas')
  out.width = W.value
  out.height = H.value
  const o = out.getContext('2d', { willReadFrequently: true })!
  if (grow.value) {
    const r = Math.abs(grow.value)
    o.filter = `blur(${r / 2}px)`
    o.drawImage(work, 0, 0)
    o.filter = 'none'
    const d = o.getImageData(0, 0, W.value, H.value)
    const cut = grow.value > 0 ? 8 : 247 // blur then a low cut grows, a high cut shrinks
    for (let i = 0; i < d.data.length; i += 4) {
      const v = d.data[i]! > cut ? 255 : 0
      d.data[i] = d.data[i + 1] = d.data[i + 2] = v
      d.data[i + 3] = 255
    }
    o.putImageData(d, 0, 0)
  } else o.drawImage(work, 0, 0)
  if (feather.value) {
    const tmp = document.createElement('canvas')
    tmp.width = W.value
    tmp.height = H.value
    const t = tmp.getContext('2d')!
    t.filter = `blur(${feather.value}px)`
    t.drawImage(out, 0, 0)
    o.clearRect(0, 0, W.value, H.value)
    o.drawImage(tmp, 0, 0)
  }
  return out
}

// the theme's background as rgb (any CSS colour it's written in)
function themeBg(): [number, number, number] {
  const c = document.createElement('canvas').getContext('2d')!
  c.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--zen-bg').trim() || '#000'
  c.fillRect(0, 0, 1, 1)
  const [r, g, b] = c.getImageData(0, 0, 1, 1).data
  return [r!, g!, b!]
}

// what a mask leaves out, fading in as the mask goes from keep to drop: the theme's background,
// or High-vis, Source's purple/black missing-texture checker (it can't be mistaken for a dark
// part of the picture), its cells scaled with the picture so the pattern reads at any size
function shadeOut(d: ImageData, w: number, alpha: number) {
  const cell = Math.max(6, Math.round(w / 40))
  const bg = highVis.value ? null : themeBg()
  for (let i = 0; i < d.data.length; i += 4) {
    const keep = d.data[i]! / 255
    if (bg) {
      d.data[i] = bg[0]
      d.data[i + 1] = bg[1]
      d.data[i + 2] = bg[2]
    } else {
      const px = (i / 4) % w
      const py = Math.floor(i / 4 / w)
      const magenta = (Math.floor(px / cell) + Math.floor(py / cell)) % 2 === 0
      d.data[i] = magenta ? 255 : 0
      d.data[i + 1] = 0
      d.data[i + 2] = magenta ? 220 : 0
    }
    d.data[i + 3] = Math.round((1 - keep) * alpha)
  }
}

// view geometry: the image fitted in the stage
const fit = ref({ x: 0, y: 0, s: 1 })
function render() {
  const v = view.value
  const st = stage.value
  if (!v || !st || !loaded.value) return
  const dpr = window.devicePixelRatio || 1
  const cw = st.clientWidth
  const ch = st.clientHeight
  v.width = cw * dpr
  v.height = ch * dpr
  v.style.width = `${cw}px`
  v.style.height = `${ch}px`
  const s = Math.min(cw / W.value, ch / H.value)
  fit.value = { x: (cw - W.value * s) / 2, y: (ch - H.value * s) / 2, s }
  const c = v.getContext('2d')!
  c.setTransform(dpr, 0, 0, dpr, 0, 0)
  c.clearRect(0, 0, cw, ch)
  const { x, y } = fit.value
  const mask = finalMask()
  if (show.value === 'mask') {
    c.drawImage(mask, x, y, W.value * s, H.value * s)
  } else {
    c.drawImage(img, x, y, W.value * s, H.value * s)
    // what's excluded (black in the mask) shaded over, the picture still showing through
    // enough to draw an edge against
    const tint = document.createElement('canvas')
    tint.width = W.value
    tint.height = H.value
    const t = tint.getContext('2d', { willReadFrequently: true })!
    t.drawImage(mask, 0, 0)
    const d = t.getImageData(0, 0, W.value, H.value)
    shadeOut(d, W.value, highVis.value ? 170 : 215)
    t.putImageData(d, 0, 0)
    c.drawImage(tint, x, y, W.value * s, H.value * s)
  }
  for (const p of points.value) {
    c.beginPath()
    c.arc(x + p.x * s, y + p.y * s, 7, 0, Math.PI * 2)
    c.fillStyle = p.keep ? '#22c55e' : '#ef4444'
    c.fill()
    c.lineWidth = 2
    c.strokeStyle = '#fff'
    c.stroke()
  }
  const r = dragRect.value
  if (r) {
    c.setLineDash([6, 4])
    c.lineWidth = 1.5
    c.strokeStyle = '#fff'
    c.strokeRect(x + Math.min(r.x0, r.x1) * s, y + Math.min(r.y0, r.y1) * s, Math.abs(r.x1 - r.x0) * s, Math.abs(r.y1 - r.y0) * s)
    c.setLineDash([])
  }
}
watch([grow, feather, show, highVis], render)

// ---- input ---------------------------------------------------------------------------------
const toImage = (e: PointerEvent) => {
  const r = view.value!.getBoundingClientRect()
  return { x: (e.clientX - r.left - fit.value.x) / fit.value.s, y: (e.clientY - r.top - fit.value.y) / fit.value.s }
}
let painting = false
let last: { x: number; y: number } | null = null
function stroke(p: { x: number; y: number }) {
  const c = wctx()
  c.strokeStyle = c.fillStyle = tool.value === 'erase' ? '#000' : '#fff'
  c.lineCap = 'round'
  c.lineWidth = brush.value
  c.beginPath()
  if (last) {
    c.moveTo(last.x, last.y)
    c.lineTo(p.x, p.y)
    c.stroke()
  } else {
    c.arc(p.x, p.y, brush.value / 2, 0, Math.PI * 2)
    c.fill()
  }
  last = p
  dirty.value = true
  render()
}
function down(e: PointerEvent) {
  if (!loaded.value || busy.value) return // saving: an edit now would be lost when Save & next moves on
  const p = toImage(e)
  if (p.x < 0 || p.y < 0 || p.x > W.value || p.y > H.value) return
  if (tool.value === 'points') {
    // a click on a dot takes it away (the dots are 7 px on screen; a little slack round them)
    const hit = points.value.findIndex((q) => Math.hypot(q.x - p.x, q.y - p.y) * fit.value.s <= 11)
    if (hit >= 0) {
      points.value = points.value.filter((_, i) => i !== hit)
      render()
      if (text.value.trim() || points.value.some((q) => q.keep)) detect()
      return
    }
    points.value = [...points.value, { x: Math.round(p.x), y: Math.round(p.y), keep: e.button !== 2 && !e.altKey }]
    render()
    detect()
    return
  }
  if (tool.value === 'rect') {
    dragRect.value = { x0: p.x, y0: p.y, x1: p.x, y1: p.y }
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    return
  }
  painting = true
  last = null
  ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  stroke(p)
}
function move(e: PointerEvent) {
  cursor.value = { x: e.offsetX, y: e.offsetY }
  if (painting) stroke(toImage(e))
  if (dragRect.value) {
    const p = toImage(e)
    dragRect.value = { ...dragRect.value, x1: Math.max(0, Math.min(W.value, p.x)), y1: Math.max(0, Math.min(H.value, p.y)) }
    render()
  }
}
function up() {
  painting = false
  last = null
  if (dragRect.value) boxOut()
}

// ---- box: take a region out (a watermark) ----------------------------------------------------
const dragRect = ref<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
function isEmpty() {
  const d = wctx().getImageData(0, 0, W.value, H.value).data
  for (let i = 0; i < d.length; i += 4 * 7) if (d[i]! > 0) return false
  return true
}
function boxOut() {
  const r = dragRect.value!
  dragRect.value = null
  const x = Math.round(Math.min(r.x0, r.x1))
  const y = Math.round(Math.min(r.y0, r.y1))
  const w = Math.round(Math.abs(r.x1 - r.x0))
  const h = Math.round(Math.abs(r.y1 - r.y0))
  if (w < 2 || h < 2) return render()
  const c = wctx()
  if (isEmpty()) {
    // a box out of nothing means nothing: start from the whole picture
    c.fillStyle = '#fff'
    c.fillRect(0, 0, W.value, H.value)
    status.value = 'The whole picture trains except the box.'
  }
  c.fillStyle = '#000'
  c.fillRect(x, y, w, h)
  dirty.value = true
  render()
}
const cursor = ref<{ x: number; y: number } | null>(null)

// ---- SAM3 ----------------------------------------------------------------------------------
let seq = 0
// what the SAM3 job is doing, said plainly: the first one loads the model, the slow part
function segMessage(job: { status: string; message?: string }) {
  const m = job.message ?? ''
  if (m.startsWith('loading')) return 'Loading SAM3… the first one after a restart takes a moment'
  if (m.startsWith('waits')) return 'Waiting for the GPU: a training run has it, SAM3 goes next'
  if (job.status === 'queued') return 'Waiting for the GPU…'
  return 'Finding…'
}
async function detect() {
  if (!item.value || (!text.value.trim() && !points.value.some((p) => p.keep))) return
  const mine = ++seq
  busy.value = true
  error.value = null
  status.value = 'Finding…'
  try {
    let { job } = await engine.segment(props.name, {
      path: item.value.path,
      text: text.value.trim(),
      positive: points.value.filter((p) => p.keep).map(({ x, y }) => ({ x, y })),
      negative: points.value.filter((p) => !p.keep).map(({ x, y }) => ({ x, y })),
      threshold: threshold.value,
    })
    while (job.status === 'queued' || job.status === 'running') {
      status.value = segMessage(job)
      await new Promise((r) => setTimeout(r, 1000))
      if (mine !== seq) return
      job = await engine.segmentJob(job.job_id)
    }
    if (mine !== seq) return
    if (job.status === 'error') throw new Error(job.message)
    const m = new Image()
    await new Promise<void>((res, rej) => {
      m.onload = () => res()
      m.onerror = () => rej(new Error('mask failed to load'))
      m.src = `/zentrainer/api/segment/${job.job_id}.png`
    })
    const c = wctx()
    c.drawImage(m, 0, 0, W.value, H.value)
    dirty.value = true
    status.value = 'Found. Click to add (left) or exclude (right), or touch up with Mask out / Paint back.'
    render()
  } catch (e) {
    if (mine === seq) error.value = e instanceof EngineError ? e.message : (e as Error).message
    status.value = ''
  } finally {
    if (mine === seq) busy.value = false
  }
}

// ---- edits ---------------------------------------------------------------------------------
function fill(white: boolean) {
  const c = wctx()
  c.fillStyle = white ? '#fff' : '#000'
  c.fillRect(0, 0, W.value, H.value)
  dirty.value = true
  render()
}
function invert() {
  const c = wctx()
  const d = c.getImageData(0, 0, W.value, H.value)
  for (let i = 0; i < d.data.length; i += 4) d.data[i] = d.data[i + 1] = d.data[i + 2] = 255 - d.data[i]!
  c.putImageData(d, 0, 0)
  dirty.value = true
  render()
}
const coverage = computed(() => {
  void dirty.value
  if (!loaded.value) return null
  const d = wctx().getImageData(0, 0, W.value, H.value).data
  let s = 0
  for (let i = 0; i < d.length; i += 4 * 16) s += d[i]!
  return s / (255 * (d.length / (4 * 16)))
})

async function save() {
  if (!item.value) return
  busy.value = true
  error.value = null
  try {
    await engine.saveMask(props.name, item.value.path, finalMask().toDataURL('image/png'))
    dirty.value = false
    hasMask.value = true
    grow.value = 0
    feather.value = 0
    status.value = ''
    bust.value++
    await refresh()
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
// save (if anything changed: an untouched image keeps what it has), then the next image
async function saveNext() {
  if (busy.value || !loaded.value) return // the next picture still loading: its canvas isn't it yet
  if (dirty.value || grow.value || feather.value) {
    await save()
    if (error.value) return // the save failed: stay on it
  }
  go(1)
}
async function remove() {
  if (!item.value || !confirm('Delete this image’s mask? It trains on the whole image again.')) return
  await engine.saveMask(props.name, item.value.path, '')
  await refresh()
  await load()
}
async function refresh() {
  items.value = (await engine.dataset(props.name)).items.filter((i) => i.kind !== 'video')
}
function go(by: number) {
  const next = items.value[index.value + by]
  if (!next) return
  if (dirty.value && !confirm('Leave without saving this mask?')) return
  router.replace({ query: { path: next.path } })
}

// ---- the list: pick images; masks of this size; apply to selected ----------------------------
const bust = ref(0) // thumbnails of masks that just changed
const thumb = (rel: string, size = 160) => `${thumbUrl(props.name, rel, size)}&v=${bust.value}`
const selected = ref<Set<string>>(new Set())
let anchor: string | null = null
const sameSize = computed(() => (item.value ? items.value.filter((i) => i.width === item.value!.width && i.height === item.value!.height) : []))
const selectedSame = computed(() => sameSize.value.filter((i) => selected.value.has(i.path)))
const sameMasks = computed(() => sameSize.value.filter((i) => i.mask && i.path !== item.value?.path))
function pick(it: DatasetItem, e: MouseEvent, viaBox = false) {
  if (!viaBox && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
    if (it.path === path.value) return
    if (dirty.value && !confirm('Leave without saving this mask?')) return
    router.replace({ query: { path: it.path } })
    return
  }
  const next = new Set(selected.value)
  if (e.shiftKey && anchor) {
    const a = items.value.findIndex((i) => i.path === anchor)
    const b = items.value.findIndex((i) => i.path === it.path)
    for (const i of items.value.slice(Math.min(a, b), Math.max(a, b) + 1)) next.add(i.path)
  } else if (next.has(it.path)) next.delete(it.path)
  else next.add(it.path)
  anchor = it.path
  selected.value = next
}
// the list on the left: search and a filter, like the lightbox's thumbnails
const search = ref('')
const filter = ref<'all' | 'size' | 'unmasked' | 'masked' | 'selected'>('all')
const FILTERS = { all: 'All', size: 'This size', unmasked: 'Unmasked', masked: 'Masked', selected: 'Selected' } as const
const shown = computed(() => {
  const q = search.value.trim().toLowerCase()
  return items.value.filter((it) => {
    if (q && !it.path.toLowerCase().includes(q) && !(it.caption ?? '').toLowerCase().includes(q)) return false
    if (filter.value === 'size') return !!item.value && it.width === item.value.width && it.height === item.value.height
    if (filter.value === 'unmasked') return !it.mask
    if (filter.value === 'masked') return !!it.mask
    if (filter.value === 'selected') return selected.value.has(it.path)
    return true
  })
})
const selectShown = () => (selected.value = new Set([...selected.value, ...shown.value.map((i) => i.path)]))
const selectSame = (onlyUnmasked = false) =>
  (selected.value = new Set(sameSize.value.filter((i) => !onlyUnmasked || !i.mask).map((i) => i.path)))
async function useMask(it: DatasetItem) {
  const m = new Image()
  await new Promise<void>((res, rej) => {
    m.onload = () => res()
    m.onerror = () => rej(new Error('mask failed to load'))
    m.src = fileUrl(it.mask!)
  })
  wctx().drawImage(m, 0, 0, W.value, H.value)
  dirty.value = true
  status.value = `Started from ${it.filename}'s mask.`
  render()
}

const loadImg = (src: string) =>
  new Promise<HTMLImageElement>((res, rej) => {
    const im = new Image()
    im.onload = () => res(im)
    im.onerror = () => rej(new Error(`couldn't load ${src}`))
    im.src = src
  })
const applyOpen = ref(false)
const applyMode = ref<'cutout' | 'replace'>('cutout')
const previews = ref<{ path: string; url: string; had: boolean }[]>([])
const PREVIEW_MAX = 48
const lastBatch = ref<{ batch: string; count: number } | null>(null)
// a selected image with the mask as it would be written: its own mask (cut out) or the new one
async function previewTile(it: DatasetItem, mask: HTMLCanvasElement): Promise<string> {
  const im = await loadImg(thumb(it.path, 256))
  const w = im.naturalWidth
  const h = im.naturalHeight
  const m = document.createElement('canvas')
  m.width = w
  m.height = h
  const mc = m.getContext('2d', { willReadFrequently: true })!
  mc.drawImage(mask, 0, 0, w, h)
  if (applyMode.value === 'cutout' && it.mask) {
    mc.globalCompositeOperation = 'darken'
    mc.drawImage(await loadImg(thumb(it.mask, 256)), 0, 0, w, h)
  }
  const d = mc.getImageData(0, 0, w, h)
  shadeOut(d, w, highVis.value ? 200 : 225)
  mc.globalCompositeOperation = 'source-over'
  mc.putImageData(d, 0, 0)
  const out = document.createElement('canvas')
  out.width = w
  out.height = h
  const oc = out.getContext('2d')!
  oc.drawImage(im, 0, 0)
  oc.drawImage(m, 0, 0)
  return out.toDataURL('image/jpeg', 0.85)
}
async function openApply() {
  applyOpen.value = true
  await buildPreviews()
}
async function buildPreviews() {
  previews.value = []
  const mask = finalMask()
  for (const it of selectedSame.value.slice(0, PREVIEW_MAX)) {
    try {
      previews.value = [...previews.value, { path: it.path, url: await previewTile(it, mask), had: !!it.mask }]
    } catch {
      // a thumbnail that won't load just isn't previewed
    }
  }
}
watch(applyMode, () => applyOpen.value && buildPreviews())
async function doApply() {
  busy.value = true
  error.value = null
  try {
    const paths = selectedSame.value.map((i) => i.path)
    const r = await engine.applyMasks(props.name, paths, finalMask().toDataURL('image/png'), applyMode.value)
    const done = r.results.filter((x) => x.mask).length
    const skipped = r.results.filter((x) => x.skipped)
    lastBatch.value = r.batch ? { batch: r.batch, count: done } : null
    applyOpen.value = false
    bust.value++
    await refresh()
    if (paths.includes(path.value)) await load()
    status.value = `Applied to ${done} image${done === 1 ? '' : 's'}` + (skipped.length ? `, ${skipped.length} skipped (${skipped[0]!.skipped})` : '')
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
async function undoApply() {
  if (!lastBatch.value) return
  busy.value = true
  try {
    const r = await engine.undoMasks(props.name, lastBatch.value.batch)
    lastBatch.value = null
    bust.value++
    await refresh()
    await load()
    status.value = `Put back the masks of ${r.restored} image${r.restored === 1 ? '' : 's'}`
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

function onKey(e: KeyboardEvent) {
  if ((e.target as HTMLElement).closest('input, textarea')) return
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault()
    save()
  } else if ((e.ctrlKey || e.metaKey) && e.key === 'ArrowRight') {
    e.preventDefault()
    saveNext()
  } else if (e.key === '[') brush.value = Math.max(4, brush.value - 8)
  else if (e.key === ']') brush.value = Math.min(400, brush.value + 8)
  else if (e.key === 'b') tool.value = 'brush'
  else if (e.key === 'e') tool.value = 'erase'
  else if (e.key === 'p') tool.value = 'points'
  else if (e.key === 'r') tool.value = 'rect'
  else if (e.key === 'ArrowLeft') go(-1)
  else if (e.key === 'ArrowRight') go(1)
}
const ro = new ResizeObserver(() => render())
onMounted(async () => {
  await refresh()
  if (!path.value && items.value[0]) router.replace({ query: { path: items.value[0].path } })
  else load()
  addEventListener('keydown', onKey)
  if (stage.value) ro.observe(stage.value)
})
onUnmounted(() => {
  removeEventListener('keydown', onKey)
  ro.disconnect()
})
watch(path, load)
</script>

<template>
  <div class="page full mask-page">
    <div class="page-head">
      <div class="titles">
        <h1>
          <RouterLink :to="`/datasets/${encodeURIComponent(name)}`" class="crumb">{{ name }}</RouterLink>
          Mask
        </h1>
        <p class="lede mono">{{ item?.path }} <span v-if="items.length" class="muted">· {{ index + 1 }} / {{ items.length }}</span></p>
      </div>
      <ZenButton icon="mdi mdi-chevron-left" :disabled="index <= 0" @click="go(-1)">Previous</ZenButton>
      <ZenButton icon="mdi mdi-chevron-right" :disabled="index >= items.length - 1" @click="go(1)">Next</ZenButton>
      <ZenButton icon="mdi mdi-content-save-outline" :disabled="busy || !loaded || !(dirty || grow || feather)" title="Save this mask (Ctrl+S)" @click="save">{{ dirty || grow || feather ? 'Save' : 'Saved' }}</ZenButton>
      <ZenButton variant="primary" icon="mdi mdi-content-save-move-outline" :disabled="busy || !loaded || index >= items.length - 1" title="Save this mask, then the next image (Ctrl+→)" @click="saveNext">Save &amp; next</ZenButton>
    </div>

    <div class="editor">
      <aside class="list card">
        <div class="list-search">
          <i class="mdi mdi-magnify" />
          <input v-model="search" placeholder="Search names and captions…" spellcheck="false" />
        </div>
        <div class="chips">
          <button v-for="(label, k) in FILTERS" :key="k" class="chip" :class="{ on: filter === k }" @click="filter = k">
            {{ label }}<template v-if="k === 'selected' && selected.size"> · {{ selected.size }}</template>
          </button>
        </div>
        <div class="list-tools small">
          <span class="muted">{{ shown.length }} shown</span>
          <span class="spacer" />
          <button class="link" :disabled="!shown.length" @click="selectShown">Select shown</button>
          <button v-if="selected.size" class="link" @click="selected = new Set()">Clear</button>
        </div>
        <div class="tiles">
          <button
            v-for="it in shown"
            :key="it.path"
            class="tile"
            :class="{ current: it.path === path, sel: selected.has(it.path), other: item && (it.width !== item.width || it.height !== item.height) }"
            :title="`${it.filename} · ${it.width}×${it.height}${it.mask ? ' · masked' : ''}. Click opens it; Ctrl/Shift-click or the tick selects`"
            @click="pick(it, $event)"
          >
            <img :src="thumb(it.path, 160)" loading="lazy" alt="" />
            <img v-if="it.mask" class="tile-mask" :src="thumb(it.mask, 64)" loading="lazy" alt="" />
            <span class="tick" @click.stop="pick(it, $event, true)"><i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" /></span>
          </button>
          <p v-if="!shown.length" class="muted small">Nothing matches.</p>
        </div>
      </aside>
      <div ref="stage" class="stage card" :class="tool" @contextmenu.prevent>
        <canvas ref="view" @pointerdown="down" @pointermove="move" @pointerup="up" @pointerleave="(cursor = null), up()" />
        <span
          v-if="cursor && tool !== 'points'"
          class="brush-ring"
          :style="{ left: `${cursor.x}px`, top: `${cursor.y}px`, width: `${brush * fit.s}px`, height: `${brush * fit.s}px` }"
        />
        <div v-if="!loaded && !error" class="loading muted small"><i class="mdi mdi-loading mdi-spin" /> Loading…</div>
      </div>

      <aside class="side">
        <section class="card grp">
          <div class="eyebrow">Tools</div>
          <div class="seg">
            <button :class="{ on: tool === 'points' }" title="P" @click="tool = 'points'"><i class="mdi mdi-gesture-tap" /> Points</button>
            <button :class="{ on: tool === 'erase' }" title="Paint what's left out of training (E)" @click="tool = 'erase'"><i class="mdi mdi-eraser" /> Mask out</button>
            <button :class="{ on: tool === 'brush' }" title="Paint back what trains (B)" @click="tool = 'brush'"><i class="mdi mdi-brush" /> Paint back</button>
            <button :class="{ on: tool === 'rect' }" title="R: drag a box to take it out (a watermark)" @click="tool = 'rect'"><i class="mdi mdi-selection" /> Box</button>
          </div>
          <p v-if="tool === 'rect'" class="muted small">Drag a box over what shouldn't train (a watermark, a caption). On an empty mask the rest of the picture is kept.</p>
          <label v-if="tool !== 'points'" class="f small">
            <span class="muted">Brush size {{ brush }} px · [ ]</span>
            <ZenSlider v-model="brush" :min="4" :max="400" :step="2" />
          </label>
          <div class="row">
            <button class="mini" @click="invert"><i class="mdi mdi-invert-colors" /> Invert</button>
            <button class="mini" title="Everything trains again: no part left out" @click="fill(true)">Keep all</button>
            <button class="mini" title="Leave everything out, to paint back only what should train" @click="fill(false)">Drop all</button>
          </div>
        </section>
        <section class="card grp">
          <div class="eyebrow">Find it (SAM3)</div>
          <div class="row">
            <ZenInput v-model="text" size="sm" placeholder="what to keep, e.g. person" @keydown.enter="detect" />
            <ZenButton size="sm" icon="mdi mdi-magnify" :disabled="busy || !text.trim()" @click="detect">Find</ZenButton>
          </div>
          <p class="muted small">Or <b>Points</b>: click keeps, right-click excludes, click a dot to remove it.</p>
          <label class="f small">
            <span class="muted">Confidence {{ threshold.toFixed(2) }}</span>
            <ZenSlider v-model="threshold" :min="0.1" :max="0.9" :step="0.05" />
          </label>
          <div v-if="points.length" class="row small">
            <span class="muted">{{ points.filter((p) => p.keep).length }} keep · {{ points.filter((p) => !p.keep).length }} exclude</span>
            <button class="link" @click="(points = []), render()">Clear points</button>
          </div>
        </section>
        <section class="card grp">
          <div class="eyebrow">Mask</div>
          <div class="seg small">
            <button :class="{ on: show === 'overlay' }" @click="show = 'overlay'">Overlay</button>
            <button :class="{ on: show === 'mask' }" @click="show = 'mask'">Mask only</button>
          </div>
          <label v-if="show === 'overlay'" class="row small" title="What's left out as a magenta checker, for edges against dark or busy pictures">
            <ZenSwitch v-model="highVis" /> High-vis
          </label>
          <label class="f small">
            <span class="muted">{{ grow >= 0 ? 'Grow' : 'Shrink' }} {{ Math.abs(grow) }} px</span>
            <ZenSlider v-model="grow" :min="-60" :max="60" :step="1" />
          </label>
          <label class="f small">
            <span class="muted">Feather {{ feather }} px</span>
            <ZenSlider v-model="feather" :min="0" :max="60" :step="1" />
          </label>
          <div v-if="status" class="small muted">{{ status }}</div>
          <div v-if="error" class="small err">{{ error }}</div>
          <div class="row small">
            <span class="muted">{{ coverage != null ? `${Math.round(coverage * 100)}% trains at full weight` : '' }}</span>
            <span class="spacer" />
            <button v-if="hasMask" class="link danger" :disabled="busy" @click="remove">Delete mask</button>
          </div>
        </section>
        <section v-if="item" class="card grp">
          <div class="eyebrow">This size · {{ item.width }}×{{ item.height }}</div>
          <p class="muted small">{{ sameSize.length }} image{{ sameSize.length === 1 ? '' : 's' }}, {{ sameSize.filter((i) => i.mask).length }} masked. Apply this mask to the ones you select</p>
          <template v-if="sameMasks.length">
            <span class="muted small">Start from one of their masks:</span>
            <div class="mask-picks">
              <button v-for="it in sameMasks.slice(0, 12)" :key="it.path" class="mask-pick" :title="`Use ${it.filename}'s mask`" @click="useMask(it)">
                <img :src="thumb(it.mask!, 128)" loading="lazy" alt="" />
              </button>
            </div>
          </template>
          <div class="row">
            <button class="mini" @click="selectSame()">Select this size</button>
            <button class="mini" @click="selectSame(true)">Unmasked only</button>
            <button v-if="selected.size" class="mini" @click="selected = new Set()">Clear</button>
          </div>
          <ZenButton icon="mdi mdi-checkbox-multiple-marked-outline" :disabled="busy || !loaded || !selectedSame.length" @click="openApply">
            Apply to {{ selectedSame.length }} selected…
          </ZenButton>
          <div v-if="selected.size > selectedSame.length" class="muted small">{{ selected.size - selectedSame.length }} selected are another size and are left out</div>
          <div v-if="lastBatch" class="row small">
            <span class="muted">Applied to {{ lastBatch.count }}</span>
            <button class="link" :disabled="busy" @click="undoApply">Undo</button>
          </div>
        </section>
      </aside>
    </div>

    <ZenModal v-model:open="applyOpen" :title="`Apply this mask to ${selectedSame.length} image${selectedSame.length === 1 ? '' : 's'}`" width="880px">
      <div class="apply">
        <div class="seg">
          <button :class="{ on: applyMode === 'cutout' }" @click="applyMode = 'cutout'">Cut out of their masks</button>
          <button :class="{ on: applyMode === 'replace' }" @click="applyMode = 'replace'">Replace their masks</button>
        </div>
        <p class="muted small">
          {{ applyMode === 'cutout' ? 'Images with a mask keep it; what this mask leaves out comes out of theirs. Images without one get this mask.' : 'Every selected image gets exactly this mask.' }}
          Their current masks are kept aside, so Undo puts them back. Shaded areas won't train.
        </p>
        <div class="previews">
          <figure v-for="pv in previews" :key="pv.path" class="pv">
            <img :src="pv.url" alt="" />
            <figcaption class="small mono">{{ pv.path.split('/').pop() }}<span v-if="pv.had" class="muted"> · had a mask</span></figcaption>
          </figure>
          <div v-if="previews.length < Math.min(selectedSame.length, PREVIEW_MAX)" class="muted small"><i class="mdi mdi-loading mdi-spin" /> Previewing…</div>
          <div v-if="selectedSame.length > PREVIEW_MAX" class="muted small">and {{ selectedSame.length - PREVIEW_MAX }} more</div>
        </div>
      </div>
      <template #footer>
        <ZenButton @click="applyOpen = false">Cancel</ZenButton>
        <ZenButton variant="primary" icon="mdi mdi-check" :disabled="busy" @click="doApply">Apply to {{ selectedSame.length }}</ZenButton>
      </template>
    </ZenModal>
  </div>
</template>

<style scoped>
/* exactly the window's height (the scroll area's content is as tall as the page, so 100% isn't
   a bound): the side panels scroll in their columns */
.mask-page {
  box-sizing: border-box;
  height: 100dvh;
  padding-bottom: 16px;
  display: flex;
  flex-direction: column;
}

.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
.editor {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 236px minmax(0, 1fr) 300px;
  grid-template-rows: minmax(0, 1fr); /* the side panels scroll; they don't stretch the page */
  gap: 14px;
}
.stage {
  position: relative;
  padding: 0;
  overflow: hidden;
  min-height: 420px;
  background: repeating-conic-gradient(color-mix(in srgb, var(--zen-text) 6%, transparent) 0% 25%, transparent 0% 50%) 50% / 24px 24px;
}
.stage canvas {
  position: absolute;
  inset: 0;
  touch-action: none;
}
.stage.points canvas {
  cursor: crosshair;
}
.stage.brush canvas,
.stage.erase canvas {
  cursor: none;
}
.brush-ring {
  position: absolute;
  transform: translate(-50%, -50%);
  border-radius: 50%;
  border: 1.5px solid #fff;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.5);
  pointer-events: none;
}
.loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
}
.side {
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
  scrollbar-width: thin;
}
.grp {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
}
.row {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}
.row :deep(.zen-input) {
  flex: 1;
}
.f {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.seg {
  display: flex;
  gap: 4px;
}
.seg button {
  flex: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  white-space: nowrap;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 5px 4px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.seg button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.mini {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 8px;
  border-radius: var(--r-sm);
  cursor: pointer;
}
.mini:hover {
  color: var(--zen-text);
  border-color: var(--zen-accent);
}
.link {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  color: var(--zen-accent);
  cursor: pointer;
}
.link.danger {
  color: var(--zen-muted);
}
.link.danger:hover {
  color: var(--zen-error, #e5484d);
}
.small {
  font-size: 12px;
}
.err {
  color: var(--zen-danger);
}
.mask-picks {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(52px, 1fr));
  gap: 6px;
}
.mask-pick {
  padding: 0;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-sm);
  overflow: hidden;
  background: #000;
  cursor: pointer;
  aspect-ratio: 1;
}
.mask-pick:hover {
  border-color: var(--zen-accent);
}
.mask-pick img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px;
  min-height: 0;
}
.list-search {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 8px;
  height: 30px;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
  color: var(--zen-muted);
}
.list-search:focus-within {
  outline: 1px solid var(--zen-accent);
}
.list-search input {
  flex: 1;
  min-width: 0;
  border: none;
  background: none;
  outline: none;
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.chip {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: var(--r-sm);
  cursor: pointer;
}
.chip.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  border-color: var(--zen-selected, var(--zen-accent));
}
.list-tools {
  display: flex;
  align-items: center;
  gap: 10px;
}
.tiles {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scrollbar-width: thin;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(64px, 1fr));
  grid-auto-rows: min-content;
  gap: 6px;
  align-content: start;
}
.tile {
  position: relative;
  aspect-ratio: 1;
  padding: 0;
  border: 2px solid transparent;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: pointer;
}
.tile > img:first-child {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.tile.other {
  opacity: 0.4;
}
.tile.current {
  border-color: var(--zen-selected, var(--zen-accent));
}
.tile.sel {
  border-color: color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 55%, transparent);
}
.tile.current.sel {
  border-color: var(--zen-selected, var(--zen-accent));
}
.tile-mask {
  position: absolute;
  right: 3px;
  bottom: 3px;
  width: 24px;
  height: 24px;
  object-fit: cover;
  border-radius: 3px;
  border: 1px solid rgba(255, 255, 255, 0.6);
}
.tick {
  position: absolute;
  left: 3px;
  top: 3px;
  font-size: 16px;
  line-height: 1;
  color: #fff;
  text-shadow: 0 0 3px rgba(0, 0, 0, 0.8);
}
.tile:not(.sel):not(:hover) .tick {
  opacity: 0.55;
}
.apply {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.previews {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
  max-height: 56vh;
  overflow-y: auto;
  scrollbar-width: thin;
}
.pv {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.pv img {
  width: 100%;
  border-radius: var(--r-sm);
  display: block;
}
.pv figcaption {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
@media (max-width: 1180px) {
  .editor {
    grid-template-columns: 180px minmax(0, 1fr) 270px;
  }
}
@media (max-width: 900px) {
  /* one column: the page scrolls as usual, the canvas keeps a usable height */
  .list {
    max-height: 260px;
  }
  .mask-page {
    height: auto;
  }
  .editor {
    grid-template-columns: 1fr;
    grid-template-rows: none;
  }
  .stage {
    height: 62vh;
  }
}
</style>
