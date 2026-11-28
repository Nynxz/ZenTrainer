<script setup lang="ts">
// Two runs side by side: head-to-head summary strip, full spec diff, overlaid curves (loss,
// validation, likeness, LR and active auxiliary objectives), and synced samples (side by side,
// swipe, or grid) with per-sample likeness badges, lightbox zoom, and themed PNG grid export.
// Keys: ←/→ step, ↑/↓ prompt, S swaps the runs.
import { ZenButton, ZenCombo, ZenCompare, ZenField, ZenLightbox, ZenSlider, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import CompareChart, { type CompareMetric, type ComparePoint, type CompareSeries } from '@/components/CompareChart.vue'
import StatusPill from '@/components/StatusPill.vue'
import SampleBadges from '@/components/run/SampleBadges.vue'
import { dur, fmtLr } from '@/components/run/context'
import { gb } from '@/stores/system'
import { EngineError, engine, viewUrl, type LikenessRow, type MetricRow, type RunDetail, type RunMeta, type SampleRow } from '@/api/engine'

const route = useRoute()
const router = useRouter()

const runs = ref<RunMeta[]>([])
const a = ref<RunDetail | null>(null)
const b = ref<RunDetail | null>(null)
const error = ref<string | null>(null)
const ida = computed({
  get: () => (typeof route.query.a === 'string' ? route.query.a : ''),
  set: (v: string) => router.replace({ query: { ...route.query, a: v || undefined } }),
})
const idb = computed({
  get: () => (typeof route.query.b === 'string' ? route.query.b : ''),
  set: (v: string) => router.replace({ query: { ...route.query, b: v || undefined } }),
})
const swap = () => router.replace({ query: { ...route.query, a: idb.value || undefined, b: ida.value || undefined } })

const fmtDate = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
const runItems = computed(() =>
  runs.value.map((r) => ({
    value: r.run_id,
    label: r.name || r.run_id,
    keywords: `${r.name} ${r.run_id}`,
    hint: `${fmtDate(r.created)} · ${r.status}${r.final_step ? ` · ${r.final_step} st` : ''}`,
  })),
)

async function loadRun(id: string, into: typeof a) {
  into.value = null
  if (!id) return
  try {
    into.value = await engine.run(id)
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  }
}
watch(ida, (v) => loadRun(v, a), { immediate: true })
watch(idb, (v) => loadRun(v, b), { immediate: true })

// colours: read once from the theme, so the export uses the same ones
const css = (name: string, fallback: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
const colorA = ref('#f0ff41')
const colorB = ref('#5aa9ff')
const nameA = computed(() => a.value?.run.name || 'Run A')
const nameB = computed(() => (b.value?.run.name && b.value.run.name !== nameA.value ? b.value.run.name : 'Run B'))

// ---- full spec diff: walks every field so new options never go unseen -----------------------
const LABELS: Record<string, string> = {
  'model.kind': 'Model kind',
  'model.unet_name': 'Model',
  'model.ckpt_name': 'Checkpoint',
  'model.vae_name': 'VAE',
  'model.clip_names': 'Text encoders',
  'model.clip_type': 'Text encoder type',
  'dataset.folder': 'Dataset',
  'dataset.caption_set': 'Caption set',
  'dataset.trigger': 'Trigger',
  'dataset.folder_triggers': 'Folder triggers',
  'dataset.resolution': 'Resolution',
  'dataset.stages': 'Resolution stages',
  'dataset.stage_blend': 'Stage blend',
  'dataset.repeats': 'Repeats',
  'dataset.flip_h': 'Horizontal flip',
  'dataset.caption_dropout': 'Caption dropout',
  'dataset.masked_loss': 'Masked training',
  'dataset.face_weight': 'Face weighting',
  'optim.steps': 'Steps',
  'optim.batch_size': 'Batch size',
  'optim.grad_accumulation_steps': 'Grad accumulation',
  'optim.optimizer': 'Optimizer',
  'optim.learning_rate': 'Learning rate',
  'optim.lr_scheduler': 'Schedule',
  'optim.warmup_steps': 'Warmup steps',
  'optim.lr_floor': 'LR floor',
  'optim.weight_decay': 'Weight decay',
  'optim.grad_clip': 'Grad clip',
  'optim.ema_decay': 'EMA decay',
  'optim.lotus_auto_lr': 'Lotus adaptive LR',
  'network.algorithm': 'Algorithm',
  'network.rank': 'Rank',
  'network.alpha': 'Alpha',
  'network.targets': 'Target layers',
  'network.blocks': 'Blocks',
  'network.dropout': 'LoRA dropout',
  'network.base_lora': 'Base LoRA',
  'network.base_strength': 'Base LoRA strength',
  'network.training_adapter': 'Training adapter',
  'objective.timesteps': 'Timesteps',
  'objective.loss_type': 'Loss type',
  'objective.min_snr_gamma': 'Min-SNR gamma',
  'objective.noise_offset': 'Noise offset',
  'objective.preserve': 'Preserve classes (DOP)',
  'objective.preserve_weight': 'DOP weight',
  'objective.preserve_every': 'DOP every',
  'objective.haar_hf': 'High-freq detail (Haar)',
  'memory.training_dtype': 'Precision',
  'memory.offloading': 'Offloading',
  'memory.gradient_checkpointing': 'Grad checkpointing',
  'memory.grad_checkpoint_skip': 'Checkpoint keep',
  'memory.checkpoint_save': 'Checkpoint save',
  'memory.compile_blocks': 'Compile blocks',
  'memory.int8_forward': 'int8 forward',
  'memory.int8_backward': 'int8 backward',
  'memory.fused_qk_rope': 'Fused Q/K RoPE',
  'memory.tread': 'TREAD',
  'samples.every': 'Sample every',
  'samples.steps': 'Sample steps',
  'samples.cfg': 'Sample CFG',
  'samples.sampler': 'Sampler',
  'samples.scheduler': 'Sample scheduler',
  'samples.model_name': 'Preview model',
  'samples.turbo_lora': 'Preview turbo LoRA',
  'samples.turbo_strength': 'Preview turbo strength',
  'validation.every': 'Validation every',
  'validation.count': 'Validation count',
  'validation.dataset': 'Validation dataset',
  'output.save_every': 'Save every',
  'output.keep': 'Keep checkpoints',
  'seed': 'Seed',
}

const SKIP_PATHS = new Set(['output.run_name', 'output.lora_name', 'output.save_dir', 'samples.prompts'])
const PRIORITY = [
  'model.unet_name',
  'model.ckpt_name',
  'dataset.folder',
  'dataset.resolution',
  'optim.optimizer',
  'optim.lotus_auto_lr',
  'optim.learning_rate',
  'optim.steps',
  'optim.batch_size',
  'optim.grad_accumulation_steps',
  'optim.lr_scheduler',
  'optim.warmup_steps',
  'network.algorithm',
  'network.rank',
  'network.alpha',
  'network.targets',
  'network.blocks',
  'network.training_adapter',
  'network.base_lora',
  'objective.timesteps',
  'objective.preserve',
  'objective.preserve_weight',
  'objective.preserve_every',
  'objective.haar_hf',
  'memory.training_dtype',
  'seed',
]

function prettyPath(path: string): string {
  if (LABELS[path]) return LABELS[path]
  const parts = path.split('.')
  const head = parts.length > 1 ? parts[0]!.replace(/^./, (c) => c.toUpperCase()) : ''
  const tail = parts.at(-1)!.replace(/_/g, ' ')
  return head ? `${head} · ${tail}` : tail.replace(/^./, (c) => c.toUpperCase())
}

// Mask sub-fields whose parent feature is turned off so stale inactive defaults never pollute a diff
function effectiveVal(spec: Record<string, unknown>, path: string, raw: unknown): unknown {
  const obj = (spec.objective ?? {}) as Record<string, unknown>
  const opt = (spec.optim ?? {}) as Record<string, unknown>
  const net = (spec.network ?? {}) as Record<string, unknown>
  const ds = (spec.dataset ?? {}) as Record<string, unknown>
  const mem = (spec.memory ?? {}) as Record<string, unknown>
  const samp = (spec.samples ?? {}) as Record<string, unknown>
  if ((path === 'objective.preserve_weight' || path === 'objective.preserve_every') && !obj.preserve) return null
  if (path === 'optim.lotus_auto_lr' && opt.optimizer !== 'Lotus') return null
  if (path === 'network.base_strength' && !net.base_lora) return null
  if (path === 'dataset.stage_blend' && (!Array.isArray(ds.stages) || !ds.stages.length)) return null
  if ((path === 'memory.grad_checkpoint_skip' || path === 'memory.checkpoint_save') && !mem.gradient_checkpointing) return null
  if (path === 'samples.turbo_strength' && !samp.turbo_lora) return null
  return raw
}

function flattenSpec(obj: unknown, prefix = '', out = new Map<string, unknown>()): Map<string, unknown> {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return out
  for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
    const p = prefix ? `${prefix}.${k}` : k
    if (SKIP_PATHS.has(p)) continue
    if (v != null && typeof v === 'object' && !Array.isArray(v)) flattenSpec(v, p, out)
    else out.set(p, v)
  }
  return out
}

function formatSpecVal(v: unknown, path = ''): string {
  if (v == null || v === '') return '–'
  if (typeof v === 'boolean') return v ? 'on' : 'off'
  if (Array.isArray(v)) {
    if (!v.length) return '–'
    if (path === 'dataset.stages') {
      return v
        .map((st) => {
          const o = st as { resolution?: number; until?: number }
          return o.resolution != null ? `${o.resolution}px @ ${Math.round((o.until ?? 0) * 100)}%` : JSON.stringify(st)
        })
        .join(', ')
    }
    return v.map((x) => (typeof x === 'object' && x != null ? JSON.stringify(x) : String(x))).join(', ')
  }
  if (typeof v === 'number' && path === 'optim.learning_rate') return fmtLr(v)
  return String(v)
}

const onlyDiff = ref(true)
const allSetupRows = computed(() => {
  if (!a.value || !b.value) return []
  const sa = a.value.spec as unknown as Record<string, unknown>
  const sb = b.value.spec as unknown as Record<string, unknown>
  const fa = flattenSpec(sa)
  const fb = flattenSpec(sb)
  const paths = [...new Set([...fa.keys(), ...fb.keys()])]
  paths.sort((p, q) => {
    const ip = PRIORITY.indexOf(p)
    const iq = PRIORITY.indexOf(q)
    if (ip !== -1 && iq !== -1) return ip - iq
    if (ip !== -1) return -1
    if (iq !== -1) return 1
    return p.localeCompare(q)
  })
  const rows: { path: string; k: string; va: string; vb: string; diff: boolean }[] = []
  for (const path of paths) {
    const ea = effectiveVal(sa, path, fa.get(path))
    const eb = effectiveVal(sb, path, fb.get(path))
    if (ea == null && eb == null) continue
    const va = formatSpecVal(ea, path)
    const vb = formatSpecVal(eb, path)
    if (va === '–' && vb === '–') continue
    rows.push({ path, k: prettyPath(path), va, vb, diff: va !== vb })
  }
  // Summarize sample prompts if count or prompt texts differ
  const pa = a.value.spec.samples?.prompts ?? []
  const pb = b.value.spec.samples?.prompts ?? []
  const pSig = (ps: typeof pa) =>
    ps.map((x) => (typeof x === 'string' ? `${x}|true` : `${x.prompt}|${x.likeness ?? true}`)).join('\n')
  const pDiff = pSig(pa) !== pSig(pb)
  if (pDiff || pa.length || pb.length) {
    rows.push({
      path: 'samples.prompts',
      k: 'Sample prompts',
      va: `${pa.length} prompt${pa.length === 1 ? '' : 's'}`,
      vb: `${pb.length} prompt${pb.length === 1 ? '' : 's'}${pDiff && pa.length === pb.length ? ' (text differs)' : ''}`,
      diff: pDiff,
    })
  }
  return rows
})
const diffCount = computed(() => allSetupRows.value.filter((r) => r.diff).length)
const setup = computed(() => (onlyDiff.value ? allSetupRows.value.filter((r) => r.diff) : allSetupRows.value))

// ---- head-to-head KPI summary ---------------------------------------------------------------
function emaSeries(rows: MetricRow[], pick: (r: MetricRow) => number | null | undefined, alpha = 0.05): ComparePoint[] {
  const out: ComparePoint[] = []
  let s: number | null = null
  for (const r of rows) {
    const v = pick(r)
    if (v == null || !Number.isFinite(v)) continue
    s = s == null ? v : (1 - alpha) * s + alpha * v
    out.push({ step: r.step, v: s })
  }
  return out
}

function rawSeries(rows: MetricRow[], pick: (r: MetricRow) => number | null | undefined): ComparePoint[] {
  const out: ComparePoint[] = []
  for (const r of rows) {
    const v = pick(r)
    if (v == null || !Number.isFinite(v)) continue
    out.push({ step: r.step, v })
  }
  return out
}

function likeSteps(d: RunDetail | null): ComparePoint[] {
  if (!d) return []
  const by = new Map<number, number[]>()
  for (const r of d.likeness ?? []) if (!r.manual && r.score != null) by.set(r.step, [...(by.get(r.step) ?? []), r.score])
  return [...by].sort((p, q) => p[0] - q[0]).map(([step, xs]) => ({ step, v: xs.reduce((s, v) => s + v, 0) / xs.length }))
}

function runStats(d: RunDetail | null) {
  if (!d) return null
  const last = d.metrics.at(-1)
  const lossEma = emaSeries(d.metrics, (r) => r.loss).at(-1)?.v ?? null
  const bestVal = d.validation.length ? Math.min(...d.validation.map((v) => v.val_loss)) : null
  const lk = likeSteps(d)
  const peakLike = lk.length ? lk.reduce((best, p) => (p.v > best.v ? p : best), lk[0]!) : null
  const lastLike = lk.at(-1) ?? null
  let vramPeak = 0
  for (const m of d.metrics) if ((m.vram_peak ?? 0) > vramPeak) vramPeak = m.vram_peak!
  const model = (d.spec.model.unet_name ?? d.spec.model.ckpt_name ?? '').replace(/\.safetensors$/, '')
  return {
    step: last?.step ?? d.run.final_step ?? 0,
    total: d.spec.optim.steps,
    it_s: last?.it_s ?? null,
    trainTime: last?.train_time ?? last?.elapsed ?? null,
    vramPeak,
    lossEma,
    bestVal,
    peakLike,
    lastLike,
    model,
    opt: d.spec.optim.optimizer === 'Lotus' && d.spec.optim.lotus_auto_lr ? 'Lotus (auto LR)' : d.spec.optim.optimizer,
  }
}
const statsA = computed(() => runStats(a.value))
const statsB = computed(() => runStats(b.value))

// ---- curves ---------------------------------------------------------------------------------
const metric = ref<CompareMetric>('loss')
const pickAux = (r: MetricRow, key: 'preserve_loss') => (r as unknown as Record<string, number | null | undefined>)[key]

const hasVal = computed(() => !!(a.value?.validation.length || b.value?.validation.length))
const hasLike = computed(() => !!(likeSteps(a.value).length || likeSteps(b.value).length))
const hasPreserve = computed(
  () => !!(a.value?.metrics.some((r) => pickAux(r, 'preserve_loss') != null) || b.value?.metrics.some((r) => pickAux(r, 'preserve_loss') != null)),
)

const metricTabs = computed<{ id: CompareMetric; label: string; disabled?: boolean }[]>(() => [
  { id: 'loss', label: 'Loss' },
  { id: 'val', label: 'Validation', disabled: !hasVal.value },
  { id: 'like', label: 'Likeness', disabled: !hasLike.value },
  { id: 'lr', label: 'LR' },
  ...(hasPreserve.value ? [{ id: 'preserve' as const, label: 'DOP' }] : []),
])

watch(metricTabs, (tabs) => {
  if (!tabs.some((t) => t.id === metric.value && !t.disabled)) metric.value = 'loss'
})

function pointsFor(d: RunDetail, m: CompareMetric): ComparePoint[] {
  switch (m) {
    case 'loss':
      return emaSeries(d.metrics, (r) => r.loss)
    case 'val':
      return d.validation.map((r) => ({ step: r.step, v: r.val_loss }))
    case 'like':
      return likeSteps(d)
    case 'lr':
      return rawSeries(d.metrics, (r) => r.lr)
    case 'preserve':
      return emaSeries(d.metrics, (r) => pickAux(r, 'preserve_loss'), 0.15)
  }
}

const series = computed<CompareSeries[]>(() =>
  [
    a.value && {
      label: nameA.value,
      color: colorA.value,
      maxStep: a.value.metrics.at(-1)?.step ?? a.value.spec.optim.steps,
      pts: pointsFor(a.value, metric.value),
    },
    b.value && {
      label: nameB.value,
      color: colorB.value,
      maxStep: b.value.metrics.at(-1)?.step ?? b.value.spec.optim.steps,
      pts: pointsFor(b.value, metric.value),
    },
  ].filter(Boolean) as CompareSeries[],
)

// ---- samples: prompts matched by text, steps from both, with likeness badges ----------------
const runSamples = (d: RunDetail | null) => (d ? d.samples.filter((s) => !s.manual && !s.base) : [])
const prompts = computed(() => {
  const out: string[] = []
  for (const s of [...runSamples(a.value), ...runSamples(b.value)]) if (!out.includes(s.prompt)) out.push(s.prompt)
  return out
})
const pi = ref(0)
watch(prompts, (ps) => {
  if (pi.value >= ps.length) pi.value = 0
})
const prompt = computed(() => prompts.value[pi.value] ?? '')

function likenessMap(d: RunDetail | null): Map<string, LikenessRow> {
  const m = new Map<string, LikenessRow>()
  for (const r of d?.likeness ?? []) if (!r.manual && !r.base) m.set(`${r.step}:${r.index}`, r)
  return m
}
const likeMapA = computed(() => likenessMap(a.value))
const likeMapB = computed(() => likenessMap(b.value))
const likeCeilA = computed(() => a.value?.likeness?.find((r) => r.ceiling != null)?.ceiling ?? null)
const likeCeilB = computed(() => b.value?.likeness?.find((r) => r.ceiling != null)?.ceiling ?? null)

function isPromptSkipped(d: RunDetail | null, text: string): boolean {
  const row = d?.spec.samples?.prompts?.find((p) => (typeof p === 'string' ? p === text : p.prompt === text))
  return typeof row === 'object' && row != null && row.likeness === false
}

const byStep = (d: RunDetail | null) => {
  const m = new Map<number, SampleRow>()
  for (const s of runSamples(d)) if (s.prompt === prompt.value) m.set(s.step, s)
  return m
}
const stepsA = computed(() => byStep(a.value))
const stepsB = computed(() => byStep(b.value))
const steps = computed(() => [...new Set([...stepsA.value.keys(), ...stepsB.value.keys()])].sort((p, q) => p - q))
const si = ref(0)
watch(steps, (s) => (si.value = Math.max(0, s.length - 1)))
const step = computed(() => steps.value[si.value] ?? 0)

function jumpToStep(target: number) {
  if (!steps.value.length) return
  let bestIdx = 0
  for (let i = 1; i < steps.value.length; i++) {
    if (Math.abs(steps.value[i]! - target) < Math.abs(steps.value[bestIdx]! - target)) bestIdx = i
  }
  si.value = bestIdx
}

// each run's sample nearest the step: runs that sample on different schedules still line up
const nearest = (m: Map<number, SampleRow>, at: number) => {
  let best: SampleRow | null = null
  for (const [st, row] of m) if (!best || Math.abs(st - at) < Math.abs(best.step - at)) best = row
  return best
}
const cellA = computed(() => nearest(stepsA.value, step.value))
const cellB = computed(() => nearest(stepsB.value, step.value))
const likeA = computed(() => (cellA.value ? likeMapA.value.get(`${cellA.value.step}:${cellA.value.index}`) ?? null : null))
const likeB = computed(() => (cellB.value ? likeMapB.value.get(`${cellB.value.step}:${cellB.value.index}`) ?? null : null))
const sampleLikeDelta = computed(() => {
  const sa = likeA.value?.score
  const sb = likeB.value?.score
  if (sa == null || sb == null) return null
  const d = sb - sa
  if (Math.abs(d) < 0.005) return { text: 'Tie', color: 'var(--zen-muted)' }
  return d > 0
    ? { text: `${nameB.value} +${d.toFixed(2)}`, color: colorB.value }
    : { text: `${nameA.value} +${Math.abs(d).toFixed(2)}`, color: colorA.value }
})
const url = (r: SampleRow | null) => (r ? viewUrl(r) : '')

const mode = ref<'side' | 'swipe' | 'grid'>('side')
const GRID_COLS = 6
// the grid's columns: up to six of the steps, evenly spread, always with the first and last
const gridSteps = computed(() => {
  const s = steps.value
  if (s.length <= GRID_COLS) return s
  return Array.from({ length: GRID_COLS }, (_, i) => s[Math.round((i * (s.length - 1)) / (GRID_COLS - 1))]!)
})
const allPrompts = ref(false)
const gridPrompts = computed(() => (allPrompts.value ? prompts.value : [prompt.value]))
const cellFor = (d: RunDetail | null, p: string, at: number) => {
  let best: SampleRow | null = null
  for (const s of runSamples(d)) if (s.prompt === p && (!best || Math.abs(s.step - at) < Math.abs(best.step - at))) best = s
  return best
}

// Lightbox for full-size sample inspection
const lb = ref<{ index: number; items: { src: string; caption?: string }[] } | null>(null)
function openSideLightbox(which: 0 | 1) {
  const items: { src: string; caption?: string }[] = []
  let target = 0
  if (cellA.value) {
    if (which === 0) target = items.length
    items.push({ src: url(cellA.value), caption: `${nameA.value} · step ${cellA.value.step} · ${prompt.value}` })
  }
  if (cellB.value) {
    if (which === 1) target = items.length
    items.push({ src: url(cellB.value), caption: `${nameB.value} · step ${cellB.value.step} · ${prompt.value}` })
  }
  if (items.length) lb.value = { index: target, items }
}
function openGridLightbox(runName: string, cell: SampleRow | null, p: string) {
  if (!cell) return
  lb.value = { index: 0, items: [{ src: url(cell), caption: `${runName} · step ${cell.step} · ${p}` }] }
}

function onKey(e: KeyboardEvent) {
  if (lb.value || (e.target as HTMLElement).closest('input, textarea')) return
  if (e.key === 'ArrowRight') si.value = Math.min(steps.value.length - 1, si.value + 1)
  else if (e.key === 'ArrowLeft') si.value = Math.max(0, si.value - 1)
  else if (e.key === 'ArrowDown') pi.value = Math.min(prompts.value.length - 1, pi.value + 1)
  else if (e.key === 'ArrowUp') pi.value = Math.max(0, pi.value - 1)
  else if (e.key === 's') swap()
  else return
  e.preventDefault()
}

// ---- export: the grid as a themed PNG ---------------------------------------------------------
const exporting = ref(false)
const LOTUS = [
  ['M0,0 C24,-16.2 16.8,-42.12 0,-54 C-16.8,-42.12 -24,-16.2 0,0Z', -70, 0.4],
  ['M0,0 C24,-16.2 16.8,-42.12 0,-54 C-16.8,-42.12 -24,-16.2 0,0Z', 70, 0.4],
  ['M0,0 C26,-19.8 18.2,-51.48 0,-66 C-18.2,-51.48 -26,-19.8 0,0Z', -36, 0.7],
  ['M0,0 C26,-19.8 18.2,-51.48 0,-66 C-18.2,-51.48 -26,-19.8 0,0Z', 36, 0.7],
  ['M0,0 C28,-23.4 19.6,-60.84 0,-78 C-19.6,-60.84 -28,-23.4 0,0Z', 0, 1],
] as const
const loadImg = (src: string) =>
  new Promise<HTMLImageElement | null>((res) => {
    const im = new Image()
    im.onload = () => res(im)
    im.onerror = () => res(null)
    im.src = src
  })
function cover(c: CanvasRenderingContext2D, im: HTMLImageElement, x: number, y: number, w: number, h: number) {
  const s = Math.max(w / im.naturalWidth, h / im.naturalHeight)
  const sw = w / s
  const sh = h / s
  c.drawImage(im, (im.naturalWidth - sw) / 2, (im.naturalHeight - sh) / 2, sw, sh, x, y, w, h)
}
function roundRect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  c.beginPath()
  c.roundRect(x, y, w, h, r)
}
async function exportGrid() {
  if (!a.value || !b.value) return
  exporting.value = true
  try {
    const bg = css('--zen-bg', '#1a1320')
    const surface = css('--zen-surface', '#211927')
    const text = css('--zen-text', '#f4eef7')
    const muted = css('--zen-muted', '#bcaecb')
    const font = getComputedStyle(document.body).fontFamily || 'sans-serif'
    const cols = gridSteps.value
    const rows = gridPrompts.value.flatMap((p) => [
      { p, d: a.value!, name: nameA.value, color: colorA.value },
      { p, d: b.value!, name: nameB.value, color: colorB.value },
    ])
    const T = 300
    const G = 12
    const L = 220
    const HEAD = 150
    const COLHEAD = 36
    const PGAP = 26
    const M = 48
    const W = M * 2 + L + cols.length * (T + G) - G
    const H = M + HEAD + COLHEAD + rows.length * (T + G) + (gridPrompts.value.length - 1) * PGAP + 40 + M
    const cv = document.createElement('canvas')
    cv.width = W
    cv.height = H
    const c = cv.getContext('2d')!
    const g = c.createLinearGradient(0, 0, W, H)
    g.addColorStop(0, bg)
    g.addColorStop(0.5, surface)
    g.addColorStop(1, bg)
    c.fillStyle = g
    c.fillRect(0, 0, W, H)
    c.save()
    c.translate(M + 34, M + 70)
    c.scale(0.62, 0.62)
    for (const [d, rot, op] of LOTUS) {
      c.save()
      c.rotate((rot * Math.PI) / 180)
      c.globalAlpha = op
      c.fillStyle = colorA.value
      c.fill(new Path2D(d))
      c.restore()
    }
    c.restore()
    c.fillStyle = colorA.value
    c.font = `800 18px ${font}`
    c.fillText('ZENTRAINER', M + 82, M + 40)
    c.fillStyle = text
    c.font = `700 40px ${font}`
    const vs = `${nameA.value}  vs  ${nameB.value}`
    c.fillText(vs.length > 64 ? `${vs.slice(0, 63)}…` : vs, M + 82, M + 86)
    c.fillStyle = muted
    c.font = `400 18px ${font}`
    const model = (s: RunDetail['spec']) => (s.model.unet_name ?? s.model.ckpt_name ?? '').replace(/\.safetensors$/, '')
    const sub = model(a.value.spec) === model(b.value.spec) ? model(a.value.spec) : `${model(a.value.spec)} / ${model(b.value.spec)}`
    c.fillText(`${sub}${gridPrompts.value.length === 1 ? `  ·  “${gridPrompts.value[0]!.slice(0, 90)}”` : ''}`, M + 82, M + 118)
    const top = M + HEAD
    c.font = `600 15px ${font}`
    c.fillStyle = muted
    cols.forEach((st, i) => c.fillText(`STEP ${st}`, M + L + i * (T + G), top + 20))
    let yy = top + COLHEAD
    for (let r = 0; r < rows.length; r++) {
      const row = rows[r]!
      if (r && r % 2 === 0) yy += PGAP
      c.fillStyle = row.color
      roundRect(c, M, yy + 8, 6, 34, 3)
      c.fill()
      c.fillStyle = text
      c.font = `700 19px ${font}`
      c.fillText(row.name.length > 18 ? `${row.name.slice(0, 17)}…` : row.name, M + 18, yy + 32)
      if (gridPrompts.value.length > 1 && r % 2 === 0) {
        c.fillStyle = muted
        c.font = `400 13px ${font}`
        const words = row.p.split(' ')
        let line = ''
        let ly = yy + 62
        for (const w of words) {
          if (c.measureText(`${line} ${w}`).width > L - 30) {
            c.fillText(line, M + 18, ly)
            line = w
            ly += 17
            if (ly > yy + T - 10) break
          } else line = line ? `${line} ${w}` : w
        }
        if (ly <= yy + T - 10) c.fillText(line, M + 18, ly)
      }
      for (let i = 0; i < cols.length; i++) {
        const x0 = M + L + i * (T + G)
        const cell = cellFor(row.d, row.p, cols[i]!)
        c.save()
        roundRect(c, x0, yy, T, T, 14)
        c.clip()
        c.fillStyle = surface
        c.fillRect(x0, yy, T, T)
        const im = cell ? await loadImg(viewUrl(cell)) : null
        if (im) cover(c, im, x0, yy, T, T)
        c.restore()
        if (cell && cell.step !== cols[i]) {
          c.fillStyle = 'rgba(0,0,0,0.55)'
          roundRect(c, x0 + 8, yy + T - 30, 90, 22, 6)
          c.fill()
          c.fillStyle = '#fff'
          c.font = `500 12px ${font}`
          c.fillText(`step ${cell.step}`, x0 + 16, yy + T - 14)
        }
      }
      yy += T + G
    }
    c.fillStyle = muted
    c.font = `400 13px ${font}`
    c.fillText(`ZenTrainer · ${new Date().toLocaleDateString()}`, M, H - M + 6)
    const blob = await new Promise<Blob | null>((res) => cv.toBlob(res, 'image/png'))
    if (!blob) throw new Error("couldn't render the image")
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `${nameA.value}-vs-${nameB.value}.png`.replace(/[^\w.-]+/g, '_')
    link.click()
    setTimeout(() => URL.revokeObjectURL(link.href), 5000)
  } catch (e) {
    error.value = String(e)
  } finally {
    exporting.value = false
  }
}

// B's colour: the theme's info colour when it reads as another hue than the accent (some themes
// make them alike), else a cool or a warm one, whichever is further from the accent
function hue(color: string): number | null {
  const c = document.createElement('canvas').getContext('2d')!
  c.fillStyle = '#000'
  c.fillStyle = color
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(c.fillStyle as string)
  if (!m) return null
  const [r, g, b] = [1, 2, 3].map((i) => parseInt(m[i]!, 16) / 255) as [number, number, number]
  const max = Math.max(r, g, b)
  const d = max - Math.min(r, g, b)
  if (d < 0.08) return null
  const h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4
  return (h * 60 + 360) % 360
}
const hueGap = (p: number | null, q: number | null) => (p == null || q == null ? 0 : Math.min(Math.abs(p - q), 360 - Math.abs(p - q)))
function pickB(accent: string, info: string): string {
  const ha = hue(accent)
  if (hue(info) != null && hueGap(ha, hue(info)) >= 70) return info
  return hueGap(ha, hue('#38bdf8')) >= hueGap(ha, hue('#f472b6')) ? '#38bdf8' : '#f472b6'
}

onMounted(async () => {
  colorA.value = css('--zen-accent', colorA.value)
  colorB.value = pickB(colorA.value, css('--zen-info', ''))
  addEventListener('keydown', onKey)
  try {
    runs.value = (await engine.runs()).sort((p, q) => q.created - p.created)
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  }
})
onUnmounted(() => removeEventListener('keydown', onKey))
</script>

<template>
  <div class="page full cmp">
    <div class="page-head">
      <div class="titles">
        <h1>Compare runs</h1>
        <p class="lede">Side-by-side settings diff, curves and samples. <span class="kbd-hint">←/→ step · ↑/↓ prompt · S swap</span></p>
      </div>
    </div>

    <section class="card pickers">
      <ZenField label="Run A" stack>
        <div class="pick"><i class="sw" :style="{ background: colorA }" /><ZenCombo v-model="ida" :items="runItems" placeholder="Pick a run…" :sorted="false" :item-height="30" /></div>
      </ZenField>
      <ZenButton icon="mdi mdi-swap-horizontal" title="Swap runs (S)" :disabled="!ida && !idb" @click="swap" />
      <ZenField label="Run B" stack>
        <div class="pick"><i class="sw" :style="{ background: colorB }" /><ZenCombo v-model="idb" :items="runItems" placeholder="Pick a run…" :sorted="false" :item-height="30" /></div>
      </ZenField>
    </section>
    <div v-if="error" class="err small">{{ error }}</div>

    <template v-if="a && b && statsA && statsB">
      <!-- Head-to-head run summary cards -->
      <div class="summary-grid">
        <section
          v-for="[d, st, name, color] in ([[a, statsA, nameA, colorA], [b, statsB, nameB, colorB]] as const)"
          :key="d.run.run_id"
          class="card run-card"
          :style="{ '--run-color': color }"
        >
          <div class="run-card-head">
            <span class="run-dot" :style="{ background: color }" />
            <RouterLink :to="`/runs/${encodeURIComponent(d.run.run_id)}`" class="run-link" :title="d.run.run_id">
              {{ name }}
              <i class="mdi mdi-arrow-top-right" />
            </RouterLink>
            <StatusPill :status="d.run.status" />
            <span class="spacer" />
            <span class="mono small muted">{{ st.step }} / {{ st.total }}</span>
          </div>
          <div class="run-sub small muted">
            <span :title="st.model">{{ st.model }}</span>
            <span>·</span>
            <span>{{ st.opt }}</span>
          </div>
          <div class="kpi-row">
            <div class="kpi">
              <span class="kpi-k">Loss (EMA)</span>
              <span class="kpi-v mono">{{ st.lossEma != null ? st.lossEma.toFixed(4) : '–' }}</span>
            </div>
            <div v-if="hasVal" class="kpi">
              <span class="kpi-k">Best val</span>
              <span class="kpi-v mono">{{ st.bestVal != null ? st.bestVal.toFixed(4) : '–' }}</span>
            </div>
            <div v-if="hasLike" class="kpi">
              <span class="kpi-k">Peak likeness</span>
              <span class="kpi-v mono">
                <template v-if="st.peakLike">{{ st.peakLike.v.toFixed(2) }} <span class="muted small">@ {{ st.peakLike.step }}</span></template>
                <template v-else>–</template>
              </span>
            </div>
            <div class="kpi">
              <span class="kpi-k">Speed</span>
              <span class="kpi-v mono">{{ st.it_s != null ? `${st.it_s.toFixed(2)} it/s` : '–' }}</span>
            </div>
            <div class="kpi">
              <span class="kpi-k">Train time</span>
              <span class="kpi-v mono">{{ dur(st.trainTime ?? undefined) }}</span>
            </div>
            <div v-if="st.vramPeak" class="kpi">
              <span class="kpi-k">Peak VRAM</span>
              <span class="kpi-v mono">{{ gb(st.vramPeak) }} GB</span>
            </div>
          </div>
        </section>
      </div>

      <!-- Setup diff + Overlaid curves -->
      <div class="two">
        <section class="card sec">
          <div class="sec-head">
            <span class="sec-title">Setup</span>
            <span class="count-pill mono" :class="{ accent: diffCount > 0 }">
              {{ diffCount === 0 ? 'identical' : `${diffCount} diff${diffCount === 1 ? '' : 's'}` }}
            </span>
            <span class="spacer" />
            <label class="small muted toggle">
              <ZenSwitch v-model="onlyDiff" size="sm" />
              <span>Only differences</span>
            </label>
          </div>
          <div class="setup-wrap">
            <table class="setup">
              <thead>
                <tr>
                  <th>Setting</th>
                  <th :style="{ color: colorA }">{{ nameA }}</th>
                  <th :style="{ color: colorB }">{{ nameB }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in setup" :key="r.path" :class="{ diff: r.diff }">
                  <td class="muted" :title="r.path">{{ r.k }}</td>
                  <td class="mono">{{ r.va }}</td>
                  <td class="mono">{{ r.vb }}</td>
                </tr>
                <tr v-if="!setup.length">
                  <td colspan="3" class="muted empty-row">Both runs share the exact same settings.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="card sec">
          <div class="sec-head">
            <span class="sec-title">Curves</span>
            <span class="spacer" />
            <div class="seg">
              <button
                v-for="t in metricTabs"
                :key="t.id"
                :class="{ on: metric === t.id }"
                :disabled="t.disabled"
                @click="metric = t.id"
              >
                {{ t.label }}
              </button>
            </div>
          </div>
          <CompareChart :series="series" :metric="metric" :height="245" :active-step="step || null" @pick-step="jumpToStep" />
        </section>
      </div>

      <!-- Samples comparison -->
      <section class="card sec">
        <div class="sec-head">
          <span class="sec-title">Samples</span>
          <span v-if="mode !== 'grid' && sampleLikeDelta" class="delta-pill mono" :style="{ borderColor: sampleLikeDelta.color, color: sampleLikeDelta.color }">
            <i class="mdi mdi-face-recognition" /> {{ sampleLikeDelta.text }}
          </span>
          <span class="spacer" />
          <div class="seg">
            <button :class="{ on: mode === 'side' }" @click="mode = 'side'"><i class="mdi mdi-view-split-vertical" /> Side by side</button>
            <button :class="{ on: mode === 'swipe' }" @click="mode = 'swipe'"><i class="mdi mdi-compare-horizontal" /> Swipe</button>
            <button :class="{ on: mode === 'grid' }" @click="mode = 'grid'"><i class="mdi mdi-view-grid-outline" /> Grid</button>
          </div>
          <ZenButton icon="mdi mdi-image-move" :disabled="exporting || !steps.length" @click="exportGrid">{{ exporting ? 'Rendering…' : 'Export grid' }}</ZenButton>
        </div>

        <p v-if="!prompts.length" class="muted small">Neither run has samples yet.</p>
        <template v-else>
          <div class="prompts">
            <button v-for="(p, i) in prompts" :key="p" class="prompt" :class="{ on: i === pi }" :title="p" @click="pi = i">
              <span class="p-idx mono">#{{ i + 1 }}</span>
              <span class="p-txt">{{ p }}</span>
              <i
                v-if="isPromptSkipped(a, p) || isPromptSkipped(b, p)"
                class="mdi mdi-eye-off-outline p-skip"
                title="Likeness skipped for this prompt"
              />
            </button>
          </div>

          <p v-if="prompt" class="prompt-full small muted">{{ prompt }}</p>

          <div v-if="mode !== 'grid'" class="stepper">
            <ZenButton size="sm" icon="mdi mdi-chevron-left" title="Previous step (←)" :disabled="si <= 0" @click="si = Math.max(0, si - 1)" />
            <ZenSlider v-model="si" :min="0" :max="Math.max(0, steps.length - 1)" :step="1" />
            <ZenButton size="sm" icon="mdi mdi-chevron-right" title="Next step (→)" :disabled="si >= steps.length - 1" @click="si = Math.min(steps.length - 1, si + 1)" />
            <span class="step-readout mono">step {{ step }}</span>
          </div>

          <div v-if="mode !== 'grid' && steps.length > 1 && steps.length <= 16" class="step-chips">
            <button v-for="(st, idx) in steps" :key="st" class="step-chip mono" :class="{ on: idx === si }" @click="si = idx">
              {{ st }}
            </button>
          </div>

          <div v-if="mode === 'side'" class="side">
            <figure
              v-for="([cell, like, ceil, skipped, name, color, idx]) in ([
                [cellA, likeA, likeCeilA, isPromptSkipped(a, prompt), nameA, colorA, 0],
                [cellB, likeB, likeCeilB, isPromptSkipped(b, prompt), nameB, colorB, 1],
              ] as const)"
              :key="name"
              class="shot"
            >
              <div class="shot-frame" :class="{ clickable: !!cell }" @click="cell && openSideLightbox(idx)">
                <img v-if="cell" :src="url(cell)" alt="" />
                <div v-else class="none muted small">No sample of this prompt</div>
                <SampleBadges v-if="cell" :likeness="like" :ceiling="ceil" :skipped="skipped" corner="bottom-left" />
                <span v-if="cell && cell.step !== step" class="step-warn mono">step {{ cell.step }}</span>
              </div>
              <figcaption class="small">
                <i class="sw" :style="{ background: color }" />
                <span class="shot-name">{{ name }}</span>
                <span v-if="cell" class="muted mono">· step {{ cell.step }}</span>
              </figcaption>
            </figure>
          </div>

          <div v-else-if="mode === 'swipe'" class="swipe">
            <ZenCompare
              v-if="cellA && cellB"
              :style="{
                aspectRatio: `${cellA.width || 1} / ${cellA.height || 1}`,
                width: `min(100%, calc(64vh / var(--zen-page-zoom) * ${cellA.width || 1} / ${cellA.height || 1}))`,
              }"
              :before="url(cellA)"
              :after="url(cellB)"
              :before-label="`${nameA} · ${cellA.step}${likeA?.score != null ? ` (${likeA.score.toFixed(2)})` : ''}`"
              :after-label="`${nameB} · ${cellB.step}${likeB?.score != null ? ` (${likeB.score.toFixed(2)})` : ''}`"
            />
            <p v-else class="muted small">Swiping needs a sample of this prompt from both runs.</p>
          </div>

          <div v-else class="gridwrap">
            <label class="small muted toggle">
              <ZenSwitch v-model="allPrompts" size="sm" />
              <span>Every prompt</span>
            </label>
            <div class="grid" :style="{ gridTemplateColumns: `150px repeat(${gridSteps.length}, minmax(0, 1fr))` }">
              <span />
              <span v-for="st in gridSteps" :key="st" class="colhead mono small muted">step {{ st }}</span>
              <template v-for="p in gridPrompts" :key="p">
                <template
                  v-for="[d, lmap, ceil, name, color] in ([
                    [a, likeMapA, likeCeilA, nameA, colorA],
                    [b, likeMapB, likeCeilB, nameB, colorB],
                  ] as const)"
                  :key="name"
                >
                  <span class="rowhead small" :title="name"><i class="sw" :style="{ background: color }" />{{ name }}</span>
                  <div
                    v-for="st in gridSteps"
                    :key="st"
                    class="gcell"
                    :class="{ clickable: !!cellFor(d, p, st) }"
                    @click="openGridLightbox(name, cellFor(d, p, st), p)"
                  >
                    <template v-if="cellFor(d, p, st)">
                      <img :src="url(cellFor(d, p, st))" loading="lazy" alt="" :title="`${name} · step ${cellFor(d, p, st)!.step}`" />
                      <SampleBadges
                        :likeness="lmap.get(`${cellFor(d, p, st)!.step}:${cellFor(d, p, st)!.index}`) ?? null"
                        :ceiling="ceil"
                        :skipped="isPromptSkipped(d, p)"
                        compact
                        corner="bottom-left"
                      />
                    </template>
                  </div>
                </template>
              </template>
            </div>
          </div>
        </template>
      </section>
    </template>
    <p v-else-if="ida || idb" class="muted small">Loading…</p>
    <div v-else class="card empty-pick muted">
      <i class="mdi mdi-compare-horizontal" />
      <span>Pick two runs above to compare settings, curves, and samples side by side.</span>
    </div>

    <ZenLightbox v-if="lb" v-model:index="lb.index" :items="lb.items" @close="lb = null" />
  </div>
</template>

<style scoped>
.kbd-hint {
  opacity: 0.75;
}
.pickers {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: end;
  gap: 12px;
  padding: 14px;
}
.pick {
  display: flex;
  align-items: center;
  gap: 8px;
}
.pick :deep(.zen-combo) {
  flex: 1;
  min-width: 0;
}
.sw {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 3px;
  flex-shrink: 0;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}
.run-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 14px 16px;
  border-top: 2px solid var(--run-color);
  min-width: 0;
}
.run-card-head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.run-dot {
  width: 9px;
  height: 9px;
  border-radius: 999px;
  flex-shrink: 0;
}
.run-link {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-weight: 600;
  color: var(--zen-text);
  text-decoration: none;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}
.run-link:hover {
  color: var(--zen-accent);
}
.run-link i {
  font-size: 13px;
  opacity: 0.65;
}
.run-sub {
  display: flex;
  align-items: center;
  gap: 6px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.run-sub span:first-child {
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 18px;
  padding-top: 6px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.kpi {
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.kpi-k {
  font-size: 10.5px;
  color: var(--zen-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.kpi-v {
  font-size: 13px;
  font-weight: 600;
  color: var(--zen-text);
}
.two {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.35fr);
  gap: 14px;
  align-items: start;
}
.sec {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px 16px;
  min-width: 0;
}
.sec-head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.sec-title {
  font-weight: 600;
}
.count-pill {
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 999px;
  background: var(--zen-surface-2);
  color: var(--zen-muted);
}
.count-pill.accent {
  color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 12%, transparent);
}
.delta-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11.5px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid currentColor;
  background: var(--zen-surface-2);
}
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  cursor: pointer;
}
.setup-wrap {
  max-height: 315px;
  overflow-y: auto;
}
.setup {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  table-layout: fixed;
}
.setup th {
  position: sticky;
  top: 0;
  background: var(--zen-surface);
  text-align: left;
  font-weight: 600;
  padding: 4px 6px 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  z-index: 1;
}
.setup th:first-child {
  width: 34%;
  color: var(--zen-muted);
}
.setup td {
  padding: 5px 6px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
  overflow-wrap: anywhere;
  vertical-align: top;
}
.setup tr.diff td:not(:first-child) {
  color: var(--zen-text);
  background: color-mix(in srgb, var(--zen-accent) 8%, transparent);
}
.empty-row {
  padding: 14px 6px !important;
  text-align: center;
}
.seg {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.seg button {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 5px 10px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.seg button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.seg button:disabled {
  opacity: 0.4;
  cursor: default;
}
.prompts {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.prompt {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  max-width: 320px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  cursor: pointer;
}
.p-idx {
  opacity: 0.65;
  font-size: 11px;
  flex-shrink: 0;
}
.p-txt {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.p-skip {
  font-size: 12px;
  opacity: 0.7;
  flex-shrink: 0;
}
.prompt.on {
  color: var(--zen-selected-text, var(--zen-text));
  border-color: var(--zen-selected, var(--zen-accent));
  background: var(--zen-selected-bg, var(--zen-surface-2));
}
.prompt-full {
  margin: -4px 0 0;
  padding: 6px 10px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  line-height: 1.4;
}
.stepper {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 8px;
}
.step-readout {
  font-size: 12px;
  font-weight: 600;
  min-width: 72px;
  text-align: right;
}
.step-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: -4px;
}
.step-chip {
  border: 1px solid color-mix(in srgb, var(--zen-border) 60%, transparent);
  background: none;
  color: var(--zen-muted);
  font-size: 11px;
  padding: 2px 7px;
  border-radius: var(--r-sm);
  cursor: pointer;
}
.step-chip.on {
  color: var(--zen-selected-text, var(--zen-text));
  border-color: var(--zen-selected, var(--zen-accent));
  background: var(--zen-selected-bg, var(--zen-surface-2));
}
.side {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
.shot {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.shot-frame {
  position: relative;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.shot-frame.clickable,
.gcell.clickable {
  cursor: zoom-in;
}
.shot-frame img {
  width: 100%;
  max-height: calc(58vh / var(--zen-page-zoom));
  object-fit: contain;
  display: block;
}
.step-warn {
  position: absolute;
  top: 6px;
  right: 6px;
  font-size: 11px;
  padding: 2px 6px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.65);
  color: var(--zen-warn, #fbbf24);
}
.shot .none {
  aspect-ratio: 1;
  display: grid;
  place-items: center;
}
.shot figcaption,
.rowhead {
  display: flex;
  align-items: center;
  gap: 6px;
}
.shot-name {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.swipe {
  display: flex;
  justify-content: center;
}
.swipe :deep(.zen-compare) {
  height: auto;
  border-radius: var(--r-md);
}
.gridwrap {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.grid {
  display: grid;
  gap: 8px;
  align-items: center;
}
.colhead {
  text-align: center;
}
.rowhead {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.gcell {
  position: relative;
  aspect-ratio: 1;
  border-radius: var(--r-sm);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.gcell img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.empty-pick {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 36px 20px;
}
.empty-pick i {
  font-size: 22px;
  opacity: 0.6;
}
.mono {
  font-family: var(--zen-mono, monospace);
}
.err {
  color: var(--zen-danger);
}
@media (max-width: 1000px) {
  .two,
  .summary-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 720px) {
  .pickers,
  .side {
    grid-template-columns: 1fr;
  }
  .grid {
    grid-template-columns: 90px repeat(3, minmax(0, 1fr)) !important;
  }
}
</style>
