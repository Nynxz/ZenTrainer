<script setup lang="ts">
// Two runs side by side: what differs in their setup, their curves on one chart, and their
// samples prompt by prompt, step by step (side by side, a swipe between them, or a grid), with a
// themed PNG of the grid to share. A and B keep their colours everywhere: accent and info.
// Keys: ←/→ step, ↑/↓ prompt, S swaps the runs.
import { ZenButton, ZenCombo, ZenCompare, ZenField, ZenSlider } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import CompareChart from '@/components/CompareChart.vue'
import { EngineError, engine, viewUrl, type RunDetail, type RunMeta, type SampleRow } from '@/api/engine'

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
  runs.value.map((r) => ({ value: r.run_id, label: r.name || r.run_id, keywords: `${r.name} ${r.run_id}`, hint: `${fmtDate(r.created)} · ${r.status}` })),
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

// ---- setup: what differs --------------------------------------------------------------------
const FIELDS: [string, (s: RunDetail['spec']) => unknown][] = [
  ['Model', (s) => s.model.unet_name ?? s.model.ckpt_name],
  ['Dataset', (s) => s.dataset.folder],
  ['Caption set', (s) => s.dataset.caption_set],
  ['Trigger', (s) => s.dataset.trigger],
  ['Resolution', (s) => s.dataset.resolution],
  ['Masked training', (s) => s.dataset.masked_loss],
  ['Steps', (s) => s.optim.steps],
  ['Batch × accumulation', (s) => `${s.optim.batch_size} × ${s.optim.grad_accumulation_steps}`],
  ['Optimizer', (s) => s.optim.optimizer],
  ['Learning rate', (s) => s.optim.learning_rate],
  ['Schedule', (s) => `${s.optim.lr_scheduler}, warmup ${s.optim.warmup_steps}`],
  ['Weight decay', (s) => s.optim.weight_decay],
  ['Network', (s) => `${s.network.algorithm} r${s.network.rank} α${s.network.alpha ?? s.network.rank}`],
  ['Layers', (s) => (Array.isArray(s.network.targets) ? s.network.targets.join(', ') : s.network.targets)],
  ['Blocks', (s) => s.network.blocks || 'all'],
  ['Base LoRA', (s) => s.network.base_lora ? `${s.network.base_lora} @ ${s.network.base_strength ?? 1}` : null],
  ['Training adapter', (s) => s.network.training_adapter],
  ['Timesteps', (s) => s.objective.timesteps],
  ['Precision', (s) => s.memory.training_dtype],
  ['Seed', (s) => s.seed],
]
const onlyDiff = ref(true)
const show = (v: unknown) => (v == null || v === '' ? '–' : typeof v === 'boolean' ? (v ? 'on' : 'off') : String(v))
const setup = computed(() => {
  if (!a.value || !b.value) return []
  return FIELDS.map(([k, f]) => {
    const va = show(f(a.value!.spec))
    const vb = show(f(b.value!.spec))
    return { k, va, vb, diff: va !== vb }
  }).filter((r) => !onlyDiff.value || r.diff)
})

// ---- curves ---------------------------------------------------------------------------------
const metric = ref<'loss' | 'val' | 'like'>('loss')
const likeSteps = (d: RunDetail) => {
  const by = new Map<number, number[]>()
  for (const r of d.likeness ?? []) if (!r.manual && r.score != null) by.set(r.step, [...(by.get(r.step) ?? []), r.score])
  return [...by].sort((p, q) => p[0] - q[0]).map(([step, xs]) => ({ step, score: xs.reduce((s, v) => s + v, 0) / xs.length }))
}
const series = computed(() =>
  [a.value && { label: nameA.value, color: colorA.value, rows: a.value.metrics, val: a.value.validation, like: likeSteps(a.value) },
   b.value && { label: nameB.value, color: colorB.value, rows: b.value.metrics, val: b.value.validation, like: likeSteps(b.value) }].filter(Boolean) as InstanceType<typeof CompareChart>['$props']['series'],
)
const hasLike = computed(() => !!(a.value?.likeness?.length || b.value?.likeness?.length))
const hasVal = computed(() => !!(a.value?.validation.length || b.value?.validation.length))

// ---- samples: prompts matched by text, steps from both --------------------------------------
const runSamples = (d: RunDetail | null) => (d ? d.samples.filter((s) => !s.manual && !s.base) : [])
const prompts = computed(() => {
  const out: string[] = []
  for (const s of [...runSamples(a.value), ...runSamples(b.value)]) if (!out.includes(s.prompt)) out.push(s.prompt)
  return out
})
const pi = ref(0)
const prompt = computed(() => prompts.value[pi.value] ?? '')
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
// each run's sample nearest the step: runs that sample on different schedules still line up
const nearest = (m: Map<number, SampleRow>, at: number) => {
  let best: SampleRow | null = null
  for (const [st, row] of m) if (!best || Math.abs(st - at) < Math.abs(best.step - at)) best = row
  return best
}
const cellA = computed(() => nearest(stepsA.value, step.value))
const cellB = computed(() => nearest(stepsB.value, step.value))
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

function onKey(e: KeyboardEvent) {
  if ((e.target as HTMLElement).closest('input, textarea')) return
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
    const T = 300 // tile
    const G = 12 // gap
    const L = 220 // row labels
    const HEAD = 150
    const COLHEAD = 36
    const PGAP = 26 // between prompts
    const M = 48
    const W = M * 2 + L + cols.length * (T + G) - G
    const H = M + HEAD + COLHEAD + rows.length * (T + G) + (gridPrompts.value.length - 1) * PGAP + 40 + M
    const cv = document.createElement('canvas')
    cv.width = W
    cv.height = H
    const c = cv.getContext('2d')!
    // background: the banner's gradient
    const g = c.createLinearGradient(0, 0, W, H)
    g.addColorStop(0, bg)
    g.addColorStop(0.5, surface)
    g.addColorStop(1, bg)
    c.fillStyle = g
    c.fillRect(0, 0, W, H)
    // header: lotus, wordmark, the two runs
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
    // column heads
    const top = M + HEAD
    c.font = `600 15px ${font}`
    c.fillStyle = muted
    cols.forEach((st, i) => c.fillText(`STEP ${st}`, M + L + i * (T + G), top + 20))
    // rows
    let yy = top + COLHEAD
    for (let r = 0; r < rows.length; r++) {
      const row = rows[r]!
      if (r && r % 2 === 0) yy += PGAP
      // label: colour chip + run name (+ prompt on the first of a pair when there are several)
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
  if (d < 0.08) return null // grey: no hue to tell apart by
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
        <p class="lede">Two runs side by side: setup, curves and samples. ←/→ steps, ↑/↓ prompts, S swaps.</p>
      </div>
    </div>

    <section class="card pickers">
      <ZenField label="Run A" stack>
        <div class="pick"><i class="sw" :style="{ background: colorA }" /><ZenCombo v-model="ida" :items="runItems" placeholder="Pick a run…" :sorted="false" :item-height="30" /></div>
      </ZenField>
      <ZenButton icon="mdi mdi-swap-horizontal" title="Swap (S)" :disabled="!ida && !idb" @click="swap" />
      <ZenField label="Run B" stack>
        <div class="pick"><i class="sw" :style="{ background: colorB }" /><ZenCombo v-model="idb" :items="runItems" placeholder="Pick a run…" :sorted="false" :item-height="30" /></div>
      </ZenField>
    </section>
    <div v-if="error" class="err small">{{ error }}</div>

    <template v-if="a && b">
      <div class="two">
        <section class="card sec">
          <div class="sec-head">
            <span class="sec-title">Setup</span>
            <span class="spacer" />
            <label class="small muted toggle"><input v-model="onlyDiff" type="checkbox" /> Only what differs</label>
          </div>
          <table class="setup">
            <thead>
              <tr><th /><th :style="{ color: colorA }">{{ nameA }}</th><th :style="{ color: colorB }">{{ nameB }}</th></tr>
            </thead>
            <tbody>
              <tr v-for="r in setup" :key="r.k" :class="{ diff: r.diff }">
                <td class="muted">{{ r.k }}</td>
                <td class="mono">{{ r.va }}</td>
                <td class="mono">{{ r.vb }}</td>
              </tr>
              <tr v-if="!setup.length"><td colspan="3" class="muted">The same setup{{ onlyDiff ? ' (in what this compares)' : '' }}.</td></tr>
            </tbody>
          </table>
        </section>

        <section class="card sec">
          <div class="sec-head">
            <span class="sec-title">Curves</span>
            <span class="spacer" />
            <div class="seg">
              <button :class="{ on: metric === 'loss' }" @click="metric = 'loss'">Loss</button>
              <button :class="{ on: metric === 'val' }" :disabled="!hasVal" @click="metric = 'val'">Validation</button>
              <button :class="{ on: metric === 'like' }" :disabled="!hasLike" @click="metric = 'like'">Likeness</button>
            </div>
          </div>
          <CompareChart :series="series" :metric="metric" :height="230" />
        </section>
      </div>

      <section class="card sec">
        <div class="sec-head">
          <span class="sec-title">Samples</span>
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
            <button v-for="(p, i) in prompts" :key="p" class="prompt" :class="{ on: i === pi }" :title="p" @click="pi = i">{{ p }}</button>
          </div>

          <div v-if="mode !== 'grid'" class="stepper">
            <span class="small muted">Step</span>
            <ZenSlider v-model="si" :min="0" :max="Math.max(0, steps.length - 1)" :step="1" />
            <b class="mono">{{ step }}</b>
          </div>

          <div v-if="mode === 'side'" class="side">
            <figure v-for="[cell, name, color] in ([[cellA, nameA, colorA], [cellB, nameB, colorB]] as const)" :key="name" class="shot">
              <img v-if="cell" :src="url(cell)" alt="" />
              <div v-else class="none muted small">No sample of this prompt</div>
              <figcaption class="small"><i class="sw" :style="{ background: color }" />{{ name }}<span v-if="cell" class="muted mono"> · step {{ cell.step }}</span></figcaption>
            </figure>
          </div>

          <div v-else-if="mode === 'swipe'" class="swipe">
            <ZenCompare v-if="cellA && cellB" :style="{ aspectRatio: `${cellA.width || 1} / ${cellA.height || 1}`, width: `min(100%, calc(64vh * ${cellA.width || 1} / ${cellA.height || 1}))` }" :before="url(cellA)" :after="url(cellB)" :before-label="`${nameA} · ${cellA.step}`" :after-label="`${nameB} · ${cellB.step}`" />
            <p v-else class="muted small">Swiping needs a sample of this prompt from both runs.</p>
          </div>

          <div v-else class="gridwrap">
            <label class="small muted toggle"><input v-model="allPrompts" type="checkbox" /> Every prompt</label>
            <div class="grid" :style="{ gridTemplateColumns: `140px repeat(${gridSteps.length}, minmax(0, 1fr))` }">
              <span />
              <span v-for="st in gridSteps" :key="st" class="colhead mono small muted">step {{ st }}</span>
              <template v-for="p in gridPrompts" :key="p">
                <template v-for="[d, name, color] in ([[a, nameA, colorA], [b, nameB, colorB]] as const)" :key="name">
                  <span class="rowhead small"><i class="sw" :style="{ background: color }" />{{ name }}</span>
                  <div v-for="st in gridSteps" :key="st" class="gcell">
                    <img v-if="cellFor(d, p, st)" :src="url(cellFor(d, p, st))" loading="lazy" alt="" :title="`${name} · step ${cellFor(d, p, st)!.step}`" />
                  </div>
                </template>
              </template>
            </div>
          </div>
        </template>
      </section>
    </template>
    <p v-else-if="ida || idb" class="muted small">Loading…</p>
    <p v-else class="muted">Pick two runs to compare.</p>
  </div>
</template>

<style scoped>
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
  margin-right: 6px;
}
.two {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr);
  gap: 14px;
}
.sec {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
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
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}
.setup {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  table-layout: fixed;
}
.setup th {
  text-align: left;
  font-weight: 600;
  padding: 4px 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.setup th:first-child {
  width: 34%;
}
.setup td {
  padding: 5px 6px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
  overflow-wrap: anywhere;
  vertical-align: top;
}
.setup tr.diff td:not(:first-child) {
  color: var(--zen-text);
  background: color-mix(in srgb, var(--zen-accent) 7%, transparent);
}
.seg {
  display: flex;
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
  max-width: 360px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  cursor: pointer;
}
.prompt.on {
  color: var(--zen-selected-text, var(--zen-text));
  border-color: var(--zen-selected, var(--zen-accent));
  background: var(--zen-selected-bg, var(--zen-surface-2));
}
.stepper {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) 56px;
  align-items: center;
  gap: 12px;
}
.side {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.shot {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.shot img {
  width: 100%;
  max-height: 56vh;
  object-fit: contain;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
}
.shot .none {
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
}
.shot figcaption,
.rowhead {
  display: flex;
  align-items: center;
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
  gap: 8px;
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
.mono {
  font-family: var(--zen-mono, monospace);
}
.err {
  color: var(--zen-danger);
}
@media (max-width: 1000px) {
  .two {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 720px) {
  .pickers {
    grid-template-columns: 1fr;
  }
  .side {
    grid-template-columns: 1fr;
  }
  .grid {
    grid-template-columns: 90px repeat(3, minmax(0, 1fr)) !important;
  }
}
</style>
