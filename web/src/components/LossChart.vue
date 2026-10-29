<script setup lang="ts">
// Loss over optimizer steps: training loss (EMA-smoothed in the accent; each step's raw loss is
// in the tooltip) and, when the run has it, the fixed-set validation loss as a second
// series with markers. Two series, one unit, one axis: a legend names them.
// Crosshair + tooltip on hover. The learning rate, when it changes (warmup, decay, Automagic
// finding its own), rides on a right-hand axis as ai-toolkit draws it: a thin solid line in
// the muted ink, with its own legend key and axis labels, so it never reads as a loss.
// Training speed (steps over their time across a 51-step window: optimizer steps only, so samples
// and checkpoints don't dip it, and smooth while a resolution change mixes two step sizes) is in the tooltip, and a line of its own when switched on; resolution stages are marked on
// the x axis, where the speed changes. Each legend key shows or hides its line; the x axis is
// steps or the run's time. With class preservation (objective.preserve), its loss (smoothed) is a
// line on its own scale from 0, like speed: it measures something else, far smaller.
import { maxOf, minOf } from '@/lib/extent'
import { SHARE_TIP } from '@/lib/likeness'
import { resumed } from '@/lib/resumed'
import { computed, ref, watch } from 'vue'
import type { MetricRow, ValidationRow } from '@/api/engine'

const props = withDefaults(
  defineProps<{
    rows: MetricRow[]
    total?: number
    smoothing?: number
    marks?: { step: number; kind: 'ckpt' | 'sample'; done: boolean }[]
    val?: ValidationRow[]
    /** with faces on: each sampled step's mean likeness (its own 0-1 scale, no axis; values on hover) */
    like?: { step: number; score: number; share?: string }[]
    /** resolution stages: from `start` the run blends into `label`'s resolution, all of it by `end` */
    stages?: { start: number; end: number; label: string }[]
    height?: number
  }>(),
  { smoothing: 0.95, marks: () => [], val: () => [], like: () => [], stages: () => [], height: 240 },
)

// which lines are hidden and the x axis, remembered per browser (only a convenience)
type Line = 'ema' | 'val' | 'like' | 'lr' | 'ckpt' | 'speed' | 'pres'
const PREFS = 'zentrainer.lossChart'
function loadPrefs(): { hidden: Line[]; mode: 'step' | 'time' } {
  try {
    const p = JSON.parse(localStorage.getItem(PREFS) ?? '')
    return { hidden: Array.isArray(p.hidden) ? p.hidden : ['speed'], mode: p.mode === 'time' ? 'time' : 'step' }
  } catch {
    return { hidden: ['speed'], mode: 'step' }
  }
}
const prefs = loadPrefs()
const hidden = ref(new Set<Line>(prefs.hidden))
const mode = ref<'step' | 'time'>(prefs.mode)
watch([hidden, mode], () => {
  try {
    localStorage.setItem(PREFS, JSON.stringify({ hidden: [...hidden.value], mode: mode.value }))
  } catch {
    /* private window or blocked storage: the choice just isn't remembered */
  }
}, { deep: true })
const shows = (l: Line) => !hidden.value.has(l)
function toggle(l: Line) {
  const h = new Set(hidden.value)
  h.has(l) ? h.delete(l) : h.add(l)
  hidden.value = h
}

const W = 720
const H = computed(() => props.height)
const pad = computed(() => ({ l: 44, r: lrShown.value && shows('lr') ? 46 : 12, t: 10, b: 24 }))

// one continuous history across resumes (lib/resumed.ts), then the steps with a loss
const points = computed(() => resumed(props.rows).filter((r) => r.loss != null) as (MetricRow & { loss: number })[])

const ema = computed(() => {
  const out: number[] = []
  let s: number | null = null
  for (const p of points.value) {
    s = s == null ? p.loss : props.smoothing * s + (1 - props.smoothing) * p.loss
    out.push(s)
  }
  return out
})

// time axis: a step's wall-clock seconds into the run (interpolated between logged steps; steps
// not trained yet sit at the end)
const timeAt = (step: number) => {
  const ps = points.value
  if (!ps.length) return 0
  if (step <= ps[0]!.step) return ps[0]!.elapsed
  let lo = 0
  let hi = ps.length - 1
  if (step >= ps[hi]!.step) return ps[hi]!.elapsed
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    ps[mid]!.step <= step ? (lo = mid) : (hi = mid)
  }
  const a = ps[lo]!
  const b = ps[hi]!
  return a.elapsed + ((step - a.step) / (b.step - a.step || 1)) * (b.elapsed - a.elapsed)
}
const domain = (step: number) => (mode.value === 'time' ? timeAt(step) : step)
const xMax = computed(() =>
  mode.value === 'time' ? Math.max(points.value.at(-1)?.elapsed ?? 1, 1) : Math.max(props.total ?? 0, points.value.at(-1)?.step ?? 1, 1),
)
// fitted to the shown loss lines: the raw per-step loss spikes (its value is in the hover
// tooltip) would otherwise squash them against the bottom
const yRange = computed(() => {
  let v = [...(shows('ema') ? ema.value : []), ...(shows('val') ? props.val.map((r) => r.val_loss) : [])]
  if (!v.length) v = [...ema.value, ...props.val.map((r) => r.val_loss)]
  if (!v.length) return [0, 1] as const
  const lo = minOf(v)
  const hi = maxOf(v)
  const padY = (hi - lo || hi || 1) * 0.08
  return [Math.max(0, lo - padY), hi + padY] as const
})

const xd = (d: number) => pad.value.l + (d / xMax.value) * (W - pad.value.l - pad.value.r)
const x = (step: number) => xd(domain(step))
const y = (v: number) => {
  const [lo, hi] = yRange.value
  return pad.value.t + (1 - (v - lo) / (hi - lo || 1)) * (H.value - pad.value.t - pad.value.b)
}

const path = (vals: number[]) =>
  points.value.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(vals[i]!).toFixed(1)}`).join('')

const valPath = computed(() =>
  props.val.map((r, i) => `${i ? 'L' : 'M'}${x(r.step).toFixed(1)},${y(r.val_loss).toFixed(1)}`).join(''),
)
const valAt = (step: number) => [...props.val].reverse().find((r) => r.step <= step)
const emaPath = computed(() => path(ema.value))

const yTicks = computed(() => {
  const [lo, hi] = yRange.value
  return [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo))
})
const xTicks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * xMax.value)))
const clock = (s: number) => {
  const m = Math.floor(s / 60)
  return m >= 60 ? `${Math.floor(m / 60)}h${String(m % 60).padStart(2, '0')}` : `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}
const fmtX = (d: number) => (mode.value === 'time' ? clock(d) : String(d))

const hover = ref<number | null>(null)
function onMove(ev: PointerEvent) {
  const svg = ev.currentTarget as SVGSVGElement
  const box = svg.getBoundingClientRect()
  const sx = ((ev.clientX - box.left) / box.width) * W
  const at = ((sx - pad.value.l) / (W - pad.value.l - pad.value.r)) * xMax.value
  let best: number | null = null
  let bestD = Infinity
  points.value.forEach((p, i) => {
    const d = Math.abs((mode.value === 'time' ? p.elapsed : p.step) - at)
    if (d < bestD) [best, bestD] = [i, d]
  })
  hover.value = best
}
// learning rate on the right axis: linear from 0, drawn only when the LR actually moves
const lrPoints = computed(() => points.value.filter((p) => p.lr != null) as (MetricRow & { loss: number; lr: number })[])
const lrShown = computed(() => {
  const v = lrPoints.value.map((p) => p.lr)
  return v.length > 1 && maxOf(v) > minOf(v) * 1.001
})
const lrMax = computed(() => maxOf(lrPoints.value.map((p) => p.lr), 1e-12))
const ly = (v: number) => pad.value.t + (1 - v / (lrMax.value * 1.05)) * (H.value - pad.value.t - pad.value.b)
const lrTicks = computed(() => [0, 0.5, 1].map((f) => f * lrMax.value))
const lrPath = computed(() => lrPoints.value.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${ly(p.lr).toFixed(1)}`).join(''))
const fmtLr = (v?: number) => (v == null ? '' : v === 0 ? '0' : v.toExponential(1))

// likeness: 0 at the bottom of the plot, 1 at the top; 0.5 (a face starts to read as the same
// person) faintly marked
const likeY = (v: number) => pad.value.t + (1 - Math.max(0, Math.min(1, v))) * (H.value - pad.value.t - pad.value.b)
const likePath = computed(() => props.like.map((r, i) => `${i ? 'L' : 'M'}${x(r.step).toFixed(1)},${likeY(r.score).toFixed(1)}`).join(''))
const likeAt = (step: number) => [...props.like].reverse().find((r) => r.step <= step) ?? null

// speed: steps over their seconds across 51 steps (the first, compiling, step left out, and any
// step over 5x the window's median: a recompile isn't the pace). An average, not a median: while
// a run blends into a new resolution its steps come at both sizes, and a median jumped between
// the two. Each step weighs less the nearer the window's ends (a triangle), so steps that differ
// in turn (class preservation every Nth step) don't make it saw as they enter and leave. Near
// either end the window keeps its whole width and slides in instead of shrinking (a one-sided one
// tips with each new step): the line reaches the newest step, level over its last SPEED_HALF
// steps at the newest window's pace. As a line it has its own scale from 0 (no axis; the value is
// on hover), like likeness
const SPEED_HALF = 25
const speed = computed(() => {
  const st = points.value.map((p) => (p.step > 1 && p.step_time ? p.step_time : null))
  const width = Math.min(2 * SPEED_HALF + 1, st.length)
  return st.map((_, i) => {
    const lo = Math.max(0, Math.min(i - SPEED_HALF, st.length - width))
    const mid = lo + (width - 1) / 2
    const w: [number, number][] = []
    for (let j = lo; j < lo + width; j++) {
      const v = st[j]
      if (v != null) w.push([v, (width + 1) / 2 - Math.abs(j - mid)])
    }
    if (!w.length) return null
    const cap = 5 * w.map(([v]) => v).sort((a, b) => a - b)[Math.floor(w.length / 2)]!
    const kept = w.filter(([v]) => v <= cap)
    return kept.reduce((a, [, k]) => a + k, 0) / kept.reduce((a, [v, k]) => a + v * k, 0)
  })
})
const hasSpeed = computed(() => speed.value.some((v) => v != null))

// class preservation's loss, smoothed per step as the training loss is (a value only every
// preserve_every-th step decays by the steps between, so it's as smooth whatever N is), on its own
// scale from 0 with room above (no axis): a steady line reads as steady, not stretched to fill it
const pres = computed(() => {
  const out: { step: number; v: number }[] = []
  let s: number | null = null
  let at = 0
  for (const r of resumed(props.rows)) {
    if (r.preserve_loss == null) continue
    const k = Math.pow(props.smoothing, Math.max(1, r.step - at))
    s = s == null ? r.preserve_loss : k * s + (1 - k) * r.preserve_loss
    at = r.step
    out.push({ step: r.step, v: s })
  }
  return out
})
const pMax = computed(() => maxOf(pres.value.map((p) => p.v), 1e-9) * 2)
const presY = (v: number) => pad.value.t + (1 - v / pMax.value) * (H.value - pad.value.t - pad.value.b)
const presPath = computed(() => pres.value.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${presY(p.v).toFixed(1)}`).join(''))
const presAt = (step: number) => [...pres.value].reverse().find((p) => p.step <= step) ?? null
const sMax = computed(() => maxOf(speed.value.filter((v): v is number => v != null), 1e-6) * 1.1)
const sy = (v: number) => pad.value.t + (1 - v / sMax.value) * (H.value - pad.value.t - pad.value.b)
const speedPath = computed(() => {
  let d = ''
  let pen = false
  points.value.forEach((p, i) => {
    const v = speed.value[i]
    if (v == null) {
      pen = false
      return
    }
    d += `${pen ? 'L' : 'M'}${x(p.step).toFixed(1)},${sy(v).toFixed(1)}`
    pen = true
  })
  return d
})
const fmtSpeed = (v: number) => (v >= 1 ? `${v.toFixed(2)} it/s` : `${(1 / v).toFixed(2)} s/it`)
// stage labels only where the run has got to on a time axis (no time for steps not trained yet)
const stageMarks = computed(() =>
  props.stages.filter((st) => mode.value === 'step' || st.start <= (points.value.at(-1)?.step ?? 0)),
)

const hovered = computed(() => (hover.value == null ? null : { p: points.value[hover.value]!, s: ema.value[hover.value]!, v: speed.value[hover.value] }))
const fmt = (v: number) => (v === 0 || Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toExponential(2))
</script>

<template>
  <figure class="loss">
    <figcaption>
      <span class="legend">
        <button class="key" :class="{ off: !shows('ema') }" title="Show or hide" @click="toggle('ema')"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-ema" /></svg> Training (smoothed)</button>
        <button v-if="val.length" class="key" :class="{ off: !shows('val') }" title="Show or hide" @click="toggle('val')"><svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" class="k-val" /><circle cx="9" cy="4" r="3" class="k-val-dot" /></svg> {{ val[val.length - 1]?.held_out ? 'Validation (held-out set)' : 'Validation (fixed set)' }}</button>
        <button v-if="like.length" class="key" :class="{ off: !shows('like') }" title="Show or hide" @click="toggle('like')"><svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" class="k-like" /><circle cx="9" cy="4" r="3" class="k-like-dot" /></svg> Likeness</button>
        <button v-if="lrShown" class="key muted" :class="{ off: !shows('lr') }" title="Show or hide" @click="toggle('lr')"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-lr" /></svg> Learning rate (right axis)</button>
        <button v-if="pres.length" class="key muted" :class="{ off: !shows('pres') }" title="Show or hide: class preservation's loss (how far the LoRA moves the class words from the base model), smoothed, on its own scale from 0 (values on hover). Flat is the aim" @click="toggle('pres')"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-pres" /></svg> Preservation</button>
        <button v-if="hasSpeed" class="key muted" :class="{ off: !shows('speed') }" title="Show or hide: training speed, its own scale from 0 (values on hover)" @click="toggle('speed')"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-speed" /></svg> Speed</button>
        <button v-if="marks.some((m) => m.kind === 'ckpt')" class="key muted" :class="{ off: !shows('ckpt') }" title="Show or hide" @click="toggle('ckpt')"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" class="k-mark" /></svg> checkpoints</button>
      </span>
      <span v-if="points.length" class="xmode" role="group" aria-label="x axis">
        <button :class="{ on: mode === 'step' }" @click="mode = 'step'">Steps</button>
        <button :class="{ on: mode === 'time' }" title="The run's wall-clock time, loading and samples included" @click="mode = 'time'">Time</button>
      </span>
    </figcaption>
    <div v-if="!points.length" class="empty muted" :style="{ height: `${height}px` }">Waiting for the first step…</div>
    <div v-else class="plot" :style="{ height: `${height}px` }">
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" @pointermove="onMove" @pointerleave="hover = null">
        <g class="grid">
          <line v-for="t in yTicks" :key="t" :x1="pad.l" :x2="W - pad.r" :y1="y(t)" :y2="y(t)" />
        </g>
        <template v-if="shows('ckpt')">
          <line
            v-for="m in marks.filter((m) => m.kind === 'ckpt')"
            :key="m.step"
            class="mark"
            :class="{ done: m.done }"
            :x1="x(m.step)"
            :x2="x(m.step)"
            :y1="pad.t"
            :y2="H - pad.b"
          />
        </template>
        <path v-if="hasSpeed && shows('speed')" class="speed-line" :d="speedPath" />
        <path v-if="pres.length && shows('pres')" class="pres-line" :d="presPath" />
        <path v-if="shows('ema')" class="ema" :d="emaPath" />
        <path v-if="lrShown && shows('lr')" class="lr-line" :d="lrPath" />
        <template v-if="shows('val')">
          <path v-if="val.length > 1" class="val" :d="valPath" />
          <circle v-for="r in val" :key="r.step" class="val-dot" :cx="x(r.step)" :cy="y(r.val_loss)" r="3.5" />
        </template>
        <template v-if="like.length && shows('like')">
          <line class="like-ref" :x1="pad.l" :x2="W - pad.r" :y1="likeY(0.5)" :y2="likeY(0.5)" />
          <path v-if="like.length > 1" class="like" :d="likePath" />
          <circle v-for="r in like" :key="`l${r.step}`" class="like-dot" :cx="x(r.step)" :cy="likeY(r.score)" r="3.5" />
        </template>
        <g v-if="hovered">
          <line class="cross" :x1="x(hovered.p.step)" :x2="x(hovered.p.step)" :y1="pad.t" :y2="H - pad.b" />
          <circle v-if="shows('ema')" class="dot" :cx="x(hovered.p.step)" :cy="y(hovered.s)" r="4" />
        </g>
      </svg>
      <div class="y-axis">
        <span v-for="t in yTicks" :key="t" :style="{ top: `${(y(t) / H) * 100}%` }">{{ fmt(t) }}</span>
      </div>
      <div v-if="lrShown && shows('lr')" class="y-axis right">
        <span class="lr-head">LR</span>
        <span v-for="t in lrTicks" :key="t" :style="{ top: `${(ly(t) / H) * 100}%` }">{{ fmtLr(t) }}</span>
      </div>
      <div
        v-if="hovered"
        class="tip"
        :style="{ left: `${(x(hovered.p.step) / W) * 100}%`, transform: `translateX(${x(hovered.p.step) > W / 2 ? '-105%' : '5%'})` }"
      >
        <div class="mono">step {{ hovered.p.step }} <span class="muted">· {{ clock(hovered.p.elapsed) }}</span></div>
        <div>loss <b class="mono">{{ fmt(hovered.p.loss) }}</b></div>
        <div>smoothed <b class="mono">{{ fmt(hovered.s) }}</b></div>
        <div v-if="valAt(hovered.p.step)">
          {{ valAt(hovered.p.step)!.held_out ? 'held-out' : 'validation' }} <b class="mono">{{ fmt(valAt(hovered.p.step)!.val_loss) }}</b>
          <span class="muted">@{{ valAt(hovered.p.step)!.step }}</span>
        </div>
        <div v-if="likeAt(hovered.p.step)">
          likeness <b class="mono">{{ likeAt(hovered.p.step)!.score.toFixed(2) }}</b>
          <span v-if="likeAt(hovered.p.step)!.share" class="muted" :title="SHARE_TIP"> · {{ likeAt(hovered.p.step)!.share }} of your photos</span>
          <span class="muted">@{{ likeAt(hovered.p.step)!.step }}</span>
        </div>
        <div v-if="presAt(hovered.p.step)">
          preservation <b class="mono">{{ fmt(presAt(hovered.p.step)!.v) }}</b>
          <span class="muted">smoothed</span>
        </div>
        <div v-if="hovered.p.lr != null" class="muted">lr <span class="mono">{{ fmtLr(hovered.p.lr) }}</span></div>
        <div v-if="hovered.v != null" class="muted">
          speed <span class="mono">{{ fmtSpeed(hovered.v) }}</span>
          <template v-if="hovered.p.step_time"> · this step <span class="mono">{{ hovered.p.step_time.toFixed(2) }}s</span></template>
        </div>
      </div>
    </div>
    <div v-if="points.length" class="x-axis">
      <span v-for="t in xTicks" :key="t" :style="{ left: `${(xd(t) / W) * 100}%` }">{{ fmtX(t) }}</span>
      <span
        v-for="st in stageMarks"
        :key="`s${st.start}`"
        class="stage"
        :style="{ left: `${(x(st.start) / W) * 100}%` }"
        :title="`resolution ${st.label.replace('→ ', '')} from step ${st.start} (fully by ${st.end})`"
        >{{ st.label }}</span
      >
    </div>
  </figure>
</template>

<style scoped>
.loss {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  align-items: center;
  font-weight: 600;
}
figcaption .muted {
  font-weight: 400;
  font-size: 12px;
}
.empty {
  display: grid;
  place-items: center;
}
.plot {
  position: relative;
}
.x-axis {
  position: relative;
  height: 14px;
  margin-top: -4px;
}
/* solid and grey: the checkpoint markers are the dotted ones, the losses the coloured ones */
.lr-line {
  fill: none;
  stroke: var(--zen-muted);
  stroke-width: 1.5;
  opacity: 0.85;
  vector-effect: non-scaling-stroke;
}
.k-lr {
  stroke: var(--zen-muted);
  stroke-width: 1.5;
}
.y-axis.right span {
  left: auto;
  right: 0;
}
.y-axis.right .lr-head {
  top: -2px;
  transform: none;
}
button.key {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  color: inherit;
  cursor: pointer;
}
button.key.off {
  opacity: 0.4;
  text-decoration: line-through;
}
.xmode {
  margin-left: auto;
  display: inline-flex;
  gap: 2px;
}
.xmode button {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  font-weight: 400;
  padding: 2px 8px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.xmode button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.pres-line,
.k-pres {
  fill: none;
  stroke: var(--chart-4, var(--zen-text));
  stroke-width: 1.25;
  stroke-dasharray: 6 3;
  opacity: 0.7;
  vector-effect: non-scaling-stroke;
}
.speed-line,
.k-speed {
  fill: none;
  stroke: var(--chart-3, var(--zen-warn));
  stroke-width: 1.25;
  stroke-dasharray: 1 3;
  stroke-linecap: round;
  opacity: 0.9;
  vector-effect: non-scaling-stroke;
}
.x-axis .stage {
  color: var(--zen-accent);
  transform: translateX(2px);
  opacity: 0.8;
  pointer-events: auto;
}
/* the plot only: legend swatches are small inline svgs */
.plot > svg {
  width: 100%;
  height: 100%;
  display: block;
  touch-action: none;
}
.grid line {
  stroke: var(--zen-border);
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}
.mark {
  stroke: var(--zen-muted);
  stroke-dasharray: 2 4;
  opacity: 0.45;
  vector-effect: non-scaling-stroke;
}
.mark.done {
  stroke: var(--zen-accent);
  opacity: 0.6;
}
.ema {
  fill: none;
  stroke: var(--zen-accent);
  vector-effect: non-scaling-stroke;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.ema {
  stroke-width: 2;
}
.cross {
  stroke: var(--zen-muted);
  stroke-dasharray: 3 3;
  vector-effect: non-scaling-stroke;
}
.dot {
  fill: var(--zen-accent);
  stroke: var(--zen-surface);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
.y-axis span,
.x-axis span {
  position: absolute;
  font-size: 10px;
  color: var(--zen-muted);
  font-family: var(--zen-mono);
  pointer-events: none;
}
.y-axis span {
  left: 0;
  transform: translateY(-50%);
}
.x-axis span {
  bottom: 0;
  transform: translateX(-50%);
}
.tip {
  position: absolute;
  top: 8px;
  padding: 6px 8px;
  background: var(--zen-surface);
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  font-size: 12px;
  pointer-events: none;
  white-space: nowrap;
  box-shadow: 0 4px 14px rgb(0 0 0 / 0.25);
}
</style>
<style scoped>
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
  font-size: 12px;
  font-weight: 400;
  color: var(--zen-muted);
}
.key {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.k-ema {
  stroke: var(--zen-accent);
  stroke-width: 2;
}
.k-val {
  stroke: var(--chart-2, var(--zen-info));
  stroke-width: 2;
  stroke-dasharray: 4 3;
}
.k-val-dot,
.val-dot {
  fill: var(--chart-2, var(--zen-info));
  stroke: var(--zen-surface);
  stroke-width: 1.5;
}
.k-mark {
  stroke: var(--zen-muted);
  stroke-dasharray: 2 3;
}
.val {
  fill: none;
  stroke: var(--chart-2, var(--zen-info));
  stroke-width: 2;
  stroke-dasharray: 5 4;
  vector-effect: non-scaling-stroke;
}
.val-dot {
  vector-effect: non-scaling-stroke;
}
.like,
.k-like {
  fill: none;
  stroke: var(--zen-ok);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
.like-dot,
.k-like-dot {
  fill: var(--zen-ok);
  stroke: var(--zen-surface);
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.like-ref {
  stroke: var(--zen-ok);
  stroke-width: 1;
  stroke-dasharray: 2 4;
  opacity: 0.3;
  vector-effect: non-scaling-stroke;
}
</style>
