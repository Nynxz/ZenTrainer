<script setup lang="ts">
// Two (or more) runs on one chart: smoothed loss, validation loss, likeness, LR, or auxiliary
// objectives. Hover shows every run's nearest value plus B - A delta; clicking emits the nearest
// step so the sample viewer can jump straight there.
import { maxOf, minOf } from '@/lib/extent'
import { computed, ref } from 'vue'

export type CompareMetric = 'loss' | 'val' | 'like' | 'lr' | 'preserve' | 'face_id'

export interface ComparePoint {
  step: number
  v: number
}

export interface CompareSeries {
  label: string
  color: string
  maxStep: number
  pts: ComparePoint[]
}

const props = withDefaults(
  defineProps<{
    series: CompareSeries[]
    metric: CompareMetric
    height?: number
    /** step currently selected in the sample viewer, drawn as a subtle marker */
    activeStep?: number | null
  }>(),
  { height: 240, activeStep: null },
)

const emit = defineEmits<{ 'pick-step': [step: number] }>()

const W = 720
const pad = { l: 52, r: 16, t: 12, b: 26 }

const all = computed(() => props.series.flatMap((s) => s.pts))
const xMax = computed(() =>
  Math.max(
    1,
    maxOf(
      all.value.map((p) => p.step),
      1,
    ),
    maxOf(
      props.series.map((s) => s.maxStep),
      1,
    ),
  ),
)
const yRange = computed(() => {
  if (props.metric === 'like') return [0, 1] as const
  const v = all.value.map((p) => p.v)
  if (!v.length) return [0, 1] as const
  const lo = minOf(v)
  const hi = maxOf(v)
  const py = (hi - lo || Math.abs(hi) || 1) * 0.08
  return [Math.max(0, lo - py), hi + py] as const
})
const x = (step: number) => pad.l + (step / xMax.value) * (W - pad.l - pad.r)
const y = (v: number) => {
  const [lo, hi] = yRange.value
  return pad.t + (1 - (v - lo) / (hi - lo || 1)) * (props.height - pad.t - pad.b)
}
const pathOf = (pts: ComparePoint[]) =>
  pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p.v).toFixed(1)}`).join('')
const yTicks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((f) => yRange.value[0] + f * (yRange.value[1] - yRange.value[0])))
const xTicks = computed(() => [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * xMax.value)))
const dots = computed(() => props.metric === 'val' || props.metric === 'like')

function fmt(v: number): string {
  if (props.metric === 'like') return v.toFixed(2)
  if (props.metric === 'lr') return v === 0 ? '0' : v < 1e-2 ? v.toExponential(1).replace('e-0', 'e-') : v.toFixed(4)
  return Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toExponential(2)
}

function fmtDelta(a: number, b: number): string {
  const d = b - a
  const sign = d >= 0 ? '+' : '−'
  const mag = Math.abs(d)
  if (props.metric === 'like') return `${sign}${mag.toFixed(2)}`
  if (props.metric === 'lr') return `${sign}${mag < 1e-2 ? mag.toExponential(1).replace('e-0', 'e-') : mag.toFixed(4)}`
  return `${sign}${mag >= 0.001 ? mag.toFixed(4) : mag.toExponential(1)}`
}

const emptyText = computed(() => {
  switch (props.metric) {
    case 'val':
      return 'Neither run has validation loss'
    case 'like':
      return 'Neither run has likeness scores'
    case 'lr':
      return 'No learning-rate history'
    case 'preserve':
      return 'Neither run logged class preservation loss'
    case 'face_id':
      return 'Neither run logged face identity loss'
    default:
      return 'No loss yet'
  }
})

const hoverStep = ref<number | null>(null)
function stepFromEvent(ev: PointerEvent | MouseEvent): number {
  const box = (ev.currentTarget as SVGSVGElement).getBoundingClientRect()
  const sx = ((ev.clientX - box.left) / box.width) * W
  return Math.max(0, Math.min(xMax.value, ((sx - pad.l) / (W - pad.l - pad.r)) * xMax.value))
}
function onMove(ev: PointerEvent) {
  hoverStep.value = stepFromEvent(ev)
}
function onClick(ev: MouseEvent) {
  emit('pick-step', Math.round(stepFromEvent(ev)))
}

const hovered = computed(() => {
  const h = hoverStep.value
  if (h == null) return null
  return props.series.map((s) => {
    let best: ComparePoint | null = null
    for (const p of s.pts) if (!best || Math.abs(p.step - h) < Math.abs(best.step - h)) best = p
    return { label: s.label, color: s.color, p: best }
  })
})
const hoverDelta = computed(() => {
  const h = hovered.value
  if (!h || h.length < 2 || !h[0]?.p || !h[1]?.p) return null
  const a = h[0].p.v
  const b = h[1].p.v
  const better = props.metric === 'like' ? b >= a : props.metric === 'lr' ? null : b <= a
  return { text: fmtDelta(a, b), better }
})
</script>

<template>
  <figure class="cmp-chart">
    <svg
      :viewBox="`0 0 ${W} ${height}`"
      class="plot"
      @pointermove="onMove"
      @pointerleave="hoverStep = null"
      @click="onClick"
    >
      <g class="grid">
        <line v-for="t in yTicks" :key="`y${t}`" :x1="pad.l" :x2="W - pad.r" :y1="y(t)" :y2="y(t)" />
      </g>
      <g class="axis">
        <text v-for="t in yTicks" :key="`yl${t}`" :x="pad.l - 6" :y="y(t) + 3" text-anchor="end">{{ fmt(t) }}</text>
        <text v-for="t in xTicks" :key="`xl${t}`" :x="x(t)" :y="height - 7" text-anchor="middle">{{ t }}</text>
      </g>
      <line v-if="metric === 'like'" class="half" :x1="pad.l" :x2="W - pad.r" :y1="y(0.5)" :y2="y(0.5)" />
      <line
        v-if="activeStep != null && activeStep > 0 && activeStep <= xMax"
        class="active-step"
        :x1="x(activeStep)"
        :x2="x(activeStep)"
        :y1="pad.t"
        :y2="height - pad.b"
      />
      <g v-for="l in series" :key="l.label">
        <path v-if="l.pts.length" :d="pathOf(l.pts)" fill="none" :stroke="l.color" stroke-width="2" stroke-linejoin="round" />
        <template v-if="dots">
          <circle v-for="p in l.pts" :key="p.step" :cx="x(p.step)" :cy="y(p.v)" r="3.5" :fill="l.color" class="dot" />
        </template>
      </g>
      <template v-if="hovered && hoverStep != null">
        <line class="cursor" :x1="x(hoverStep)" :x2="x(hoverStep)" :y1="pad.t" :y2="height - pad.b" />
        <circle
          v-for="h in hovered"
          :key="`h-${h.label}`"
          v-show="h.p"
          :cx="h.p ? x(h.p.step) : 0"
          :cy="h.p ? y(h.p.v) : 0"
          r="4"
          :fill="h.color"
          class="dot"
        />
      </template>
    </svg>
    <div v-if="!all.length" class="empty muted small">{{ emptyText }}</div>
    <div class="readout small">
      <template v-if="hovered">
        <span v-for="h in hovered" :key="h.label" class="ro">
          <i class="sw" :style="{ background: h.color }" />{{ h.label }}:
          <b class="mono">{{ h.p ? fmt(h.p.v) : '–' }}</b><span v-if="h.p" class="muted"> @ {{ h.p.step }}</span>
        </span>
        <span
          v-if="hoverDelta"
          class="ro delta mono"
          :class="{ good: hoverDelta.better === true, bad: hoverDelta.better === false }"
        >
          Δ {{ hoverDelta.text }}
        </span>
      </template>
      <span v-else class="muted">Hover for values · click a step to jump samples</span>
    </div>
  </figure>
</template>

<style scoped>
.cmp-chart {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  position: relative;
}
.plot {
  width: 100%;
  height: auto;
  display: block;
  cursor: crosshair;
}
.grid line {
  stroke: color-mix(in srgb, var(--zen-border) 55%, transparent);
  stroke-width: 1;
}
.axis text {
  fill: var(--zen-muted);
  font-size: 10px;
  font-family: var(--zen-mono, monospace);
}
.half {
  stroke: var(--zen-muted);
  stroke-dasharray: 2 4;
  opacity: 0.45;
}
.active-step {
  stroke: var(--zen-accent);
  stroke-width: 1.25;
  stroke-dasharray: 4 3;
  opacity: 0.55;
}
.dot {
  stroke: var(--zen-surface);
  stroke-width: 2;
}
.cursor {
  stroke: var(--zen-muted);
  stroke-dasharray: 3 3;
}
.empty {
  position: absolute;
  inset: 16px 0 30px 0;
  display: grid;
  place-items: center;
  pointer-events: none;
}
.readout {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 14px;
  min-height: 20px;
}
.ro {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.delta {
  padding: 1px 6px;
  border-radius: 999px;
  background: var(--zen-surface-2);
  font-size: 11px;
}
.delta.good {
  color: var(--zen-ok);
}
.delta.bad {
  color: var(--zen-warn);
}
.sw {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  display: inline-block;
}
.mono {
  font-family: var(--zen-mono, monospace);
}
</style>
