<script setup lang="ts">
// What the LoRA spends its steps on. A flow model learns layout and pose at heavy noise
// (composition), shapes and features in the middle (form), faces, texture and fine detail at
// light noise (detail). The chart shows where this run's steps land along that, against the
// model's own schedule; the slider leans it toward either end (it sets the timestep shift).
import { ZenSlider } from '@nynxz/zenkit-ui'
import { computed, toRef } from 'vue'
import type { NoiseProfile, TrainSpec } from '@/api/engine'
import NoiseChart, { type NoiseSeries } from '@/components/NoiseChart.vue'
import { remembered } from '@/lib/remembered'
import { bands, draws, FLEX_WEIGHTS, histogram, levels, useSchedule } from '@/lib/timesteps'

const props = defineProps<{ objective: TrainSpec['objective']; profile: NoiseProfile }>()

const BASE = 9 // focus -1..1 -> shift 9..1/9: an even split at 0, ~82% in one end band at the ends
const shiftOf = (f: number) => Number((BASE ** -f).toFixed(2))
const focusOf = (shift: number) => -Math.log(shift) / Math.log(BASE)

const ownDraws = computed(() => draws({ ...props.objective, timesteps: 'uniform', shift: 0, bias: 'balanced' }, props.profile))
const exact = useSchedule(toRef(props, 'profile'), () => props.objective)
const nowDraws = computed(() => draws(props.objective, props.profile, 20000, exact.value))
const own = computed(() => bands(ownDraws.value))
const now = computed(() => bands(nowDraws.value))
const scheduled = computed(() => props.objective.timesteps === 'scheduler')
const series = computed((): NoiseSeries[] => [
  scheduled.value
    ? { label: `This run: ${props.objective.timestep_scheduler ?? 'simple'}'s ${props.objective.timestep_scheduler_steps ?? 20} steps (ticks)`, hist: histogram(nowDraws.value, 48, 6), levels: levels(nowDraws.value), kind: 'area' }
    : { label: 'This run', hist: histogram(nowDraws.value), kind: 'area' },
  { label: props.profile.model ? `${props.profile.model}'s own schedule` : "The model's own schedule", hist: histogram(ownDraws.value), kind: 'line' },
])
// the chart folds away (remembered); folded, one line still says where the steps go
const chartOpen = remembered('zentrainer.newRun.noiseChart', false)
const pct = (v: number) => `${Math.round(v * 100)}%`
const ZONES = ['Composition', 'Form', 'Detail']
// the slider's position for the model's own schedule: the shift that gives its detail share
const ownFocus = computed(() => focusOf(Math.max(1e-3, (1 / Math.max(own.value[2]!, 1e-3) - 1) / 2)))
const focus = computed({
  get: () => (props.objective.shift > 0 ? focusOf(props.objective.shift) : ownFocus.value),
  set: (f: number) => (props.objective.shift = shiftOf(f)),
})
</script>

<template>
  <div class="focus">
    <button class="summary" :aria-expanded="chartOpen" title="Share of the steps in each part of the schedule: this run / the model's own" @click="chartOpen = !chartOpen">
      <span v-for="(z, i) in ZONES" :key="z" class="zs" :class="`z${i}`">{{ z }} <b>{{ pct(now[i]!) }}</b><span class="muted"> / {{ pct(own[i]!) }}</span></span>
      <span class="spacer" />
      <span class="small muted toggle">{{ chartOpen ? 'Hide chart' : 'Show chart' }}<i :class="chartOpen ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /></span>
    </button>
    <NoiseChart v-if="chartOpen" :series="series" :zones="now" :zones-compare="own" :weights="objective.weighting === 'flex' ? FLEX_WEIGHTS : null" />
    <div v-if="!scheduled" class="lean">
      <span class="small muted">Lean</span>
      <div class="slide">
        <ZenSlider v-model="focus" :min="-1" :max="1" :step="0.05" />
        <div class="mark" :style="{ left: `${((ownFocus + 1) / 2) * 100}%` }" title="The model's own" />
      </div>
      <span class="small muted state">
        {{ objective.shift > 0 ? `shift ${objective.shift.toFixed(2)}` : "model's own" }}
        <button v-if="objective.shift > 0" class="link" title="Back to the model's own schedule" @click="objective.shift = 0">reset</button>
      </span>
    </div>
  </div>
</template>

<style scoped>
.focus {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.summary {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 14px;
  padding: 6px 10px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
}
.summary:hover {
  border-color: var(--zen-border);
}
.zs b {
  color: var(--zen-accent);
  font-variant-numeric: tabular-nums;
}
.spacer {
  flex: 1;
}
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  white-space: nowrap;
}
.lean {
  display: grid;
  /* the state column keeps one width, so the slider doesn't resize as the shift's digits change */
  grid-template-columns: auto minmax(0, 1fr) 9.5em;
  align-items: center;
  gap: 10px;
}
.slide {
  position: relative;
}
.mark {
  position: absolute;
  top: 50%;
  width: 2px;
  height: 14px;
  margin: -7px 0 0 -1px;
  background: var(--zen-muted);
  border-radius: 1px;
  pointer-events: none;
}
.state {
  white-space: nowrap;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.link {
  border: none;
  background: none;
  padding: 0 0 0 4px;
  font: inherit;
  color: var(--zen-accent);
  cursor: pointer;
}
</style>
