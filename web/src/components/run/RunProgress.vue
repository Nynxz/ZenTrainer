<script setup lang="ts">
// The run page's one slim progress line: steps with the save and sample notches, or, when the
// run isn't stepping, what it's doing instead (sampling, compiling, queued, preparing, starting).
import { computed, ref, watch } from 'vue'
import { dur, useRunContext } from './context'

const { detail, sampling, last, total, step, progress, status, running, marks, now } = useRunContext()

const eta = computed(() => {
  const l = last.value
  if (!l?.it_s || status.value !== 'training') return null
  return (total.value - l.step) / l.it_s
})

// Track when a training status message (e.g. torch.compile) started so the user sees live elapsed
// time rather than a screen that looks frozen.
const activeMsg = computed(() => (status.value === 'training' ? detail.value?.run.message || '' : ''))
const msgSince = ref<number | null>(null)
watch(
  activeMsg,
  (m, prev) => {
    if (m && m !== prev) msgSince.value = Date.now() / 1000
    else if (!m) msgSince.value = null
  },
  { immediate: true },
)

// what the run is doing when it isn't stepping: shown in the step bar's place (same size), with
// its own progress where there is one (a sample's steps, the dataset's "(12/120)")
const phase = computed(() => {
  const d = detail.value
  if (!d || !running.value) return null
  const sm = sampling.value
  if (sm && sm.phase !== 'done') {
    const baseline = step.value === 0 && d.spec.samples.baseline
    const st = sm.s_total ?? 0
    const s0 = sm.s_step ?? 0
    return {
      icon: 'mdi mdi-image-filter-center-focus-weak',
      label: baseline ? 'Baseline' : 'Sampling',
      text: (baseline ? 'The model before training · ' : '') + (sm.prompt ?? ''),
      frac: st ? s0 / st : 0,
      indeterminate: !st,
      right: `${sm.count ? `${(sm.index ?? 0) + 1}/${sm.count} · ` : ''}${st && s0 >= st ? 'decoding' : st ? `${s0}/${st}` : ''}`,
    }
  }
  if (status.value === 'queued') return { icon: 'mdi mdi-clock-outline', label: 'Queued', text: 'Waiting for its turn', frac: 0, indeterminate: false, right: '' }
  if (status.value === 'loading') {
    const msg = d.run.message || 'Loading the model and the dataset'
    const m = msg.match(/\((\d+)\/(\d+)\)\s*$/)
    return {
      icon: 'mdi mdi-database-arrow-down-outline',
      label: 'Preparing',
      text: m ? msg.slice(0, m.index).trim() : msg,
      frac: m ? Number(m[1]) / Number(m[2]) : 0,
      indeterminate: !m,
      right: m ? `${m[1]}/${m[2]}` : '',
    }
  }
  if (status.value === 'training' && activeMsg.value) {
    const compiling = /compil/i.test(activeMsg.value)
    const elapsed = msgSince.value != null ? Math.max(0, now.value - msgSince.value) : 0
    return {
      icon: compiling ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-progress-helper',
      label: compiling ? 'Compiling' : 'Working',
      text: activeMsg.value,
      frac: 0,
      indeterminate: true,
      right: `${step.value > 0 ? `step ${step.value + 1} · ` : ''}${dur(elapsed)}`,
    }
  }
  if (status.value === 'training' && !last.value) {
    return { icon: 'mdi mdi-progress-helper', label: 'Starting', text: 'First step', frac: 0, indeterminate: true, right: '' }
  }
  return null
})
const etaLine = computed(() => {
  const pct = `${Math.round(progress.value * 100)}%`
  if (eta.value != null) return `${pct} · ${dur(eta.value)} left`
  if (status.value === 'done') return `${pct} · in ${dur(last.value?.train_time)}`
  return pct
})
</script>

<template>
  <!-- one slim line: fixed-width text columns either side, so the bar (and its notches) never moves -->
  <div v-if="phase" class="progline card phase">
    <span class="pl-step"><i :class="phase.icon" /> {{ phase.label }}</span>
    <div class="ph-mid">
      <span class="ph-text ellipsis" :title="phase.text">{{ phase.text }}</span>
      <div class="bar thin" :class="{ indeterminate: phase.indeterminate }"><div :style="phase.indeterminate ? undefined : { width: `${phase.frac * 100}%` }" /></div>
    </div>
    <span class="pl-eta mono muted">{{ phase.right }}</span>
  </div>
  <div v-else class="progline card">
    <span class="pl-step mono"><b>{{ step }}</b><span class="muted"> / {{ total }}</span></span>
    <div class="track">
      <div class="bar"><div :style="{ width: `${progress * 100}%` }" /></div>
      <span
        v-for="m in marks"
        :key="`${m.kind}${m.step}`"
        class="notch"
        :class="[m.kind, { done: m.done }]"
        :style="{ left: `${(m.step / total) * 100}%` }"
        :title="`${m.kind === 'ckpt' ? 'Checkpoint' : 'Samples'} at step ${m.step}${m.done ? ' (done)' : ''}`"
      />
    </div>
    <span class="pl-eta mono muted">{{ etaLine }}</span>
  </div>
</template>

<style scoped>
.phase .pl-step {
  color: var(--zen-accent);
  font-weight: 600;
}
.ph-mid {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 3px;
  height: 18px; /* the step bar's height: switching between them doesn't move the page */
  min-width: 0;
}
.ph-text {
  font-size: 11px;
  color: var(--zen-muted);
  line-height: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bar.thin {
  height: 3px;
}
.bar.indeterminate {
  position: relative;
}
.bar.indeterminate div {
  position: absolute;
  inset: 0 auto 0 0;
  width: 34%;
  border-radius: var(--r-sm);
  animation: slide 1.5s ease-in-out infinite;
}
@keyframes slide {
  0% {
    transform: translateX(-110%);
  }
  100% {
    transform: translateX(310%);
  }
}
.progline {
  display: grid;
  grid-template-columns: 11ch minmax(0, 1fr) 20ch;
  align-items: center;
  gap: 16px;
  padding: 12px 18px;
  font-variant-numeric: tabular-nums;
}
.pl-step {
  font-size: 13px;
  white-space: nowrap;
}
.pl-eta {
  font-size: 12px;
  text-align: right;
  white-space: nowrap;
}
.track {
  position: relative;
  padding: 5px 0;
}
.bar {
  height: 6px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.bar div {
  height: 100%;
  background: var(--zen-accent);
}
.notch {
  position: absolute;
  top: 1px;
  width: 2px;
  height: 14px;
  margin-left: -1px;
  border-radius: var(--r-sm);
  background: var(--zen-muted);
  opacity: 0.5;
}
.notch.sample {
  top: 4px;
  width: 8px;
  height: 8px;
  margin-left: -4px;
  border-radius: var(--r-sm);
  background: var(--zen-surface);
  border: 1.5px solid var(--zen-muted);
}
/* done marks in the text colour: the accent would vanish into the filled bar */
.notch.ckpt.done {
  background: var(--zen-text);
  opacity: 0.85;
}
.notch.sample.done {
  background: var(--zen-text);
  border-color: var(--zen-text);
  opacity: 0.85;
}
@media (max-width: 1000px) {
  .progline {
    grid-template-columns: auto minmax(0, 1fr);
  }
  .pl-eta {
    grid-column: 1 / -1;
    text-align: left;
  }
}
</style>
