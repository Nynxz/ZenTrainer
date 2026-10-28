<script setup lang="ts">
// The rail's numbers, in two cards: Time (how long, how fast, the step's phases) and Learning
// (LR, grad norm, losses, likeness), each value's meaning behind an ⓘ.
import { ZenTip } from '@nynxz/zenkit-ui'
import { computed } from 'vue'
import { gb } from '@/stores/system'
import { SHARE_TIP, likeShare } from '@/lib/likeness'
import { isCollapsed, toggleCollapsed } from './collapse'
import { dur, fmtLr, useRunContext } from './context'

const { detail, last, vramPeak, wall, likeLast, likeFirst, likeCeil } = useRunContext()

// a grad norm or loss as it fits the cell: 3 decimals while it's ordinary, "6.3e16" when a step's
// gradients blew up (the full number is in the cell's hover)
function fmtBig(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–'
  return Math.abs(v) >= 1000 ? v.toExponential(1).replace('e+', 'e') : v.toFixed(digits)
}

const valLast = computed(() => detail.value?.validation.at(-1))
const valFirst = computed(() => detail.value?.validation[0])
interface Stat {
  k: string
  v: string
  /** what it means, behind an ⓘ */
  tip?: string
  mono?: boolean
  /** the whole value, on hover, when v is shortened */
  full?: string
  delta?: { good: boolean; text: string } | null
}
const statGroups = computed((): { title: string; items: Stat[] }[] => {
  const d = detail.value
  if (!d) return []
  const l = last.value
  const valDelta =
    valLast.value && valFirst.value && valLast.value !== valFirst.value
      ? {
          good: valLast.value.val_loss < valFirst.value.val_loss,
          text: ` ${valLast.value.val_loss < valFirst.value.val_loss ? '▼' : '▲'}${Math.abs(((valLast.value.val_loss - valFirst.value.val_loss) / valFirst.value.val_loss) * 100).toFixed(0)}%`,
        }
      : null
  // likeness rises as it learns, from about 0 (a stranger) before training: the change is in
  // points, not percent (0.05 -> 0.6 would read +1100%)
  const likeDelta =
    likeLast.value && likeFirst.value && likeLast.value.step !== likeFirst.value.step
      ? {
          good: likeLast.value.score >= likeFirst.value.score,
          text: ` ${likeLast.value.score >= likeFirst.value.score ? '▲' : '▼'}${Math.abs(likeLast.value.score - likeFirst.value.score).toFixed(2)}`,
        }
      : null
  return [
    {
      title: 'Time',
      items: [
        { k: 'Training', v: dur(l?.train_time), tip: 'Time spent in training steps only: no loading, sampling or validation' },
        { k: 'Total', v: dur(wall.value), tip: 'From start to now, with loading, sampling and validation' }, // start to now, loading and sampling included
        { k: 'Speed', v: `${l?.it_s?.toFixed(2) ?? '–'} it/s`, tip: 'Training steps per second, lately. The first steps are slower (caching, compiling)' },
        { k: 'Epoch', v: String(l?.epoch ?? '–'), tip: 'Passes through the dataset so far, repeats included' },
        ...(vramPeak.value ? [{ k: 'Peak VRAM', v: `${gb(vramPeak.value)} GB`, tip: 'The most GPU memory a training step has used. Sampling can use more, briefly' }] : []),
        ...(l?.phases
          ? [
              { k: 'Forward / back', v: `${l.phases.forward.toFixed(1)} / ${l.phases.backward.toFixed(1)} s`, tip: 'Seconds per step running the model forward, and back for the gradients: nearly all of a step' },
              { k: 'Data / optimizer', v: `${l.phases.data.toFixed(2)} / ${l.phases.optimizer.toFixed(2)} s`, tip: 'Seconds per step fetching the batch, and updating the LoRA. Near 0 is right' },
            ]
          : []),
      ],
    },
    {
      title: 'Learning',
      items: [
        { k: 'Learning rate', v: fmtLr(l?.lr), tip: "The step size now. Automagic and Prodigy set their own as they go; others follow the schedule" },
        { k: 'Grad norm', v: fmtBig(l?.grad_norm), full: l?.grad_norm != null ? String(l.grad_norm) : undefined, tip: 'How strongly this step pulled on the LoRA (before clipping). Spikes are hard pictures; a steady climb can mean it’s going unstable' },
        {
          k: 'Validation',
          v: valLast.value?.val_loss != null ? valLast.value.val_loss.toFixed(4) : '–',
          delta: valDelta,
          tip: 'Loss on a fixed set: the same pictures at the same noise, every time, so it moves only when the LoRA does (the training loss jumps with each step’s random noise). The arrow: change since before training. It’s drawn from the training pictures, so it shows learning, not overfitting: judge that by the samples',
        },
        { k: 'Loss', v: fmtBig(l?.loss, 4), full: l?.loss != null ? String(l.loss) : undefined, tip: 'This step’s loss. Noisy: each step picks a random picture and noise level. The chart’s smoothed line is the trend' },
        ...(likeLast.value
          ? [{
              k: 'Likeness',
              v: (likeDelta ? likeLast.value.score.toFixed(2) : `${likeLast.value.score.toFixed(2)} @ ${likeLast.value.step}`)
                + (likeCeil.value ? ` (${likeShare(likeLast.value.score, likeCeil.value)})` : ''),
              delta: likeDelta,
              tip: `At step ${likeLast.value.step}: how much the samples' faces look like the dataset's (mean of the step's samples): 1 is identical, about 0.5 and up reads as the same person, near 0 a stranger.${likeCeil.value ? ` In brackets: ${SHARE_TIP.toLowerCase()}, ${likeCeil.value.toFixed(2)} here.` : ''} The arrow: change since the first scored samples. ${likeFirst.value && likeFirst.value.step !== likeLast.value.step ? `It was ${likeFirst.value.score.toFixed(2)} at step ${likeFirst.value.step}.` : ''}`,
            }]
          : []),
      ],
    },
  ]
})
</script>

<template>
  <section v-for="g in statGroups" :key="g.title" class="card">
    <header
      class="sec-head fold"
      :class="{ folded: isCollapsed(g.title) }"
      role="button"
      tabindex="0"
      :aria-expanded="!isCollapsed(g.title)"
      :title="isCollapsed(g.title) ? 'Show' : 'Fold away'"
      @click="toggleCollapsed(g.title)"
      @keydown.enter.self.prevent="toggleCollapsed(g.title)"
      @keydown.space.self.prevent="toggleCollapsed(g.title)"
    >
      <span class="sec-title">{{ g.title }}</span><span class="spacer" />
      <i class="mdi fold-chev" :class="isCollapsed(g.title) ? 'mdi-chevron-down' : 'mdi-chevron-up'" />
    </header>
    <dl v-if="!isCollapsed(g.title)" class="sgrid">
      <div v-for="it in g.items" :key="it.k" class="scell">
        <dt><ZenTip v-if="it.tip" :text="it.tip">{{ it.k }}</ZenTip><template v-else>{{ it.k }}</template></dt>
        <dd :class="{ mono: it.mono !== false }">
          <span class="v" :title="it.full ?? it.v">{{ it.v }}</span><span v-if="it.delta" class="delta" :class="it.delta.good ? 'good' : 'bad'">{{ it.delta.text.trim() }}</span>
        </dd>
      </div>
    </dl>
  </section>
</template>

<style scoped src="./section.css"></style>
<style scoped>
.sgrid {
  margin: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 14px;
}
.scell {
  min-width: 0;
}
.scell dt {
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 3px;
}

/* the change (the green or amber arrow) is what matters most: when the cell is too narrow for
   value and change it goes under the value, never cut off; only the value can shorten */
.scell dd {
  margin: 2px 0 0;
  font-size: 13px;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  column-gap: 4px;
  white-space: nowrap;
}
.scell dd .v {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}
.delta {
  flex: none;
  font-size: 11px;
}
dt {
  font-size: 12px;
  color: var(--zen-muted);
  white-space: nowrap;
}
dd {
  margin: 0;
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.good {
  color: var(--zen-ok);
  font-size: 11px;
}
.bad {
  color: var(--zen-warn);
  font-size: 11px;
}
</style>
