<script setup lang="ts">
// Per-sample indicator pills (matching DatasetGrid's corner badges): face likeness score,
// missing-face / multi-face warnings, and skipped-likeness markers on Latest, Progression and
// Manual sample tiles.
import { computed } from 'vue'
import type { LikenessRow } from '@/api/engine'
import { likeShare } from '@/lib/likeness'

const props = withDefaults(
  defineProps<{
    /** this sample's likeness row, when scored */
    likeness?: LikenessRow | null
    /** fallback dataset ceiling (when the row itself predates per-row ceiling) */
    ceiling?: number | null
    /** prompt has likeness: false in the spec */
    skipped?: boolean
    /** smaller pills for Progression / Manual thumbnails */
    compact?: boolean
    /** corner placement to stay clear of tile numbers, play icons, and bottom labels */
    corner?: 'bottom-left' | 'bottom-right' | 'top-right'
  }>(),
  {
    likeness: null,
    ceiling: null,
    skipped: false,
    compact: false,
    corner: 'bottom-left',
  },
)

const effCeiling = computed(() => props.likeness?.ceiling ?? props.ceiling)
const share = computed(() => (props.likeness?.score != null ? likeShare(props.likeness.score, effCeiling.value) : ''))
const multi = computed(() => (props.likeness?.faces ?? 0) > 1)
const scoreTip = computed(() => {
  const r = props.likeness
  if (!r || r.score == null) return ''
  const parts = [`Face likeness ${r.score.toFixed(2)}`]
  if (share.value && effCeiling.value) parts.push(`${share.value} of dataset ceiling ${effCeiling.value.toFixed(2)}`)
  if (multi.value) parts.push(`${r.faces} faces detected (scored closest)`)
  return parts.join(' · ')
})
const hasAny = computed(() => props.skipped || props.likeness != null)
</script>

<template>
  <span v-if="hasAny" class="s-badges" :class="[corner, { compact }]">
    <span
      v-if="skipped"
      class="s-badge muted"
      title="Likeness skipped for this prompt (not counted in average likeness)"
    >
      <i class="mdi mdi-eye-off-outline" />
    </span>
    <template v-else-if="likeness">
      <span
        v-if="likeness.score != null"
        class="s-badge mono"
        :title="scoreTip"
      >
        <i class="mdi mdi-face-recognition" /> {{ likeness.score.toFixed(2) }}
      </span>
      <span
        v-else
        class="s-badge warn"
        title="No face detected in this sample (not counted in step average)"
      >
        <i class="mdi mdi-face-man-outline" /> 0
      </span>
      <span
        v-if="multi"
        class="s-badge warn mono"
        :title="`${likeness.faces} faces detected — scored the closest match`"
      >
        <i class="mdi mdi-account-multiple-outline" /> {{ likeness.faces }}
      </span>
    </template>
  </span>
</template>

<style scoped>
.s-badges {
  position: absolute;
  z-index: 2;
  display: flex;
  align-items: center;
  gap: 4px;
  pointer-events: none;
}
.s-badges.bottom-left {
  left: 6px;
  bottom: 6px;
}
.s-badges.bottom-right {
  right: 6px;
  bottom: 6px;
}
.s-badges.top-right {
  top: 6px;
  right: 6px;
}
.s-badges.compact.bottom-left {
  left: 4px;
  bottom: 4px;
  gap: 3px;
}
.s-badges.compact.top-right {
  top: 4px;
  right: 4px;
  gap: 3px;
}
.s-badge {
  font-size: 11px;
  min-width: 22px;
  height: 22px;
  padding: 0 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.6);
  color: #fff;
  pointer-events: auto;
}
.s-badges.compact .s-badge {
  font-size: 10px;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  gap: 2px;
}
.s-badge.warn {
  color: var(--zen-warn, #fbbf24);
}
.s-badge.muted {
  color: rgb(255 255 255 / 0.7);
}
</style>
