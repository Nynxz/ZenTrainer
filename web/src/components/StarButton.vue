<script setup lang="ts">
// Star a model family, or a dataset. Safe inside a dropdown option: the click doesn't pick the option.
import { computed } from 'vue'
import { isDatasetStarred, isStarred, toggleDatasetStar, toggleStar } from '@/stores/favorites'

const props = defineProps<{ id: string; name?: string; kind?: 'model' | 'dataset' }>()
const dataset = computed(() => props.kind === 'dataset')
const on = computed(() => (dataset.value ? isDatasetStarred(props.id) : isStarred(props.id)))
const tip = computed(() =>
  on.value
    ? `Unstar ${props.name ?? (dataset.value ? 'this dataset' : 'this model')}`
    : `Star ${props.name ?? (dataset.value ? 'this dataset' : 'this model')}: it goes to the top of ${dataset.value ? 'the dataset picker' : 'model lists'}`,
)
const toggle = () => (dataset.value ? toggleDatasetStar(props.id) : toggleStar(props.id))
</script>

<template>
  <button
    type="button"
    class="star"
    :class="{ on }"
    :aria-pressed="on"
    :title="tip"
    @click.stop.prevent="toggle"
    @mousedown.stop
    @pointerdown.stop
  >
    <i :class="on ? 'mdi mdi-star' : 'mdi mdi-star-outline'" />
  </button>
</template>

<style scoped>
.star {
  display: inline-grid;
  place-items: center;
  width: 22px;
  height: 22px;
  flex: none;
  padding: 0;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  opacity: 0.55;
  cursor: pointer;
  font-size: 15px;
}
.star:hover,
.star:focus-visible {
  opacity: 1;
  color: var(--zen-text);
}
.star.on {
  opacity: 1;
  color: var(--zen-warn);
}
</style>
