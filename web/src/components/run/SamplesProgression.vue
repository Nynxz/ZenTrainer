<script setup lang="ts">
// Progression: each sample prompt across training, a row of thumbnails per prompt and a column
// per sampled step, newest at the right (scrolled into view when it lands). Edit runs can head
// each row with its input picture.
import { nextTick, onMounted, ref, watch } from 'vue'
import { viewUrl } from '@/api/engine'
import ScrollArea from '@/components/ScrollArea.vue'
import { useRunContext } from './context'
import SampleBadges from './SampleBadges.vue'
import { isVideo, rowHasVideo, useSamplesContext } from './samples'

defineProps<{ size: number }>()
const { detail, total, likeCeil } = useRunContext()
const { steps, prompts, cell, openGrid, syncing, toggleSync, hoverPlay, hoverPause, inputFor, hasInputs, compare, likenessFor, isLikenessSkipped } = useSamplesContext()

// a step that isn't on the sampling schedule: someone pressed Sample now
const isManual = (s: number) => {
  const every = detail.value?.spec.samples.every ?? 0
  return s !== 0 && s !== total.value && (!every || s % every !== 0)
}
const gridScroll = ref<InstanceType<typeof ScrollArea> | null>(null)
// newest steps in view: when new ones land, and when Progression opens
async function toEnd() {
  await nextTick()
  gridScroll.value?.scrollToEnd()
}
watch(() => steps.value.length, toEnd)
onMounted(toEnd)
</script>

<template>
  <ScrollArea ref="gridScroll" axis="x" class="matrix-scroll">
    <div class="matrix" :style="{ '--t': `${size}px` }">
      <div class="mrow mhead">
        <div v-if="hasInputs && compare" class="mstep mono input-col">input</div>
        <div
          v-for="s in steps"
          :key="s"
          class="mstep mono"
          :class="{ manual: isManual(s) }"
          :title="isManual(s) ? 'Sampled on demand (Sample now)' : undefined"
        >
          {{ s === 0 ? 'base' : s }}<i v-if="isManual(s)" class="mdi mdi-gesture-tap" />
        </div>
      </div>
      <template v-for="[index, prompt] in prompts" :key="index">
        <div class="mlabel" :title="prompt">
          <span class="pnum">{{ index + 1 }}</span>
          <button
            v-if="rowHasVideo(steps.map((s) => cell(index, s)))"
            class="rowplay"
            :class="{ on: syncing === `p${index}` }"
            :title="syncing === `p${index}` ? 'Stop' : 'Play the whole row together'"
            @click="toggleSync(`p${index}`, ($event.currentTarget as HTMLElement).parentElement?.nextElementSibling)"
          >
            <i :class="syncing === `p${index}` ? 'mdi mdi-stop' : 'mdi mdi-play'" />
          </button>
          <span class="ellipsis">{{ prompt }}</span>
        </div>
        <div class="mrow" :data-syncing="syncing === `p${index}` || undefined">
          <div v-if="hasInputs && compare" class="mcell input-cell">
            <img v-if="inputFor(index)" :src="inputFor(index)!" loading="lazy" alt="The input picture" title="The picture this prompt edits" />
          </div>
          <div v-for="s in steps" :key="s" class="mcell">
            <button v-if="cell(index, s)" @click="openGrid(index, s)">
              <video
                v-if="isVideo(cell(index, s)!)"
                :src="viewUrl(cell(index, s)!)"
                muted
                loop
                playsinline
                preload="metadata"
                @mouseenter="hoverPlay"
                @mouseleave="hoverPause"
              />
              <img v-else :src="viewUrl(cell(index, s)!)" loading="lazy" :alt="prompt" />
            </button>
            <SampleBadges
              v-if="cell(index, s)"
              compact
              :likeness="likenessFor(cell(index, s))"
              :ceiling="likeCeil"
              :skipped="isLikenessSkipped(index)"
            />
          </div>
        </div>
      </template>
    </div>
  </ScrollArea>
</template>

<style scoped>
.rowplay.on {
  color: var(--zen-selected, var(--zen-accent));
}
.rowplay {
  flex: none;
  width: 20px;
  height: 20px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  padding: 0;
}
.rowplay:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.matrix-scroll {
  height: auto;
}
.matrix {
  padding-bottom: 12px;
}
.mrow {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
  width: max-content;
}
.pnum {
  display: inline-grid;
  place-items: center;
  min-width: 18px;
  height: 18px;
  margin-right: 4px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  color: var(--zen-text);
  font-family: var(--zen-mono);
  font-size: 10px;
}
.mlabel {
  position: sticky;
  left: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  width: max-content;
  max-width: min(100%, 720px);
  margin: 8px 0 6px;
  font-size: 12px;
  color: var(--zen-muted);
}
.mlabel .ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mstep.manual {
  color: var(--zen-text);
}
.mstep.manual i {
  margin-left: 3px;
  font-size: 11px;
  color: var(--zen-accent);
}
.mstep {
  width: var(--t);
  flex: none;
  font-size: 11px;
  color: var(--zen-muted);
  text-align: center;
}
.mcell {
  position: relative;
  width: var(--t);
  height: var(--t);
  flex: none;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.mcell img,
.mcell video {
  width: var(--t);
  height: var(--t);
  object-fit: contain;
  display: block;
  transition: transform 0.15s ease;
}
/* the input picture heads each row, set apart from the renders */
.input-cell {
  box-shadow: inset 0 0 0 1px var(--zen-border);
  margin-right: 6px;
}
.input-col {
  margin-right: 6px;
  color: var(--zen-muted);
}
.mcell button {
  display: block;
  padding: 0;
  border: none;
  background: none;
  cursor: zoom-in;
}
.mcell button:hover img {
  transform: scale(1.04);
}
</style>
