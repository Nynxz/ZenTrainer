<script setup lang="ts">
// Latest: one tile per sample prompt with its newest render, filling the card (2-4 columns). An
// edit's tile wipes between its input picture and the render.
import { ZenCompare } from '@nynxz/zenkit-ui'
import { computed } from 'vue'
import { viewUrl } from '@/api/engine'
import { useRunContext } from './context'
import { isVideo, useSamplesContext } from './samples'

const { detail, sampling } = useRunContext()
const { promptCount, latestFor, promptText, liveIndex, openGrid, hoverPlay, hoverPause, inputFor, compare } = useSamplesContext()
// an edit's render beside its input: a wipe, with the viewer a button away
const wipe = (i: number) => compare.value && !!inputFor(i) && !!latestFor(i) && !isVideo(latestFor(i)!)

const latestCols = computed(() => Math.min(4, Math.max(2, promptCount.value)))
// tiles take the samples' own shape, so neither stills nor videos are cropped
const sampleAspect = computed(() => {
  const sp = detail.value?.spec.samples
  return sp?.width && sp?.height ? `${sp.width} / ${sp.height}` : '1'
})
const openLatest = (index: number) => openGrid(index, latestFor(index)!.step)
</script>

<template>
  <div class="tiles" :style="{ '--cols': latestCols, '--ar': sampleAspect }">
    <div v-for="i in promptCount" :key="i" class="tile-wrap">
      <div class="tile" :class="{ live: liveIndex === i - 1 }">
        <template v-if="liveIndex === i - 1">
          <img v-if="sampling?.preview" :src="sampling.preview" alt="" />
          <div class="tile-bar"><div :style="{ width: `${sampling?.s_total ? ((sampling.s_step ?? 0) / sampling.s_total) * 100 : 0}%` }" /></div>
          <span class="tile-badge mono">{{ sampling?.s_step }}/{{ sampling?.s_total }}</span>
        </template>
        <template v-else-if="wipe(i - 1)">
          <div class="tile-wipe"><ZenCompare :before="inputFor(i - 1)!" :after="viewUrl(latestFor(i - 1)!)" before-label="Input" :after-label="latestFor(i - 1)!.step === 0 ? 'Baseline' : `Step ${latestFor(i - 1)!.step}`" /></div>
          <button class="tile-open" title="Open in the viewer" @click="openLatest(i - 1)"><i class="mdi mdi-arrow-expand" /></button>
        </template>
        <button v-else-if="latestFor(i - 1)" class="tile-btn" :class="{ stale: sampling && sampling.phase === 'sampling' && (sampling.index ?? 0) < i - 1 }" @click="openLatest(i - 1)">
          <video
            v-if="isVideo(latestFor(i - 1)!)"
            :src="viewUrl(latestFor(i - 1)!)"
            muted
            loop
            playsinline
            preload="metadata"
            @mouseenter="hoverPlay"
            @mouseleave="hoverPause"
          />
          <i v-if="isVideo(latestFor(i - 1)!)" class="mdi mdi-play play-badge" />
          <img v-else :src="viewUrl(latestFor(i - 1)!)" alt="" />
        </button>
        <div v-else class="tile-empty"><i class="mdi mdi-image-outline" /></div>
        <span class="tile-num mono">{{ i }}</span>
      </div>
      <div class="tile-cap" :title="promptText(i - 1)">{{ promptText(i - 1) }}</div>
    </div>
  </div>
</template>

<style scoped>
/* tile size follows the column width only, never what's inside: no reflow when renders land */
.tiles {
  display: grid;
  grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr));
  gap: 12px;
}
.tile-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.tile {
  position: relative;
  aspect-ratio: var(--ar, 1);
  border-radius: var(--r-lg);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.tile.live {
  box-shadow: 0 0 0 2px var(--zen-accent);
}
.tile img,
.tile video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.play-badge {
  position: absolute;
  right: 8px;
  bottom: 8px;
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  font-size: 14px;
  pointer-events: none;
  transition: opacity var(--zen-dur-fast);
}
.tile-btn:hover .play-badge {
  opacity: 0;
}
.tile-btn {
  position: absolute;
  inset: 0;
  padding: 0;
  border: none;
  background: none;
  cursor: zoom-in;
}
.tile-btn.stale img {
  opacity: 0.45;
}
.tile-wipe {
  position: absolute;
  inset: 0;
}
.tile-wipe :deep(.zen-compare) {
  width: 100%;
  height: 100%;
}
/* the whole picture, not ZenCompare's cover crop: an edit comes out its input's shape, not the
   tile's, and both letterbox alike so they still line up. UPSTREAM: a fit prop on ZenCompare */
.tile-wipe :deep(.zen-compare img) {
  object-fit: contain;
}
.tile-open {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
  cursor: zoom-in;
}
.tile-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-size: 26px;
  color: var(--zen-muted);
}
.tile-bar {
  position: absolute;
  left: 8px;
  right: 8px;
  bottom: 8px;
  height: 4px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.35);
  overflow: hidden;
}
.tile-bar div {
  height: 100%;
  background: var(--zen-accent);
}
.tile-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
}
.tile-cap {
  font-size: 11px;
  line-height: 1.4;
  color: var(--zen-muted);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 4.2em; /* three lines, so tiles line up; the whole prompt is its tooltip */
}
.tile-num {
  position: absolute;
  top: 6px;
  left: 6px;
  min-width: 18px;
  height: 18px;
  display: grid;
  place-items: center;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
  font-size: 10px;
  pointer-events: none;
}
</style>
