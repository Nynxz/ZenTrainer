<script setup lang="ts">
// The dataset as a grid of uniform cards (zoomable): the picture with its badges (mask, video,
// repeats, image and face issues, no caption), mute and select on hover, the caption below.
import { ZenTip } from '@nynxz/zenkit-ui'
import { onMounted, onUnmounted, ref } from 'vue'
import { fileInfo, pct, secs, useDatasetContext } from './context'
import MoreRows from './MoreRows.vue'
import { zoomOf } from '@/lib/zoom'

const { zoom, inputs, fit, showMasks, activeSet, cap, selected, isMuted, setMuted, imageFlags, faceOf, faceIssue, faceNote, shown, visible, more, toggleSelect, onItemClick, onItemMenu, full, thumb, thumbSize, boxing, maskHighVis } = useDatasetContext()

// drag to select a box of cards (mouse and pen; touch scrolls), starting on a card or in the empty
// space around them: the page's margins, gaps and below the last row. A plain drag starts a new
// selection, Ctrl/Cmd/Shift adds to it; a press that barely moves stays a click. Near the top or
// bottom of the page it scrolls, and the box follows.
const grid = ref<HTMLElement | null>(null)
const box = ref<{ x: number; y: number; w: number; h: number } | null>(null)
let press: { x: number; y: number; cx: number; cy: number; add: boolean; base: Set<string> } | null = null
let pointer = { x: 0, y: 0 }
let scroller: HTMLElement | null = null
let raf = 0
let swallowClick = false

// anywhere on the page below the toolbar: a card's picture or caption, the gaps, the margins, the
// strip between the toolbar and the first row. Not a control (a card's own buttons, links, fields),
// the side panel or a banner, nor the toolbar or anything above it.
function startsBox(e: MouseEvent) {
  const t = e.target as HTMLElement
  if (e.button !== 0 || !grid.value) return false
  if (grid.value.contains(t)) return !t.closest('.check, .mute-btn, .info-btn')
  const bar = grid.value.closest('.page')?.querySelector('.sticky-bar')
  if (!bar || e.clientY < bar.getBoundingClientRect().bottom) return false
  return !t.closest('button, a, input, textarea, select, label, [contenteditable], [role], .ds-side, .drawer-scrim, .reviewing, .notice, .sel-bar')
}
// in the grid's own coordinates, which scrolling doesn't change (and its own px: the page may be zoomed)
function local(cx: number, cy: number) {
  const g = grid.value!.getBoundingClientRect()
  const z = zoomOf(grid.value)
  return { x: (cx - g.left) / z, y: (cy - g.top) / z, g, z }
}
function onDown(e: PointerEvent) {
  if (e.pointerType === 'touch' || !startsBox(e)) return
  const p = local(e.clientX, e.clientY)
  press = { x: p.x, y: p.y, cx: e.clientX, cy: e.clientY, add: e.ctrlKey || e.metaKey || e.shiftKey, base: new Set(selected) }
  pointer = { x: e.clientX, y: e.clientY }
  addEventListener('pointermove', onMove)
  addEventListener('pointerup', onUp)
  addEventListener('pointercancel', onUp)
}
// the browser's own drag of a picture, or of selected text, would cancel ours
function onMouseDown(e: MouseEvent) {
  if (startsBox(e)) e.preventDefault()
}
function onMove(e: PointerEvent) {
  pointer = { x: e.clientX, y: e.clientY }
  if (!press) return
  if (!box.value) {
    if (Math.hypot(e.clientX - press.cx, e.clientY - press.cy) < 6) return
    getSelection()?.removeAllRanges()
    boxing.value = true
    raf = requestAnimationFrame(edgeScroll)
  }
  update()
}
// the part of the page that shows: below the sticky toolbar, which the box mustn't run under
function view() {
  const v = scroller!.getBoundingClientRect()
  const bar = grid.value!.closest('.page')?.querySelector('.sticky-bar')
  return { left: v.left, right: v.right, top: Math.max(v.top, bar?.getBoundingClientRect().bottom ?? v.top), bottom: v.bottom }
}
function update() {
  if (!press || !grid.value || !scroller) return
  const v = view()
  const p = local(Math.min(Math.max(pointer.x, v.left), v.right), Math.min(Math.max(pointer.y, v.top), v.bottom))
  const b = { x: Math.min(p.x, press.x), y: Math.min(p.y, press.y), w: Math.abs(p.x - press.x), h: Math.abs(p.y - press.y) }
  box.value = b
  const next = press.add ? new Set(press.base) : new Set<string>()
  for (const el of grid.value.querySelectorAll<HTMLElement>('.card-item')) {
    const r = el.getBoundingClientRect()
    const [l, t, w, h] = [(r.left - p.g.left) / p.z, (r.top - p.g.top) / p.z, r.width / p.z, r.height / p.z]
    if (l < b.x + b.w && l + w > b.x && t < b.y + b.h && t + h > b.y) next.add(el.dataset.path!)
  }
  for (const k of [...selected]) if (!next.has(k)) selected.delete(k)
  for (const k of next) selected.add(k)
}
function edgeScroll() {
  if (!press || !scroller) return
  const v = view()
  const edge = 48
  const dy = pointer.y < v.top + edge ? pointer.y - (v.top + edge) : pointer.y > v.bottom - edge ? pointer.y - (v.bottom - edge) : 0
  if (dy) {
    scroller.scrollTop += dy * 0.4
    update()
  }
  raf = requestAnimationFrame(edgeScroll)
}
function onUp() {
  removeEventListener('pointermove', onMove)
  removeEventListener('pointerup', onUp)
  removeEventListener('pointercancel', onUp)
  cancelAnimationFrame(raf)
  if (box.value) {
    // the release would otherwise click the card under it open; no click comes when it ends elsewhere
    swallowClick = true
    setTimeout(() => (swallowClick = false))
  }
  press = null
  box.value = null
  boxing.value = false
}
function onClickCapture(e: MouseEvent) {
  if (!swallowClick) return
  swallowClick = false
  e.stopPropagation()
  e.preventDefault()
}
// listened for on the whole scrolling page, so a box can start beside the grid too
onMounted(() => {
  scroller = grid.value?.closest('.viewport') ?? null
  scroller?.addEventListener('pointerdown', onDown)
  scroller?.addEventListener('mousedown', onMouseDown)
})
onUnmounted(() => {
  onUp()
  scroller?.removeEventListener('pointerdown', onDown)
  scroller?.removeEventListener('mousedown', onMouseDown)
})
</script>

<template>
  <div
    ref="grid"
    class="grid"
    :class="{ boxing: box }"
    :style="{ '--card': `${zoom}px` }"
    @click.capture="onClickCapture"
  >
    <div v-if="box" class="marquee" :style="{ left: `${box.x}px`, top: `${box.y}px`, width: `${box.w}px`, height: `${box.h}px` }" />
    <article
      v-for="(it, i) in visible"
      :key="it.path"
      :data-path="it.path"
      class="card-item"
      :class="{ sel: selected.has(it.path), muted: isMuted(it) }"
      @contextmenu="onItemMenu(it, i, $event)"
    >
      <button class="thumb" :title="it.path" :data-peek="full(it)" @click="onItemClick(it, i, $event)">
        <img :src="thumb(it, thumbSize)" loading="lazy" :alt="cap(it)" :class="{ 'mask-dim': showMasks && it.mask && maskHighVis }" :style="{ objectFit: fit }" />
        <!-- the mask view: what the mask leaves out is a purple/black checker (Source's missing
             texture: unmistakable on any picture), and only what it trains shows -->
        <span v-if="showMasks && it.mask" class="mask-out" :class="{ hi: maskHighVis }" />
        <img
          v-if="showMasks && it.mask"
          :src="thumb(it, thumbSize)"
          loading="lazy"
          alt=""
          class="mask-lit"
          :style="{ objectFit: fit, maskImage: `url(${thumb({ path: it.mask }, thumbSize)})`, WebkitMaskImage: `url(${thumb({ path: it.mask }, thumbSize)})`, maskSize: fit, WebkitMaskSize: fit }"
        />
        <!-- an edit pair: its input picture in the corner (the big one is the target) -->
        <img v-if="inputs(it).length" :src="thumb({ path: inputs(it)[0]! }, 256)" loading="lazy" alt="" class="control-inset" :title="`Input: ${inputs(it).join(', ')}`" />
        <span class="badges">
          <span v-if="it.mask" class="badge" :class="{ warn: it.mask_mismatch }" :title="it.mask_mismatch ? 'This mask is a different shape from its image: it won\'t line up' : `Masked: ${pct(it.mask_coverage)} trains at full weight`">
            <i :class="it.mask_mismatch ? 'mdi mdi-alert-outline' : 'mdi mdi-layers-outline'" /> {{ pct(it.mask_coverage) }}
          </span>
          <span v-if="it.kind === 'video'" class="badge" :title="`Video: ${it.frames ?? '?'} frames at ${it.fps ?? '?'} fps`"><i class="mdi mdi-play" /> {{ secs(it.duration) }}</span>
          <span v-if="it.repeats > 1" class="badge" :title="`Repeats ${it.repeats}×`">×{{ it.repeats }}</span>
          <span v-if="imageFlags[it.path]?.length" class="badge warn" :title="imageFlags[it.path]!.join(', ')"><i class="mdi mdi-image-filter-center-focus-weak" /> {{ imageFlags[it.path]!.length }}</span>
          <span v-if="faceOf(it)" class="badge" :class="{ warn: !!faceIssue(it) }" :title="faceIssue(it) ?? faceNote(it)">
            <i :class="faceOf(it)!.faces === 0 ? 'mdi mdi-face-man-outline' : 'mdi mdi-face-recognition'" />
            {{ faceOf(it)!.faces === 0 ? '0' : faceOf(it)!.likeness != null ? faceOf(it)!.likeness!.toFixed(2) : faceOf(it)!.faces }}
          </span>
          <span v-if="!cap(it)" class="badge warn" title="No caption"><i class="mdi mdi-text-box-remove-outline" /></span>
        </span>
      </button>
      <span class="info-btn" @click.stop><ZenTip :text="fileInfo(it)" label="File details" /></span>
      <button
        class="mute-btn"
        :class="{ on: isMuted(it) }"
        :title="isMuted(it) ? `Muted in “${activeSet}”: click to train on it again` : `Mute in “${activeSet}”: kept, but not trained on`"
        @click.stop="setMuted([it.path], !isMuted(it))"
      >
        <i :class="isMuted(it) ? 'mdi mdi-eye-off' : 'mdi mdi-eye-outline'" />
      </button>
      <label class="check" :class="{ on: selected.has(it.path) }" @click.prevent="toggleSelect(it, i, $event)">
        <i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" />
      </label>
      <button class="card-body" :title="cap(it) || 'No caption'" @click="onItemClick(it, i, $event)">
        <span class="cap-text" :class="{ empty: !cap(it) }">{{ cap(it) || 'No caption' }}</span>
      </button>
    </article>
    <MoreRows v-if="visible.length < shown.length" @more="more" />
  </div>
</template>

<style scoped src="./items.css"></style>
<style scoped>
/* grid: the slider sets the smallest card; columns stretch to fill the row evenly */
.grid {
  position: relative;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(var(--card), 1fr));
  gap: 12px;
  align-items: start;
}
.grid.boxing {
  user-select: none;
}
/* the drag-select box: the primary colour, rounded like the cards' inner controls */
.marquee {
  position: absolute;
  z-index: 3;
  pointer-events: none;
  border: 1px solid var(--zen-accent);
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-accent) 16%, transparent);
}
.card-item {
  transform-origin: top left;
  position: relative;
  display: flex;
  flex-direction: column;
  border-radius: var(--r-xl);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 60%, transparent);
  transition:
    border-color 0.15s ease,
    box-shadow 0.15s ease;
}
/* hover in the primary colour, softer than a selection's full accent ring */
.card-item:hover {
  border-color: color-mix(in srgb, var(--zen-accent) 60%, transparent);
}
/* selected: a 2px border laid exactly over the card's outline (its own border and a pixel in), over
   the picture. Nothing reaches past the card, which scrolled under the sticky toolbar (lined up with
   the outer columns) would show beside it */
.card-item.sel {
  border-color: var(--zen-selected, var(--zen-accent));
}
.card-item.sel::after {
  content: '';
  position: absolute;
  inset: -1px;
  z-index: 3;
  border: 2px solid var(--zen-selected, var(--zen-accent));
  border-radius: var(--r-xl);
  pointer-events: none;
}
/* every card the same size: a square thumbnail (image positioned inside, so its own
   aspect can't stretch the card) over a fixed-height caption block */
.thumb {
  position: relative;
  display: block;
  width: 100%;
  aspect-ratio: 1;
  padding: 0;
  border: none;
  background: var(--zen-surface-2);
  cursor: zoom-in;
  overflow: hidden;
  /* the card's top corners: the card itself doesn't clip, so a selected card's border isn't cut */
  border-radius: calc(var(--r-xl) - 1px) calc(var(--r-xl) - 1px) 0 0;
}
.thumb img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.card-item.muted .thumb img:not(.control-inset):not(.mask-ov) {
  opacity: 0.35;
  filter: grayscale(0.6);
}
.info-btn {
  position: absolute;
  top: 8px;
  right: 40px;
  z-index: 2;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border-radius: var(--r-sm);
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.info-btn :deep(*) {
  color: inherit;
}
.card-item:hover .info-btn,
.info-btn:focus-within {
  opacity: 1;
}
.mute-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 2;
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.card-item:hover .mute-btn,
.mute-btn.on,
.mute-btn:focus-visible {
  opacity: 1;
}
.mute-btn.on {
  color: var(--zen-warn);
}
.thumb .control-inset {
  position: absolute;
  left: 6px;
  top: 6px;
  width: 34%;
  height: auto;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--r-sm);
  border: 2px solid var(--zen-surface);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
}
.badges {
  position: absolute;
  right: 6px;
  bottom: 6px;
  display: flex;
  gap: 4px;
}
.badge {
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
}
.badge.warn {
  color: #fbbf24;
}
.card-item:hover .check,
.grid:has(.check.on) .check,
.grid.boxing .check {
  opacity: 1;
}
.card-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  height: 52px;
  flex: none;
  padding: 9px 10px 8px;
  border: none;
  background: none;
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.cap-text {
  font-size: 12px;
  line-height: 1.4;
  height: 2.8em; /* exactly two lines; the clamp lives here, padding stays outside */
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
.cap-text.empty {
  color: var(--zen-warn);
}
/* the mask view: what the mask leaves out faded into the theme's background (as the mask
   editor does it), or with High-vis Source's purple/black checker over a hidden picture */
.thumb img.mask-dim {
  opacity: 0;
}
.thumb .mask-out {
  position: absolute;
  inset: 0;
  background: var(--zen-bg);
  opacity: 0.84; /* the editor's 215/255 */
  pointer-events: none;
}
.thumb .mask-out.hi {
  background: repeating-conic-gradient(#f0f 0 25%, #000 0 50%) 0 0 / 16px 16px;
  opacity: 1;
}
.thumb img.mask-lit {
  mask-mode: luminance;
  mask-repeat: no-repeat;
  mask-position: center;
  pointer-events: none;
}
</style>
