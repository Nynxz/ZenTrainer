<script setup lang="ts">
// Scroll container with an overlay scrollbar: the native bar is hidden (so content never
// shifts when it starts or stops overflowing) and a themed thumb floats over the edge.
// Scrolling itself stays native: wheel, touch, keyboard, momentum, scroll-into-view.
// `axis="x"` scrolls sideways instead, with the thumb along the bottom edge.
// `endInset` (px) is how much of the far end something lies over (the app's log drawer): the
// track stops short of it, and the thumb measures the part still seen against the content without
// the scroll room the caller added below it for the same reason.
// UPSTREAM: an `overlay` mode for ZenKit's ZenScroll.
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = withDefaults(defineProps<{ axis?: 'x' | 'y'; endInset?: number }>(), { axis: 'y', endInset: 0 })
const X = props.axis === 'x'
// the scroll metrics along the chosen axis, less what endInset covers
const pos = (el: HTMLElement) => (X ? el.scrollLeft : el.scrollTop)
const full = (el: HTMLElement) => (X ? el.scrollWidth : el.scrollHeight) - props.endInset
const span = (el: HTMLElement) => (X ? el.clientWidth : el.clientHeight) - props.endInset

const viewport = ref<HTMLElement | null>(null)
const thumbTop = ref(0)
const thumbSize = ref(0)
const scrollable = ref(false)
const active = ref(false) // recently scrolled or dragging: show the bar
let idle: ReturnType<typeof setTimeout> | undefined
let ro: ResizeObserver | undefined

const MIN_THUMB = 28

function measure() {
  const el = viewport.value
  if (!el) return
  const [scrollTop, scrollHeight, clientHeight] = [pos(el), full(el), span(el)]
  // a tall log can leave only a sliver of track: no bar then, and a thumb never over half of it
  const track = clientHeight - 8
  scrollable.value = scrollHeight > clientHeight + 1 && track >= 16
  const size = Math.max(Math.min(MIN_THUMB, track / 2), (clientHeight / scrollHeight) * track)
  thumbSize.value = size
  const max = scrollHeight - clientHeight
  thumbTop.value = 4 + (max > 0 ? (scrollTop / max) * (track - size) : 0)
}

function wake() {
  active.value = true
  clearTimeout(idle)
  idle = setTimeout(() => (active.value = dragging ? true : false), 900)
}

let quiet = false // the next scroll event is ours (scrollToEnd), not the user's
let quietUntil = 0 // ...or every scroll event until then (an animated scroll of ours)
function onScroll() {
  measure()
  if (quiet) quiet = false
  else if (performance.now() >= quietUntil) wake()
}

// drag the thumb
let dragging = false
let startY = 0 // pointer position along the axis
let startTop = 0
function onThumbDown(e: PointerEvent) {
  const el = viewport.value
  if (!el) return
  dragging = true
  startY = X ? e.clientX : e.clientY
  startTop = pos(el)
  ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  e.preventDefault()
}
function onThumbMove(e: PointerEvent) {
  const el = viewport.value
  if (!dragging || !el) return
  const track = span(el) - 8 - thumbSize.value
  const ratio = (full(el) - span(el)) / Math.max(1, track)
  const to = startTop + ((X ? e.clientX : e.clientY) - startY) * ratio
  if (X) el.scrollLeft = to
  else el.scrollTop = to
}
function onThumbUp() {
  dragging = false
  wake()
}
// click the track: page toward the click
function onTrackDown(e: PointerEvent) {
  const el = viewport.value
  if (!el || e.target !== e.currentTarget) return
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
  const dir = (X ? e.clientX - rect.left : e.clientY - rect.top) < thumbTop.value ? -1 : 1
  const by = dir * span(el) * 0.9
  el.scrollBy({ [X ? 'left' : 'top']: by, behavior: 'smooth' })
}

watch(() => props.endInset, () => nextTick(measure))

onMounted(async () => {
  await nextTick()
  measure()
  ro = new ResizeObserver(measure)
  if (viewport.value) {
    ro.observe(viewport.value)
    // content size changes (route changes, images loading) don't resize the viewport
    for (const child of Array.from(viewport.value.children)) ro.observe(child)
  }
})
onBeforeUnmount(() => {
  ro?.disconnect()
  clearTimeout(idle)
})

defineExpose({
  viewport,
  /** scroll to `to` over `ms` (eased) without flashing the bar; `to` may be a function, re-read
   *  each frame, for a target that moves while it runs (content growing) */
  glideTo(to: number | (() => number), ms: number) {
    const el = viewport.value
    if (!el) return
    const target = () => (typeof to === 'function' ? to() : to)
    const from = pos(el)
    const start = performance.now()
    quietUntil = start + ms + 50
    const set = (v: number) => (X ? (el.scrollLeft = v) : (el.scrollTop = v))
    if (ms <= 0) return set(target())
    const step = (now: number) => {
      const k = Math.min(1, (now - start) / ms)
      const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2
      set(from + (target() - from) * e)
      if (k < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  },
  /** jump to the end without flashing the bar (following a log, not the user scrolling) */
  scrollToEnd() {
    const el = viewport.value
    if (!el) return
    const to = full(el) - span(el)
    if (Math.abs(pos(el) - to) < 1) return
    quiet = true
    if (X) el.scrollLeft = to
    else el.scrollTop = to
  },
  /** re-observe content after the slot's root element changes (e.g. a route swap) */
  refresh() {
    if (!ro || !viewport.value) return
    ro.disconnect()
    ro.observe(viewport.value)
    for (const child of Array.from(viewport.value.children)) ro.observe(child)
    measure()
  },
})
</script>

<template>
  <div class="scroll-area" :class="{ active, scrollable, x: X }">
    <div ref="viewport" class="viewport" @scroll.passive="onScroll"><slot /></div>
    <div v-show="scrollable" class="track" :style="endInset ? { [X ? 'right' : 'bottom']: `${endInset}px` } : undefined" @pointerdown="onTrackDown">
      <div
        class="thumb"
        :style="X ? { width: `${thumbSize}px`, transform: `translateX(${thumbTop}px)` } : { height: `${thumbSize}px`, transform: `translateY(${thumbTop}px)` }"
        @pointerdown.stop="onThumbDown"
        @pointermove="onThumbMove"
        @pointerup="onThumbUp"
        @pointercancel="onThumbUp"
      />
    </div>
  </div>
</template>

<style scoped>
.scroll-area {
  position: relative;
  min-height: 0;
  height: 100%;
  overflow: hidden;
}
.viewport {
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  scrollbar-width: none; /* the overlay replaces it; no gutter, no shift */
}
.viewport::-webkit-scrollbar {
  display: none;
}
.track {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 12px;
  z-index: 4;
  opacity: 0;
  transition: opacity 0.2s ease;
}
.scroll-area:hover .track,
.scroll-area.active .track {
  opacity: 1;
}
.thumb {
  position: absolute;
  top: 0;
  right: 3px;
  width: 6px;
  border-radius: var(--r-sm);
  background: var(--zen-scrollbar, color-mix(in srgb, var(--zen-text) 28%, transparent));
  transition:
    width 0.15s ease,
    right 0.15s ease,
    background 0.15s ease;
  touch-action: none;
  cursor: pointer;
}
.track:hover .thumb,
.thumb:active {
  width: 8px;
  right: 2px;
  background: var(--zen-accent);
}
.x .viewport {
  height: auto;
  overflow-x: auto;
  overflow-y: hidden;
}
.x .track {
  top: auto;
  left: 0;
  width: auto;
  height: 12px;
}
.x .thumb {
  top: auto;
  right: auto;
  left: 0;
  bottom: 3px;
  width: auto;
  height: 6px;
}
.x .track:hover .thumb,
.x .thumb:active {
  width: auto;
  right: auto;
  height: 8px;
  bottom: 2px;
}
</style>
