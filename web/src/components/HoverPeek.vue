<script setup lang="ts">
// Hold Shift over any picture or video (sample tiles, dataset items, cards) for a large copy of
// it in the middle of the screen, videos playing with sound at a quarter volume (M while peeking
// mutes or unmutes, remembered). An element can name what to show with data-peek="url" (a still that
// stands for a clip, a thumb for a full-size file). A picture that changes while it's peeked (a
// sample's live preview, a frame each step) is followed, and a small one is scaled up to the screen.
// A dataset thumbnail peeks as its biggest one. Not where the picture is large already (the
// lightbox, the mask editor's canvas), and not for a Shift-click: that selects a range.
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const src = ref<string | null>(null)
const video = ref(false)
const time = ref(0)
const SOUND = 'zentrainer.peek.sound'
const sound = ref((() => {
  try {
    return localStorage.getItem(SOUND) !== '0'
  } catch {
    return true
  }
})())
function toggleSound() {
  sound.value = !sound.value
  try {
    localStorage.setItem(SOUND, sound.value ? '1' : '0')
  } catch {
    // storage blocked
  }
}
let shift = false
// a click while Shift is down (selecting a range): no peek until Shift is let go
let clicked = false
let x = 0
let y = 0
// the picture under the pointer while peeking: told when its src changes (a live preview's next
// frame) or when it's swapped out of the page (the finished sample replacing it)
let watched: HTMLImageElement | null = null
const onChange = new MutationObserver(() => {
  if (!watched?.isConnected) update()
  else if (watched.currentSrc && watched.currentSrc !== src.value) src.value = watched.currentSrc
})
// the picture's own shape, to fill the screen with it whatever its pixel size
const nat = ref<{ w: number; h: number } | null>(null)
const fit = computed(() => (nat.value ? { width: `min(86vw, calc(86vh * ${nat.value.w} / ${nat.value.h}))`, height: 'auto' } : {}))
function sized(e: Event) {
  const im = e.target as HTMLImageElement
  if (im.naturalWidth && im.naturalHeight) nat.value = { w: im.naturalWidth, h: im.naturalHeight }
}

const VIDEO = /\.(mp4|webm|mov|mkv)([?&#]|$)/i  // also /files/view?filename=x.mp4&subfolder=...
const LARGE = '.zlb, .mask-page .stage, [data-no-peek]'
// a dataset thumbnail (…/thumb?path=…&size=160): the biggest the server makes
function biggest(url: string) {
  if (!url.includes('/zentrainer/api/datasets/') || !url.includes('/thumb?')) return url
  const u = new URL(url, location.href)
  u.searchParams.set('size', '1024')
  return u.pathname + u.search
}

function under(): { url: string; video: boolean; time: number; img?: HTMLImageElement } | null {
  for (const el of document.elementsFromPoint(x, y)) {
    if (el.closest('.peek')) continue
    if (el.closest(LARGE)) return null
    const tagged = (el as HTMLElement).closest<HTMLElement>('[data-peek]')
    if (tagged?.dataset.peek) return { url: tagged.dataset.peek, video: VIDEO.test(tagged.dataset.peek), time: 0 }
    if (el instanceof HTMLVideoElement && el.currentSrc) return { url: el.currentSrc, video: true, time: el.currentTime }
    if (el instanceof HTMLImageElement && el.currentSrc && el.naturalWidth > 1) {
      const url = biggest(el.currentSrc)
      return { url, video: false, time: 0, img: url === el.currentSrc ? el : undefined }
    }
  }
  return null
}

function update() {
  const hit = shift && !clicked ? under() : null
  if (hit?.img !== watched) {
    onChange.disconnect()
    watched = hit?.img ?? null
    if (watched) {
      onChange.observe(watched, { attributes: true, attributeFilter: ['src', 'srcset'] })
      if (watched.parentElement) onChange.observe(watched.parentElement, { childList: true })
    }
  }
  if (hit?.url === src.value) return
  if (!hit || hit.video) nat.value = null
  src.value = hit?.url ?? null
  video.value = !!hit?.video
  time.value = hit?.time ?? 0
}

function onMove(e: PointerEvent) {
  x = e.clientX
  y = e.clientY
  shift = e.shiftKey
  if (!shift) clicked = false
  if (shift || src.value) update()
}
function onDown(e: PointerEvent) {
  if (!e.shiftKey) return
  clicked = true
  update()
}
function onKey(e: KeyboardEvent) {
  if (e.type === 'keydown' && src.value && video.value && (e.key === 'm' || e.key === 'M')) {
    e.preventDefault()
    toggleSound()
    return
  }
  if (e.key !== 'Shift') return
  const t = e.target as HTMLElement | null
  if (e.type === 'keydown' && t?.closest('input, textarea, [contenteditable="true"]')) return // typing capitals
  shift = e.type === 'keydown'
  if (!shift) clicked = false
  update()
}
function off() {
  shift = false
  update()
}
function start(e: Event) {
  const v = e.target as HTMLVideoElement
  if (time.value) v.currentTime = time.value // carry on from where the tile was
}

onMounted(() => {
  window.addEventListener('pointermove', onMove, { passive: true })
  window.addEventListener('pointerdown', onDown, { capture: true, passive: true })
  window.addEventListener('keydown', onKey)
  window.addEventListener('keyup', onKey)
  window.addEventListener('blur', off)
})
onBeforeUnmount(() => {
  onChange.disconnect()
  window.removeEventListener('pointermove', onMove)
  window.removeEventListener('pointerdown', onDown, { capture: true })
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('keyup', onKey)
  window.removeEventListener('blur', off)
})
</script>

<template>
  <Transition name="peek">
    <div v-if="src" class="peek" aria-hidden="true">
      <video v-if="video" :key="src" :src="src" autoplay loop playsinline :muted="!sound" :volume="0.25" @loadedmetadata="start" />
      <img v-else :src="src" :style="fit" alt="" @load="sized" />
      <span v-if="video" class="peek-hint"><i :class="sound ? 'mdi mdi-volume-high' : 'mdi mdi-volume-off'" /> M</span>
    </div>
  </Transition>
</template>

<style scoped>
.peek {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  pointer-events: none;
  background: color-mix(in srgb, var(--zen-bg) 55%, transparent);
  backdrop-filter: blur(3px);
}
.peek video,
.peek img {
  max-width: 86vw;
  max-height: 86vh;
  border-radius: var(--r-lg, 12px);
  border: 1px solid var(--zen-border);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
  background: #000;
  display: block;
}
.peek-hint {
  position: fixed;
  bottom: 18px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 12px;
  padding: 3px 10px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
}
.peek-enter-active,
.peek-leave-active {
  transition: opacity var(--zen-dur-fast) ease;
}
.peek-enter-from,
.peek-leave-to {
  opacity: 0;
}
</style>
