<script setup lang="ts">
import { ZenLotus } from '@nynxz/zenkit-ui'
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppSidebar from './components/AppSidebar.vue'
import HoverPeek from './components/HoverPeek.vue'
import LogDrawer from './components/LogDrawer.vue'
import ScrollArea from './components/ScrollArea.vue'
import { drawerHeight, drawerOpen, startLogSync } from '@/stores/logs'
import { startRunSync } from '@/stores/runs'

const KEY = 'zentrainer.sidebar.collapsed'
const read = () => {
  try {
    return localStorage.getItem(KEY) === '1'
  } catch {
    return false
  }
}
const collapsed = ref(read())
const mobileOpen = ref(false)
watch(collapsed, (v) => {
  try {
    localStorage.setItem(KEY, v ? '1' : '0')
  } catch {
    // storage blocked
  }
})

// a window this narrow (but wider than the mobile bar) shows the sidebar folded whatever was
// saved: the expanded one takes a third of an 800px window. Expanding it there lasts the session
const tightQuery = window.matchMedia('(max-width: 1100px) and (min-width: 721px)')
const tight = ref(tightQuery.matches)
const tightExpanded = ref(false)
const onTight = (e: MediaQueryListEvent) => (tight.value = e.matches)
const folded = computed(() => (tight.value ? !tightExpanded.value : collapsed.value))
function toggleFold() {
  if (tight.value) tightExpanded.value = !tightExpanded.value
  else collapsed.value = !collapsed.value
}

onMounted(() => {
  startRunSync()
  startLogSync()
  tightQuery.addEventListener('change', onTight)
})
onUnmounted(() => tightQuery.removeEventListener('change', onTight))

// the log drawer docks over the page's bottom. Scrolled to the end, the page rides up with it
// (its end stays in view above the drawer) and back down when it closes; anywhere else
// nothing moves. The .log-room spacer is the scroll room that makes this possible.
const motionMs = () => {
  const v = getComputedStyle(document.documentElement).getPropertyValue('--zen-dur').trim()
  return v ? parseFloat(v) * (v.endsWith('ms') ? 1 : 1000) : 200
}
const roomPx = ref(0)
watch(drawerOpen, async (open) => {
  const el = scroller.value?.viewport
  // scrolled down to the end: a page that fits (or sits at the top) is left alone
  const atEnd = !!el && el.scrollTop > 0 && el.scrollHeight - el.scrollTop - el.clientHeight < 4
  if (open) {
    roomPx.value = drawerHeight.value
    await nextTick()
    if (el && atEnd) scroller.value?.glideTo(() => el.scrollHeight - el.clientHeight, motionMs())
  } else if (el && el.scrollHeight - el.scrollTop - el.clientHeight < roomPx.value) {
    // looking at the room: glide back down to the content's end, then drop the room
    const ms = motionMs()
    scroller.value?.glideTo(Math.max(0, el.scrollHeight - roomPx.value - el.clientHeight), ms)
    setTimeout(() => !drawerOpen.value && (roomPx.value = 0), ms + 30)
  } else {
    roomPx.value = 0
  }
})
watch(drawerHeight, (h) => drawerOpen.value && (roomPx.value = h))

// a page's scroll, kept by its address: Back and Forward land where it was, a page opened anew
// starts at the top. The page fills in as its data comes, so the restore keeps at it for a moment
// until there's room to scroll that far. Then re-observe the new content for the scrollbar.
const route = useRoute()
const scroller = ref<InstanceType<typeof ScrollArea> | null>(null)
const scrolls = new Map<string, number>()
// the browser's Back or Forward: the router's history says so before it navigates (a popstate
// listener of our own would hear it only after the navigation, microtasks running between listeners)
let popped = false
const router = useRouter()
const stopListening = router.options.history.listen(() => (popped = true))
// a pop that doesn't land on another page (a leave guard cancels it, or only the query changes)
// mustn't make the next ordinary link restore an old scroll
const stopAfter = router.afterEach((to, from, failure) => {
  if (failure || to.path === from.path) popped = false
})
onUnmounted(() => (stopListening(), stopAfter()))
function scrollTo(top: number) {
  let frames = 0
  // the user scrolling (wheel, touch, keys) ends it: the restore doesn't fight them
  let stopped = false
  const stop = () => (stopped = true)
  const opts = { once: true, passive: true, capture: true } as const
  for (const ev of ['wheel', 'touchstart', 'keydown', 'pointerdown'] as const) addEventListener(ev, stop, opts)
  const step = () => {
    const el = scroller.value?.viewport
    if (!el || stopped) return
    el.scrollTop = top
    if (Math.abs(el.scrollTop - top) > 2 && frames++ < 90) requestAnimationFrame(step)
  }
  step()
}
watch(
  () => route.path,
  async (to, from) => {
    const el = scroller.value?.viewport
    if (el && from) {
      // still the old page: the watch runs before it's swapped. The 200 latest pages, the oldest go
      scrolls.delete(from)
      scrolls.set(from, el.scrollTop)
      if (scrolls.size > 200) scrolls.delete(scrolls.keys().next().value!)
    }
    const top = popped ? (scrolls.get(to) ?? 0) : 0
    popped = false
    await nextTick()
    scrollTo(top)
    scroller.value?.refresh()
  },
)
</script>

<template>
  <div class="shell">
    <AppSidebar
      :collapsed="folded"
      :mobile-open="mobileOpen"
      @toggle="toggleFold"
      @navigate="mobileOpen = false"
    />
    <div v-if="mobileOpen" class="scrim" @click="mobileOpen = false" />
    <div class="main">
      <header class="mobile-bar">
        <button class="menu" aria-label="Open menu" @click="mobileOpen = true"><i class="mdi mdi-menu" /></button>
        <ZenLotus class="lotus" /> <b class="zt-title">ZenTrainer</b>
      </header>
      <ScrollArea ref="scroller" class="content">
        <!-- a page marked remount (an app) starts fresh when its address changes, not reused -->
        <main><RouterView v-slot="{ Component, route: r }"><component :is="Component" :key="r.meta.remount ? r.path : undefined" /></RouterView></main>
        <!-- room to scroll the page's end up above the open log; below the content, so nothing visible moves -->
        <div class="log-room" :style="{ height: `${roomPx}px` }" aria-hidden="true" />
      </ScrollArea>
      <LogDrawer />
      <HoverPeek />
    </div>
  </div>
</template>

<style scoped>
.shell {
  display: flex;
  height: 100%;
}
.main {
  position: relative; /* the log drawer docks over its bottom */
  overflow: clip; /* its slide-in starts below the edge: without this the page grows a native scrollbar */
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}
.content {
  flex: 1;
}
.mobile-bar {
  display: none;
}
.scrim {
  display: none;
}
@media (max-width: 720px) {
  .mobile-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 48px;
    padding: 0 10px;
    border-bottom: 1px solid var(--zen-border);
    background: var(--zen-surface);
  }
  .menu {
    width: 32px;
    height: 32px;
    border: none;
    background: none;
    color: var(--zen-text);
    font-size: 20px;
    cursor: pointer;
  }
  .lotus {
    width: 20px;
    height: 20px;
  }
  .scrim {
    display: block;
    position: fixed;
    inset: 0;
    z-index: 40;
    background: rgb(0 0 0 / 0.4);
  }
}
</style>
