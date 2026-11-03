<script setup lang="ts">
// ComfyUI's console, folded up over the bottom of the main area: an overlay, so the page
// underneath never moves. Follows the tail until you scroll up; drag the top edge to resize.
// Ctrl+` toggles it from anywhere, Esc closes it while it has focus. A second tab charts
// VRAM and RAM with the log's model loads and errors marked on the time axis.
import { ZenInput } from '@nynxz/zenkit-ui'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { DRAWER_HEIGHT_KEY, DRAWER_MIN_H, clearLogs, drawerHeight, drawerOpen, logCounts, logLines, logPartial, toggleDrawer, type LogLevel } from '@/stores/logs'
import { RAM_SERIES, VRAM_SERIES } from '@/lib/memSeries'
import { gb, memLatest, samples, startSystemSync } from '@/stores/system'
import LoadedModels from './LoadedModels.vue'
import MemChart, { type Marker } from './MemChart.vue'
import ScrollArea from './ScrollArea.vue'
import { zoomOf } from '@/lib/zoom'

// ---- tabs (remembered per viewer) --------------------------------------------------------------
const TAB_KEY = 'zentrainer.logs.tab'
type Tab = 'log' | 'memory'
const readTab = (): Tab => {
  try {
    return localStorage.getItem(TAB_KEY) === 'memory' ? 'memory' : 'log'
  } catch {
    return 'log'
  }
}
const tab = ref<Tab>(readTab())
watch(tab, (v) => {
  try {
    localStorage.setItem(TAB_KEY, v)
  } catch {
    // storage blocked: this session keeps it
  }
})

const height = drawerHeight
const level = ref<'all' | 'warn' | 'error'>('all')
const query = ref('')
const following = ref(true)
const copied = ref(false)
const scroller = ref<InstanceType<typeof ScrollArea> | null>(null)
const root = ref<HTMLElement | null>(null)

const RANK: Record<LogLevel, number> = { info: 0, warn: 1, error: 2 }
const shown = computed(() => {
  const min = level.value === 'all' ? 0 : RANK[level.value]
  const q = query.value.trim().toLowerCase()
  return logLines.value.filter((l) => RANK[l.level] >= min && (!q || l.text.toLowerCase().includes(q)))
})
const showPartial = computed(() => level.value === 'all' && !query.value.trim() && logPartial.value.trim() !== '')

const time = (t: string) => (t ? t.slice(11, 19) : '')

// ---- memory tab ------------------------------------------------------------------------------
// Log times are ComfyUI's local wall clock without a zone (datetime.now().isoformat()); the
// browser reads a zoneless ISO date-time as local too, which matches when both run on one
// machine. Samples are epoch seconds (time.time()).
const EVENT = /Requested to load|loaded completely|loaded partially|Loading model|Unloading|Exception|Traceback|OutOfMemory/i
const markers = computed<Marker[]>(() => {
  const out: Marker[] = []
  for (const l of logLines.value) {
    if (!l.t || !(l.level === 'error' || EVENT.test(l.text))) continue
    const ms = Date.parse(l.t.slice(0, 23)) // JS reads milliseconds, ComfyUI writes micro
    if (Number.isNaN(ms)) continue
    const first = l.text.split('\n').find((x) => x.trim() && !/^Traceback/.test(x)) ?? l.text
    const error = l.level === 'error' || /Exception|Traceback|OutOfMemory/i.test(l.text)
    out.push({ t: ms / 1000, kind: error ? 'error' : 'event', label: `${time(l.t)}  ${(error && l.text.includes('\n') ? l.text.split('\n').at(-1)! : first).slice(0, 160)}` })
  }
  return out
})
// the charts fill the drawer: two across, or stacked (scrolling) when it's narrow
const memWidth = ref(1200)
let ro: ResizeObserver | undefined
const stacked = computed(() => memWidth.value < 900)
const CHROME = 46 + 16 + 30 + 16 // header, body padding, caption, x ticks
const plotH = computed(() => Math.max(70, stacked.value ? Math.max(120, (height.value - CHROME) / 2 - 20) : height.value - CHROME - 8))

// ---- follow the tail -------------------------------------------------------------------------
const viewport = () => scroller.value?.viewport ?? null
function toBottom() {
  scroller.value?.scrollToEnd()
  following.value = true
}
function onScroll() {
  const el = viewport()
  if (el) following.value = el.scrollHeight - el.scrollTop - el.clientHeight < 24
}
watch(
  () => [shown.value.length, logPartial.value, drawerOpen.value],
  async () => {
    if (!drawerOpen.value || !following.value) return
    await nextTick()
    toBottom()
  },
)
// new filter, or back from the Memory tab: start from the latest lines
watch([level, query, () => tab.value === 'log'], async () => {
  if (tab.value !== 'log') return
  await nextTick()
  toBottom()
})

// ---- resize from the top edge ------------------------------------------------------------------
let dragStart = 0
let startH = 0
function onDragDown(e: PointerEvent) {
  dragStart = e.clientY
  startH = height.value
  ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  e.preventDefault()
}
function onDragMove(e: PointerEvent) {
  if (!dragStart) return
  const max = Math.max(DRAWER_MIN_H, (root.value?.parentElement?.clientHeight ?? 800) * 0.85)
  const moved = (dragStart - e.clientY) / zoomOf(root.value) // screen px to the page's
  height.value = Math.round(Math.min(max, Math.max(DRAWER_MIN_H, startH + moved)))
}
function onDragUp() {
  if (!dragStart) return
  dragStart = 0
  try {
    localStorage.setItem(DRAWER_HEIGHT_KEY, String(height.value))
  } catch {
    // storage blocked: this session keeps it
  }
}

async function copyAll() {
  const text = shown.value.map((l) => `${time(l.t)}  ${l.text}`).join('\n')
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch {
    // clipboard blocked (insecure origin)
  }
}

// ---- keys ------------------------------------------------------------------------------------
function onKey(e: KeyboardEvent) {
  if (e.ctrlKey && !e.altKey && !e.metaKey && (e.key === '`' || e.code === 'Backquote')) {
    e.preventDefault()
    toggleDrawer()
  }
}
onMounted(() => {
  window.addEventListener('keydown', onKey)
  startSystemSync() // idempotent; the sidebar meters normally start it
  ro = new ResizeObserver(([e]) => e && (memWidth.value = e.contentRect.width))
  if (root.value) ro.observe(root.value)
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  ro?.disconnect()
})
</script>

<template>
  <Transition name="drawer">
    <section
      v-show="drawerOpen"
      ref="root"
      class="drawer"
      :style="{ height: `${height}px` }"
      role="region"
      aria-label="ComfyUI log"
      @keydown.esc="toggleDrawer(false)"
    >
      <div class="grip" title="Drag to resize" @pointerdown="onDragDown" @pointermove="onDragMove" @pointerup="onDragUp" @pointercancel="onDragUp" />
      <header class="bar" :class="{ narrow: stacked }">
        <div class="seg tabs" role="tablist">
          <button role="tab" :aria-selected="tab === 'log'" :class="{ on: tab === 'log' }" @click="tab = 'log'"><i class="mdi mdi-console-line" /> Log</button>
          <button role="tab" :aria-selected="tab === 'memory'" :class="{ on: tab === 'memory' }" @click="tab = 'memory'"><i class="mdi mdi-memory" /> Memory</button>
        </div>
        <template v-if="tab === 'log'">
        <div class="seg">
          <button :class="{ on: level === 'all' }" @click="level = 'all'">All</button>
          <button :class="{ on: level === 'warn' }" title="Warnings and errors" @click="level = 'warn'">
            <span class="wide-only">Warnings</span><i class="mdi mdi-alert-outline narrow-only" /> <span class="n mono">{{ logCounts.warn + logCounts.error }}</span>
          </button>
          <button :class="{ on: level === 'error' }" title="Errors" @click="level = 'error'">
            <span class="wide-only">Errors</span><i class="mdi mdi-alert-circle-outline narrow-only" /> <span class="n mono" :class="{ hot: logCounts.error }">{{ logCounts.error }}</span>
          </button>
        </div>
        <div class="filter"><ZenInput v-model="query" size="sm" placeholder="Filter" /></div>
        <span class="spacer" />
        <!-- fixed slot: appears without moving the buttons beside it -->
        <span class="jump-slot">
          <button v-if="!following" class="chip-btn" title="Follow new lines" @click="toBottom"><i class="mdi mdi-arrow-collapse-down" /><span class="wide-only">Latest</span></button>
        </span>
        <button class="icon-btn" :title="copied ? 'Copied' : 'Copy what is shown'" @click="copyAll">
          <i :class="copied ? 'mdi mdi-check' : 'mdi mdi-content-copy'" />
        </button>
        <button class="icon-btn" title="Clear the view (ComfyUI keeps its log)" @click="clearLogs"><i class="mdi mdi-notification-clear-all" /></button>
        </template>
        <template v-else>
          <span class="readout mono" title="Device memory in use / total">
            <span class="k">VRAM</span><span class="v">{{ gb(memLatest?.vram_used) }}</span><span class="of">/ {{ gb(memLatest?.vram_total) }} GB</span>
          </span>
          <span class="readout mono wide-only" title="PyTorch allocated">
            <span class="k">alloc</span><span class="v">{{ gb(memLatest?.torch_allocated) }}</span><span class="of">GB</span>
          </span>
          <span class="readout mono" title="System RAM in use / total">
            <span class="k">RAM</span><span class="v">{{ gb(memLatest?.ram_used) }}</span><span class="of">/ {{ gb(memLatest?.ram_total) }} GB</span>
          </span>
          <span class="spacer" />
          <span class="legend-note wide-only">
            <span class="mk-key" /> model load
            <span class="mk-key err" /> error
          </span>
        </template>
        <button class="icon-btn" title="Close (Ctrl+`)" @click="toggleDrawer(false)"><i class="mdi mdi-chevron-down" /></button>
      </header>
      <ScrollArea v-if="tab === 'memory'" class="body">
        <div class="mem" :class="{ stacked }">
          <MemChart title="VRAM (GB)" :samples="samples" :series="VRAM_SERIES" :total="memLatest?.vram_total" :height="plotH" :markers="markers" />
          <MemChart title="RAM (GB)" :samples="samples" :series="RAM_SERIES" :total="memLatest?.ram_total" :height="plotH" :markers="markers" />
        </div>
        <LoadedModels class="loaded" />
      </ScrollArea>
      <ScrollArea v-show="tab === 'log'" ref="scroller" class="body" @scroll.capture="onScroll">
        <div class="lines mono">
          <div v-for="l in shown" :key="l.id" class="line" :class="l.level">
            <span class="t">{{ time(l.t) }}</span>
            <span class="m">{{ l.text }}</span>
          </div>
          <div v-if="showPartial" class="line partial">
            <span class="t" />
            <span class="m">{{ logPartial }}</span>
          </div>
          <div v-if="!shown.length && !showPartial" class="empty">{{ logLines.length ? 'Nothing matches.' : 'No output yet.' }}</div>
        </div>
      </ScrollArea>
    </section>
  </Transition>
</template>

<style scoped>
.drawer {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 30;
  display: flex;
  flex-direction: column;
  background: var(--zen-surface);
  border-top: 1px solid var(--zen-border);
  border-radius: var(--r-lg) var(--r-lg) 0 0;
  box-shadow: 0 -12px 32px rgb(0 0 0 / 0.28);
}
.drawer-enter-active,
.drawer-leave-active {
  transition:
    transform var(--zen-dur, 0.2s) ease,
    opacity var(--zen-dur, 0.2s) ease;
}
.drawer-enter-from,
.drawer-leave-to {
  transform: translateY(16px);
  opacity: 0;
}
.grip {
  position: absolute;
  top: -4px;
  left: 0;
  right: 0;
  height: 9px;
  cursor: ns-resize;
  touch-action: none;
  z-index: 2;
}
.grip::after {
  content: '';
  position: absolute;
  top: 6px;
  left: 50%;
  width: 36px;
  height: 3px;
  margin-left: -18px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-text) 22%, transparent);
  transition: background var(--zen-dur-fast, 0.12s) ease;
}
.grip:hover::after {
  background: var(--zen-accent);
}
.bar {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 46px;
  flex: none;
  padding: 6px 10px 0 14px;
  min-width: 0;
  overflow: hidden;
}
.title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}
.seg {
  display: inline-flex;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.seg button,
.chip-btn {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 3px 9px;
  border-radius: var(--r-md);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}
.seg button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.seg button:hover,
.chip-btn:hover {
  color: var(--zen-text);
}
.n {
  font-size: 11px;
  min-width: 2ch;
  text-align: right;
  color: var(--zen-muted);
}
.n.hot {
  color: var(--zen-danger);
}
.filter {
  flex: 0 1 200px;
  min-width: 90px;
}
.bar > * {
  flex-shrink: 0;
}
.bar > .filter,
.bar > .spacer {
  flex-shrink: 1;
}
.narrow-only {
  display: none;
}
.narrow .wide-only {
  display: none;
}
.narrow .narrow-only {
  display: inline;
}
.narrow .jump-slot {
  width: 36px;
}
.tabs button {
  font-weight: 600;
}
/* fixed-width readouts: numbers change without moving anything */
.readout {
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  font-size: 12px;
  white-space: nowrap;
}
.readout .k {
  color: var(--zen-muted);
  font-family: inherit;
  font-size: 11px;
}
.readout .v {
  width: 5ch;
  text-align: right;
  color: var(--zen-text);
}
.readout .of {
  width: 9ch;
  color: var(--zen-muted);
}
.legend-note {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--zen-muted);
  white-space: nowrap;
  margin-right: 6px;
}
.mk-key {
  width: 0;
  height: 12px;
  border-left: 1px dotted var(--zen-muted);
  margin-left: 6px;
}
.mk-key.err {
  border-left: 1px solid var(--zen-danger);
}
.loaded {
  padding: 4px 16px 16px;
}
.mem {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  padding: 4px 16px 12px;
}
.mem.stacked {
  grid-template-columns: 1fr;
}
/* one-line captions so side-by-side plots line up; the legend clips rather than wraps */
.mem :deep(figcaption) {
  flex-wrap: nowrap;
  overflow: hidden;
  white-space: nowrap;
}
.mem :deep(.legend) {
  flex-wrap: nowrap;
}
.jump-slot {
  display: inline-flex;
  justify-content: flex-end;
  width: 84px;
}
.icon-btn {
  width: 28px;
  height: 28px;
  display: inline-grid;
  place-items: center;
  border: none;
  background: none;
  color: var(--zen-muted);
  border-radius: var(--r-md);
  cursor: pointer;
  font-size: 16px;
}
.icon-btn:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.body {
  flex: 1;
  min-height: 0;
}
.lines {
  padding: 4px 12px 12px;
  font-size: 12px;
  line-height: 1.5;
}
.line {
  display: grid;
  grid-template-columns: 8ch minmax(0, 1fr);
  gap: 10px;
  padding: 0 6px;
  border-radius: var(--r-sm);
}
.line:hover {
  background: color-mix(in srgb, var(--zen-text) 5%, transparent);
}
.t {
  color: var(--zen-muted);
  opacity: 0.7;
  user-select: none;
}
.m {
  white-space: pre-wrap;
  word-break: break-word;
  min-width: 0;
}
.line.warn .m {
  color: var(--zen-warn);
}
.line.error {
  background: color-mix(in srgb, var(--zen-danger) 9%, transparent);
}
.line.error .m {
  color: color-mix(in srgb, var(--zen-danger) 75%, var(--zen-text));
}
.line.partial .m {
  color: var(--zen-muted);
}
.empty {
  padding: 18px 6px;
  color: var(--zen-muted);
  font-family: inherit;
}
@media (max-width: 720px) {
  .filter {
    display: none;
  }
}
</style>
