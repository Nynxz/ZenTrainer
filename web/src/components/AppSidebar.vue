<script setup lang="ts">
// The app rail. Every row keeps its icon in a fixed 44px column, so folding only
// changes the width: nothing moves, labels just fade. The right edge (highlighted on
// hover) or "[" folds it. Themes live in Settings. Colours come from the pack's --sidebar-* tokens when it
// has them, else from the general surface tokens.
import { ZenLotus } from '@nynxz/zenkit-ui'
import ScrollArea from './ScrollArea.vue'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { activeRuns, latest, lossTrail, queue, refreshQueue, refreshRuns } from '@/stores/runs'
import { activeCaptions, startCaptionSync } from '@/stores/captions'
import { settings } from '@/stores/settings'
import { comfyBehind, comfyOld, engineStale, restartEngine, restarting, startStatusSync, webBuildable, webNewer, webStale, ztBehind, ztVersion } from '@/stores/status'
import { activeDownloads, fmtBytes, startDownloadSync } from '@/stores/downloads'
import { disks, gb, memLatest, startSystemSync } from '@/stores/system'
import { drawerOpen, toggleDrawer, unseenErrors } from '@/stores/logs'
import { engine, type LoadedModel } from '@/api/engine'

const props = defineProps<{ collapsed: boolean; mobileOpen: boolean }>()
const emit = defineEmits<{ toggle: []; navigate: [] }>()

// the mark: blooms when clicked, breathes while something trains
const lotus = ref<InstanceType<typeof ZenLotus> | null>(null)
const simple = computed(() => settings.mode === 'simple')
const training = computed(() => activeRuns.value.some((r) => r.status === 'training'))
const route = useRoute()

interface NavItem {
  to: string
  icon: string
  label: string
  count?: () => number
  /** behind Settings → Experimental pages */
  experimental?: boolean
}
// six places: no group headings needed
const items: NavItem[] = [
  // everything that runs: training (and its queue) and caption jobs
  { to: '/jobs', icon: 'mdi mdi-chart-timeline-variant', label: 'Jobs', count: () => activeRuns.value.length + activeCaptions.value.length },
  { to: '/datasets', icon: 'mdi mdi-image-multiple-outline', label: 'Datasets' },
  { to: '/models', icon: 'mdi mdi-cube-outline', label: 'Models' },
  { to: '/apps', icon: 'mdi mdi-view-grid-outline', label: 'Apps', experimental: true },
  { to: '/tools', icon: 'mdi mdi-toolbox-outline', label: 'Tools', experimental: true },
  { to: '/gallery', icon: 'mdi mdi-image-album', label: 'Gallery', experimental: true },
]
const isActive = (to: string) =>
  to === '/jobs' ? route.path.startsWith('/jobs') || (route.path.startsWith('/runs/') && route.path !== '/runs/new') : route.path === to || route.path.startsWith(to + '/')

// ---- runs -------------------------------------------------------------------------
const live = computed(() =>
  activeRuns.value.map((r) => {
    const m = latest[r.run_id]
    const frac = m?.total ? m.step / m.total : 0
    return { ...r, step: m?.step ?? 0, total: m?.total ?? 0, it_s: m?.it_s, frac, spark: spark(lossTrail[r.run_id]) }
  }),
)

// the run that's going gets the card; the ones waiting, in queue order, become its "next" line
// (a card per queued run was a stack of empty rings). Nothing running (a paused queue): the line alone
const running = computed(() => live.value.filter((r) => r.status !== 'queued'))
const waiting = computed(() => {
  const order = queue.value.items.map((i) => i.run_id)
  return live.value.filter((r) => r.status === 'queued').sort((a, b) => order.indexOf(a.run_id) - order.indexOf(b.run_id))
})

// a paused queue: resume it from here, as Jobs' Resume queue does
const resuming = ref(false)
async function resumeQueue() {
  resuming.value = true
  try {
    await engine.editQueue({ action: 'resume' })
    await Promise.all([refreshQueue(), refreshRuns()])
  } catch (e) {
    alert(String((e as Error).message ?? e))
  } finally {
    resuming.value = false
  }
}

function spark(values: number[] | undefined): string {
  if (!values || values.length < 2) return ''
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  return values
    .map(
      (v, i) =>
        `${i ? 'L' : 'M'}${((i / (values.length - 1)) * 100).toFixed(1)},${(22 - ((v - lo) / (hi - lo || 1)) * 20).toFixed(1)}`,
    )
    .join('')
}

// ---- memory --------------------------------------------------------------------------
const meters = computed(() => {
  const m = memLatest.value
  if (!m) return []
  const out: { label: string; used: number; total: number; text?: string }[] = []
  if (m.vram_total) out.push({ label: 'VRAM', used: m.vram_used ?? 0, total: m.vram_total })
  if (m.ram_total) out.push({ label: 'RAM', used: m.ram_used ?? 0, total: m.ram_total })
  // the disk outputs land on: checkpoints and resume states fill it fast
  const d = disks.value[0]
  if (d) out.push({ label: 'Disk', used: d.used, total: d.total, text: `${gb(d.free)} GB free` })
  return out
})

// ---- downloads: the one in progress, and how many wait behind it -----------------------
const dl = computed(() => {
  const list = activeDownloads.value
  const cur = list.find((d) => d.status === 'downloading') ?? list[0]
  if (!cur) return null
  return { name: cur.name, pct: cur.total ? cur.done / cur.total : 0, done: cur.done, total: cur.total, speed: cur.speed, more: list.length - 1 }
})

// ---- caption jobs: the one running (or first waiting), and how many more ---------------
const cap = computed(() => {
  const list = activeCaptions.value
  const cur = list.find((j) => j.status === 'running') ?? list[0]
  if (!cur) return null
  const total = cur.total || cur.preview || 0
  const meta = cur.status === 'queued' ? 'waiting' : cur.message.startsWith('loading') ? 'loading' : `${cur.done}/${total}`
  return { job: cur, pct: total ? cur.done / total : 0, meta, more: list.length - 1 }
})

// ---- housekeeping: an update to load (rebuild the page, restart ComfyUI), or ComfyUI has one -
// the version beside the name: which build this is, and whether there's a newer one
const versionTitle = computed(() => {
  const v = ztVersion.value
  if (!v) return ''
  const build = `ZenTrainer ${v.describe ?? `v${v.version}`}${v.dirty ? ', with local changes' : ''}`
  if (!v.upstream) return build
  return ztBehind.value ? `${build}: ${ztBehind.value} new commit${ztBehind.value === 1 ? '' : 's'} upstream (see Settings)` : `${build}: up to date`
})
const housekeeping = computed(() => !!(comfyOld.value || engineStale.value || webStale.value || webNewer.value || ztBehind.value || comfyBehind.value))
const reload = () => location.reload()
const trainingNow = computed(() => activeRuns.value.some((r) => r.status === 'training' || r.status === 'loading'))
// what the square does: rebuild the page and/or restart ComfyUI, whichever is behind
const update = computed(() => {
  const what = [webStale.value && 'rebuild the web app', engineStale.value && 'restart ComfyUI to load the engine'].filter(Boolean).join(', then ')
  if (engineStale.value && trainingNow.value) return { ok: false, title: 'An update is waiting: it loads once nothing is training' }
  if (webStale.value && !webBuildable.value)
    return { ok: false, title: 'The web app’s source changed (a pull?), but pnpm isn’t here to rebuild it: run `pnpm install && pnpm build` in web/' }
  return { ok: true, title: `Updated code is on disk: ${what}`, confirm: `Update now: ${what}?${engineStale.value ? ' The restart takes about a minute.' : ''}` }
})
async function onUpdate() {
  const u = update.value
  if (!u.ok || !confirm(u.confirm)) return
  await restartEngine().catch((e) => alert(String((e as Error).message ?? e)))
}

// ---- folding & keys ------------------------------------------------------------------
function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement
  if (e.metaKey || e.ctrlKey || e.altKey || t.closest('input, textarea, select, [contenteditable]')) return
  if (e.key === '[') emit('toggle')
}

// ---- what ComfyUI holds right now, on every page: a training model, the captioner, SAM3 while it
// loads for a mask... (each by its file and how much of it is in VRAM). Read every few seconds while
// the sidebar is open and the page is on screen
const loaded = ref<LoadedModel[]>([])
let loadedTimer: ReturnType<typeof setTimeout> | undefined
async function pollLoaded() {
  if (!props.collapsed || props.mobileOpen) {
    if (document.visibilityState === 'visible') loaded.value = (await engine.loadedModels().catch(() => loaded.value)).filter((m) => m.bytes > 0)
  }
  loadedTimer = setTimeout(pollLoaded, 3000)
}
const KIND_ICON: Record<string, string> = { 'diffusion model': 'mdi-cube-outline', 'text encoder': 'mdi-format-text', vae: 'mdi-image-filter-center-focus' }
const loadedRows = computed(() =>
  [...loaded.value]
    .sort((a, b) => b.vram - a.vram)
    .map((m) => {
      const file = (m.file ?? m.name).split(' + ')[0]!.split(/[\\/]/).pop()!.replace(/\.(safetensors|sft|ckpt|pt|pth|bin)$/, '')
      return { key: `${m.name}:${m.file}`, label: file, icon: KIND_ICON[m.kind ?? ''] ?? 'mdi-shape-outline', vram: m.vram, bytes: m.bytes, kind: m.kind ?? 'model', full: m.file ?? m.name }
    }),
)
onMounted(() => {
  pollLoaded()
  startSystemSync()
  startDownloadSync()
  startCaptionSync()
  startStatusSync()
  addEventListener('keydown', onKey)
})
onUnmounted(() => {
  clearTimeout(loadedTimer)
  removeEventListener('keydown', onKey)
})
</script>

<template>
  <aside class="rail" :class="{ collapsed: collapsed && !mobileOpen, open: mobileOpen }" aria-label="Sidebar">
    <button
      class="edge"
      :title="collapsed ? 'Expand sidebar ([)' : 'Fold sidebar ([)'"
      :aria-label="collapsed ? 'Expand sidebar' : 'Fold sidebar'"
      :aria-expanded="!collapsed"
      @click="emit('toggle')"
    />
    <RouterLink to="/jobs" class="row brand" :title="training ? 'Training…' : undefined" @click="lotus?.bloom(); emit('navigate')">
      <!-- no breathing while training: a constant animation kept the page repainting for the whole run (the click still blooms) -->
      <span class="ic"><ZenLotus ref="lotus" class="lotus" :breathing="false" /></span>
      <span class="lbl"><span class="zt-title">ZenTrainer</span><span v-if="ztVersion" class="ver" :title="versionTitle">v{{ ztVersion.version }}</span></span>
    </RouterLink>

    <RouterLink :to="simple ? '/simple' : '/runs/new'" class="row cta" @click="emit('navigate')">
      <span class="ic"><i class="mdi mdi-plus" /></span>
      <span class="lbl">{{ simple ? 'Train a LoRA' : 'New run' }}</span>
    </RouterLink>
    <ScrollArea class="nav-scroll"><nav>
      <div v-for="item in items.filter((i) => !i.experimental || settings.experimental)" :key="item.to" class="nav-row">
      <RouterLink
        :to="item.to"
        class="row item"
        :class="{ active: isActive(item.to) }"
        :title="collapsed ? item.label : undefined"
        @click="emit('navigate')"
      >
        <span class="ic">
          <i :class="item.icon" />
          <span v-if="item.count?.()" class="dot-count" />
        </span>
        <span class="lbl">{{ item.label }}</span>
        <span v-if="item.count?.()" class="lbl count">{{ item.count() }}</span>
      </RouterLink>
      <!-- a paused queue: resume it here, as the update square beside Settings loads an update -->
      <button
        v-if="item.to === '/jobs' && queue.paused && (!collapsed || mobileOpen)"
        class="sq"
        :disabled="resuming"
        :title="waiting.length ? `Queue paused: resume it (${waiting.length} waiting)` : 'Queue paused: resume it (a run you queue waits until then)'"
        @click="resumeQueue"
      >
        <i :class="resuming ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-play'" />
      </button>
      </div>
    </nav></ScrollArea>

    <!-- what's training, above the memory meters: one click to it. The queue rides on the card's
         foot as one "next" line (all of it in the tooltip), so the card stays put as runs come and go -->
    <div v-if="running.length || waiting.length" class="live">
      <RouterLink
        v-for="r in running"
        :key="r.run_id"
        :to="simple ? `/simple/run/${r.run_id}` : `/runs/${r.run_id}`"
        class="live-run"
        :title="`${r.name}: ${r.status} ${Math.round(r.frac * 100)}%`"
        @click="emit('navigate')"
      >
        <span class="ic">
          <svg viewBox="0 0 36 36" class="ring" :class="r.status">
            <circle cx="18" cy="18" r="15" class="track" />
            <circle cx="18" cy="18" r="15" class="arc" :stroke-dasharray="`${r.frac * 94.25} 94.25`" />
          </svg>
          <span class="pct">{{ Math.round(r.frac * 100) }}</span>
        </span>
        <span class="lbl live-body">
          <span class="live-name ellipsis">{{ r.name }}</span>
          <svg v-if="r.spark" viewBox="0 0 100 24" preserveAspectRatio="none" class="spark"><path :d="r.spark" /></svg>
          <span class="live-meta">
            {{ r.status === 'training' ? `${r.step}/${r.total} · ${r.it_s?.toFixed(2) ?? '–'} it/s` : r.status }}
          </span>
        </span>
      </RouterLink>
      <div v-if="waiting.length" class="next-row">
        <RouterLink
          to="/jobs"
          class="live-next"
          :class="{ paused: queue.paused }"
          :title="`${queue.paused ? 'Queue paused. ' : ''}Queued, in order:\n${waiting.map((w, i) => `${i + 1}. ${w.name}`).join('\n')}`"
          @click="emit('navigate')"
        >
          <span class="ic"><i :class="queue.paused ? 'mdi mdi-pause-circle-outline' : 'mdi mdi-tray-full'" /><span class="q-count">{{ waiting.length }}</span></span>
          <span class="lbl next-body"><span class="muted">{{ queue.paused ? 'Paused' : 'Next' }}</span><span class="ellipsis">{{ waiting[0]!.name }}</span><span v-if="waiting.length > 1" class="mono muted">+{{ waiting.length - 1 }}</span></span>
        </RouterLink>
      </div>
    </div>



    <RouterLink
      v-if="cap"
      :to="`/jobs/caption/${cap.job.job_id}`"
      class="row mem"
      :title="cap.job.message || `${cap.job.preview ? 'Trying captions on' : 'Captioning'} ${cap.job.dataset}${cap.job.file ? `: ${cap.job.file}` : ''}`"
      @click="emit('navigate')"
    >
      <span class="ic"><i class="mdi mdi-creation-outline" /></span>
      <span class="lbl mem-body">
        <span class="mem-row">
          <span class="mem-head"><span class="ellipsis">{{ cap.job.rewrite?.length ? 'Rewrite prompts' : `${cap.job.preview ? 'Try' : 'Caption'} ${cap.job.dataset}` }}</span><span class="mono">{{ cap.meta }}{{ cap.more ? ` +${cap.more}` : '' }}</span></span>
          <span class="meter"><span :style="{ width: `${cap.pct * 100}%` }" /></span>
        </span>
      </span>
    </RouterLink>

    <div v-if="dl" class="row mem" :title="`${dl.name}: ${fmtBytes(dl.done)} of ${fmtBytes(dl.total)} · ${fmtBytes(dl.speed)}/s`">
      <span class="ic"><i class="mdi mdi-download" /></span>
      <span class="lbl mem-body">
        <span class="mem-row">
          <span class="mem-head"><span class="ellipsis">{{ dl.name }}</span><span class="mono">{{ Math.floor(dl.pct * 100) }}%{{ dl.more ? ` +${dl.more}` : '' }}</span></span>
          <span class="meter"><span :style="{ width: `${dl.pct * 100}%` }" /></span>
        </span>
      </span>
    </div>

    <RouterLink v-if="meters.length" to="/system" class="row mem" :class="{ active: isActive('/system') }" title="System memory" @click="emit('navigate')">
      <span class="ic"><i class="mdi mdi-memory" /></span>
      <span class="lbl mem-body">
        <!-- what's loaded, first: each with a ring for how much of it is in VRAM (the rest offloaded) -->
        <span v-if="loadedRows.length" class="loaded">
          <span v-for="m in loadedRows.slice(0, 4)" :key="m.key" class="loaded-row" :title="`${m.full}\n${m.kind} · ${gb(m.vram)} of ${gb(m.bytes)} GB in VRAM`">
            <svg class="lring" viewBox="0 0 16 16" aria-hidden="true">
              <circle cx="8" cy="8" r="6" class="track" />
              <circle cx="8" cy="8" r="6" class="fill" :stroke-dasharray="`${Math.min(1, m.vram / (m.bytes || 1)) * 37.7} 37.7`" />
            </svg>
            <span class="ellipsis">{{ m.label }}</span>
            <span class="mono">{{ gb(m.vram) }}</span>
          </span>
          <span v-if="loadedRows.length > 4" class="loaded-more mono">+{{ loadedRows.length - 4 }} more</span>
        </span>
        <span v-for="m in meters" :key="m.label" class="mem-row">
          <span class="mem-head"><span>{{ m.label }}</span><span class="mono">{{ 'text' in m ? m.text : `${gb(m.used)}/${gb(m.total)} GB` }}</span></span>
          <span class="meter"><span :style="{ width: `${(m.used / m.total) * 100}%` }" :class="{ hot: m.used / m.total > 0.9 }" /></span>
        </span>
      </span>
    </RouterLink>

    <div class="foot">
      <button
        class="row item log-row"
        :class="{ active: drawerOpen }"
        :title="collapsed ? 'Log (Ctrl+`)' : 'ComfyUI log (Ctrl+`)'"
        @click="toggleDrawer()"
      >
        <span class="ic">
          <i class="mdi mdi-console-line" />
          <span v-if="unseenErrors" class="dot-count err" />
        </span>
        <span class="lbl">Log</span>
        <span v-if="unseenErrors" class="lbl count err" :title="`${unseenErrors} new error${unseenErrors === 1 ? '' : 's'}`">{{ unseenErrors > 99 ? '99+' : unseenErrors }}</span>
        <kbd v-else class="lbl key">Ctrl `</kbd>
      </button>
      <div class="foot-row">
        <RouterLink
          to="/settings"
          class="row item"
          :class="{ active: isActive('/settings') }"
          :title="collapsed ? 'Settings' : undefined"
          @click="emit('navigate')"
        >
          <span class="ic"><i class="mdi mdi-cog-outline" /><span v-if="housekeeping" class="dot-count" /></span>
          <span class="lbl">Settings</span>
        </RouterLink>
        <!-- housekeeping, open sidebar only: restart to load an update, or ZenTrainer or ComfyUI has one -->
        <template v-if="!collapsed || mobileOpen">
          <span v-if="comfyOld" class="sq warn" :title="comfyOld"><i class="mdi mdi-alert-outline" /></span>
          <button
            v-else-if="engineStale || webStale"
            class="sq"
            :disabled="!update.ok || restarting"
            :title="restarting ? 'Updating…' : update.title"
            @click="onUpdate"
          >
            <i :class="restarting ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-restart'" />
          </button>
          <button v-else-if="webNewer" class="sq" title="The web app was rebuilt: reload to see the new version" @click="reload()">
            <i class="mdi mdi-refresh" />
          </button>
          <RouterLink v-else-if="ztBehind" to="/settings" class="sq" :title="`ZenTrainer update available: ${ztBehind} new commit${ztBehind === 1 ? '' : 's'} (see Settings)`" @click="emit('navigate')">
            <i class="mdi mdi-arrow-up-circle-outline" />
          </RouterLink>
          <RouterLink v-else-if="comfyBehind" to="/settings" class="sq" :title="`ComfyUI update available: ${comfyBehind} commits behind upstream`" @click="emit('navigate')">
            <i class="mdi mdi-arrow-up-circle-outline" />
          </RouterLink>
        </template>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.rail {
  /* pack sidebar tokens first, general surfaces as fallback */
  --sb-bg: var(--sidebar, color-mix(in srgb, var(--zen-surface) 70%, var(--zen-bg)));
  --sb-fg: var(--sidebar-foreground, var(--zen-text));
  --sb-muted: var(--zen-muted);
  --sb-primary: var(--sidebar-primary, var(--zen-accent));
  --sb-primary-fg: var(--sidebar-primary-foreground, var(--zen-accent-text));
  --sb-accent: var(--sidebar-accent, color-mix(in srgb, var(--zen-accent) 14%, transparent));
  --sb-accent-fg: var(--sidebar-accent-foreground, var(--zen-text));
  --sb-border: var(--sidebar-border, color-mix(in srgb, var(--zen-border) 70%, transparent));
  --w: 244px;
  --ic: 44px;
  width: var(--w);
  flex: none;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 10px 10px;
  background: var(--sb-bg);
  color: var(--sb-fg);
  border-right: 1px solid var(--sb-border);
  transition: width 0.2s cubic-bezier(0.2, 0, 0, 1);
  overflow: hidden;
  position: relative;
  user-select: none;
}
.rail.collapsed {
  --w: calc(var(--ic) + 20px);
}
/* the fold handle: a strip over the right border that lights up on hover */
.edge {
  position: absolute;
  top: 0;
  right: -1px;
  bottom: 0;
  width: 7px;
  z-index: 5;
  padding: 0;
  border: none;
  background: none;
  cursor: w-resize !important;
}
.collapsed .edge {
  cursor: e-resize !important;
}
.edge::after {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 2px;
  background: var(--sb-primary);
  opacity: 0;
  transition: opacity 0.15s ease;
}
.edge:hover::after,
.edge:focus-visible::after {
  opacity: 0.7;
}
.rail a,
.rail button {
  cursor: pointer;
}

/* every row: [44px icon column][label]; the icon never moves */
.row,
.live-run,
.live-next {
  display: flex;
  align-items: center;
  min-height: 36px;
  border-radius: var(--r-md);
  color: inherit;
  flex: none;
}
.ic {
  position: relative;
  width: var(--ic);
  height: 36px;
  flex: none;
  display: grid;
  place-items: center;
  font-size: 18px;
}
.lbl {
  white-space: nowrap;
  transition: opacity 0.15s ease;
}
.collapsed .lbl {
  opacity: 0;
  pointer-events: none;
}
.ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}

.ver {
  margin-left: 7px;
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 10.5px;
  font-weight: 500;
  letter-spacing: 0;
  color: var(--zen-muted);
  opacity: 0.8;
}
.brand {
  min-height: 40px;
  margin-bottom: 6px;
  font-weight: 700;
  font-size: 15px;
  letter-spacing: -0.01em;
}
.lotus {
  width: 24px;
  height: 24px;
  color: var(--zen-accent);
}
.cta {
  background: var(--sb-primary);
  color: var(--sb-primary-fg);
  font-weight: 600;
  margin-bottom: 8px;
}
.cta:hover {
  filter: brightness(1.07);
}

.nav-scroll {
  flex: 1;
  height: auto;
  margin-bottom: 8px;
}
nav {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-top: 4px;
}
.item {
  position: relative;
  color: var(--sb-muted);
}
.item:hover {
  color: var(--sb-fg);
  background: color-mix(in srgb, var(--sb-accent) 60%, transparent);
}
.item.active {
  color: var(--sb-accent-fg);
  background: var(--sb-accent);
}
.item.active::before {
  content: '';
  position: absolute;
  left: -10px;
  top: 8px;
  bottom: 8px;
  width: 3px;
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  background: var(--sb-primary);
}
.item.active .ic i {
  color: var(--sb-primary);
}
.count {
  margin-left: auto;
  margin-right: 10px;
  min-width: 20px;
  height: 18px;
  padding: 0 6px;
  border-radius: var(--r-sm);
  display: grid;
  place-items: center;
  font-size: 11px;
  font-weight: 650;
  background: var(--sb-primary);
  color: var(--sb-primary-fg);
}
.dot-count {
  position: absolute;
  top: 7px;
  right: 10px;
  width: 7px;
  height: 7px;
  border-radius: var(--r-sm);
  background: var(--sb-primary);
  opacity: 0;
  transition: opacity 0.15s ease;
}
.collapsed .dot-count {
  opacity: 1;
}
.log-row {
  width: 100%;
  border: none;
  background: none;
  font: inherit;
  text-align: left;
  padding: 0;
}
.count.err,
.dot-count.err {
  background: var(--zen-danger);
  color: #fff;
}
.key {
  margin-left: auto;
  margin-right: 10px;
  font: 600 10px var(--zen-mono);
  padding: 1px 5px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, currentColor 12%, transparent);
  opacity: 0.8;
}
.collapsed .key {
  opacity: 0;
}

/* live run: progress ring in the icon column, details beside it */
.live {
  display: flex;
  flex-direction: column;
  flex: none;
  margin-bottom: 4px;
  border: 1px solid var(--sb-border);
  background: color-mix(in srgb, var(--zen-surface) 70%, transparent);
  border-radius: var(--r-lg);
  transition: border-color 0.15s ease, background 0.15s ease;
}
.collapsed .live {
  border-color: transparent;
  background: none;
}
.live:hover {
  border-color: var(--sb-primary);
}
.live-run {
  min-height: 54px;
}
.live-run .ic {
  height: 54px;
}
.ring {
  width: 32px;
  height: 32px;
  transform: rotate(-90deg);
}
.ring circle {
  fill: none;
  stroke-width: 3;
}
.ring .track {
  stroke: color-mix(in srgb, var(--sb-muted) 25%, transparent);
}
.ring .arc {
  stroke: var(--sb-primary);
  stroke-linecap: round;
  transition: stroke-dasharray 0.4s ease;
}
.ring.paused .arc {
  stroke: var(--zen-warn);
}
.pct {
  position: absolute;
  font: 600 9px var(--zen-mono);
}
.live-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-right: 10px;
}
.live-name {
  font-weight: 650;
  font-size: 12px;
}
.spark {
  width: 100%;
  height: 16px;
}
.spark path {
  fill: none;
  stroke: var(--sb-primary);
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
  stroke-linejoin: round;
}
.live-meta {
  font: 10px var(--zen-mono);
  color: var(--sb-muted);
}

/* the queue: the card's foot, the next run and how many more */
.live-next {
  min-height: 28px;
  color: var(--sb-muted);
  font-size: 11px;
}
.live-run + .next-row {
  border-top: 1px solid var(--sb-border);
}
.next-row {
  display: flex;
  align-items: center;
  gap: 4px;
}
.next-row > .live-next {
  flex: 1;
  min-width: 0;
}
.nav-row {
  display: flex;
  align-items: center;
  gap: 4px;
}
.nav-row > .item {
  flex: 1;
  min-width: 0;
}
.live-next.paused .ic {
  color: var(--zen-warn);
}
.collapsed .next-row {
  border-top-color: transparent;
}
.live-next:hover {
  color: var(--sb-fg);
}
.live-next .ic {
  height: 26px;
  font-size: 15px;
}
.q-count {
  position: absolute;
  right: 8px;
  top: -2px;
  font: 600 9px var(--zen-mono);
  color: var(--sb-primary);
}
.next-body {
  flex: 1;
  min-width: 0;
  display: flex;
  gap: 6px;
  padding-right: 10px;
}
.next-body .ellipsis {
  flex: 1;
}

/* memory */
.mem {
  color: var(--sb-muted);
  min-height: 52px;
}
.mem:hover,
.mem.active {
  color: var(--sb-fg);
  background: color-mix(in srgb, var(--sb-accent) 60%, transparent);
}
.mem-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 6px 10px 6px 0;
  font-size: 11px;
}
.loaded {
  display: flex;
  flex-direction: column;
  gap: 3px;
  margin-bottom: 2px;
  padding-bottom: 6px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.lring {
  width: 13px;
  height: 13px;
  transform: rotate(-90deg);
}
.lring circle {
  fill: none;
  stroke-width: 2.5;
}
.lring .track {
  stroke: color-mix(in srgb, var(--zen-text) 15%, transparent);
}
.lring .fill {
  stroke: var(--zen-accent);
  stroke-linecap: round;
}
.loaded-row {
  display: grid;
  grid-template-columns: 14px minmax(0, 1fr) auto;
  align-items: center;
  gap: 5px;
  font-size: 10.5px;
  color: var(--zen-muted);
}
.loaded-more {
  font-size: 10px;
  color: var(--zen-muted);
  padding-left: 19px;
}
.mem-row {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.mem-head {
  display: flex;
  justify-content: space-between;
  gap: 6px;
}
.meter {
  height: 4px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--sb-muted) 22%, transparent);
  overflow: hidden;
}
.meter span {
  display: block;
  height: 100%;
  background: var(--sb-primary);
  transition: width 0.5s ease;
}
.meter span.hot {
  background: var(--zen-warn);
}

/* footer */
.foot {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-top: 6px;
  margin-top: 4px;
  border-top: 1px solid var(--sb-border);
}
@media (max-width: 720px) {
  .rail,
  .rail.collapsed {
    --w: 272px;
    position: fixed;
    inset: 0 auto 0 0;
    z-index: 50;
    transform: translateX(-100%);
    transition: transform 0.2s ease;
    box-shadow: 0 0 40px rgb(0 0 0 / 0.35);
  }
  .edge {
    display: none;
  }
  .rail.open {
    transform: none;
  }
}
/* Settings, and beside it (open sidebar only) a square for housekeeping: an update to load */
.foot-row {
  display: flex;
  align-items: center;
  gap: 4px;
}
.foot-row > .item {
  flex: 1;
  min-width: 0;
}
.sq {
  width: 36px;
  height: 36px;
  flex: none;
  display: grid;
  place-items: center;
  font-size: 18px;
  border: 1px solid var(--sb-border);
  border-radius: var(--r-md);
  background: none;
  color: var(--sb-primary);
  cursor: pointer;
}
/* a square only shows with the sidebar open: it fades in once the unfold's 0.2s widening is done,
   rather than appearing squeezed into the still-narrow rail */
.sq {
  animation: sq-in 0.15s ease 0.2s both;
}
@keyframes sq-in {
  from {
    opacity: 0;
  }
}
/* the glyph's own line box sat it a little high in the square */
.sq > i {
  display: block;
  line-height: 1;
}
.sq:hover:not(:disabled) {
  background: color-mix(in srgb, var(--sb-accent) 60%, transparent);
}
.sq:disabled {
  cursor: default;
  color: var(--sb-muted);
}
.sq.warn {
  color: var(--zen-warn);
  cursor: help;
}
</style>
