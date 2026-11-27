<script setup lang="ts">
// A dataset's analysis (engine datasets/analysis.py): pre-training audit & cleanup cockpit.
// Overview KPIs + focused tabs (Issues, Duplicates, Buckets, Captions, Spread) with lightbox,
// right-click actions, and one-click mute/delete so problems can be fixed right here.
import {
  ZenButton,
  ZenCombo,
  ZenContextMenu,
  ZenEmpty,
  ZenLightbox,
  ZenToggleGroup,
  type ContextMenuItem,
  type LightboxItem,
} from '@nynxz/zenkit-ui'
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { EngineError, engine, thumbUrl, type DatasetAnalysis } from '@/api/engine'
import { ask, tellError } from '@/lib/ask'
import { remembered } from '@/lib/remembered'

const props = defineProps<{ name: string }>()
const route = useRoute()
const router = useRouter()

type Tab = 'issues' | 'duplicates' | 'buckets' | 'captions' | 'spread'
const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'issues', label: 'Issues', icon: 'mdi-alert-circle-outline' },
  { id: 'duplicates', label: 'Duplicates', icon: 'mdi-content-duplicate' },
  { id: 'buckets', label: 'Buckets', icon: 'mdi-grid' },
  { id: 'captions', label: 'Captions', icon: 'mdi-text-box-outline' },
  { id: 'spread', label: 'Spread', icon: 'mdi-chart-bar' },
]

const RESOLUTIONS = ['256', '512', '768', '1024', '1280', '1536']
const resolution = ref<string>(RESOLUTIONS.includes(route.query.resolution as string) ? (route.query.resolution as string) : '1024')
const activeSet = ref<string>((route.query.set as string) || 'default')
const sets = ref<string[]>(['default'])
const muted = ref<Set<string>>(new Set())
const data = ref<DatasetAnalysis | null>(null)
const loading = ref(false)
const busy = ref(false)
const error = ref<string | null>(null)
const notice = ref<string | null>(null)
const tab = ref<Tab>(TABS.some((t) => t.id === route.query.tab) ? (route.query.tab as Tab) : 'issues')
let pickedInitialTab = TABS.some((t) => t.id === route.query.tab)

function flash(msg: string) {
  notice.value = msg
  setTimeout(() => notice.value === msg && (notice.value = null), 2600)
}

async function loadMeta() {
  try {
    const ds = await engine.dataset(props.name)
    sets.value = ds.sets.length ? ds.sets : ['default']
    if (!sets.value.includes(activeSet.value)) activeSet.value = sets.value[0]!
    const m = new Set<string>()
    for (const it of ds.items) {
      if (it.muted?.includes(activeSet.value)) m.add(it.path)
    }
    muted.value = m
  } catch {
    // analysis endpoint will report any fatal error
  }
}

async function loadAnalysis() {
  loading.value = true
  error.value = null
  try {
    data.value = await engine.datasetAnalysis(props.name, activeSet.value, Number(resolution.value))
    if (!pickedInitialTab && data.value) {
      pickedInitialTab = true
      const hasIssues = data.value.pictures.some((p) => p.flags.length)
      const hasDups = data.value.duplicates.length > 0
      tab.value = hasIssues ? 'issues' : hasDups ? 'duplicates' : 'buckets'
    }
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

watch(
  [() => props.name, activeSet, resolution],
  async () => {
    const q: Record<string, string> = { resolution: resolution.value }
    if (activeSet.value !== 'default') q.set = activeSet.value
    if (tab.value) q.tab = tab.value
    router.replace({ query: q })
    await Promise.all([loadMeta(), loadAnalysis()])
  },
  { immediate: true },
)

watch(tab, (t) => {
  if (route.query.tab !== t) router.replace({ query: { ...route.query, tab: t } })
})

const thumb = (path: string, size = 256) => thumbUrl(props.name, path, size)
const pct = (x: number) => `${Math.round(x * 100)}%`
const isMuted = (path: string) => muted.value.has(path)
const picMap = computed(() => new Map((data.value?.pictures ?? []).map((p) => [p.path, p])))
const fmtSize = (sz?: [number, number] | null) => (sz ? `${sz[0]}×${sz[1]}` : '—')

// ---- actions: mute / unmute / delete / context menu / lightbox -----------------------------------
async function setMuted(paths: string[], on: boolean) {
  const unique = [...new Set(paths)]
  if (!unique.length) return
  busy.value = true
  try {
    await engine.datasetMuted(props.name, activeSet.value, unique, on)
    const next = new Set(muted.value)
    for (const p of unique) {
      if (on) next.add(p)
      else next.delete(p)
    }
    muted.value = next
    flash(`${on ? 'Muted' : 'Unmuted'} ${unique.length} ${unique.length === 1 ? 'picture' : 'pictures'} in “${activeSet.value}”`)
  } catch (e) {
    tellError(e, 'Couldn’t update muted state')
  } finally {
    busy.value = false
  }
}

async function removePaths(paths: string[]) {
  const unique = [...new Set(paths)]
  if (!unique.length) return
  const what = unique.length === 1 ? `“${unique[0]!.split('/').at(-1)}”` : `${unique.length} pictures`
  if (
    !(await ask({
      title: `Delete ${what}?`,
      message: 'With their captions and masks. This can’t be undone.',
      ok: 'Delete',
      danger: true,
    }))
  ) {
    return
  }
  busy.value = true
  try {
    for (const p of unique) await engine.deleteItem(props.name, p)
    lightbox.value = null
    flash(`Deleted ${unique.length} ${unique.length === 1 ? 'picture' : 'pictures'}`)
    await Promise.all([loadMeta(), loadAnalysis()])
  } catch (e) {
    tellError(e, 'Couldn’t delete')
    await Promise.all([loadMeta(), loadAnalysis()])
  } finally {
    busy.value = false
  }
}

const menu = ref<InstanceType<typeof ZenContextMenu> | null>(null)
const menuItems = ref<ContextMenuItem[]>([])
function onPicMenu(path: string, e: MouseEvent, list: string[] = [path]) {
  const m = isMuted(path)
  const file = path.split('/').at(-1) ?? path
  menuItems.value = [
    { heading: file },
    { label: 'Preview', icon: 'mdi-open-in-app', run: () => openLightbox(list, path) },
    { label: m ? `Unmute in “${activeSet.value}”` : `Mute in “${activeSet.value}”`, icon: m ? 'mdi-eye-outline' : 'mdi-eye-off-outline', run: () => setMuted([path], !m) },
    { label: 'Edit mask', icon: 'mdi-draw', run: () => router.push({ path: `/datasets/${encodeURIComponent(props.name)}/mask`, query: { path } }) },
    { label: 'Copy path', icon: 'mdi-content-copy', run: () => navigator.clipboard?.writeText(path) },
    '-',
    { label: 'Delete…', icon: 'mdi-trash-can-outline', danger: true, run: () => removePaths([path]) },
  ]
  menu.value?.show(e)
}

const lightbox = ref<number | null>(null)
const lbPaths = ref<string[]>([])
const lbFit = ref<'cover' | 'contain'>('contain')
function openLightbox(paths: string[], target: string) {
  lbPaths.value = paths
  const idx = paths.indexOf(target)
  lightbox.value = idx >= 0 ? idx : 0
}
const lbItems = computed<LightboxItem[]>(() =>
  lbPaths.value.map((path) => {
    const info = picMap.value.get(path)
    const parts = [fmtSize(info?.size), info?.flags.length ? info.flags.join(', ') : null, isMuted(path) ? 'muted' : null].filter(Boolean)
    return {
      src: thumb(path, 1600),
      thumb: thumb(path, 256),
      label: path,
      meta: parts.join(' · '),
    }
  }),
)

// ---- pictures to check, by issue -----------------------------------------------------------------
const FLAG_INFO: Record<string, string> = {
  small: 'Shorter side under 768 px: it gets upscaled.',
  blurry: 'Much softer than the rest of the set.',
  dark: 'Very dark overall.',
  'blown out': 'Very bright overall: detail lost in the highlights.',
  flat: 'Very little contrast.',
  bars: 'Flat bars along the edges (letterboxing or a frame).',
  greyscale: 'No colour, in a set that has colour.',
  transparent: 'See-through parts: most models train on whatever colour sits under them (often black). Qwen Image 2.1 keeps the transparency.',
  compressed: 'Heavily compressed JPEG: its blocky artefacts can be learned.',
  'near-duplicate': 'A re-saved, resized, lightly edited or mirrored copy of another picture: it counts twice.',
}
const flagFilter = ref<string | null>(null)
const hideMuted = ref(false)
const allFlagged = computed(() => (data.value?.pictures ?? []).filter((p) => p.flags.length))
const flagged = computed(() =>
  allFlagged.value.filter((p) => (!flagFilter.value || p.flags.includes(flagFilter.value)) && (!hideMuted.value || !isMuted(p.path))),
)
const flagChips = computed(() => Object.entries(data.value?.flag_counts ?? {}).sort((a, b) => b[1] - a[1]))
const unmutedFlagged = computed(() => flagged.value.filter((p) => !isMuted(p.path)).map((p) => p.path))

// ---- near-duplicates -----------------------------------------------------------------------------
function worseSide(a: string, b: string): 'a' | 'b' | null {
  const pa = picMap.value.get(a)
  const pb = picMap.value.get(b)
  const areaA = pa?.size ? pa.size[0] * pa.size[1] : 0
  const areaB = pb?.size ? pb.size[0] * pb.size[1] : 0
  if (areaA && areaB && Math.abs(areaA - areaB) / Math.max(areaA, areaB) > 0.08) return areaA < areaB ? 'a' : 'b'
  const sa = pa?.sharpness ?? 0
  const sb = pb?.sharpness ?? 0
  if (sa && sb && Math.abs(sa - sb) / Math.max(sa, sb) > 0.08) return sa < sb ? 'a' : 'b'
  return 'b'
}
function worseReason(side: 'a' | 'b', a: string, b: string): string | null {
  const w = worseSide(a, b)
  if (w !== side) return null
  const pa = picMap.value.get(a)
  const pb = picMap.value.get(b)
  const areaA = pa?.size ? pa.size[0] * pa.size[1] : 0
  const areaB = pb?.size ? pb.size[0] * pb.size[1] : 0
  if (areaA && areaB && Math.abs(areaA - areaB) / Math.max(areaA, areaB) > 0.08) return 'smaller'
  const sa = pa?.sharpness ?? 0
  const sb = pb?.sharpness ?? 0
  if (sa && sb && Math.abs(sa - sb) / Math.max(sa, sb) > 0.08) return 'softer'
  return 'copy'
}
const activeDuplicates = computed(() =>
  (data.value?.duplicates ?? []).filter((d) => !hideMuted.value || (!isMuted(d.a) && !isMuted(d.b))),
)
const dupDropCandidates = computed(() => {
  const out = new Set<string>()
  for (const d of data.value?.duplicates ?? []) {
    if (isMuted(d.a) || isMuted(d.b)) continue
    const w = worseSide(d.a, d.b)
    if (w) out.add(w === 'a' ? d.a : d.b)
  }
  return [...out]
})

// ---- buckets -------------------------------------------------------------------------------------
const maxBucket = computed(() => Math.max(1, ...(data.value?.buckets.rows.map((r) => r.count) ?? [1])))
const openBucket = ref<string | null>(null)
const bucketKey = (size: [number, number]) => size.join('x')
type Placed = { bucket: [number, number]; kept: number; size: [number, number] }
const asTrained = remembered('zentrainer.analysis.asTrained', false)
const wide = (c: Placed) => c.size[0] / c.size[1] > c.bucket[0] / c.bucket[1]
const frame = (c: Placed | undefined) => (c ? { aspectRatio: asTrained.value ? `${c.bucket[0]} / ${c.bucket[1]}` : `${c.size[0]} / ${c.size[1]}` } : {})
function bands(c: Placed) {
  const lost = `${((1 - c.kept) / 2) * 100}%`
  return wide(c)
    ? [{ side: 'left', style: { left: '0', top: '0', bottom: '0', width: lost } }, { side: 'right', style: { right: '0', top: '0', bottom: '0', width: lost } }]
    : [{ side: 'top', style: { top: '0', left: '0', right: '0', height: lost } }, { side: 'bottom', style: { bottom: '0', left: '0', right: '0', height: lost } }]
}
const whole = (c: Placed) => c.kept >= 0.995
const cutLabel = (c: Placed) => (whole(c) ? 'full frame' : `cuts ${pct(1 - c.kept)} ${wide(c) ? 'sides' : 'top/bottom'}`)
const placed = (path: string) => data.value?.buckets.placed[path]
function bucketKept(paths: string[]): number {
  const vals = paths.map((p) => placed(p)?.kept).filter((v): v is number => v != null)
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 1
}

// ---- distributions -------------------------------------------------------------------------------
interface Hist {
  key: string
  label: string
  note: string
  bins: number[]
  lo: number
  hi: number
  median: number | null
  warn: [number, number][]
  fmt: (x: number) => string
}
const BINS = 30
function histogram(values: number[], lo: number, hi: number): number[] {
  const bins = new Array(BINS).fill(0)
  for (const v of values) bins[Math.min(BINS - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * BINS)))]++
  return bins
}
const median = (v: number[]) => (v.length ? [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)]! : null)
const hists = computed<Hist[]>(() => {
  const ps = data.value?.pictures ?? []
  const vals = (f: (p: (typeof ps)[number]) => number | null | undefined) => ps.map(f).filter((v): v is number => v != null)
  const make = (key: string, label: string, note: string, v: number[], lo: number, hi: number, warn: [number, number][], fmt: (x: number) => string): Hist => ({
    key, label, note, lo, hi, warn, fmt, bins: histogram(v, lo, hi), median: median(v),
  })
  const sides = vals((p) => (p.size ? Math.min(...p.size) : null))
  const maxSide = Math.max(2048, Math.ceil(Math.max(0, ...sides) / 512) * 512)
  const ratio = (x: number) => (Math.abs(x) < 0.05 ? '1:1' : x > 0 ? `${(2 ** x).toFixed(1)}:1` : `1:${(2 ** -x).toFixed(1)}`)
  return [
    make('brightness', 'Brightness', 'Dark to bright. Shaded: flagged dark or blown out.', vals((p) => p.brightness), 0, 1, [[0, 0.15], [0.85, 1]], pct),
    make('contrast', 'Contrast', 'Flat to punchy. Shaded: flagged flat.', vals((p) => p.contrast), 0, 0.4, [[0, 0.06]], (x) => x.toFixed(2)),
    make('sharpness', 'Sharpness', 'Against the set’s median (1×). Shaded: flagged blurry.',
      vals((p) => (p.sharpness ? Math.log2(p.sharpness) : null)), -4, 4, [[-4, -2]], (x) => `${2 ** x < 1 ? +(2 ** x).toFixed(2) : Math.round(2 ** x)}×`),
    make('side', 'Shorter side', 'Pixels. Shaded: under 768, upscaled to train.', sides, 0, maxSide, [[0, 768]], (x) => `${Math.round(x)}`),
    make('aspect', 'Aspect ratio', 'Tall on the left, wide on the right.', vals((p) => p.aspect), -2, 2, [], ratio),
  ]
})
const histMax = (h: Hist) => Math.max(1, ...h.bins)
const at = (h: Hist, x: number) => `${((Math.min(h.hi, Math.max(h.lo, x)) - h.lo) / (h.hi - h.lo)) * 100}%`
const medianSide = computed(() => {
  const sides = (data.value?.pictures ?? []).map((p) => (p.size ? Math.min(...p.size) : null)).filter((v): v is number => v != null)
  const m = median(sides)
  return m != null ? `${Math.round(m)} px` : '—'
})

const tabCount = (id: Tab): number | null => {
  if (!data.value) return null
  if (id === 'issues') return allFlagged.value.length
  if (id === 'duplicates') return data.value.duplicates.length
  if (id === 'buckets') return data.value.buckets.rows.length
  if (id === 'captions') return data.value.captions.frequent.length || null
  return null
}
</script>

<template>
  <div class="page full analysis">
    <div class="page-head">
      <div class="titles">
        <h1>
          <RouterLink to="/datasets" class="crumb">Datasets</RouterLink>
          <RouterLink :to="`/datasets/${encodeURIComponent(name)}`" class="crumb">{{ name }}</RouterLink>
          Analysis
        </h1>
        <p v-if="data" class="lede">
          {{ data.pictures.length }} pictures · {{ data.captions.captions }} captioned
          <template v-if="muted.size"> · {{ muted.size }} muted in “{{ activeSet }}”</template>
          <i v-if="loading" class="mdi mdi-loading mdi-spin" />
        </p>
      </div>
      <div class="head-actions">
        <ZenCombo
          v-if="sets.length > 1"
          :model-value="activeSet"
          :items="sets.map((s) => ({ value: s, label: s }))"
          :searchable="false"
          :sorted="false"
          size="sm"
          title="Caption set"
          @update:model-value="activeSet = String($event)"
        />
        <RouterLink :to="{ path: '/runs/new', query: { dataset: name } }">
          <ZenButton size="sm" variant="primary" icon="mdi mdi-play">Train</ZenButton>
        </RouterLink>
        <RouterLink :to="`/datasets/${encodeURIComponent(name)}`">
          <ZenButton size="sm" icon="mdi mdi-arrow-left">Dataset</ZenButton>
        </RouterLink>
      </div>
    </div>

    <div v-if="error" class="card err small"><i class="mdi mdi-alert-circle-outline" /> {{ error }}</div>
    <div v-else-if="!data" class="card muted small"><i class="mdi mdi-loading mdi-spin" /> Looking at every picture… the first time takes a moment; after that only changed ones are read again.</div>

    <template v-if="data">
      <!-- KPI strip: one-glance summary, each card jumps straight to its section -->
      <div class="kpis">
        <button class="kpi card" :class="{ on: tab === 'issues', warn: allFlagged.length > 0 }" @click="tab = 'issues'">
          <span class="kpi-label">Issues</span>
          <b class="kpi-val mono">{{ allFlagged.length }}</b>
          <span class="kpi-sub small muted">{{ allFlagged.length ? `${flagChips.length} ${flagChips.length === 1 ? 'type' : 'types'}` : 'All clean' }}</span>
        </button>
        <button class="kpi card" :class="{ on: tab === 'duplicates', warn: data.duplicates.length > 0 }" @click="tab = 'duplicates'">
          <span class="kpi-label">Duplicates</span>
          <b class="kpi-val mono">{{ data.duplicates.length }}</b>
          <span class="kpi-sub small muted">{{ data.duplicates.length ? `${dupDropCandidates.length} unmuted` : 'No close copies' }}</span>
        </button>
        <button class="kpi card" :class="{ on: tab === 'buckets' }" @click="tab = 'buckets'">
          <span class="kpi-label">Buckets ({{ data.buckets.resolution }}px)</span>
          <b class="kpi-val mono">{{ pct(data.buckets.kept_median) }}</b>
          <span class="kpi-sub small muted">{{ data.buckets.rows.length }} buckets{{ data.buckets.lonely.length ? ` · ${data.buckets.lonely.length} lonely` : '' }}</span>
        </button>
        <button class="kpi card" :class="{ on: tab === 'captions', warn: data.captions.captions < data.pictures.length || data.captions.frequent.length > 0 }" @click="tab = 'captions'">
          <span class="kpi-label">Captions</span>
          <b class="kpi-val mono">{{ data.captions.captions }}/{{ data.pictures.length }}</b>
          <span class="kpi-sub small muted">{{ data.captions.frequent.length ? `${data.captions.frequent.length} repeated >50%` : 'Balanced' }}</span>
        </button>
        <button class="kpi card" :class="{ on: tab === 'spread' }" @click="tab = 'spread'">
          <span class="kpi-label">Median size</span>
          <b class="kpi-val mono">{{ medianSide }}</b>
          <span class="kpi-sub small muted">Shorter side</span>
        </button>
      </div>

      <!-- Focused tab bar -->
      <div class="tabbar">
        <div class="tabs">
          <button v-for="t in TABS" :key="t.id" class="tab" :class="{ on: tab === t.id }" @click="tab = t.id">
            <i class="mdi" :class="t.icon" />
            <span>{{ t.label }}</span>
            <span v-if="tabCount(t.id)" class="badge mono">{{ tabCount(t.id) }}</span>
          </button>
        </div>
        <button v-if="muted.size && (tab === 'issues' || tab === 'duplicates')" class="chip" :class="{ on: hideMuted }" @click="hideMuted = !hideMuted">
          <i class="mdi" :class="hideMuted ? 'mdi-eye-off-outline' : 'mdi-eye-outline'" />
          {{ hideMuted ? 'Hiding muted' : 'Hide muted' }}
        </button>
      </div>

      <!-- 1. Issues -->
      <section v-if="tab === 'issues'" class="card col">
        <div class="head">
          <h2>Pictures to check</h2>
          <div v-if="flagged.length" class="bulk">
            <ZenButton
              v-if="unmutedFlagged.length"
              size="sm"
              icon="mdi mdi-eye-off-outline"
              :disabled="busy"
              :title="`Keep these files, but exclude them from training in “${activeSet}”`"
              @click="setMuted(unmutedFlagged, true)"
            >
              Mute {{ unmutedFlagged.length }}
            </ZenButton>
            <ZenButton
              size="sm"
              icon="mdi mdi-trash-can-outline"
              :disabled="busy"
              @click="removePaths(flagged.map((p) => p.path))"
            >
              Delete {{ flagged.length }}…
            </ZenButton>
          </div>
        </div>
        <div v-if="flagChips.length" class="chips">
          <button class="chip" :class="{ on: !flagFilter }" @click="flagFilter = null">All <span class="muted">{{ allFlagged.length }}</span></button>
          <button v-for="[f, n] in flagChips" :key="f" class="chip" :class="{ on: flagFilter === f }" :title="FLAG_INFO[f]" @click="flagFilter = flagFilter === f ? null : f">
            {{ f }} <span class="muted">{{ n }}</span>
          </button>
        </div>
        <p v-if="flagFilter && FLAG_INFO[flagFilter]" class="muted small hint">{{ FLAG_INFO[flagFilter] }}</p>
        <ZenEmpty v-if="!flagged.length" icon="mdi mdi-check-circle-outline" title="Nothing to fix">
          {{ allFlagged.length && hideMuted ? 'Every flagged picture is muted in this set.' : 'No picture has an image issue.' }}
        </ZenEmpty>
        <div v-else class="grid">
          <figure
            v-for="p in flagged.slice(0, 120)"
            :key="p.path"
            class="tile clickable"
            :class="{ mutedTile: isMuted(p.path) }"
            @click="openLightbox(flagged.map((x) => x.path), p.path)"
            @contextmenu="onPicMenu(p.path, $event, flagged.map((x) => x.path))"
          >
            <div class="thumb-wrap">
              <img :src="thumb(p.path)" loading="lazy" :alt="p.path" />
              <span v-if="isMuted(p.path)" class="muted-pill">Muted</span>
              <div class="tile-actions" @click.stop>
                <button :title="isMuted(p.path) ? 'Unmute' : `Mute in “${activeSet}”`" :disabled="busy" @click="setMuted([p.path], !isMuted(p.path))">
                  <i class="mdi" :class="isMuted(p.path) ? 'mdi-eye-outline' : 'mdi-eye-off-outline'" />
                </button>
                <button title="Edit mask" @click="router.push({ path: `/datasets/${encodeURIComponent(name)}/mask`, query: { path: p.path } })">
                  <i class="mdi mdi-draw" />
                </button>
                <button class="danger" title="Delete picture" :disabled="busy" @click="removePaths([p.path])">
                  <i class="mdi mdi-trash-can-outline" />
                </button>
              </div>
            </div>
            <figcaption>
              <span class="ellipsis small mono" :title="p.path">{{ p.path }}</span>
              <span class="small muted mono">{{ fmtSize(p.size) }}</span>
              <span class="tags"><span v-for="f in p.flags" :key="f" class="tag" :title="FLAG_INFO[f]">{{ f }}</span></span>
            </figcaption>
          </figure>
        </div>
        <p v-if="flagged.length > 120" class="muted small">and {{ flagged.length - 120 }} more</p>
      </section>

      <!-- 2. Near-duplicates -->
      <section v-if="tab === 'duplicates'" class="card col">
        <div class="head">
          <h2>Near-duplicates</h2>
          <ZenButton
            v-if="dupDropCandidates.length"
            size="sm"
            icon="mdi mdi-eye-off-outline"
            :disabled="busy"
            title="Mute the smaller or softer copy of every active duplicate pair"
            @click="setMuted(dupDropCandidates, true)"
          >
            Mute {{ dupDropCandidates.length }} smaller/softer {{ dupDropCandidates.length === 1 ? 'copy' : 'copies' }}
          </ZenButton>
        </div>
        <p class="muted small hint">Copies of the same picture (re-saved, resized, lightly edited or mirrored) count twice in training. Mute or delete the weaker copy directly.</p>
        <ZenEmpty v-if="!activeDuplicates.length" icon="mdi mdi-check-circle-outline" title="No duplicates to resolve">
          {{ data.duplicates.length && hideMuted ? 'Every duplicate pair already has a muted copy.' : 'No two pictures are close copies.' }}
        </ZenEmpty>
        <div v-else class="pairs">
          <div v-for="d in activeDuplicates.slice(0, 40)" :key="d.a + d.b" class="pair">
            <div class="pair-head small">
              <span><b>{{ d.kind }}</b><span v-if="d.mirrored" class="muted"> · mirrored</span></span>
              <span class="muted mono">{{ Math.round(d.corr * 100) }}% match</span>
            </div>
            <div class="pair-sides">
              <div
                v-for="side in (['a', 'b'] as const)"
                :key="side"
                class="pair-item"
                :class="{ mutedTile: isMuted(d[side]) }"
                @contextmenu="onPicMenu(d[side], $event, [d.a, d.b])"
              >
                <div class="thumb-wrap clickable" @click="openLightbox([d.a, d.b], d[side])">
                  <img :src="thumb(d[side], 256)" loading="lazy" :alt="d[side]" />
                  <span v-if="isMuted(d[side])" class="muted-pill">Muted</span>
                  <span v-else-if="worseReason(side, d.a, d.b)" class="worse-pill">{{ worseReason(side, d.a, d.b) }}</span>
                </div>
                <span class="ellipsis small mono" :title="d[side]">{{ d[side] }}</span>
                <span class="small muted mono">
                  {{ fmtSize(picMap.get(d[side])?.size) }}
                  <template v-if="picMap.get(d[side])?.sharpness"> · {{ picMap.get(d[side])!.sharpness!.toFixed(2) }}×</template>
                </span>
                <div class="pair-btns">
                  <ZenButton size="sm" :disabled="busy" :icon="isMuted(d[side]) ? 'mdi mdi-eye-outline' : 'mdi mdi-eye-off-outline'" @click="setMuted([d[side]], !isMuted(d[side]))">
                    {{ isMuted(d[side]) ? 'Unmute' : 'Mute' }}
                  </ZenButton>
                  <ZenButton size="sm" icon="mdi mdi-trash-can-outline" :disabled="busy" @click="removePaths([d[side]])">Delete</ZenButton>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 3. Buckets -->
      <section v-if="tab === 'buckets'" class="card col">
        <div class="head">
          <h2>Buckets</h2>
          <ZenToggleGroup
            :model-value="asTrained ? 'trained' : 'marked'"
            :options="[{ value: 'marked', label: 'Cut parts marked' }, { value: 'trained', label: 'As trained' }]"
            size="sm"
            @update:model-value="asTrained = $event === 'trained'"
          />
          <ZenToggleGroup v-model="resolution" :options="RESOLUTIONS.map((r) => ({ value: r, label: r }))" size="sm" />
        </div>
        <p class="muted small hint">
          Each picture trains at the bucket closest to its shape (~{{ data.buckets.resolution }}² px), centre-cropped to fit.
          Half the set keeps at least <b>{{ pct(data.buckets.kept_median) }}</b>. Click any bucket to inspect its crops, or right-click a picture to mute/delete it.
        </p>
        <div class="bars">
          <template v-for="r in data.buckets.rows" :key="bucketKey(r.size)">
            <button class="bar-row bucket-row clickable" :class="{ open: openBucket === bucketKey(r.size) }" @click="openBucket = openBucket === bucketKey(r.size) ? null : bucketKey(r.size)">
              <span class="mono small label">
                <i :class="openBucket === bucketKey(r.size) ? 'mdi mdi-chevron-down' : 'mdi mdi-chevron-right'" />
                {{ r.size[0] }}×{{ r.size[1] }}
                <span v-if="r.count === 1" class="lonely-tag">lonely</span>
              </span>
              <span class="track"><span class="fill" :style="{ width: pct(r.count / maxBucket) }" /></span>
              <span class="mono small count">{{ r.count }} · {{ pct(bucketKept(r.pictures)) }}</span>
            </button>
            <div v-if="openBucket === bucketKey(r.size)" class="bucket-pics">
              <figure
                v-for="path in r.pictures.slice(0, 60)"
                :key="path"
                class="tile clickable"
                :class="{ mutedTile: isMuted(path) }"
                @click="openLightbox(r.pictures, path)"
                @contextmenu="onPicMenu(path, $event, r.pictures)"
              >
                <div class="crop" :style="frame(placed(path))">
                  <img :src="thumb(path)" loading="lazy" :alt="path" />
                  <template v-if="!asTrained && placed(path) && !whole(placed(path)!)">
                    <span v-for="b in bands(placed(path)!)" :key="b.side" class="cut" :class="b.side" :style="b.style" />
                  </template>
                </div>
                <figcaption>
                  <span class="ellipsis small mono" :title="path">{{ path }}</span>
                  <span v-if="placed(path)" class="small" :class="{ warn: placed(path)!.kept < 0.85 }">{{ cutLabel(placed(path)!) }}</span>
                </figcaption>
              </figure>
              <p v-if="r.pictures.length > 60" class="muted small">and {{ r.pictures.length - 60 }} more</p>
            </div>
          </template>
        </div>
        <p v-if="data.buckets.lonely.length" class="warn small">
          <i class="mdi mdi-account-alert-outline" /> {{ data.buckets.lonely.length }} {{ data.buckets.lonely.length === 1 ? 'picture is' : 'pictures are' }} alone in a bucket:
          with batch size &gt; 1 it can't share a batch.
        </p>
        <template v-if="data.buckets.cropped.length">
          <div class="eyebrow">Cropped the most (&gt;15% lost)</div>
          <div class="grid">
            <figure
              v-for="c in data.buckets.cropped.slice(0, 24)"
              :key="c.path"
              class="tile clickable"
              :class="{ mutedTile: isMuted(c.path) }"
              @click="openLightbox(data.buckets.cropped.map((x) => x.path), c.path)"
              @contextmenu="onPicMenu(c.path, $event, data.buckets.cropped.map((x) => x.path))"
            >
              <div class="crop" :style="frame(c)">
                <img :src="thumb(c.path)" loading="lazy" :alt="c.path" />
                <template v-if="!asTrained">
                  <span v-for="b in bands(c)" :key="b.side" class="cut" :class="b.side" :style="b.style" />
                </template>
              </div>
              <figcaption>
                <span class="ellipsis small mono" :title="c.path">{{ c.path }}</span>
                <span class="small"><b>{{ cutLabel(c) }}</b> · {{ c.bucket[0] }}×{{ c.bucket[1] }}</span>
              </figcaption>
            </figure>
          </div>
        </template>
      </section>

      <!-- 4. Captions -->
      <section v-if="tab === 'captions'" class="card col">
        <div class="head">
          <h2>Captions</h2>
          <span v-if="data.captions.style" class="muted small">
            read as {{ data.captions.style === 'tags' ? 'tags' : 'sentences' }}<template v-if="data.captions.subject">, leaving out “{{ data.captions.subject }}”</template>
          </span>
        </div>
        <ZenEmpty v-if="!data.captions.captions" icon="mdi mdi-text-box-remove-outline" title="No captions">Caption the set to see what it repeats.</ZenEmpty>
        <template v-else>
          <div v-if="data.captions.frequent.length" class="frequent">
            <i class="mdi mdi-alert-outline" />
            <div>
              <b>Most of the set has these:</b>&nbsp;<span v-for="(f, i) in data.captions.frequent" :key="f.text"><template v-if="i">, </template>“{{ f.text }}” ({{ pct(f.share) }})</span>.
              <div class="muted small">
                The LoRA will lean towards them. If one isn't part of what you're training, add pictures without it, or vary it.
              </div>
            </div>
          </div>
          <div class="bars">
            <div v-for="p in data.captions.phrases" :key="p.text" class="bar-row">
              <span class="small label ellipsis" :title="p.text">{{ p.text }}</span>
              <span class="track"><span class="fill" :class="{ hot: p.share > 0.5 }" :style="{ width: pct(p.share) }" /></span>
              <span class="mono small count">{{ pct(p.share) }}</span>
            </div>
          </div>
        </template>
      </section>

      <!-- 5. Spread -->
      <section v-if="tab === 'spread'" class="card col">
        <div class="head"><h2>Spread</h2></div>
        <div class="hists">
          <div v-for="h in hists" :key="h.key" class="hist">
            <div class="hist-head">
              <b>{{ h.label }}</b>
              <span v-if="h.median != null" class="mono small muted">median {{ h.fmt(h.median) }}</span>
            </div>
            <div class="plot">
              <span v-for="(w, i) in h.warn" :key="i" class="zone" :style="{ left: at(h, w[0]), width: `calc(${at(h, w[1])} - ${at(h, w[0])})` }" />
              <div class="hbars">
                <span v-for="(b, i) in h.bins" :key="i" class="hb" :style="{ height: b ? `${Math.max(4, (b / histMax(h)) * 100)}%` : '0' }" :title="`${b} pictures`" />
              </div>
              <span v-if="h.median != null" class="med" :style="{ left: at(h, h.median) }" />
            </div>
            <div class="axis small muted mono"><span>{{ h.fmt(h.lo) }}</span><span>{{ h.fmt((h.lo + h.hi) / 2) }}</span><span>{{ h.fmt(h.hi) }}</span></div>
            <p class="note small muted">{{ h.note }}</p>
          </div>
        </div>
      </section>
    </template>

    <div v-if="notice" class="float-notice zen-surface small">{{ notice }}</div>
    <ZenContextMenu ref="menu" :items="menuItems" />
    <ZenLightbox v-if="lightbox != null && lbItems.length" v-model:index="lightbox" v-model:thumb-fit="lbFit" :items="lbItems" :searchable="false" @close="lightbox = null">
      <template #side="{ index }">
        <div v-if="lbPaths[index]" class="lb-side">
          <b class="mono small ellipsis" :title="lbPaths[index]">{{ lbPaths[index] }}</b>
          <div class="small muted mono">{{ fmtSize(picMap.get(lbPaths[index]!)?.size) }}</div>
          <div v-if="picMap.get(lbPaths[index]!)?.flags.length" class="tags">
            <span v-for="f in picMap.get(lbPaths[index]!)!.flags" :key="f" class="tag" :title="FLAG_INFO[f]">{{ f }}</span>
          </div>
          <div class="lb-actions">
            <ZenButton size="sm" :disabled="busy" :icon="isMuted(lbPaths[index]!) ? 'mdi mdi-eye-outline' : 'mdi mdi-eye-off-outline'" @click="setMuted([lbPaths[index]!], !isMuted(lbPaths[index]!))">
              {{ isMuted(lbPaths[index]!) ? 'Unmute' : `Mute in “${activeSet}”` }}
            </ZenButton>
            <ZenButton size="sm" icon="mdi mdi-draw" @click="router.push({ path: `/datasets/${encodeURIComponent(name)}/mask`, query: { path: lbPaths[index]! } })">
              Edit mask
            </ZenButton>
            <ZenButton size="sm" icon="mdi mdi-trash-can-outline" :disabled="busy" @click="removePaths([lbPaths[index]!])">
              Delete…
            </ZenButton>
          </div>
        </div>
      </template>
    </ZenLightbox>
  </div>
</template>

<style scoped>
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
  flex-wrap: wrap;
}
.head-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.col {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}
.head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.head h2 {
  margin: 0;
  font-size: 15px;
  flex: 1;
}
.bulk {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.hint {
  margin: 0;
}
.err {
  color: var(--zen-danger);
}
.warn {
  color: var(--zen-warn);
  margin: 0;
}

/* ---- KPI strip & tab bar ------------------------------------------------------------------------ */
.kpis {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}
.kpi {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding: 10px 12px;
  text-align: left;
  color: var(--zen-text);
  font: inherit;
  cursor: pointer;
  transition: border-color var(--zen-dur-fast) ease, background var(--zen-dur-fast) ease;
}
.kpi:hover {
  border-color: var(--zen-accent);
}
.kpi.on {
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 8%, var(--zen-surface));
}
.kpi-label {
  font-size: 11px;
  color: var(--zen-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.kpi-val {
  font-size: 20px;
  line-height: 1.2;
}
.kpi.warn .kpi-val {
  color: var(--zen-warn);
}
.tabbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}
.tabs {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.tab {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 11px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md, 8px);
  background: var(--zen-surface);
  color: var(--zen-muted);
  font: inherit;
  font-size: 12.5px;
  cursor: pointer;
}
.tab:hover {
  color: var(--zen-text);
  border-color: color-mix(in srgb, var(--zen-text) 25%, var(--zen-border));
}
.tab.on {
  color: var(--zen-text);
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 12%, var(--zen-surface));
}
.badge {
  font-size: 11px;
  padding: 0 6px;
  border-radius: 99px;
  background: var(--zen-surface-2);
  color: var(--zen-text);
}

/* ---- chips and picture tiles -------------------------------------------------------------------- */
.chips {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--zen-border);
  background: none;
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  padding: 3px 10px;
  border-radius: 99px;
  cursor: pointer;
}
.chip:hover {
  background: var(--zen-surface-2);
}
.chip.on {
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 12%, transparent);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(156px, 1fr));
  gap: 10px;
}
.tile {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.tile.clickable {
  cursor: pointer;
}
.mutedTile {
  opacity: 0.55;
}
.thumb-wrap {
  position: relative;
  border-radius: var(--r-sm);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.tile img,
.pair-item img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: contain;
  background: var(--zen-surface-2);
  border-radius: var(--r-sm);
  display: block;
}
.muted-pill,
.worse-pill {
  position: absolute;
  top: 6px;
  left: 6px;
  font-size: 10.5px;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--zen-bg) 82%, transparent);
  color: var(--zen-muted);
}
.worse-pill {
  color: var(--zen-warn);
}
.tile-actions {
  position: absolute;
  top: 6px;
  right: 6px;
  display: flex;
  gap: 3px;
  opacity: 0;
  transition: opacity var(--zen-dur-fast) ease;
}
.tile:hover .tile-actions,
.tile:focus-within .tile-actions {
  opacity: 1;
}
.tile-actions button {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-bg) 84%, transparent);
  color: var(--zen-text);
  font-size: 13px;
  cursor: pointer;
}
.tile-actions button:hover {
  border-color: var(--zen-accent);
  color: var(--zen-accent);
}
.tile-actions button.danger:hover {
  border-color: var(--zen-danger);
  color: var(--zen-danger);
}
.tile figcaption {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.tags {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.tag {
  font-size: 10.5px;
  padding: 0 6px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--zen-warn) 15%, transparent);
  color: var(--zen-warn);
}
.crop {
  position: relative;
  border-radius: var(--r-sm);
  overflow: hidden;
}
.crop img {
  width: 100%;
  height: 100%;
  aspect-ratio: auto;
  object-fit: cover;
  border-radius: 0;
}
.cut {
  position: absolute;
  pointer-events: none;
  background: repeating-linear-gradient(
    45deg,
    color-mix(in srgb, var(--zen-warn) 70%, transparent) 0 3px,
    color-mix(in srgb, black 45%, transparent) 3px 7px
  );
}
.cut.left { border-right: 2px dashed var(--zen-warn); }
.cut.right { border-left: 2px dashed var(--zen-warn); }
.cut.top { border-bottom: 2px dashed var(--zen-warn); }
.cut.bottom { border-top: 2px dashed var(--zen-warn); }

/* ---- bars & buckets ----------------------------------------------------------------------------- */
.bars {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.bar-row {
  display: grid;
  grid-template-columns: minmax(90px, 220px) minmax(0, 1fr) 54px;
  gap: 10px;
  align-items: center;
}
.bucket-row {
  grid-template-columns: minmax(110px, 200px) minmax(0, 1fr) 86px;
}
.bar-row.clickable {
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  padding: 4px 6px;
  border-radius: var(--r-sm);
  text-align: left;
  cursor: pointer;
}
.bar-row.clickable:hover,
.bar-row.open {
  background: var(--zen-surface-2);
}
.lonely-tag {
  margin-left: 6px;
  font-size: 10px;
  padding: 0 5px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--zen-warn) 15%, transparent);
  color: var(--zen-warn);
}
.label {
  min-width: 0;
}
.track {
  height: 8px;
  border-radius: 99px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.fill {
  display: block;
  height: 100%;
  border-radius: 99px;
  background: color-mix(in srgb, var(--zen-text) 35%, transparent);
}
.fill.hot {
  background: var(--zen-warn);
}
.count {
  text-align: right;
  color: var(--zen-muted);
  white-space: nowrap;
}
.bucket-pics {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px;
  padding: 8px 6px 12px;
}
.eyebrow {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--zen-muted);
  margin-top: 6px;
}
.frequent {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-warn) 40%, transparent);
  background: color-mix(in srgb, var(--zen-warn) 8%, transparent);
  font-size: 13px;
}
.frequent > i {
  color: var(--zen-warn);
  font-size: 18px;
}

/* ---- histograms --------------------------------------------------------------------------------- */
.hists {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 22px 28px;
}
.hist {
  display: grid;
  grid-template-rows: 20px 84px auto auto;
  gap: 6px;
  min-width: 0;
}
.hist-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  font-size: 13px;
  white-space: nowrap;
}
.plot {
  position: relative;
  border-bottom: 1px solid var(--zen-border);
}
.zone {
  position: absolute;
  top: 0;
  bottom: 0;
  background: color-mix(in srgb, var(--zen-warn) 12%, transparent);
  pointer-events: none;
}
.hbars {
  position: relative;
  height: 100%;
  display: flex;
  align-items: flex-end;
  gap: 1px;
}
.hb {
  flex: 1;
  background: color-mix(in srgb, var(--zen-text) 45%, transparent);
  border-radius: 1px 1px 0 0;
}
.med {
  position: absolute;
  top: 0;
  bottom: -2px;
  width: 2px;
  background: var(--zen-accent);
  pointer-events: none;
}
.axis {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
}
.note {
  margin: 0;
}

/* ---- duplicates --------------------------------------------------------------------------------- */
.pairs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 12px;
}
.pair {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  background: var(--zen-surface);
}
.pair-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}
.pair-sides {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.pair-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.pair-btns {
  display: flex;
  gap: 4px;
  margin-top: 2px;
  flex-wrap: wrap;
}
.clickable {
  cursor: pointer;
}

/* ---- lightbox side & toast ---------------------------------------------------------------------- */
.lb-side {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
}
.lb-actions {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.float-notice {
  position: fixed;
  bottom: 18px;
  left: 50%;
  transform: translateX(-50%);
  padding: 7px 14px;
  border: 1px solid var(--zen-border);
  border-radius: 99px;
  z-index: 60;
}

@media (max-width: 640px) {
  .kpis {
    grid-template-columns: repeat(2, 1fr);
  }
  .pairs {
    grid-template-columns: 1fr;
  }
  .tile-actions {
    opacity: 1;
  }
}
</style>
