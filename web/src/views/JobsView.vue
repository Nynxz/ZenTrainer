<script setup lang="ts">
// Everything that runs, on one page: training runs with their queue (what's training now,
// what's next, still reorderable: those haven't been handed to ComfyUI yet) and caption jobs;
// then everything that has finished, newest first. Each opens its own page with its log.
import { ZenButton, ZenEmpty, ZenInput } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { engine, type CaptionJob, type RunMeta } from '@/api/engine'
import StatusPill from '@/components/StatusPill.vue'
import StickyBar from '@/components/StickyBar.vue'
import { activeCaptions, captionJobs, startCaptionSync } from '@/stores/captions'
import { engineError, latest, queue, refreshQueue, refreshRuns, runs } from '@/stores/runs'

const captions = ref<CaptionJob[]>([])
async function refreshCaptions() {
  try {
    captions.value = await engine.captionJobs()
  } catch {
    captions.value = []
  }
}
onMounted(() => Promise.all([refreshRuns(), refreshQueue(), refreshCaptions(), startCaptionSync()]))
// live caption events: merge them in, and pick up jobs started elsewhere
watch(
  () => Object.values(captionJobs).map((j) => `${j.job_id}:${j.status}:${j.done}`).join(),
  () => {
    const byId = new Map(captions.value.map((j) => [j.job_id, j]))
    for (const j of Object.values(captionJobs)) byId.set(j.job_id, { ...byId.get(j.job_id), ...j })
    captions.value = [...byId.values()].sort((a, b) => (b.created ?? 0) - (a.created ?? 0))
  },
)

type Kind = 'all' | 'train' | 'caption'
const kind = ref<Kind>('all')

async function act(body: Parameters<typeof engine.editQueue>[0]) {
  await engine.editQueue(body)
  await Promise.all([refreshQueue(), refreshRuns()])
}
function move(i: number, by: number) {
  const order = queue.value.items.map((r) => r.run_id)
  const j = i + by
  if (j < 0 || j >= order.length) return
  ;[order[i], order[j]] = [order[j]!, order[i]!]
  act({ action: 'order', order })
}

const current = computed(() => queue.value.current)
const queued = computed(() => queue.value.items)
type Row = { kind: 'train'; t: number; run: RunMeta } | { kind: 'caption'; t: number; job: CaptionJob }
// two finished runs ticked in History open side by side on the compare page; a third tick replaces
// the oldest
const router = useRouter()
const picked = ref<string[]>([])
function pick(id: string) {
  picked.value = picked.value.includes(id) ? picked.value.filter((x) => x !== id) : [...picked.value, id].slice(-2)
}
const pickedName = (id: string) => runs.value.find((r) => r.run_id === id)?.name ?? id
const compare = () => router.push({ path: '/compare', query: { a: picked.value[0], b: picked.value[1] } })

// the history's search: a run by its name or id, a caption job by what it captioned
const query = ref('')
const matches = (...texts: (string | undefined)[]) => {
  const q = query.value.trim().toLowerCase()
  return !q || texts.some((t) => t?.toLowerCase().includes(q))
}
const history = computed<Row[]>(() => {
  const live = new Set([current.value?.run_id, ...queued.value.map((r) => r.run_id)])
  const rows: Row[] = []
  if (kind.value !== 'caption')
    for (const r of runs.value) if (!live.has(r.run_id) && matches(r.name, r.run_id)) rows.push({ kind: 'train', t: r.created, run: r })
  if (kind.value !== 'train')
    for (const j of captions.value)
      if (!['queued', 'running'].includes(j.status) && matches(capTitle(j), j.dataset, j.set_name)) rows.push({ kind: 'caption', t: j.created ?? 0, job: j })
  return rows.sort((a, b) => b.t - a.t)
})
const liveCaptions = computed(() => (kind.value === 'train' ? [] : activeCaptions.value))
const capTitle = (j: CaptionJob) => (j.rewrite?.length ? 'Rewrite prompts' : `${j.preview ? 'Try captions' : 'Caption'} ${j.dataset}`)
const capMeta = (j: CaptionJob) =>
  j.status === 'queued' ? j.message || 'waiting' : j.message?.startsWith('loading') ? 'loading model' : `${j.done}/${j.total || j.preview || '…'}`
const capPct = (j: CaptionJob) => ((j.total || j.preview) ? (j.done / (j.total || j.preview || 1)) * 100 : 0)
const capSub = (j: CaptionJob) =>
  [j.rewrite?.length ? `${j.rewrite.length} prompts` : `into “${j.set_name}”`, j.preview ? 'try-on' : `${j.written?.length ?? j.done} written`, j.clip_name.replace(/\.safetensors$/, '')].join(' · ')
const progress = (id: string) => {
  const m = latest[id]
  return m?.total ? { step: m.step, total: m.total, pct: (m.step / m.total) * 100 } : null
}
const when = (t: number) => new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
const clock = (t: number) => new Date(t * 1000).toLocaleTimeString([], { timeStyle: 'short' })
// "today 17:03", "yesterday 09:40", "21 Oct 17:03"
const day = (t: number) => {
  const d = new Date(t * 1000)
  const days = Math.round((new Date().setHours(0, 0, 0, 0) - new Date(d).setHours(0, 0, 0, 0)) / 86400000)
  const date = days === 0 ? 'today' : days === 1 ? 'yesterday' : d.toLocaleDateString([], { day: 'numeric', month: 'short' })
  return `${date} ${clock(t)}`
}
// "42s", "12m", "1h 05m"
const dur = (s: number) => {
  const m = Math.floor(s / 60)
  return m < 1 ? `${Math.round(s)}s` : m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`
}
const ENDED: Record<string, { verb: string; icon: string; tone: string }> = {
  done: { verb: 'Finished', icon: 'mdi-flag-checkered', tone: 'ok' },
  stopped: { verb: 'Stopped', icon: 'mdi-stop-circle-outline', tone: '' },
  error: { verb: 'Failed', icon: 'mdi-alert-circle-outline', tone: 'bad' },
  interrupted: { verb: 'Interrupted', icon: 'mdi-power-plug-off-outline', tone: 'bad' },
  rejected: { verb: 'Rejected', icon: 'mdi-cancel', tone: 'bad' },
}
// a run's go as icons: when it started, how long it ran, how and when it ended (the full dates on
// hover); runs from before start times were kept show when they were made
type Part = { icon: string; text: string; title: string; tone?: string }
const timing = (r: RunMeta): Part[] => {
  if (!r.started) return [{ icon: 'mdi-calendar-outline', text: day(r.created), title: `Made ${when(r.created)}` }]
  const e = ENDED[r.status]
  const end = r.ended ?? (e ? r.updated : Date.now() / 1000)
  const parts: Part[] = [
    { icon: 'mdi-play-outline', text: day(r.started), title: `Started ${when(r.started)}` },
    { icon: 'mdi-timer-outline', text: dur(end - r.started), title: e ? `Ran for ${dur(end - r.started)}` : `Running for ${dur(end - r.started)}` },
  ]
  if (e && r.ended) parts.push({ icon: e.icon, text: clock(r.ended), title: `${e.verb} ${when(r.ended)}`, tone: e.tone })
  return parts
}
const summary = computed(() => {
  const parts = [`${runs.value.length} run${runs.value.length === 1 ? '' : 's'}`, `${captions.value.filter((j) => !j.preview).length} caption jobs`]
  if (current.value) parts.push('1 training')
  if (activeCaptions.value.length) parts.push(`${activeCaptions.value.length} captioning`)
  if (queued.value.length) parts.push(`${queued.value.length} queued`)
  return parts.join(' · ')
})
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1>Jobs</h1>
        <p class="lede">
          {{ summary }}
          <template v-if="queue.paused"> · <b class="warn">queue paused</b></template>
        </p>
      </div>
      <ZenButton
        v-if="queue.paused || queued.length"
        :variant="queue.paused ? 'primary' : 'default'"
        :icon="queue.paused ? 'mdi mdi-play' : 'mdi mdi-pause'"
        :title="queue.paused ? 'Let queued runs start again' : 'Let the current run finish, then hold the rest'"
        @click="act({ action: queue.paused ? 'resume' : 'pause' })"
        >{{ queue.paused ? 'Resume queue' : 'Pause queue' }}</ZenButton
      >
      <RouterLink to="/compare"><ZenButton icon="mdi mdi-compare-horizontal" title="Two runs side by side: setup, curves and samples. Or tick two in History">Compare runs</ZenButton></RouterLink>
      <RouterLink to="/runs/new"><ZenButton variant="primary" icon="mdi mdi-plus">New run</ZenButton></RouterLink>
    </div>
    <div v-if="engineError" class="card muted">Can't reach the engine: {{ engineError }}</div>
    <ZenEmpty v-else-if="!runs.length && !captions.length" title="Nothing has run yet" icon="mdi mdi-chart-line">
      Start a training run from <RouterLink to="/runs/new"><u>New run</u></RouterLink>, or auto-caption a dataset.
    </ZenEmpty>

    <template v-else>
      <StickyBar>
        <div class="page-toolbar card">
          <div class="view-switch text">
            <button :class="{ on: kind === 'all' }" @click="kind = 'all'">All</button>
            <button :class="{ on: kind === 'train' }" @click="kind = 'train'"><i class="mdi mdi-chart-timeline-variant" /> Training</button>
            <button :class="{ on: kind === 'caption' }" @click="kind = 'caption'"><i class="mdi mdi-creation-outline" /> Captioning</button>
          </div>
          <span class="spacer" />
          <div class="search"><ZenInput v-model="query" placeholder="Search runs and caption jobs" size="sm" /></div>
        </div>
      </StickyBar>
      <section v-if="liveCaptions.length || (current && kind !== 'caption')" class="sec">
        <div class="eyebrow">Now</div>
        <RouterLink v-for="j in liveCaptions" :key="j.job_id" :to="`/jobs/caption/${j.job_id}`" class="row card now">
          <div class="name">
            <i class="mdi mdi-creation-outline kind-ic" />{{ capTitle(j) }}
            <div class="muted mono small">{{ capSub(j) }}</div>
          </div>
          <div class="mini">
            <div class="bar"><div :style="{ width: `${capPct(j)}%` }" /></div>
            <span class="muted mono small">{{ capMeta(j) }}</span>
          </div>
          <StatusPill :status="j.status" :label="j.status === 'running' ? 'captioning' : undefined" />
        </RouterLink>
      </section>
      <section v-if="current && kind !== 'caption'" class="sec">
        <RouterLink :to="`/runs/${current.run_id}`" class="row card now">
          <div class="name">
            {{ current.name }}
            <div class="muted mono small">{{ current.run_id }}<template v-if="current.started"> · <i class="mdi mdi-timer-outline" /> {{ dur(Date.now() / 1000 - current.started) }} since {{ clock(current.started) }}</template></div>
          </div>
          <div class="mini">
            <div class="bar"><div :style="{ width: `${progress(current.run_id)?.pct ?? 0}%` }" /></div>
            <span class="muted mono small">{{ progress(current.run_id) ? `${progress(current.run_id)!.step}/${progress(current.run_id)!.total}` : 'starting' }}</span>
          </div>
          <StatusPill :status="current.status" />
        </RouterLink>
      </section>

      <section v-if="queued.length && kind !== 'caption'" class="sec">
        <div class="eyebrow">Up next · {{ queued.length }}</div>
        <div v-for="(r, i) in queued" :key="r.run_id" class="row card queued">
          <RouterLink :to="`/runs/${r.run_id}`" class="name">
            <span class="pos mono">{{ i + 1 }}</span>
            <span>
              {{ r.name }}
              <span class="muted mono small block">{{ r.run_id }} · {{ r.resume_from ? 'resumes' : 'new' }}</span>
            </span>
          </RouterLink>
          <div class="acts">
            <button
              v-if="current"
              :title="`Swap to this: train it now. ${current.name} stops (saving its state) and goes back in the queue right after it, to resume from there${queue.paused ? '. The queue stays paused for the rest' : ''}`"
              @click="act({ action: 'swap', run_id: r.run_id })"
            >
              <i class="mdi mdi-swap-horizontal" />
            </button>
            <RouterLink class="edit" :to="`/runs/new?edit=${r.run_id}`" title="Change its settings or name; it keeps its place"><i class="mdi mdi-pencil-outline" /></RouterLink>
            <button title="Move to top" :disabled="i === 0" @click="act({ action: 'top', run_id: r.run_id })"><i class="mdi mdi-format-vertical-align-top" /></button>
            <button title="Move up" :disabled="i === 0" @click="move(i, -1)"><i class="mdi mdi-chevron-up" /></button>
            <button title="Move down" :disabled="i === queued.length - 1" @click="move(i, 1)"><i class="mdi mdi-chevron-down" /></button>
            <button class="danger" title="Remove from the queue (the run stays, unstarted)" @click="act({ action: 'remove', run_id: r.run_id })"><i class="mdi mdi-close" /></button>
          </div>
          <StatusPill :status="r.status" />
        </div>
      </section>

      <p v-if="query.trim() && !history.length" class="muted small">{{ current || queued.length || liveCaptions.length ? 'Nothing in the history matches' : 'No runs or caption jobs match' }} “{{ query.trim() }}”.</p>
      <section v-if="history.length" class="sec">
        <div v-if="current || queued.length || liveCaptions.length" class="eyebrow">History</div>
        <template v-for="h in history" :key="h.kind === 'train' ? h.run.run_id : h.job.job_id">
          <RouterLink v-if="h.kind === 'train'" :to="`/runs/${h.run.run_id}`" class="row card" :class="{ picked: picked.includes(h.run.run_id) }">
            <div class="name">
              <button
                class="tick"
                :title="picked.includes(h.run.run_id) ? 'Untick' : 'Tick two runs to compare them'"
                @click.prevent.stop="pick(h.run.run_id)"
              >
                <i :class="picked.includes(h.run.run_id) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" />
              </button>
              <i class="mdi mdi-chart-timeline-variant kind-ic" />{{ h.run.name }}
              <div class="muted mono small">{{ h.run.run_id }}</div>
            </div>
            <div class="times small">
              <span v-for="(t, ti) in timing(h.run)" :key="ti" :class="t.tone" :title="t.title"><i class="mdi" :class="t.icon" />{{ t.text }}</span>
            </div>
            <StatusPill :status="h.run.status" />
          </RouterLink>
          <RouterLink v-else :to="`/jobs/caption/${h.job.job_id}`" class="row card">
            <div class="name">
              <i class="mdi mdi-creation-outline kind-ic" />{{ capTitle(h.job) }}
              <div class="muted mono small">{{ capSub(h.job) }}</div>
            </div>
            <div class="times small"><span :title="when(h.t)"><i class="mdi mdi-calendar-outline" />{{ day(h.t) }}</span></div>
            <StatusPill :status="h.job.status" />
          </RouterLink>
        </template>
      </section>

      <div v-if="picked.length" class="compare-bar card">
        <i class="mdi mdi-compare-horizontal" />
        <span class="small">
          <b>{{ pickedName(picked[0]!) }}</b>
          <template v-if="picked[1]"> and <b>{{ pickedName(picked[1]) }}</b></template>
          <span v-else class="muted"> · tick one more</span>
        </span>
        <span class="spacer" />
        <button class="link small" @click="picked = []">Clear</button>
        <ZenButton variant="primary" icon="mdi mdi-arrow-right" :disabled="picked.length < 2" @click="compare">Compare</ZenButton>
      </div>
    </template>
  </div>
</template>

<style scoped>
.page-toolbar .search {
  flex: 0 1 300px;
}
.kind-ic {
  color: var(--zen-muted);
  margin-right: 8px;
  font-weight: 400;
}
.sec {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.warn {
  color: var(--zen-warn);
}
/* when a run ran: icons and short values in one line, the full dates on hover */
.times {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 4px 12px;
  color: var(--zen-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.times i {
  margin-right: 4px;
  opacity: 0.8;
}
.times .ok {
  color: var(--zen-ok);
}
.times .bad {
  color: var(--zen-danger);
}
.row {
  display: grid;
  grid-template-columns: 1fr minmax(0, auto) 110px;
  gap: 16px;
  align-items: center;
  padding: 12px 16px;
}
a.row:hover,
.queued a.name:hover {
  border-color: var(--zen-accent);
  color: var(--zen-accent);
}
a.row:hover {
  color: inherit;
}
.tick {
  border: none;
  background: none;
  padding: 0 6px 0 0;
  font-size: 16px;
  line-height: 1;
  color: var(--zen-muted);
  cursor: pointer;
  opacity: 0.35;
  vertical-align: -2px;
}
.row:hover .tick,
.row.picked .tick,
.tick:focus-visible {
  opacity: 1;
}
.row.picked {
  border-color: var(--zen-accent);
}
.row.picked .tick {
  color: var(--zen-accent);
}
.compare-bar {
  position: sticky;
  bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
}
.compare-bar .spacer {
  flex: 1;
}
.compare-bar .link {
  border: none;
  background: none;
  color: var(--zen-accent);
  font: inherit;
  cursor: pointer;
}
.now {
  border-color: color-mix(in srgb, var(--zen-accent) 50%, transparent);
}
.name {
  font-weight: 600;
  min-width: 0;
}
.queued .name {
  display: flex;
  align-items: center;
  gap: 12px;
}
.pos {
  width: 22px;
  flex: none;
  text-align: center;
  color: var(--zen-muted);
  font-weight: 400;
}
.block {
  display: block;
}
.small {
  font-size: 11px;
  font-weight: 400;
}
.mini {
  display: flex;
  align-items: center;
  gap: 8px;
}
.bar {
  flex: 1;
  height: 4px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.bar div {
  height: 100%;
  background: var(--zen-accent);
  transition: width var(--zen-dur, 0.3s) ease;
}
.acts {
  display: flex;
  gap: 2px;
  justify-content: flex-end;
}
.acts button,
.acts .edit {
  display: inline-grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font-size: 16px;
  cursor: pointer;
}
.acts .edit:hover,
.acts button:hover:not(:disabled) {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.acts button.danger:hover {
  color: var(--zen-danger);
}
.acts button:disabled {
  opacity: 0.3;
  cursor: default;
}
@media (max-width: 720px) {
  .row {
    grid-template-columns: 1fr auto;
  }
  .row > :nth-child(2) {
    display: none;
  }
}
</style>
