<script setup lang="ts">
// One caption job: what it was asked to do, its progress, every caption it wrote, and its log.
// Live while it runs (the websocket's caption events trigger a re-read).
import { ZenButton } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { EngineError, engine, thumbUrl, type CaptionJob } from '@/api/engine'
import StatusPill from '@/components/StatusPill.vue'
import { captionJobs, startCaptionSync } from '@/stores/captions'

const props = defineProps<{ id: string }>()
const job = ref<CaptionJob | null>(null)
const error = ref<string | null>(null)

async function load() {
  try {
    job.value = await engine.captionJob(props.id)
    error.value = null
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  }
}
onMounted(() => Promise.all([load(), startCaptionSync()]))
// re-read on its events, at most every 800 ms
let pending: ReturnType<typeof setTimeout> | undefined
watch(
  () => {
    const j = captionJobs[props.id]
    return j ? `${j.status}:${j.done}:${j.message}` : ''
  },
  (v) => {
    if (!v || pending) return
    pending = setTimeout(() => {
      pending = undefined
      load()
    }, 800)
  },
)

const active = computed(() => !!job.value && ['queued', 'running'].includes(job.value.status))
const total = computed(() => job.value?.total || job.value?.preview || 0)
const pct = computed(() => (total.value ? (job.value!.done / total.value) * 100 : 0))
const rewrite = computed(() => !!job.value?.rewrite?.length)
const title = computed(() =>
  !job.value ? 'Caption job' : rewrite.value ? 'Rewrite prompts' : `${job.value.preview ? 'Try captions' : 'Caption'} ${job.value.dataset}`,
)
const written = computed(() => {
  const j = job.value
  if (!j) return []
  if (j.preview) return (j.results ?? []).map((r) => ({ path: r.path, text: r.text, prompt: r.prompt }))
  return Object.entries(j.captions ?? {}).map(([path, text]) => ({ path, text, prompt: undefined as string | undefined }))
})

// a caption shown readably: a structured (JSON) one by its short_description, the whole of it
// pretty-printed on demand; long ones clamped until opened
function asJson(t: string): Record<string, unknown> | null {
  const s = t.trim()
  if (!s.startsWith('{')) return null
  try {
    const o = JSON.parse(s)
    return o && typeof o === 'object' && !Array.isArray(o) ? o : null
  } catch {
    return null
  }
}
const openKeys = ref(new Set<string>())
const isOpen = (k: string) => openKeys.value.has(k)
function toggle(k: string) {
  const n = new Set(openKeys.value)
  n.has(k) ? n.delete(k) : n.add(k)
  openKeys.value = n
}
const keyOf = (w: { path: string; prompt?: string }, i: number) => w.path || `p${i}`
const facts = computed(() => {
  const j = job.value
  if (!j) return []
  return [
    ...(rewrite.value
      ? [['Prompts', String(j.rewrite!.length)]]
      : [
          ['Dataset', j.dataset],
          ['Into set', j.set_name],
          ['Images', j.preview ? `try-on of ${j.preview}` : j.mode ?? 'missing'],
          ['Replaced', j.replaced ? `${j.replaced} existing (undo puts them back)` : 'none'],
        ]),
    ['Model', j.clip_name],
    ['Kept as', j.format === 'fibo_json' ? 'Fibo structured JSON' : 'text'],
    ...(j.subject ? [['Subject', j.subject]] : []),
  ] as [string, string][]
})
const when = (t?: number | null) => (t ? new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' }) : '')
const clock = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
const took = computed(() => {
  const j = job.value
  if (!j?.created || !j.finished) return ''
  const s = Math.round(j.finished - j.created)
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`
})

async function act(action: 'cancel' | 'undo') {
  if (!job.value) return
  if (action === 'undo' && !confirm(`Undo this job? ${written.value.length} caption(s) in “${job.value.set_name}” go back to how they were (any you edited since stay).`)) return
  try {
    await engine.captionAction(job.value.job_id, action)
    await load()
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  }
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/jobs" class="crumb">Jobs</RouterLink> {{ title }}</h1>
        <p v-if="job" class="lede">
          {{ when(job.created) }}<template v-if="took"> · took {{ took }}</template>
        </p>
      </div>
      <template v-if="job">
        <StatusPill :status="job.status" :label="job.status === 'running' ? 'captioning' : undefined" />
        <RouterLink v-if="!rewrite" :to="`/datasets/${encodeURIComponent(job.dataset)}${!job.preview && written.length ? `?review=${job.job_id}` : ''}`">
          <ZenButton icon="mdi mdi-image-multiple-outline">{{ !job.preview && written.length ? 'Review in dataset' : 'Open dataset' }}</ZenButton>
        </RouterLink>
        <ZenButton v-if="active" icon="mdi mdi-stop" title="Stop at the next image; what's written stays (and can be undone)" @click="act('cancel')">Cancel</ZenButton>
        <ZenButton v-else-if="!job.preview && job.status !== 'undone' && written.length" icon="mdi mdi-undo-variant" @click="act('undo')">Undo</ZenButton>
      </template>
    </div>
    <div v-if="error" class="card err small mono">{{ error }}</div>

    <template v-if="job">
      <div v-if="active || job.message" class="card prog">
        <div class="line">
          <b class="small nowrap">{{ job.status === 'queued' ? 'Queued' : active ? `${job.done} of ${total || '…'}` : job.status }}</b>
          <span class="muted small ellipsis" :title="job.message || undefined">{{ job.message }}</span>
        </div>
        <div v-if="active" class="bar"><div :style="{ width: `${pct}%` }" /></div>
      </div>

      <div class="cols">
        <section class="card">
          <div class="eyebrow">{{ rewrite ? 'Rewrites' : job.preview ? 'Try-on (not saved)' : 'Captions written' }} · {{ written.length }}</div>
          <div v-if="!written.length" class="muted small">None yet.</div>
          <div v-for="(w, i) in written" :key="keyOf(w, i)" class="cap" :class="{ text: rewrite }">
            <img v-if="!rewrite" :src="thumbUrl(job.dataset, w.path, 128)" loading="lazy" alt="" />
            <div class="cap-body">
              <div v-if="rewrite" class="was small muted" :title="w.prompt">{{ w.prompt }}</div>
              <div v-else class="mono muted small ellipsis" :title="w.path">{{ w.path }}</div>
              <template v-if="!w.text"><p class="muted">No usable answer: kept as it was.</p></template>
              <template v-else-if="asJson(w.text)">
                <p class="clamp" :class="{ open: isOpen(keyOf(w, i)) }">{{ asJson(w.text)!.short_description }}</p>
                <pre v-if="isOpen(keyOf(w, i))" class="json">{{ JSON.stringify(asJson(w.text), null, 2) }}</pre>
                <button class="link small" @click="toggle(keyOf(w, i))">{{ isOpen(keyOf(w, i)) ? 'Hide JSON' : `JSON · ${Object.keys(asJson(w.text)!).length} keys` }}</button>
              </template>
              <template v-else>
                <p class="clamp" :class="{ open: isOpen(keyOf(w, i)) }">{{ w.text }}</p>
                <button v-if="w.text.length > 320" class="link small" @click="toggle(keyOf(w, i))">{{ isOpen(keyOf(w, i)) ? 'Less' : 'More' }}</button>
              </template>
            </div>
          </div>
        </section>
        <aside class="side">
          <section class="card">
            <div class="eyebrow">Job</div>
            <dl>
              <template v-for="[k, v] in facts" :key="k"><dt>{{ k }}</dt><dd :class="{ mono: k === 'Model' }">{{ v }}</dd></template>
            </dl>
            <details v-if="job.prompt" class="prompt">
              <summary class="small muted">Instruction</summary>
              <p class="small instr">{{ job.prompt }}</p>
            </details>
          </section>
          <section class="card">
            <div class="eyebrow">Log</div>
            <div class="log mono">
              <div v-for="(l, i) in job.log ?? []" :key="i" :class="l.level">
                <span class="t">{{ clock(l.t) }}</span> <span :title="l.msg.length > 240 ? l.msg : undefined">{{ l.msg.length > 240 ? `${l.msg.slice(0, 240)}…` : l.msg }}</span>
              </div>
              <div v-if="!job.log?.length" class="muted">No log for this job (it ran before jobs kept one).</div>
            </div>
          </section>
        </aside>
      </div>
    </template>
  </div>
</template>

<style scoped>
.titles h1 {
  display: flex;
  align-items: baseline;
  gap: 6px;
}
.prog {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.line {
  display: flex;
  gap: 10px;
  align-items: baseline;
  min-width: 0;
}
.line .ellipsis {
  min-width: 0;
}
.nowrap {
  white-space: nowrap;
}
.bar {
  height: 4px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.bar div {
  height: 100%;
  background: var(--zen-accent);
}
.cols {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 380px);
  gap: 16px;
  align-items: start;
}
.cols > section,
.side > section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.side {
  display: flex;
  flex-direction: column;
  gap: 16px;
  position: sticky;
  top: 16px;
}
.cap {
  display: grid;
  grid-template-columns: 64px minmax(0, 1fr);
  gap: 12px;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.cap img {
  width: 64px;
  height: 64px;
  object-fit: cover;
  border-radius: var(--r-md);
}
.cap.text {
  grid-template-columns: minmax(0, 1fr);
}
.cap-body {
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
}
.cap-body > * {
  max-width: 100%;
}
.cap p {
  margin: 0;
  font-size: 13px;
  line-height: 1.55;
  overflow-wrap: anywhere;
}
.cap .clamp:not(.open) {
  display: -webkit-box;
  -webkit-line-clamp: 4;
  line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.cap .was {
  font-style: italic;
  overflow-wrap: anywhere;
}
.cap .json {
  width: 100%;
  margin: 2px 0 0;
  padding: 8px 10px;
  max-height: 340px;
  overflow: auto;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
  font-family: var(--zen-mono);
  font-size: 11px;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  scrollbar-width: thin;
}
.link {
  border: none;
  background: none;
  padding: 0;
  color: var(--zen-accent);
  font: inherit;
  cursor: pointer;
}
dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 12px;
  margin: 0;
  font-size: 12px;
}
dt {
  color: var(--zen-muted);
}
dd {
  margin: 0;
  min-width: 0;
  overflow-wrap: anywhere;
}
.prompt p {
  margin: 6px 0 0;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.prompt .instr {
  max-height: 280px;
  overflow: auto;
  scrollbar-width: thin;
}
.log {
  font-size: 11px;
  line-height: 1.6;
  max-height: 50vh;
  overflow: auto;
  scrollbar-width: thin;
}
.log > div {
  overflow-wrap: anywhere;
}
.log .t {
  color: var(--zen-muted);
}
.log .warn {
  color: var(--zen-warn);
}
.log .error {
  color: var(--zen-danger);
}
.err {
  color: var(--zen-danger);
}
@media (max-width: 900px) {
  .cols {
    grid-template-columns: 1fr;
  }
  .side {
    position: static;
  }
}
</style>
