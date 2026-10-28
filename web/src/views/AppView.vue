<script setup lang="ts">
// A generic app, drawn from its manifest: inputs on the left (grouped, with sweeps),
// results on the right as a gallery or, for sweeps, a labelled grid. Batches run on
// ComfyUI's queue; each cell's progress streams over the websocket.
import { ZenButton, ZenEmpty, ZenLightbox } from '@nynxz/zenkit-ui'
import type { LightboxItem } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { onComfyMessage } from '@/api/comfy'
import { engine, viewUrl, type AppInfo, type AppInput as AppInputDef, type Batch, type BatchCell } from '@/api/engine'
import AppInput from '@/components/AppInput.vue'
import GenerateLayout from '@/components/apps/GenerateLayout.vue'
import { runs } from '@/stores/runs'

const props = defineProps<{ id: string }>()
const route = useRoute()

const app = ref<AppInfo | null>(null)
const error = ref<string | null>(null)
const values = reactive<Record<string, unknown>>({})
const sweeps = reactive<Record<string, unknown[] | undefined>>({})
const batch = ref<Batch | null>(null)
const history = ref<Batch[]>([])
const running = ref(false)
const progress = reactive<Record<string, { value: number; max: number }>>({})
const groupsOpen = reactive<Record<string, boolean>>({ image: true, sampler: false, advanced: false })

const KEY = () => `zentrainer.app.${props.id}`

async function load() {
  const { apps, errors } = await engine.apps()
  app.value = apps.find((a) => a.id === props.id) ?? null
  if (!app.value) {
    error.value = errors.find((e) => e.startsWith(props.id)) ?? 'No such app.'
    return
  }
  let saved: { values?: Record<string, unknown>; sweeps?: Record<string, unknown[]> } = {}
  try {
    saved = JSON.parse(localStorage.getItem(KEY()) ?? '{}')
  } catch {
    // storage blocked
  }
  for (const inp of app.value.inputs) {
    const d = inp.default
    if (inp.sweep && Array.isArray(d) && inp.type !== 'loras' && inp.type !== 'resolution') {
      sweeps[inp.id] = saved.sweeps?.[inp.id] ?? [...d]
      values[inp.id] = saved.values?.[inp.id] ?? d[0]
    } else {
      values[inp.id] = saved.values?.[inp.id] ?? (inp.type === 'model' ? { kind: 'checkpoint' } : d)
      if (saved.sweeps?.[inp.id]) sweeps[inp.id] = saved.sweeps[inp.id]
    }
  }
  // opened from a run's checkpoint (?run=…&lora=…): that LoRA, on the model it was trained on
  const q = route.query
  if (typeof q.lora === 'string') {
    for (const inp of app.value.inputs) {
      if (inp.type === 'lora') {
        values[inp.id] = q.lora
        delete sweeps[inp.id]
      }
    }
  }
  // opened from Merge LoRAs (?loras=[{name, strength}]): that stack, which renders exactly as
  // the merged file would (before any shrink)
  if (typeof q.loras === 'string') {
    try {
      const stack = (JSON.parse(q.loras) as { name: string; strength: number }[]).map((r) => ({ ...r, on: true }))
      for (const inp of app.value.inputs) if (inp.type === 'loras') values[inp.id] = stack
    } catch {
      // a bad link: keep the saved stack
    }
  }
  if (typeof q.run === 'string') {
    const spec = await engine.run(q.run).then((r) => r.spec).catch(() => null)
    const model = app.value.inputs.find((i) => i.type === 'model')
    if (spec && model) values[model.id] = { ...spec.model }
  }
  history.value = await engine.appBatches(props.id)
  batch.value = history.value[0] ?? null
}
watch([values, sweeps], () => {
  try {
    localStorage.setItem(KEY(), JSON.stringify({ values, sweeps }))
  } catch {
    // storage blocked
  }
}, { deep: true })

// ---- layout ---------------------------------------------------------------------------------
const GROUPS: Record<string, string> = { image: 'Image', sampler: 'Sampler', advanced: 'Advanced' }
const topInputs = computed(() => app.value?.inputs.filter((i) => !i.group) ?? [])
const grouped = computed(() => {
  const out: [string, AppInputDef[]][] = []
  for (const g of Object.keys(GROUPS)) {
    const list = app.value?.inputs.filter((i) => i.group === g) ?? []
    if (list.length) out.push([g, list])
  }
  return out
})

// runs with checkpoints, for Test LoRA's "all checkpoints of a run"
const runLoras = ref<{ name: string; loras: string[] }[]>([])
async function loadRunLoras() {
  const done = runs.value.filter((r) => ['done', 'stopped', 'interrupted', 'error'].includes(r.status)).slice(0, 12)
  const details = await Promise.all(done.map((r) => engine.run(r.run_id).catch(() => null)))
  runLoras.value = details
    .filter((d): d is NonNullable<typeof d> => !!d && d.checkpoints.length > 0)
    .map((d) => ({ name: `${d.run.name} (${d.run.run_id.slice(9, 15)})`, loras: d.checkpoints.map((c) => c.lora_name) }))
}

// ---- run ------------------------------------------------------------------------------------------
const activeSweeps = computed(() =>
  Object.fromEntries(Object.entries(sweeps).filter(([, v]) => Array.isArray(v) && v.length)) as Record<string, unknown[]>,
)
const cellCount = computed(() => Object.values(activeSweeps.value).reduce((n, v) => n * v.length, 1))
const missing = computed(() => {
  const m = app.value?.inputs.find((i) => i.type === 'model')
  if (!m) return null
  const v = values[m.id] as { kind?: string; ckpt_name?: string; unet_name?: string; clip_name?: string; vae_name?: string } | undefined
  if (!v) return 'Pick a model'
  if (v.kind === 'split' ? !(v.unet_name && v.clip_name && v.vae_name) : !v.ckpt_name) return 'Pick a model'
  return null
})
async function run() {
  if (!app.value) return
  running.value = true
  error.value = null
  try {
    const b = await engine.runApp(app.value.id, { ...values }, activeSweeps.value)
    batch.value = { ...b, cells: b.cells.map((c) => ({ ...c, status: 'queued', images: [] })), done: 0, total: b.cells.length, status: 'queued' }
    history.value = [batch.value, ...history.value]
    poll()
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  } finally {
    running.value = false
  }
}
let timer: ReturnType<typeof setTimeout> | undefined
async function poll() {
  clearTimeout(timer)
  if (!batch.value) return
  const b = await engine.batch(batch.value.batch_id).catch(() => null)
  if (b && batch.value?.batch_id === b.batch_id) {
    batch.value = b
    const i = history.value.findIndex((h) => h.batch_id === b.batch_id)
    if (i >= 0) history.value[i] = b
  }
  if (batch.value && batch.value.status !== 'done') timer = setTimeout(poll, 1500)
}

let off = () => {}
onMounted(() => {
  load().then(() => {
    if (batch.value && batch.value.status !== 'done') poll()
  })
  if (props.id === 'test-lora') loadRunLoras()
  off = onComfyMessage((m) => {
    if (m.type === 'progress') {
      const d = m.data as { value: number; max: number; prompt_id?: string }
      if (d.prompt_id) progress[d.prompt_id] = { value: d.value, max: d.max }
    } else if (m.type === 'executed' || m.type === 'execution_success' || m.type === 'execution_error') {
      if (batch.value?.cells.some((c) => c.prompt_id === m.data.prompt_id)) poll()
    }
  })
})
onUnmounted(() => {
  off()
  clearTimeout(timer)
})

// ---- results ------------------------------------------------------------------------------------
const view = computed(() => app.value?.outputs[0]?.view ?? 'gallery')
const axes = computed(() => {
  const s = batch.value?.sweep ?? {}
  return Object.entries(s).filter(([, v]) => Array.isArray(v) && v.length > 1) as [string, unknown[]][]
})
const xAxis = computed(() => axes.value[0])
const yAxis = computed(() => axes.value[1])
const pageAxes = computed(() => axes.value.slice(2))
const page = reactive<Record<string, unknown>>({})
const labelOf = (key: string, v: unknown) => {
  const t = app.value?.inputs.find((i) => i.id === key)?.type
  if (t === 'lora') {
    // checkpoints of one run share a long prefix: show what differs ("step 250", "final")
    const name = String(v).split('/').at(-1)!.replace(/\.safetensors$/, '')
    const m = name.match(/_(?:([a-z]+)-)?step0*(\d+)$/) // _step000250, _stopped-step000021
    if (m) return m[1] ? `${m[1]} · step ${m[2]}` : `step ${m[2]}`
    return /_\d{8}-\d{6}-[0-9a-f]{6}$/.test(name) ? 'final' : name // <name>_<run id> is the final save
  }
  return String(v)
}
const keyLabel = (key: string) => app.value?.inputs.find((i) => i.id === key)?.label ?? key
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)
function cellAt(x: unknown, y: unknown): BatchCell | undefined {
  return batch.value?.cells.find(
    (c) =>
      (!xAxis.value || same(c.coords[xAxis.value[0]], x)) &&
      (!yAxis.value || same(c.coords[yAxis.value[0]], y)) &&
      pageAxes.value.every(([k, vals]) => same(c.coords[k], page[k] ?? vals[0])),
  )
}
const lb = ref<{ items: LightboxItem[]; index: number } | null>(null)
const describe = (c: BatchCell) =>
  Object.entries(c.coords).map(([k, v]) => `${keyLabel(k)}: ${labelOf(k, v)}`).join(' · ')
function open(c: BatchCell) {
  const cells = (batch.value?.cells ?? []).filter((x) => x.images.length)
  lb.value = {
    items: cells.map((x) => ({ src: viewUrl(x.images[0]!), label: describe(x) || app.value?.name, meta: String(x.coords.prompt ?? batch.value?.inputs.prompt ?? '') })),
    index: Math.max(0, cells.indexOf(c)),
  }
}
const pct = (c: BatchCell) => {
  const p = progress[c.prompt_id]
  return p && p.max ? p.value / p.max : 0
}
const when = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
</script>

<template>
  <div class="page wide">
    <div v-if="error && !app" class="card muted">{{ error }}</div>
    <template v-else-if="app">
      <div class="page-head">
        <div class="titles">
          <RouterLink to="/apps" class="crumb">Apps</RouterLink>
          <h1>{{ app.name }}</h1>
        </div>
      </div>

      <GenerateLayout v-if="app.layout === 'generate'" :app="app" />
      <div v-else class="layout">
        <aside class="inspector card">
          <AppInput
            v-for="inp in topInputs"
            :key="inp.id"
            v-model:value="values[inp.id]"
            v-model:sweep="sweeps[inp.id]"
            :input="inp"
            :runs="runLoras"
          />
          <div v-for="[g, list] in grouped" :key="g" class="group">
            <button class="group-head" @click="groupsOpen[g] = !groupsOpen[g]">
              <i :class="groupsOpen[g] ? 'mdi mdi-chevron-down' : 'mdi mdi-chevron-right'" /> {{ GROUPS[g] }}
            </button>
            <template v-if="groupsOpen[g]">
              <AppInput v-for="inp in list" :key="inp.id" v-model:value="values[inp.id]" v-model:sweep="sweeps[inp.id]" :input="inp" />
            </template>
          </div>
          <div class="run-bar">
            <div v-if="error" class="err mono">{{ error }}</div>
            <ZenButton variant="primary" icon="mdi mdi-play" block :disabled="running || !!missing || cellCount > 64" @click="run">
              {{ missing ?? (cellCount > 64 ? `${cellCount} images: max 64` : cellCount > 1 ? `Run ${cellCount} images` : 'Run') }}
            </ZenButton>
          </div>
        </aside>

        <section class="stage">
          <div v-if="batch" class="card result">
            <div class="result-head">
              <b>{{ batch.status === 'done' ? 'Done' : batch.status === 'running' ? 'Running' : 'Queued' }}</b>
              <span class="muted small">{{ batch.done }}/{{ batch.total }} · {{ when(batch.created) }}</span>
              <span class="spacer" />
              <div v-for="[k, vals] in pageAxes" :key="k" class="pages">
                <span class="muted small">{{ keyLabel(k) }}</span>
                <button v-for="v in vals" :key="String(v)" class="chip" :class="{ on: same(page[k] ?? vals[0], v) }" @click="page[k] = v">
                  {{ labelOf(k, v) }}
                </button>
              </div>
            </div>

            <!-- grid: x across, y down -->
            <div v-if="view === 'grid' && xAxis" class="grid-wrap">
              <table class="grid">
                <thead>
                  <tr>
                    <th class="corner">
                      <span v-if="yAxis" class="muted small">{{ keyLabel(yAxis[0]) }} ↓</span>
                      <span class="muted small">{{ keyLabel(xAxis[0]) }} →</span>
                    </th>
                    <th v-for="x in xAxis[1]" :key="String(x)" class="xh" :title="String(x)">{{ labelOf(xAxis[0], x) }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="y in yAxis ? yAxis[1] : [null]" :key="String(y)">
                    <th v-if="yAxis" class="yh" :title="String(y)">{{ labelOf(yAxis[0], y) }}</th>
                    <th v-else class="yh" />
                    <td v-for="x in xAxis[1]" :key="String(x)">
                      <template v-if="cellAt(x, y)">
                        <button v-if="cellAt(x, y)!.images.length" class="cell" @click="open(cellAt(x, y)!)">
                          <img :src="viewUrl(cellAt(x, y)!.images[0]!)" loading="lazy" alt="" />
                        </button>
                        <div v-else class="cell pending" :class="{ err: cellAt(x, y)!.status === 'error' }" :title="cellAt(x, y)!.error ?? ''">
                          <template v-if="cellAt(x, y)!.status === 'error'"><i class="mdi mdi-alert-outline" /></template>
                          <div v-else class="cell-bar"><div :style="{ width: `${pct(cellAt(x, y)!) * 100}%` }" /></div>
                        </div>
                      </template>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- gallery -->
            <div v-else class="gallery">
              <template v-for="c in batch.cells" :key="c.prompt_id">
                <button v-if="c.images.length" class="g-item" :title="describe(c)" @click="open(c)">
                  <img :src="viewUrl(c.images[0]!)" loading="lazy" alt="" />
                  <span v-if="describe(c)" class="g-cap">{{ describe(c) }}</span>
                </button>
                <div v-else class="g-item pending" :class="{ err: c.status === 'error' }" :title="c.error ?? ''">
                  <template v-if="c.status === 'error'"><i class="mdi mdi-alert-outline" /> <span class="small">{{ c.error ?? 'failed' }}</span></template>
                  <div v-else class="cell-bar"><div :style="{ width: `${pct(c) * 100}%` }" /></div>
                </div>
              </template>
            </div>
          </div>
          <ZenEmpty v-else :title="`Run ${app.name}`" :icon="app.icon ?? 'mdi mdi-play'">Results show up here.</ZenEmpty>

          <div v-if="history.length > 1" class="history">
            <div class="eyebrow">Recent</div>
            <div class="h-strip">
              <button
                v-for="h in history"
                :key="h.batch_id"
                class="h-item"
                :class="{ on: h.batch_id === batch?.batch_id }"
                :title="`${when(h.created)} · ${h.total} image${h.total === 1 ? '' : 's'}`"
                @click="batch = h; poll()"
              >
                <img v-if="h.cells.find((c) => c.images.length)" :src="viewUrl(h.cells.find((c) => c.images.length)!.images[0]!)" alt="" />
                <i v-else class="mdi mdi-image-outline" />
                <span v-if="h.total > 1" class="h-n">{{ h.total }}</span>
              </button>
            </div>
          </div>
        </section>
      </div>

      <ZenLightbox v-if="lb" v-model:index="lb.index" :items="lb.items" @close="lb = null" />
    </template>
  </div>
</template>

<style scoped>
.wide {
  max-width: 1440px;
}
.layout {
  display: grid;
  grid-template-columns: 340px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
.inspector {
  position: sticky;
  top: 8px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 16px;
}
.group {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 60%, transparent);
}
.group-head {
  align-self: flex-start;
  border: none;
  background: none;
  color: var(--zen-text);
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  padding: 0;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.run-bar {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 6px;
}
.err {
  font-size: 11px;
  color: var(--zen-danger);
  white-space: pre-wrap;
}
.stage {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}
.result {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.result-head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.pages {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
}
.chip {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: var(--r-md);
  cursor: pointer;
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.chip.on {
  color: var(--zen-selected-text, var(--zen-text));
  border-color: var(--zen-selected, var(--zen-accent));
  background: var(--zen-selected-bg, color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 14%, transparent));
}
.grid-wrap {
  overflow-x: auto;
  scrollbar-width: thin;
}
.grid {
  border-collapse: separate;
  border-spacing: 6px;
}
.grid th {
  font-size: 11px;
  font-weight: 600;
  color: var(--zen-muted);
  text-align: center;
  white-space: nowrap;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.corner {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
}
.yh {
  text-align: right !important;
  max-width: 180px !important;
  white-space: normal !important;
  font-weight: 500 !important;
  padding-right: 4px;
}
.grid td {
  padding: 0;
}
.cell {
  display: block;
  width: 200px;
  height: 200px;
  padding: 0;
  border: none;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: zoom-in;
}
.cell img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.pending {
  display: grid;
  place-items: center;
  cursor: default;
  color: var(--zen-muted);
}
.pending.err {
  color: var(--zen-danger);
}
.cell-bar {
  width: 60%;
  height: 4px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-muted) 25%, transparent);
  overflow: hidden;
}
.cell-bar div {
  height: 100%;
  background: var(--zen-accent);
  transition: width 0.3s ease;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
}
.g-item {
  position: relative;
  aspect-ratio: 1;
  padding: 0;
  border: none;
  border-radius: var(--r-lg);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: zoom-in;
}
.g-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.g-item.pending {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px;
}
.g-cap {
  position: absolute;
  left: 6px;
  right: 6px;
  bottom: 6px;
  font-size: 11px;
  padding: 3px 6px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
  text-align: left;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.history {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.h-strip {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  scrollbar-width: thin;
  padding-bottom: 4px;
}
.h-item {
  position: relative;
  flex: none;
  width: 72px;
  height: 72px;
  padding: 0;
  border: 2px solid transparent;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  color: var(--zen-muted);
  cursor: pointer;
}
.h-item.on {
  border-color: var(--zen-selected, var(--zen-accent));
}
.h-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.h-n {
  position: absolute;
  right: 3px;
  bottom: 3px;
  font-size: 10px;
  padding: 0 4px;
  border-radius: var(--r-sm);
  background: rgb(0 0 0 / 0.6);
  color: #fff;
}
@media (max-width: 1000px) {
  .layout {
    grid-template-columns: 1fr;
  }
  .inspector {
    position: static;
  }
}
</style>
