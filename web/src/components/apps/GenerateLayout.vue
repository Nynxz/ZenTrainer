<script setup lang="ts">
// The "generate" layout: any app.json with layout "generate" is drawn by this. A controls panel
// (scrolls on its own, Generate pinned at its foot) and a stage: the current image large, the
// live sampling preview while it renders, a grid for a batch. Earlier images live in the
// Gallery; the stage keeps a short recent row. A `family` input picks the model and with it the
// workflow: the form starts from that workflow's own values and hides inputs it doesn't have.
import { ZenButton, ZenCombo, ZenField, ZenInput, ZenLightbox, ZenNumber, ZenSwitch } from '@nynxz/zenkit-ui'
import type { LightboxItem } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { inputChoices, onComfyMessage, onComfyPreview } from '@/api/comfy'
import { engine, viewUrl, type AppInfo, type AppInput, type Batch, type BatchCell, type ModelFamily } from '@/api/engine'
import LoraStack, { type LoraRow } from '@/components/LoraStack.vue'
import StarButton from '@/components/StarButton.vue'
import { starredFirst } from '@/stores/favorites'

const props = defineProps<{ app: AppInfo }>()
const route = useRoute()

const values = reactive<Record<string, unknown>>({})
const randomSeed = ref(true)
const count = ref(1)
const applies = ref<string[] | null>(null)
const error = ref<string | null>(null)
const families = ref<ModelFamily[]>([])
const showNegative = ref(false)

const kind = (i: AppInput) => (i.component ?? i.type) as AppInput['type']
const byKind = (k: AppInput['type']) => props.app.inputs.find((i) => kind(i) === k) ?? null
const familyInput = computed(() => byKind('family'))
const fileInput = computed(() => byKind('model-file'))
const promptInput = computed(() => byKind('prompt'))
const sizeInput = computed(() => byKind('size') ?? byKind('resolution'))
const lorasInput = computed(() => byKind('loras'))
const seedInput = computed(() => byKind('seed'))
const applied = (i: AppInput | null) => !!i && (!applies.value || applies.value.includes(i.id))
const negativeInput = computed(() => props.app.inputs.find((i) => kind(i) === 'text' && /negative/i.test(i.id)) ?? null)
const SPECIAL = ['family', 'model-file', 'prompt', 'size', 'resolution', 'loras', 'seed']
// whatever else the app declares, drawn generically
const otherInputs = computed(() => props.app.inputs.filter((i) => !SPECIAL.includes(kind(i)) && i !== negativeInput.value && applied(i)))
const family = computed(() => (familyInput.value ? (values[familyInput.value.id] as string) || '' : ''))
const fam = computed(() => families.value.find((f) => f.id === family.value) ?? null)

// ---- per-viewer memory -----------------------------------------------------------------------
const KEY = computed(() => `zentrainer.app.${props.app.id}`)
function remember() {
  try {
    localStorage.setItem(KEY.value, JSON.stringify({ values, randomSeed: randomSeed.value, count: count.value }))
  } catch {
    // storage blocked
  }
}
function recall(): { values?: Record<string, unknown>; randomSeed?: boolean; count?: number } {
  try {
    return JSON.parse(localStorage.getItem(KEY.value) ?? '{}')
  } catch {
    return {}
  }
}

// ---- the model: its workflow's own values ----------------------------------------------------
const familyItems = computed(() =>
  starredFirst(
    props.app.families.map((id) => families.value.find((f) => f.id === id) ?? ({ id, name: id, missing: [], profile: null } as unknown as ModelFamily)),
    (f) => f.id,
  ).map((f) => ({ value: f.id, label: f.name, keywords: f.name, f })),
)
const autoFile = ref('')
const unetFiles = ref<string[]>([])
const ckptFiles = ref<string[]>([])
const modelFile = computed({
  get: () => (fileInput.value ? ((values[fileInput.value.id] as string) ?? '') : ''),
  set: (v: string) => fileInput.value && (values[fileInput.value.id] = v),
})
const modelFileItems = computed(() => [
  { value: '', label: `Auto: ${autoFile.value || 'the workflow’s build'}`, keywords: 'auto' },
  ...(fam.value?.kind === 'checkpoint' ? ckptFiles.value : unetFiles.value).map((f) => ({ value: f, label: f, keywords: f })),
])

async function loadDefaults(keepPrompt: boolean) {
  try {
    const d = await engine.appDefaults(props.app.id, family.value || null)
    applies.value = d.inputs
    if (fileInput.value) {
      autoFile.value = String(d.values[fileInput.value.id] ?? '')
      values[fileInput.value.id] = ''
    }
    for (const i of props.app.inputs) {
      if (kind(i) === 'family' || kind(i) === 'model-file') continue
      if (keepPrompt && (kind(i) === 'prompt' || kind(i) === 'text') && values[i.id]) continue // your prompt follows you
      if (i.id in d.values && d.values[i.id] != null) values[i.id] = d.values[i.id]
      else if (!(i.id in values)) values[i.id] = i.default ?? null
    }
  } catch (e) {
    error.value = String(e)
  }
}
watch(family, (now, before) => before !== undefined && now !== before && loadDefaults(true))

// ---- size: aspect presets at about a megapixel -----------------------------------------------
const ASPECTS: [string, number, number][] = [
  ['1:1', 1024, 1024],
  ['3:4', 896, 1152],
  ['4:3', 1152, 896],
  ['2:3', 832, 1216],
  ['3:2', 1216, 832],
  ['9:16', 768, 1344],
  ['16:9', 1344, 768],
]
const size = computed(() => (sizeInput.value && Array.isArray(values[sizeInput.value.id]) ? (values[sizeInput.value.id] as number[]) : [1024, 1024]))
const setSize = (w: number, h: number) => sizeInput.value && (values[sizeInput.value.id] = [w, h])

// ---- running -----------------------------------------------------------------------------------
const running = ref(false)
const canRun = computed(() => !running.value && (!familyInput.value || !!family.value))
async function run() {
  if (!canRun.value) return
  running.value = true
  error.value = null
  try {
    const inputs: Record<string, unknown> = { ...values }
    const sweep: Record<string, unknown[]> = {}
    if (seedInput.value) {
      if (randomSeed.value) inputs[seedInput.value.id] = -1
      if (count.value > 1)
        sweep[seedInput.value.id] = randomSeed.value
          ? Array(count.value).fill(-1)
          : Array.from({ length: count.value }, (_, k) => Number(inputs[seedInput.value!.id] ?? 0) + k)
    }
    if (negativeInput.value && (!showNegative.value || !applied(negativeInput.value))) delete inputs[negativeInput.value.id]
    for (const i of props.app.inputs) if (!applied(i) && kind(i) !== 'family') delete inputs[i.id]
    const b = await engine.runApp(props.app.id, inputs, sweep)
    current.value = { ...b, cells: b.cells.map((c) => ({ ...c, status: 'queued', images: [] })), done: 0, total: b.cells.length, status: 'queued' }
    focus.value = 0
    poll(b.batch_id)
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  } finally {
    running.value = false
  }
}

// the batch on the stage, and the recent ones under it
const current = ref<Batch | null>(null)
const recent = ref<Batch[]>([])
const focus = ref(0)
const progress = reactive<Record<string, { value: number; max: number }>>({})
const preview = ref<{ promptId: string | null; url: string } | null>(null)
const timers = new Map<string, ReturnType<typeof setTimeout>>()
async function poll(id: string) {
  clearTimeout(timers.get(id))
  const b = await engine.batch(id).catch(() => null)
  if (b && current.value?.batch_id === id) current.value = b
  if (b && b.status === 'done') {
    recent.value = [b, ...recent.value.filter((r) => r.batch_id !== id)].slice(0, 12)
    return
  }
  timers.set(id, setTimeout(() => poll(id), 1500))
}
const offs: (() => void)[] = []
onMounted(() => {
  offs.push(
    onComfyMessage((m) => {
      const d = m.data as { prompt_id?: string; value?: number; max?: number }
      if (m.type === 'progress' && d.prompt_id) progress[d.prompt_id] = { value: d.value ?? 0, max: d.max ?? 0 }
      else if (['executed', 'execution_success', 'execution_error'].includes(m.type) && current.value?.cells.some((c) => c.prompt_id === d.prompt_id))
        poll(current.value.batch_id)
    }),
    onComfyPreview((promptId, url) => {
      if (!current.value || current.value.status === 'done') return
      if (promptId && !current.value.cells.some((c) => c.prompt_id === promptId)) return
      preview.value = { promptId, url }
    }),
  )
})
onUnmounted(() => {
  offs.forEach((f) => f())
  for (const t of timers.values()) clearTimeout(t)
})

const cells = computed(() => current.value?.cells ?? [])
const pct = (c: BatchCell) => {
  const p = progress[c.prompt_id]
  return p && p.max ? p.value / p.max : 0
}
const stepText = (c: BatchCell) => {
  const p = progress[c.prompt_id]
  return p && p.max ? `step ${p.value} / ${p.max}` : c.status === 'queued' ? 'queued' : ''
}
const livePreview = (c: BatchCell) => (preview.value && (!preview.value.promptId || preview.value.promptId === c.prompt_id) && c.status !== 'done' ? preview.value.url : null)
// the output's shape, so the small live preview fills the same box the result will
const aspectOf = (c: BatchCell) => {
  const v = current.value && sizeInput.value ? valuesOf(current.value, c)[sizeInput.value.id] : null
  return Array.isArray(v) && v.length === 2 ? `${v[0]} / ${v[1]}` : undefined
}
// one image on the stage: wheel to zoom, drag to pan (ZenKit's inline viewer)
const single = computed<LightboxItem[]>(() => {
  const c = cells.value.length === 1 ? cells.value[0] : null
  const src = c?.images[0] ? viewUrl(c.images[0]) : null
  return src ? [{ src }] : []
})
const imageOf = (c: BatchCell) => (c.images[0] ? viewUrl(c.images[0]) : null)
const valuesOf = (b: Batch, c: BatchCell) => ({ ...b.inputs, ...c.coords })
const metaLine = (b: Batch, c: BatchCell) => {
  const v = valuesOf(b, c)
  const name = families.value.find((f) => f.id === v[familyInput.value?.id ?? ''])?.name
  const sz = sizeInput.value && Array.isArray(v[sizeInput.value.id]) ? (v[sizeInput.value.id] as number[]).join('×') : ''
  const seed = seedInput.value && v[seedInput.value.id] != null && v[seedInput.value.id] !== -1 ? `seed ${v[seedInput.value.id]}` : ''
  return [name, sz, seed].filter(Boolean).join(' · ')
}

function reuse(b: Batch, c: BatchCell) {
  for (const [k, v] of Object.entries(valuesOf(b, c))) values[k] = v
  if (seedInput.value && seedInput.value.id in c.coords) randomSeed.value = c.coords[seedInput.value.id] === -1
  if (negativeInput.value && values[negativeInput.value.id]) showNegative.value = true
}
const copied = ref(false)
async function copyPrompt(b: Batch, c: BatchCell) {
  const p = promptInput.value ? String(valuesOf(b, c)[promptInput.value.id] ?? '') : ''
  try {
    await navigator.clipboard.writeText(p)
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch {
    // clipboard blocked
  }
}

// the viewer: everything this app made (its gallery), opened at the image you clicked
const lb = ref<{ items: LightboxItem[]; index: number } | null>(null)
async function openViewer(b: Batch, start: BatchCell) {
  const all = await engine.gallery(props.app.id, 1000).then((r) => r.items).catch(() => [])
  const toItem = (bb: Batch, c: BatchCell, src: string): LightboxItem => ({
    src,
    label: String(promptInput.value ? valuesOf(bb, c)[promptInput.value.id] ?? '' : ''),
    meta: metaLine(bb, c),
  })
  const items = all.map((g) => toItem({ inputs: g.values } as unknown as Batch, { coords: {} } as unknown as BatchCell, viewUrl(g.image)))
  const at = all.findIndex((g) => g.batch_id === b.batch_id && g.prompt_id === start.prompt_id)
  if (at >= 0) lb.value = { items, index: at }
  else {
    // not in the gallery yet (still being written): the batch first, then the rest
    const shots = b.cells.filter((c) => c.images.length)
    lb.value = { items: [...shots.map((c) => toItem(b, c, imageOf(c)!)), ...items], index: Math.max(0, shots.indexOf(start)) }
  }
}

// ---- start ----------------------------------------------------------------------------------
onMounted(async () => {
  const saved = recall()
  randomSeed.value = saved.randomSeed ?? true
  count.value = saved.count ?? 1
  for (const i of props.app.inputs) values[i.id] = i.default ?? null
  if (familyInput.value) {
    families.value = await engine.families().catch(() => [])
    const fid = saved.values?.[familyInput.value.id] as string | undefined
    values[familyInput.value.id] = fid && props.app.families.includes(fid) ? fid : (props.app.families[0] ?? '')
  }
  if (fileInput.value) [unetFiles.value, ckptFiles.value] = await Promise.all([inputChoices('UNETLoader', 'unet_name'), inputChoices('CheckpointLoaderSimple', 'ckpt_name')])
  await loadDefaults(false)
  for (const [k, v] of Object.entries(saved.values ?? {})) if (k in values && k !== familyInput.value?.id) values[k] = v
  if (negativeInput.value && values[negativeInput.value.id]) showNegative.value = true
  recent.value = (await engine.appBatches(props.app.id, 12).catch(() => [])).filter((b) => b.cells.some((c) => c.images.length))
  current.value = recent.value[0] ?? null
  // opened from the Gallery's Reuse: that image's settings
  const reuseRef = typeof route.query.reuse === 'string' ? route.query.reuse.split('/') : null
  if (reuseRef) {
    const b = await engine.batch(reuseRef[0]!).catch(() => null)
    const c = b?.cells.find((x) => x.prompt_id === reuseRef[1])
    if (b && c) {
      reuse(b, c)
      current.value = b
      focus.value = Math.max(0, b.cells.indexOf(c))
    }
  }
})
watch([values, randomSeed, count], remember, { deep: true })
</script>

<template>
  <div class="gen">
    <!-- controls: scroll on their own; Generate stays at the foot -->
    <aside class="panel card">
      <div class="panel-body">
        <div v-if="familyInput" class="block">
          <ZenCombo v-model="values[familyInput.id] as string" :items="familyItems" placeholder="Choose a model…" :item-height="34" :menu-width="380" class="model-combo">
            <template #option="{ item }">
              <div class="fam-opt">
                <StarButton :id="(item.f as ModelFamily).id" :name="item.label" />
                <b>{{ item.label }}</b>
                <span class="spacer" />
                <span class="muted small" :class="{ ok: !(item.f as ModelFamily).missing?.length }">
                  {{ (item.f as ModelFamily).missing?.length ? `${(item.f as ModelFamily).missing.length} to download` : 'ready' }}
                </span>
              </div>
            </template>
          </ZenCombo>
          <div v-if="fileInput" class="file-line">
            <span class="muted small">Build</span>
            <ZenCombo v-model="modelFile" :items="modelFileItems" :pinned="['']" :item-height="30" :menu-width="440" size="sm" class="file-combo" />
          </div>
          <p v-if="fam?.missing.length" class="warn small">
            <i class="mdi mdi-download-outline" /> Some files aren’t installed: <RouterLink to="/models">get them on Models</RouterLink>
          </p>
        </div>

        <div v-if="promptInput" class="block">
          <ZenInput v-model="values[promptInput.id] as string" type="textarea" :rows="6" autosize :placeholder="promptInput.placeholder ?? 'Describe the image'" class="prompt" />
          <template v-if="negativeInput && applied(negativeInput)">
            <button v-if="!showNegative" class="link small" @click="showNegative = true"><i class="mdi mdi-plus" /> Negative prompt</button>
            <div v-else class="neg">
              <ZenInput v-model="values[negativeInput.id] as string" type="textarea" :rows="2" placeholder="What to avoid" />
              <button class="link small" title="Don't use a negative prompt" @click="showNegative = false"><i class="mdi mdi-close" /></button>
            </div>
          </template>
        </div>

        <div v-if="sizeInput && applied(sizeInput)" class="block">
          <div class="label">Size</div>
          <div class="aspects">
            <button
              v-for="[name, w, h] in ASPECTS"
              :key="name"
              class="aspect"
              :class="{ on: size[0] === w && size[1] === h }"
              :title="`${w} × ${h}`"
              @click="setSize(w, h)"
            >
              <span class="shape" :style="{ aspectRatio: `${w} / ${h}` }" />
              {{ name }}
            </button>
          </div>
          <div class="wh">
            <ZenNumber :model-value="size[0]!" :min="64" :step="16" :precision="0" size="sm" @update:model-value="setSize($event, size[1]!)" />
            <span class="muted">×</span>
            <ZenNumber :model-value="size[1]!" :min="64" :step="16" :precision="0" size="sm" @update:model-value="setSize(size[0]!, $event)" />
          </div>
        </div>

        <div v-if="lorasInput && applied(lorasInput)" class="block">
          <div class="label">LoRAs</div>
          <LoraStack v-model="values[lorasInput.id] as LoraRow[]" />
        </div>

        <div class="block adv">
          <ZenField v-if="seedInput" :label="seedInput.label ?? 'Seed'" stack :tip="seedInput.tip">
            <div class="seed">
              <ZenNumber v-model="values[seedInput.id] as number" :min="0" :step="1" :precision="0" :disabled="randomSeed" size="sm" />
              <button class="icon-btn" :class="{ on: randomSeed }" :title="randomSeed ? 'A new seed every image (click to keep this one)' : 'Keep this seed (click for a new one every image)'" @click="randomSeed = !randomSeed">
                <i class="mdi mdi-dice-multiple-outline" />
              </button>
            </div>
          </ZenField>
          <template v-for="i in otherInputs" :key="i.id">
            <ZenField v-if="kind(i) === 'number' || kind(i) === 'slider'" :label="i.label ?? i.id" stack :tip="i.tip">
              <ZenNumber v-model="values[i.id] as number" :min="i.min" :max="i.max" :step="i.step ?? 1" :precision="(i.step ?? 1) < 1 ? 2 : 0" size="sm" />
            </ZenField>
            <ZenField v-else-if="kind(i) === 'toggle'" :label="i.label ?? i.id" stack :tip="i.tip">
              <div class="sw-row"><ZenSwitch v-model="values[i.id] as boolean" /></div>
            </ZenField>
            <ZenField v-else-if="kind(i) === 'choice'" :label="i.label ?? i.id" stack :tip="i.tip">
              <ZenCombo v-model="values[i.id] as string" :items="(i.options ?? []).map((o) => ({ value: o, label: o, keywords: o }))" :sorted="false" size="sm" />
            </ZenField>
            <ZenField v-else-if="kind(i) === 'text'" :label="i.label ?? i.id" stack :tip="i.tip" class="wide">
              <ZenInput v-model="values[i.id] as string" />
            </ZenField>
          </template>
        </div>
      </div>

      <div class="panel-foot">
        <div v-if="error" class="err small mono">{{ error }}</div>
        <div class="go-row">
          <div class="seg" title="Images per run">
            <button v-for="n in [1, 2, 4]" :key="n" :class="{ on: count === n }" @click="count = n">{{ n }}</button>
          </div>
          <ZenButton variant="primary" icon="mdi mdi-creation" :disabled="!canRun" class="go" @click="run">
            {{ running ? 'Queuing…' : count > 1 ? `Generate ${count}` : 'Generate' }}
          </ZenButton>
        </div>
      </div>
    </aside>

    <section class="stage">
      <div class="canvas card" :class="`n${Math.min(cells.length, 4)}`">
        <template v-if="cells.length">
          <div v-for="(c, n) in cells.slice(0, 4)" :key="c.prompt_id" class="shot" :class="{ focus: n === focus }" @click="focus = n">
            <ZenLightbox v-if="imageOf(c) && single.length" inline controls="none" :items="single" :index="0" class="zoomer" />
            <img v-else-if="imageOf(c)" :src="imageOf(c)!" alt="" :style="{ aspectRatio: aspectOf(c) }" @dblclick="openViewer(current!, c)" />
            <img v-else-if="livePreview(c)" :src="livePreview(c)!" alt="" class="live" :style="{ aspectRatio: aspectOf(c) }" />
            <div v-else class="wait">
              <i :class="c.status === 'error' ? 'mdi mdi-alert-outline' : 'mdi mdi-creation-outline'" />
              <span class="small">{{ c.status === 'error' ? c.error ?? 'failed' : 'waiting for the GPU' }}</span>
            </div>
            <div v-if="!imageOf(c) && c.status !== 'error'" class="prog">
              <span class="mono small">{{ stepText(c) }}</span>
              <div class="bar"><span :style="{ width: `${pct(c) * 100}%` }" /></div>
            </div>
            <div v-if="imageOf(c)" class="acts">
              <button class="act" title="Open in the viewer" @click.stop="openViewer(current!, c)"><i class="mdi mdi-arrow-expand" /></button>
              <button class="act" title="Put its settings back in the form" @click.stop="reuse(current!, c)"><i class="mdi mdi-backup-restore" /></button>
              <button class="act" :title="copied ? 'Copied' : 'Copy the prompt'" @click.stop="copyPrompt(current!, c)"><i :class="copied ? 'mdi mdi-check' : 'mdi mdi-content-copy'" /></button>
              <a class="act" :href="imageOf(c)!" :download="c.images[0]!.filename" title="Download" @click.stop><i class="mdi mdi-download" /></a>
            </div>
          </div>
        </template>
        <div v-else class="empty">
          <i class="mdi mdi-image-filter-hdr" />
          <b>Your images land here</b>
          <span class="muted small">Pick a model, write a prompt, Generate.</span>
        </div>
      </div>
      <div class="under">
        <span class="muted small ellipsis">{{ current && cells[focus] ? metaLine(current, cells[focus]!) : '' }}</span>
        <span class="spacer" />
        <div v-if="recent.length" class="recent">
          <button
            v-for="b in recent.slice(0, 8)"
            :key="b.batch_id"
            class="mini"
            :class="{ on: current?.batch_id === b.batch_id }"
            :title="new Date(b.created * 1000).toLocaleString()"
            @click="current = b; focus = 0"
          >
            <img v-if="b.cells[0]?.images[0]" :src="viewUrl(b.cells[0]!.images[0]!)" alt="" loading="lazy" />
          </button>
        </div>
        <RouterLink :to="`/gallery?app=${app.id}`" class="link small">Gallery <i class="mdi mdi-arrow-right" /></RouterLink>
      </div>
    </section>
    <ZenLightbox v-if="lb" v-model:index="lb.index" :items="lb.items" :searchable="false" @close="lb = null" />
  </div>
</template>

<style scoped>
.gen {
  display: grid;
  grid-template-columns: 360px minmax(0, 1fr);
  gap: 16px;
  height: calc(100vh - 150px);
  min-height: 560px;
}
@media (max-width: 900px) {
  .gen {
    grid-template-columns: 1fr;
    height: auto;
  }
}
.panel {
  display: flex;
  flex-direction: column;
  padding: 0;
  min-height: 0;
  overflow: hidden;
}
.panel-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scrollbar-width: thin;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.panel-foot {
  padding: 12px 14px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 60%, transparent);
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.label {
  font-size: 11px;
  color: var(--zen-muted);
}
.file-line {
  display: flex;
  align-items: center;
  gap: 8px;
}
.file-combo {
  flex: 1;
  min-width: 0;
}
.prompt :deep(textarea) {
  min-height: 120px;
  font-size: 14px;
  line-height: 1.5;
}
.neg {
  display: flex;
  gap: 6px;
  align-items: flex-start;
}
.neg > :first-child {
  flex: 1;
}
.aspects {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 4px;
}
.aspect {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 6px 0 4px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 60%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 10px;
  cursor: pointer;
}
.aspect .shape {
  height: 18px;
  max-width: 26px;
  border: 1.5px solid currentColor;
  border-radius: 2px;
}
.aspect:hover,
.aspect.on {
  color: var(--zen-selected-text, var(--zen-text));
}
.aspect.on {
  border-color: var(--zen-selected, var(--zen-accent));
  background: var(--zen-selected-bg, color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 10%, transparent));
}
.wh {
  display: flex;
  align-items: center;
  gap: 8px;
}
.wh > :not(span) {
  flex: 1;
}
.adv {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.adv .wide {
  grid-column: 1 / -1;
}
.seed {
  display: flex;
  gap: 6px;
  align-items: center;
}
.seed > :first-child {
  flex: 1;
  min-width: 0;
}
.sw-row {
  display: flex;
  align-items: center;
  min-height: 30px;
}
.fam-opt {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}
.ok {
  color: var(--zen-ok);
}
.warn {
  margin: 0;
  color: var(--zen-warn);
}
.warn a,
.link {
  color: var(--zen-accent);
}
.link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  align-self: flex-start;
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  cursor: pointer;
  text-decoration: none;
}
.go-row {
  display: flex;
  gap: 10px;
  align-items: center;
}
.go {
  flex: 1;
}
.seg {
  display: inline-flex;
  height: 32px;
  box-sizing: border-box;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.seg button {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 0 10px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.seg button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.icon-btn {
  width: 30px;
  height: 30px;
  flex: none;
  display: inline-grid;
  place-items: center;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.icon-btn.on {
  color: var(--zen-selected, var(--zen-accent));
  border-color: color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 50%, transparent);
}
.err {
  color: var(--zen-danger);
  overflow-wrap: anywhere;
}
/* ---- stage ---- */
.stage {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
  min-height: 0;
}
.canvas {
  flex: 1;
  min-height: 0;
  padding: 12px;
  display: grid;
  gap: 10px;
  background: color-mix(in srgb, var(--zen-bg) 70%, var(--zen-surface));
}
.canvas.n2 {
  grid-template-columns: 1fr 1fr;
}
.canvas.n3,
.canvas.n4 {
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
}
.shot {
  position: relative;
  min-height: 0;
  min-width: 0;
  display: grid;
  place-items: center;
  border-radius: var(--r-lg);
  overflow: hidden;
}
.shot img {
  height: 100%;
  width: auto;
  max-width: 100%;
  object-fit: contain;
  border-radius: var(--r-md);
  display: block;
}
.zoomer {
  position: absolute;
  inset: 0;
}
.shot img.live {
  filter: saturate(0.9);
}
.canvas:not(.n1) .shot.focus img {
  outline: 2px solid var(--zen-accent);
  outline-offset: 2px;
}
.wait {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--zen-muted);
}
.wait i {
  font-size: 28px;
}
.prog {
  position: absolute;
  left: 16px;
  right: 16px;
  bottom: 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-items: center;
}
.prog .mono {
  padding: 2px 8px;
  border-radius: var(--r-md);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
}
.prog .mono:empty {
  display: none;
}
.bar {
  width: min(320px, 80%);
  height: 4px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-text) 15%, transparent);
  overflow: hidden;
}
.bar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
  transition: width var(--zen-dur-fast, 0.15s) linear;
}
.acts {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 2;
  display: flex;
  gap: 4px;
  opacity: 0;
  transition: opacity var(--zen-dur-fast, 0.15s) ease;
}
.shot:hover .acts,
.shot:focus-within .acts {
  opacity: 1;
}
.act {
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-md);
  background: rgb(0 0 0 / 0.55);
  color: #fff;
  cursor: pointer;
  text-decoration: none;
}
.act:hover {
  background: rgb(0 0 0 / 0.75);
}
.empty {
  grid-column: 1 / -1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  color: var(--zen-text);
}
.empty i {
  font-size: 40px;
  color: var(--zen-muted);
}
.under {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 44px;
}
.ellipsis {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.recent {
  display: flex;
  gap: 6px;
}
.mini {
  width: 40px;
  height: 40px;
  padding: 0;
  border: 2px solid transparent;
  border-radius: var(--r-md);
  background: var(--zen-surface);
  overflow: hidden;
  cursor: pointer;
}
.mini.on {
  border-color: var(--zen-selected, var(--zen-accent));
}
.mini img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.small {
  font-size: 11px;
}
.mono {
  font-family: var(--font-mono, ui-monospace, monospace);
}
</style>
