<script setup lang="ts">
// Auto-captioning for one dataset, as a job: pick the model, the style, which caption set it
// writes and which images; try it on a few first; watch it run, cancel it, and afterwards
// review what it wrote or undo it. It never rewrites captions in "default" without asking.
import { ZenButton, ZenCombo, ZenInput, ZenNumber, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { setSetting, settings } from '@/stores/settings'
import { EVENTS, EngineError, engine, thumbUrl, type CaptionFormat, type CaptionJob, type CaptionLora, type CaptionMode, type CaptionStyle, type DatasetItem, type ModelSenses } from '@/api/engine'

const props = defineProps<{ dataset: string; items: DatasetItem[]; sets: string[]; selected: string[]; subject?: string }>()
const emit = defineEmits<{ changed: [set: string]; review: [job: CaptionJob] }>()

const styles = ref<CaptionStyle[]>([])
const models = ref<string[]>([])
const hearing = ref<string[]>([]) // models that also hear a video's soundtrack (Gemma 4)
const hasVideo = computed(() => props.items.some((it) => it.kind === 'video'))
const senses = ref<Record<string, ModelSenses>>({})
// each model's senses as icons, coloured for this dataset: fine, short of what it needs (warn), or not needed (dim)
function senseIcons(name: string) {
  // an older engine sends only which models hear: those take whole clips, the rest stills
  const s = senses.value[name] ?? (models.value.includes(name)
    ? { image: true, video: hearing.value.includes(name) ? 'clip' : 'stills', audio: hearing.value.includes(name), frames: 4 }
    : null)
  if (!s) return []
  const vid = hasVideo.value
  return [
    { icon: 'mdi mdi-image-outline', tone: vid ? 'dim' : 'ok', tip: 'Sees pictures' },
    {
      icon: 'mdi mdi-movie-open-outline',
      tone: !vid ? 'dim' : s.video === 'clip' ? 'ok' : 'warn',
      tip: s.video === 'clip' ? 'Watches a video as a timestamped clip' : `Sees a video as ${s.frames} stills: motion and timing are guesses`,
    },
    {
      icon: s.audio ? 'mdi mdi-music-note' : 'mdi mdi-music-note-off',
      tone: !vid ? 'dim' : s.audio ? 'ok' : 'warn',
      tip: s.audio ? "Hears a video's soundtrack" : "Can't hear: captions won't describe the sound",
    },
  ]
}
// LoRAs on the captioning model only (abliterated, uncensored, prompt styles): never training
const teLoras = ref<string[]>([])
const loras = ref<{ name: string; strength: number }[]>([])
const loraLink = ref('')
const loraChoices = ref<{ file: string; size: number | null }[]>([])
const loraNote = ref('')
try {
  loras.value = JSON.parse(localStorage.getItem("zentrainer.caption.loras") ?? '[]')
} catch {
  // storage blocked
}
watch(
  loras,
  (v) => {
    try {
      localStorage.setItem("zentrainer.caption.loras", JSON.stringify(v))
    } catch {
      // storage blocked
    }
  },
  { deep: true },
)
// the catalog's LoRAs that fit the selected model
const catalog = ref<CaptionLora[]>([])
// a LoRA by a name you can read: the catalog's label for one of its files, else the file's name
// without its folder or extension (the whole path on hover, the folder under it in the menu)
const loraLabel = (n: string) => {
  const file = n.split('/').pop() ?? n
  return catalog.value.find((e) => e.file.split('/').pop() === file)?.label ?? file.replace(/\.safetensors$/, '')
}
const loraItems = computed(() =>
  teLoras.value.map((n) => ({ value: n, label: loraLabel(n), keywords: n, dir: n.includes('/') ? n.slice(0, n.lastIndexOf('/')) : '' })),
)
const suggested = computed(() => catalog.value.filter((e) => new RegExp(e.base, 'i').test(form.clip_name)))
const sizeOf = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(n / 1e6))} MB`)
const isOn = (e: CaptionLora) => loras.value.some((l) => l.name.split('/').at(-1) === e.file.split('/').at(-1))
async function useSuggested(e: CaptionLora) {
  if (isOn(e)) {
    loras.value = loras.value.filter((l) => l.name.split('/').at(-1) !== e.file.split('/').at(-1))
    return
  }
  if (e.installed) {
    loras.value.push({ name: e.installed, strength: 1 })
    return
  }
  loraNote.value = ''
  try {
    const r = await engine.addCaptionLora('', undefined, e.id)
    if (r.name) {
      if (!teLoras.value.includes(r.name)) teLoras.value = [...teLoras.value, r.name]
      loras.value.push({ name: r.name, strength: 1 })
      loraNote.value = `downloading ${e.label} (${sizeOf(e.size)}): ready when the download finishes`
    }
  } catch (err) {
    loraNote.value = err instanceof Error ? err.message : String(err)
  }
}
const addLora = () => loras.value.push({ name: teLoras.value.find((n) => !loras.value.some((l) => l.name === n)) ?? '', strength: 1 })
async function fetchLora(file?: string) {
  loraNote.value = ''
  try {
    const r = await engine.addCaptionLora(loraLink.value, file)
    if (r.choices) {
      loraChoices.value = r.choices
      return
    }
    loraChoices.value = []
    loraLink.value = ''
    if (r.name) {
      if (!teLoras.value.includes(r.name)) teLoras.value = [...teLoras.value, r.name]
      loras.value.push({ name: r.name, strength: 1 })
      loraNote.value = `downloading ${r.name.split('/').at(-1)}: it's ready when the download finishes`
    }
  } catch (e) {
    loraNote.value = e instanceof Error ? e.message : String(e)
  }
}
const jobs = ref<CaptionJob[]>([])
const error = ref<string | null>(null)
const PREFS = 'zentrainer.caption'
// length: '' = the style's own limit, 0 = no limit; vary: sample with a new seed each run (re-rolls)
// advanced: a fixed seed (null: none), sampling settings and a custom length, for when the defaults
// don't suit a model (one that loops until it's cut off wants a repetition penalty)
const form = reactive({
  clip_name: '', style: 'nl', prompt: '', format: 'text' as CaptionFormat, set: 'nl', mode: 'missing' as CaptionMode, length: '' as '' | number, vary: false,
  advanced: false, fixedSeed: null as number | null, temperature: 0.7, top_p: 0.95, top_k: 50, repetition_penalty: 1.0, customLength: null as number | null,
  // pictures with a mask: show the model only what it keeps, so the caption is of what the LoRA learns
  hide_masked: true,
})
/** Whether this run samples (so temperature, top-p and top-k apply): Vary, or a fixed seed. */
const sampled = computed(() => form.vary || (form.advanced && form.fixedSeed != null))
const LENGTHS: { v: '' | number; label: string; tip: string }[] = [
  { v: '', label: 'Style', tip: "The style's own: no limit for most (up to 4096 tokens: the model ends its caption itself), 300 for a short caption" },
  { v: 600, label: '600', tip: 'Up to 600 tokens' },
  { v: 1200, label: '1200', tip: 'Up to 1200 tokens' },
  { v: 0, label: 'No limit', tip: 'As long as the model wants (stopped at 4096 tokens, in case it never ends)' },
]
const seed = () => Math.floor(Math.random() * 2 ** 31)
try {
  const { subject: _s, ...saved } = JSON.parse(localStorage.getItem(PREFS) ?? '{}') // subject: now the dataset's
  Object.assign(form, saved, { mode: 'missing' })
} catch {
  // storage blocked
}
watch(form, () => {
  try {
    const { mode: _m, ...keep } = form
    localStorage.setItem(PREFS, JSON.stringify(keep))
  } catch {
    // storage blocked
  }
})
// what to call the subject belongs to the dataset (its settings), not to this browser
const subject = ref(props.subject ?? '')
watch(() => props.subject, (v) => (subject.value = v ?? ''))
let subjectTimer: ReturnType<typeof setTimeout> | undefined
watch(subject, (v) => {
  clearTimeout(subjectTimer)
  if (v.trim() === (props.subject ?? '')) return
  subjectTimer = setTimeout(() => engine.datasetSettings(props.dataset, { caption_subject: v.trim() }).catch(() => {}), 500)
})

watch(
  () => form.style,
  (st, old) => {
    // the set follows a built-in style while it was still following it
    if (st !== 'custom' && !st.startsWith('u:') && (form.set === old || !form.set)) form.set = st
  },
)

onMounted(async () => {
  const info = await engine.captionInfo(props.dataset)
  styles.value = info.styles
  notes.value = info.notes ?? null
  models.value = info.models // loadable only, vision-language first
  hearing.value = info.hearing ?? []
  senses.value = info.senses ?? {}
  teLoras.value = info.loras ?? []
  catalog.value = info.lora_catalog ?? []
  jobs.value = info.jobs.filter((j) => !j.preview)
  preview.value = info.jobs.find((j) => j.preview) ?? null // a try-on still running, or just done
  if (!models.value.includes(form.clip_name))
    form.clip_name = models.value.find((n) => /qwen3vl_4b/i.test(n)) ?? models.value.find((n) => /qwen3vl/i.test(n)) ?? models.value[0] ?? ''
})

// ---- what a run would do -------------------------------------------------------------------
const has = (it: DatasetItem) => !!it.captions?.[form.set]
const targets = computed(() => {
  if (form.mode === 'selected') {
    const want = new Set(props.selected)
    return props.items.filter((it) => want.has(it.path))
  }
  return form.mode === 'all' ? props.items : props.items.filter((it) => !has(it))
})
const replaces = computed(() => targets.value.filter(has).length)
const counts = computed(() => ({
  missing: props.items.filter((it) => !has(it)).length,
  all: props.items.length,
  selected: props.selected.length,
}))
watch(
  () => props.selected.length,
  (n) => {
    if (n && form.mode !== 'selected') form.mode = 'selected'
    else if (!n && form.mode === 'selected') form.mode = 'missing'
  },
)
const summary = computed(() => {
  const n = targets.value.length
  if (!n) return form.mode === 'missing' ? `Every image already has a “${form.set}” caption` : 'No images to caption'
  const fresh = n - replaces.value
  const parts = []
  if (fresh) parts.push(`writes ${fresh} new`)
  if (replaces.value) parts.push(`replaces ${replaces.value} existing`)
  return `${n} image${n === 1 ? '' : 's'}: ${parts.join(', ')} in “${form.set}”`
})
// ---- your saved styles: kept in your ComfyUI user settings -----------------------------------
const mine = computed(() => settings.captionStyles)
const own = computed(() => mine.value.find((x) => `u:${x.id}` === form.style) ?? null)
function pick(id: string) {
  form.style = id
  const m = mine.value.find((x) => `u:${x.id}` === id)
  if (m) {
    form.prompt = m.prompt
    form.format = m.format ?? 'text'
  } else if (id !== 'custom') form.format = styles.value.find((st) => st.id === id)?.format ?? 'text'
}
// how the answer is kept: a built-in style says, yours and custom ones have the switch
const formatOf = computed<CaptionFormat>(() => (editable.value ? form.format : (styles.value.find((st) => st.id === form.style)?.format ?? 'text')))
const editable = computed(() => form.style === 'custom' || !!own.value)
const styleItems = computed(() => [
  ...styles.value.map((st) => ({ value: st.id, label: st.label })),
  ...mine.value.map((st) => ({ value: `u:${st.id}`, label: `${st.label} · yours` })),
  { value: 'custom', label: 'Custom…' },
])

// ---- what the model is told: shown, so nobody has to take the built-in prompts on trust ----------
const notes = ref<{ video: string; sound: string; subject: string; mask?: string } | null>(null)
const maskedCount = computed(() => props.items.filter((i) => i.mask).length)
const maskNote = computed(() => (maskedCount.value && form.hide_masked && notes.value?.mask ? notes.value.mask.trim() : ''))
const showPrompt = ref(false)
const stylePrompt = computed(() => (editable.value ? form.prompt : styles.value.find((st) => st.id === form.style)?.prompt ?? ''))
const clipNote = computed(() => (!hasVideo.value || !notes.value ? '' : hearing.value.includes(form.clip_name) ? notes.value.sound : notes.value.video))
const subjectNote = computed(() => (subject.value.trim() && notes.value ? notes.value.subject.replace('{subject}', subject.value.trim()).trim() : ''))
function editCopy() {
  form.prompt = stylePrompt.value
  form.style = 'custom'
}
const dirty = computed(() => !!own.value && (own.value.prompt !== form.prompt || (own.value.format ?? 'text') !== form.format))
function saveStyle() {
  if (own.value) {
    setSetting('captionStyles', mine.value.map((x) => (x.id === own.value!.id ? { ...x, prompt: form.prompt, format: form.format } : x)))
    return
  }
  const label = prompt('Name this caption style:')?.trim()
  if (!label) return
  const id = `${label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'style'}-${Date.now().toString(36).slice(-4)}`
  setSetting('captionStyles', [...mine.value, { id, label, prompt: form.prompt, format: form.format }])
  form.style = `u:${id}`
}
function removeStyle() {
  if (!own.value || !confirm(`Delete the caption style “${own.value.label}”?`)) return
  setSetting('captionStyles', mine.value.filter((x) => x.id !== own.value!.id))
  form.style = 'custom'
}

// a structured caption shown readably: pretty-printed (compact JSON has no spaces to wrap at); the
// running job's one line shows its short_description
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
const pretty = (t: string) => {
  const o = asJson(t)
  return o ? JSON.stringify(o, null, 1) : t
}
const gist = (t: string) => {
  const o = asJson(t)
  return o ? `${String(o.short_description ?? '')} {…}` : t
}

const validSet = computed(() => form.set === 'default' || /^[a-z0-9][a-z0-9_-]{0,23}$/.test(form.set))

// ---- jobs ------------------------------------------------------------------------------------
const active = computed(() => jobs.value.find((j) => !j.preview && ['queued', 'running'].includes(j.status)) ?? null)
const history = computed(() => jobs.value.filter((j) => !j.preview && !['queued', 'running'].includes(j.status)).slice(0, 6))
const preview = ref<CaptionJob | null>(null)
const busy = ref(false)

function body(extra: Record<string, unknown> = {}) {
  return {
    clip_name: form.clip_name,
    clip_type: 'auto', // the engine loads it as the plain chat model
    set: form.set,
    mode: form.mode,
    paths: form.mode === 'selected' ? props.selected : undefined,
    subject: subject.value.trim() || undefined,
    loras: loras.value.filter((l) => l.name),
    ...(maskedCount.value ? { hide_masked: form.hide_masked } : {}),
    ...(editable.value ? { prompt: form.prompt, format: form.format } : { style: form.style }),
    ...(form.advanced && form.customLength ? { max_length: form.customLength } : form.length !== '' ? { max_length: form.length } : {}),
    ...(form.advanced && form.fixedSeed != null ? { seed: form.fixedSeed } : form.vary ? { seed: seed() } : {}),
    ...(form.advanced ? { temperature: form.temperature, top_p: form.top_p, top_k: form.top_k, repetition_penalty: form.repetition_penalty } : {}),
    ...extra,
  }
}
/** The job's cut-off captions again, with no length limit, into the same set. */
async function recaptionCut(j: CaptionJob) {
  const n = j.cut?.length ?? 0
  if (!n) return
  if (j.set_name === 'default' && !confirm(`Re-caption the ${n} cut-off caption${n === 1 ? '' : 's'} in the default set, with no length limit?\n\nUndo puts the cut ones back.`))
    return
  busy.value = true
  error.value = null
  try {
    const { job } = await engine.startCaption(props.dataset, body({ mode: 'selected', paths: j.cut, set: j.set_name, max_length: 0, replace_default: j.set_name === 'default' }))
    jobs.value = [job, ...jobs.value]
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
/** Show only the images whose caption was cut off (as Review shows what a job wrote). */
function reviewCut(j: CaptionJob) {
  emit('review', { ...j, written: j.cut ?? [] })
}
async function start(previewN = 0) {
  if (!previewN && form.set === 'default' && replaces.value) {
    const ok = confirm(
      `Replace ${replaces.value} existing caption${replaces.value === 1 ? '' : 's'} in the default set?\n\n` +
        'These are the captions training uses. The job keeps what they said, so Undo puts them back.',
    )
    if (!ok) return
  }
  busy.value = true
  error.value = null
  try {
    const { job } = await engine.startCaption(props.dataset, body({ preview: previewN, replace_default: !previewN && form.set === 'default' }))
    if (previewN) preview.value = job
    else {
      jobs.value = [job, ...jobs.value]
      preview.value = null
    }
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
async function act(j: CaptionJob, action: 'cancel' | 'undo') {
  if (action === 'undo' && !confirm(`Undo this job? ${j.written?.length ?? j.done} caption(s) in “${j.set_name}” go back to how they were (any you edited since stay).`))
    return
  try {
    const r = await engine.captionAction(j.job_id, action)
    upsert(r.job)
    if (action === 'undo') emit('changed', j.set_name)
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  }
}
async function review(j: CaptionJob) {
  emit('review', j.written ? j : await engine.captionJob(j.job_id))
}
function upsert(j: CaptionJob) {
  if (j.preview) {
    if (preview.value?.job_id === j.job_id) preview.value = { ...preview.value, ...j }
    else if (!preview.value && ['queued', 'running'].includes(j.status)) preview.value = j
    return
  }
  const i = jobs.value.findIndex((x) => x.job_id === j.job_id)
  jobs.value = i < 0 ? [j, ...jobs.value] : jobs.value.map((x, n) => (n === i ? { ...x, ...j } : x))
}
let off = () => {}
onMounted(() => {
  off = onComfyMessage((m) => {
    if (m.type !== EVENTS.caption || m.data.dataset !== props.dataset) return
    const j = m.data as unknown as CaptionJob
    const before = jobs.value.find((x) => x.job_id === j.job_id)?.done ?? 0
    upsert(j)
    if (!j.preview && (j.done !== before || ['done', 'stopped', 'error'].includes(j.status))) emit('changed', j.set_name)
  })
})
onUnmounted(() => off())

const ago = (t?: number) => {
  if (!t) return ''
  const s = Date.now() / 1000 - t
  return s < 60 ? 'just now' : s < 3600 ? `${Math.round(s / 60)}m ago` : s < 86400 ? `${Math.round(s / 3600)}h ago` : new Date(t * 1000).toLocaleDateString()
}
const thumb = (path: string) => thumbUrl(props.dataset, path, 256)
</script>

<template>
  <div class="cp card">
    <div class="row">
      <div class="f model"><span class="eyebrow">Model</span>
        <ZenCombo
          v-model="form.clip_name"
          :items="models.map((n) => ({ value: n, label: n }))"
          :sorted="false"
          :item-height="30"
          placeholder="A vision-language model (Qwen-VL, Qwen3.5, Gemma 4)"
          size="sm"
        >
          <template #option="{ item }">
            <span class="mopt">
              <span class="senses"><i v-for="s in senseIcons(String(item.value))" :key="s.icon" :class="[s.icon, s.tone]" :title="s.tip" /></span>
              <span class="mname">{{ item.label }}</span>
            </span>
          </template>
          <template #selected="{ item }">
            <span v-if="item" class="mopt">
              <span class="senses"><i v-for="s in senseIcons(String(item.value))" :key="s.icon" :class="[s.icon, s.tone]" :title="s.tip" /></span>
              <span class="mname">{{ item.label }}</span>
            </span>
            <span v-else class="muted">A vision-language model</span>
          </template>
        </ZenCombo>
        <button
          v-if="hasVideo && hearing.length && !hearing.includes(form.clip_name)"
          class="hear-tip"
          :title="`${hearing[0]} hears each clip's soundtrack too: captions can describe the music and how the edit follows the beat`"
          @click="form.clip_name = hearing[0]!"
        >
          <i class="mdi mdi-music-note" /> Caption with sound
        </button>
        <div class="loras">
          <div v-for="(l, i) in loras" :key="i" class="lora">
            <ZenCombo v-model="l.name" :items="loraItems" :item-height="36" placeholder="A text-encoder LoRA" size="sm" :title="l.name">
              <template #selected="{ item }">
                <span v-if="item" class="lname ellipsis">{{ item.label }}</span>
                <span v-else class="muted">A text-encoder LoRA</span>
              </template>
              <template #option="{ item }">
                <span class="lopt">
                  <span class="ellipsis">{{ item.label }}</span>
                  <span v-if="(item as { dir?: string }).dir" class="muted small ellipsis">{{ (item as { dir?: string }).dir }}</span>
                </span>
              </template>
            </ZenCombo>
            <ZenNumber v-model="l.strength" :min="-2" :max="2" :step="0.05" :precision="2" size="sm" />
            <button class="x" title="Remove" @click="loras.splice(i, 1)"><i class="mdi mdi-close" /></button>
          </div>
          <div class="lora-add">
            <button class="hear-tip" title="Patch the captioning model with a text-encoder LoRA (captioning only, never training)" @click="addLora"><i class="mdi mdi-plus" /> LoRA</button>
            <ZenInput v-model="loraLink" placeholder="or a Hugging Face link" size="sm" @keydown.enter="fetchLora()" />
            <ZenButton v-if="loraLink" size="sm" icon="mdi mdi-download-outline" @click="fetchLora()">Get</ZenButton>
          </div>
          <div v-if="suggested.length" class="suggested">
            <span class="muted small">For this model:</span>
            <button v-for="e in suggested" :key="e.id" class="chip-s" :class="{ on: isOn(e) }" :title="`${e.note}. ${e.repo}`" @click="useSuggested(e)">
              <i :class="isOn(e) ? 'mdi mdi-check' : e.installed ? 'mdi mdi-plus' : 'mdi mdi-download-outline'" />
              {{ e.label }}<span v-if="!e.installed" class="muted"> · {{ sizeOf(e.size) }}</span>
            </button>
          </div>
          <div v-if="loraChoices.length" class="choices">
            <button v-for="c in loraChoices" :key="c.file" class="hear-tip choice" :title="c.file" @click="fetchLora(c.file)">{{ c.file.split('/').pop() }}</button>
          </div>
          <div v-if="loraNote" class="muted small">{{ loraNote }}</div>
        </div>
      </div>
      <div class="f style"><span class="eyebrow">Style</span>
        <div class="style-row">
          <ZenCombo :model-value="form.style" :items="styleItems" :sorted="false" :searchable="false" size="sm" @update:model-value="pick(String($event))" />
          <button class="mini" :class="{ on: showPrompt }" title="Show exactly what the model is told" @click="showPrompt = !showPrompt"><i class="mdi mdi-text-box-search-outline" /> Prompt</button>
        </div>
      </div>
    </div>
    <div v-if="showPrompt && !editable" class="told">
      <p v-if="clipNote" class="told-wrap"><span class="eyebrow">Clips first get</span>{{ clipNote }}</p>
      <p class="told-main" :class="{ long: stylePrompt.length > 600 }">{{ stylePrompt }}</p>
      <p v-if="formatOf === 'fibo_json'" class="told-wrap"><span class="eyebrow">Kept as</span>Fibo 2.0's structured JSON caption (Bria's VGL prompt and schema): checked and stored compact; an image whose answer isn't valid JSON after 3 tries keeps what it had.</p>
      <p v-if="subjectNote" class="told-wrap"><span class="eyebrow">Then</span>{{ subjectNote }}</p>
      <p v-if="maskNote" class="told-wrap"><span class="eyebrow">Masked pictures</span>{{ maskNote }}</p>
      <div class="own-acts">
        <ZenButton size="sm" icon="mdi mdi-pencil-outline" title="Start a custom style from this prompt: the built-in stays as it is" @click="editCopy">Edit a copy</ZenButton>
      </div>
    </div>
    <div v-if="editable" class="own">
      <p v-if="showPrompt && clipNote" class="told-wrap"><span class="eyebrow">Clips first get</span>{{ clipNote }}</p>
      <ZenInput v-model="form.prompt" type="textarea" :rows="3" placeholder="What should the model write? It sees the image, e.g. “Describe the outfit and pose in one sentence.”" />
      <label class="fmt" title="Check the answer is a JSON object with a short_description and keep it compact (Fibo 2.0's caption format); an image whose answer isn't gets 3 tries, then keeps what it had">
        <ZenSwitch :model-value="form.format === 'fibo_json'" @update:model-value="form.format = $event ? 'fibo_json' : 'text'" />
        <span class="small">Answer is a structured JSON caption</span>
      </label>
      <div class="own-acts">
        <ZenButton class="nowrap" size="sm" icon="mdi mdi-content-save-outline" :disabled="!form.prompt.trim() || (!!own && !dirty)" :title="own ? 'Save your changes to this style (kept on this ComfyUI, for every browser)' : 'Keep this instruction as a style of your own (kept on this ComfyUI, for every browser)'" @click="saveStyle">
          {{ own ? (dirty ? 'Save changes' : 'Saved') : 'Save as style' }}
        </ZenButton>
        <button v-if="own" class="mini" title="Delete this style" @click="removeStyle"><i class="mdi mdi-trash-can-outline" /></button>
      </div>
      <p v-if="showPrompt && subjectNote" class="told-wrap"><span class="eyebrow">Then</span>{{ subjectNote }}</p>
      <p v-if="showPrompt && maskNote" class="told-wrap"><span class="eyebrow">Masked pictures</span>{{ maskNote }}</p>
    </div>

    <div class="row">
      <label class="f set"><span class="eyebrow">Into set</span>
        <ZenInput v-model="form.set" size="sm" :class="{ bad: !validSet }" list="cp-sets" />
        <datalist id="cp-sets"><option v-for="s in sets" :key="s" :value="s" /></datalist>
      </label>
      <div class="f"><span class="eyebrow">Images</span>
        <div class="seg">
          <button :class="{ on: form.mode === 'missing' }" title="Only images with no caption in this set" @click="form.mode = 'missing'">Missing {{ counts.missing }}</button>
          <button :class="{ on: form.mode === 'all' }" title="Every image (existing captions in this set are replaced)" @click="form.mode = 'all'">All {{ counts.all }}</button>
          <button :class="{ on: form.mode === 'selected' }" :disabled="!counts.selected" title="The images you've selected" @click="form.mode = 'selected'">Selected {{ counts.selected }}</button>
        </div>
      </div>
      <label class="f subject"><span class="eyebrow">Call the subject</span>
        <ZenInput v-model="subject" size="sm" placeholder="optional, e.g. a trigger word" title="Saved with this dataset" />
      </label>
    </div>

    <div class="row">
      <div class="f"><span class="eyebrow">Length</span>
        <div class="seg">
          <button v-for="l in LENGTHS" :key="String(l.v)" :class="{ on: form.length === l.v }" :title="l.tip" @click="form.length = l.v">{{ l.label }}</button>
        </div>
      </div>
      <div class="f"><span class="eyebrow">Wording</span>
        <div class="seg">
          <button :class="{ on: !form.vary }" title="The model's single most likely caption: the same every time" @click="form.vary = false">Same each run</button>
          <button :class="{ on: form.vary }" title="Sampled with a new seed each run, so captioning an image again words it differently (re-roll a caption: select it, caption Selected)" @click="form.vary = true">Vary</button>
        </div>
      </div>
      <div v-if="maskedCount" class="f"><span class="eyebrow">Masked parts</span>
        <div class="seg">
          <button :class="{ on: form.hide_masked }" :title="`The ${maskedCount} pictures with a mask show the model only what it keeps: the rest in flat grey, cropped to the kept part, so the caption describes what the LoRA learns`" @click="form.hide_masked = true">Hidden</button>
          <button :class="{ on: !form.hide_masked }" title="The whole picture, mask or not" @click="form.hide_masked = false">Shown</button>
        </div>
      </div>
      <button class="mini adv-toggle" :class="{ on: form.advanced }" title="Seed, sampling and length settings" @click="form.advanced = !form.advanced">
        <i :class="form.advanced ? 'mdi mdi-chevron-up' : 'mdi mdi-tune-variant'" /> Advanced
      </button>
    </div>

    <div v-if="form.advanced" class="row adv">
      <div class="f"><span class="eyebrow">Seed</span>
        <div class="seed-row">
          <ZenNumber :model-value="form.fixedSeed ?? 0" :min="0" :max="2147483647" :step="1" :precision="0" size="sm" title="Sample with this seed: the same seed gives the same caption again, another seed another wording" @update:model-value="form.fixedSeed = $event" />
          <button class="mini" title="A new random seed" @click="form.fixedSeed = seed()"><i class="mdi mdi-dice-5-outline" /></button>
          <button class="mini" :disabled="form.fixedSeed == null" title="No fixed seed: Wording decides (the same each run, or a new seed each run)" @click="form.fixedSeed = null"><i class="mdi mdi-close" /></button>
        </div>
        <span class="small muted">{{ form.fixedSeed == null ? (form.vary ? 'a new one each run' : 'none: the most likely caption') : 'fixed: repeatable' }}</span>
      </div>
      <label class="f" :class="{ off: !sampled }" :title="sampled ? 'Lower: more literal and predictable. Higher: more varied' : 'Only when sampling (Vary, or a fixed seed)'">
        <span class="eyebrow">Temperature</span>
        <ZenNumber v-model="form.temperature" :min="0.05" :max="2" :step="0.05" :precision="2" size="sm" :disabled="!sampled" />
      </label>
      <label class="f" :class="{ off: !sampled }" :title="sampled ? 'Sample only from the likeliest words that make up this share of the odds' : 'Only when sampling (Vary, or a fixed seed)'">
        <span class="eyebrow">Top-p</span>
        <ZenNumber v-model="form.top_p" :min="0.05" :max="1" :step="0.05" :precision="2" size="sm" :disabled="!sampled" />
      </label>
      <label class="f" :class="{ off: !sampled }" :title="sampled ? 'Sample only from this many of the likeliest words (0: no limit)' : 'Only when sampling (Vary, or a fixed seed)'">
        <span class="eyebrow">Top-k</span>
        <ZenNumber v-model="form.top_k" :min="0" :max="500" :step="5" :precision="0" size="sm" :disabled="!sampled" />
      </label>
      <label class="f" title="Above 1 discourages repeating words. For a model that loops (and a red scarf, and a red scarf…) until it's cut off: try 1.05–1.15">
        <span class="eyebrow">Repetition penalty</span>
        <ZenNumber v-model="form.repetition_penalty" :min="1" :max="2" :step="0.01" :precision="2" size="sm" />
      </label>
      <label class="f" title="Tokens a caption may take, instead of the Length choice (up to 4096; empty: use Length)">
        <span class="eyebrow">Max tokens</span>
        <ZenNumber :model-value="form.customLength ?? 0" :min="0" :max="4096" :step="50" :precision="0" size="sm" @update:model-value="form.customLength = $event || null" />
      </label>
    </div>

    <div class="foot">
      <span class="small" :class="form.set === 'default' && replaces ? 'warn' : 'muted'">
        <i v-if="form.set === 'default' && replaces" class="mdi mdi-alert-outline" />
        {{ summary }}<template v-if="replaces"> · undoable</template>
      </span>
      <span class="spacer" />
      <span class="foot-acts">
      <ZenButton size="sm" icon="mdi mdi-eye-outline" :disabled="busy || !form.clip_name || !targets.length || !!active" title="Caption 3 of these without saving anything" @click="start(3)">Try on 3</ZenButton>
      <ZenButton size="sm" variant="primary" icon="mdi mdi-creation-outline" :disabled="busy || !form.clip_name || !validSet || !targets.length || !!active" @click="start()">
        Caption {{ targets.length || '' }}
      </ZenButton>
      </span>
    </div>
    <div v-if="error" class="err small mono">{{ error }}</div>

    <!-- try-on results: nothing saved -->
    <div v-if="preview" class="preview">
      <div class="line">
        <b class="small">Preview</b>
        <span class="muted small">{{ preview.status === 'done' ? 'not saved: run it to write them' : preview.status === 'running' ? `${preview.done}/${preview.total || 3}${preview.message ? ` · ${preview.message}` : ''}` : preview.status === 'queued' && preview.message ? preview.message : preview.status }}</span>
        <span class="spacer" />
        <button class="x" title="Close the preview" @click="preview = null"><i class="mdi mdi-close" /></button>
      </div>
      <div class="pv-grid">
        <div v-for="r in preview.results ?? []" :key="r.path" class="pv">
          <img :src="thumb(r.path)" alt="" />
          <pre v-if="asJson(r.text)" class="small json">{{ pretty(r.text) }}</pre>
          <p v-else class="small">{{ r.text }}</p>
          <p v-if="r.cut" class="small warn"><i class="mdi mdi-content-cut" /> cut off: raise the length</p>
        </div>
        <div v-for="n in Math.max(0, (preview.total || 3) - (preview.results?.length ?? 0))" :key="`w${n}`" class="pv waiting">
          <div class="ph"><i class="mdi mdi-creation-outline" /></div>
          <p class="small muted">{{ preview.status === 'queued' ? 'waiting for the GPU' : 'writing…' }}</p>
        </div>
      </div>
      <div v-if="preview.message && preview.status !== 'done'" class="err small">{{ preview.message }}</div>
    </div>

    <!-- the running job -->
    <div v-if="active" class="job live">
      <div class="line">
        <b class="small nowrap">{{ active.status === 'queued' ? 'Queued' : `Captioning ${active.done}/${active.total || '…'}` }}</b>
        <span class="muted small ellipsis grow" :title="active.message || undefined">into “{{ active.set_name }}”<template v-if="active.message"> · {{ active.message }}</template></span>
        <ZenButton class="nogrow" size="sm" icon="mdi mdi-stop" title="Stop at the next image; what's written stays (and can be undone)" @click="act(active, 'cancel')">Cancel</ZenButton>
      </div>
      <div class="bar"><div :style="{ width: `${active.total ? (active.done / active.total) * 100 : 0}%` }" /></div>
      <div v-if="active.text" class="small latest" :title="pretty(active.text)"><span class="mono muted">{{ active.file }}</span> {{ gist(active.text) }}</div>
      <div v-if="active.cut?.length" class="small warn"><i class="mdi mdi-content-cut" /> {{ active.cut.length }} cut off so far: {{ active.cut.slice(-3).join(', ') }}{{ active.cut.length > 3 ? '…' : '' }}</div>
    </div>

    <!-- what earlier jobs did -->
    <div v-if="history.length" class="history">
      <div class="eyebrow">Recent jobs</div>
      <div v-for="j in history" :key="j.job_id" class="job-row">
        <i :class="j.status === 'done' ? 'mdi mdi-check-circle-outline ok' : j.status === 'undone' ? 'mdi mdi-undo-variant muted' : 'mdi mdi-alert-circle-outline warn'" />
        <span class="small"><b>{{ j.set_name }}</b> · {{ j.done }} written<template v-if="j.replaced"> ({{ j.replaced }} replaced)</template></span>
        <span class="muted small ellipsis" :title="j.message">{{ j.status === 'done' ? ago(j.finished ?? j.created) : j.message || j.status }}</span>
        <span class="spacer" />
        <template v-if="j.cut?.length && j.status !== 'undone'">
          <button class="mini warn" :title="`Show only the ${j.cut.length} image(s) whose caption hit the length limit: ${j.cut.join(', ')}`" @click="reviewCut(j)"><i class="mdi mdi-content-cut" /> {{ j.cut.length }} cut off</button>
          <button class="mini" :disabled="busy || !!active" title="Caption the cut-off ones again with no length limit, into the same set" @click="recaptionCut(j)">Re-caption them</button>
        </template>
        <button v-if="j.done && j.status !== 'undone'" class="mini" title="Show only the images this job captioned" @click="review(j)">Review</button>
        <button v-if="j.done && j.status !== 'undone'" class="mini" title="Put back what this job changed" @click="act(j, 'undo')">Undo</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.adv-toggle {
  align-self: flex-end;
  margin-left: auto;
  white-space: nowrap;
}
.adv-toggle.on {
  color: var(--zen-selected, var(--zen-accent));
}
.adv {
  flex-wrap: wrap;
}
.adv .f.off {
  opacity: 0.5;
}
.seed-row {
  display: flex;
  gap: 4px;
  align-items: center;
}
.style-row {
  display: flex;
  gap: 6px;
  align-items: center;
}
.style-row :deep(.zen-combo) {
  flex: 1;
  min-width: 0;
}
.style-row .mini {
  flex: none;
  white-space: nowrap;
}
.mini.on {
  color: var(--zen-selected, var(--zen-accent));
}
.told {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.told-main.long {
  max-height: 220px;
  overflow: auto;
}
.fmt {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}
.told-main,
.told-wrap {
  margin: 0;
  padding: 8px 10px;
  border: 1px solid var(--zen-border);
  border-radius: var(--zen-radius, 7px);
  font-size: 12px;
  line-height: 1.45;
  white-space: pre-wrap;
}
.told-wrap {
  color: var(--zen-muted);
  border-style: dashed;
}
.told-wrap .eyebrow {
  display: block;
  margin-bottom: 2px;
}
.cp {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: flex-end;
}
.f {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.model {
  flex: 2 1 260px;
}
.f.style {
  flex: 1 1 220px;
}
.type {
  flex: 0 1 140px;
}
.set {
  flex: 0 1 160px;
}
.subject {
  flex: 1 1 200px;
}
.bad :deep(input) {
  outline: 1px solid var(--zen-danger);
}
.seg {
  display: inline-flex;
  height: 30px;
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
  white-space: nowrap;
}
.seg button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.seg button:disabled {
  opacity: 0.4;
  cursor: default;
}
.foot,
.line,
.job-row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.foot {
  flex-wrap: wrap;
}
.foot-acts {
  display: flex;
  gap: 6px;
  margin-left: auto;
  white-space: nowrap;
}
.small {
  font-size: 12px;
}
.warn {
  color: var(--zen-warn);
}
.ok {
  color: var(--zen-ok);
}
.err {
  color: var(--zen-danger);
  white-space: pre-wrap;
}
.preview,
.job,
.history {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.own {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.own-acts {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.own-acts .nowrap {
  flex: none;
  white-space: nowrap;
}
.pv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
}
.pv {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}
.pv img,
.pv .ph {
  width: 72px;
  height: 72px;
  flex: none;
  object-fit: cover;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
  display: grid;
  place-items: center;
  color: var(--zen-muted);
}
.pv p {
  margin: 0;
  line-height: 1.45;
  max-height: 120px;
  overflow: auto;
  scrollbar-width: thin;
}
.x,
.mini {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  border-radius: var(--r-md);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.x {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
}
.mini {
  height: 24px;
  padding: 0 8px;
}
.x:hover,
.mini:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
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
  transition: width var(--zen-dur, 0.3s) ease;
}
.line .grow {
  flex: 1 1 0;
  min-width: 0;
}
.line .nogrow,
.line .nowrap {
  flex: none;
  white-space: nowrap;
}
.pv .json {
  flex: 1;
  min-width: 0;
  margin: 0;
  max-height: 160px;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-family: var(--zen-mono);
  font-size: 11px;
  line-height: 1.35;
  scrollbar-width: thin;
}
.pv p {
  min-width: 0;
  overflow-wrap: anywhere;
}
.latest {
  overflow-wrap: anywhere;
  color: var(--zen-text);
  line-height: 1.45;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.ellipsis {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.loras {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 6px;
}
.lora {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 84px 22px;
  gap: 4px;
  align-items: center;
}
.lname {
  display: block;
  min-width: 0;
}
.lopt {
  display: flex;
  flex-direction: column;
  min-width: 0;
  line-height: 1.25;
}
.choice {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.lora-add {
  display: flex;
  gap: 6px;
  align-items: center;
}
.lora-add > :nth-child(2) {
  flex: 1;
  min-width: 0;
}
.suggested {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}
.chip-s {
  font: inherit;
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 85%, transparent);
  background: none;
  color: var(--zen-text);
  cursor: pointer;
}
.chip-s:hover,
.chip-s.on {
  border-color: var(--zen-selected, var(--zen-accent));
}
.chip-s.on i {
  color: var(--zen-selected, var(--zen-accent));
}
.choices {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.x {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  padding: 0;
}
.x:hover {
  color: var(--zen-text);
}
.mopt {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  width: 100%;
}
.mname {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.senses {
  display: inline-flex;
  gap: 3px;
  flex: none;
  width: 52px; /* a fixed column: names line up after it */
  font-size: 14px;
}
.senses .ok {
  color: var(--zen-accent);
}
.senses .warn {
  color: var(--zen-warn);
}
.senses .dim {
  color: var(--zen-muted);
  opacity: 0.55;
}
.hear-tip {
  margin-top: 4px;
  align-self: flex-start;
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  font-size: 11px;
  color: var(--zen-accent);
  cursor: pointer;
}
.hear-tip:hover {
  text-decoration: underline;
}
</style>
