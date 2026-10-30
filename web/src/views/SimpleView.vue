<script setup lang="ts">
// Simple mode: pick a recipe card, add photos, name it, train. Everything else (settings, captions,
// model files) comes from the recipe; what's missing is downloaded and the photos are captioned
// before the run starts, each as a visible step.
import { makerLogo } from '@/lib/makers'
import { ZenButton, ZenInput, ZenLightbox, type LightboxItem } from '@nynxz/zenkit-ui'
import TemplateCard from '@/components/TemplateCard.vue'
import CaptionRing from '@/components/CaptionRing.vue'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { EngineError, defaultSpec, engine, thumbUrl, type DatasetFaces, type DatasetInfo, type DatasetItem } from '@/api/engine'
import { captionJobs, startCaptionSync } from '@/stores/captions'
import { downloadFor, downloads, finished, fmtBytes, startDownloadSync } from '@/stores/downloads'
import { RECIPES, buildSpec, plan, recipe as findRecipe, triggerFrom, type Plan, type Recipe } from '@/lib/recipes'
import { settings } from '@/stores/settings'

const props = defineProps<{ id?: string }>()
const router = useRouter()
const r = computed<Recipe | null>(() => (props.id ? findRecipe(props.id) : null))
// "/templates/..." is ComfyUI's own (template thumbnails, maker logos); else our web/public
// a card picture: a path in the app, never another site's ("//host/x.png") or a URL with a scheme
const asset = (p: string) =>
  p.startsWith('//') || /^[a-z][a-z0-9+.-]*:/i.test(p) ? '' : p.startsWith('/') ? p : `${import.meta.env.BASE_URL}${p}`

// ---- the gallery ---------------------------------------------------------------------------
const kinds = [
  { id: 'all', label: 'All' },
  { id: 'character', label: 'People & characters' },
  { id: 'style', label: 'Styles' },
  { id: 'general', label: 'Any subject' },
] as const
const kind = ref<(typeof kinds)[number]['id']>('all')
const shown = computed(() => RECIPES.filter((x) => kind.value === 'all' || x.kind === kind.value))
// All: the recipes under a title per kind (one kind picked: just its cards)
const groups = computed(() =>
  kinds.filter((k) => k.id !== 'all' && (kind.value === 'all' || kind.value === k.id))
    .map((k) => ({ ...k, recipes: shown.value.filter((x) => x.kind === k.id) }))
    .filter((g) => g.recipes.length),
)
// the photos: two whole rows, then the rest on asking (38 squares push the rest of the page away)
const allPhotos = ref(false)
const thumbsEl = ref<HTMLElement | null>(null)
const perRow = ref(7)
const FEW = computed(() => perRow.value * 2)
const rowWatch = new ResizeObserver(([e]) => {
  const w = e?.contentRect.width ?? 0
  perRow.value = Math.max(1, Math.floor((w + 6) / (72 + 6))) // the grid's 72px columns and 6px gap
})
watch(thumbsEl, (el, was) => {
  if (was) rowWatch.unobserve(was)
  if (el) rowWatch.observe(el)
})

// ---- the photos ----------------------------------------------------------------------------
const name = ref('')
const trigger = ref('')
const triggerEdited = ref(false)
watch(name, (n) => {
  if (!triggerEdited.value) trigger.value = triggerFrom(n)
})
const datasets = ref<DatasetInfo[]>([])
const source = ref<'new' | 'existing'>('new')
const existing = ref('')
const choosing = ref(true) // the list open; picking one closes it to that one, Change opens it again
// the datasets you have, newest first; searchable once there are many
const dsQuery = ref('')
const dsShown = computed(() => {
  const q = dsQuery.value.trim().toLowerCase()
  return [...datasets.value].filter((d) => d.images && (!q || d.folder.toLowerCase().includes(q))).sort((a, b) => (b.modified ?? 0) - (a.modified ?? 0))
})
const dsThumb = (d: DatasetInfo, t: DatasetInfo['thumbs'][number]) =>
  thumbUrl(d.folder, t.path ?? t.subfolder.split('/').slice(1).concat(t.filename).join('/'), 160)
const ago = (t?: number) => {
  if (!t) return ''
  const s = Date.now() / 1000 - t
  return s < 3600 ? `${Math.max(1, Math.round(s / 60))}m ago` : s < 86400 ? `${Math.round(s / 3600)}h ago` : `${Math.round(s / 86400)}d ago`
}
const dsName = computed(() => (source.value === 'existing' ? existing.value : name.value ? `simple_${triggerFrom(name.value)}` : ''))
const items = ref<DatasetItem[]>([])
const uploading = ref(false)
const error = ref<string | null>(null)

async function loadItems() {
  if (!dsName.value) return (items.value = [])
  try {
    items.value = (await engine.dataset(dsName.value)).items
  } catch {
    items.value = []
  }
  checkPhotos()
}

// ---- each photo checked: the dataset's own picture checks (blurry, small, duplicate...) and, with
// faces on, its faces (none, several, someone else's), shown on the thumbnails ----------
const flags = ref<Record<string, string[]>>({})
const faces = ref<DatasetFaces | null>(null)
const checking = ref(false)
let checkTimer: ReturnType<typeof setTimeout> | undefined
function checkPhotos() {
  clearTimeout(checkTimer)
  if (!dsName.value || !items.value.length) {
    flags.value = {}
    faces.value = null
    return
  }
  checkTimer = setTimeout(async () => {
    const ds = dsName.value
    checking.value = true
    try {
      const [st, fc] = await Promise.all([
        engine.datasetStats(ds).catch(() => null),
        r.value?.kind === 'character' ? engine.datasetFaces(ds).catch(() => null) : Promise.resolve(null),
      ])
      if (ds !== dsName.value) return
      const f: Record<string, string[]> = {}
      for (const [k, v] of Object.entries(st?.current.flags ?? {})) f[k.replace(/^[\\/]+/, '')] = v
      flags.value = f
      faces.value = fc?.available ? fc : null
    } finally {
      checking.value = false
    }
  }, 700)
}
// what's worth fixing, in plain words: the serious ones mark the photo
const SERIOUS = new Set(['blurry', 'small', 'near-duplicate', 'no face', 'several people', 'someone else?'])
const WORDS: Record<string, string> = {
  blurry: 'blurry', small: 'low resolution', 'near-duplicate': 'a near-duplicate', similar: 'very like another',
  dark: 'dark', 'blown out': 'blown out', flat: 'flat contrast', bars: 'has borders', greyscale: 'black and white',
  'no face': 'no face found', 'several people': 'more than one person', 'someone else?': 'may be someone else',
}
function issuesOf(it: DatasetItem): string[] {
  const out = [...(flags.value[it.path] ?? [])]
  const fc = faces.value?.items?.[it.path]
  if (fc) {
    if (fc.faces === 0) out.push('no face')
    else if (fc.faces > 1) out.push('several people')
    else if (fc.outlier) out.push('someone else?')
  }
  return out
}
const serious = (it: DatasetItem) => issuesOf(it).some((i) => SERIOUS.has(i))
const verdict = computed(() => {
  if (!items.value.length) return ''
  const bad = items.value.filter(serious).length
  const mild = items.value.filter((it) => !serious(it) && issuesOf(it).length).length
  if (!bad && !mild) return `All ${items.value.length} look good`
  return [`${items.value.length - bad - mild} look good`, bad ? `${bad} worth replacing` : '', mild ? `${mild} could be better` : '']
    .filter(Boolean)
    .join(' · ')
})
watch(dsName, loadItems)
// the panel's example prompts, with this LoRA's word in them
const word = computed(() => trigger.value || (r.value?.kind === 'style' ? 'mystyle' : 'yourname'))
const examples = computed(() => (r.value?.samples ?? []).map((s) => s.replaceAll('[trigger]', word.value)))
const editingWord = ref(false)
// what's still missing before it can train, in order
const needs = computed(() => {
  const out: string[] = []
  if (!name.value.trim()) out.push('Give it a name')
  if (r.value && !enough.value) out.push(count.value ? `Add ${r.value.photos.min - count.value} more photo${r.value.photos.min - count.value === 1 ? '' : 's'}` : `Add ${r.value.photos.min} or more photos`)
  return out
})
// a photo full size: its caption and anything worth fixing; ←/→ through them all
const lb = ref<number | null>(null)
const lbItems = computed<LightboxItem[]>(() =>
  items.value.map((it) => ({
    src: thumbUrl(dsName.value, it.path, 1600),
    label: it.filename,
    meta: [issuesOf(it).map((i) => WORDS[i] ?? i).join(', '), it.caption].filter(Boolean).join(' · '),
  })),
)

async function addFiles(files: File[]) {
  const pics = files.filter((f) => /^image\//.test(f.type))
  if (!pics.length) return
  if (!name.value.trim()) {
    error.value = 'Give it a name first: the photos are kept in a folder called after it.'
    return
  }
  uploading.value = true
  error.value = null
  try {
    if (!datasets.value.some((d) => d.folder === dsName.value)) {
      await engine.createDataset(dsName.value).catch(() => {})
      datasets.value = await engine.datasets()
    }
    const res = await engine.upload(dsName.value, pics)
    if (res.errors.length) error.value = res.errors.join('; ')
    await loadItems()
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    uploading.value = false
  }
}
const over = ref(false)
function onDrop(e: DragEvent) {
  over.value = false
  addFiles([...(e.dataTransfer?.files ?? [])])
}
const picker = ref<HTMLInputElement | null>(null)
async function removeItem(it: DatasetItem) {
  await engine.deleteItem(dsName.value, it.path)
  await loadItems()
}

const count = computed(() => items.value.length)
const uncaptioned = computed(() => items.value.filter((i) => !i.caption).length)
const enough = computed(() => !!r.value && count.value >= r.value.photos.min)

// ---- the plan: files, downloads, time --------------------------------------------------------
const p = ref<Plan | null>(null)
watch(
  r,
  async (x) => {
    p.value = null
    if (x) p.value = await plan(x).catch((e) => ({ problems: [String(e)] }) as Plan)
  },
  { immediate: true },
)
const downloadBytes = computed(() => (p.value?.downloads ?? []).reduce((n, d) => n + (d.size ?? 0), 0) + (p.value?.adapter?.size ?? 0))
const steps = computed(() => r.value?.steps ?? 0)

// ---- training: downloads, then captions, then the run ---------------------------------------
type Phase = 'idle' | 'downloading' | 'captioning' | 'starting'
const phase = ref<Phase>('idle')
const capJob = ref<string | null>(null)
const pendingDownloads = computed(() =>
  (p.value?.downloads ?? []).map((d) => ({ ...d, job: downloadFor(d.folder, d.name) })),
)
const capProgress = computed(() => {
  const j = capJob.value ? captionJobs[capJob.value] : undefined
  return j ? { done: j.done, total: j.total || count.value, status: j.status, loading: j.message?.startsWith('loading') } : null
})

function waitFor(check: () => boolean | string, every = 1000): Promise<void> {
  return new Promise((resolve, reject) => {
    const tick = () => {
      const v = check()
      if (typeof v === 'string') return reject(new Error(v))
      if (v) return resolve()
      setTimeout(tick, every)
    }
    tick()
  })
}

// the best describer installed: bigger and newer describe better (Qwen3.5 9B, then Qwen3-VL 8B, then 4B)
function describer(models: string[]): string | undefined {
  const order = [/qwen3\.?5[-_]?9b(?!.*(flux|prompt|upsampl|pe_))/i, /qwen3.?vl[-_]?8b/i, /qwen3.?vl[-_]?4b/i, /qwen3.?vl/i, /gemma[-_]?4/i]
  for (const re of order) {
    const hit = models.find((m) => re.test(m))
    if (hit) return hit
  }
  return models[0]
}
const describing = ref(false)
async function describe() {
  const x = r.value
  if (!x) return
  await loadItems()
  if (!items.value.some((i) => !i.caption)) return
  const info = await engine.captionInfo(dsName.value)
  const model = describer(info.models)
  if (!model) throw new Error('no vision-language model to describe the photos with: install Qwen3-VL (Models page)')
  const { job } = await engine.startCaption(dsName.value, {
    clip_name: model,
    clip_type: 'auto',
    set: 'default',
    mode: 'missing',
    ...(x.caption.style ? { style: x.caption.style } : { prompt: x.caption.prompt }),
    subject: x.caption.subject ? trigger.value : undefined,
  } as Parameters<typeof engine.startCaption>[1])
  capJob.value = job.job_id
  await waitFor(() => {
    const j = captionJobs[job.job_id]
    if (j?.status === 'error') return `describing failed: ${j.message}`
    if (j?.status === 'stopped') return 'describing was stopped'
    return j?.status === 'done'
  })
  await loadItems()
}
async function describeNow() {
  error.value = null
  describing.value = true
  phase.value = 'captioning'
  try {
    await describe()
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : (e as Error).message ?? String(e)
  } finally {
    describing.value = false
    phase.value = 'idle'
  }
}

async function train() {
  const x = r.value
  if (!x || !p.value) return
  error.value = null
  try {
    // 1. models
    if (p.value.downloads.length || p.value.adapter) {
      phase.value = 'downloading'
      for (const d of p.value.downloads) await engine.download(x.family, d.folder, d.name)
      // the adapter too: a run that starts before it lands is refused (no LoRA ... here)
      const adapter = p.value.adapter ? await engine.downloadAdapter(p.value.adapter.id) : null
      await waitFor(() => {
        const jobs = p.value!.downloads.map((d) => downloadFor(d.folder, d.name))
        // by its own id: an older try at the same file (failed, say) mustn't stand in for it
        if (adapter) jobs.push(downloads[adapter.id] ?? adapter)
        const bad = jobs.find((j) => j && (j.status === 'error' || j.status === 'cancelled'))
        if (bad) return `${bad.name}: ${bad.error ?? bad.status}`
        return jobs.every((j) => j?.status === 'done')
      })
      p.value = await plan(x) // now installed: the names ComfyUI lists them by
    }
    // 2. descriptions for photos without one
    await loadItems()
    if (items.value.some((i) => !i.caption)) {
      phase.value = 'captioning'
      await describe()
    }
    // 3. the run
    phase.value = 'starting'
    const base = defaultSpec()
    base.output.keep_states = settings.keepStates
    await loadItems()
    const spec = buildSpec(x, p.value, base, {
      name: name.value.trim(),
      trigger: trigger.value,
      dataset: dsName.value,
      captions: items.value.map((i) => i.caption),
      masks: items.value.filter((i) => i.mask).length,
    })
    const run = await engine.start(spec)
    router.push(`/simple/run/${run.run_id}`)
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : (e as Error).message ?? String(e)
    phase.value = 'idle'
  }
}

onMounted(async () => {
  startDownloadSync()
  startCaptionSync()
  datasets.value = await engine.datasets().catch(() => [])
})
watch(finished, () => r.value && plan(r.value).then((x) => phase.value === 'idle' && (p.value = x)))
let leaving = false
onUnmounted(() => {
  leaving = true
  rowWatch.disconnect()
})
void leaving
</script>

<template>
  <div class="page simple">
    <!-- the gallery -->
    <template v-if="!r">
      <div class="page-head">
        <div class="titles">
          <h1>Train a LoRA</h1>
          <p class="lede">Pick what you want to teach. Everything is set up for you.</p>
        </div>
        <RouterLink to="/runs/new"><ZenButton variant="ghost" icon="mdi mdi-tune-variant">All settings</ZenButton></RouterLink>
      </div>
      <div class="kinds">
        <button v-for="k in kinds" :key="k.id" :class="{ on: kind === k.id }" @click="kind = k.id">{{ k.label }}</button>
      </div>
      <section v-for="g in groups" :key="g.id" class="group">
        <div v-if="kind === 'all'" class="group-title">{{ g.label }}<span>{{ g.recipes.length }}</span></div>
        <div class="cards">
            <TemplateCard
              v-for="x in g.recipes"
              :key="x.id"
              :to="`/simple/${x.id}`"
              :title="x.title.split(' · ')[1] ?? x.title"
              :description="x.blurb"
              :image="x.thumb ? asset(x.thumb.src) : null"
              :after="x.thumb?.after ? asset(x.thumb.after) : null"
              :mode="x.thumb?.mode"
              :before-label="x.thumb?.before_label"
              :after-label="x.thumb?.after_label"
              picture-only
              :logo="makerLogo(x.family)?.src ?? null"
              :tags="[x.model]"
              :flag="x.validated ? undefined : 'draft'"
              flag-title="Not trained and checked end to end yet: settings from similar runs"
              :meta="[x.photos.ideal]"
            />
        </div>
      </section>
    </template>

    <!-- one recipe -->
    <template v-else>
      <div class="page-head">
        <div class="titles">
          <h1><RouterLink to="/simple" class="crumb">Train</RouterLink> {{ r.title }}</h1>
          <p class="lede">{{ r.blurb }}</p>
        </div>
      </div>

      <div class="recipe">
        <div class="flow">
          <section class="card step">
            <div class="num">1</div>
            <div class="sbody">
              <h2>What's it called?</h2>
              <ZenInput v-model="name" :placeholder="r.kind === 'style' ? 'e.g. Watercolour postcards' : r.kind === 'general' ? 'e.g. Biscuit the dog' : 'e.g. Grandma Rose'" />
              <div class="trig small">
                <span class="muted">You'll bring it in with</span>
                <input v-if="editingWord" v-model="trigger" class="trig-in mono" spellcheck="false" autofocus @input="triggerEdited = true" @blur="editingWord = false" @keydown.enter="editingWord = false" />
                <button v-else class="trig-chip mono" :class="{ empty: !trigger }" title="Change the word" @click="editingWord = true">
                  {{ trigger || 'a word from the name' }} <i class="mdi mdi-pencil-outline" />
                </button>
                <span class="muted">in a prompt.</span>
              </div>
            </div>
          </section>

          <section class="card step" :class="{ locked: !name.trim() }">
            <div class="num">2</div>
            <div v-if="!name.trim()" class="sbody">
              <h2>{{ r.kind === 'style' ? 'Add images in the style' : 'Add photos' }}</h2>
              <p class="muted small locked-note">Name it first: the photos are kept in a folder called after it.</p>
            </div>
            <div v-else class="sbody">
              <div class="h-row">
                <h2>{{ r.kind === 'style' ? 'Add images in the style' : 'Add photos' }}</h2>
                <div class="seg small">
                  <button :class="{ on: source === 'new' }" @click="source = 'new'">New photos</button>
                  <button :class="{ on: source === 'existing' }" :disabled="!datasets.length" @click="source = 'existing'">A dataset I have</button>
                </div>
              </div>
              <div v-if="source === 'existing'" class="ds-pick">
                <div v-if="existing && !choosing" class="ds-chosen">
                  <template v-for="d in datasets.filter((x) => x.folder === existing)" :key="d.folder">
                    <img v-if="d.thumbs[0]" class="ds-thumb" :src="dsThumb(d, d.thumbs[0])" alt="" />
                    <span class="ds-text">
                      <span class="ds-name ellipsis">{{ d.folder }}</span>
                      <span class="ds-meta small muted">{{ d.images }} {{ d.images === 1 ? 'image' : 'images' }}<template v-if="d.modified"> · {{ ago(d.modified) }}</template></span>
                    </span>
                  </template>
                  <ZenButton size="sm" icon="mdi mdi-swap-horizontal" @click="choosing = true">Change</ZenButton>
                </div>
                <ZenInput v-if="choosing && datasets.length > 8" v-model="dsQuery" size="sm" placeholder="Search your datasets" />
                <div v-if="choosing" class="ds-grid">
                  <button v-for="d in dsShown" :key="d.folder" class="ds" :class="{ on: existing === d.folder }" :title="d.folder" @click="(existing = d.folder), (choosing = false)">
                    <img v-if="d.thumbs[0]" class="ds-thumb" :src="dsThumb(d, d.thumbs[0])" loading="lazy" alt="" />
                    <span v-else class="ds-thumb none"><i class="mdi mdi-image-off-outline" /></span>
                    <span class="ds-text">
                      <span class="ds-name ellipsis">{{ d.folder }}</span>
                      <span class="ds-meta small muted">{{ d.images }} {{ d.images === 1 ? 'image' : 'images' }}<template v-if="d.modified"> · {{ ago(d.modified) }}</template></span>
                    </span>
                    <CaptionRing :done="d.captioned" :total="d.images" :scores="d.health?.scores" :size="22" />
                  </button>
                  <div v-if="!dsShown.length" class="muted small">No dataset matches “{{ dsQuery }}”.</div>
                </div>
              </div>
              <div
                v-else
                class="drop"
                :class="{ over, compact: items.length > 0 }"
                @dragover.prevent="over = true"
                @dragleave="over = false"
                @drop.prevent="onDrop"
                @click="picker?.click()"
              >
                <i :class="uploading ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-image-plus-outline'" />
                <span>{{ uploading ? 'Adding…' : items.length ? 'Drop more, or click to add' : 'Drop photos here, or click to choose' }}</span>
                <input ref="picker" type="file" accept="image/*" multiple hidden @change="addFiles([...(($event.target as HTMLInputElement).files ?? [])])" />
              </div>
              <div v-if="!items.length" class="tips">
                <span class="muted small"><b>{{ r.photos.ideal }}</b> works best:</span>
                <span v-for="t in r.photos.tips" :key="t" class="tip small"><i class="mdi mdi-check" />{{ t }}</span>
              </div>
              <template v-else>
                <div class="verdict small">
                  <span :class="{ warn: !enough }">
                    <b>{{ count }}</b> {{ count === 1 ? 'photo' : 'photos' }}<template v-if="!enough"> · add at least {{ r.photos.min - count }} more</template>
                  </span>
                  <span v-if="checking" class="muted"><i class="mdi mdi-loading mdi-spin" /> checking…</span>
                  <span v-else-if="verdict" class="muted">{{ verdict }}</span>
                </div>
                <div ref="thumbsEl" class="thumbs">
                  <div v-for="(it, i) in allPhotos ? items : items.slice(0, FEW)" :key="it.path" class="th" :class="{ bad: serious(it), meh: !serious(it) && issuesOf(it).length }"
                       :title="issuesOf(it).length ? issuesOf(it).map((x) => WORDS[x] ?? x).join(', ') : 'Looks good'" @click="lb = i">
                    <img :src="thumbUrl(dsName, it.path, 192)" loading="lazy" alt="" />
                    <span v-if="issuesOf(it).length" class="flag"><i :class="serious(it) ? 'mdi mdi-alert' : 'mdi mdi-information-outline'" /></span>
                    <button v-if="source === 'new'" class="x" title="Remove" @click.stop="removeItem(it)"><i class="mdi mdi-close" /></button>
                  </div>
                </div>
                <button v-if="items.length > FEW" class="more small" @click="allPhotos = !allPhotos">
                  {{ allPhotos ? 'Show fewer' : `Show all ${items.length}` }} <i :class="allPhotos ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
                </button>
                <ZenLightbox v-if="lb != null" v-model:index="lb" :items="lbItems" :searchable="false" @close="lb = null" />
                <div class="describe">
                  <i class="mdi mdi-text-box-outline" />
                  <div class="describe-text small">
                    <b>{{ uncaptioned ? `${uncaptioned} to describe` : 'All described' }}</b>
                    <span class="muted">
                      Each photo gets a short description, so it learns {{ r.kind === 'style' ? 'the look, not the things in them' : 'what’s in them' }}; your word goes in for you.
                      {{ r.caption.style === 'fibo_json' ? 'Written in Fibo’s own structured format.' : '' }}
                      Open a photo to read its description.
                    </span>
                  </div>
                  <ZenButton v-if="uncaptioned" size="sm" icon="mdi mdi-creation-outline" :disabled="phase !== 'idle'" @click="describeNow">
                    {{ describing ? 'Describing…' : 'Describe now' }}
                  </ZenButton>
                  <RouterLink v-else :to="`/datasets/${encodeURIComponent(dsName)}`" class="small">Edit them</RouterLink>
                </div>
              </template>
            </div>
          </section>
        </div>

        <!-- what you'll get -->
        <aside class="card get">
          <div v-if="r.thumb" class="cover"><img :src="asset(r.thumb.src)" alt="" /></div>
          <div class="get-body">
            <div class="model">
              <img v-if="makerLogo(r.family)" class="maker" :class="{ mono: makerLogo(r.family)!.mono }" :src="makerLogo(r.family)!.src" alt="" />
              <b>{{ r.model }}</b>
              <span class="muted small">{{ makerLogo(r.family)?.maker }}</span>
            </div>
            <dl class="facts">
              <dt>Steps</dt><dd>{{ steps.toLocaleString() }}</dd>
              <dt>Photos</dt><dd>{{ r.photos.ideal }}</dd>
              <dt>Models</dt>
              <dd>
                <span v-if="!p"><i class="mdi mdi-loading mdi-spin" /></span>
                <span v-else-if="!p.downloads.length && !p.adapter" class="ok"><i class="mdi mdi-check-circle" /> ready</span>
                <span v-else><i class="mdi mdi-download-outline" /> {{ fmtBytes(downloadBytes) }} to download</span>
              </dd>
            </dl>
            <!-- training: always in view, beside the steps -->
            <div class="go-box">
              <div v-for="q in p?.problems ?? []" :key="q" class="err small"><i class="mdi mdi-alert-circle-outline" /> {{ q }}</div>
              <div v-if="phase !== 'idle'" class="doing">
                <template v-if="phase === 'downloading'">
                  <div v-for="d in pendingDownloads" :key="d.name" class="prow">
                    <span class="ellipsis small">Downloading {{ d.name }}</span>
                    <div class="pbar"><span :style="{ width: `${d.job?.total ? (d.job.done / d.job.total) * 100 : 0}%` }" /></div>
                  </div>
                </template>
                <div v-else-if="phase === 'captioning'" class="prow">
                  <span class="small">{{ capProgress?.loading ? 'Getting the describer ready…' : `Describing photos ${capProgress?.done ?? 0} of ${capProgress?.total ?? count}` }}</span>
                  <div class="pbar"><span :style="{ width: `${capProgress?.total ? (capProgress.done / capProgress.total) * 100 : 0}%` }" /></div>
                </div>
                <div v-else class="prow small"><i class="mdi mdi-loading mdi-spin" /> Starting…</div>
              </div>
              <ul v-else-if="needs.length" class="needs small">
                <li v-for="n in needs" :key="n"><i class="mdi mdi-circle-small" />{{ n }}</li>
              </ul>
              <div v-else class="muted small">
                {{ uncaptioned ? `Describes ${uncaptioned} photo${uncaptioned === 1 ? '' : 's'} first, then trains.` : 'Every photo is described: training starts right away.' }}
              </div>
              <div v-if="error" class="err small">{{ error }}</div>
              <ZenButton
                class="go"
                variant="primary"
                icon="mdi mdi-play"
                :disabled="phase !== 'idle' || !!needs.length || !p || !!p.problems.length"
                @click="train"
              >
                Train {{ name.trim() || 'it' }}
              </ZenButton>
            </div>
            <div class="eyebrow">Try it with prompts like</div>
            <div class="examples">
              <span v-for="e in examples" :key="e" class="small">{{ e }}</span>
            </div>
            <p v-if="r.note" class="muted small note">{{ r.note }}</p>
            <p v-if="!r.validated" class="muted small note"><i class="mdi mdi-flask-outline" /> A new recipe: its settings come from similar runs, not yet one trained end to end.</p>
          </div>
        </aside>
      </div>
    </template>
  </div>
</template>

<style scoped>
.kinds,
.seg {
  display: flex;
  gap: 4px;
}
.kinds button,
.seg button {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 5px 12px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.kinds button.on,
.seg button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.seg button:disabled {
  opacity: 0.4;
  cursor: default;
}
.simple {
  width: 100%;
  max-width: 1080px;
  margin-inline: auto;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}
.group {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.group + .group {
  margin-top: 10px;
}
.group-title {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted);
}
.group-title span {
  font-weight: 400;
  opacity: 0.7;
}
.more {
  align-self: flex-start;
  border: 0;
  background: none;
  padding: 0;
  color: var(--zen-accent);
  font: inherit;
  cursor: pointer;
}
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
.recipe {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 16px;
  align-items: start;
}
.flow {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}
.h-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}
/* what you'll get: the recipe at a glance, beside the steps */
.get {
  position: sticky;
  top: 16px;
  padding: 0;
  overflow: hidden;
}
.cover {
  aspect-ratio: 16 / 7;
  display: grid;
  place-items: center;
  color: rgb(255 255 255 / 0.85);
  font-size: 44px;
}
.cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.get-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px 16px 16px;
}
.model {
  display: flex;
  align-items: center;
  gap: 8px;
}
.model .maker {
  width: 22px;
  height: 22px;
  border-radius: 5px;
  object-fit: contain;
}
:global(html[data-mode='dark'] .model .maker.mono) {
  filter: invert(1);
}
.facts {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 14px;
  margin: 0;
  font-size: 12px;
}
.facts dt {
  color: var(--zen-muted);
}
.facts dd {
  margin: 0;
}
.facts .ok {
  color: var(--zen-ok);
}
.examples {
  margin: -4px 0 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: var(--zen-text);
}
/* each a prompt you could paste: on the field's colour, the word it brings in first */
.examples span {
  padding: 7px 10px;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
  border-left: 2px solid var(--zen-accent);
  line-height: 1.4;
}
.step.locked .num {
  opacity: 0.45;
}
.locked-note {
  margin: 0;
}
.describe {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
}
.describe > i {
  font-size: 18px;
  color: var(--zen-muted);
}
.describe-text {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  line-height: 1.45;
}
.go-box {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-accent) 6%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 25%, transparent);
}
.go-box :deep(.zen-button.go) {
  justify-content: center;
}
.needs {
  margin: 0;
  padding: 0;
  list-style: none;
  color: var(--zen-muted);
}
.needs i {
  color: var(--zen-warn);
}
.th {
  cursor: zoom-in;
}
.note {
  margin: 0;
  line-height: 1.45;
}
@media (max-width: 920px) {
  .recipe {
    grid-template-columns: 1fr;
  }
  .get {
    position: static;
  }
}
.step {
  display: grid;
  grid-template-columns: 36px 1fr;
  gap: 14px;
  padding: 18px;
}
.num {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: color-mix(in srgb, var(--zen-accent) 18%, transparent);
  color: var(--zen-accent);
  font-weight: 700;
}
.sbody {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}
.sbody h2 {
  margin: 4px 0 0;
  font-size: 16px;
}
.trig {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.trig-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 8px;
  border: 1px solid color-mix(in srgb, var(--zen-accent) 45%, transparent);
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-accent) 12%, transparent);
  color: var(--zen-text);
  font-size: 12px;
  cursor: pointer;
}
.trig-chip.empty {
  border-style: dashed;
  color: var(--zen-muted);
  background: none;
}
.trig-chip i {
  font-size: 12px;
  opacity: 0.6;
}
.trig-in {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
  border-radius: var(--r-sm);
  padding: 2px 6px;
  font-size: 12px;
  width: 16ch;
}
.drop {
  border: 2px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  border-radius: var(--r-lg, 12px);
  padding: 28px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--zen-muted);
  cursor: pointer;
  transition:
    border-color var(--zen-dur-fast),
    background var(--zen-dur-fast);
}
.drop i {
  font-size: 28px;
}
.drop.compact {
  flex-direction: row;
  justify-content: center;
  padding: 12px;
}
.drop.compact i {
  font-size: 20px;
}
.verdict {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  align-items: center;
}
.drop:hover,
.drop.over {
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 6%, transparent);
}
.sel {
  padding: 8px;
  border-radius: var(--r-md);
  border: 1px solid var(--zen-border);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
  font: inherit;
}
.ds-pick {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ds-chosen {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 6px 6px 6px;
  border: 1px solid var(--zen-selected, var(--zen-accent));
  border-radius: var(--r-md);
  background: var(--zen-selected-bg, color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 8%, transparent));
}
.ds-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 4px;
  max-height: 232px;
  overflow: auto;
  padding: 2px;
}
.ds {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  padding: 5px 8px 5px 5px;
  border: 1px solid transparent;
  border-radius: var(--r-md);
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.ds:hover {
  background: var(--zen-surface-2);
}
.ds.on {
  border-color: var(--zen-selected, var(--zen-accent));
  background: var(--zen-selected-bg, color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 8%, transparent));
}
.ds-thumb {
  width: 36px;
  height: 36px;
  flex: none;
  border-radius: var(--r-sm);
  object-fit: cover;
  background: var(--zen-surface-2);
  display: grid;
  place-items: center;
  color: var(--zen-muted);
}
.ds-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}
.ds-name {
  font-size: 12px;
  font-weight: 600;
}
.ds-meta {
  font-size: 11px;
}
.tips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  align-items: center;
}
.tip i {
  color: var(--zen-ok);
  margin-right: 4px;
}
.thumbs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
  gap: 6px;
}
.th {
  position: relative;
  aspect-ratio: 1;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.th img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.th .x {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: none;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--zen-dur-fast);
}
.th:hover .x {
  opacity: 1;
}
/* a photo worth replacing (blurry, tiny, a duplicate, the wrong or no face), or one that could be better */
.th.bad {
  outline: 2px solid var(--zen-danger);
  outline-offset: -2px;
}
.th.meh {
  outline: 2px solid color-mix(in srgb, var(--zen-warn) 70%, transparent);
  outline-offset: -2px;
}
.th .flag {
  position: absolute;
  left: 4px;
  bottom: 4px;
  display: grid;
  place-items: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: rgb(0 0 0 / 0.6);
  font-size: 13px;
}
.th.bad .flag {
  color: #ff8a80;
}
.th.meh .flag {
  color: #ffd180;
}
.count.warn,
.warn {
  color: var(--zen-warn);
}
.strip-sum {
  display: flex;
  flex-wrap: wrap;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  overflow: hidden;
}
.strip-sum > span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 12px;
  font-size: 12px;
  border-right: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.strip-sum > span:last-child {
  border-right: none;
}
.strip-sum .ok i {
  color: var(--zen-ok);
}
.strip-sum .warn {
  color: var(--zen-warn);
}
.checks {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.checks i {
  margin-right: 6px;
  color: var(--zen-muted);
}
.checks i.ok {
  color: var(--zen-ok);
}
.doing {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.prow {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.pbar {
  height: 6px;
  border-radius: 3px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.pbar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
.err {
  color: var(--zen-danger);
}
.step :deep(.zen-button) {
  align-self: flex-start;
}
.step :deep(.zen-button.go) {
  align-self: stretch;
  justify-content: center;
}
</style>
