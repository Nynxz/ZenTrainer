<script setup lang="ts">
// The run form's recipes: Simple mode's, listed by model, and your own (saved from the form), and
// for the one picked what it would do to this run (its model files, a length for this dataset,
// each setting it changes) before it's applied. The run's name, dataset, trigger and captions
// stay; Undo takes it back. The dialog has a fixed height: the list and the detail each scroll on
// their own (the app's overlay scrollbars). A search narrows the list; on a phone it fills the
// screen and goes list, then detail.
import { ZenButton, ZenInput, ZenModal } from '@nynxz/zenkit-ui'
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, toRaw, watch } from 'vue'
import type { Preset, TrainSpec } from '@/api/engine'
import ScrollArea from '@/components/ScrollArea.vue'
import { makerLogo } from '@/lib/makers'
import { BIG_DATASET, RECIPES, applyTo, changes, plan, type Plan, type Recipe } from '@/lib/recipes'
import { SECTIONS, fmt, full, label } from '@/lib/specText'
import { fmtBytes } from '@/stores/downloads'
import { merge, useRunFormContext } from './form'

const open = defineModel<boolean>('open', { default: false })
const { spec, familyId, dataset, runPresets, applyRecipe, deleteRecipe } = useRunFormContext()

type Mine = Preset<Partial<TrainSpec>>
/** one row of the list, a built-in recipe or one of yours, as the list and the detail show it */
interface Entry {
  id: string
  title: string
  name: string
  model: string
  blurb: string
  adapter: string
  meta: string
  thumb: string | null
  cover: { from: string; to: string; icon: string }
  draft: boolean
  note?: string
  recipe?: Recipe
  mine?: Mine
}
const KINDS = { character: 'Character', style: 'Style', video: 'Video', general: 'Any subject' } as const
const asset = (p: string) => (p.startsWith('/') ? p : `${import.meta.env.BASE_URL}${p}`)
const builtIn = (r: Recipe): Entry => {
  const adapter = r.spec.network?.algorithm ?? 'LoRA' // a LoKr's rank isn't a LoRA's, so it's named
  return {
    id: r.id, title: r.title, name: r.title.split(' · ')[1] ?? r.title, model: r.model, blurb: r.blurb, adapter,
    meta: `${adapter} · ${r.photos.ideal}`, thumb: r.thumb ? asset(r.thumb.src) : null, cover: r.cover, draft: !r.validated,
    note: r.note, recipe: r,
  }
}
const yours = (p: Mine): Entry => {
  const adapter = p.data.network?.algorithm ?? 'LoRA'
  const file = p.data.model?.unet_name ?? p.data.model?.ckpt_name
  return {
    id: `mine:${p.id}`, title: p.name, name: p.name, model: 'Yours', adapter,
    blurb: p.description || (file ? `Your settings, on ${short(file)}.` : 'Your settings (keeps the model you pick).'),
    meta: `${adapter}${p.data.optim?.steps ? ` · ${p.data.optim.steps.toLocaleString()} steps` : ''}`,
    thumb: null, cover: { from: 'var(--zen-accent)', to: 'color-mix(in srgb, var(--zen-accent) 35%, #000)', icon: 'mdi mdi-account-outline' },
    draft: false, mine: p,
  }
}

const query = ref('')
// every word somewhere in what the row says (name, model, kind, description, tips)
const haystack = (e: Entry) =>
  [e.title, e.model, e.adapter, e.blurb, e.note ?? '', ...(e.recipe ? [KINDS[e.recipe.kind], ...e.recipe.photos.tips] : [])].join(' ').toLowerCase()
const matches = (e: Entry) => query.value.toLowerCase().split(/\s+/).filter(Boolean).every((w) => haystack(e).includes(w))
// built in by model (the form's own first), then yours
const groups = computed(() => {
  const out: { key: string; label: string; family: string | null; entries: Entry[] }[] = []
  for (const r of RECIPES) {
    const e = builtIn(r)
    if (!matches(e)) continue
    let g = out.find((x) => x.family === r.family)
    if (!g) out.push((g = { key: r.family, label: r.model, family: r.family, entries: [] }))
    g.entries.push(e)
  }
  out.sort((a, b) => Number(b.family === familyId.value) - Number(a.family === familyId.value))
  const mine = runPresets.value.map(yours).filter(matches)
  if (mine.length) out.push({ key: 'mine', label: 'Yours', family: null, entries: mine })
  return out
})
const flat = computed(() => groups.value.flatMap((g) => g.entries))

const picked = shallowRef<Entry | null>(null) // as it is: a ref's proxy can't be copied into a spec
// every built-in recipe's files, worked out once when the dialog opens: picking one is then
// instant, with nothing in the detail swapping out while it loads (a string: why it couldn't be)
const plans = ref<Record<string, Plan | string>>({})
async function planAll() {
  plans.value = {}
  await Promise.all(RECIPES.map(async (r) => {
    try {
      plans.value[r.id] = await plan(r)
    } catch (e) {
      plans.value[r.id] = e instanceof Error ? e.message : String(e)
    }
  }))
}
const planOf = computed(() => (picked.value?.recipe ? plans.value[picked.value.recipe.id] : undefined))
const planned = computed(() => (typeof planOf.value === 'object' ? planOf.value : null))
const failed = computed(() => (typeof planOf.value === 'string' ? planOf.value : null))
const planning = computed(() => !!picked.value?.recipe && planOf.value === undefined)

// a phone: the dialog fills the screen and shows the list, then an entry's detail
const phoneQuery = window.matchMedia('(max-width: 720px)')
const phone = ref(phoneQuery.matches)
const onPhone = (e: MediaQueryListEvent) => (phone.value = e.matches)
phoneQuery.addEventListener('change', onPhone)
onBeforeUnmount(() => phoneQuery.removeEventListener('change', onPhone))
const step = ref<'list' | 'detail'>('list')

watch(open, async (o) => {
  if (!o) return
  void planAll()
  query.value = ''
  step.value = 'list'
  confirmDelete.value = false
  picked.value = flat.value[0] ?? null // never an empty detail
  if (!phone.value) {
    await nextTick()
    document.querySelector<HTMLInputElement>('.recipe-search input')?.focus()
  }
})
// a search (or a delete) that hides the picked entry picks the first that's left
watch(flat, (list) => {
  if (!list.some((e) => e.id === picked.value?.id)) picked.value = list[0] ?? null
})
function choose(e: Entry) {
  picked.value = e
  step.value = 'detail'
}
// a new entry's detail starts at its top, and the bar measures its new length
const detailScroll = ref<InstanceType<typeof ScrollArea> | null>(null)
watch(picked, async () => {
  confirmDelete.value = false
  await nextTick()
  if (detailScroll.value?.viewport) detailScroll.value.viewport.scrollTop = 0
})
watch([picked, planned], async () => {
  await nextTick()
  detailScroll.value?.refresh()
})

const photos = computed(() => dataset.value?.images ?? null)
/** yours over the form: its settings and model; the run's name, dataset, trigger and captions stay */
function mineOver(p: Mine, current: TrainSpec): TrainSpec {
  const out = JSON.parse(JSON.stringify(current)) as TrainSpec // the form's: parts of it are proxies
  merge(out as unknown as Record<string, unknown>, JSON.parse(JSON.stringify(p.data)) as Record<string, unknown>)
  const d = current.dataset
  Object.assign(out.dataset, { folder: d.folder, trigger: d.trigger, trigger_prepend: d.trigger_prepend, caption_set: d.caption_set, caption_mix: d.caption_mix })
  out.name = current.name
  return out
}
const built = computed<{ spec: TrainSpec | null; error: string | null }>(() => {
  const e = picked.value
  if (!e) return { spec: null, error: null }
  try {
    if (e.mine) return { spec: mineOver(e.mine, toRaw(spec) as TrainSpec), error: null }
    if (!planned.value) return { spec: null, error: null }
    return { spec: applyTo(e.recipe!, planned.value, toRaw(spec) as TrainSpec), error: null }
  } catch (err) {
    return { spec: null, error: err instanceof Error ? err.message : String(err) }
  }
})
const next = computed(() => built.value.spec)

// where a built-in recipe's length comes from: the run it was checked at, or a draft's guess
const stepsWhy = computed(() => {
  const r = picked.value?.recipe
  if (!r) return ''
  return r.validated ? 'the length it was checked at' : 'a draft: its best guess at a length'
})
// a big set sees each picture only a few times in a recipe's length
const bigSet = computed(() => !!picked.value?.recipe && photos.value !== null && photos.value > BIG_DATASET)
// the numbers people look at first, before and after
const stats = computed<{ label: string; from: string; to: string; why?: string }[]>(() => {
  const n = next.value
  if (!n) return []
  const now = toRaw(spec) as TrainSpec
  // a run that starts lower trains at each stage's resolution, then its own: 768 → 1024
  const resolution = (x: TrainSpec) => [...(x.dataset.stages ?? []).map((st) => st.resolution), x.dataset.resolution].join(' → ')
  const lr = (v: number) => (v < 0.001 ? v.toExponential(0).replace('e-', 'e−') : String(v))
  return [
    { label: 'Steps', from: now.optim.steps.toLocaleString(), to: n.optim.steps.toLocaleString(), why: stepsWhy.value },
    { label: 'Adapter', from: `${now.network.algorithm} · ${now.network.rank}`, to: `${n.network.algorithm} · ${n.network.rank}` },
    { label: 'Learning rate', from: lr(now.optim.learning_rate), to: lr(n.optim.learning_rate) },
    { label: 'Resolution', from: resolution(now), to: resolution(n) },
  ]
})
// the model files it trains with: a built-in's as worked out here, yours as saved
const files = computed<{ key: string; name: string | null; download: { size: number | null } | null }[]>(() => {
  const e = picked.value
  if (e?.recipe && planned.value) {
    return Object.entries(planned.value.model).map(([folder, name]) => ({
      key: folder, name, download: planned.value!.downloads.find((d) => d.folder === folder) ?? null,
    }))
  }
  const m = e?.mine?.data.model
  if (!m) return []
  return (m.kind === 'checkpoint' ? [['ckpt', m.ckpt_name]] : [['unet', m.unet_name], ['clip', m.clip_name], ['vae', m.vae_name]])
    .filter(([, n]) => n)
    .map(([key, name]) => ({ key: key!, name: name ?? null, download: null }))
})

const grouped = computed(() => {
  const out: { section: string; rows: { label: string; from: string; to: string; full: string }[] }[] = []
  if (!next.value) return out
  for (const c of changes(toRaw(spec) as TrainSpec, next.value)) {
    const [head, ...rest] = c.key.split('.')
    const section = SECTIONS[head!] ?? head!
    let g = out.find((x) => x.section === section)
    if (!g) out.push((g = { section, rows: [] }))
    g.rows.push({ label: label(rest, head!), from: fmt(c.from), to: fmt(c.to), full: `${c.key}: ${full(c.from)} → ${full(c.to)}` })
  }
  return out
})
const changeCount = computed(() => grouped.value.reduce((n, g) => n + g.rows.length, 0))

function short(name: string | null | undefined) {
  return name?.split('/').pop() ?? '—'
}
const cover = (e: Entry) => ({ background: `linear-gradient(135deg, ${e.cover.from}, ${e.cover.to})` })

const confirmDelete = ref(false)
async function remove() {
  const p = picked.value?.mine
  if (!p) return
  if (!confirmDelete.value) {
    confirmDelete.value = true
    return
  }
  await deleteRecipe(p.id)
  confirmDelete.value = false
}

function apply() {
  if (!picked.value || !next.value) return
  applyRecipe({ id: picked.value.id, title: picked.value.title }, next.value)
  open.value = false
}
</script>

<template>
  <ZenModal v-model:open="open" width="1080px" height="min(720px, 88vh)">
    <template #header>
      <div class="head" :class="{ phone }">
        <button v-if="phone && step === 'detail'" class="back" @click="step = 'list'"><i class="mdi mdi-chevron-left" /> Recipes</button>
        <div v-else class="head-text">
          <b class="zt-title">Recipes</b>
          <span class="muted">Settings for each model, sized to your dataset</span>
        </div>
        <div v-if="!(phone && step === 'detail')" class="recipe-search">
          <i class="mdi mdi-magnify" />
          <ZenInput v-model="query" size="sm" placeholder="Search recipes" />
        </div>
      </div>
    </template>

    <div class="picker" :class="{ phone, detailed: step === 'detail' }">
      <ScrollArea class="list">
        <nav class="list-in">
          <section v-for="g in groups" :key="g.key" class="group">
            <header class="g-head">
              <img v-if="g.family && makerLogo(g.family)" :src="makerLogo(g.family)!.src" alt="" />
              <i v-else-if="!g.family" class="mdi mdi-account-outline g-icon" />
              <span>{{ g.label }}</span>
              <span v-if="g.family && g.family === familyId" class="here" title="The model picked in the form">selected</span>
            </header>
            <button
              v-for="e in g.entries"
              :key="e.id"
              type="button"
              class="item recipe-item"
              :class="{ on: picked?.id === e.id && !phone }"
              :style="{ '--a': e.cover.from }"
              @click="choose(e)"
            >
              <span class="thumb" :style="e.thumb ? undefined : cover(e)">
                <img v-if="e.thumb" :src="e.thumb" alt="" loading="lazy" />
                <i v-else :class="e.cover.icon" />
              </span>
              <span class="item-text">
                <span class="item-title">
                  {{ e.name }}
                  <span v-if="e.draft" class="draft" title="Not trained and checked end to end yet: settings from similar runs">draft</span>
                </span>
                <span class="item-blurb">{{ phone ? e.blurb : e.meta }}</span>
              </span>
              <i v-if="phone" class="mdi mdi-chevron-right chev" />
            </button>
          </section>
          <div v-if="!groups.length" class="empty">
            <i class="mdi mdi-magnify-close" />
            <p>No recipe matches “{{ query }}”.</p>
            <button class="link" @click="query = ''">Show them all</button>
          </div>
          <p v-else-if="!runPresets.length && !query" class="small muted mine-hint">
            <i class="mdi mdi-content-save-outline" /> Save a run's settings with <b>Save as recipe</b> and they're here too.
          </p>
        </nav>
      </ScrollArea>

      <ScrollArea ref="detailScroll" class="detail">
        <article v-if="picked" class="detail-in">
          <div class="hero" :style="picked.thumb ? undefined : cover(picked)">
            <img v-if="picked.thumb" :src="picked.thumb" alt="" />
            <i v-else :class="picked.cover.icon" class="hero-icon" />
            <div class="hero-text">
              <span class="hero-kind">{{ picked.recipe ? `${KINDS[picked.recipe.kind]} · ${picked.model}` : 'Your recipe' }}</span>
              <h3 class="zt-title">{{ picked.name }}</h3>
            </div>
          </div>
          <p class="blurb">{{ picked.blurb }}</p>

          <div v-if="planning" class="muted small"><i class="mdi mdi-loading mdi-spin" /> Checking the model files…</div>
          <p v-else-if="failed || built.error" class="small warn"><i class="mdi mdi-alert-outline" /> Couldn't work out this recipe here: {{ failed || built.error }}</p>
          <template v-else-if="next">
            <div class="stats">
              <div v-for="s in stats" :key="s.label" class="stat" :class="{ same: s.from === s.to }" :title="[s.why, s.from === s.to ? 'as the form has it' : `was ${s.from}`].filter(Boolean).join('; ')">
                <span class="stat-label">{{ s.label }}</span>
                <span class="stat-to mono">{{ s.to }}</span>
              </div>
            </div>
            <p v-if="bigSet && next" class="small muted sizing">
              <i class="mdi mdi-information-outline" /> {{ photos }} pictures: {{ next.optim.steps.toLocaleString() }} steps shows each one about
              {{ Math.max(1, Math.round((next.optim.steps * (next.optim.batch_size || 1)) / photos!)) }} times. A set this big may want more.
            </p>

            <h4>Model files</h4>
            <div v-if="files.length" class="files">
              <span v-for="f in files" :key="f.key" class="file" :class="{ get: f.download }" :title="f.name ?? ''">
                <i :class="f.download ? 'mdi mdi-download-outline' : picked.mine ? 'mdi mdi-file-outline' : 'mdi mdi-check-circle'" />
                <span class="mono">{{ short(f.name) }}</span>
                <span v-if="f.download" class="file-note">download{{ f.download.size ? ` · ${fmtBytes(f.download.size)}` : '' }}</span>
              </span>
              <span v-if="planned?.adapter" class="file get">
                <i class="mdi mdi-download-outline" /> {{ planned.adapter.name }} <span class="file-note">training adapter</span>
              </span>
            </div>
            <p v-else class="small muted">Keeps the model picked in the form.</p>
            <p v-for="p in planned?.problems ?? []" :key="p" class="small warn"><i class="mdi mdi-alert-outline" /> {{ p }}</p>

            <h4>Every setting it changes <span class="count">{{ changeCount }}</span></h4>
            <div v-if="changeCount" class="changes">
              <template v-for="g in grouped" :key="g.section">
                <div class="sec">{{ g.section }}</div>
                <div v-for="row in g.rows" :key="g.section + row.label" class="row" :title="row.full">
                  <span class="label">{{ row.label }}</span>
                  <span class="from mono">{{ row.from }}</span>
                  <i class="mdi mdi-arrow-right" />
                  <span class="to mono">{{ row.to }}</span>
                </div>
              </template>
            </div>
            <p v-else class="small muted">Nothing: the form already has these settings.</p>
            <p v-if="picked.note" class="small muted note"><i class="mdi mdi-information-outline" /> {{ picked.note }}</p>
            <p v-if="phone" class="small muted note"><i class="mdi mdi-lock-outline" /> Your name, dataset, trigger word and captions stay. Undo puts everything else back.</p>
          </template>

          <div v-if="picked.mine" class="mine-actions">
            <button class="del" :class="{ sure: confirmDelete }" @click="remove">
              <i class="mdi mdi-trash-can-outline" /> {{ confirmDelete ? `Delete “${picked.name}”? Click again` : 'Delete this recipe' }}
            </button>
          </div>
        </article>
      </ScrollArea>
    </div>

    <template #footer>
      <div class="foot" :class="{ phone }">
        <span v-if="!phone" class="muted small"><i class="mdi mdi-lock-outline" /> Your name, dataset, trigger word and captions stay. Undo puts everything else back.</span>
        <span class="grow" />
        <ZenButton @click="open = false">Cancel</ZenButton>
        <ZenButton v-if="!phone || step === 'detail'" variant="primary" icon="mdi mdi-check" :disabled="!next || planning" @click="apply">
          {{ picked ? (phone ? 'Apply' : `Apply ${picked.title}`) : 'Apply' }}
        </ZenButton>
      </div>
    </template>
  </ZenModal>
</template>

<style scoped>
/* ZenModal's body scrolls by itself; here it doesn't: the two panes do */
:global(.zen-modal:has(.picker) .zm-body) {
  padding: 0;
  overflow: hidden;
}
.head {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;
  min-width: 0;
}
.head-text {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}
.head-text b {
  font-size: 15px;
}
.head-text .muted {
  font-size: 12px;
}
.recipe-search {
  position: relative;
  margin-left: auto;
  width: 220px;
}
.recipe-search > i {
  position: absolute;
  left: 8px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--zen-muted);
  font-size: 15px;
  pointer-events: none;
  z-index: 1;
}
.recipe-search :deep(input),
.recipe-search :deep(.zen-input) {
  padding-left: 28px !important; /* past the magnifier: ZenInput sets its own */
  width: 100%;
}
.back {
  font: inherit;
  font-size: 14px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 2px;
  border: none;
  background: none;
  color: var(--zen-text);
  cursor: pointer;
  padding: 4px 4px 4px 0;
}
.back i {
  font-size: 20px;
}

.picker {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr);
  height: 100%;
  min-height: 0;
}
.list,
.detail {
  min-height: 0;
}
.list :deep(.viewport),
.detail :deep(.viewport) {
  overscroll-behavior: contain;
}

/* the list: by model, small covers */
.list {
  border-right: 1px solid var(--zen-border);
  background: color-mix(in srgb, var(--zen-surface-2) 55%, transparent);
}
.list-in {
  padding: 10px 8px 16px;
}
.group + .group {
  margin-top: 10px;
}
.g-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted);
}
.g-head img {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  background: #fff;
}
.here {
  margin-left: auto;
  text-transform: none;
  letter-spacing: 0;
  font-weight: 500;
  color: var(--zen-accent);
}
.item {
  --a: var(--zen-accent);
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 6px 8px;
  border: none;
  border-radius: 8px;
  background: none;
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
  position: relative;
}
.item:hover {
  background: color-mix(in srgb, var(--zen-text) 6%, transparent);
}
.item.on {
  background: color-mix(in srgb, var(--a) 16%, transparent);
}
.item.on::before {
  content: '';
  position: absolute;
  left: -8px;
  top: 8px;
  bottom: 8px;
  width: 3px;
  border-radius: 0 3px 3px 0;
  background: var(--a);
}
.thumb {
  flex: none;
  width: 52px;
  height: 40px;
  border-radius: 6px;
  overflow: hidden;
  display: grid;
  place-items: center;
  color: #fff;
  font-size: 18px;
}
.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.item-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.item-title {
  font-weight: 600;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.item-blurb {
  font-size: 12px;
  color: var(--zen-muted);
}
.draft {
  font-size: 10px;
  font-weight: 500;
  padding: 0 6px;
  border-radius: 999px;
  border: 1px dashed var(--zen-warn, #d9a400);
  color: var(--zen-warn, #d9a400);
}
.empty {
  padding: 40px 16px;
  text-align: center;
  color: var(--zen-muted);
}
.empty i {
  font-size: 32px;
  opacity: 0.6;
}
.empty p {
  margin: 6px 0;
  font-size: 13px;
}
.link {
  font: inherit;
  font-size: 12px;
  border: none;
  background: none;
  color: var(--zen-accent);
  cursor: pointer;
}
.chev {
  margin-left: auto;
  color: var(--zen-muted);
  font-size: 18px;
}

.g-icon {
  font-size: 15px;
}
.mine-hint {
  margin: 14px 8px 0;
  line-height: 1.45;
}
.mine-actions {
  margin-top: 22px;
  padding-top: 14px;
  border-top: 1px solid var(--zen-border);
}
.del {
  font: inherit;
  font-size: 12px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  border-radius: var(--r-md);
  border: 1px solid var(--zen-border);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.del:hover,
.del.sure {
  color: var(--zen-danger, #ff5a4e);
  border-color: color-mix(in srgb, var(--zen-danger, #ff5a4e) 55%, var(--zen-border));
}
/* the detail */
.detail-in {
  padding: 0 24px 24px;
}
/* a compact band: the recipe's name bottom left, its icon large and faint on the right */
.hero {
  position: relative;
  margin: 0 -24px;
  height: 128px;
  overflow: hidden;
}
.hero img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.hero::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.72), rgba(0, 0, 0, 0) 60%);
}
.hero-icon {
  position: absolute;
  right: 28px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 92px;
  line-height: 1;
  color: rgba(255, 255, 255, 0.28);
}
.hero-text {
  position: absolute;
  left: 24px;
  bottom: 16px;
  z-index: 1;
  color: #fff;
}
.hero-kind {
  font-size: 12px;
  opacity: 0.85;
}
.hero h3 {
  margin: 2px 0 0;
  font-size: 24px;
  font-weight: 650;
}
.blurb {
  margin: 16px 0 14px;
  line-height: 1.5;
}
.stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--zen-border);
  background: var(--zen-surface-2);
}
.stat-label {
  font-size: 11px;
  color: var(--zen-muted);
}
.stat-to {
  font-size: 18px;
  font-weight: 600;
  color: var(--zen-accent);
}
.stat.same .stat-to {
  color: var(--zen-text);
}
.sizing {
  margin: 8px 0 0;
}
h4 {
  margin: 20px 0 8px;
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
}
.count {
  font-weight: 500;
  font-size: 11px;
  padding: 0 7px;
  border-radius: 999px;
  background: var(--zen-surface-2);
  color: var(--zen-muted);
}
.files {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.file {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 100%;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--zen-border);
  font-size: 12px;
}
.file .mdi-check-circle {
  color: var(--zen-success, #3fb950);
}
.file.get {
  border-style: dashed;
  border-color: var(--zen-warn, #d9a400);
}
.file.get i {
  color: var(--zen-warn, #d9a400);
}
.file .mono {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.file-note {
  color: var(--zen-muted);
  white-space: nowrap;
}
.changes {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) auto minmax(0, 1fr);
  column-gap: 10px;
  border: 1px solid var(--zen-border);
  border-radius: 8px;
  overflow: hidden;
  font-size: 12px;
}
.sec {
  grid-column: 1 / -1;
  padding: 6px 12px;
  font-size: 11px;
  font-weight: 600;
  color: var(--zen-muted);
  background: var(--zen-surface-2);
  border-top: 1px solid var(--zen-border);
}
.sec:first-child {
  border-top: none;
}
.row {
  display: contents;
}
.row > * {
  padding: 4px 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.label {
  padding-left: 12px !important;
}
.from {
  color: var(--zen-muted);
  text-decoration: line-through;
  text-decoration-color: color-mix(in srgb, var(--zen-muted) 50%, transparent);
}
.row i {
  color: var(--zen-muted);
}
.to {
  color: var(--zen-accent);
  padding-right: 12px !important;
}
.note {
  margin-top: 14px;
}
.warn {
  color: var(--zen-warn, #d9a400);
}
.foot {
  display: flex;
  align-items: center;
  gap: 8px;
}
.grow {
  flex: 1;
}
/* a phone: the whole screen, list then detail */
@media (max-width: 720px) {
  :global(.zen-modal:has(.picker)) {
    width: 100vw !important;
    max-width: 100vw !important;
    height: 100dvh !important;
    max-height: 100dvh !important;
    border-radius: 0 !important;
    border: none !important;
  }
  :global(.zen-modal-back:has(.picker)) {
    align-items: stretch;
  }
}
.head.phone {
  flex-wrap: wrap;
  row-gap: 8px;
}
.head.phone .recipe-search {
  order: 3;
  width: 100%;
  margin-left: 0;
}
.head.phone .recipe-search :deep(input) {
  font-size: 16px; /* no zoom on focus in iOS */
}
.picker.phone {
  grid-template-columns: 1fr;
}
.picker.phone .list {
  border-right: none;
}
.picker.phone:not(.detailed) .detail,
.picker.phone.detailed .list {
  display: none;
}
.picker.phone .item {
  padding: 10px 8px;
}
.picker.phone .thumb {
  width: 64px;
  height: 48px;
}
.picker.phone .item-blurb {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.picker.phone .detail-in {
  padding: 0 16px 20px;
}
.picker.phone .hero {
  margin: 0 -16px;
  height: 150px;
}
.picker.phone .hero-text {
  left: 16px;
}
.picker.phone .stats {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.picker.phone .changes {
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
}
.picker.phone .changes .from {
  display: none;
}
.foot.phone :deep(button) {
  flex: 1;
  justify-content: center;
}
.foot.phone .grow {
  display: none;
}
</style>
