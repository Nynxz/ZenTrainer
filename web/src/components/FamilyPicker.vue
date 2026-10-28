<script setup lang="ts">
// Pick a model as one thing (Krea 2, Z-Image, SDXL…) instead of three files. Families come
// from ComfyUI's own templates plus the builds in each family's Hugging Face repo; every part
// shows whether it's installed and downloads in place if not. "Custom" keeps the manual
// file-by-file picker for anything the catalog doesn't know.
import { ZenButton, ZenCombo } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { inputChoices } from '@/api/comfy'
import { engine, type ModelFamily, type ModelFile, type ModelFolder, type ModelProfile, type SampleDefaults } from '@/api/engine'
import { fmtAdded, fmtBytes, finished, hfToken, isNew, startDownloadSync, targets } from '@/stores/downloads'
import FileState from './FileState.vue'
import StarButton from './StarButton.vue'
import { isStarred, starredFirst } from '@/stores/favorites'
import { isOther, makerLogo } from '@/lib/makers'
import type { ModelValue } from './ModelPicker.vue'

export interface SampleSuggestion extends SampleDefaults {
  family: string
  folder: ModelFolder
  size: number | null
  /** the name ComfyUI lists it by, once installed */
  installed: string | null
}

const model = defineModel<ModelValue>({ required: true })
const emit = defineEmits<{
  /** files still to download before this model can train (0 = ready) */
  missing: [number]
  /** the family's suggested sampling model + settings, when it differs from the trained one */
  sampleSuggestion: [SampleSuggestion | null]
  custom: [boolean]
  /** the family's hand-picked settings (null = none); picked = the user chose this family just now */
  profile: [ModelProfile | null, boolean]
  /** the picked family's id (null: none, or custom files) */
  family: [string | null]
  /** browse mode: the field was clicked (New run opens its model browser) */
  browse: []
}>()
/** browse: a field showing the model that opens a dialog (New run's), instead of the dropdown */
const props = defineProps<{ browse?: boolean }>()

const CUSTOM = '__custom'
const families = ref<ModelFamily[]>([])
const famId = ref<string>('')
let userPicked = false
const fam = ref<ModelFamily | null>(null)
const loadingRepo = ref(false)
const error = ref<string | null>(null)

const PARTS: Record<'split' | 'checkpoint', { folder: ModelFolder; label: string; key: keyof ModelValue }[]> = {
  split: [
    { folder: 'diffusion_models', label: 'Diffusion model', key: 'unet_name' },
    { folder: 'text_encoders', label: 'Text encoder', key: 'clip_name' },
    { folder: 'vae', label: 'VAE', key: 'vae_name' },
  ],
  checkpoint: [{ folder: 'checkpoints', label: 'Checkpoint', key: 'ckpt_name' }],
}
const parts = computed(() => (fam.value ? PARTS[fam.value.kind] : []))

const base = (n?: string | null) => (n ?? '').replace(/\\/g, '/').split('/').pop() ?? ''
// what ComfyUI has in each folder, as its loaders list them: a file of yours outside the catalog (a
// quant you made, a renamed copy) is still installed, and a duplicated run keeps it rather than
// showing it empty and asking to download the catalog's
const LOADERS: Partial<Record<ModelFolder, [string, string]>> = {
  checkpoints: ['CheckpointLoaderSimple', 'ckpt_name'], diffusion_models: ['UNETLoader', 'unet_name'],
  text_encoders: ['CLIPLoader', 'clip_name'], vae: ['VAELoader', 'vae_name'],
}
const onDisk = ref<Partial<Record<ModelFolder, string[]>>>({})
async function loadOnDisk() {
  const lists = await Promise.all(
    Object.entries(LOADERS).map(async ([folder, [node, input]]) => [folder, await inputChoices(node, input).catch(() => [])] as const),
  )
  onDisk.value = Object.fromEntries(lists)
}
/** a file of yours: installed here, not in the family's catalog */
function yours(folder: ModelFolder, name?: string | null): ModelFile | undefined {
  if (!name) return undefined
  const hit = (onDisk.value[folder] ?? []).find((n) => n === name || base(n) === base(name))
  return hit ? { name: base(name), folder, url: '', size: null, distilled: false, from_template: false, installed: hit } : undefined
}
const fileOf = (folder: ModelFolder, name?: string | null) =>
  fam.value?.files[folder]?.find((f) => f.name === base(name)) ?? yours(folder, name)

// which family the current spec already uses (a duplicated run, a preset)
function detect(list: ModelFamily[]): string {
  const m = model.value
  const key = base(m.kind === 'checkpoint' ? m.ckpt_name : m.unet_name)
  if (!key) return ''
  const hit = list.find((f) => (f.files[m.kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models'] ?? []).some((x) => x.name === key))
  return hit?.id ?? CUSTOM
}

// Hugging Face is listed in the background and cached for a few hours; a model its makers
// added since then shows up on Re-list
const relisting = ref(false)
const relistNote = ref<string | null>(null)
async function relist() {
  relisting.value = true
  relistNote.value = null
  try {
    const r = await engine.relistFamilies()
    await loadFamilies()
    relistNote.value = r.new.length ? `New: ${r.new.join(', ')}` : 'Up to date: no new models'
  } catch (e) {
    relistNote.value = String((e as Error).message ?? e)
  } finally {
    relisting.value = false
  }
}

async function loadFamilies() {
  try {
    families.value = await engine.families()
    redetect()
  } catch (e) {
    error.value = String(e)
  }
}
// the spec's files changed from outside (a duplicated run loading, a preset): follow them,
// unless they still belong to the family already showing
function redetect() {
  if (!families.value.length) return
  const m = model.value
  const key = base(m.kind === 'checkpoint' ? m.ckpt_name : m.unet_name)
  const cur = families.value.find((f) => f.id === famId.value)
  if (cur && (cur.files[m.kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models'] ?? []).some((x) => x.name === key)) return
  if (!key && famId.value) return
  const next = detect(families.value)
  if (next !== famId.value) {
    userPicked = false
    famId.value = next
  }
}
watch(() => [model.value.kind, model.value.ckpt_name, model.value.unet_name], redetect)

async function open(id: string, refresh = false) {
  if (!id || id === CUSTOM) {
    fam.value = null
    return
  }
  fam.value = families.value.find((f) => f.id === id) ?? null
  loadingRepo.value = true
  try {
    const full = await engine.family(id, true, refresh)
    if (famId.value === id) fam.value = full
  } catch {
    // offline: the template files are still there
  } finally {
    loadingRepo.value = false
  }
}

// choosing a family fills the spec with its recommended files
function apply(f: ModelFamily) {
  const rec = f.recommended.train
  const name = (folder: ModelFolder) => {
    const n = rec[folder]
    const file = n ? f.files[folder]?.find((x) => x.name === n) : undefined
    return file ? (file.installed ?? file.name) : null
  }
  model.value =
    f.kind === 'checkpoint'
      ? { ...model.value, kind: 'checkpoint', ckpt_name: name('checkpoints') }
      : {
          ...model.value,
          kind: 'split',
          unet_name: name('diffusion_models'),
          clip_name: name('text_encoders'),
          clip_type: f.clip_type,
          vae_name: name('vae'),
        }
}

watch(famId, async (id, old) => {
  emit('custom', id === CUSTOM)
  await open(id)
  // a family picked by hand fills the files; one detected from the spec keeps them
  const picked = !!fam.value && userPicked && id !== old
  if (picked) apply(fam.value!)
  emit('profile', fam.value?.profile ?? null, picked)
  emit('family', fam.value?.id ?? null)
})
const pick = (id: string | number) => {
  userPicked = true
  famId.value = String(id)
}

function setFile(folder: ModelFolder, key: keyof ModelValue, name: string | number) {
  const f = fileOf(folder, String(name))
  model.value = { ...model.value, [key]: f ? (f.installed ?? f.name) : String(name) }
}

// install state changes as downloads land: refresh the family and swap in the local names
watch(finished, async () => {
  void loadOnDisk()
  if (!fam.value) return
  const full = await engine.family(fam.value.id, true).catch(() => null)
  if (!full) return
  fam.value = full
  for (const p of parts.value) {
    const f = fileOf(p.folder, model.value[p.key] as string)
    if (f?.installed && model.value[p.key] !== f.installed) model.value = { ...model.value, [p.key]: f.installed }
  }
  families.value = families.value.map((x) => (x.id === full.id ? { ...x, missing: full.missing } : x))
})

const missingParts = computed(() => parts.value.filter((p) => !fileOf(p.folder, model.value[p.key] as string)?.installed))
watch(
  [missingParts, fam],
  () => emit('missing', fam.value ? missingParts.value.length : 0),
  { immediate: true },
)
// the suggestion only makes sense while training on a different (base) build
watch(
  [fam, () => model.value.unet_name, () => model.value.ckpt_name],
  () => {
    const f = fam.value
    const s = f?.recommended.sample ?? null
    if (!f || !s) return emit('sampleSuggestion', null)
    const folder: ModelFolder = f.kind === 'checkpoint' ? 'checkpoints' : 'diffusion_models'
    const trained = base(f.kind === 'checkpoint' ? model.value.ckpt_name : model.value.unet_name)
    // any installed distilled build will do (the template's precision first); else the template's
    const tmpl = fileOf(folder, s.model)
    const prec = (n: string) => n.match(/bf16|fp16|fp8\w*|int8\w*|nvfp4|mxfp8/)?.[0] ?? ''
    const distilled = (f.files[folder] ?? []).filter((x) => x.distilled && x.name !== trained)
    const pick =
      distilled.find((x) => x.installed && prec(x.name) === prec(s.model)) ?? distilled.find((x) => x.installed) ?? tmpl
    if (!pick || pick.name === trained) return emit('sampleSuggestion', null)
    emit('sampleSuggestion', { ...s, model: pick.name, family: f.id, folder, size: pick.size, installed: pick.installed })
  },
  { immediate: true },
)

async function download(folder: ModelFolder, name: string) {
  if (!fam.value) return
  error.value = null
  try {
    await engine.download(fam.value.id, folder, name)
  } catch (e) {
    error.value = String(e)
  }
}
async function downloadAll() {
  for (const p of missingParts.value) {
    const f = fileOf(p.folder, model.value[p.key] as string)
    if (f) await download(p.folder, f.name)
  }
}
const missingBytes = computed(() =>
  missingParts.value.reduce((n, p) => n + (fileOf(p.folder, model.value[p.key] as string)?.size ?? 0), 0),
)

// models few people train (the catalog's "other") wait behind the footer's toggle, unless you
// starred one (then it's with your stars, at the top) or it's the one picked
const showOther = ref(false)
// the closed picker's logo: its family's, when the item is a family (not Custom)
const logoOf = (item: unknown) => {
  const f = (item as { f?: ModelFamily | null } | null)?.f
  return f ? makerLogo(f.id) : null
}
const hidden = (f: ModelFamily) => isOther(f.id) && !isStarred(f.id) && f.id !== famId.value
const otherCount = computed(() => families.value.filter(hidden).length)
const familyItems = computed(() => [
  ...starredFirst(families.value, (f) => f.id)
    .filter((f) => !hidden(f))
    .map((f) => ({ value: f.id, label: f.name, keywords: `${f.name} ${f.clip_type ?? ''} ${f.repos.join(' ')}`, f })),
  ...(showOther.value
    ? families.value
        .filter(hidden)
        .map((f) => ({ value: f.id, label: f.name, keywords: `${f.name} ${f.clip_type ?? ''} ${f.repos.join(' ')}`, f }))
    : []),
  { value: CUSTOM, label: 'Custom: pick the files yourself', keywords: 'custom manual files', f: null },
])
// the catalog's files, and the one picked when it's one of yours (listed so it shows, tagged)
const fileItems = (folder: ModelFolder, current?: string | null) => {
  const items = (fam.value?.files[folder] ?? []).map((f) => ({ value: f.name, label: f.name, keywords: f.name, file: f, mine: false }))
  const own = yours(folder, current)
  if (own && !items.some((i) => i.value === own.name)) items.push({ value: own.name, label: own.name, keywords: own.name, file: own, mine: true })
  return items
}
// installed: the tick alone; else what's left to fetch, by size (the count on hover)
const statusOf = (f: ModelFamily) => {
  if (!f.missing.length) return ''
  const bytes = f.missing.reduce((n, m) => n + (m.size ?? 0), 0)
  return bytes ? fmtBytes(bytes) : `${f.missing.length}`
}
const statusTip = (f: ModelFamily) =>
  f.missing.length ? `${f.missing.length} file${f.missing.length === 1 ? '' : 's'} to download` : 'Installed'

onMounted(() => {
  startDownloadSync()
  loadFamilies()
  void loadOnDisk()
})
defineExpose({ refresh: () => fam.value && open(fam.value.id, true), pick, CUSTOM })
</script>

<template>
  <div class="fp">
    <button v-if="props.browse" type="button" class="fam-field" :class="{ empty: !fam && famId !== CUSTOM }" @click="emit('browse')">
      <!-- a maker's logo on white (its own colours); otherwise a tile in the app's own -->
      <span class="fam-logo" :class="{ blank: !(fam && makerLogo(fam.id)) }">
        <img v-if="fam && makerLogo(fam.id)" :src="makerLogo(fam.id)!.src" :class="{ mono: makerLogo(fam.id)!.mono }" alt="" />
        <i v-else :class="famId === CUSTOM ? 'mdi mdi-file-cog-outline' : fam ? 'mdi mdi-cube-outline' : 'mdi mdi-plus'" />
      </span>
      <span class="fam-text">
        <b>{{ fam?.name ?? (famId === CUSTOM ? 'Custom: your own files' : 'Choose a model') }}</b>
        <span v-if="fam" class="muted small">{{ fam.missing.length ? `${statusOf(fam)} to download` : 'installed' }}</span>
        <span v-else-if="famId !== CUSTOM" class="muted small">The models in your ComfyUI, with their recipes</span>
      </span>
      <span class="fam-change">{{ fam || famId === CUSTOM ? 'Change' : 'Choose' }} <i class="mdi mdi-chevron-right" /></span>
    </button>
    <ZenCombo
      v-else
      :model-value="famId"
      :items="familyItems"
      placeholder="Choose a model…"
      :sorted="false"
      :item-height="40"
      :menu-width="460"
      @update:model-value="pick"
    >
      <template #option="{ item }">
        <div class="fam-opt">
          <StarButton v-if="item.f" :id="(item.f as ModelFamily).id" :name="item.label" />
          <img
            v-if="item.f && makerLogo((item.f as ModelFamily).id)"
            class="maker"
            :class="{ mono: makerLogo((item.f as ModelFamily).id)!.mono }"
            :src="makerLogo((item.f as ModelFamily).id)!.src"
            :title="makerLogo((item.f as ModelFamily).id)!.maker"
            alt=""
          />
          <span v-else-if="item.f" class="maker none" />
          <b>{{ item.label }}</b>
          <i
            v-if="item.f && (item.f as ModelFamily).profile"
            class="mdi tick"
            :class="(item.f as ModelFamily).profile!.status === 'verified' ? 'mdi-check-decagram' : 'mdi-check-circle-outline'"
            :title="(item.f as ModelFamily).profile!.status === 'verified' ? 'Tuned: trained and checked with hand-picked settings' : 'Hand-picked settings from the model makers'"
          />
          <span v-if="item.f && isNew((item.f as ModelFamily).added)" class="tag new" :title="`Added ${fmtAdded((item.f as ModelFamily).added)}`">new</span>
          <span v-if="item.f" class="fam-status" :class="{ ready: !(item.f as ModelFamily).missing.length }" :title="statusTip(item.f as ModelFamily)">
            <i :class="(item.f as ModelFamily).missing.length ? 'mdi mdi-download-outline' : 'mdi mdi-check-circle'" />
            {{ statusOf(item.f as ModelFamily) }}
          </span>
        </div>
      </template>
      <!-- the closed picker: the model with its maker's logo -->
      <template #selected="{ item }">
        <span v-if="item" class="sel">
          <img v-if="logoOf(item)" class="maker" :class="{ mono: logoOf(item)!.mono }" :src="logoOf(item)!.src" alt="" />
          <span class="sel-name">{{ item.label }}</span>
        </span>
        <span v-else class="muted">Choose a model…</span>
      </template>
      <template #footer>
        <div class="fp-menu-foot">
          <button v-if="otherCount" class="foot-btn" @click.stop="showOther = !showOther">
            <i :class="showOther ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
            {{ showOther ? 'Hide other models' : `Other models · ${otherCount}` }}
          </button>
          <span class="spacer" />
          <span v-if="relistNote" class="relist-note small muted" :title="relistNote">{{ relistNote }}</span>
          <button
            class="foot-btn"
            :disabled="relisting"
            title="Re-list the model makers' Hugging Face repos now: models they've added show up without waiting for the cache"
            @click.stop="relist"
          >
            <i class="mdi mdi-refresh" :class="{ spin: relisting }" /> {{ relisting ? 'Re-listing…' : 'Re-list' }}
          </button>
        </div>
      </template>
    </ZenCombo>

    <div v-if="fam" class="parts">
      <div v-for="p in parts" :key="p.folder" class="part">
        <span class="p-label">{{ p.label }}</span>
        <ZenCombo
          :model-value="base(model[p.key] as string)"
          :items="fileItems(p.folder, model[p.key] as string)"
          :item-height="34"
          :menu-width="440"
          @update:model-value="setFile(p.folder, p.key, $event)"
        >
          <template #option="{ item }">
            <div class="file-opt">
              <i :class="(item.file as ModelFile).installed ? 'mdi mdi-check-circle ok' : 'mdi mdi-cloud-download-outline muted'" />
              <span class="f-name">{{ item.label }}</span>
              <span v-if="(item.file as ModelFile).distilled" class="tag" title="Distilled (turbo/lightning): fast to sample, not a good base to train on">turbo</span>
              <span v-if="item.mine" class="tag" title="Installed here, not one of the family's files: yours">yours</span>
              <span class="f-size mono">{{ fmtBytes((item.file as ModelFile).size) }}</span>
            </div>
          </template>
        </ZenCombo>
        <!-- fixed-width status column: installed / download / progress, without shifting the row -->
        <div class="p-state">
          <template v-for="f in [fileOf(p.folder, model[p.key] as string)]" :key="p.folder">
            <FileState v-if="f && fam" :family="fam.id" :file="f" @error="error = $event" />
            <span v-else class="muted">not in the catalog</span>
          </template>
        </div>
      </div>

      <div class="fp-foot">
        <template v-if="missingParts.length > 1">
          <ZenButton class="dl-all" size="sm" icon="mdi mdi-download-multiple" @click="downloadAll">Download all · {{ fmtBytes(missingBytes) }}</ZenButton>
        </template>
        <span class="muted small ellipsis" :title="targets[parts[0]?.folder ?? 'diffusion_models']">
          <template v-if="missingParts.length">Downloads go to {{ targets[parts[0]?.folder ?? 'diffusion_models'] ?? 'your models folder' }}</template>
          <template v-else-if="fam.repos.length">From {{ fam.repos.join(', ') }}</template>
        </span>
        <span class="spacer" />
        <span v-if="loadingRepo" class="muted small"><i class="mdi mdi-loading mdi-spin" /> listing builds…</span>
        <button v-else class="link small" title="Re-list the builds on Hugging Face" @click="open(fam.id, true)"><i class="mdi mdi-refresh" /></button>
      </div>
    </div>
    <div v-if="fam?.gated && missingParts.length" class="gate small">
      <i class="mdi mdi-lock-outline" />
      <span>
        Gated: accept the license on
        <a :href="`https://huggingface.co/${fam.repos[0]}`" target="_blank" rel="noopener">Hugging Face</a>
        <template v-if="!hfToken">, then set <span class="mono">HF_TOKEN</span> (a read token) and restart ZenTrainer</template>
        <template v-else> with the account your <span class="mono">HF_TOKEN</span> belongs to</template>.
      </span>
    </div>
    <div v-if="error" class="err small mono">{{ error }}</div>
  </div>
</template>

<style scoped>
.fp {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.maker {
  width: 22px;
  height: 22px;
  flex: none;
  border-radius: 5px;
  object-fit: contain;
}
/* black-on-transparent marks: white on dark themes (the whole selector inside :global, or Vue
   drops everything after it and the rule lands on <html> itself) */
:global(html[data-mode='dark'] .fam-opt .maker.mono),
:global(html[data-mode='dark'] .sel .maker.mono) {
  filter: invert(1);
}
.maker.none {
  display: inline-block;
}
.fp-menu-foot {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.fp-menu-foot .spacer {
  flex: 1;
}
.sel {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.sel .maker {
  width: 18px;
  height: 18px;
}
.sel-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.foot-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex: none;
  padding: 6px 8px;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
/* ZenCombo keeps the last hovered option highlighted when the pointer moves on to the footer:
   no highlight while the footer's under the pointer. UPSTREAM: ZenCombo clears its active
   option on leaving the list */
:global(.zen-combo-menu:has(.zc-footer:hover) .zc-opt.active) {
  background: none;
}
.foot-btn:disabled {
  opacity: 0.6;
  cursor: default;
}
.foot-btn:hover:not(:disabled) {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.relist-note {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.spin {
  animation: relist-spin 0.9s linear infinite;
}
@keyframes relist-spin {
  to {
    transform: rotate(360deg);
  }
}
.fam-opt {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-width: 0;
}
.fam-opt b {
  flex: 0 1 auto; /* its own width: the tick and "new" sit right after the name */
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
}
.fam-status {
  margin-left: auto;
  font-size: 11px;
  color: var(--zen-muted);
  white-space: nowrap;
}
.fam-status.ready,
.ok {
  color: var(--zen-ok);
}
.parts {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border-radius: var(--r-lg);
  background: color-mix(in srgb, var(--zen-surface-2) 50%, transparent);
}
.part {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr) 150px;
  align-items: center;
  gap: 10px;
}
.p-label {
  font-size: 12px;
  color: var(--zen-muted);
}
.p-state {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  font-size: 12px;
  white-space: nowrap;
  min-width: 0;
}
.file-opt {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-width: 0;
}
.f-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.f-size {
  font-size: 11px;
  color: var(--zen-muted);
}
.gate {
  display: flex;
  gap: 8px;
  align-items: baseline;
  padding: 8px 10px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-warn) 10%, transparent);
}
.gate i {
  color: var(--zen-warn);
}
.gate a {
  color: var(--zen-accent);
}
.tick {
  color: var(--zen-accent);
  font-size: 14px;
  margin-left: -4px;
}
.tag.new {
  background: color-mix(in srgb, var(--zen-ok) 18%, transparent);
  color: var(--zen-ok);
}
.tag {
  font-size: 10px;
  padding: 0 6px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-warn) 18%, transparent);
  color: var(--zen-warn);
}
.fp-foot {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 26px;
  margin-top: 2px;
}
/* one line: the folder path beside it truncates instead */
.dl-all {
  flex-shrink: 0;
  white-space: nowrap;
}
.small {
  font-size: 11px;
}
.ellipsis {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.link {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  font: inherit;
}
.link:hover {
  color: var(--zen-accent);
}
.err {
  color: var(--zen-danger);
}
@media (max-width: 700px) {
  .part {
    grid-template-columns: 1fr auto;
  }
  .p-label {
    grid-column: 1 / -1;
  }
}

/* browse mode's field: the model, as New run's dataset field shows its dataset */
.fam-field {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 6px 10px 6px 6px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  background: var(--zen-surface);
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.fam-field:hover {
  border-color: color-mix(in srgb, var(--zen-accent) 50%, var(--zen-border));
}
.fam-logo {
  width: 36px;
  height: 36px;
  flex: none;
  border-radius: 8px;
  display: grid;
  place-items: center;
  background: #fff;
  overflow: hidden;
  color: #555;
}
.fam-logo img {
  width: 70%;
  height: 70%;
  object-fit: contain;
}
.fam-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}
.fam-text b {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fam-field.empty {
  border-style: dashed;
}
.fam-logo.blank {
  background: var(--zen-surface-2);
  color: var(--zen-accent);
  font-size: 18px;
}
.fam-change {
  font-size: 12px;
  color: var(--zen-accent);
  display: inline-flex;
  align-items: center;
}
</style>
