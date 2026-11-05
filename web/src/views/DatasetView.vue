<script setup lang="ts">
// One dataset. Grid (uniform cards, zoomable), table (sortable, inline captions) or edit pairs;
// click any image for the lightbox with a caption editor beside it. Drop files anywhere
// to upload: images, same-name .txt captions, and masks (<name>-masklabel.png). The parts are in
// components/dataset/ and share the dataset through its context (context.ts).
import { ZenButton, ZenCombo, ZenContextMenu, ZenEmpty, ZenInput, ZenLightbox, ZenSlider } from '@nynxz/zenkit-ui'
import { computed, nextTick, onMounted, onUnmounted, provide, ref } from 'vue'
import { useRouter } from 'vue-router'
import { engine } from '@/api/engine'
import { reducedMotion } from '@/stores/settings'
import StickyBar from '@/components/StickyBar.vue'
import { DATASET, useDataset } from '@/components/dataset/context'
import DatasetGrid from '@/components/dataset/DatasetGrid.vue'
import DatasetLightboxSide from '@/components/dataset/DatasetLightboxSide.vue'
import DatasetPairs from '@/components/dataset/DatasetPairs.vue'
import DatasetSide from '@/components/dataset/DatasetSide.vue'
import DatasetTable from '@/components/dataset/DatasetTable.vue'
import { askText } from '@/lib/ask'
import { handoff, useFileDrop } from '@/lib/drop'

const props = defineProps<{ name: string }>()
const router = useRouter()

const ds = useDataset(() => props.name)
provide(DATASET, ds)
const { view, zoom, fit, sideOpen, items, sets, activeSet, controlSets, activeControl, loading, error, notice, bust, load, filter, query, selected, boxing, reviewing, counts, setMuted, faces, flash, removeMany, menu, menuItems, allShownSelected, toggleAll, layout, lightbox, lbItems, shown } = ds

// In a narrow window the panel would land under every picture, out of reach: there it's a drawer
// over the right edge instead, opened by the same button, closed on its own (not remembered)
const narrowQuery = window.matchMedia('(max-width: 1100px)')
const narrow = ref(narrowQuery.matches)
const drawerOpen = ref(false)
const onNarrow = (e: MediaQueryListEvent) => (narrow.value = e.matches)
narrowQuery.addEventListener('change', onNarrow)
onUnmounted(() => narrowQuery.removeEventListener('change', onNarrow))
const panelShown = computed(() => (narrow.value ? drawerOpen.value : sideOpen.value))

// the side panel changes the grid's columns: each card on screen takes its new place at once and
// grows or shrinks there from its old size, rather than snapping (no flying across the grid)
async function toggleSide() {
  if (narrow.value) {
    drawerOpen.value = !drawerOpen.value
    return
  }
  const reduce = reducedMotion()
  const vh = window.innerHeight
  const cards = reduce ? [] : [...document.querySelectorAll<HTMLElement>('.ds-main .card-item')]
  const before = new Map<HTMLElement, DOMRect>()
  for (const el of cards) {
    const r = el.getBoundingClientRect()
    if (r.bottom > -100 && r.top < vh + 100) before.set(el, r)
  }
  sideOpen.value = !sideOpen.value
  await nextTick()
  for (const [el, a] of before) {
    const b = el.getBoundingClientRect()
    if (!b.width || !b.height) continue
    const sx = a.width / b.width
    const sy = a.height / b.height
    if (Math.abs(sx - 1) < 0.01 && Math.abs(sy - 1) < 0.01) continue
    el.animate(
      [{ transform: `scale(${sx}, ${sy})` }, { transform: 'none' }],
      { duration: 240, easing: 'cubic-bezier(0.2, 0.7, 0.2, 1)' },
    )
  }
}

async function addSet() {
  const name = (
    await askText({
      title: 'New caption set',
      message: 'Another caption per picture, e.g. tags, nl or short: a run picks which set it trains on.',
      ok: 'Add',
      input: { placeholder: 'tags', check: (v) => (/^[a-z0-9][a-z0-9_-]{0,23}$/.test(v.toLowerCase()) ? null : 'Lowercase letters, digits, - and _ (24 at most)') },
    })
  )?.toLowerCase()
  if (!name) return
  if (!sets.value.includes(name)) sets.value = [...sets.value, name]
  activeSet.value = name
}
// the toolbar's caption set and filter menus: every entry always there, so nothing shifts
const setItems = computed(() => [
  ...sets.value.map((st) => ({ value: st, label: `${st} · ${items.value.filter((i) => i.captions?.[st]).length}/${items.value.length}` })),
  { value: '__add__', label: '+ New caption set…' },
])
function pickSet(v: string) {
  if (v === '__add__') addSet()
  else activeSet.value = v
}
const filterItems = computed(() => [
  { value: 'all', label: `All · ${items.value.length}` },
  { value: 'uncaptioned', label: `Uncaptioned · ${counts.value.uncaptioned}` },
  { value: 'masked', label: `Masked · ${counts.value.masked}` },
  { value: 'unmasked', label: 'Unmasked' },
  { value: 'muted', label: `Muted in “${activeSet.value}” · ${counts.value.muted}` },
  { value: 'images', label: `Image issues · ${counts.value.imageIssues}` },
  { value: 'faces', label: `Face issues · ${faces.value?.items ? counts.value.faceIssues : '–'}` },
  ...(controlSets.value.length ? [{ value: 'pairs', label: `With a “${activeControl.value}” input · ${counts.value.pairs}` }] : []),
])
function onEsc(e: KeyboardEvent) {
  // Escape in a dialog or menu (a ZenKit layer) is that one's: cancelling "Delete these?" keeps them selected
  if ((e.target as HTMLElement).closest?.('[data-zen-layer]')) return
  if (e.key === 'Escape' && narrow.value && drawerOpen.value && lightbox.value == null) {
    drawerOpen.value = false
    return
  }
  if (e.key !== 'Escape' || lightbox.value != null || !selected.size) return
  if ((e.target as HTMLElement).closest('input, textarea, [contenteditable]')) return
  selected.clear()
}

// ---- masks: the editor opens on the first selected picture, else the first one ---------------
const maskStart = computed(() => {
  const pictures = items.value.filter((i) => i.kind !== 'video')
  return pictures.find((i) => selected.has(i.path)) ?? pictures[0] ?? null
})
function openMasks() {
  if (maskStart.value) router.push({ path: `/datasets/${encodeURIComponent(props.name)}/mask`, query: { path: maskStart.value.path } })
}

// ---- upload -------------------------------------------------------------------------------
const uploading = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const dragging = useFileDrop(({ files }) => upload(files))
// A drop or pick goes up in requests of about 48 MB (a tunnel or proxy on the way won't take a
// gigabyte in one; what's landed stays when one fails), the toast counting them in and the
// pictures showing as they come. Drops take turns: a second during the first waits for it
const BATCH = 48 * 1024 ** 2
let turn = Promise.resolve()
function upload(files: File[]) {
  turn = turn.then(() => send(files))
}
async function send(files: File[]) {
  if (!files.length) return flash('Nothing to add in that drop')
  uploading.value = true
  const saved: string[] = []
  const errors: string[] = []
  let stopped: string | null = null
  let refreshed = Date.now()
  try {
    for (let i = 0; i < files.length; ) {
      notice.value = `Adding files… ${i.toLocaleString()} of ${files.length.toLocaleString()}`
      let j = i + 1
      for (let bytes = files[i]!.size; j < files.length && bytes + files[j]!.size <= BATCH; j++) bytes += files[j]!.size
      const r = await engine.upload(props.name, files.slice(i, j))
      saved.push(...r.saved)
      errors.push(...r.errors)
      i = j
      if (i < files.length && Date.now() - refreshed > 4000) {
        refreshed = bust.value = Date.now()
        await load()
      }
    }
  } catch (e) {
    stopped = String((e as Error).message ?? e)
  } finally {
    uploading.value = false
  }
  bust.value = Date.now()
  await load() // clears `error`: what was skipped, and why, goes up after it
  const n = `${saved.length} file${saved.length === 1 ? '' : 's'}`
  flash(stopped ? `${n} added, then the upload stopped` : `${n} added` + (errors.length ? `, ${errors.length} skipped` : ''))
  if (stopped) errors.push(stopped)
  if (errors.length) error.value = errors.join('\n')
}
onMounted(() => {
  addEventListener('keydown', onEsc)
  // dropped on the Datasets page, which made this dataset for them
  if (handoff.name === props.name && handoff.files.length) upload(handoff.files.splice(0))
})
onUnmounted(() => removeEventListener('keydown', onEsc))
</script>

<template>
  <div class="page full">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/datasets" class="crumb">Datasets</RouterLink> {{ name }}</h1>
        <p class="lede">
          {{ items.length - counts.videos }} image{{ items.length - counts.videos === 1 ? '' : 's' }}<template v-if="counts.videos"> · {{ counts.videos }} video{{ counts.videos === 1 ? '' : 's' }}</template>
          <template v-if="counts.uncaptioned"> · <span class="warn">{{ counts.uncaptioned }} uncaptioned</span></template>
          <template v-if="counts.mismatched"> · <span class="warn" title="A mask with a different shape from its image is stretched over it and won't line up">{{ counts.mismatched }} mask{{ counts.mismatched === 1 ? '' : 's' }} don't fit</span></template>
        </p>
      </div>
      <ZenButton size="sm" icon="mdi mdi-vector-selection" :disabled="!maskStart" title="Paint or find (SAM3) where each picture trains; ←/→ moves between pictures" @click="openMasks">Mask editor</ZenButton>
      <ZenButton size="sm" icon="mdi mdi-upload" :disabled="uploading" @click="fileInput?.click()">Add files</ZenButton>
      <ZenButton size="sm" variant="primary" icon="mdi mdi-play" @click="router.push({ path: '/runs/new', query: { dataset: name } })">Train on this</ZenButton>
      <input ref="fileInput" type="file" multiple accept="image/*,video/*,.txt" hidden @change="upload([...(($event.target as HTMLInputElement).files ?? [])])" />
    </div>

    <div v-if="error" class="card err mono">{{ error }}</div>

    <!-- one fixed-height row: nothing in it appears, disappears or wraps -->
    <StickyBar>
    <div class="page-toolbar card">
      <div class="view-switch" role="tablist" aria-label="View">
        <button :class="{ on: layout === 'grid' }" title="Grid" @click="view = 'grid'"><i class="mdi mdi-view-grid-outline" /></button>
        <button :class="{ on: layout === 'list' }" title="Table" @click="view = 'list'"><i class="mdi mdi-view-list-outline" /></button>
        <button v-if="controlSets.length" :class="{ on: layout === 'pairs' }" :title="`Edit pairs: “${activeControl}” inputs → result, with the instruction`" @click="(activeControl ||= controlSets[0] ?? ''), (view = 'pairs')"><i class="mdi mdi-swap-horizontal" /></button>
      </div>
      <label class="zoom" title="Card size" :class="{ off: layout === 'list' }">
        <i class="mdi mdi-image-size-select-small" />
        <ZenSlider v-model="zoom" :min="120" :max="420" :step="10" :disabled="layout === 'list'" />
      </label>
      <div class="tb-field" title="Caption set: the captions shown and edited, and which pictures are muted">
        <i class="mdi mdi-text-box-multiple-outline" />
        <ZenCombo :model-value="activeSet" :items="setItems" :sorted="false" :searchable="false" size="sm" @update:model-value="pickSet(String($event))" />
      </div>
      <div class="tb-field" title="Show only some of the pictures">
        <i class="mdi mdi-filter-variant" />
        <ZenCombo v-model="filter" :items="filterItems" :sorted="false" :searchable="false" size="sm" />
      </div>
      <div class="search"><ZenInput v-model="query" placeholder="Search captions or names" size="sm" /></div>
      <button class="icon-btn" :class="{ on: panelShown }" :title="panelShown ? 'Hide the side panel' : 'Show the side panel: health, faces, auto-caption, bulk edit, display'" @click="toggleSide">
        <i class="mdi mdi-dock-right" />
      </button>
    </div>
    </StickyBar>

    <div class="ds-body" :class="{ 'with-side': sideOpen && !narrow }">
      <main class="ds-main">
    <div v-if="reviewing" class="reviewing">
      <i class="mdi mdi-filter-outline" />
      <span class="small">Showing the {{ reviewing.paths.size }} image{{ reviewing.paths.size === 1 ? '' : 's' }} a caption job wrote into <b>“{{ reviewing.job.set_name }}”</b></span>
      <button class="chip-btn" @click="reviewing = null"><i class="mdi mdi-close" /> Show all</button>
    </div>

    <ZenEmpty v-if="!loading && !items.length" title="Empty dataset" icon="mdi mdi-image-plus-outline">
      Drop images, videos, same-name .txt captions and masks (<span class="mono">name-masklabel.png</span>) anywhere on this page.
    </ZenEmpty>

    <DatasetGrid v-else-if="layout === 'grid'" />
    <DatasetPairs v-else-if="layout === 'pairs'" />
    <DatasetTable v-else />
      </main>

      <div v-if="narrow && drawerOpen" class="drawer-scrim" @click="drawerOpen = false" />
      <DatasetSide v-if="panelShown" :name="name" :drawer="narrow" />
    </div>

    <!-- a message and the selection's actions float over the bottom of the page: neither moves anything -->
    <div class="float">
      <Transition name="float">
        <div v-if="notice" class="notice zen-surface"><i v-if="uploading" class="mdi mdi-loading mdi-spin" />{{ notice }}</div>
      </Transition>
      <Transition name="float">
        <div v-if="selected.size || boxing" class="sel-bar zen-surface">
          <b>{{ selected.size }}<span class="lbl"> selected</span></b>
          <button class="chip-btn" :title="allShownSelected ? 'Deselect the shown images' : 'Select every shown image'" @click="toggleAll">
            <i :class="allShownSelected ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-multiple-blank-outline'" /><span class="lbl">{{ allShownSelected ? 'Deselect shown' : 'Select shown' }}</span>
          </button>
          <span class="sel-sep" />
          <button class="chip-btn" :title="`Keep them, but don't train on them in “${activeSet}”`" @click="setMuted([...selected], true)"><i class="mdi mdi-eye-off-outline" /><span class="lbl">Mute</span></button>
          <button class="chip-btn" :title="`Train on them again in “${activeSet}”`" @click="setMuted([...selected], false)"><i class="mdi mdi-eye-outline" /><span class="lbl">Unmute</span></button>
          <button class="chip-btn" title="Open the mask editor on the first selected picture" @click="openMasks"><i class="mdi mdi-vector-selection" /><span class="lbl">Masks</span></button>
          <button class="chip-btn danger" title="Delete the selected images with their captions and masks (asks first)" @click="removeMany([...selected])"><i class="mdi mdi-trash-can-outline" /><span class="lbl">Delete</span></button>
          <span class="sel-sep" />
          <button class="chip-btn" title="Clear the selection (Esc)" @click="selected.clear()"><i class="mdi mdi-close" /></button>
        </div>
      </Transition>
    </div>

    <ZenContextMenu ref="menu" :items="menuItems" />

    <ZenLightbox v-if="lightbox != null && lbItems.length" v-model:index="lightbox" v-model:thumb-fit="fit" :items="lbItems" :searchable="false" @close="lightbox = null">
      <template #side="{ index }">
        <DatasetLightboxSide v-if="shown[index]" :it="shown[index]" :name="name" />
      </template>
    </ZenLightbox>

    <div v-if="dragging" class="drop-cover"><i class="mdi mdi-tray-arrow-down" /> Drop to add to {{ name }}</div>
  </div>
</template>

<style scoped>
.warn {
  color: var(--zen-warn);
}
.err {
  border-color: var(--zen-danger);
  white-space: pre-wrap;
  font-size: 12px;
}
.chip-btn.danger:hover {
  color: var(--zen-danger);
  border-color: var(--zen-danger);
}
.chip-btn {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: var(--r-md);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.chip-btn.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  border-color: var(--zen-selected, var(--zen-accent));
}
.chip-btn {
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.chip-btn:hover {
  color: var(--zen-text);
}
.zoom {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
}
.zoom .zen-slider {
  width: 110px;
}
.search {
  width: 220px;
}
.reviewing {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 40%, transparent);
  background: color-mix(in srgb, var(--zen-accent) 8%, transparent);
}
.reviewing > :nth-child(2) {
  flex: 1;
}
/* ---- page layout: a slim header, one fixed toolbar, the pictures and a side panel ---- */
/* everything gives way before anything is cut off: the side panel's button is always there */
.page-toolbar .zoom {
  flex: 0 1 150px;
  min-width: 70px;
}
.page-toolbar .zoom.off {
  opacity: 0.35;
}
.tb-field {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 0 1 190px;
  min-width: 110px;
  color: var(--zen-muted);
}
.tb-field :deep(.zen-combo) {
  flex: 1;
  min-width: 0;
}
.icon-btn {
  flex: none;
  width: 32px;
  height: 30px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  font-size: 18px;
  cursor: pointer;
}
.icon-btn:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.icon-btn.on {
  color: var(--zen-selected, var(--zen-accent));
}
.ds-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 14px;
  align-items: start;
}
.ds-body.with-side {
  grid-template-columns: minmax(0, 1fr) 320px;
}
.drawer-scrim {
  position: fixed;
  inset: 0;
  z-index: 39;
  background: rgba(0, 0, 0, 0.4);
}
.ds-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
/* a message (files added, captions changed) and the selection's actions float over the page
   bottom, the message above the bar: neither moves anything, and one coming or going doesn't move
   the other. Their look (border, corners, background, shadow) is the theme's floating surface, as
   the right-click menu's is */
.float {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 18px;
  z-index: 50;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  pointer-events: none;
}
.float > * {
  pointer-events: auto;
}
.notice {
  display: flex;
  align-items: center;
  gap: 8px;
  max-width: calc(100vw / var(--zen-page-zoom) - 32px);
  padding: 6px 14px;
  font-size: 13px;
}
.sel-bar {
  display: flex;
  align-items: center;
  /* too many for the window (a phone, a large Text size): a second row, not buttons off screen */
  flex-wrap: wrap;
  justify-content: center;
  gap: 4px 8px;
  width: max-content; /* its own width, not the half window left: 50% would leave it */
  max-width: calc(100vw / var(--zen-page-zoom) - 32px);
  white-space: nowrap;
  padding: 6px 8px 6px 14px;
  font-size: 13px;
}
/* a narrow window: icons and the count only (each button's title still says what it does) */
@media (max-width: 760px) {
  .sel-bar .lbl,
  .sel-bar .sel-sep {
    display: none;
  }
}
.sel-sep {
  width: 1px;
  height: 18px;
  background: var(--zen-border);
}
.float-enter-active,
.float-leave-active {
  transition: opacity var(--zen-dur-fast), transform var(--zen-dur-fast);
}
.float-enter-from,
.float-leave-to {
  opacity: 0;
  transform: translateY(8px);
}
@media (max-width: 900px) {
  .page-toolbar .zoom,
  .tb-field > .mdi {
    display: none;
  }
}
</style>
