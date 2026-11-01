<script setup lang="ts">
// The dataset page's side panel: health, faces, auto-caption, bulk edit and display. Pinned
// beside the pictures, or, in a narrow window, a drawer over the right edge.
import { ZenButton, ZenInput, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, onUnmounted, ref, watch } from 'vue'
import { engine } from '@/api/engine'
import { remembered } from '@/lib/remembered'
import { activeCaptions } from '@/stores/captions'
import CaptionPanel from '@/components/CaptionPanel.vue'
import DatasetStats from '@/components/DatasetStats.vue'
import { useDatasetContext } from './context'
import { zoomOf } from '@/lib/zoom'

const props = defineProps<{ name: string; drawer: boolean }>()
const { fit, showMasks, maskHighVis, view, items, sets, activeSet, controlSets, activeControl, selected, load, review, counts, imageFlags, statsPanel, faces, facesBusy, facesStatus, facesShort, captionSubject, facesSetting, toggleFaces, flash } =
  useDatasetContext()

// The side panel stays pinned without a scrollbar of its own: under the sticky toolbar when it fits
// the window, or, when it's taller, scrolling with the page until its bottom reaches the window's
// bottom and staying there (a negative sticky top), so it never scrolls out of sight
const sideEl = ref<HTMLElement | null>(null)
const sideTop = ref(58)
const PIN = 66 // the sticky toolbar (10px down, 46px tall) + a 10px gap
function placeSide() {
  const h = sideEl.value?.offsetHeight ?? 0
  sideTop.value = Math.min(PIN, window.innerHeight / zoomOf() - h - 12)
}
// as a drawer (a narrow window) it opens under the toolbar, where it sits beside the pictures in a
// wide one, overlaid instead: following the toolbar as the page scrolls (it sticks at the top)
const drawerTop = ref(66)
function placeDrawer() {
  const bar = document.querySelector('.page-toolbar')
  drawerTop.value = Math.max(12, Math.round((bar ? bar.getBoundingClientRect().bottom / zoomOf() : 56) + 10))
}
watch(
  () => props.drawer,
  (on) => {
    window.removeEventListener('scroll', placeDrawer, true)
    window.removeEventListener('resize', placeDrawer)
    if (!on) return
    placeDrawer()
    window.addEventListener('scroll', placeDrawer, { passive: true, capture: true })
    window.addEventListener('resize', placeDrawer, { passive: true })
  },
  { immediate: true },
)
onUnmounted(() => {
  window.removeEventListener('scroll', placeDrawer, true)
  window.removeEventListener('resize', placeDrawer)
})
let sideObs: ResizeObserver | null = null
watch(sideEl, (el) => {
  sideObs?.disconnect()
  if (!el) return
  sideObs = new ResizeObserver(placeSide)
  sideObs.observe(el)
})
window.addEventListener('resize', placeSide)
onUnmounted(() => {
  sideObs?.disconnect()
  window.removeEventListener('resize', placeSide)
})

// ---- bulk edit ------------------------------------------------------------------------------
const bulkText = ref('')
const findText = ref('')
// the captions as they were before the last bulk edit, so it can be undone
const bulkUndo = ref<{ set: string; before: Record<string, string> } | null>(null)
async function bulk(op: 'prefix' | 'suffix' | 'replace') {
  if (!bulkText.value && op !== 'replace') return
  const set = activeSet.value
  const before = Object.fromEntries(items.value.map((it) => [it.path, it.captions[set] ?? '']))
  const touched = Object.values(before).filter((c) => (op === 'replace' ? c.includes(findText.value) : true)).length
  const what = op === 'replace' ? `Replace “${findText.value}” with “${bulkText.value}”` : `${op === 'prefix' ? 'Put' : 'Append'} “${bulkText.value}” ${op === 'prefix' ? 'at the start of' : 'to'}`
  if (!touched || !confirm(`${what} ${op === 'replace' ? 'in' : ''} ${touched} caption${touched === 1 ? '' : 's'} in “${set}”? You can undo it from here.`)) return
  const { changed } = await engine.bulkCaptions(props.name, op, bulkText.value, findText.value, set)
  bulkUndo.value = changed ? { set, before } : null
  flash(`${changed} caption${changed === 1 ? '' : 's'} updated`)
  await load()
}
async function undoBulk() {
  const u = bulkUndo.value
  if (!u) return
  let n = 0
  for (const it of items.value) {
    const old = u.before[it.path]
    if (old != null && (it.captions[u.set] ?? '') !== old) {
      await engine.setCaption(props.name, it.path, old, u.set)
      n++
    }
  }
  bulkUndo.value = null
  flash(`${n} caption${n === 1 ? '' : 's'} put back`)
  await load()
}

// ---- edit inputs: control sets, named layers of input pictures beside the dataset -----------
const inputsOpen = ref(true)
const inputCount = (set: string) => items.value.filter((it) => it.control_sets?.[set]?.length).length
const SET_RE = /^[a-z0-9][a-z0-9_-]{0,23}$/
function askSetName(text: string, now = ''): string | null {
  const v = prompt(text, now)?.trim().toLowerCase()
  if (!v) return null
  if (!SET_RE.test(v)) {
    alert('Use lowercase letters, digits, - and _ (max 24).')
    return null
  }
  return v
}
async function newInputSet() {
  const set = askSetName('New input set (e.g. faceswap, depth, sketch):')
  if (!set) return
  await engine.createControlSet(props.name, set)
  await load()
  activeControl.value = set
}
async function renameInputSet() {
  const from = activeControl.value
  const to = askSetName(`Rename “${from}” to:`, from)
  if (!to || to === from) return
  try {
    await engine.renameControlSet(props.name, from, to)
  } catch (e) {
    return alert(String((e as Error).message ?? e))
  }
  await load()
  activeControl.value = to
}
async function deleteInputSet() {
  const set = activeControl.value
  const n = items.value.reduce((sum, it) => sum + (it.control_sets?.[set]?.length ?? 0), 0) // files, not pictures
  if (!confirm(`Delete the input set “${set}”${n ? ` and its ${n} input picture${n === 1 ? '' : 's'}` : ''}? The dataset's pictures stay.`)) return
  await engine.deleteControlSet(props.name, set)
  flash(`“${set}” deleted`)
  await load()
}
const inputFiles = ref<HTMLInputElement | null>(null)
async function addInputs(files: FileList | null) {
  if (!files?.length) return
  const set = activeControl.value
  const r = await engine.uploadControls(props.name, set, [...files])
  if (inputFiles.value) inputFiles.value.value = ''
  flash(`${r.saved.length} input${r.saved.length === 1 ? '' : 's'} added to “${set}”`)
  if (r.errors.length) alert(`${r.errors.length} not added:\n${r.errors.slice(0, 12).join('\n')}`)
  await load()
  if (r.saved.length) view.value = 'pairs'
}

// ---- health: folds away, remembered per browser; folded, it still loads (the grid's picture
// flags come from it), it's only hidden ---------------------------------------------------------
const healthOpen = remembered('zentrainer.datasetSide.health', true)
// faces and how pictures show: settings, folded away until wanted
const settingsOpen = remembered('zentrainer.datasetSide.settings', false)

// ---- auto-caption: CaptionPanel runs it as a job -----------------------------------------
// open while this dataset has a caption job going, so coming back to the page shows it
const capRunning = computed(() => activeCaptions.value.some((j) => j.dataset === props.name))
// captions: auto-caption and bulk edit, one section (open while a caption job runs here)
const capOpen = ref(capRunning.value)
const capTab = ref<'auto' | 'bulk'>((() => {
  try {
    return localStorage.getItem('zentrainer.datasetSide.capTab') === 'bulk' ? 'bulk' : 'auto'
  } catch {
    return 'auto'
  }
})())
watch(capTab, (v) => {
  try {
    localStorage.setItem('zentrainer.datasetSide.capTab', v)
  } catch {
    // storage blocked
  }
})
watch(capRunning, (v) => {
  if (v) [capOpen.value, capTab.value] = [true, 'auto']
})
function onCaptioned(set: string) {
  if (!sets.value.includes(set)) sets.value = [...sets.value, set]
  load()
}
</script>

<template>
  <aside ref="sideEl" class="ds-side" :class="{ drawer }" :style="drawer ? { top: `${drawerTop}px` } : { top: `${sideTop}px` }">
    <section class="side-sec">
      <header class="side-head clickable" :title="healthOpen ? 'Fold away' : 'Show the dataset\'s health'" @click="healthOpen = !healthOpen">
        <span>Health <span class="muted small set-name">“{{ activeSet }}”</span></span>
        <i :class="healthOpen ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
      </header>
      <DatasetStats
        v-if="items.length"
        v-show="healthOpen"
        :key="`${name}:${activeSet}`"
        ref="statsPanel"
        :name="name"
        :set="activeSet"
        compact
        @loaded="(s) => (imageFlags = s.current.flags ?? {})"
      />
      <RouterLink
        v-if="healthOpen && items.length"
        :to="{ path: `/datasets/${encodeURIComponent(name)}/analysis`, query: activeSet !== 'default' ? { set: activeSet } : {} }"
        class="side-note analysis-link"
      >
        <i class="mdi mdi-chart-box-outline" /> Full analysis: buckets, captions, duplicates <i class="mdi mdi-arrow-right" />
      </RouterLink>
      <!-- with faces on, how they read: part of the dataset's health -->
      <p v-if="healthOpen && facesSetting && (facesBusy || faces?.items)" class="side-note faces-line">
        <i :class="facesBusy ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-face-recognition'" />
        <template v-if="facesBusy">{{ facesStatus }}</template>
        <template v-else-if="faces?.items">
          {{ faces.with_face }}/{{ faces.total }} with a face{{ faces.one_identity === false ? ' · mixed people' : ` · alike ${faces.identity?.toFixed(2) ?? '–'}` }}{{ counts.faceIssues ? ` · ${counts.faceIssues} to look at` : '' }}
        </template>
      </p>
    </section>
    <section class="side-sec">
      <header class="side-head clickable" @click="capOpen = !capOpen">
        <span>Captions</span><i :class="capOpen ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
      </header>
      <div v-if="capOpen" class="cap-tabs">
        <button :class="{ on: capTab === 'auto' }" @click="capTab = 'auto'"><i class="mdi mdi-creation-outline" /> Auto-caption</button>
        <button :class="{ on: capTab === 'bulk' }" @click="capTab = 'bulk'"><i class="mdi mdi-format-text-variant-outline" /> Bulk edit</button>
      </div>
      <CaptionPanel v-if="capOpen && capTab === 'auto'" :dataset="name" :items="items" :sets="sets" :selected="[...selected]" :subject="captionSubject" class="side-caption" @changed="onCaptioned" @review="review" />
      <div v-if="capOpen && capTab === 'bulk'" class="side-bulk">
        <span class="muted small">Every caption in “{{ activeSet }}”:</span>
        <ZenInput v-model="bulkText" placeholder="trigger word / text" size="sm" />
        <div class="side-row">
          <ZenButton size="sm" title="Put the text at the start of every caption" @click="bulk('prefix')">Prefix</ZenButton>
          <ZenButton size="sm" title="Append the text to every caption" @click="bulk('suffix')">Suffix</ZenButton>
        </div>
        <ZenInput v-model="findText" placeholder="find (to replace with the text)" size="sm" />
        <ZenButton size="sm" :disabled="!findText" title="Replace 'find' with the text everywhere" @click="bulk('replace')">Replace</ZenButton>
        <ZenButton v-if="bulkUndo" size="sm" icon="mdi mdi-undo" :title="`Put the captions in “${bulkUndo.set}” back as they were before the last bulk edit`" @click="undoBulk">Undo last edit</ZenButton>
      </div>
    </section>
    <section class="side-sec">
      <header class="side-head clickable" @click="inputsOpen = !inputsOpen">
        <span>Edit inputs</span><i :class="inputsOpen ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
      </header>
      <template v-if="inputsOpen">
        <p v-if="!controlSets.length" class="side-note">
          Input pictures for edit training, in named sets beside these pictures. The dataset stays as it is: a run chooses whether to use a set.
        </p>
        <div v-else class="input-sets" role="radiogroup" aria-label="Input set shown">
          <button
            v-for="c in controlSets"
            :key="c"
            class="input-set"
            :class="{ on: c === activeControl }"
            role="radio"
            :aria-checked="c === activeControl"
            :title="c === activeControl ? `Showing “${c}”'s inputs: click again to hide them` : `Show “${c}”'s inputs in the pairs view, the grid and the lightbox`"
            @click="activeControl = c === activeControl ? '' : c"
          >
            <i :class="c === activeControl ? 'mdi mdi-radiobox-marked' : 'mdi mdi-radiobox-blank'" />
            <span class="ellipsis">{{ c }}</span>
            <span class="spacer" />
            <span class="mono small muted">{{ inputCount(c) }}/{{ items.length }}</span>
          </button>
        </div>
        <div class="side-row">
          <ZenButton size="sm" icon="mdi mdi-plus" title="A new, empty input set" @click="newInputSet">New set</ZenButton>
          <ZenButton v-if="activeControl" size="sm" icon="mdi mdi-upload" :title="`Add input pictures to “${activeControl}”`" @click="inputFiles?.click()">Add inputs</ZenButton>
          <span class="spacer" />
          <button v-if="activeControl" class="icon-btn" :title="`Rename “${activeControl}”`" @click="renameInputSet"><i class="mdi mdi-pencil-outline" /></button>
          <button v-if="activeControl" class="icon-btn" :title="`Delete “${activeControl}” and its inputs`" @click="deleteInputSet"><i class="mdi mdi-delete-outline" /></button>
        </div>
        <p v-if="activeControl" class="side-note">
          Name each input like its picture: <span class="mono">photo.png</span>, and <span class="mono">photo.2.png</span> for a second.
        </p>
        <input ref="inputFiles" type="file" multiple hidden accept="image/*,video/*,audio/*" @change="addInputs(($event.target as HTMLInputElement).files)" />
      </template>
    </section>
    <section class="side-sec">
      <header class="side-head clickable" @click="settingsOpen = !settingsOpen">
        <span>Settings</span><i :class="settingsOpen ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" />
      </header>
      <template v-if="settingsOpen">
      <div class="side-row" :title="faces && !faces.available ? faces.hint : 'Faces checked on open, a Faces score in its health, and likeness for this dataset\'s runs'">
        <span class="small">Check faces</span><span class="spacer" />
        <span v-if="facesBusy" class="small muted faces-busy" :title="facesStatus"><i class="mdi mdi-loading mdi-spin" /> {{ facesShort }}</span>
        <ZenSwitch :model-value="facesSetting" :disabled="facesBusy || (!!faces && !faces.available)" @update:model-value="toggleFaces" />
      </div>
      <p v-if="faces && !faces.available && !facesBusy" class="side-note">{{ faces.hint }}</p>
      <div class="side-row">
        <span class="small">Crop to the card</span><span class="spacer" />
        <ZenSwitch :model-value="fit === 'cover'" @update:model-value="fit = $event ? 'cover' : 'contain'" />
      </div>
      <div class="side-row" :title="counts.masked ? 'Shows what each mask trains: the rest dims' : 'No masks in this dataset'">
        <span class="small">Show masks</span><span class="spacer" />
        <ZenSwitch :model-value="showMasks" :disabled="!counts.masked" @update:model-value="showMasks = $event" />
      </div>
      <div class="side-row" title="What a mask leaves out as a purple/black checker instead of faded into the background: unmistakable on dark or busy pictures. The mask editor's High-vis switch is the same setting">
        <span class="small">High-vis</span><span class="spacer" />
        <ZenSwitch v-model="maskHighVis" :disabled="!counts.masked || !showMasks" />
      </div>
      </template>
    </section>
  </aside>
</template>

<style scoped>
@keyframes side-in {
  from {
    opacity: 0;
    transform: translateX(16px);
  }
}
.ds-side.drawer {
  position: fixed;
  right: 12px;
  bottom: 12px;
  z-index: 40;
  width: min(340px, calc(100vw - 24px));
  overflow-y: auto;
  scrollbar-width: thin;
  overscroll-behavior: contain;
  filter: drop-shadow(0 18px 40px rgba(0, 0, 0, 0.55));
}
.ds-side {
  /* pinned (top set by placeSide), never a scrollbar of its own */
  position: sticky;
  animation: side-in 0.24s cubic-bezier(0.2, 0.7, 0.2, 1);
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.side-sec {
  padding: 10px 12px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-lg, 12px);
  background: var(--zen-surface);
}
.side-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
}
.cap-tabs {
  display: flex;
  gap: 4px;
}
.cap-tabs button {
  flex: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 5px 8px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.cap-tabs button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.faces-line {
  display: flex;
  align-items: center;
  gap: 6px;
}
.faces-busy {
  margin-right: 8px;
  font-variant-numeric: tabular-nums;
}
.set-name {
  font-weight: 400;
  margin-left: 4px;
}
.side-head.clickable {
  cursor: pointer;
}
.side-head.clickable i {
  color: var(--zen-muted);
}
.side-note {
  margin: 6px 0 0;
  font-size: 12px;
  line-height: 1.5;
  color: var(--zen-muted);
}
.side-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}
.side-bulk {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 8px;
}
.input-sets {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-top: 8px;
}
.input-set {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 30px;
  padding: 0 8px;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
}
.input-set:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.input-set.on {
  color: var(--zen-selected-text, var(--zen-text));
}
.input-set.on i {
  color: var(--zen-selected, var(--zen-accent));
}
.icon-btn {
  width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.icon-btn:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.side-caption {
  margin-top: 8px;
  border: none !important;
  padding: 0 !important;
  background: none !important;
}
.analysis-link {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
  text-decoration: none;
}
.analysis-link:hover {
  color: var(--zen-text);
}
</style>
