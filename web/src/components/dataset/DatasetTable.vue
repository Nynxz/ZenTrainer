<script setup lang="ts">
// The dataset as a list: a card per picture, its caption at full width and editable in place, and
// what's worth knowing about it (size, a mask, its face) as small chips beside the name. Sorting
// is a bar on top instead of column heads.
import { vAutosize } from '@/lib/autosize'
import { kb, pct, useDatasetContext } from './context'
import MoreRows from './MoreRows.vue'

const { cap, k, drafts, saving, save, sortKey, sortDir, selected, isMuted, faceOf, faceIssue, faceNote, shown, visible, more, sortBy, toggleSelect, onItemClick, onItemMenu, allShownSelected, toggleAll, full, thumb, fit } = useDatasetContext()
const SORTS = [
  { key: 'name', label: 'Name' },
  { key: 'res', label: 'Size' },
  { key: 'size', label: 'File' },
  { key: 'caption', label: 'Caption' },
] as const
</script>

<template>
  <div class="list">
    <div class="list-bar small">
      <label class="check-all" @click.prevent="toggleAll">
        <i :class="allShownSelected ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" /> All
      </label>
      <span class="spacer" />
      <span class="muted">Sort by</span>
      <button v-for="s in SORTS" :key="s.key" class="sort" :class="{ on: sortKey === s.key }" @click="sortBy(s.key)">
        {{ s.label }}<i v-if="sortKey === s.key" :class="sortDir > 0 ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" />
      </button>
    </div>

    <div
      v-for="(it, i) in visible"
      :key="it.path"
      class="row card"
      :class="{ sel: selected.has(it.path), dirty: drafts[k(it)] != null, muted: isMuted(it) }"
      @contextmenu="onItemMenu(it, i, $event)"
    >
      <label class="c-check" @click.prevent="toggleSelect(it, i, $event)"><i :class="selected.has(it.path) ? 'mdi mdi-checkbox-marked' : 'mdi mdi-checkbox-blank-outline'" /></label>
      <button class="c-thumb" :data-peek="full(it)" @click="onItemClick(it, i, $event)"><img :src="thumb(it, 160)" loading="lazy" alt="" :style="{ objectFit: fit }" /></button>
      <div class="body">
        <div class="head">
          <span class="mono ellipsis name" :title="it.path">{{ it.path }}</span>
          <span class="chip mono">{{ it.width ? `${it.width}×${it.height}` : '–' }}</span>
          <span class="chip mono">{{ kb(it.bytes) }}</span>
          <span v-if="it.mask" class="chip mono" :class="{ warn: it.mask_mismatch }" :title="it.mask_mismatch ? 'Its mask is a different shape from the picture: it won\'t line up' : `Mask: trains ${pct(it.mask_coverage)} of it`">
            <i class="mdi mdi-vector-selection" />{{ pct(it.mask_coverage) }}
          </span>
          <span v-if="faceOf(it)" class="chip mono" :class="{ warn: !!faceIssue(it) }" :title="faceIssue(it) ?? faceNote(it)">
            <i :class="faceIssue(it) ? 'mdi mdi-alert-outline' : 'mdi mdi-face-recognition'" />{{ faceOf(it)!.faces === 0 ? 'no face' : faceOf(it)!.faces > 1 ? `${faceOf(it)!.faces} faces` : faceOf(it)!.likeness?.toFixed(2) }}
          </span>
          <span v-if="isMuted(it)" class="chip" title="Kept, but not trained on in this caption set"><i class="mdi mdi-eye-off-outline" />muted</span>
          <span v-if="saving[k(it)]" class="muted saving">saving…</span>
        </div>
        <textarea
          v-autosize
          class="cap-edit"
          :class="{ empty: !cap(it) && drafts[k(it)] == null }"
          :value="drafts[k(it)] ?? cap(it)"
          placeholder="No caption: add one"
          rows="1"
          @input="drafts[k(it)] = ($event.target as HTMLTextAreaElement).value"
          @blur="save(it)"
          @keydown.ctrl.enter="($event.target as HTMLTextAreaElement).blur()"
        />
      </div>
    </div>
    <MoreRows v-if="visible.length < shown.length" @more="more" />
  </div>
</template>

<style scoped src="./items.css"></style>
<style scoped>
.list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.list-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 4px 2px;
}
.check-all {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
  cursor: pointer;
}
.check-all i {
  font-size: 16px;
}
.sort {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 2px 8px;
  border: 1px solid transparent;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  cursor: pointer;
}
.sort:hover {
  color: var(--zen-text);
}
.sort.on {
  color: var(--zen-selected-text, var(--zen-text));
  border-color: color-mix(in srgb, var(--zen-border) 80%, transparent);
  background: var(--zen-selected-bg, var(--zen-surface-2));
}
.row {
  display: grid;
  grid-template-columns: 22px 64px minmax(0, 1fr);
  gap: 12px;
  align-items: start;
  padding: 8px 10px;
}
.row.sel {
  border-color: color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 60%, transparent);
  background: var(--zen-selected-bg, color-mix(in srgb, var(--zen-selected, var(--zen-accent)) 8%, transparent));
}
.row.dirty .cap-edit {
  border-color: color-mix(in srgb, var(--zen-accent) 60%, transparent);
}
.row.muted .c-thumb img {
  opacity: 0.35;
}
.c-check {
  padding-top: 2px;
  font-size: 18px;
  color: var(--zen-muted);
  cursor: pointer;
}
.row.sel .c-check {
  color: var(--zen-selected, var(--zen-accent));
}
.c-thumb {
  width: 64px;
  height: 64px;
  padding: 0;
  border: none;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: zoom-in;
}
.c-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover; /* Crop to the card off: contain, set inline */
  display: block;
}
.body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.head {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  flex-wrap: wrap;
}
.name {
  font-size: 12px;
  font-weight: 600;
  min-width: 0;
  max-width: 100%;
  margin-right: 4px;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 0 6px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  color: var(--zen-muted);
  font-size: 10.5px;
  line-height: 18px;
}
.chip.warn {
  color: var(--zen-warn);
  background: color-mix(in srgb, var(--zen-warn) 14%, transparent);
}
.saving {
  font-size: 11px;
}
</style>
