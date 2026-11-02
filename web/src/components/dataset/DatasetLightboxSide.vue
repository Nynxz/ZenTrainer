<script setup lang="ts">
// Beside a picture in the lightbox: its inputs (an edit pair), its caption to edit, its captions
// in the other sets (to compare, or copy one across), its mask, and the file's details.
import { reactive } from 'vue'
import type { DatasetItem } from '@/api/engine'
import { vAutosize } from '@/lib/autosize'
import { isVideoPath, kb, pct, secs, useDatasetContext } from './context'

const props = defineProps<{ it: DatasetItem; name: string }>()
const { activeSet, activeControl, inputs, cap, k, drafts, saving, save, ctrlView, thumb } = useDatasetContext()

// the image's captions in the other sets, to compare (and copy one across)
const otherSets = (it: DatasetItem) => Object.entries(it.captions ?? {}).filter(([st, t]) => st !== activeSet.value && t)
async function useCaption(it: DatasetItem, text: string) {
  drafts[k(it)] = text
  await save(it)
}
// the lightbox's other caption sets: a few lines each until clicked open (no inner scrollbars)
const openSets = reactive(new Set<string>())
const toggleSet = (st: string) => (openSets.has(st) ? openSets.delete(st) : openSets.add(st))
</script>

<template>
  <div class="side">
    <template v-if="inputs(it).length">
      <div class="eyebrow">{{ inputs(it).length > 1 ? 'Inputs' : 'Input' }} · {{ activeControl }}</div>
      <div class="side-inputs">
        <template v-for="c in inputs(it)" :key="c">
          <video v-if="isVideoPath(c)" :src="ctrlView(it, c)" muted loop autoplay playsinline :title="c" />
          <img v-else :src="thumb({ path: c }, 384)" alt="" :title="c" />
        </template>
      </div>
    </template>
    <div class="eyebrow">{{ inputs(it).length ? 'Instruction' : 'Caption' }}</div>
    <textarea
      v-autosize
      class="side-cap"
      :value="drafts[k(it)] ?? cap(it)"
      placeholder="No caption: add one"
      rows="3"
      @input="drafts[k(it)] = ($event.target as HTMLTextAreaElement).value"
      @blur="save(it)"
      @keydown.ctrl.enter="($event.target as HTMLTextAreaElement).blur()"
    />
    <div class="muted small">
      {{ saving[k(it)] ? 'saving…' : drafts[k(it)] != null ? 'unsaved: click away or Ctrl+Enter' : 'saved' }}
    </div>
    <div v-for="[st, text] in otherSets(it)" :key="st" class="other-set">
      <div class="other-head">
        <span class="eyebrow">{{ st }}</span>
        <span class="spacer" />
        <button class="mini" :title="`Copy this into “${activeSet}”`" @click="useCaption(it, text)">Use this</button>
      </div>
      <p :class="{ open: openSets.has(st) }" :title="openSets.has(st) ? '' : 'Show all'" @click="toggleSet(st)">{{ text }}</p>
    </div>
    <div v-if="it.mask" class="mask-info">
      <img :src="thumb({ path: it.mask! }, 256)" alt="" />
      <div class="small">
        <b>Mask</b> · {{ pct(it.mask_coverage) }} trains at full weight
        <div v-if="it.mask_mismatch" class="warn">A different shape from the image: it's stretched over it and won't line up.</div>
        <div class="muted">M toggles it over the picture</div>
      </div>
    </div>
    <RouterLink v-if="it.kind !== 'video'" :to="{ path: `/datasets/${encodeURIComponent(props.name)}/mask`, query: { path: it.path } }" class="edit-mask">
      <i class="mdi mdi-vector-selection" /> {{ it.mask ? 'Edit mask' : 'Make a mask' }} <span class="muted">· SAM3, brush, feather</span>
    </RouterLink>
    <dl>
      <dt>File</dt><dd class="mono">{{ it.path }}</dd>
      <dt>Size</dt><dd class="mono">{{ it.width }}×{{ it.height }} · {{ kb(it.bytes) }}</dd>
      <template v-if="it.kind === 'video'">
        <dt>Video</dt><dd class="mono">{{ secs(it.duration) }} · {{ it.fps }} fps · {{ it.frames }} frames</dd>
      </template>
      <dt v-if="it.repeats > 1">Repeats</dt><dd v-if="it.repeats > 1">×{{ it.repeats }}</dd>
    </dl>
  </div>
</template>

<style scoped>
.warn {
  color: var(--zen-warn);
}
.other-set {
  align-self: stretch;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-surface-2) 60%, transparent);
}
.other-set p {
  margin: 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--zen-muted);
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
  cursor: pointer;
}
.other-set p.open {
  display: block;
  cursor: default;
}
.other-head {
  display: flex;
  align-items: center;
}
.mini {
  height: 22px;
  padding: 0 8px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.mini:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.edit-mask {
  font-size: 12px;
  padding: 6px 10px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.edit-mask:hover {
  border-color: var(--zen-accent);
}
.mask-info {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}
.mask-info img {
  width: 64px;
  height: 64px;
  flex: none;
  object-fit: contain;
  border-radius: var(--r-md);
  background: #000;
}
.side-inputs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
  gap: 6px;
  margin-bottom: 6px;
}
.side-inputs img,
.side-inputs video {
  width: 100%;
  border-radius: var(--r-sm);
  display: block;
}
/* lightbox side panel */
.side {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: flex-start;
  min-width: 0;
}
/* nothing wider than the panel: long captions (a JSON one has no spaces to break at) wrap anywhere */
.side > * {
  max-width: 100%;
  min-width: 0;
}
.side :is(p, dd, .small) {
  overflow-wrap: anywhere;
}
.side-cap {
  display: block;
  box-sizing: border-box;
  width: 100%;
  overflow-wrap: anywhere;
  resize: none;
  font: inherit;
  font-size: 13px;
  line-height: 1.6;
  min-height: 5em;
  max-height: calc(55vh / var(--zen-page-zoom));
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: color-mix(in srgb, var(--zen-text) 25%, transparent) transparent;
  padding: 10px 12px;
  border-radius: var(--r-md);
  border: 1px solid var(--zen-border);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
}
.side-cap:focus {
  outline: none;
  border-color: var(--zen-accent);
}
dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 12px;
  margin: 4px 0;
  font-size: 12px;
}
dt {
  color: var(--zen-muted);
}
dd {
  margin: 0;
  word-break: break-all;
}
</style>
