<script setup lang="ts">
// The run's samples as a list: each its prompt and, if it wants, its own size (and length, on
// video models). A sample left on "Run size" follows the run's width/height/frames.
import { ZenCombo, ZenNumber } from '@nynxz/zenkit-ui'
import { computed } from 'vue'
import { vAutosize } from '@/lib/autosize'
import type { SamplePrompt } from '@/api/engine'

const model = defineModel<(string | SamplePrompt)[]>({ required: true })
const props = defineProps<{
  width: number
  height: number
  frames: number
  video: boolean
  trigger: string
  /** with a trigger per folder: the folders' words, which a prompt names to show that folder */
  words?: string[]
  /** edit runs: the dataset's input pictures to pick from (each sample edits one) */
  controls?: { value: string; thumb: string }[]
}>()

const rows = computed(() => model.value.map((p) => (typeof p === 'string' ? { prompt: p } : p)))
// plain strings while nothing is overridden, so older specs and saved sets stay as they were
function put(i: number, patch: Partial<SamplePrompt>) {
  const next = { ...rows.value[i]!, ...patch }
  const bare = next.width == null && next.height == null && next.frames == null && next.seed == null && !next.control
  const list = [...model.value]
  list[i] = bare ? next.prompt : next
  model.value = list
}
const add = () => (model.value = [...model.value, model.value.length ? '' : '[trigger], '])
const remove = (i: number) => (model.value = model.value.filter((_, n) => n !== i))
function move(i: number, by: number) {
  const j = i + by
  if (j < 0 || j >= model.value.length) return
  const list = [...model.value]
  ;[list[i], list[j]] = [list[j]!, list[i]!]
  model.value = list
}

const SIZES: [number, number][] = [
  [1024, 1024],
  [832, 1216],
  [1216, 832],
  [768, 1344],
  [1344, 768],
  [480, 832],
  [832, 480],
  [512, 512],
]
const sizeItems = computed(() => [
  { value: '', label: `Run size (${props.width}×${props.height})` },
  ...SIZES.map(([w, h]) => ({ value: `${w}x${h}`, label: `${w}×${h}${w === h ? ' square' : w < h ? ' portrait' : ' landscape'}` })),
])
const controlItems = computed(() => [
  { value: '', label: 'No input picture' },
  ...(props.controls ?? []).map((c) => ({ value: c.value, label: c.value })),
])
const controlThumb = (v?: string) => props.controls?.find((c) => c.value === v)?.thumb
const sizeOf = (r: SamplePrompt) => (r.width && r.height ? `${r.width}x${r.height}` : '')
function setSize(i: number, v: string | number) {
  const [w, h] = String(v).split('x').map(Number)
  put(i, w && h ? { width: w, height: h } : { width: undefined, height: undefined })
}
</script>

<template>
  <div class="spl">
    <div v-for="(r, i) in rows" :key="i" class="row">
      <span class="n mono">{{ i + 1 }}</span>
      <textarea
        v-autosize
        class="prompt"
        rows="1"
        :value="r.prompt"
        placeholder="[trigger], a portrait photo, soft light"
        @input="put(i, { prompt: ($event.target as HTMLTextAreaElement).value })"
      />
      <div class="opts">
        <template v-if="controls">
          <img v-if="controlThumb(r.control)" class="cthumb" :src="controlThumb(r.control)" alt="" />
          <ZenCombo
            :model-value="r.control ?? ''"
            :items="controlItems"
            :sorted="false"
            size="sm"
            title="The picture this sample edits; the sample comes out its shape"
            @update:model-value="put(i, { control: String($event) || undefined })"
          />
        </template>
        <ZenCombo v-if="!controls || !r.control" :model-value="sizeOf(r)" :items="sizeItems" :sorted="false" :searchable="false" size="sm" @update:model-value="setSize(i, $event)" />
        <label v-if="video" class="frames" title="Frames (at the run's fps); empty follows the run">
          <ZenNumber
            :model-value="r.frames ?? frames"
            :min="1"
            :max="361"
            :step="17"
            :precision="0"
            size="sm"
            @update:model-value="put(i, { frames: $event === frames ? undefined : $event })"
          />
          <span class="muted">frames</span>
        </label>
        <div class="acts">
          <button title="Move up" :disabled="i === 0" @click="move(i, -1)"><i class="mdi mdi-chevron-up" /></button>
          <button title="Move down" :disabled="i === rows.length - 1" @click="move(i, 1)"><i class="mdi mdi-chevron-down" /></button>
          <button class="del" title="Remove this sample" @click="remove(i)"><i class="mdi mdi-close" /></button>
        </div>
      </div>
    </div>
    <button class="add" @click="add"><i class="mdi mdi-plus" /> Add a sample</button>
    <span v-if="words?.length" class="hint muted">A prompt shows a folder by naming its word: {{ words.join(', ') }}. [trigger] becomes {{ trigger ? `“${trigger}”` : 'nothing' }}.</span>
    <span v-else class="hint muted">[trigger] becomes {{ trigger ? `“${trigger}”` : 'your trigger word' }}.</span>
  </div>
</template>

<style scoped>
.spl {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.row {
  display: grid;
  grid-template-columns: 20px minmax(0, 1fr);
  gap: 4px 10px;
  padding: 8px 10px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-surface-2) 40%, transparent);
}
.n {
  color: var(--zen-muted);
  font-size: 11px;
  padding-top: 6px;
  grid-row: span 2;
}
.prompt {
  display: block;
  width: 100%;
  resize: none;
  font: inherit;
  font-size: 13px;
  line-height: 1.5;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--zen-text);
  max-height: calc(6 * 1.5em + 10px);
  overflow-y: auto;
  scrollbar-width: thin;
}
.prompt:hover {
  border-color: color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.prompt:focus {
  outline: none;
  border-color: var(--zen-accent);
  background: var(--zen-field-bg, var(--zen-input));
}
.opts {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.opts > :first-child {
  width: 220px;
}
.cthumb {
  width: 26px;
  height: 26px;
  border-radius: var(--r-sm);
  object-fit: cover;
  flex: none;
}
.frames {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
}
.frames :deep(.zen-number) {
  width: 110px;
}
.acts {
  display: flex;
  gap: 2px;
  margin-left: auto;
}
.acts button {
  width: 26px;
  height: 26px;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.acts button:hover:not(:disabled) {
  background: var(--zen-surface-2);
  color: var(--zen-text);
}
.acts button.del:hover {
  color: var(--zen-danger);
}
.acts button:disabled {
  opacity: 0.3;
  cursor: default;
}
.add {
  align-self: flex-start;
  border: 1px dashed color-mix(in srgb, var(--zen-border) 90%, transparent);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 6px 12px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.add:hover {
  border-color: var(--zen-accent);
  color: var(--zen-text);
}
.hint {
  font-size: 11px;
}
</style>
