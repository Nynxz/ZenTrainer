<script setup lang="ts">
// The training adapter picker (kind "training"): adapters we know for the picked family
// (downloaded on pick), ones added from a Hugging Face link, and files already in a
// loras/training_adapters folder. Kind "base" picks a base LoRA the same way, from any loras folder.
// v-model is the name ComfyUI lists the file by (what the spec stores), '' for none.
import { ZenButton, ZenCombo, ZenInput } from '@nynxz/zenkit-ui'
import { computed, ref, watch } from 'vue'
import { downloadFor, finished, fmtBytes } from '@/stores/downloads'
import { EngineError, engine, type AdapterKind, type TrainingAdapter } from '@/api/engine'

const model = defineModel<string>({ required: true })
const props = withDefaults(defineProps<{ family: string | null; kind?: AdapterKind }>(), { kind: 'training' })

const NONE = '__none'
const list = ref<TrainingAdapter[]>([])
const pending = ref<string | null>(null) // id downloading for this pick
const error = ref<string | null>(null)

async function load() {
  try {
    list.value = await engine.adapters(props.family, props.kind)
  } catch {
    list.value = []
  }
}
watch(() => props.family, load, { immediate: true })

// a download finished: re-read install state, and select the one picked
watch(finished, async () => {
  await load()
  const a = list.value.find((x) => x.id === pending.value)
  if (a?.installed) {
    model.value = a.installed
    pending.value = null
  }
})

// the family's adapters, custom and local ones; other families' only when no family is picked
const shown = computed(() => list.value.filter((a) => a.fits || a.source !== 'curated' || !props.family))
const selected = computed(() => list.value.find((a) => a.installed && a.installed === model.value))
const items = computed(() => {
  const rows: Record<string, unknown>[] = [{ value: NONE, label: 'None' }]
  for (const a of shown.value) rows.push({ value: a.id, label: a.label, keywords: `${a.label} ${a.repo} ${a.name}`, a })
  if (model.value && !selected.value) rows.push({ value: `raw:${model.value}`, label: model.value })
  return rows as { value: string; label: string }[]
})
const value = computed({
  get: () => (pending.value ? pending.value : selected.value?.id ?? (model.value ? `raw:${model.value}` : NONE)),
  set: (v: string) => pick(v),
})

const dl = computed(() => {
  const a = list.value.find((x) => x.id === pending.value)
  return a ? downloadFor('loras', a.name) : undefined
})
const pct = computed(() => (dl.value?.total ? Math.floor((dl.value.done / dl.value.total) * 100) : 0))

async function pick(v: string) {
  error.value = null
  if (v === NONE) {
    pending.value = null
    model.value = ''
    return
  }
  if (v.startsWith('raw:')) return
  const a = list.value.find((x) => x.id === v)
  if (!a) return
  if (a.installed) {
    pending.value = null
    model.value = a.installed
    return
  }
  pending.value = a.id
  try {
    await engine.downloadAdapter(a.id)
  } catch (e) {
    pending.value = null
    error.value = e instanceof EngineError ? e.message : String(e)
  }
}
watch(
  () => dl.value?.status,
  (s) => {
    if (s === 'error' || s === 'cancelled') {
      error.value = s === 'error' ? dl.value?.error ?? 'download failed' : null
      pending.value = null
    }
  },
)
const cancel = () => dl.value && engine.cancelDownload(dl.value.id)

// suggest the family's adapter while none is set (MiniMax H3: the image adapter)
const suggestion = computed(() => (!model.value && !pending.value ? shown.value.find((a) => a.suggest && a.fits) : undefined))

// ---- add from Hugging Face ---------------------------------------------------------------
const adding = ref(false)
const link = ref('')
const busy = ref(false)
const choices = ref<{ file: string; size: number | null }[]>([])
async function add(file?: string) {
  if (!link.value.trim()) return
  busy.value = true
  error.value = null
  try {
    const r = await engine.addAdapter(link.value.trim(), props.family, file, props.kind)
    if (r.choices) {
      choices.value = r.choices
      return
    }
    choices.value = []
    adding.value = false
    link.value = ''
    await load()
    if (r.adapter) await pick(r.adapter.id)
  } catch (e) {
    error.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
async function forget(a: TrainingAdapter) {
  await engine.removeAdapter(a.id)
  if (selected.value?.id === a.id) model.value = ''
  await load()
}
</script>

<template>
  <div class="ap">
    <ZenCombo v-model="value" :items="items" :sorted="false" :item-height="44" :menu-width="440" placeholder="None">
      <template #option="{ item }">
        <div v-if="item.a" class="ap-opt">
          <div class="ap-main">
            <span class="ellipsis">{{ item.label }}</span>
            <span class="ap-sub ellipsis">{{ (item.a as TrainingAdapter).note || (item.a as TrainingAdapter).repo || (item.a as TrainingAdapter).name }}</span>
          </div>
          <span v-if="(item.a as TrainingAdapter).installed" class="ap-tag"><i class="mdi mdi-check" /></span>
          <span v-else class="ap-tag"><i class="mdi mdi-download" /> {{ fmtBytes((item.a as TrainingAdapter).size) }}</span>
          <button
            v-if="(item.a as TrainingAdapter).source === 'custom'"
            class="ap-x"
            title="Remove from the list (the file stays)"
            @click.stop="forget(item.a as TrainingAdapter)"
          >
            <i class="mdi mdi-close" />
          </button>
        </div>
        <span v-else>{{ item.label }}</span>
      </template>
    </ZenCombo>

    <div v-if="pending && dl" class="ap-line">
      <div class="ap-bar"><span :style="{ width: `${pct}%` }" /></div>
      <span class="mono">{{ pct }}%</span>
      <button class="ap-link" @click="cancel">Cancel</button>
    </div>
    <div v-else-if="suggestion" class="ap-line">
      <span class="muted">Suggested: {{ suggestion.label }}</span>
      <button class="ap-link" @click="pick(suggestion.id)">{{ suggestion.installed ? 'Use' : `Download ${fmtBytes(suggestion.size)}` }}</button>
    </div>
    <div v-else-if="selected?.note" class="ap-line muted ellipsis">{{ selected.note }}</div>

    <div v-if="adding" class="ap-add">
      <ZenInput v-model="link" placeholder="https://huggingface.co/owner/repo" @keydown.enter="add()" />
      <ZenButton size="sm" :disabled="busy || !link.trim()" @click="add()">Add</ZenButton>
    </div>
    <div v-if="adding && choices.length" class="ap-files">
      <button v-for="c in choices" :key="c.file" class="ap-file" :disabled="busy" @click="add(c.file)">
        <span class="mono ellipsis">{{ c.file }}</span><span class="muted nowrap">{{ fmtBytes(c.size) }}</span>
      </button>
    </div>
    <div class="ap-line">
      <span v-if="error" class="ap-err ellipsis">{{ error }}</span>
      <span class="spacer" />
      <button class="ap-link" @click="(adding = !adding), (choices = []), (error = null)">
        {{ adding ? 'Cancel' : 'Add from Hugging Face' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.ap {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.ap-opt {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-width: 0;
}
.ap-main {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}
.ap-sub {
  font-size: 11px;
  color: var(--zen-muted);
}
.ap-tag {
  font-size: 11px;
  color: var(--zen-muted);
  white-space: nowrap;
}
.ap-x {
  border: none;
  background: none;
  color: var(--zen-muted);
  padding: 2px;
  border-radius: var(--r-sm, 4px);
  cursor: pointer;
}
.ap-x:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.ap-line {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 18px;
  font-size: 11px;
  min-width: 0;
}
.ap-bar {
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.ap-bar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
.ap-link {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  font-size: 11px;
  color: var(--zen-accent);
  cursor: pointer;
  white-space: nowrap;
}
.ap-link:hover {
  text-decoration: underline;
}
.ap-err {
  color: var(--zen-danger, #e5484d);
}
.ap-add {
  display: flex;
  gap: 6px;
  align-items: center;
}
.ap-add :deep(.zen-input) {
  flex: 1;
}
.ap-files {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 160px;
  overflow: auto;
}
.ap-file {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  border: 1px solid var(--zen-border);
  background: none;
  color: var(--zen-text);
  font-size: 11px;
  padding: 4px 8px;
  border-radius: var(--r-sm, 4px);
  cursor: pointer;
  min-width: 0;
}
.ap-file:hover {
  background: var(--zen-surface-2);
}
.nowrap {
  white-space: nowrap;
}
</style>
