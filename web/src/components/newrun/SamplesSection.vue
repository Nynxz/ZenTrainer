<script setup lang="ts">
// 4 Samples: the prompts (and saved prompt sets), how often they render, with which model, and the
// sampler's settings.
import { ZenButton, ZenCombo, ZenField, ZenInput, ZenNumber, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, nextTick, reactive, ref, watch } from 'vue'
import { engine, thumbUrl } from '@/api/engine'
import SamplePromptList from '@/components/SamplePromptList.vue'
import { downloadFor, fmtBytes } from '@/stores/downloads'
import { items, merge, useRunFormContext } from './form'

const { spec, choices, sampleSuggestion, promptSets, pairItems, runInputs, isVideoModel } = useRunFormContext()

const advanced = reactive({ samples: false })
const usingSuggestion = computed(
  () => !!sampleSuggestion.value && spec.samples.model_name === sampleSuggestion.value.installed && spec.samples.steps === sampleSuggestion.value.steps,
)
function useSuggestion() {
  const s = sampleSuggestion.value
  if (!s?.installed) return
  Object.assign(spec.samples, { model_name: s.installed, steps: s.steps, cfg: s.cfg, sampler: s.sampler, scheduler: s.scheduler })
}
// the LoRA under samples (spec: turbo_lora): a "+" until one is added, then a picker with its
// strength and an x. A turbo one (by its name, as models/families.py DISTILLED) offers few-step
// settings, and clearing it offers the ones from before (when known)
const TURBO = { steps: 8, cfg: 1 }
const turboOpen = ref(!!spec.samples.turbo_lora)
const turboCleared = ref(false)
const plain = ref<{ steps: number; cfg: number } | null>(null)  // the settings from before a turbo LoRA, once seen
const isTurboSteps = () => spec.samples.steps <= 12 && spec.samples.cfg <= 1.5
const DISTILLED = /turbo|distill|lightning|hyper|lcm|schnell|dmd|\bfast\b/i
const turboSteps = computed(() => DISTILLED.test(spec.samples.turbo_lora ?? '') && !isTurboSteps() && !spec.samples.sigmas?.length)
const plainSteps = computed(() => turboCleared.value && !!plain.value && isTurboSteps() && !spec.samples.sigmas?.length)
watch(() => spec.samples.turbo_lora, (v) => v && (turboOpen.value = true))  // a recipe or an import set one
function pickTurbo(v: unknown) {
  spec.samples.turbo_lora = v == null || v === '' ? null : String(v)
  if (DISTILLED.test(spec.samples.turbo_lora ?? '') && !isTurboSteps()) plain.value = { steps: spec.samples.steps, cfg: spec.samples.cfg }
}
function clearTurbo() {
  turboCleared.value = DISTILLED.test(spec.samples.turbo_lora ?? '')
  spec.samples.turbo_lora = null
  turboOpen.value = false
}
function setSteps(v: { steps: number; cfg: number }) {
  Object.assign(spec.samples, v)
}
// the scheduler, or "custom": the sigmas typed here (steps then follow from them)
const scheduleItems = computed(() => [...items(choices.scheduler ?? ['normal']), { value: 'custom', label: 'Custom sigmas', keywords: 'custom sigmas manual' }])
const schedule = computed({
  get: () => (spec.samples.sigmas?.length ? 'custom' : spec.samples.scheduler),
  set: (v: string) => {
    if (v === 'custom') spec.samples.sigmas = spec.samples.sigmas?.length ? spec.samples.sigmas : [1, 0.9, 0.75, 0.55, 0.35, 0.15, 0]
    else {
      spec.samples.sigmas = []
      spec.samples.scheduler = v
    }
  },
})
const sigmasText = ref((spec.samples.sigmas ?? []).join(', '))
watch(() => spec.samples.sigmas, (v) => {
  if ((v ?? []).join(', ') !== sigmasText.value.split(/[\s,]+/).filter(Boolean).map(Number).join(', ')) sigmasText.value = (v ?? []).join(', ')
})
function setSigmas(input: string | number) {
  const text = String(input)
  sigmasText.value = text
  const v = text.split(/[\s,]+/).filter(Boolean).map(Number)
  if (v.length >= 2 && v.every((x, i) => Number.isFinite(x) && x >= 0 && (!i || x <= v[i - 1]!))) spec.samples.sigmas = v
}
const sigmasBad = computed(() => schedule.value === 'custom' && sigmasText.value.split(/[\s,]+/).filter(Boolean).map(Number).join(', ') !== (spec.samples.sigmas ?? []).join(', '))
const slgLayersText = computed({
  get: () => spec.samples.slg_layers.join(', '),
  set: (v: string) => (spec.samples.slg_layers = (v.match(/\d+/g) ?? []).map(Number)),
})
const sampleModel = computed({
  get: () => spec.samples.model_name ?? '',
  set: (v: string) => (spec.samples.model_name = v || null),
})
const insertTrigger = () => {
  spec.samples.prompts = [...spec.samples.prompts, '[trigger], ']
}

// ---- Fibo 2.0: sample prompts rewritten as its structured JSON captions -------------------------
// Fibo is trained on, and prompted with, JSON scene descriptions; a language model expands each
// plain prompt into one (Bria's VGL prompt, on a local text encoder). [trigger] is kept: the
// model sees the trigger word in its place and it goes back to [trigger] after.
const isFibo = computed(() => spec.model.kind === 'split' && spec.model.clip_type === 'fibo2')
const rw = reactive({ open: false, models: [] as string[], model: '', busy: false, note: '', undo: null as null | typeof spec.samples.prompts })
const isJson = (t: string) => t.trim().startsWith('{') && t.trim().endsWith('}')
async function openRewrite() {
  rw.open = !rw.open
  if (!rw.open || rw.models.length) return
  try {
    rw.models = (await engine.captionInfo()).models
    rw.model = rw.models.find((n) => /qwen3[-_.]?vl[-_]?(8|30|32)b/i.test(n)) ?? rw.models.find((n) => /gemma[-_]?4/i.test(n)) ?? rw.models.find((n) => /qwen3[-_.]?vl/i.test(n)) ?? rw.models[0] ?? ''
  } catch (e) {
    rw.note = String((e as Error).message ?? e)
  }
}
async function rewrite() {
  const trig = spec.dataset.trigger.trim()
  const list = spec.samples.prompts
  const todo = list.map((p, i) => ({ i, text: typeof p === 'string' ? p : p.prompt })).filter((r) => r.text.trim() && !isJson(r.text))
  if (!todo.length) return void (rw.note = 'Nothing to rewrite: every prompt is already JSON')
  rw.busy = true
  rw.note = `Rewriting ${todo.length} prompt${todo.length === 1 ? '' : 's'}…`
  try {
    let job = await engine.rewritePrompts({ clip_name: rw.model, prompts: todo.map((r) => (trig ? r.text.replaceAll('[trigger]', trig) : r.text)) })
    while (['queued', 'running'].includes(job.status)) {
      await new Promise((ok) => setTimeout(ok, 1500))
      job = await engine.captionJob(job.job_id)
      rw.note = job.status === 'queued' ? job.message || 'waiting for the GPU…' : `Rewriting… ${job.done}/${todo.length}`
    }
    if (job.status !== 'done') throw new Error(job.message || job.status)
    const next = [...list]
    let failed = 0
    todo.forEach((r, n) => {
      let out = job.results?.[n]?.text ?? ''
      if (!out) return void failed++
      if (trig) out = out.replaceAll(trig, '[trigger]')
      const old = next[r.i]!
      next[r.i] = typeof old === 'string' ? out : { ...old, prompt: out }
    })
    rw.undo = list
    spec.samples.prompts = next
    rw.note = `Rewrote ${todo.length - failed}` + (failed ? `; ${failed} kept as they were (no valid JSON)` : '')
  } catch (e) {
    rw.note = String((e as Error).message ?? e)
  } finally {
    rw.busy = false
  }
}
function undoRewrite() {
  if (rw.undo) spec.samples.prompts = rw.undo
  rw.undo = null
  rw.note = 'Back to the prompts as they were'
}
// edit pairs: a sample's input picture comes from the dataset's pairs
const controlOptions = computed(() =>
  pairItems.value.flatMap((it) => runInputs(it).map((c) => ({ value: c, thumb: thumbUrl(spec.dataset.folder, c, 128) }))),
)
function samplesFromPairs() {
  spec.samples.prompts = pairItems.value.slice(0, 2).map((it) => ({ prompt: it.caption, control: runInputs(it)[0] }))
}

// ---- prompt sets ---------------------------------------------------------------------------
// Picking a set loads it into the form to edit; once edited it shows as modified and can be
// saved back over itself (a built-in gets a user copy that takes its place), saved as a new
// set, or reverted.
const SET_KEYS = ['prompts', 'negative', 'width', 'height', 'steps', 'cfg', 'sampler', 'scheduler'] as const
const promptSet = ref<string>('')
const currentSet = computed(() => promptSets.value.find((p) => p.id === promptSet.value) ?? null)
const loadSet = () => currentSet.value && merge(spec.samples as unknown as Record<string, unknown>, currentSet.value.data as Record<string, unknown>)
watch(promptSet, loadSet)
const setModified = computed(() => {
  const set = currentSet.value
  if (!set) return false
  const d = set.data as Record<string, unknown>
  const now = spec.samples as unknown as Record<string, unknown>
  return Object.keys(d).some((k) => JSON.stringify(d[k]) !== JSON.stringify(now[k]))
})
const naming = ref(false)
const newSetName = ref('')
const savingSet = ref(false)
const nameBox = ref<HTMLElement | null>(null)
async function startNaming() {
  newSetName.value = currentSet.value ? `${currentSet.value.name} (copy)` : ''
  naming.value = true
  await nextTick()
  nameBox.value?.querySelector('input')?.select()
}
async function savePromptSet(name: string) {
  if (!name.trim()) return
  savingSet.value = true
  try {
    const data = JSON.parse(JSON.stringify(Object.fromEntries(SET_KEYS.map((k) => [k, spec.samples[k]])))) // reactive arrays don't structuredClone
    const saved = await engine.savePreset('prompts', name.trim(), data)
    const row = saved.preset as (typeof promptSets.value)[number]
    promptSets.value = [...promptSets.value.filter((p) => p.id !== row.id), row]
    promptSet.value = row.id
    naming.value = false
  } finally {
    savingSet.value = false
  }
}
</script>

<template>
<section class="card sec">
  <h2><span class="num">4</span> Samples</h2>
  <!-- one fixed-height row: the set picker and its actions, or the name field when saving -->
  <div v-if="naming" class="sample-head" @keydown.enter.prevent="savePromptSet(newSetName)" @keydown.esc="naming = false">
    <div ref="nameBox"><ZenInput v-model="newSetName" size="sm" placeholder="Name this prompt set" /></div>
    <ZenButton size="sm" variant="primary" icon="mdi mdi-content-save-outline" :disabled="!newSetName.trim() || savingSet" @click="savePromptSet(newSetName)">Save</ZenButton>
    <ZenButton size="sm" @click="naming = false">Cancel</ZenButton>
  </div>
  <div v-else class="sample-head">
    <div class="set-pick">
      <ZenCombo
        v-model="promptSet"
        :items="promptSets.map((p) => ({ value: p.id, label: p.name }))"
        placeholder="Saved prompt sets…"
        empty-text="No saved sets yet"
        size="sm"
      />
      <span v-if="setModified" class="set-dot" title="Edited since it was loaded" />
    </div>
    <template v-if="currentSet && setModified">
      <ZenButton size="sm" icon="mdi mdi-content-save-outline" :disabled="savingSet" :title="currentSet.builtin ? 'Save your edits as your own version of this built-in set' : `Overwrite “${currentSet.name}”`" @click="savePromptSet(currentSet.name)">Save</ZenButton>
      <ZenButton size="sm" icon="mdi mdi-content-save-plus-outline" @click="startNaming">Save as new</ZenButton>
      <ZenButton size="sm" icon="mdi mdi-undo-variant" title="Back to the saved prompts" @click="loadSet">Revert</ZenButton>
    </template>
    <ZenButton v-else-if="!currentSet" size="sm" icon="mdi mdi-content-save-outline" :disabled="!spec.samples.prompts.length" @click="startNaming">Save as set</ZenButton>
    <span class="spacer" />
    <ZenButton v-if="spec.dataset.edit && pairItems.length" size="sm" icon="mdi mdi-swap-horizontal" title="Sample two of the dataset's pairs: their instruction on their input picture" @click="samplesFromPairs">From pairs</ZenButton>
    <ZenButton v-if="isFibo" size="sm" icon="mdi mdi-code-json" :variant="rw.open ? 'primary' : 'default'" title="Rewrite the prompts as Fibo 2.0's structured JSON captions, the form it was trained on" @click="openRewrite">Fibo JSON</ZenButton>
    <ZenButton size="sm" icon="mdi mdi-tag-plus-outline" @click="insertTrigger">Insert [trigger]</ZenButton>
  </div>
  <div v-if="isFibo && rw.open" class="rewrite">
    <span class="small">Rewrite with</span>
    <ZenCombo v-model="rw.model" :items="rw.models.map((m) => ({ value: m, label: m.replace(/\.safetensors$/, '') }))" size="sm" class="rw-model" />
    <ZenButton size="sm" variant="primary" icon="mdi mdi-auto-fix" :disabled="rw.busy || !rw.model" @click="rewrite">Rewrite</ZenButton>
    <ZenButton v-if="rw.undo && !rw.busy" size="sm" icon="mdi mdi-undo-variant" @click="undoRewrite">Undo</ZenButton>
    <span class="muted small">{{ rw.note || 'Each plain prompt becomes a structured caption (Bria\'s VGL prompt): bigger models write better ones.' }}</span>
  </div>
  <SamplePromptList
    v-model="spec.samples.prompts"
    :width="spec.samples.width"
    :height="spec.samples.height"
    :frames="spec.samples.frames ?? 1"
    :video="isVideoModel"
    :trigger="spec.dataset.trigger"
    :controls="spec.dataset.edit ? controlOptions : undefined"
  />
  <div v-if="sampleSuggestion && !usingSuggestion" class="suggest">
    <i class="mdi mdi-lightning-bolt-outline" />
    <span>
      Preview faster on <b>{{ sampleSuggestion.model }}</b>: {{ sampleSuggestion.steps }} steps, CFG {{ sampleSuggestion.cfg }},
      {{ sampleSuggestion.sampler }} / {{ sampleSuggestion.scheduler }}, the settings ComfyUI's template uses with it.
    </span>
    <button v-if="sampleSuggestion.installed" class="chip-btn" @click="useSuggestion">Use</button>
    <template v-else-if="['queued', 'downloading'].includes(downloadFor(sampleSuggestion.folder, sampleSuggestion.model)?.status ?? '')">
      <span class="mono small">{{ Math.floor(((downloadFor(sampleSuggestion.folder, sampleSuggestion.model)!.done) / (downloadFor(sampleSuggestion.folder, sampleSuggestion.model)!.total || 1)) * 100) }}%</span>
    </template>
    <button v-else class="chip-btn" @click="engine.download(sampleSuggestion.family, sampleSuggestion.folder, sampleSuggestion.model)">
      <i class="mdi mdi-download" /> {{ fmtBytes(sampleSuggestion.size) || 'Download' }}
    </button>
  </div>
  <div class="grid-2">
    <ZenField label="Every" stack tip="Steps between samples; 0 = off (only Sample now). The save interval puts samples, and a likeness score, on every checkpoint">
      <ZenNumber v-model="spec.samples.every" :min="0" :step="50" :precision="0" />
    </ZenField>
    <ZenField label="Baseline before training" stack tip="Render once with the untouched model"><div class="sw-row"><ZenSwitch v-model="spec.samples.baseline" /><span class="sw-state">{{ spec.samples.baseline ? 'On' : 'Off' }}</span></div></ZenField>
    <ZenField label="Sample with another model" stack tip="E.g. train on raw, preview on turbo (same family)">
      <ZenCombo
        v-model="sampleModel"
        :items="[{ value: '', label: 'The model being trained' }, ...items(spec.model.kind === 'split' ? choices.unet : choices.ckpt)]"
        :pinned="['']"
        :item-height="30"
      />
    </ZenField>
    <ZenField v-if="spec.samples.model_name" label="LoRA strength" stack>
      <ZenNumber v-model="spec.samples.lora_strength" :min="0" :max="2" :step="0.05" :precision="2" />
    </ZenField>
  </div>
  <button class="more" @click="advanced.samples = !advanced.samples">
    <i :class="advanced.samples ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /> Sampler settings
  </button>
  <div v-if="advanced.samples" class="grid-2">
    <ZenField label="Size" stack>
      <div class="inline">
        <ZenNumber v-model="spec.samples.width" :min="256" :step="64" :precision="0" />
        <span class="muted">×</span>
        <ZenNumber v-model="spec.samples.height" :min="256" :step="64" :precision="0" />
      </div>
    </ZenField>
    <ZenField v-if="isVideoModel" label="Frames / fps" stack tip="Video samples: frames per sample (snapped to a length the VAE decodes; 1 = a still) and the frame rate they're saved at">
      <div class="inline">
        <ZenNumber :model-value="spec.samples.frames ?? 1" :min="1" :max="361" :step="17" :precision="0" @update:model-value="spec.samples.frames = $event" />
        <ZenNumber :model-value="spec.samples.fps ?? 24" :min="1" :max="60" :step="1" :precision="0" @update:model-value="spec.samples.fps = $event" />
      </div>
    </ZenField>
    <ZenField label="LoRA under samples" stack class="wide-f" tip="On under every sample, never in training or the saved LoRA. Usually a turbo LoRA, so previews take a few steps: MiniMax H3's minimax_h3_turbo_v4_step600_ema at 0.75, 6 steps, CFG 1">
      <div v-if="turboOpen" class="turbo">
        <div class="turbo-pick">
          <ZenCombo :model-value="spec.samples.turbo_lora ?? ''" :items="items(choices.loras)" placeholder="Pick a LoRA…" :item-height="30" @update:model-value="pickTurbo" />
          <label class="strength" title="Its strength under samples; 0 samples without it">
            <span class="muted small">Strength</span>
            <ZenNumber :model-value="spec.samples.turbo_strength ?? 0.75" :min="0" :max="1.5" :step="0.05" :precision="2" @update:model-value="spec.samples.turbo_strength = $event" />
          </label>
          <button type="button" class="remove" title="Sample without it" @click="clearTurbo"><i class="mdi mdi-close" /></button>
        </div>
        <p v-if="turboSteps" class="hint muted small">
          A turbo LoRA samples in a few steps; these are {{ spec.samples.steps }} at CFG {{ spec.samples.cfg }}.
          <button type="button" class="link" @click="setSteps(TURBO)">Use {{ TURBO.steps }} steps, CFG {{ TURBO.cfg }}</button>
        </p>
      </div>
      <div v-else class="turbo-add">
        <button type="button" class="add-lora" @click="turboOpen = true"><i class="mdi mdi-plus" /> Add a LoRA under samples</button>
        <p v-if="plainSteps" class="hint muted small">
          Without the turbo LoRA, {{ spec.samples.steps }} steps at CFG {{ spec.samples.cfg }} are likely too few.
          <button v-if="plain" type="button" class="link" @click="setSteps(plain)">Use {{ plain.steps }} steps, CFG {{ plain.cfg }}</button>
        </p>
      </div>
    </ZenField>
    <ZenField label="Steps / CFG" stack>
      <div class="inline">
        <ZenNumber v-model="spec.samples.steps" :min="1" :step="1" :precision="0" />
        <ZenNumber v-model="spec.samples.cfg" :min="0" :step="0.5" :precision="1" />
      </div>
    </ZenField>
    <ZenField label="Sampler" stack><ZenCombo v-model="spec.samples.sampler" :items="items(choices.sampler ?? ['euler'])" /></ZenField>
    <ZenField label="Scheduler" stack><ZenCombo v-model="schedule" :items="scheduleItems" :sorted="false" /></ZenField>
    <ZenField v-if="schedule === 'custom'" label="Sigmas" stack class="wide-f" :tip="`The noise levels sampling steps through, high to low (one step between each pair: ${Math.max(0, (spec.samples.sigmas?.length ?? 1) - 1)} steps), in the model's own sigma scale: 1 to 0 for flow models (Krea 2, Flux, MOSS). Steps above then doesn't apply`">
      <ZenInput :model-value="sigmasText" :class="{ bad: sigmasBad }" placeholder="1, 0.9, 0.75, 0.55, 0.35, 0.15, 0" @update:model-value="setSigmas" />
      <span v-if="sigmasBad" class="small warn">two or more numbers, high to low, none below 0</span>
    </ZenField>
    <ZenField label="Seed" stack><ZenNumber v-model="spec.samples.seed" :min="0" :step="1" :precision="0" /></ZenField>
    <ZenField label="Skip-layer guidance" stack tip="Negative pass: skip layers in the unconditional pass (free). Extra pass: one more forward per step">
      <div class="seg">
        <button v-for="m in (['off', 'uncond', 'standard'] as const)" :key="m" :class="{ on: spec.samples.slg_mode === m }" @click="spec.samples.slg_mode = m">
          {{ { off: 'Off', uncond: 'Negative pass', standard: 'Extra pass' }[m] }}
        </button>
      </div>
    </ZenField>
    <ZenField v-if="spec.samples.slg_mode !== 'off'" label="Skipped layers" stack tip="Block indices, from 0">
      <ZenInput v-model="slgLayersText" placeholder="22" />
    </ZenField>
    <template v-if="spec.samples.slg_mode === 'standard'">
      <ZenField label="SLG scale" stack><ZenNumber v-model="spec.samples.slg_scale" :min="0" :step="0.5" :precision="1" /></ZenField>
      <ZenField label="SLG range" stack tip="Fraction of the steps it runs over">
        <div class="inline">
          <ZenNumber v-model="spec.samples.slg_start" :min="0" :max="1" :step="0.01" :precision="2" />
          <ZenNumber v-model="spec.samples.slg_end" :min="0" :max="1" :step="0.01" :precision="2" />
        </div>
      </ZenField>
    </template>
    <ZenField label="Negative" stack class="wide-f"><ZenInput v-model="spec.samples.negative" /></ZenField>
  </div>
</section>
</template>

<style scoped src="./form.css"></style>
<style scoped>
.rewrite {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin: 6px 0 8px;
}
.rewrite .rw-model {
  min-width: 220px;
  max-width: 340px;
}
.sample-head {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.sample-head {
  min-height: 32px;
}
.sample-head > :first-child {
  width: 240px;
}
.set-pick {
  position: relative;
}
.set-dot {
  position: absolute;
  top: 50%;
  right: 30px;
  width: 7px;
  height: 7px;
  margin-top: -3.5px;
  border-radius: var(--r-sm);
  background: var(--zen-warn);
  pointer-events: none;
}
.suggest {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: var(--r-lg);
  font-size: 12px;
  line-height: 1.45;
  background: color-mix(in srgb, var(--zen-accent) 9%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 30%, transparent);
}
.suggest > i {
  font-size: 18px;
  color: var(--zen-accent);
}
.suggest > span:not(.muted) {
  flex: 1;
}
/* the turbo LoRA: as a row of Other LoRAs (LoraUses.vue) */
.turbo-add {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.turbo {
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.turbo-pick {
  display: flex;
  align-items: center;
  gap: 10px;
}
.turbo-pick > :first-child {
  flex: 1;
  min-width: 0;
}
.strength {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 170px;
  flex: none;
}
.remove {
  flex: none;
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  padding: 2px;
  font-size: 16px;
}
.remove:hover {
  color: var(--zen-text);
}
.hint {
  margin: 0;
  line-height: 1.4;
}
.hint .link {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  color: var(--zen-accent);
  cursor: pointer;
}
.add-lora {
  align-self: flex-start;
  width: fit-content;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 5px 12px;
  border: 1px dashed var(--zen-border);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.add-lora:hover {
  color: var(--zen-text);
  border-color: color-mix(in srgb, var(--zen-accent) 50%, var(--zen-border));
}
</style>
