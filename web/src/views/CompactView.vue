<script setup lang="ts">
// Compact a LoRA or LoKr to the size you want. Every layer's change is split into directions,
// all of the file's are ranked together by how much of the change each carries per parameter,
// and the copy keeps the top of that ranking: rank goes where the change is. Weighed against
// the base model, a change beside a large weight counts for less (engine/zentrainer/loras/compact.py).
import { ZenButton, ZenCombo, ZenEmpty, ZenInput, ZenNumber, ZenToggleGroup } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { inputChoices } from '@/api/comfy'
import { EngineError, engine, type CompactAnalysis, type CompactRequest } from '@/api/engine'
import { fmtBytes } from '@/stores/downloads'

const route = useRoute()
const loras = ref<string[]>([])
const bases = ref<{ value: string; label: string }[]>([])
const name = ref(typeof route.query.add === 'string' ? route.query.add : '')
const base = ref('') // "folder:name", or none
const format = ref<string>('same')
const mode = ref<string>('keep')
const keep = ref(99)
const sizeMB = ref(50)
const maxRank = ref(0)
const dtype = ref<string>('bf16')
const outName = ref('')

const analysis = ref<CompactAnalysis | null>(null)
const measuring = ref(false)
const error = ref<string | null>(null)
const saving = ref(false)
const saveError = ref<string | null>(null)
const result = ref<{ lora_name: string; bytes: number; mean_rank: number; kept: number; layers_kept: number; layers: number } | null>(null)

const BYTES: Record<string, number> = { bf16: 2, fp16: 2, fp32: 4 }
const bytesPer = computed(() => BYTES[dtype.value] ?? 2)

onMounted(async () => {
  loras.value = await inputChoices('LoraLoader', 'lora_name').catch(() => [])
  const files = await engine.tensorFiles().catch(() => ({}) as Awaited<ReturnType<typeof engine.tensorFiles>>)
  bases.value = [
    ...(files.diffusion_models ?? []).map((n) => ({ value: `diffusion_models:${n}`, label: n })),
    ...(files.checkpoints ?? []).map((n) => ({ value: `checkpoints:${n}`, label: `${n} (checkpoint)` })),
  ]
})
const loraItems = computed(() => loras.value.map((n) => ({ value: n, label: n })))
const baseItems = computed(() => [{ value: '', label: 'None: measure the change alone' }, ...bases.value])

const request = (): CompactRequest => {
  const [folder, ...rest] = base.value.split(':')
  return {
    name: name.value,
    format: format.value === 'lora' ? 'lora' : 'same',
    base: base.value ? { folder: folder as 'diffusion_models' | 'checkpoints', name: rest.join(':') } : null,
    max_rank: maxRank.value || 0,
    target: mode.value === 'keep' ? { keep: keep.value / 100 } : { params: Math.round((sizeMB.value * 1e6) / bytesPer.value) },
  }
}

let timer: ReturnType<typeof setTimeout> | undefined
let precisionFor = '' // the file whose precision the toggle was last set from
let seq = 0
watch(
  [name, base, format, mode, keep, sizeMB, maxRank, dtype],
  ([n], old) => {
    clearTimeout(timer)
    result.value = null
    if (!n) {
      analysis.value = null
      return
    }
    if (!old || n !== old[0]) outName.value = `${n.split('/').pop()!.replace(/\.safetensors$/i, '')}_compact`
    timer = setTimeout(async () => {
      const mine = ++seq
      measuring.value = true
      error.value = null
      try {
        const a = await engine.analyzeCompact(request())
        if (mine === seq) {
          // a new file starts at its own precision, so sizes compare like for like
          if (a.original.precision && precisionFor !== n) dtype.value = a.original.precision
          precisionFor = n
          analysis.value = a
        }
      } catch (e) {
        if (mine === seq) error.value = e instanceof EngineError ? e.message : String(e)
      } finally {
        if (mine === seq) measuring.value = false
      }
    }, 250)
  },
  { immediate: true },
)

const dtypeOptions = computed(() =>
  ['bf16', 'fp16', 'fp32'].map((d) => ({ value: d, label: d === analysis.value?.original.precision ? `${d} (original)` : d })),
)
/** what saving in another precision than the file's does to its size */
const precisionNote = computed(() => {
  const from = analysis.value?.original.precision
  if (!from || from === dtype.value) return ''
  const a = BYTES[from] ?? 2
  const b = BYTES[dtype.value] ?? 2
  const what = b < a ? 'half the size per number' : b > a ? 'twice the size per number' : 'the same size per number'
  return `The file is ${from}; ${dtype.value} is ${what}, and the sizes above are in ${dtype.value}.`
})
const hasLokr = computed(() => analysis.value?.kind.some((k) => k !== 'lora') ?? false)
const planBytes = computed(() => (analysis.value ? analysis.value.plan.params * bytesPer.value : 0))

// ---- the levels on offer: each share kept, with the size it comes to ------------------------------
const LEVEL_NOTES: Record<number, string> = { 0.9: 'weaker', 0.95: 'close', 0.99: 'looks the same', 1: 'everything' }
const custom = ref(false) // a target typed in, instead of one of the levels
const levels = computed(() => {
  const a = analysis.value
  if (!a) return []
  return a.levels.map((l) => {
    const bytes = l.params * bytesPer.value
    return { ...l, bytes, ratio: bytes / (a.original.bytes || 1), note: LEVEL_NOTES[l.keep] ?? '', on: !custom.value && mode.value === 'keep' && Math.abs(keep.value / 100 - l.keep) < 1e-6 }
  })
})
function pickLevel(share: number) {
  custom.value = false
  mode.value = 'keep'
  keep.value = +(share * 100).toFixed(1)
}
const ratio = computed(() => planBytes.value / (analysis.value?.original.bytes || 1))
const pct = (r: number) => (r > 0 && r < 0.01 ? '<1%' : `${Math.round(r * 100)}%`)
/** a size against the original, as a bar's width (a bigger file fills it) */
const barWidth = (r: number) => `${Math.max(1, Math.min(100, r * 100))}%`

// ---- ranks by layer --------------------------------------------------------------------------------
const showLayers = ref(false)
const bars = computed(() => {
  const a = analysis.value
  if (!a) return []
  return a.plan.ranks.map((k, i) => ({ k, full: a.full_ranks[i] ?? 1, name: a.names[i] ?? '' }))
})

async function save() {
  saving.value = true
  saveError.value = null
  result.value = null
  try {
    result.value = await engine.compactLora({ ...request(), out_name: outName.value.trim(), dtype: dtype.value as 'bf16' | 'fp16' | 'fp32' })
  } catch (e) {
    saveError.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="page compact">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/tools" class="crumb">Tools</RouterLink> Compact a LoRA</h1>
        <p class="lede">The smallest file that keeps what matters: rank goes to the layers that change most.</p>
      </div>
    </div>

    <section class="card col">
      <div class="grid-2">
        <label class="f"><span class="small muted">LoRA or LoKr</span><ZenCombo v-model="name" :items="loraItems" placeholder="Pick a LoRA…" :item-height="30" /></label>
        <label class="f">
          <span class="small muted">Base model (optional)</span>
          <ZenCombo v-model="base" :items="baseItems" placeholder="None: measure the change alone" :item-height="30" />
        </label>
      </div>
      <span v-if="base" class="small muted">Each layer's change is measured against that layer's own weight in the model: a change beside a large weight counts for less. Only the weights' sizes are read.</span>
      <span v-if="analysis?.base_matched != null" class="small" :class="analysis.base_matched ? 'muted' : 'err'">
        {{ analysis.base_matched }} of {{ analysis.names.length }} layers found in the base model{{ analysis.base_matched ? '' : ': is it the model this LoRA was trained on?' }}
      </span>
    </section>

    <div v-if="error" class="card err small">{{ error }}</div>
    <ZenEmpty v-else-if="!name" icon="mdi mdi-arrow-collapse-all" title="Pick a LoRA">Every layer is measured once; after that, moving the target is instant.</ZenEmpty>
    <section v-else-if="!analysis" class="card small muted"><i class="mdi mdi-loading mdi-spin" /> Measuring every layer… a large LoKr takes a little while; moving the target afterwards is instant.</section>

    <section v-if="analysis && !error" class="card col result">
      <!-- the levels: pick one, or type a target -->
      <div class="levels">
        <div class="lv orig">
          <span class="lname">Original <span v-if="analysis.original.precision" class="pill mono">{{ analysis.original.precision }}</span></span>
          <span class="track"><span class="fill" style="width: 100%" /></span>
          <span class="lsize mono">{{ fmtBytes(analysis.original.bytes) }}</span>
        </div>
        <button v-for="l in levels" :key="l.keep" class="lv" :class="{ on: l.on, over: l.ratio > 1 }" @click="pickLevel(l.keep)">
          <span class="lname">
            <i :class="l.on ? 'mdi mdi-radiobox-marked' : 'mdi mdi-radiobox-blank'" />
            <b>{{ +(l.keep * 100).toFixed(1) }}%</b>
            <span class="muted">{{ l.note }}</span>
          </span>
          <span class="track"><span class="fill" :style="{ width: barWidth(l.ratio) }" /></span>
          <span class="lsize mono">{{ fmtBytes(l.bytes) }} <span class="muted">{{ pct(l.ratio) }}</span></span>
        </button>
        <div class="lv" :class="{ on: custom, over: custom && ratio > 1 }" @click="custom = true">
          <span class="lname">
            <i :class="custom ? 'mdi mdi-radiobox-marked' : 'mdi mdi-radiobox-blank'" />
            <b>Custom</b>
          </span>
          <span class="ctl">
            <ZenToggleGroup v-model="mode" :options="[{ value: 'keep', label: 'Keep %' }, { value: 'size', label: 'Size MB' }]" size="sm" @update:model-value="custom = true" />
            <ZenNumber v-if="mode === 'keep'" v-model="keep" :min="50" :max="100" :step="0.5" :precision="1" class="num" @update:model-value="custom = true" />
            <ZenNumber v-else v-model="sizeMB" :min="0.1" :max="100000" :step="5" :precision="1" class="num" @update:model-value="custom = true" />
          </span>
          <span v-if="custom" class="lsize mono">{{ fmtBytes(planBytes) }} <span class="muted">{{ pct(ratio) }}</span></span>
        </div>
      </div>

      <div class="facts">
        <span><b>{{ (analysis.plan.kept * 100).toFixed(2) }}%</b> of the change kept</span>
        <span><b>{{ analysis.plan.layers_kept }}/{{ analysis.plan.layers }}</b> layers</span>
        <span>rank <b>{{ analysis.plan.mean_rank.toFixed(1) }}</b> on average, up to {{ analysis.plan.max_rank }}</span>
        <i v-if="measuring" class="mdi mdi-loading mdi-spin" />
      </div>

      <!-- options -->
      <div class="controls">
        <div class="ctl">
          <span class="small muted">Precision</span>
          <ZenToggleGroup v-model="dtype" :options="dtypeOptions" size="sm" />
        </div>
        <div class="ctl">
          <span class="small muted">Rank cap</span>
          <ZenNumber v-model="maxRank" :min="0" :max="4096" :step="4" :precision="0" class="num" />
        </div>
        <div v-if="hasLokr" class="ctl">
          <ZenToggleGroup v-model="format" :options="[{ value: 'same', label: 'Keep as LoKr' }, { value: 'lora', label: 'Make a LoRA' }]" size="sm" />
        </div>
      </div>
      <span class="small muted">
        The share is how much of the LoRA's change the copy keeps. Rank cap 0 is no cap.
        <template v-if="precisionNote"> {{ precisionNote }}</template>
        <template v-if="hasLokr && format === 'lora'"> A LoKr's Kronecker structure is very efficient: as a LoRA the same share usually takes many times the size.</template>
        <template v-if="analysis.unsupported.length"> {{ analysis.unsupported.length }} layers (LoHa or Tucker LoKr) are copied as they are.</template>
      </span>

      <!-- per layer, on request -->
      <button class="fold small" @click="showLayers = !showLayers">
        <i :class="showLayers ? 'mdi mdi-chevron-down' : 'mdi mdi-chevron-right'" /> Rank by layer
      </button>
      <div v-if="showLayers" class="bars">
        <span v-for="b in bars" :key="b.name" class="bar" :title="`${b.name}: rank ${b.k} of ${b.full}`">
          <span :style="{ height: `${(100 * b.k) / Math.max(1, b.full)}%` }" />
        </span>
      </div>

      <!-- saving -->
      <div class="savebox">
        <div class="saverow">
          <span class="eyebrow">Save</span>
          <div class="grow"><ZenInput v-model="outName" placeholder="e.g. my-lora_compact" size="sm" /></div>
          <ZenButton variant="primary" size="sm" icon="mdi mdi-arrow-collapse-all" :disabled="!outName.trim() || saving" @click="save">{{ saving ? 'Compacting…' : 'Compact' }}</ZenButton>
          <RouterLink :to="`/apps/text-to-image?lora=${encodeURIComponent(name)}`"><ZenButton size="sm" icon="mdi mdi-creation-outline">Try the original</ZenButton></RouterLink>
        </div>
        <div class="small"><span class="muted">Saves as</span> <span class="mono out">LoRAs / compact/{{ outName.trim() || '…' }}.safetensors</span></div>
        <div v-if="saveError" class="err small">{{ saveError }}</div>
        <div v-if="result" class="done">
          <i class="mdi mdi-check-circle" />
          <div class="grow">
            <div class="mono out">{{ result.lora_name }}</div>
            <div class="small muted">{{ fmtBytes(result.bytes) }} · average rank {{ result.mean_rank.toFixed(1) }} · {{ (result.kept * 100).toFixed(2) }}% kept</div>
          </div>
          <RouterLink :to="`/apps/text-to-image?lora=${encodeURIComponent(result.lora_name)}`"><ZenButton size="sm" icon="mdi mdi-creation-outline">Try it</ZenButton></RouterLink>
          <RouterLink :to="{ path: '/tools/inspect', query: { folder: 'loras', name: result.lora_name } }"><ZenButton size="sm" icon="mdi mdi-file-tree-outline">Inspect it</ZenButton></RouterLink>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.compact {
  max-width: 960px;
}
.col {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}
.grow {
  flex: 1;
  min-width: 0;
}
.f {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.num {
  width: 104px;
}
.err {
  color: var(--zen-danger);
}

/* ---- the levels ---------------------------------------------------------------------------- */
.levels {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  overflow: hidden;
}
.lv {
  display: grid;
  grid-template-columns: minmax(170px, auto) minmax(80px, 1fr) 130px;
  gap: 16px;
  align-items: center;
  min-height: 38px;
  padding: 4px 14px;
  border: none;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
  background: none;
  color: var(--zen-text);
  font: inherit;
  font-size: 13px;
  text-align: left;
  cursor: pointer;
}
.lv:last-child {
  border-bottom: none;
}
.lv:hover {
  background: color-mix(in srgb, var(--zen-surface-2) 60%, transparent);
}
.lv.on {
  background: color-mix(in srgb, var(--zen-accent) 8%, transparent);
}
.lv.orig {
  cursor: default;
  color: var(--zen-muted);
  background: none;
}
.lname {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}
.lname i {
  font-size: 16px;
  color: var(--zen-muted);
}
.lv.on .lname i {
  color: var(--zen-accent);
}
.lname b {
  font-variant-numeric: tabular-nums;
  min-width: 44px;
}
.lname .muted {
  font-size: 12px;
}
.lv.orig .lname {
  padding-left: 24px;
}
.pill {
  font-size: 10.5px;
  padding: 0 6px;
  border-radius: 4px;
  line-height: 18px;
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
.track {
  height: 6px;
  border-radius: 99px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.fill {
  display: block;
  height: 100%;
  border-radius: 99px;
  background: color-mix(in srgb, var(--zen-text) 30%, transparent);
}
.lv.on .fill {
  background: var(--zen-accent);
}
.lv.over .fill {
  background: var(--zen-warn);
}
.lv.orig .fill {
  background: color-mix(in srgb, var(--zen-text) 18%, transparent);
}
.lsize {
  text-align: right;
  font-size: 12px;
  white-space: nowrap;
}
.lv .ctl {
  grid-column: span 1;
}
.facts {
  display: flex;
  gap: 6px 18px;
  font-size: 12px;
  color: var(--zen-muted);
  flex-wrap: wrap;
  align-items: center;
}
.facts b {
  color: var(--zen-text);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
@media (max-width: 640px) {
  .lv {
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 6px 12px;
    padding: 8px 12px;
  }
  .track {
    grid-column: 1 / -1;
    grid-row: 2;
  }
  .lv .ctl {
    grid-column: 1 / -1;
    grid-row: 2;
  }
}

/* ---- target and options ---------------------------------------------------------------------- */
.controls {
  display: flex;
  gap: 10px 24px;
  flex-wrap: wrap;
  align-items: center;
}
.ctl {
  display: flex;
  gap: 8px;
  align-items: center;
}
.fold {
  align-self: flex-start;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 0;
  cursor: pointer;
}
.fold:hover {
  color: var(--zen-text);
}
.bars {
  display: flex;
  align-items: stretch;
  gap: 1px;
  height: 40px;
}
.bar {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: flex-end;
  background: color-mix(in srgb, var(--zen-text) 7%, transparent);
  border-radius: 1px;
}
.bar > span {
  width: 100%;
  background: color-mix(in srgb, var(--zen-accent) 70%, transparent);
  border-radius: 1px;
}
.bar:hover > span {
  background: var(--zen-accent);
}

/* ---- saving ---------------------------------------------------------------------------------- */
.savebox {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid var(--zen-border);
}
.saverow {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}
.saverow .grow {
  min-width: 200px;
}
.out {
  overflow-wrap: anywhere;
}
.done {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  flex-wrap: wrap;
  padding: 10px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-ok) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-ok) 30%, transparent);
}
.done > i {
  color: var(--zen-ok);
  font-size: 20px;
}
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
</style>
