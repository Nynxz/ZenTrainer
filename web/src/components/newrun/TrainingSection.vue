<script setup lang="ts">
// 3 Training: steps, optimizer, LR, rank, schedule and where the LoRAs go; under Advanced the
// optimizer's finer settings, the network, and the objective (with Focus for flow models).
import { ZenCombo, ZenField, ZenInput, ZenNumber, ZenSwitch, ZenTip } from '@nynxz/zenkit-ui'
import { computed, reactive, ref, watch } from 'vue'
import { engine, type OptimizerInfo } from '@/api/engine'
import FocusField from '@/components/FocusField.vue'
import { fmtBytes } from '@/stores/downloads'
import { setSetting } from '@/stores/settings'
import LoraUses from './LoraUses.vue'
import { items, useRunFormContext } from './form'

const { spec, loraPlaces, choices, familyId, optimizers, dataset, datasets, opt, epochs, noise, resuming, preserveOn, stage0 } = useRunFormContext()
// validation on a held-out dataset: any but the one training
const heldOutItems = computed(() => [
  { value: '', label: 'From the training set' },
  ...datasets.value.filter((d) => d.folder !== spec.dataset.folder).map((d) => ({ value: d.folder, label: d.folder })),
])

// Preserve classes: turning it on fills in what the dataset's captions call the subject and its
// counterpart (a woman's LoRA bleeds into men too), or woman, man, person
const classes = ref<{ captions: number; words: { word: string; count: number }[]; suggest: string[] } | null>(null)
// preserveOn (form.ts) is its own on/off, set by the switch and by every flow that brings a spec in
// (form.ts syncPreserve); emptying the words field to retype leaves it on. Turned on before the
// captions' words load, it holds a stand-in that they replace when they come, unless it's been
// edited since (fallbackShown)
const FALLBACK = 'woman, man, person'
const fallbackShown = ref(false)
function setPreserve(on: boolean) {
  preserveOn.value = on
  const words = classes.value?.suggest.join(', ')
  fallbackShown.value = on && !words
  spec.objective.preserve = on ? words || FALLBACK : ''
}
function editWords(v: string) {
  fallbackShown.value = false
  spec.objective.preserve = v
}
// a slider has no trigger to swap and the switch hides with it: turning one on turns this off
watch(() => spec.slider?.kind, (kind) => { if (kind && preserveOn.value) setPreserve(false) })
let classTicket = 0
watch(() => spec.dataset.folder, async (folder) => {
  const ticket = ++classTicket
  classes.value = null
  if (!folder) return
  try {
    const got = await engine.datasetClassWords(folder)
    if (ticket !== classTicket) return
    classes.value = got
    // switched on before the captions' words came: theirs replace the stand-in, never words typed
    if (preserveOn.value && fallbackShown.value && got.suggest.length) {
      fallbackShown.value = false
      spec.objective.preserve = got.suggest.join(', ')
    }
  } catch {
    /* no suggestion: the field still takes words by hand */
  }
}, { immediate: true })

const advanced = reactive({ training: false })
// LoRA and LoKr: LoHa and OFT aren't ready (LoHa can't train on an int8 base). A run that already
// uses one keeps it in the list, so its form doesn't show an empty field.
// the optimizers it can train with now (and the run's own, if it isn't); the rest are a count
const optimizerItems = computed(() =>
  optimizers.value
    .filter((o) => o.available || o.id === spec.optim.optimizer)
    .map((o) => ({ value: o.id, label: o.label, keywords: o.note, o })),
)
const notInstalled = computed(() => optimizers.value.filter((o) => !o.available && o.id !== spec.optim.optimizer))
const algorithms = computed(() => {
  const ready = ['LoRA', 'LoKr']
  return ready.includes(spec.network.algorithm) ? ready : [...ready, spec.network.algorithm]
})
// ---- where checkpoints go: one of ComfyUI's loras folders ---------------------------------
const loraDir = computed({
  get: () => spec.output.lora_dir ?? loraPlaces.value.find((p) => p.default)?.path ?? '',
  set: (v: string) => {
    const place = loraPlaces.value.find((p) => p.path === v)
    spec.output.lora_dir = !place || place.default ? null : v
    setSetting('loraFolder', spec.output.lora_dir ?? '')
  },
})
const loraPlaceItems = computed(() =>
  loraPlaces.value.map((p) => ({ value: p.path, label: p.path, keywords: p.path, free: `${fmtBytes(p.free)} free` })),
)
const loraPlaceHint = computed(() => {
  const p = loraPlaces.value.find((x) => x.path === loraDir.value)
  if (!p) return 'One of ComfyUI’s LoRA folders'
  return p.default ? 'Inside this install. Your model library’s loras folder keeps them with the rest, visible to your other ComfyUI' : `In a zentrainer/ subfolder · ${fmtBytes(p.free)} free`
})
// ---- bound helpers ---------------------------------------------------------------------------
// which saves along the way stay on disk (engine harness/keep.py); the best-by rules put samples
// or validation on every save, so each has a score
// likeness is only scored for a dataset with Check faces on: say so on those choices
const noFaces = computed(() => !!dataset.value && !dataset.value.faces)
const keepItems = computed(() => {
  const later = noFaces.value ? ' (needs Check faces)' : ''
  return [
    { value: 'all', label: 'All' },
    { value: 'latest', label: 'Latest' },
    { value: 'likeness', label: `Best likeness${later}` },
    { value: 'val_loss', label: 'Lowest validation loss' },
    { value: 'likeness_new_best', label: `Each new best likeness${later}` },
    { value: 'val_loss_new_best', label: 'Each new lowest validation loss' },
  ]
})
const keepNeedsFaces = computed(() => noFaces.value && spec.output.keep.startsWith('likeness'))
const keepOnTurbo = computed(() => spec.output.keep.startsWith('likeness') && !!spec.samples.turbo_lora && (spec.samples.turbo_strength ?? 0.75) > 0)
const keepTip =
  'Which saves stay on disk; the rest are deleted as the run goes. The final LoRA, the newest save and saves made by hand or at a stop always stay. ' +
  "Best likeness: its samples' faces against the dataset's (a character, with faces on for its dataset). " +
  "Lowest validation loss is measured on the training pictures, so it mostly keeps the later saves. " +
  'Each new best: every save that beat all the saves before it stays, so a run keeps each of its record-setters (no count)'
const alphaAuto = computed({
  get: () => spec.network.alpha == null,
  set: (v: boolean) => (spec.network.alpha = v ? null : spec.network.rank),
})
const alphaValue = computed({
  get: () => spec.network.alpha ?? spec.network.rank,
  set: (v: number) => (spec.network.alpha = v),
})
// ---- AI Toolkit's timestep types as ours: its linear and sigmoid draw σ unshifted (shift 1), its
// shift keeps the model's own schedule (shift 0), and weighted is linear with its Flex.1 weights
type AitkType = '' | 'linear' | 'sigmoid' | 'shift' | 'weighted'
const aitk = computed<AitkType>({
  get: () => {
    const o = spec.objective
    if (o.timesteps === 'scheduler') return ''
    if (o.weighting) return o.weighting === 'flex' && o.timesteps === 'uniform' && o.shift === 1 ? 'weighted' : ''
    if (o.timesteps === 'logit_normal') return o.shift === 1 && o.logit_mean === 0 && o.logit_std === 1 && !o.uniform_mix ? 'sigmoid' : ''
    return o.shift === 1 ? 'linear' : o.shift === 0 ? 'shift' : ''
  },
  set: (v: AitkType) => {
    if (!v) return
    const o = spec.objective
    o.timesteps = v === 'sigmoid' ? 'logit_normal' : 'uniform'
    o.shift = v === 'shift' ? 0 : 1
    o.weighting = v === 'weighted' ? 'flex' : ''
    if (v === 'sigmoid') Object.assign(o, { logit_mean: 0, logit_std: 1, uniform_mix: 0 })
  },
})
// ComfyUI's schedulers, as the picked model's profile lists them
const schedulerNames = computed(() => Object.keys(noise.value.schedules ?? {}).length
  ? Object.keys(noise.value.schedules!)
  : ['simple', 'sgm_uniform', 'karras', 'exponential', 'ddim_uniform', 'beta', 'normal', 'linear_quadratic', 'kl_optimal'])
const AITK_ITEMS = [
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'linear', label: 'Linear' },
  { value: 'shift', label: 'Shift' },
  { value: 'weighted', label: 'Weighted' },
  { value: '', label: 'Custom' },
]
// a preset, or "custom": a list of patterns over layer names (an imported or hand-written spec)
const targets = computed({
  get: () => (typeof spec.network.targets === 'string' ? spec.network.targets : 'custom'),
  set: (v: string) => {
    if (v !== 'custom') spec.network.targets = v as typeof spec.network.targets
    else if (typeof spec.network.targets === 'string') spec.network.targets = []
  },
})
const patterns = computed({
  get: () => (Array.isArray(spec.network.targets) ? spec.network.targets.join('\n') : ''),
  set: (v: string) => (spec.network.targets = v.split('\n').map((p) => p.trim()).filter(Boolean)),
})
</script>

<template>
<section class="card sec">
  <h2><span class="num">3</span> Training</h2>
  <div class="grid-2">
    <ZenField label="Steps" stack :hint="epochs != null ? `≈ ${epochs.toFixed(1)} epochs of ${dataset?.steps_per_epoch} images` : ''">
      <ZenNumber v-model="spec.optim.steps" :min="1" :step="100" :precision="0" />
    </ZenField>
    <ZenField label="Optimizer" stack :tip="opt ? `${opt.note} Suggested LR ${opt.lr_hint}.`.trim() : ''">
      <ZenCombo
        v-model="spec.optim.optimizer"
        :items="optimizerItems"
        :sorted="false"
        :pick-guard="(v: string | number) => !optimizers.find((o) => o.id === v)?.available"
        :menu-width="360"
        :item-height="34"
      >
        <!-- what each one is for, while choosing: an ⓘ per option (clicking it doesn't pick) -->
        <template #option="{ item }">
          <div class="opt-row" :class="{ off: !(item.o as OptimizerInfo).available }">
            <span class="opt-name">{{ item.label }}</span>
            <span v-if="(item.o as OptimizerInfo).note" class="opt-tip" @click.stop @mousedown.stop @pointerdown.stop>
              <ZenTip :text="(item.o as OptimizerInfo).note" :label="`About ${item.label}`" />
            </span>
            <span class="spacer" />
            <span class="muted mono small">{{ (item.o as OptimizerInfo).available ? `lr ${(item.o as OptimizerInfo).lr_hint}` : 'not installed' }}</span>
          </div>
        </template>
        <template v-if="notInstalled.length" #footer>
          <div class="opt-more small muted">
            {{ notInstalled.length }} more with optional packages
            <ZenTip :text="`${notInstalled.map((o) => o.label).join(', ')}: pip install -r requirements-optional.txt in ComfyUI's Python, then restart`" label="Which, and how" />
          </div>
        </template>
      </ZenCombo>
    </ZenField>
    <ZenField label="Learning rate" stack>
      <ZenNumber v-model="spec.optim.learning_rate" :min="0" :step="0.00001" :precision="6" />
    </ZenField>
    <!-- what trains and its size together: the rank means a different thing for each -->
    <ZenField label="Adapter" stack :tip="resuming ? 'Kept from the run it resumes: its saved state is this adapter at this rank' : 'LoRA, or LoKr (smaller files, often as good). Rank: 8–16 for a character, 16–32 for a style as a LoRA; 4–8 as a LoKr'">
      <div class="adapter-row" :class="{ locked: resuming }">
        <div class="view-switch text" role="radiogroup" aria-label="Adapter">
          <button v-for="a in algorithms" :key="a" role="radio" :aria-checked="spec.network.algorithm === a" :class="{ on: spec.network.algorithm === a }" @click="resuming || (spec.network.algorithm = a)">{{ a }}</button>
        </div>
        <label class="rank"><span class="muted small">Rank</span><ZenNumber v-model="spec.network.rank" :min="1" :max="256" :step="1" :precision="0" :disabled="resuming" /></label>
      </div>
    </ZenField>
    <ZenField label="Schedule" stack>
      <ZenCombo v-model="spec.optim.lr_scheduler" :items="items(['constant', 'linear', 'cosine'])" :searchable="false" :sorted="false" />
    </ZenField>
    <ZenField label="Save every" stack tip="0 = only the final LoRA">
      <ZenNumber v-model="spec.output.save_every" :min="0" :step="50" :precision="0" />
    </ZenField>
    <ZenField v-if="spec.output.save_every" label="Keep checkpoints" stack :tip="keepTip">
      <div class="sw-row">
        <ZenCombo v-model="spec.output.keep" :items="keepItems" :sorted="false" :searchable="false" />
        <ZenNumber v-if="spec.output.keep !== 'all' && !spec.output.keep.endsWith('_new_best')" v-model="spec.output.keep_count" :min="1" :max="99" :step="1" :precision="0" />
      </div>
      <p v-if="keepNeedsFaces" class="warn-note">
        {{ dataset!.folder }} has Check faces off, so its samples get no likeness to rank by: turn it on on the dataset's page (Settings), or keep another way.
      </p>
      <p v-else-if="keepOnTurbo" class="warn-note">
        Samples render with a LoRA under them, so likeness is scored on those renders rather than on your LoRA alone. Set its strength to 0 in Samples to score without it.
      </p>
    </ZenField>
    <ZenField label="Save LoRAs to" stack class="span-2" :tip="loraPlaceHint">
      <ZenCombo v-model="loraDir" :items="loraPlaceItems" :sorted="false" :searchable="false" :item-height="40" :menu-width="520">
        <template #option="{ item }">
          <div class="place-opt">
            <span class="mono ellipsis">{{ item.label }}</span>
            <span class="spacer" />
            <span class="muted small mono">{{ item.free }}</span>
          </div>
        </template>
      </ZenCombo>
    </ZenField>
    <ZenField label="Resume states" stack tip="Optimizer state for resuming a stopped run; often bigger than the LoRA" :hint="spec.output.keep_states === 'none' ? 'A stopped or crashed run can’t be continued, only started over' : undefined">
      <ZenCombo
        v-model="spec.output.keep_states"
        :items="[{ value: 'latest', label: 'Latest only' }, { value: 'all', label: 'Every checkpoint' }, { value: 'none', label: 'None (no resuming)' }]"
        :sorted="false"
        :searchable="false"
      />
    </ZenField>
    <ZenField label="Save as" stack tip="What the LoRA files are written in. bf16 is half of fp32 and loses nothing you'd see; training still runs in the LoRA precision, and resuming uses the full weights">
      <ZenCombo
        v-model="spec.output.save_dtype"
        :items="[{ value: 'bf16', label: 'bf16 (recommended)' }, { value: 'fp16', label: 'fp16' }, { value: 'fp32', label: 'fp32 (twice the size)' }]"
        :sorted="false"
        :searchable="false"
      />
    </ZenField>
  </div>
  <button class="more" @click="advanced.training = !advanced.training">
    <i :class="advanced.training ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /> Advanced training
  </button>
  <div v-if="advanced.training" class="adv">
    <div class="subgroup">
      <div class="sub-title">Optimizer</div>
      <div class="grid-2">
        <ZenField label="Warmup steps" stack><ZenNumber v-model="spec.optim.warmup_steps" :min="0" :step="10" :precision="0" /></ZenField>
        <ZenField label="Batch size" stack :tip="stage0 ? `Pictures per step at ${spec.dataset.resolution} px` : ''"><ZenNumber v-model="spec.optim.batch_size" :min="1" :step="1" :precision="0" /></ZenField>
        <ZenField v-if="stage0" :label="`Batch size at ${stage0.resolution}`" stack tip="Pictures per step during the lower-resolution stage; lower-resolution steps use less memory, so a bigger batch fits there">
          <ZenNumber :model-value="stage0.batch_size ?? spec.optim.batch_size" :min="1" :step="1" :precision="0" @update:model-value="stage0.batch_size = $event" />
        </ZenField>
        <ZenField label="Grad accumulation" stack><ZenNumber v-model="spec.optim.grad_accumulation_steps" :min="1" :step="1" :precision="0" /></ZenField>
        <ZenField label="LoRA+" stack tip="The LoRA's up matrices learn this many times faster than the down ones: trains faster at the same stability. 4 is a safe start, the paper uses 16. Off for LoKr">
          <div class="sw-row">
            <ZenSwitch :model-value="(spec.optim.lora_plus ?? 0) > 1" @update:model-value="spec.optim.lora_plus = $event ? 4 : 0" />
            <ZenNumber v-if="(spec.optim.lora_plus ?? 0) > 1" :model-value="spec.optim.lora_plus ?? 4" :min="1.5" :max="32" :step="1" :precision="1" @update:model-value="spec.optim.lora_plus = $event" />
            <span v-else class="sw-state">Off</span>
          </div>
        </ZenField>
        <ZenField label="Weight averaging" stack tip="Checkpoints, samples and the saved LoRA use a running average of the recent steps (EMA) instead of whichever step the save landed on: steadier likeness, less wobble between checkpoints. 0.98 ≈ the last 50 steps">
          <div class="sw-row">
            <ZenSwitch :model-value="(spec.optim.ema ?? 0) > 0" @update:model-value="spec.optim.ema = $event ? 0.98 : 0" />
            <ZenNumber v-if="(spec.optim.ema ?? 0) > 0" :model-value="spec.optim.ema ?? 0.98" :min="0.5" :max="0.999" :step="0.005" :precision="3" @update:model-value="spec.optim.ema = $event" />
            <span v-else class="sw-state">Off</span>
          </div>
        </ZenField>
        <ZenField v-if="spec.optim.optimizer === 'Automagic3' || spec.optim.optimizer === 'Automagic4'" label="LR floor (%)" stack tip="Automagic never lowers its learning rate below this share of the start LR. v3 can talk itself down to almost nothing once the trend is learned (50% kept likeness climbing on MOSS); v4 settles on its own, so it rarely needs one. 0 = no floor">
          <ZenNumber :model-value="Math.round((spec.optim.lr_floor ?? 0) * 100)" :min="0" :max="100" :step="5" :precision="0" @update:model-value="spec.optim.lr_floor = $event / 100" />
        </ZenField>
        <ZenField label="Weight decay" stack><ZenNumber v-model="spec.optim.weight_decay" :min="0" :step="0.001" :precision="4" /></ZenField>
        <ZenField label="Grad clip" stack tip="Max grad norm; 0 = off"><ZenNumber v-model="spec.optim.max_grad_norm" :min="0" :step="0.1" :precision="2" /></ZenField>
      </div>
    </div>
    <div class="subgroup" :class="{ locked: resuming }">
      <div class="sub-title">Network</div>
      <div class="grid-2">
        <ZenField label="Layers" stack>
          <ZenCombo v-model="targets" :items="items(['attn', 'mlp', 'attn_mlp', 'all', 'custom'], { attn: 'Attention', mlp: 'MLP', attn_mlp: 'Attention + MLP', all: 'Every layer', custom: 'Custom' })" :searchable="false" :sorted="false" />
        </ZenField>
        <ZenField label="Alpha" stack :tip="alphaAuto ? 'Auto = rank (scale 1.0)' : 'Scale is alpha / rank'">
          <div class="sw-row">
            <ZenSwitch v-model="alphaAuto" />
            <ZenNumber v-if="!alphaAuto" v-model="alphaValue" :min="0.01" :step="1" :precision="2" />
            <span v-else class="sw-state">Auto (= rank)</span>
          </div>
        </ZenField>
        <ZenField label="LoRA precision" stack tip="fp32: small steps land. bf16 rounds most AdamW steps away; use it only with Automagic (stochastic rounding)">
          <ZenCombo v-model="spec.network.lora_dtype" :items="items(['fp32', 'bf16'], { fp32: 'fp32 (recommended)', bf16: 'bf16' })" :searchable="false" :sorted="false" />
        </ZenField>
        <ZenField label="Other LoRAs" stack class="full-row" tip="A LoRA to train more on, one to sit under the new LoRA, or one that's on only while training: pick the file, then how it's used">
          <LoraUses />
        </ZenField>
        <ZenField v-if="targets === 'custom'" label="Layer patterns" stack class="full-row" tip="One per line: a layer gets an adapter when its name matches any of them (regular expressions over names like diffusion_model.layers.30.proj.ffn_down). The Blocks and Skip blocks below still apply">
          <textarea v-model.lazy="patterns" class="patterns mono" rows="2" spellcheck="false" placeholder="proj\.ffn_(up_gate|down)$" />
          <p v-if="!patterns.trim() && !spec.network.train_bias_norm && !(spec.dataset.edit && spec.dataset.edit_method === 'concat' && spec.dataset.control_sets?.length)" class="warn-note">
            No patterns yet, so nothing would train: add the layers to train, or pick a preset above.
          </p>
        </ZenField>
        <ZenField label="Blocks" stack tip="Only these transformer blocks get adapters, e.g. 20-49: the backward stops at the first one, so steps are quicker, and the early blocks (layout, motion) stay untouched. Empty = every block">
          <ZenInput :model-value="spec.network.blocks ?? ''" placeholder="all" @update:model-value="spec.network.blocks = String($event).trim()" />
        </ZenField>
        <ZenField label="Skip blocks" stack tip="These blocks get no adapters, e.g. 22 or 22, 30-31 (AI Toolkit's ignore_if_contains). Skip-layer guidance skips a block in the uncond: left untrained, skipping it still means skipping the base's block">
          <ZenInput :model-value="spec.network.skip_blocks ?? ''" placeholder="none" @update:model-value="spec.network.skip_blocks = String($event).trim()" />
        </ZenField>
        <ZenField label="Train biases & norms" stack><div class="sw-row"><ZenSwitch v-model="spec.network.train_bias_norm" /><span class="sw-state">{{ spec.network.train_bias_norm ? 'On' : 'Off' }}</span></div></ZenField>
      </div>
    </div>
    <div class="subgroup">
      <div class="sub-title">Objective</div>
      <div class="grid-2">
        <ZenField label="Target" stack tip="Native: ε for SD, velocity for flow models">
          <ZenCombo v-model="spec.objective.target" :items="items(['native', 'x0'], { native: 'Model native', x0: 'Denoised (x0)' })" :searchable="false" :sorted="false" />
        </ZenField>
        <ZenField v-if="noise.flow" label="Timestep type" stack tip="AI Toolkit's names for the settings below. Sigmoid and Linear draw the noise level as is (shift 1), Shift follows the model's own schedule, Weighted is Linear with AI Toolkit's loss weights. Changing anything below makes it Custom">
          <ZenCombo v-model="aitk" :items="AITK_ITEMS" :searchable="false" :sorted="false" />
        </ZenField>
        <ZenField label="Timesteps" stack tip="Logit-normal (AI Toolkit's sigmoid) favours mid noise; uniform (linear) spends the same on every level">
          <ZenCombo v-model="spec.objective.timesteps" :items="items(['uniform', 'logit_normal', 'scheduler'], { uniform: 'Uniform (linear)', logit_normal: 'Logit-normal (sigmoid)', scheduler: 'A ComfyUI scheduler' })" :searchable="false" :sorted="false" @update:model-value="$event === 'scheduler' && (spec.objective.shift = 0)" />
        </ZenField>
        <ZenField v-if="spec.objective.timesteps === 'scheduler'" label="Scheduler / steps" stack tip="Trains on exactly the noise levels this scheduler samples at in this many steps, each as often: the LoRA learns where it'll be used. Few steps (4–8) for a LoRA meant for fast sampling. Uses the model's own shift">
          <div class="inline">
            <ZenCombo :model-value="spec.objective.timestep_scheduler ?? 'simple'" :items="items(schedulerNames)" :searchable="false" :sorted="false" @update:model-value="spec.objective.timestep_scheduler = String($event)" />
            <ZenNumber :model-value="spec.objective.timestep_scheduler_steps ?? 20" :min="1" :max="200" :step="1" :precision="0" @update:model-value="spec.objective.timestep_scheduler_steps = $event" />
          </div>
        </ZenField>
        <ZenField label="Timestep bias" stack tip="Lean the steps to one end, whatever the timesteps above. High noise teaches composition, layout and pose (AI Toolkit's content); low noise detail and texture (its style). Balanced = as drawn">
          <ZenCombo :model-value="spec.objective.bias ?? 'balanced'" :items="items(['balanced', 'high_noise', 'low_noise'], { balanced: 'Balanced', high_noise: 'High noise', low_noise: 'Low noise' })" :searchable="false" :sorted="false" @update:model-value="spec.objective.bias = $event as 'balanced' | 'high_noise' | 'low_noise'" />
        </ZenField>
        <ZenField v-if="(spec.objective.bias ?? 'balanced') !== 'balanced'" label="Bias strength" stack tip="How hard it leans: the draw's power. 3 is AI Toolkit's (half the steps in the eighth of the schedule at that end); 2 is gentler">
          <ZenNumber :model-value="spec.objective.bias_power ?? 3" :min="1" :max="10" :step="0.5" :precision="1" @update:model-value="spec.objective.bias_power = $event" />
        </ZenField>
        <ZenField label="Loss weighting" stack tip="Weights each step's loss by its noise level. Balanced: measured on this model as it trains, so every noise level pulls on the LoRA about equally (EDM2's idea); the timesteps still decide how often each is trained. AI Toolkit (flow models, its &quot;weighted&quot; type): a fixed curve measured on Flex.1, up to 1.5x at heavy noise, under half near clean">
          <ZenCombo :model-value="spec.objective.weighting ?? ''" :items="items(noise.flow ? ['', 'balance', 'flex'] : ['', 'balance'], { '': 'None', balance: 'Balanced (measured)', flex: 'AI Toolkit (Flex.1)' })" :searchable="false" :sorted="false" @update:model-value="spec.objective.weighting = $event as '' | 'flex' | 'balance'" />
        </ZenField>
        <ZenField label="Spread evenly" stack tip="Noise levels from an even sequence instead of random draws: at batch 1, a few steps already cover the whole schedule, and the loss curve is steadier. Same distribution">
          <div class="sw-row"><ZenSwitch :model-value="!!spec.objective.spread" @update:model-value="spec.objective.spread = $event" /><span class="sw-state">{{ spec.objective.spread ? 'On' : 'Off' }}</span></div>
        </ZenField>
        <ZenField v-if="spec.objective.timesteps === 'logit_normal'" label="Centre / spread" stack tip="Logit-normal's mean and standard deviation. Mean 0 centres on mid noise; positive leans to heavy noise (composition), negative to light noise (detail). MOSS pretrained at 1.09 / 0.95">
          <div class="inline">
            <ZenNumber v-model="spec.objective.logit_mean" :min="-3" :max="3" :step="0.1" :precision="2" />
            <ZenNumber v-model="spec.objective.logit_std" :min="0.1" :max="3" :step="0.1" :precision="2" />
          </div>
        </ZenField>
        <ZenField v-if="spec.objective.timesteps === 'logit_normal'" label="Uniform mix (%)" stack tip="Share of steps drawn uniformly instead: a little training at the extremes logit-normal rarely visits. 1% is plenty">
          <ZenNumber :model-value="Math.round((spec.objective.uniform_mix ?? 0) * 100)" :min="0" :max="50" :step="1" :precision="0" @update:model-value="spec.objective.uniform_mix = $event / 100" />
        </ZenField>
        <ZenField v-if="noise.known && noise.flow" label="Where the steps go" stack class="full-row" tip="Each step trains at one noise level. Left is full noise, where the LoRA learns composition (layout, pose, colour); right is nearly clean, where it learns detail (faces, texture). The higher the curve, the more steps there. The dashed line is the model's own schedule; the timesteps, bias and Lean above all move the curve. Leaning toward detail usually helps likeness">
          <FocusField :objective="spec.objective" :profile="noise" />
        </ZenField>
        <ZenField v-else-if="spec.objective.timesteps !== 'scheduler'" label="Timestep shift" stack tip="Flow models. 0 = the model's own (MOSS 5: composition, colour). 1–3 trains more on fine detail like faces">
          <ZenNumber v-model="spec.objective.shift" :min="0" :max="10" :step="0.5" :precision="1" />
        </ZenField>
        <ZenField label="Loss" stack><ZenCombo v-model="spec.objective.loss" :items="items(['mse', 'l1', 'huber', 'smooth_l1'])" :searchable="false" :sorted="false" /></ZenField>
        <ZenField v-if="!spec.slider?.kind" label="Preserve classes" stack tip="AI Toolkit's differential output preservation. Each step also trains its caption with a class word (woman, man…) in place of the trigger to give what the base model gives, so the LoRA changes only what the trigger asks for: “a man” stays a man instead of turning into your character. Needs a trigger word. On every 2nd step (the default) a run takes about 1.7x as long">
          <div class="sw-row"><ZenSwitch :model-value="preserveOn" @update:model-value="setPreserve" /><span class="sw-state">{{ preserveOn ? 'On' : 'Off' }}</span></div>
        </ZenField>
        <ZenField v-if="preserveOn" label="Class words" stack tip="Comma-separated: what the captions call your subject, and what it shouldn't leak into. Filled in from the dataset's captions when you turn it on">
          <ZenInput :model-value="spec.objective.preserve ?? ''" placeholder="woman, man, person" @update:model-value="editWords(String($event))" />
        </ZenField>
        <ZenField v-if="preserveOn" label="Preserve weight / every" stack tip="The preservation loss's weight beside the training loss (1 = as strong), and how often it runs: every Nth step. 2 (the default) held the class as well as every step at 1.7x the time, against 2.3x">
          <div class="inline">
            <ZenNumber :model-value="spec.objective.preserve_weight ?? 1" :min="0.05" :max="10" :step="0.25" :precision="2" @update:model-value="spec.objective.preserve_weight = $event" />
            <ZenNumber :model-value="spec.objective.preserve_every ?? 2" :min="1" :max="50" :step="1" :precision="0" @update:model-value="spec.objective.preserve_every = $event" />
          </div>
        </ZenField>
        <ZenField v-if="preserveOn" label="Plain prompts (%)" stack tip="The share of preservation passes on a plain prompt (“a photo of a woman”) at heavy noise, instead of the step's own caption with the class word in. The captions hold prompts that describe the picture; the plain ones hold the short prompts people write, where a picture made from noise can still drift toward your character. 0 = the captions only">
          <ZenNumber :model-value="Math.round((spec.objective.preserve_plain ?? 0) * 100)" :min="0" :max="100" :step="10" :precision="0" @update:model-value="spec.objective.preserve_plain = $event / 100" />
        </ZenField>
        <ZenField label="Validation every" stack tip="Fixed-noise loss for a smooth curve; 0 = off">
          <ZenNumber v-model="spec.validation.every" :min="0" :step="50" :precision="0" />
        </ZenField>
        <ZenField v-if="spec.validation.every" label="Validate on" stack tip="Where the validation pictures come from. A held-out dataset (pictures, or edit pairs, the LoRA never trains on, laid out like the training one) shows whether it generalises; from the training set, how well it fits">
          <ZenCombo v-model="spec.validation.dataset" :items="heldOutItems" :sorted="false" />
        </ZenField>
        <ZenField v-if="spec.validation.every" label="Validation pictures" stack :tip="spec.validation.dataset ? 'How many of the held-out pictures to measure (each at several noise levels): more is a steadier curve and a longer pass' : 'How many training pictures to measure, each at several noise levels'">
          <ZenNumber v-model="spec.validation.images" :min="1" :max="64" :step="1" :precision="0" />
        </ZenField>
      </div>
    </div>
  </div>
</section>
</template>

<style scoped src="./form.css"></style>
<style scoped>
.warn-note {
  margin: 4px 0 0;
  font-size: 12px;
  color: var(--zen-warn);
}
.patterns {
  width: 100%;
  resize: vertical;
  font-size: 12px;
  color: var(--zen-text);
  background: var(--zen-field-bg);
  border: 1px solid var(--zen-border);
  border-radius: var(--zen-radius);
  padding: 6px 10px;
}
.adv {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.subgroup {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.sub-title {
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted);
  padding-bottom: 6px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.place-opt {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-width: 0;
}
.place-opt .ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.opt-row {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  min-width: 0;
  font-size: 12px;
}
.opt-tip {
  display: inline-flex;
}
.opt-row.off {
  opacity: 0.5;
}

/* the adapter and its rank on one line */
.adapter-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.adapter-row .rank {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
}
.adapter-row .rank > :last-child {
  flex: 1;
  min-width: 0;
}

.opt-more {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
}
</style>
