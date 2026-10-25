<script setup lang="ts">
// The form's sticky summary: what you're about to run, what looks off, and the buttons that start
// it (or save the run being edited).
import { ZenButton, ZenField, ZenInput } from '@nynxz/zenkit-ui'
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { EngineError, engine, promptText } from '@/api/engine'
import { useRunFormContext } from './form'

const router = useRouter()
const { spec, modelMissing, dataset, opt, epochs, modelLabel, stage0, applied, editing, retrying, resuming, error } = useRunFormContext()
const submitting = ref(false)

const warnings = computed(() => {
  const w: string[] = []
  const slider = !!spec.slider?.kind
  if (spec.slider?.kind === 'text' && !(spec.slider.positive.trim() && spec.slider.negative.trim())) w.push('A slider needs both ends: Positive and Negative.')
  if (spec.slider?.kind === 'pairs' && !spec.slider.pair_set) w.push("An image slider needs its pairs: pick the other end's picture set.")
  if (!slider && !spec.dataset.trigger) w.push('No trigger word: the likeness has nothing specific to attach to.')
  const masked = dataset.value?.masked ?? 0
  if (masked && !spec.dataset.masked_loss)
    w.push(`${masked} picture${masked === 1 ? ' has a mask' : 's have masks'}, but Masked training is off: the parts the masks leave out train too.`)
  if (!spec.samples.prompts.length) w.push('No sample prompts: you will only see the loss curves.')
  else if (!spec.samples.every) w.push('Samples are off (Every 0): only Sample now renders them, so checkpoints get no samples or likeness. Set Every to the save interval.')
  else if (spec.save_every && spec.save_every % spec.samples.every && !spec.output.keep.startsWith('likeness')) w.push(`Samples every ${spec.samples.every} don't land on the saves every ${spec.save_every}: checkpoints get no samples or likeness to compare.`)
  else if (!slider && spec.dataset.trigger && !spec.samples.prompts.map(promptText).some((p) => p.includes('[trigger]') || p.includes(spec.dataset.trigger)))
    w.push("Sample prompts don't use [trigger].")
  if (Array.isArray(spec.network.targets) && !spec.network.targets.length) w.push('Layers is Custom with no patterns: no layer would get an adapter.')
  const keep = spec.save_every ? spec.output.keep.replace('_new_best', '') : 'all'
  if (keep === 'likeness' && !spec.samples.prompts.length)
    w.push('Keeping the best likeness needs sample prompts to score (Samples).')
  if (keep === 'val_loss' && !spec.validation.every) w.push('Keeping the lowest validation loss needs validation on (Advanced training → Validation every).')
  if (opt.value?.id === 'Prodigy' && spec.optim.learning_rate < 0.5) w.push('Prodigy sets its own step size: use LR 1.0.')
  if (opt.value && opt.value.id !== 'Prodigy' && spec.optim.learning_rate >= 0.01) w.push('That LR is very high for this optimizer.')
  const automagic = ['Automagic', 'Automagic3', 'Automagic4'].includes(opt.value?.id ?? '')
  if (opt.value?.id === 'Automagic' && spec.optim.learning_rate > 1e-4) w.push("Automagic's LR is only where each weight starts: use 1e-6 and let it climb.")
  if (opt.value?.id === 'Automagic3' && spec.optim.learning_rate < 2e-5) w.push('Automagic v3 climbs only on a steady trend, which small batches rarely give: from a low start it can stay near it the whole run. Start around 1e-4.')
  if (spec.network.lora_dtype === 'bf16' && !automagic) w.push('bf16 LoRA weights lose most small steps with this optimizer: use fp32.')
  if (opt.value && !opt.value.available) w.push(`${opt.value.label} isn't installed (scripts/zt extras optimizers).`)
  if (epochs.value != null && epochs.value < 3) w.push(`Only ${epochs.value.toFixed(1)} epochs: probably too few steps.`)
  return w
})
const missing = computed(() => {
  const m = spec.model
  if (modelMissing.value) return modelMissing.value === 1 ? 'Download the missing model file' : `Download ${modelMissing.value} model files`
  if (!spec.dataset.folder) return 'Pick a dataset'
  if (m.kind === 'checkpoint' && !m.ckpt_name) return 'Pick a checkpoint'
  if (m.kind === 'split' && !(m.unet_name && m.clip_name && m.clip_type && m.vae_name)) return 'Pick all model parts'
  return null
})

async function saveEdit() {
  if (!editing.value) return
  submitting.value = true
  error.value = null
  try {
    await engine.editRun(editing.value, spec)
    if (resuming.value) await engine.resume(editing.value)
    router.push(retrying.value || resuming.value ? `/runs/${editing.value}` : '/jobs')
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  } finally {
    submitting.value = false
  }
}

async function start(position: 'next' | 'end' = 'next') {
  submitting.value = true
  error.value = null
  try {
    const run = await engine.start(spec, position)
    router.push(`/runs/${run.run_id}`)
  } catch (e) {
    const comfy = e instanceof EngineError ? (e.body as { comfy?: unknown })?.comfy : null
    error.value = comfy ? `${(e as Error).message}\n${JSON.stringify(comfy, null, 2)}` : String((e as Error).message ?? e)
  } finally {
    submitting.value = false
  }
}
</script>

<template>
      <aside class="summary card">
<ZenField label="Run name" stack><ZenInput v-model="spec.name" /></ZenField>
<dl>
  <dt>Model</dt><dd class="ellipsis" :title="modelLabel ?? ''">{{ modelLabel || '–' }}</dd>
  <dt>Dataset</dt><dd>{{ spec.dataset.folder || '–' }}<span v-if="dataset" class="muted"> · {{ dataset.images }} imgs</span></dd>
  <dt>Captions</dt><dd>{{ spec.dataset.caption_mix.length ? spec.dataset.caption_mix.join(' + ') : spec.dataset.caption_set }}</dd>
  <template v-if="spec.slider?.kind === 'text'"><dt>Slider</dt><dd class="ellipsis" :title="`${spec.slider.negative} ⟷ ${spec.slider.positive}`">{{ spec.slider.negative || '…' }} ⟷ {{ spec.slider.positive || '…' }}</dd></template>
  <template v-else-if="spec.slider?.kind === 'pairs'"><dt>Slider</dt><dd>image pairs<span v-if="spec.slider.pair_set" class="muted"> · {{ spec.slider.pair_set }}</span></dd></template>
  <template v-else><dt>Trigger</dt><dd>{{ spec.dataset.trigger || '–' }}</dd></template>
  <dt>Steps</dt><dd class="mono">{{ spec.optim.steps }}<span v-if="epochs != null" class="muted"> · {{ epochs.toFixed(1) }} ep</span></dd>
  <dt>Optimizer</dt><dd>{{ opt?.label ?? spec.optim.optimizer }} <span class="mono muted">{{ spec.optim.learning_rate }}</span></dd>
  <dt>Network</dt><dd class="mono">{{ spec.network.algorithm }} r{{ spec.network.rank }}</dd>
  <dt>Resolution</dt><dd class="mono">{{ stage0 ? `${spec.dataset.stages!.map((st) => st.resolution).join(' → ')} → ${spec.dataset.resolution}` : spec.dataset.resolution }}</dd>
  <dt>Samples</dt><dd>{{ spec.samples.prompts.length }} prompt{{ spec.samples.prompts.length === 1 ? '' : 's' }}{{ spec.samples.every ? ` / ${spec.samples.every} steps` : '' }}</dd>
</dl>
<div v-if="applied.length" class="muted small">Presets: {{ applied.join(' + ') }}</div>
<ul v-if="warnings.length" class="warns">
  <li v-for="w in warnings" :key="w"><i class="mdi mdi-alert-outline" /> {{ w }}</li>
</ul>
<div v-if="error" class="err mono">{{ error }}</div>
<template v-if="editing">
  <ZenButton variant="primary" icon="mdi mdi-content-save-outline" block :disabled="submitting || !!missing" :title="missing ?? (resuming ? 'Saves these settings and resumes it from its last state' : retrying ? 'Saves these settings and queues it to run next' : 'Keeps its place in the queue')" @click="saveEdit">
    {{ missing ?? (resuming ? 'Save & resume' : retrying ? 'Save & retry' : 'Save changes') }}
  </ZenButton>
  <ZenButton icon="mdi mdi-close" block @click="router.push('/jobs')">Cancel</ZenButton>
</template>
<template v-else>
  <ZenButton variant="primary" icon="mdi mdi-play" block :disabled="submitting || !!missing" :title="missing ?? 'Starts now, or right after the current run'" @click="start('next')">
    {{ missing ?? 'Start training' }}
  </ZenButton>
  <ZenButton icon="mdi mdi-tray-plus" block :disabled="submitting || !!missing" title="Line it up behind everything queued" @click="start('end')">Add to queue</ZenButton>
</template>
      </aside>
</template>

<style scoped src="./form.css"></style>
<style scoped>
.summary {
  position: sticky;
  top: 8px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
}
dl {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 6px 12px;
  margin: 0;
  font-size: 12px;
}
dt {
  color: var(--zen-muted);
}
dd {
  margin: 0;
}
.warns {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 12px;
}
.warns li {
  display: flex;
  gap: 6px;
  color: var(--zen-warn);
}
.err {
  font-size: 11px;
  white-space: pre-wrap;
  color: var(--zen-danger);
}
@media (max-width: 1000px) {
  .summary {
    position: static;
  }
}
</style>
