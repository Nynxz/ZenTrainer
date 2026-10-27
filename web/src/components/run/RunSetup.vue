<script setup lang="ts">
// The rail's Setup card: what the run is, as label/value lines that wrap rather than cut off,
// and the full spec behind a link.
import { ZenModal } from '@nynxz/zenkit-ui'
import { computed, ref } from 'vue'
import RunDatasetLightbox from './RunDatasetLightbox.vue'
import { fmtLr, useRunContext } from './context'

const { detail } = useRunContext()

const specOpen = ref(false)
const OPT_NAMES: Record<string, string> = { Automagic3: 'Automagic v3', Automagic4: 'Automagic v4', AdamWScheduleFree: 'AdamW schedule-free', AdamW8bit: 'AdamW 8-bit' }
const ADAPTIVE = new Set(['Automagic', 'Automagic3', 'Automagic4', 'Prodigy']) // they set their own LR: the schedule hardly matters
const baseName = (p: string) => p.split(/[\\/]/).pop()!.replace(/\.safetensors$/, '')
// a dataset in the datasets folder (by name, not a path elsewhere) opens in the lightbox, on the run's caption set
const showData = ref<{ name: string; set: string } | null>(null)
const datasetOpen = (folder: string, set?: string) =>
  /^([\\/]|[A-Za-z]:)/.test(folder) ? undefined : () => (showData.value = { name: folder, set: set || 'default' })
// lead: the start of v that opens (the dataset's name)
const setup = computed((): { k: string; v: string; title?: string; lead?: string; open?: () => void }[] => {
  const sp = detail.value?.spec
  if (!sp) return []
  const model = sp.model.unet_name ?? sp.model.ckpt_name ?? ''
  const m = sp.memory
  const mem = [
    m.offloading ? 'offload' : '',
    m.int8_forward !== false && /int8/i.test(model) ? 'int8 fwd' : '',
    /int8/i.test(model) ? { w8a8: 'W8A8', fused: 'fused int8', bf16: '' }[m.int8_backward ?? 'w8a8'] : '',
    m.compile_blocks === true ? 'compile' : (m.compile_blocks ?? 'auto') === 'auto' ? 'compile auto' : '',
    m.fused_qk_rope === true ? 'fused q/k rope' : '',
    m.gradient_checkpointing ? (m.grad_checkpoint_skip === 'auto' ? 'checkpoint · keep auto' : m.grad_checkpoint_skip ? `checkpoint · keep ${m.grad_checkpoint_skip}` : 'checkpoint') : '',
    m.tread ? `TREAD ${Math.round(m.tread * 100)}%` : '',
  ].filter(Boolean)
  const opt = sp.optim
  return [
    { k: 'Model', v: model.replace(/\.safetensors$/, ''), title: model },
    {
      k: 'Dataset',
      v: [sp.dataset.folder, sp.dataset.caption_set && sp.dataset.caption_set !== 'default' ? sp.dataset.caption_set : '', sp.dataset.trigger ? `“${sp.dataset.trigger}”` : ''].filter(Boolean).join(' · '),
      lead: sp.dataset.folder,
      open: datasetOpen(sp.dataset.folder, sp.dataset.caption_set),
    },
    {
      k: 'Training',
      v: `${opt.steps} steps · batch ${opt.batch_size} · ${[...(sp.dataset.stages ?? []).map((st) => st.resolution), sp.dataset.resolution].join(' → ')}px`,
      title: sp.dataset.stages?.length
        ? `${sp.dataset.stages.map((st) => `${st.resolution} px until ${Math.round(st.until * 100)}%`).join(', ')}, then ${sp.dataset.resolution} px; each change blends over ${Math.round((sp.dataset.stage_blend ?? 0.2) * 100)}% of the run`
        : undefined,
    },
    {
      k: 'Optimizer',
      v: [OPT_NAMES[opt.optimizer] ?? opt.optimizer, `from ${fmtLr(opt.learning_rate)}`, ADAPTIVE.has(opt.optimizer) ? '' : opt.lr_scheduler].filter(Boolean).join(' · '),
    },
    { k: 'Network', v: `${sp.network.algorithm} r${sp.network.rank} · ${Array.isArray(sp.network.targets) ? `${sp.network.targets.length} layer types` : sp.network.targets.replace('_', ' + ')}${sp.dataset.masked_loss ? ' · masked' : ''}` },
    ...(sp.network.base_lora
      ? [{
          k: 'Base LoRA',
          v: `${baseName(sp.network.base_lora)} at ${sp.network.base_strength ?? 1} · load it beside this LoRA`,
          title: `Trained on top of ${sp.network.base_lora}: the LoRA expects it, at the same strength`,
        }]
      : []),
    ...(sp.network.training_adapter
      ? [{ k: 'Trained over', v: `${baseName(sp.network.training_adapter)} · training only`, title: sp.network.training_adapter }]
      : []),
    ...(mem.length ? [{ k: 'Memory', v: mem.join(' · ') }] : []),
  ]
})
</script>

<template>
  <section class="card">
    <header class="sec-head">
      <span class="sec-title">Setup</span>
      <span class="spacer" />
      <button class="link" @click="specOpen = true"><i class="mdi mdi-code-json" /> Full spec</button>
    </header>
    <dl class="kv">
      <div v-for="it in setup" :key="it.k" class="kv-row">
        <dt>{{ it.k }}</dt>
        <dd :title="it.title ?? it.v">
          <template v-if="it.open && it.lead"><button class="to" title="Show its pictures" @click="it.open">{{ it.lead }}</button>{{ it.v.slice(it.lead.length) }}</template>
          <template v-else>{{ it.v }}</template>
        </dd>
      </div>
    </dl>
  </section>
  <ZenModal v-if="detail" v-model:open="specOpen" title="Spec" width="760px">
    <pre class="mono spec">{{ JSON.stringify(detail.spec, null, 2) }}</pre>
  </ZenModal>
  <RunDatasetLightbox v-if="showData" :name="showData.name" :set="showData.set" @close="showData = null" />
</template>

<style scoped src="./section.css"></style>
<style scoped>
/* setup: label / value lines; values wrap instead of cutting off */
.kv {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.kv-row {
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr);
  gap: 10px;
  font-size: 12px;
  line-height: 1.4;
}
.kv-row dt {
  color: var(--zen-muted);
}
.kv-row dd {
  margin: 0;
  white-space: normal;
  overflow: visible;
  overflow-wrap: anywhere;
}
dt {
  font-size: 12px;
  color: var(--zen-muted);
  white-space: nowrap;
}
dd {
  margin: 0;
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.link {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  white-space: nowrap;
}
.link:hover {
  color: var(--zen-accent);
}
.to {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  text-align: left;
  color: inherit;
  cursor: pointer;
  text-decoration: underline dotted color-mix(in srgb, var(--zen-text) 40%, transparent);
  text-underline-offset: 3px;
}
.to:hover {
  color: var(--zen-accent);
  text-decoration-color: currentColor;
}
.to:focus-visible {
  outline: 1px solid var(--zen-accent);
  outline-offset: 2px;
  border-radius: 2px;
}
.spec {
  margin: 0;
  font-size: 12px;
  white-space: pre-wrap;
}
</style>
