<script setup lang="ts">
// Memory & speed: training precision, attention, checkpointing, offloading, the int8 paths, TREAD
// and compiling; folded away under one line saying what's on.
import { ZenCombo, ZenField, ZenNumber, ZenSwitch } from '@nynxz/zenkit-ui'
import { reactive } from 'vue'
import { items, useRunFormContext } from './form'

const { spec, trainAttention } = useRunFormContext()

const advanced = reactive({ memory: false })
</script>

<template>
<section class="card sec">
  <button class="more head" @click="advanced.memory = !advanced.memory">
    <i :class="advanced.memory ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /> Memory & speed
    <span class="muted small">{{ spec.memory.training_dtype }} · checkpointing {{ spec.memory.gradient_checkpointing ? 'on' : 'off' }}{{ spec.memory.offloading ? ' · offload' : '' }}</span>
  </button>
  <div v-if="advanced.memory" class="grid-2">
    <ZenField label="Training dtype" stack><ZenCombo v-model="spec.memory.training_dtype" :items="items(['bf16', 'fp16', 'fp32'])" :searchable="false" :sorted="false" /></ZenField>
    <ZenField label="Attention" stack tip="Only backends with a backward pass"><ZenCombo v-model="spec.memory.attention" :items="items(trainAttention)" :searchable="false" /></ZenField>
    <ZenField label="Gradient checkpointing" stack tip="Recompute each block in backward"><div class="sw-row"><ZenSwitch v-model="spec.memory.gradient_checkpointing" /><span class="sw-state">{{ spec.memory.gradient_checkpointing ? 'On' : 'Off' }}</span></div></ZenField>
    <ZenField v-if="spec.memory.gradient_checkpointing" label="Keep last blocks" stack tip="checkpoint_skip: the last N blocks skip checkpointing: they keep their activations (more VRAM) and skip the recompute (faster). Spare VRAM spent on speed">
      <ZenNumber v-model="spec.memory.checkpoint_skip" :min="0" :max="64" :step="1" :precision="0" />
    </ZenField>
    <ZenField label="Offload weights" stack tip="Stream weights from RAM"><div class="sw-row"><ZenSwitch v-model="spec.memory.offloading" /><span class="sw-state">{{ spec.memory.offloading ? 'On' : 'Off' }}</span></div></ZenField>
    <ZenField label="FP8 backward" stack tip="fp8 bases: the backward's matmuls in fp8 too. Faster, less precise"><div class="sw-row"><ZenSwitch v-model="spec.memory.fp8_backward" /><span class="sw-state">{{ spec.memory.fp8_backward ? 'On' : 'Off' }}</span></div></ZenField>
    <ZenField label="TREAD (video)" stack tip="On video steps this share of the clip's tokens skips the middle blocks and rejoins unchanged: those blocks do about half the work at 50%. Stills, samples and validation run whole. Best with some stills in the dataset: they anchor the LoRA. MiniMax H3">
      <div class="sw-row">
        <ZenSwitch :model-value="(spec.memory.tread ?? 0) > 0" @update:model-value="spec.memory.tread = $event ? 0.5 : 0" />
        <ZenNumber v-if="(spec.memory.tread ?? 0) > 0" :model-value="Math.round((spec.memory.tread ?? 0.5) * 100)" :min="10" :max="90" :step="5" :precision="0" @update:model-value="spec.memory.tread = $event / 100" />
        <span v-else class="sw-state">Off</span>
      </div>
    </ZenField>
    <ZenField v-if="(spec.memory.tread ?? 0) > 0" label="Share of steps routed" stack tip="TREAD on only this share of clip steps; the rest run whole. Routing every step of a dataset with no stills lets the LoRA learn to make up for the routing (measured on H3: over-sharpened samples). Whole steps keep it working on the model it'll actually be used on">
      <ZenNumber :model-value="Math.round((spec.memory.tread_steps ?? 1) * 100)" :min="0" :max="100" :step="10" :precision="0" @update:model-value="spec.memory.tread_steps = $event / 100" />
    </ZenField>
    <ZenField label="Compile blocks" stack tip="torch.compile on the model's repeated transformer block: the small operations around the big matmuls (norms, modulation, activations, the LoRA add) fuse into fewer kernels. The first steps and the first sample are slower while it compiles. Auto compiles runs of 500 steps or more when Triton is installed. Works with offloading too: most of the gain while little of the model streams (MiniMax H3 at 512 px: 32% faster a step), about even when most of it does (the copies set the pace)">
      <div class="sw-row"><ZenSwitch :model-value="spec.memory.compile_blocks === true" @update:model-value="spec.memory.compile_blocks = $event" /><span class="sw-state">{{ spec.memory.compile_blocks == null ? 'Auto' : spec.memory.compile_blocks ? 'On' : 'Off' }}</span><button v-if="spec.memory.compile_blocks != null" type="button" class="sw-auto" @click="spec.memory.compile_blocks = null">Auto</button></div>
    </ZenField>
    <ZenField label="int8 forward" stack tip="int8 ConvRot bases (MiniMax H3, Krea 2 int8): the frozen layers run ComfyUI's int8 kernel forward, activations quantized, instead of rebuilding a bf16 weight for each matmul. Off is slower: for comparing quality">
      <div class="sw-row"><ZenSwitch :model-value="spec.memory.int8_forward !== false" @update:model-value="spec.memory.int8_forward = $event" /><span class="sw-state">{{ spec.memory.int8_forward !== false ? 'On' : 'Off' }}</span></div>
    </ZenField>
    <ZenField label="int8 backward" stack tip="int8 ConvRot bases (MiniMax H3, Krea 2 int8): how each frozen layer's input gradient is computed. W8A8 (the default): the gradient quantized to int8 too, on int8 tensor cores, about 2× the bf16 backward on an RTX 5090; rounded stochastically so it stays unbiased, noisier per step than 16-bit. Fused: a Triton kernel at full precision instead of a rebuilt bf16 weight (Fizgig's, by @mabseyuk), saving that weight's VRAM. bf16: the weight rebuilt in bf16, the slowest">
      <ZenCombo :model-value="spec.memory.int8_backward ?? 'w8a8'" :items="items(['w8a8', 'fused', 'bf16'], { w8a8: 'W8A8', fused: 'Fused', bf16: 'bf16' })" :searchable="false" :sorted="false" @update:model-value="spec.memory.int8_backward = $event as 'w8a8' | 'fused' | 'bf16'" />
    </ZenField>
    <ZenField label="Fused q/k norm + rope" stack tip="Krea 2, MOSS and MiniMax H3: the attention's q/k norms and rope run as one op with its own backward instead of six separate ones (fewer copies and casts; 6–7% a step on Krea 2 and MOSS at 1024, 9% on H3 at 512). Same maths, slightly less rounding. Auto uses it whenever the blocks run uncompiled; compiled blocks fuse these already">
      <div class="sw-row"><ZenSwitch :model-value="spec.memory.fused_qk_rope !== false" @update:model-value="spec.memory.fused_qk_rope = $event" /><span class="sw-state">{{ spec.memory.fused_qk_rope == null ? 'Auto' : spec.memory.fused_qk_rope ? 'On' : 'Off' }}</span><button v-if="spec.memory.fused_qk_rope != null" type="button" class="sw-auto" @click="spec.memory.fused_qk_rope = null">Auto</button></div>
    </ZenField>
    <ZenField label="Bypass mode" stack tip="Adapters as forward hooks beside each layer instead of patched into its weight. Quantized bases always train this way"><div class="sw-row"><ZenSwitch v-model="spec.memory.bypass_mode" /><span class="sw-state">{{ spec.memory.bypass_mode ? 'On' : 'Off' }}</span></div></ZenField>
  </div>
</section>
</template>

<style scoped src="./form.css"></style>
