<script setup lang="ts">
// The rail's In VRAM card, while the run is live: the run's own models among what ComfyUI
// holds (matched by file), and how much of each is on the GPU.
import { computed, onMounted, onUnmounted } from 'vue'
import { loadedModels, watchLoaded } from '@/stores/loaded'
import { gb } from '@/stores/system'
import { isCollapsed, toggleCollapsed } from './collapse'
import { useRunContext } from './context'

const { detail, running } = useRunContext()

const runLoaded = computed(() => {
  const m = detail.value?.spec.model
  if (!m || !running.value) return []
  const files = new Set(
    [m.unet_name, m.ckpt_name, m.clip_name, m.vae_name, (m as { audio_vae_name?: string | null }).audio_vae_name, detail.value?.spec.samples.model_name]
      .filter(Boolean)
      .map((f) => String(f).split('/').pop()),
  )
  return loadedModels.value.filter((x) => x.file && x.file.split(' + ').some((f) => files.has(f)))
})
let stopLoaded = () => {}
onMounted(() => (stopLoaded = watchLoaded()))
onUnmounted(() => stopLoaded())
</script>

<template>
  <section v-if="runLoaded.length" class="card loaded-card">
    <header class="sec-head fold" :class="{ folded: isCollapsed('In VRAM') }" :title="isCollapsed('In VRAM') ? 'Show' : 'Fold away'" @click="toggleCollapsed('In VRAM')">
      <span class="sec-title">In VRAM</span>
      <span class="sec-sub">loaded now</span>
      <span class="spacer" />
      <i class="mdi fold-chev" :class="isCollapsed('In VRAM') ? 'mdi-chevron-down' : 'mdi-chevron-up'" />
    </header>
    <div v-for="m in isCollapsed('In VRAM') ? [] : runLoaded" :key="m.file ?? m.name" class="lrow" :title="`${m.file}: ${gb(m.vram)} of ${gb(m.bytes)} GB in VRAM`">
      <span class="lname ellipsis">{{ (m.file ?? m.name).replace(/\.safetensors$/, '') }}</span>
      <span class="mono small muted">{{ gb(m.vram) }}/{{ gb(m.bytes) }} GB</span>
      <div class="lbar"><span :style="{ width: `${m.bytes ? (m.vram / m.bytes) * 100 : 0}%` }" /></div>
    </div>
  </section>
</template>

<style scoped src="./section.css"></style>
<style scoped>
.loaded-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.loaded-card .sec-head {
  margin-bottom: 4px;
}
.lrow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 3px 8px;
  font-size: 12px;
}
.lbar {
  grid-column: span 2;
  height: 3px;
  border-radius: 2px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.lbar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
.lname {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
