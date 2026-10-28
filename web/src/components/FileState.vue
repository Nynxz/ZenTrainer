<script setup lang="ts">
// One model file's state, in a fixed-width slot: installed, downloading (bar + cancel), or a
// download button with its size. Used by the model picker and the Models page.
import { computed } from 'vue'
import { engine, type ModelFile } from '@/api/engine'
import { downloadFor, fmtBytes } from '@/stores/downloads'

const props = defineProps<{ family: string; file: ModelFile }>()
const emit = defineEmits<{ error: [string] }>()

const dl = computed(() => downloadFor(props.file.folder, props.file.name))
const active = computed(() => dl.value && (dl.value.status === 'queued' || dl.value.status === 'downloading'))
const pct = computed(() => (dl.value?.total ? dl.value.done / dl.value.total : 0))

async function start() {
  try {
    await engine.download(props.family, props.file.folder, props.file.name)
  } catch (e) {
    emit('error', String(e))
  }
}
</script>

<template>
  <div class="fs">
    <span v-if="file.installed" class="ok" :title="file.installed"><i class="mdi mdi-check-circle" /> installed</span>
    <template v-else-if="active && dl">
      <div class="bar" :title="`${fmtBytes(dl.done)} of ${fmtBytes(dl.total)}${dl.speed ? ` · ${fmtBytes(dl.speed)}/s` : ''}`">
        <div :style="{ width: `${pct * 100}%` }" />
      </div>
      <span class="mono pct">{{ dl.status === 'queued' ? 'queued' : `${Math.floor(pct * 100)}%` }}</span>
      <button class="ib" title="Cancel (the partial file is kept, so it resumes)" @click="engine.cancelDownload(dl.id)"><i class="mdi mdi-close" /></button>
    </template>
    <button v-else class="dl" :class="{ err: dl?.status === 'error' }" :title="dl?.error ?? `Download from ${file.url}`" @click="start">
      <i :class="dl?.status === 'error' ? 'mdi mdi-alert-outline' : 'mdi mdi-download'" />
      {{ dl?.status === 'error' ? 'retry' : fmtBytes(file.size) || 'download' }}
    </button>
  </div>
</template>

<style scoped>
.fs {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  font-size: 12px;
  white-space: nowrap;
  min-width: 0;
}
.ok {
  color: var(--zen-ok);
}
.dl {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border: 1px solid color-mix(in srgb, var(--zen-accent) 50%, transparent);
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-accent) 10%, transparent);
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.dl:hover {
  background: color-mix(in srgb, var(--zen-accent) 22%, transparent);
}
.dl.err {
  border-color: var(--zen-danger);
  background: none;
}
.bar {
  width: 70px;
  height: 4px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.bar div {
  height: 100%;
  background: var(--zen-accent);
}
.pct {
  min-width: 4ch;
  text-align: right;
  font-size: 11px;
}
.ib {
  width: 22px;
  height: 22px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.ib:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
</style>
