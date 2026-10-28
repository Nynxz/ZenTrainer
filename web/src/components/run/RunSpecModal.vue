<script setup lang="ts">
// A run's full spec: every setting by part, in words (as Recipes lists the ones it changes), or the
// JSON itself; either way it copies or downloads as JSON, ready for `zt train` or a new run.
import { ZenButton, ZenModal } from '@nynxz/zenkit-ui'
import { computed, ref } from 'vue'
import { specRows } from '@/lib/specText'

const props = defineProps<{ spec: Record<string, unknown>; name: string }>()
const open = defineModel<boolean>('open', { default: false })

const view = ref<'settings' | 'json'>('settings')
const json = computed(() => JSON.stringify(props.spec, null, 2))
const groups = computed(() => specRows(props.spec))

const copied = ref(false)
async function copy() {
  await navigator.clipboard.writeText(json.value)
  copied.value = true
  setTimeout(() => (copied.value = false), 1500)
}
function download() {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([json.value + '\n'], { type: 'application/json' }))
  a.download = `${props.name || 'run'}.json`
  a.click()
  URL.revokeObjectURL(a.href)
}
</script>

<template>
  <ZenModal v-model:open="open" width="760px">
    <template #header>
      <div class="head">
        <b class="zt-title">Spec</b>
        <span class="muted name">{{ name }}</span>
        <span class="spacer" />
        <div class="view-switch text" role="tablist" aria-label="View">
          <button :class="{ on: view === 'settings' }" @click="view = 'settings'">Settings</button>
          <button :class="{ on: view === 'json' }" @click="view = 'json'">JSON</button>
        </div>
      </div>
    </template>

    <div v-if="view === 'settings'" class="settings">
      <template v-for="g in groups" :key="g.section">
        <div class="sec">{{ g.section }}</div>
        <div v-for="r in g.rows" :key="g.section + r.label" class="row" :title="r.full">
          <span class="label">{{ r.label }}</span>
          <span class="value mono">{{ r.value }}</span>
        </div>
      </template>
    </div>
    <pre v-else class="mono json">{{ json }}</pre>

    <template #footer>
      <div class="foot">
        <span class="muted small">As JSON: <code>zt train</code> takes it, or paste it on a New run page</span>
        <span class="spacer" />
        <ZenButton icon="mdi mdi-download-outline" @click="download">Download .json</ZenButton>
        <ZenButton variant="primary" :icon="copied ? 'mdi mdi-check' : 'mdi mdi-content-copy'" @click="copy">{{ copied ? 'Copied' : 'Copy' }}</ZenButton>
      </div>
    </template>
  </ZenModal>
</template>

<style scoped>
.head,
.foot {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}
.name {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
/* as Recipes lists a recipe's changes: a part's name over its settings, one a row */
.settings {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr);
  column-gap: 12px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  overflow: hidden;
  font-size: 12px;
}
.sec {
  grid-column: 1 / -1;
  padding: 6px 12px;
  font-size: 11px;
  font-weight: 600;
  color: var(--zen-muted);
  background: var(--zen-surface-2);
  border-top: 1px solid var(--zen-border);
}
.sec:first-child {
  border-top: none;
}
.row {
  display: contents;
}
.row > * {
  padding: 4px 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.label {
  padding-left: 12px;
}
.value {
  color: var(--zen-accent);
  padding-right: 12px;
}
.json {
  margin: 0;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
