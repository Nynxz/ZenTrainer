<script setup lang="ts">
// What a cleanup on the System page will delete, listed before anything goes: by run or dataset,
// each item with its size (and why it goes, for checkpoints), the total, and what stays. Delete runs
// `run`; Cancel or closing the dialog deletes nothing.
import { ZenButton, ZenModal } from '@nynxz/zenkit-ui'
import { computed, ref, watch } from 'vue'
import { fmtBytes } from '@/stores/downloads'

export interface CleanupGroup {
  name: string
  items: { label: string; bytes: number; note?: string }[]
}
export interface Cleanup {
  title: string
  groups: CleanupGroup[]
  /** what stays, said under the list */
  keeps: string
  /** what the groups are: "run" or "dataset" */
  unit: string
  action: string
  run: () => Promise<void>
  /** still finding what goes (a trim's dry run): said with a spinner, nothing can be deleted yet */
  loading?: string
}

const props = defineProps<{ cleanup: Cleanup | null }>()
const emit = defineEmits<{ close: [] }>()
const open = computed({ get: () => !!props.cleanup, set: (v) => !v && !busy.value && emit('close') })
const busy = ref(false)
const error = ref('')
const unfolded = ref(new Set<number>())
watch(() => props.cleanup, () => {
  error.value = ''
  unfolded.value = new Set()
})
const FOLD = 6 // a group with more items starts folded to its total
const total = computed(() => (props.cleanup?.groups ?? []).reduce((n, g) => n + g.items.reduce((m, i) => m + i.bytes, 0), 0))
const count = computed(() => (props.cleanup?.groups ?? []).reduce((n, g) => n + g.items.length, 0))
const groupBytes = (g: CleanupGroup) => g.items.reduce((n, i) => n + i.bytes, 0)
function toggle(i: number) {
  const s = new Set(unfolded.value)
  if (!s.delete(i)) s.add(i)
  unfolded.value = s
}
async function go() {
  if (!props.cleanup) return
  busy.value = true
  error.value = ''
  try {
    await props.cleanup.run()
    emit('close')
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <ZenModal v-model:open="open" :title="cleanup?.title" width="560px" height="auto">
    <div v-if="cleanup?.loading" class="cleanup finding">
      <i class="mdi mdi-loading mdi-spin" /> {{ cleanup.loading }}
    </div>
    <div v-else-if="cleanup" class="cleanup">
      <p class="sum">
        <b>{{ count }} item{{ count === 1 ? '' : 's' }}</b> from {{ cleanup.groups.length }} {{ cleanup.unit }}{{ cleanup.groups.length === 1 ? '' : 's' }},
        <b>{{ fmtBytes(total) }}</b> in all.
      </p>
      <ul class="groups">
        <li v-for="(g, gi) in cleanup.groups" :key="g.name + gi" class="group">
          <!-- one item: one line (its name, what goes, the size) instead of a header repeating the item -->
          <div v-if="g.items.length === 1 && !g.items[0]!.note" class="single">
            <span class="gname">{{ g.name }}</span>
            <span class="ilabel muted small">{{ g.items[0]!.label }}</span>
            <span class="ibytes mono">{{ fmtBytes(g.items[0]!.bytes) }}</span>
          </div>
          <button
            v-else
            type="button"
            class="ghead"
            :class="{ fold: g.items.length > FOLD }"
            :aria-expanded="g.items.length <= FOLD || unfolded.has(gi)"
            :disabled="g.items.length <= FOLD"
            @click="toggle(gi)"
          >
            <i v-if="g.items.length > FOLD" :class="unfolded.has(gi) ? 'mdi mdi-chevron-down' : 'mdi mdi-chevron-right'" />
            <span class="gname">{{ g.name }}</span>
            <span class="muted small">{{ g.items.length }} · {{ fmtBytes(groupBytes(g)) }}</span>
          </button>
          <ul v-if="!(g.items.length === 1 && !g.items[0]!.note) && (g.items.length <= FOLD || unfolded.has(gi))" class="items">
            <li v-for="(it, ii) in g.items" :key="ii" class="item">
              <span class="ilabel mono">{{ it.label }}</span>
              <span class="ibytes mono">{{ fmtBytes(it.bytes) }}</span>
              <span v-if="it.note" class="inote muted small">{{ it.note }}</span>
            </li>
          </ul>
        </li>
      </ul>
      <p class="keeps muted small"><i class="mdi mdi-shield-check-outline" /> {{ cleanup.keeps }}</p>
      <p v-if="error" class="err small"><i class="mdi mdi-alert-outline" /> {{ error }}</p>
    </div>
    <template #footer>
      <div class="foot">
        <ZenButton :disabled="busy" @click="emit('close')">Cancel</ZenButton>
        <ZenButton variant="danger" :icon="busy ? 'mdi mdi-loading mdi-spin' : 'mdi mdi-delete-outline'" :disabled="busy || !count || !!cleanup?.loading" @click="go">
          {{ busy ? 'Deleting…' : cleanup?.loading ? cleanup.action : `${cleanup?.action ?? 'Delete'} · ${fmtBytes(total)}` }}
        </ZenButton>
      </div>
    </template>
  </ZenModal>
</template>

<style scoped>
.cleanup {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.finding {
  flex-direction: row;
  align-items: center;
  gap: 8px;
  padding: 18px 2px;
  color: var(--zen-muted);
}
.foot {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  width: 100%;
}
.cleanup p {
  margin: 0;
  line-height: 1.45;
}
.groups,
.items {
  list-style: none;
  margin: 0;
  padding: 0;
}
.groups {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: min(calc(48vh / var(--zen-page-zoom)), 420px);
  overflow: auto;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  padding: 6px;
}
.ghead {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  border: 0;
  background: none;
  color: var(--zen-text);
  font: inherit;
  padding: 4px 6px;
  border-radius: var(--r-sm);
  text-align: left;
}
.ghead.fold {
  cursor: pointer;
}
.ghead.fold:hover {
  background: var(--zen-surface-2);
}
.ghead:disabled {
  cursor: default;
}
.gname {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
}
.single {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto 5.5em; /* the size a fixed column: the labels line up */
  align-items: baseline;
  gap: 12px;
  padding: 4px 6px;
}
.items {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 0 6px 4px 22px;
}
.item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0 10px;
  font-size: 12px;
}
.ilabel {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ibytes {
  color: var(--zen-muted);
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.inote {
  grid-column: 1 / -1;
}
.err {
  color: var(--zen-warn);
}
</style>
