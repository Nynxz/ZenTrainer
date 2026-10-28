<script setup lang="ts">
import { ZenButton, ZenInput } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { engine, type Health, type StorageReport } from '@/api/engine'
import { fmtBytes } from '@/stores/downloads'
import MemChart from '@/components/MemChart.vue'
import { RAM_SERIES as RAM, VRAM_SERIES as VRAM } from '@/lib/memSeries'
import { gb, memLatest, samples, startSystemSync } from '@/stores/system'

const health = ref<Health | null>(null)
onMounted(async () => {
  startSystemSync()
  health.value = await engine.health().catch(() => null)
})

const peak = (k: 'vram_used' | 'ram_used' | 'torch_allocated') =>
  Math.max(0, ...samples.value.map((s) => s[k] ?? 0))
const l = computed(() => memLatest.value)

// ---- storage -------------------------------------------------------------------------------
const store = ref<StorageReport | null>(null)
const storeError = ref<string | null>(null)
const note = ref('')
async function loadStorage() {
  try {
    store.value = await engine.storage()
  } catch (e) {
    storeError.value = String(e)
  }
}
onMounted(loadStorage)
const RESUMABLE = ['stopped', 'error', 'interrupted', 'rejected']
const ACTIVE_S = ['queued', 'loading', 'training', 'paused']
const anyActive = computed(() => (store.value?.runs ?? []).some((r) => ACTIVE_S.includes(r.status)))
const usage = computed(() => {
  const st = store.value
  if (!st) return []
  const sum = (f: (r: StorageReport['runs'][number]) => number) => st.runs.reduce((n, r) => n + f(r), 0)
  const rows = [
    { label: 'LoRA checkpoints', bytes: sum((r) => r.checkpoints) },
    { label: 'Resume states', bytes: sum((r) => r.states), hint: 'Optimizer state saved with every checkpoint; only stopped runs need it' },
    { label: 'Samples', bytes: sum((r) => r.samples) },
    { label: 'Dataset images', bytes: st.datasets.reduce((n, d) => n + d.images, 0) },
    { label: 'Dataset caches', bytes: st.datasets.reduce((n, d) => n + d.cache, 0), hint: 'Latents and text embeddings training made from your datasets: rebuilt when needed' },
    { label: 'App outputs', bytes: st.apps },
  ]
  const max = Math.max(1, ...rows.map((r) => r.bytes))
  return rows.map((r) => ({ ...r, frac: r.bytes / max }))
})
const total = computed(() => usage.value.reduce((n, r) => n + r.bytes, 0))
// every finished run's states at once; stopped runs keep theirs to resume from
const LIVE = ['queued', 'loading', 'training', 'paused']
const sweepable = computed(() => (store.value?.runs ?? []).filter((r) => r.states && !RESUMABLE.includes(r.status) && !LIVE.includes(r.status)))
// the runs table: biggest first (or by a column), the 12 first until Show all, the runs that keep nothing out of
// the way behind a count, and a search (by name or id) that looks through all of them
const RUNS_SHOWN = 12
const runQuery = ref('')
const allRuns = ref(false)
const emptyShown = ref(false)
type StoredRun = StorageReport['runs'][number]
const runSize = (r: StoredRun) => r.checkpoints + r.states + r.samples
// a column's header sorts by it: a name A to Z, a size biggest first (Total, the default)
type RunSort = 'name' | 'checkpoints' | 'states' | 'samples' | 'total'
const runSort = ref<RunSort>('total')
const sortValue = (r: StoredRun, k: RunSort) => (k === 'total' ? runSize(r) : k === 'name' ? 0 : r[k])
const runsFound = computed(() => {
  const q = runQuery.value.trim().toLowerCase()
  const k = runSort.value
  return [...(store.value?.runs ?? [])]
    .filter((r) => !q || r.name.toLowerCase().includes(q) || r.run_id.includes(q))
    .sort((a, b) => (k === 'name' ? a.name.localeCompare(b.name) : sortValue(b, k) - sortValue(a, k) || runSize(b) - runSize(a)))
})
const emptyRuns = computed(() => runsFound.value.filter((r) => !runSize(r)).length)
const runsListed = computed(() => runsFound.value.filter((r) => emptyShown.value || runSize(r)))
// the biggest 12, and the empty ones too once asked for (sorted by size they'd fall past the 12)
const runRows = computed(() => {
  if (allRuns.value || runQuery.value.trim()) return runsListed.value
  const top = runsListed.value.filter((r) => runSize(r)).slice(0, RUNS_SHOWN)
  return emptyShown.value ? [...top, ...runsListed.value.filter((r) => !runSize(r))] : top
})
const kept = computed(() => (store.value?.runs ?? []).filter((r) => r.states && RESUMABLE.includes(r.status)))
function cleanAllStates() {
  const bytes = sweepable.value.reduce((n, r) => n + r.states, 0)
  const keep = kept.value.length
    ? `\n\nKept: ${kept.value.length} stopped or failed run${kept.value.length === 1 ? '' : 's'} (${kept.value.map((r) => r.name).join(', ')}) still need theirs to resume.`
    : ''
  clean(
    { kind: 'states_all' },
    `Delete the resume states of ${sweepable.value.length} finished run${sweepable.value.length === 1 ? '' : 's'} (${fmtBytes(bytes)})`,
    `Every LoRA checkpoint and sample stays. Finished runs can't be resumed from their checkpoints afterwards (Train more starts fresh anyway).${keep}`,
  )
}
async function clean(body: { kind: 'states'; run_id: string } | { kind: 'states_all' } | { kind: 'cache'; folder: string }, what: string, warning: string) {
  if (!confirm(`${what}?\n\n${warning}`)) return
  note.value = ''
  try {
    const r = await engine.cleanStorage(body)
    note.value = `Freed ${fmtBytes(r.freed)}` + (r.runs != null ? ` from ${r.runs} run${r.runs === 1 ? '' : 's'}` : '')
    await loadStorage()
  } catch (e) {
    note.value = String((e as Error).message ?? e)
  }
}

const restarting = ref(false)
const restartError = ref<string | null>(null)
async function restart() {
  if (!confirm('Restart ComfyUI? The app reconnects when it is back.')) return
  restartError.value = null
  try {
    await engine.restart()
  } catch (e) {
    restartError.value = String((e as Error).message ?? e)
    return
  }
  restarting.value = true
  await new Promise((r) => setTimeout(r, 2000))
  for (let i = 0; i < 240; i++) {
    try {
      await engine.health()
      location.reload()
      return
    } catch {
      await new Promise((r) => setTimeout(r, 500))
    }
  }
  restarting.value = false
  restartError.value = "ComfyUI didn't come back; check its console."
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1>System</h1>
        <p class="lede">
          {{ health?.capabilities.device_name ?? health?.capabilities.device ?? 'GPU' }}
        </p>
      </div>
      <ZenButton icon="mdi mdi-restart" :disabled="restarting" @click="restart">Restart ComfyUI</ZenButton>
    </div>
    <div v-if="restartError" class="card err mono">{{ restartError }}</div>

    <div class="stats">
      <div class="card stat">
        <div class="eyebrow">VRAM</div>
        <div class="big mono">{{ gb(l?.vram_used) }} <span class="muted">/ {{ gb(l?.vram_total) }} GB</span></div>
        <div class="muted small">peak {{ gb(peak('vram_used')) }} GB</div>
      </div>
      <div class="card stat">
        <div class="eyebrow">PyTorch allocated</div>
        <div class="big mono">{{ gb(l?.torch_allocated) }} <span class="muted">GB</span></div>
        <div class="muted small">reserved {{ gb(l?.torch_reserved) }} GB · peak {{ gb(peak('torch_allocated')) }}</div>
      </div>
      <div class="card stat">
        <div class="eyebrow">RAM</div>
        <div class="big mono">{{ gb(l?.ram_used) }} <span class="muted">/ {{ gb(l?.ram_total) }} GB</span></div>
        <div class="muted small">ComfyUI {{ gb(l?.proc_rss) }} GB · peak {{ gb(peak('ram_used')) }}</div>
      </div>
    </div>

    <div class="card"><MemChart title="VRAM (GB)" :samples="samples" :series="VRAM" :total="l?.vram_total" /></div>
    <div class="card"><MemChart title="RAM (GB)" :samples="samples" :series="RAM" :total="l?.ram_total" /></div>

    <section class="storage">
      <div class="st-head">
        <h2>Storage</h2>
        <span class="muted small">{{ store ? `${fmtBytes(total)} from training and apps` : 'measuring…' }}</span>
        <span class="spacer" />
        <!-- fixed slot: cleanup results land here without moving anything -->
        <span class="note small mono">{{ note }}</span>
        <button class="chip-btn" title="Measure again" @click="loadStorage"><i class="mdi mdi-refresh" /> Refresh</button>
      </div>
      <div v-if="storeError" class="card err mono">{{ storeError }}</div>
      <template v-if="store">
        <div class="disks">
          <div v-for="d in store.disks" :key="d.path" class="card disk" :title="d.path">
            <div class="eyebrow">{{ d.labels.join(' · ') }}</div>
            <div class="big mono">{{ fmtBytes(d.free) }} <span class="muted">free of {{ fmtBytes(d.total) }}</span></div>
            <div class="meter"><span :class="{ hot: d.used / d.total > 0.9 }" :style="{ width: `${(d.used / d.total) * 100}%` }" /></div>
            <div class="muted small mono ellipsis">{{ d.path }}</div>
          </div>
        </div>

        <div class="card">
          <div class="eyebrow">What’s using space</div>
          <div class="usage">
            <template v-for="u in usage" :key="u.label">
              <span class="u-label" :title="u.hint">{{ u.label }}<i v-if="u.hint" class="mdi mdi-information-outline muted" /></span>
              <span class="u-bar"><span :style="{ width: `${Math.max(u.frac * 100, u.bytes ? 1 : 0)}%` }" /></span>
              <span class="u-size mono">{{ fmtBytes(u.bytes) || '0' }}</span>
            </template>
          </div>
        </div>

        <div class="card">
          <div class="tbl-head">
            <span class="eyebrow">Runs</span>
            <span class="muted small">{{ store.runs.length }}</span>
            <span v-if="anyActive" class="muted small">cleanup waits until training finishes</span>
            <div class="run-search"><ZenInput v-model="runQuery" size="sm" placeholder="Search runs" /></div>
            <button
              v-if="sweepable.length"
              class="chip-btn quiet"
              :disabled="anyActive"
              :title="anyActive ? 'Wait for training to finish' : 'Delete the resume states of every finished run (asks first; stopped runs keep theirs)'"
              @click="cleanAllStates"
            >
              <i class="mdi mdi-broom" /> Delete all finished runs' states · {{ fmtBytes(sweepable.reduce((n, r) => n + r.states, 0)) }}
            </button>
          </div>
          <div class="tbl-scroll"><table class="tbl runs-tbl">
            <thead>
              <tr>
                <th v-for="c in ([['name', 'Run'], ['checkpoints', 'Checkpoints'], ['states', 'Resume states'], ['samples', 'Samples'], ['total', 'Total']] as const)" :key="c[0]">
                  <button class="sort" :class="{ on: runSort === c[0] }" :title="c[0] === 'name' ? 'Sort by name' : `Sort by ${c[1].toLowerCase()}, biggest first`" @click="runSort = c[0]">
                    {{ c[1] }}<i v-if="runSort === c[0]" :class="c[0] === 'name' ? 'mdi mdi-arrow-up' : 'mdi mdi-arrow-down'" />
                  </button>
                </th>
                <th />
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in runRows" :key="r.run_id">
                <td><RouterLink :to="`/runs/${r.run_id}`" class="name">{{ r.name }}</RouterLink><span class="muted small"> · {{ r.status }}</span></td>
                <td class="mono">{{ r.checkpoint_count ? `${r.checkpoint_count} · ${fmtBytes(r.checkpoints)}` : '–' }}</td>
                <td class="mono">{{ r.states ? fmtBytes(r.states) : '–' }}</td>
                <td class="mono">{{ r.samples ? fmtBytes(r.samples) : '–' }}</td>
                <td class="mono">{{ runSize(r) ? fmtBytes(runSize(r)) : '–' }}</td>
                <td class="act">
                  <button
                    v-if="r.states"
                    class="chip-btn quiet"
                    :disabled="anyActive || RESUMABLE.includes(r.status)"
                    :title="RESUMABLE.includes(r.status) ? 'This run can still be resumed: it needs these' : anyActive ? 'Wait for training to finish' : 'Delete the optimizer states; the LoRAs stay'"
                    @click="clean({ kind: 'states', run_id: r.run_id }, `Delete ${r.name}'s resume states (${fmtBytes(r.states)})`, 'The LoRA checkpoints and samples stay. The run can no longer be resumed from its checkpoints (it is finished, so that only matters for Train more, which starts fresh anyway).')"
                  >
                    Delete states
                  </button>
                </td>
              </tr>
            </tbody>
          </table></div>
          <div v-if="runsListed.length > RUNS_SHOWN || emptyRuns || (runQuery.trim() && !runsFound.length)" class="tbl-foot muted small">
            <template v-if="runQuery.trim() && !runsFound.length">No runs match “{{ runQuery.trim() }}”.</template>
            <template v-else>
              <button v-if="!allRuns && !runQuery.trim() && runsListed.length > RUNS_SHOWN" class="more-link" @click="allRuns = true">Show all {{ runsListed.length }}</button>
              <button v-else-if="allRuns && !runQuery.trim() && runsListed.length > RUNS_SHOWN" class="more-link" @click="allRuns = false">Show the first {{ RUNS_SHOWN }}</button>
              <template v-if="emptyRuns">
                <span v-if="runsListed.length > RUNS_SHOWN && !runQuery.trim()"> · </span>
                {{ emptyRuns }} keep nothing on disk:
                <button class="more-link" @click="emptyShown = !emptyShown">{{ emptyShown ? 'hide them' : 'show them' }}</button>
              </template>
            </template>
          </div>
        </div>

        <div class="card">
          <div class="tbl-head"><span class="eyebrow">Datasets</span></div>
          <div class="tbl-scroll"><table class="tbl">
            <thead>
              <tr><th>Dataset</th><th>Images</th><th>Caches</th><th /></tr>
            </thead>
            <tbody>
              <tr v-for="d in store.datasets" :key="d.folder">
                <td><RouterLink :to="`/datasets/${d.folder}`" class="name">{{ d.folder }}</RouterLink></td>
                <td class="mono">{{ d.count }} · {{ fmtBytes(d.images) }}</td>
                <td class="mono">{{ d.cache ? fmtBytes(d.cache) : '–' }}</td>
                <td class="act">
                  <button
                    v-if="d.cache"
                    class="chip-btn quiet"
                    :disabled="anyActive"
                    :title="anyActive ? 'Wait for training to finish' : 'Delete the latents and text embeddings training cached from this dataset'"
                    @click="clean({ kind: 'cache', folder: d.folder }, `Clear ${d.folder}'s caches (${fmtBytes(d.cache)})`, 'Your images and captions stay. The next run on this dataset re-encodes them, which takes a little longer.')"
                  >
                    Clear caches
                  </button>
                </td>
              </tr>
            </tbody>
          </table></div>
        </div>
      </template>
    </section>

    <div v-if="restarting" class="overlay"><i class="mdi mdi-loading mdi-spin" /> Restarting ComfyUI…</div>
  </div>
</template>

<style scoped>
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.big {
  font-size: 22px;
  font-weight: 650;
}
.big .muted {
  font-size: 13px;
  font-weight: 400;
}
.small {
  font-size: 11px;
}
.storage {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 8px;
}
.st-head {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 32px;
}
.st-head h2 {
  margin: 0;
  font-size: 18px;
}
.note {
  min-width: 14ch;
  text-align: right;
  color: var(--zen-muted);
}
.disks {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.disk {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.meter {
  height: 6px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.meter span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
.meter span.hot {
  background: var(--zen-danger);
}
.usage {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr) 9ch;
  gap: 8px 14px;
  align-items: center;
  margin-top: 10px;
  font-size: 12px;
}
.u-label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.u-bar {
  height: 8px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  overflow: hidden;
}
.u-bar span {
  display: block;
  height: 100%;
  background: color-mix(in srgb, var(--zen-accent) 75%, var(--zen-surface-2));
}
.u-size {
  text-align: right;
}
.tbl-head {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 30px;
}
.tbl-head .run-search {
  margin-left: auto;
  width: 220px;
}
.tbl-head .run-search + .chip-btn {
  margin-left: 0;
}
.tbl-head .chip-btn {
  margin-left: auto;
}
/* a phone: the tables keep their columns and scroll sideways, rather than squeeze into overlap */
.tbl-scroll {
  overflow-x: auto;
}
@media (max-width: 720px) {
  .tbl-scroll > .tbl {
    min-width: 640px;
  }
}
/* the runs table keeps its columns where they are when it's sorted or searched */
.runs-tbl {
  table-layout: fixed;
}
.runs-tbl th:nth-child(1) {
  width: 34%;
}
.runs-tbl th:nth-child(n + 2):nth-child(-n + 5) {
  width: 13%;
}
.tbl th .sort {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  color: inherit;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 3px;
}
.tbl th .sort:hover,
.tbl th .sort.on {
  color: var(--zen-text);
}
.tbl-foot {
  padding: 10px 8px 2px;
}
.more-link {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  color: var(--zen-accent);
  cursor: pointer;
}
.more-link:hover {
  text-decoration: underline;
}
.tbl {
  width: 100%;
  margin-top: 8px;
  border-collapse: collapse;
  font-size: 12px;
}
.tbl th {
  text-align: left;
  font-weight: 500;
  font-size: 11px;
  color: var(--zen-muted);
  padding: 6px 8px;
}
.tbl td {
  padding: 7px 8px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
  white-space: nowrap;
}
.tbl .name {
  font-weight: 600;
}
.tbl .name:hover {
  color: var(--zen-accent);
}
.tbl .act {
  text-align: right;
  width: 1%;
}
.chip-btn {
  height: 28px;
  padding: 0 10px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.chip-btn.quiet {
  color: var(--zen-muted);
}
.chip-btn.quiet:hover:not(:disabled) {
  color: var(--zen-danger);
  border-color: color-mix(in srgb, var(--zen-danger) 50%, transparent);
}
.chip-btn:disabled {
  opacity: 0.4;
  cursor: default;
}
.err {
  border-color: var(--zen-danger);
  font-size: 12px;
}
.overlay {
  position: fixed;
  inset: 0;
  z-index: 70;
  display: grid;
  place-items: center;
  font-size: 16px;
  font-weight: 600;
  background: color-mix(in srgb, var(--zen-bg) 80%, transparent);
  backdrop-filter: blur(4px);
}
</style>
