<script setup lang="ts">
// Every model family ComfyUI ships a template for (plus Comfy-Org and model makers' ComfyUI
// repos): what you have of it, its repo, and the rest one click away. Newest first by default.
// Opening a family lists every build in its Hugging Face repo.
import { ZenEmpty, ZenInput } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { engine, type ModelFamily, type ModelFolder } from '@/api/engine'
import FileState from '@/components/FileState.vue'
import StarButton from '@/components/StarButton.vue'
import { isStarred, starred, starredFirst } from '@/stores/favorites'
import { finished, fmtAdded, fmtBytes, hfToken, isNew, startDownloadSync, targets } from '@/stores/downloads'
import StickyBar from '@/components/StickyBar.vue'

const families = ref<ModelFamily[]>([])
const loaded = ref(false)
const error = ref<string | null>(null)
const query = ref('')
const filter = ref<'all' | 'starred' | 'ready' | 'missing'>('all')
const sort = ref<'newest' | 'name'>('newest')
const open = ref<string | null>(null)
const detail = ref<Record<string, ModelFamily>>({})
const listing = ref<string | null>(null)

const FOLDERS: { folder: ModelFolder; label: string }[] = [
  { folder: 'checkpoints', label: 'Checkpoints' },
  { folder: 'diffusion_models', label: 'Diffusion models' },
  { folder: 'text_encoders', label: 'Text encoders' },
  { folder: 'vae', label: 'VAE' },
]

async function load() {
  try {
    families.value = await engine.families()
  } catch (e) {
    error.value = String(e)
  } finally {
    loaded.value = true
  }
}
async function expand(id: string, refresh = false) {
  open.value = open.value === id && !refresh ? null : id
  if (open.value !== id) return
  listing.value = id
  try {
    detail.value[id] = await engine.family(id, true, refresh)
  } finally {
    listing.value = null
  }
}
watch(finished, async () => {
  await load()
  if (open.value) detail.value[open.value] = await engine.family(open.value, true)
})

const ready = (f: ModelFamily) => !f.missing.length
const shown = computed(() => {
  const q = query.value.trim().toLowerCase()
  const list = families.value.filter(
    (f) =>
      (filter.value === 'all' || (filter.value === 'starred' ? isStarred(f.id) : (filter.value === 'ready') === ready(f))) &&
      (!q || `${f.name} ${f.clip_type ?? ''} ${f.repos.join(' ')}`.toLowerCase().includes(q)),
  )
  // the server sends newest first; starred ones lead either way
  return starredFirst(sort.value === 'name' ? [...list].sort((a, b) => a.name.localeCompare(b.name)) : list, (f) => f.id)
})
const counts = computed(() => ({ ready: families.value.filter(ready).length, missing: families.value.filter((f) => !ready(f)).length }))
const famOf = (id: string) => detail.value[id] ?? families.value.find((f) => f.id === id)!
const missingBytes = (f: ModelFamily) => f.missing.reduce((n, m) => n + (m.size ?? 0), 0)

onMounted(() => {
  startDownloadSync()
  load()
})
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="titles">
        <h1>Models</h1>
      </div>
    </div>

    <StickyBar>
    <div class="page-toolbar card">
      <div class="view-switch text">
        <button :class="{ on: filter === 'all' }" @click="filter = 'all'">All {{ families.length }}</button>
        <button :class="{ on: filter === 'starred' }" @click="filter = 'starred'"><i class="mdi mdi-star-outline" /> Starred {{ starred.size }}</button>
        <button :class="{ on: filter === 'ready' }" @click="filter = 'ready'">Ready {{ counts.ready }}</button>
        <button :class="{ on: filter === 'missing' }" @click="filter = 'missing'">Not installed {{ counts.missing }}</button>
      </div>
      <span class="spacer" />
      <div class="view-switch text" title="Sort">
        <button :class="{ on: sort === 'newest' }" @click="sort = 'newest'"><i class="mdi mdi-sort-calendar-descending" /> Newest</button>
        <button :class="{ on: sort === 'name' }" @click="sort = 'name'"><i class="mdi mdi-sort-alphabetical-ascending" /> A–Z</button>
      </div>
      <div class="search"><ZenInput v-model="query" placeholder="Search models" size="sm" /></div>
    </div>
    </StickyBar>

    <div v-if="error" class="card err mono">{{ error }}</div>
    <ZenEmpty v-if="loaded && !shown.length" title="No models match" icon="mdi mdi-cube-outline" />

    <div class="list">
      <section v-for="f in shown" :key="f.id" class="fam card" :class="{ open: open === f.id }">
        <div class="fam-row">
        <span class="star-slot"><StarButton :id="f.id" :name="f.name" /></span>
        <button class="fam-head" @click="expand(f.id)">
          <i class="mdi mdi-chevron-right chev" />
          <span class="name">{{ f.name }}</span>
          <i
            v-if="f.profile"
            class="mdi tick"
            :class="f.profile.status === 'verified' ? 'mdi-check-decagram' : 'mdi-check-circle-outline'"
            :title="f.profile.status === 'verified' ? 'Tuned: trained and checked with hand-picked settings' : `Hand-picked settings: ${f.profile.summary}`"
          />
          <span class="chip mono">{{ f.kind === 'checkpoint' ? 'checkpoint' : f.clip_type }}</span>
          <span v-if="isNew(f.added)" class="tag new" :title="`Added ${fmtAdded(f.added)}`">new</span>
          <span v-if="f.source === 'repo'" class="tag" title="ComfyUI has no example workflow for this model yet, so its files come straight from its Hugging Face repo, and there are no suggested sampler settings for previews">from repo</span>
          <span v-if="f.gated" class="tag" title="The repo asks you to accept its license before downloading"><i class="mdi mdi-lock-outline" /> gated</span>
          <span class="spacer" />
          <span class="added muted">{{ fmtAdded(f.added) }}</span>
          <span class="status" :class="{ ready: ready(f) }">
            <i :class="ready(f) ? 'mdi mdi-check-circle' : 'mdi mdi-download-outline'" />
            {{ ready(f) ? 'ready to train' : `${f.missing.length} to download${missingBytes(f) ? ` · ${fmtBytes(missingBytes(f))}` : ''}` }}
          </span>
        </button>
        </div>
        <div v-if="open === f.id" class="fam-body">
          <div class="links">
            <a v-for="r in f.repos" :key="r" :href="`https://huggingface.co/${r}`" target="_blank" rel="noopener" class="link">
              <i class="mdi mdi-open-in-new" /> {{ r }}
            </a>
            <span class="muted small">{{ f.source === 'repo' ? 'no ComfyUI example workflow yet: files read from the repo, no suggested sample settings' : `from ${f.templates.length} ComfyUI template${f.templates.length === 1 ? '' : 's'}` }}</span>
            <span class="spacer" />
            <span v-if="listing === f.id" class="muted small"><i class="mdi mdi-loading mdi-spin" /> listing builds…</span>
            <button v-else class="link small" title="Re-list the builds on Hugging Face" @click="expand(f.id, true)"><i class="mdi mdi-refresh" /> refresh</button>
          </div>
          <div v-if="f.gated && !ready(f)" class="gate small">
            <i class="mdi mdi-lock-outline" />
            <span>
              Gated: accept the license on
              <a :href="`https://huggingface.co/${f.repos[0]}`" target="_blank" rel="noopener">Hugging Face</a>
              <template v-if="!hfToken">, then set <span class="mono">HF_TOKEN</span> (a read token) and restart ZenTrainer</template>
              <template v-else> with the account your <span class="mono">HF_TOKEN</span> belongs to</template>.
            </span>
          </div>
          <template v-for="g in FOLDERS" :key="g.folder">
            <div v-if="famOf(f.id).files[g.folder]?.length" class="group">
              <div class="eyebrow">{{ g.label }}</div>
              <div v-for="file in famOf(f.id).files[g.folder]" :key="file.name" class="file">
                <span class="f-name mono" :title="file.name">{{ file.name }}</span>
                <span v-if="famOf(f.id).recommended.train[g.folder] === file.name" class="tag rec" title="What New Run picks for training">train</span>
                <span v-if="file.distilled" class="tag turbo" title="Distilled (turbo/lightning): fast to sample, not a good base to train on">turbo</span>
                <span v-if="file.from_template" class="tag" title="What ComfyUI's template loads">template</span>
                <span class="spacer" />
                <FileState :family="f.id" :file="file" @error="error = $event" />
              </div>
            </div>
          </template>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.page-toolbar .search {
  flex: 0 1 260px;
}
.chip-btn {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: var(--r-md);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.chip-btn.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  border-color: var(--zen-selected, var(--zen-accent));
}
.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.fam {
  padding: 0;
}
.fam-row {
  display: flex;
  align-items: center;
}
.star-slot {
  padding-left: 12px;
}
.fam-head {
  flex: 1;
  min-width: 0;
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  border: none;
  background: none;
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.chev {
  color: var(--zen-muted);
  transition: transform var(--zen-dur, 0.2s) ease;
}
.open .chev {
  transform: rotate(90deg);
}
.name {
  font-weight: 650;
}
.status {
  font-size: 12px;
  color: var(--zen-muted);
  white-space: nowrap;
}
.status.ready {
  color: var(--zen-ok);
}
.fam-body {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 0 16px 16px 40px;
}
.links {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  text-decoration: none;
}
.link:hover {
  color: var(--zen-accent);
}
.group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.group .eyebrow {
  margin-bottom: 4px;
}
.file {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 32px;
  padding: 2px 0;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 40%, transparent);
}
.f-name {
  font-size: 12px;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tag {
  font-size: 10px;
  padding: 0 6px;
  border-radius: var(--r-sm);
  background: var(--zen-surface-2);
  color: var(--zen-muted);
  white-space: nowrap;
}
.tick {
  color: var(--zen-accent);
  font-size: 15px;
  margin-left: -4px;
}
.tag.new {
  background: color-mix(in srgb, var(--zen-ok) 18%, transparent);
  color: var(--zen-ok);
}
.added {
  font-size: 11px;
  width: 8ch;
  text-align: right;
  white-space: nowrap;
}
.gate {
  display: flex;
  gap: 8px;
  align-items: baseline;
  padding: 8px 10px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-warn) 10%, transparent);
  color: var(--zen-text);
}
.gate i {
  color: var(--zen-warn);
}
.gate a {
  color: var(--zen-accent);
}
.tag.turbo {
  background: color-mix(in srgb, var(--zen-warn) 18%, transparent);
  color: var(--zen-warn);
}
.tag.rec {
  background: color-mix(in srgb, var(--zen-accent) 20%, transparent);
  color: var(--zen-text);
}
.small {
  font-size: 11px;
}
.err {
  border-color: var(--zen-danger);
  font-size: 12px;
}
</style>
