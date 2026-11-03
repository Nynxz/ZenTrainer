<script setup lang="ts">
// Gallery: every image the apps made, newest first, grouped by day. Search the prompts, narrow
// to a model; open one for its settings, then Reuse (back to the app with them), copy the
// prompt, download, or delete it.
import { ZenCombo, ZenEmpty, ZenInput, ZenLightbox } from '@nynxz/zenkit-ui'
import type { LightboxItem } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { engine, viewUrl, type AppInfo, type AppInput, type GalleryItem, type ModelFamily } from '@/api/engine'
import { ask } from '@/lib/ask'

const route = useRoute()
const router = useRouter()
const items = ref<GalleryItem[]>([])
const apps = ref<AppInfo[]>([])
const families = ref<ModelFamily[]>([])
const loaded = ref(false)
const error = ref<string | null>(null)
const query = ref('')
const model = ref('')
const appFilter = computed(() => (typeof route.query.app === 'string' ? route.query.app : ''))

onMounted(async () => {
  try {
    const [g, a, f] = await Promise.all([engine.gallery(appFilter.value || undefined, 1000), engine.apps(), engine.families().catch(() => [])])
    items.value = g.items
    apps.value = a.apps
    families.value = f
  } catch (e) {
    error.value = String(e)
  } finally {
    loaded.value = true
  }
})

// which of an app's inputs is the prompt, the model, the size...: by component kind
const appOf = (id: string) => apps.value.find((a) => a.id === id)
const kind = (i: AppInput) => i.component ?? i.type
const inputOf = (it: GalleryItem, k: string) => appOf(it.app)?.inputs.find((i) => kind(i) === k) ?? null
const valueOf = (it: GalleryItem, k: string) => {
  const i = inputOf(it, k)
  return i ? it.values[i.id] : undefined
}
const promptOf = (it: GalleryItem) => String(valueOf(it, 'prompt') ?? '')
const familyOf = (it: GalleryItem) => String(valueOf(it, 'family') ?? '')
const familyName = (id: string) => families.value.find((f) => f.id === id)?.name ?? id
const sizeOf = (it: GalleryItem) => {
  const v = valueOf(it, 'size') ?? valueOf(it, 'resolution')
  return Array.isArray(v) ? v.join(' × ') : ''
}

const modelItems = computed(() => [
  { value: '', label: 'All models', keywords: 'all' },
  ...[...new Set(items.value.map(familyOf).filter(Boolean))].map((id) => ({ value: id, label: familyName(id), keywords: familyName(id) })),
])
const shown = computed(() => {
  const q = query.value.trim().toLowerCase()
  return items.value.filter((it) => (!model.value || familyOf(it) === model.value) && (!q || promptOf(it).toLowerCase().includes(q)))
})
const days = computed(() => {
  const out = new Map<string, { it: GalleryItem; n: number }[]>()
  shown.value.forEach((it, n) => {
    const d = new Date(it.created * 1000).toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })
    out.set(d, [...(out.get(d) ?? []), { it, n }])
  })
  return [...out.entries()]
})

// ---- the viewer -------------------------------------------------------------------------------
const index = ref<number | null>(null)
const lbItems = computed<LightboxItem[]>(() =>
  shown.value.map((it) => ({ src: viewUrl(it.image), label: promptOf(it) || it.image.filename, meta: [familyName(familyOf(it)), sizeOf(it)].filter(Boolean).join(' · ') })),
)
// the settings worth showing, in the app's own order and labels
function details(it: GalleryItem) {
  const app = appOf(it.app)
  const rows: [string, string][] = []
  for (const i of app?.inputs ?? []) {
    const v = it.values[i.id]
    const k = kind(i)
    if (v == null || v === '' || k === 'prompt') continue
    if (k === 'family') rows.push(['Model', familyName(String(v))])
    else if (k === 'size' || k === 'resolution') Array.isArray(v) && rows.push(['Size', v.join(' × ')])
    else if (k === 'loras') {
      const on = (v as { name: string; strength: number; on?: boolean }[]).filter((l) => l.name && l.on !== false)
      if (on.length) rows.push(['LoRAs', on.map((l) => `${l.name.split('/').pop()} · ${l.strength}`).join('\n')])
    } else if (k === 'seed' && v === -1) continue
    else rows.push([i.label ?? i.id, String(v)])
  }
  return rows
}

function reuse(it: GalleryItem) {
  router.push(`/apps/${it.app}?reuse=${encodeURIComponent(it.batch_id)}/${encodeURIComponent(it.prompt_id)}`)
}
const copied = ref(false)
async function copy(it: GalleryItem) {
  try {
    await navigator.clipboard.writeText(promptOf(it))
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch {
    // clipboard blocked
  }
}
async function remove(it: GalleryItem) {
  if (!(await ask({ title: 'Delete this image?', message: 'The file is removed from the output folder.', ok: 'Delete', danger: true }))) return
  try {
    await engine.deleteGalleryItem(it.batch_id, it.prompt_id)
    const pos = shown.value.indexOf(it)
    items.value = items.value.filter((x) => !(x.batch_id === it.batch_id && x.prompt_id === it.prompt_id))
    if (index.value != null) index.value = shown.value.length ? Math.min(pos, shown.value.length - 1) : null
  } catch (e) {
    error.value = String(e)
  }
}
</script>

<template>
  <div class="page full">
    <div class="page-head">
      <div class="titles">
        <h1>Gallery</h1>
        <p class="lede">
          {{ items.length }} image{{ items.length === 1 ? '' : 's' }} <template v-if="appFilter">
            in <b>{{ appOf(appFilter)?.name ?? appFilter }}</b> · <RouterLink to="/gallery" class="link">show all</RouterLink></template
          >
        </p>
      </div>
      <div class="filters">
        <ZenInput v-model="query" placeholder="Search prompts" icon="mdi mdi-magnify" clearable class="search" />
        <ZenCombo v-model="model" :items="modelItems" :sorted="false" class="model" />
      </div>
    </div>
    <div v-if="error" class="card err mono small">{{ error }}</div>
    <ZenEmpty v-if="loaded && !items.length" title="Nothing here yet" icon="mdi mdi-image-multiple-outline">
      Images you make in <RouterLink to="/apps"><u>Apps</u></RouterLink> collect here.
    </ZenEmpty>
    <ZenEmpty v-else-if="loaded && !shown.length" title="No matches" icon="mdi mdi-magnify" />

    <section v-for="[day, list] in days" :key="day" class="day">
      <div class="eyebrow">{{ day }} <span class="muted">· {{ list.length }}</span></div>
      <div class="grid">
        <button v-for="{ it, n } in list" :key="`${it.batch_id}/${it.prompt_id}/${it.image.filename}`" class="tile" :title="promptOf(it)" @click="index = n">
          <img :src="viewUrl(it.image)" alt="" loading="lazy" />
          <span class="cap small">{{ familyName(familyOf(it)) }}</span>
        </button>
      </div>
    </section>

    <ZenLightbox v-if="index != null && lbItems.length" v-model:index="index" :items="lbItems" :searchable="false" @close="index = null">
      <template #side="{ index: i }">
        <div v-if="shown[i]" class="side">
          <div v-if="promptOf(shown[i]!)" class="prompt">{{ promptOf(shown[i]!) }}</div>
          <dl>
            <template v-for="[k, v] in details(shown[i]!)" :key="k">
              <dt>{{ k }}</dt>
              <dd>{{ v }}</dd>
            </template>
            <dt>Made</dt>
            <dd>{{ new Date(shown[i]!.created * 1000).toLocaleString() }}</dd>
            <dt>App</dt>
            <dd>{{ appOf(shown[i]!.app)?.name ?? shown[i]!.app }}</dd>
          </dl>
          <div class="side-acts">
            <button class="sbtn primary" title="Open the app with these settings" @click="reuse(shown[i]!)"><i class="mdi mdi-backup-restore" /> Reuse settings</button>
            <button class="sbtn" :title="copied ? 'Copied' : 'Copy the prompt'" @click="copy(shown[i]!)"><i :class="copied ? 'mdi mdi-check' : 'mdi mdi-content-copy'" /></button>
            <a class="sbtn" :href="viewUrl(shown[i]!.image)" :download="shown[i]!.image.filename" title="Download"><i class="mdi mdi-download" /></a>
          </div>
          <!-- out of the way: not beside Reuse, nowhere near the viewer's close button -->
          <button class="del" @click="remove(shown[i]!)"><i class="mdi mdi-trash-can-outline" /> Delete image</button>
        </div>
      </template>
    </ZenLightbox>
  </div>
</template>

<style scoped>
.filters {
  display: flex;
  gap: 8px;
  align-items: center;
}
.search {
  width: 260px;
}
.model {
  width: 200px;
}
.day {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
  gap: 8px;
}
.tile {
  position: relative;
  aspect-ratio: 1;
  padding: 0;
  border: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
  border-radius: var(--r-lg);
  background: var(--zen-surface);
  overflow: hidden;
  cursor: zoom-in;
}
.tile img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform var(--zen-dur, 0.3s) ease;
}
.tile:hover {
  border-color: var(--zen-accent);
}
.tile:hover img {
  transform: scale(1.03);
}
.cap {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 16px 8px 6px;
  text-align: left;
  color: #fff;
  background: linear-gradient(transparent, rgb(0 0 0 / 0.6));
  opacity: 0;
  transition: opacity var(--zen-dur-fast, 0.15s) ease;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tile:hover .cap {
  opacity: 1;
}
.side {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.side-acts {
  display: flex;
  gap: 6px;
}
.sbtn {
  height: 32px;
  min-width: 32px;
  padding: 0 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  border-radius: var(--r-md);
  background: var(--zen-surface);
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  text-decoration: none;
}
.sbtn.primary {
  flex: 1;
  background: var(--zen-accent);
  border-color: var(--zen-accent);
  color: var(--zen-accent-text, #fff);
}
.sbtn:not(.primary):hover {
  background: var(--zen-surface-2);
}
.del {
  align-self: flex-start;
  margin-top: 8px;
  padding: 0;
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.del:hover {
  color: var(--zen-danger);
}
.prompt {
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
  user-select: text;
}
dl {
  margin: 0;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 12px;
  font-size: 12px;
}
dt {
  color: var(--zen-muted);
}
dd {
  margin: 0;
  white-space: pre-line;
  overflow-wrap: anywhere;
  user-select: text;
}
.link {
  color: var(--zen-accent);
}
.err {
  color: var(--zen-danger);
}
@media (max-width: 720px) {
  .filters {
    width: 100%;
  }
  .search,
  .model {
    flex: 1;
    width: auto;
  }
}
</style>
