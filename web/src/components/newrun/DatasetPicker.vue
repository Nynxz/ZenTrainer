<script setup lang="ts">
// New run's dataset picker, as Recipes is laid out: the datasets on the left (starred first, a
// search, sorted by recent, name or size) and the one highlighted on the right, its pictures and
// what's in it, before Use puts it in the run. Stars follow you (ComfyUI's per-user settings).
import { ZenButton, ZenInput, ZenModal } from '@nynxz/zenkit-ui'
import { computed, ref, watch } from 'vue'
import { thumbUrl, type DatasetInfo } from '@/api/engine'
import CaptionRing from '@/components/CaptionRing.vue'
import ScrollArea from '@/components/ScrollArea.vue'
import StarButton from '@/components/StarButton.vue'
import RunDatasetLightbox from '@/components/run/RunDatasetLightbox.vue'
import { isDatasetStarred } from '@/stores/favorites'

const props = defineProps<{ datasets: DatasetInfo[]; current: string }>()
const emit = defineEmits<{ pick: [folder: string] }>()
const open = defineModel<boolean>('open', { default: false })

const query = ref('')
type Sort = 'recent' | 'name' | 'size'
const sort = ref<Sort>('recent')
const picked = ref<string>(props.current)
watch(open, (o) => {
  if (o) (picked.value = props.current || props.datasets[0]?.folder || ''), (query.value = '')
})

const sorted = computed(() => {
  const q = query.value.trim().toLowerCase()
  const by: Record<Sort, (a: DatasetInfo, b: DatasetInfo) => number> = {
    recent: (a, b) => (b.modified ?? 0) - (a.modified ?? 0),
    name: (a, b) => a.folder.localeCompare(b.folder),
    size: (a, b) => b.images - a.images,
  }
  return props.datasets.filter((d) => !q || d.folder.toLowerCase().includes(q)).sort(by[sort.value])
})
// the list under Starred says how it's sorted, starred or not
const SORT_LABEL: Record<Sort, string> = { recent: 'Recently changed', name: 'A to Z', size: 'Most pictures' }
const groups = computed(() => {
  const starred = sorted.value.filter((d) => isDatasetStarred(d.folder))
  const rest = sorted.value.filter((d) => !isDatasetStarred(d.folder))
  return [
    ...(starred.length ? [{ key: 'starred', label: 'Starred', items: starred }] : []),
    ...(rest.length ? [{ key: 'all', label: SORT_LABEL[sort.value], items: rest }] : []),
  ]
})
const shown = computed(() => props.datasets.find((d) => d.folder === picked.value) ?? null)
// its pictures in the lightbox, over the picker: '' opens on the first, a path on that one
const gallery = ref<string | null>(null)
const relPath = (t: DatasetInfo['thumbs'][number]) => t.path ?? t.subfolder.split('/').slice(1).concat(t.filename).join('/')

const thumb = (d: DatasetInfo, t: DatasetInfo['thumbs'][number], size = 256) =>
  thumbUrl(d.folder, t.path ?? t.subfolder.split('/').slice(1).concat(t.filename).join('/'), size)
const fmtBytes = (b?: number) => (b == null ? '–' : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(b / 1e6))} MB`)
const ago = (t?: number) => {
  if (!t) return '–'
  const s = Date.now() / 1000 - t
  return s < 3600 ? `${Math.max(1, Math.round(s / 60))} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} days ago`
}
const facts = computed(() => {
  const d = shown.value
  if (!d) return []
  return [
    { k: 'Pictures', v: d.videos ? `${d.images - d.videos} · ${d.videos} videos` : String(d.images) },
    { k: 'Captioned', v: `${d.captioned} of ${d.images}` },
    { k: 'Caption sets', v: d.sets.join(', ') || 'default' },
    { k: 'Masks', v: d.masked ? String(d.masked) : 'none' },
    ...(d.pairs ? [{ k: 'Edit pairs', v: String(d.pairs) }] : []),
    { k: 'Size', v: fmtBytes(d.bytes) },
    { k: 'Changed', v: ago(d.modified) },
  ]
})

function use(folder = picked.value) {
  if (!folder) return
  emit('pick', folder)
  open.value = false
}
</script>

<template>
  <ZenModal v-model:open="open" width="980px" height="min(660px, 86vh)">
    <template #header>
      <div class="head">
        <div class="head-text">
          <b class="zt-title">Datasets</b>
          <span class="muted">What this run trains on: star the ones you use</span>
        </div>
        <div class="view-switch text" title="Sort">
          <button :class="{ on: sort === 'recent' }" @click="sort = 'recent'">Recent</button>
          <button :class="{ on: sort === 'name' }" @click="sort = 'name'">A–Z</button>
          <button :class="{ on: sort === 'size' }" @click="sort = 'size'">Size</button>
        </div>
        <div class="ds-search"><ZenInput v-model="query" size="sm" placeholder="Search datasets" /></div>
      </div>
    </template>

    <div class="picker">
      <ScrollArea class="list">
        <nav class="list-in">
          <section v-for="g in groups" :key="g.key" class="group">
            <header class="g-head"><i v-if="g.key === 'starred'" class="mdi mdi-star" /> {{ g.label }}</header>
            <!-- a row, not a button: the star inside it is one -->
            <div
              v-for="d in g.items"
              :key="d.folder"
              role="button"
              tabindex="0"
              class="item"
              :class="{ on: picked === d.folder }"
              :aria-pressed="picked === d.folder"
              @click="picked = d.folder"
              @dblclick="use(d.folder)"
              @keydown.enter.prevent="picked === d.folder ? use(d.folder) : (picked = d.folder)"
            >
              <span class="thumb"><img v-if="d.thumbs[0]" :src="thumb(d, d.thumbs[0], 96)" alt="" loading="lazy" /><i v-else class="mdi mdi-image-off-outline" /></span>
              <span class="item-text">
                <span class="item-title">{{ d.folder }}<span v-if="d.folder === current" class="here">in this run</span></span>
                <span class="item-blurb">{{ d.images }} pictures · {{ d.sets.length || 1 }} caption set{{ (d.sets.length || 1) === 1 ? '' : 's' }}</span>
              </span>
              <StarButton :id="d.folder" :name="d.folder" kind="dataset" />
            </div>
          </section>
          <div v-if="!groups.length" class="empty muted small">
            No dataset matches “{{ query }}”.
          </div>
        </nav>
      </ScrollArea>

      <ScrollArea class="detail">
        <article v-if="shown" class="detail-in">
          <!-- always two rows of four, empty tiles for a smaller set: the panel doesn't jump between datasets -->
          <div class="mosaic">
            <button v-for="t in shown.thumbs.slice(0, 8)" :key="t.filename" type="button" class="pic" title="See it big, with its caption" @click="gallery = relPath(t)">
              <img :src="thumb(shown, t, 320)" alt="" loading="lazy" />
            </button>
            <span v-for="n in Math.max(0, 8 - shown.thumbs.slice(0, 8).length)" :key="`gap${n}`" class="gap"><i v-if="!shown.thumbs.length && n === 1" class="mdi mdi-image-off-outline" /></span>
          </div>
          <div class="title-row">
            <h3 class="zt-title">{{ shown.folder }}</h3>
            <span class="ring" :title="shown.health ? 'Its health, from the last check' : shown.captioned < shown.images ? 'How much is captioned' : ''">
              <CaptionRing v-if="shown.health || shown.captioned < shown.images" :done="shown.captioned" :total="shown.images" :scores="shown.health?.scores" :size="34" />
            </span>
            <span class="spacer" />
            <button type="button" class="browse" @click="gallery = ''"><i class="mdi mdi-image-multiple-outline" /> Browse {{ shown.images }} pictures</button>
            <RouterLink :to="`/datasets/${encodeURIComponent(shown.folder)}`" class="open" @click="open = false">View dataset <i class="mdi mdi-arrow-right" /></RouterLink>
          </div>
          <dl class="facts">
            <div v-for="f in facts" :key="f.k"><dt>{{ f.k }}</dt><dd class="mono">{{ f.v }}</dd></div>
          </dl>
          <p v-if="shown.captioned < shown.images" class="small warn"><i class="mdi mdi-alert-outline" /> {{ shown.images - shown.captioned }} pictures have no caption yet: auto-caption them on the dataset's page first.</p>
        </article>
        <div v-else class="empty muted small">Pick a dataset on the left.</div>
      </ScrollArea>
      <RunDatasetLightbox v-if="gallery !== null && shown" :name="shown.folder" :set="shown.sets[0] ?? 'default'" :start="gallery || undefined" @close="gallery = null" />
    </div>

    <template #footer>
      <div class="foot">
        <span class="muted small">Double-click a dataset, or Enter twice, to use it</span>
        <span class="spacer" />
        <ZenButton variant="primary" icon="mdi mdi-check" :disabled="!shown" @click="use()">{{ shown ? `Use ${shown.folder}` : 'Use' }}</ZenButton>
      </div>
    </template>
  </ZenModal>
</template>

<style scoped>
/* ZenModal's body scrolls by itself; here it doesn't: the two panes do (as in Recipes) */
:global(.zen-modal:has(.picker) .zm-body) {
  padding: 0;
  overflow: hidden;
}
.head,
.foot {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  min-width: 0;
}
.head-text {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
  margin-right: auto;
}
.head-text b {
  font-size: 15px;
}
.head-text .muted {
  font-size: 12px;
}
.ds-search {
  width: 200px;
}
.picker {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  height: 100%;
  min-height: 0;
}
.list,
.detail {
  min-height: 0;
}
.list {
  border-right: 1px solid var(--zen-border);
  background: color-mix(in srgb, var(--zen-surface-2) 55%, transparent);
}
.list-in {
  padding: 10px 8px 16px;
}
.group + .group {
  margin-top: 10px;
}
.g-head {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted);
}
.g-head .mdi-star {
  color: var(--zen-accent);
}
.item {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 6px 8px;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
  position: relative;
}
.item:hover {
  background: color-mix(in srgb, var(--zen-text) 6%, transparent);
}
.item.on {
  background: color-mix(in srgb, var(--zen-accent) 14%, transparent);
}
.item.on::before {
  content: '';
  position: absolute;
  left: -8px;
  top: 8px;
  bottom: 8px;
  width: 3px;
  border-radius: var(--r-sm);
  background: var(--zen-accent);
}
.thumb {
  width: 40px;
  height: 40px;
  flex: none;
  border-radius: var(--r-md);
  overflow: hidden;
  display: grid;
  place-items: center;
  background: var(--zen-surface-2);
  color: var(--zen-muted);
}
.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.item-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}
.item-title {
  font-weight: 600;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.item-blurb {
  font-size: 11px;
  color: var(--zen-muted);
}
.here {
  font-size: 10px;
  font-weight: 500;
  color: var(--zen-accent);
}
.detail-in {
  padding: 16px 20px 20px;
}
/* up to eight of its pictures, whole rows of four */
.mosaic {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 4px;
  border-radius: var(--r-lg);
  overflow: hidden;
  background: var(--zen-surface-2);
}
.mosaic .pic {
  display: block;
  padding: 0;
  border: none;
  background: none;
  cursor: zoom-in;
}
.mosaic img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  display: block;
  transition: filter 0.12s ease;
}
.mosaic .pic:hover img {
  filter: brightness(1.12);
}
.mosaic .gap {
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  background: color-mix(in srgb, var(--zen-surface-2) 70%, transparent);
  color: var(--zen-muted);
  font-size: 24px;
}
.title-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 14px 0 10px;
  min-height: 36px; /* the ring's room, whether a dataset has one or not */
}
.ring {
  width: 34px;
  height: 34px;
  flex: none;
}
.title-row h3 {
  margin: 0;
  font-size: 22px;
}
/* an action here (the lightbox) as a button; leaving for the dataset's page as a quieter link */
.browse {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  background: var(--zen-surface);
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.browse:hover {
  border-color: color-mix(in srgb, var(--zen-accent) 50%, var(--zen-border));
}
.open {
  font-size: 12px;
  color: var(--zen-muted);
  display: inline-flex;
  align-items: center;
  gap: 2px;
  margin-left: 6px;
}
.open:hover {
  color: var(--zen-text);
}
.facts {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 8px;
  margin: 0;
}
.facts > div {
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  padding: 8px 10px;
}
.facts dt {
  font-size: 11px;
  color: var(--zen-muted);
}
.facts dd {
  margin: 2px 0 0;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.warn {
  color: var(--zen-warn, #d9a400);
  margin-top: 12px;
}
.empty {
  padding: 20px;
}
@media (max-width: 720px) {
  .picker {
    grid-template-columns: 1fr;
  }
  .detail {
    display: none;
  }
}
</style>
