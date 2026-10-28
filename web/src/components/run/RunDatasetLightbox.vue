<script setup lang="ts">
// A dataset in the lightbox, read-only: over the run page (the Setup card's Dataset line), or New
// run's dataset picker. Each picture with its caption in the run's caption set (`start`: the picture
// it opens on); the dataset page edits them.
import { ZenLightbox, type LightboxItem } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref } from 'vue'
import { engine, thumbUrl, viewUrl, type DatasetItem } from '@/api/engine'
import { kb } from '@/components/dataset/context'

const props = defineProps<{ name: string; set: string; start?: string }>()
const emit = defineEmits<{ close: [] }>()

const items = ref<DatasetItem[]>([])
const error = ref<string | null>(null)
const index = ref(0)
onMounted(async () => {
  try {
    items.value = (await engine.dataset(props.name)).items
    if (props.start) index.value = Math.max(0, items.value.findIndex((it) => it.path === props.start))
    if (!items.value.length) error.value = 'This dataset has no pictures now.'
  } catch (e) {
    error.value = String((e as Error).message ?? e)
  }
})

const thumb = (path: string, size: number) => thumbUrl(props.name, path, size)
const lbItems = computed<LightboxItem[]>(() =>
  items.value.map((it) => ({
    src: viewUrl({ ...it, type: 'input' }),
    kind: it.kind === 'video' ? ('video' as const) : undefined,
    thumb: thumb(it.path, 256),
    label: it.path,
    meta: it.width ? `${it.width}×${it.height} · ${kb(it.bytes)}` : kb(it.bytes),
    overlay: it.mask ? thumb(it.mask, 1024) : undefined,
  })),
)
const open = computed(() => ({ path: `/datasets/${encodeURIComponent(props.name)}`, query: props.set !== 'default' ? { set: props.set } : {} }))
</script>

<template>
  <ZenLightbox v-if="lbItems.length" v-model:index="index" :items="lbItems" @close="emit('close')">
    <template #side="{ index: i }">
      <div v-if="items[i]" class="side">
        <div class="eyebrow">Caption{{ set !== 'default' ? ` · ${set}` : '' }}</div>
        <p class="cap" :class="{ none: !items[i].captions?.[set] }">{{ items[i].captions?.[set] || 'No caption' }}</p>
        <div v-if="items[i].muted?.includes(set)" class="small warn">Muted in “{{ set }}”: not trained on</div>
        <div v-if="items[i].repeats > 1" class="small muted">Repeats ×{{ items[i].repeats }}</div>
        <RouterLink :to="open" class="small to"><i class="mdi mdi-image-multiple-outline" /> Edit on the dataset's page</RouterLink>
      </div>
    </template>
  </ZenLightbox>
  <div v-else-if="error" class="err small" role="alert">{{ error }} <button class="link" @click="emit('close')">Close</button></div>
</template>

<style scoped>
.side {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}
.cap {
  margin: 0;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  max-height: 55vh;
  overflow-y: auto;
  scrollbar-width: thin;
}
.cap.none {
  color: var(--zen-muted);
}
.warn {
  color: var(--zen-warn);
}
.to {
  color: var(--zen-muted);
  text-decoration: none;
}
.to:hover {
  color: var(--zen-accent);
}
.err {
  color: var(--zen-danger, var(--zen-warn));
}
.link {
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  cursor: pointer;
}
</style>
