<script setup lang="ts">
// "Save as recipe": a spec's settings and model as one of your recipes (lib/myRecipes.ts), from the
// run form or a run's page. Says what it keeps and what it leaves out, and catches the two ways a
// save repeats one you have: the same name (it replaces that one) or the same settings.
import { ZenButton, ZenField, ZenInput, ZenModal } from '@nynxz/zenkit-ui'
import { computed, ref, watch } from 'vue'
import type { TrainSpec } from '@/api/engine'
import { loadMyRecipes, myRecipes, recipeData, sameRecipe, saveMyRecipe } from '@/lib/myRecipes'

const props = defineProps<{ spec: TrainSpec | null; suggest?: string }>()
const open = defineModel<boolean>('open', { default: false })
const emit = defineEmits<{ saved: [name: string] }>()

const name = ref('')
const note = ref('')
const saving = ref(false)
const error = ref<string | null>(null)
watch(open, (o) => {
  if (!o) return
  void loadMyRecipes()
  name.value = props.suggest ?? ''
  note.value = ''
  error.value = null
})

const data = computed(() => (props.spec ? recipeData(props.spec) : null))
const slug = (n: string) => n.replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase() // spec.py slugify
const replacing = computed(() => {
  const n = name.value.trim()
  return n ? myRecipes.value.find((p) => p.id === slug(n) || p.name.toLowerCase() === n.toLowerCase()) ?? null : null
})
const same = computed(() => (data.value ? sameRecipe(data.value) : null))
const modelFile = computed(() => {
  const m = props.spec?.model
  return (m?.kind === 'checkpoint' ? m.ckpt_name : m?.unet_name)?.split('/').pop() ?? null
})

async function save() {
  if (!name.value.trim() || !data.value) return
  saving.value = true
  error.value = null
  try {
    await saveMyRecipe(name.value.trim(), note.value.trim(), data.value)
    emit('saved', name.value.trim())
    open.value = false
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <ZenModal v-model:open="open" title="Save as recipe" width="440px" height="auto">
    <div class="save-form">
      <ZenField label="Name" stack>
        <ZenInput v-model="name" placeholder="e.g. My character settings" @keydown.enter="save" />
      </ZenField>
      <ZenField label="Description" stack>
        <ZenInput v-model="note" placeholder="Optional: what it's for" />
      </ZenField>
      <p class="small muted">
        These settings<template v-if="modelFile"> and the model (<span class="mono">{{ modelFile }}</span>)</template>; not the
        run's name, dataset, trigger word or captions.
      </p>
      <p v-if="same && same.id !== replacing?.id" class="small warn">
        <i class="mdi mdi-content-duplicate" /> Your recipe “{{ same.name }}” has exactly these settings already.
      </p>
      <p v-if="replacing" class="small warn"><i class="mdi mdi-alert-outline" /> Replaces your recipe “{{ replacing.name }}”.</p>
      <p v-if="error" class="small warn"><i class="mdi mdi-alert-outline" /> {{ error }}</p>
    </div>
    <template #footer>
      <ZenButton @click="open = false">Cancel</ZenButton>
      <ZenButton variant="primary" icon="mdi mdi-content-save-outline" :disabled="!name.trim() || !data || saving" @click="save">
        {{ replacing ? 'Replace' : same ? 'Save a copy' : 'Save' }}
      </ZenButton>
    </template>
  </ZenModal>
</template>

<style scoped>
.save-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.save-form p {
  margin: 0;
  line-height: 1.45;
}
.warn {
  color: var(--zen-warn, #d9a400);
}
</style>
