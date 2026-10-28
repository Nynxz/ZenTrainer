<script setup lang="ts">
// New run's model dialog: two views of one window, Models (the families, each with its recipes)
// and Recipes (every recipe, yours too, with what it changes). A view puts its own controls in the
// header beside the tabs and its buttons in the footer (Teleport, to the slots below); a recipe's
// Details on Models goes to it on Recipes, the window staying put.
import { ZenModal } from '@nynxz/zenkit-ui'
import ModelBrowser from './ModelBrowser.vue'
import RecipePicker from './RecipePicker.vue'

defineProps<{ current: string | null }>()
const emit = defineEmits<{ pick: [id: string]; recipe: [id: string] }>()
const open = defineModel<boolean>('open', { default: false })
const view = defineModel<'models' | 'recipes'>('view', { default: 'models' })
/** the recipe the Recipes view opens on (null: the first) */
const start = defineModel<string | null>('start', { default: null })

function details(id: string | null) {
  start.value = id
  view.value = 'recipes'
}
</script>

<template>
  <ZenModal v-model:open="open" width="1080px" height="min(720px, 88vh)">
    <template #header>
      <div class="md-head">
        <div class="view-switch text tabs" role="tablist" aria-label="View">
          <button role="tab" :aria-selected="view === 'models'" :class="{ on: view === 'models' }" @click="view = 'models'"><i class="mdi mdi-cube-outline" /> Models</button>
          <button role="tab" :aria-selected="view === 'recipes'" :class="{ on: view === 'recipes' }" @click="(start = null), (view = 'recipes')"><i class="mdi mdi-book-open-page-variant-outline" /> Recipes</button>
        </div>
        <div id="zt-md-head" class="md-slot" />
      </div>
    </template>

    <ModelBrowser v-if="view === 'models'" v-model:open="open" :current="current" @pick="emit('pick', $event)" @recipe="emit('recipe', $event)" @details="details" />
    <RecipePicker v-else v-model:open="open" :start="start" />

    <template #footer>
      <div id="zt-md-foot" class="md-slot" />
    </template>
  </ZenModal>
</template>

<style scoped>
.md-head {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;
  min-width: 0;
}
.tabs button {
  height: 30px;
  font-size: 13px;
}
/* where a view's controls and buttons go: the rest of the header, the whole footer */
.md-slot {
  display: flex;
  align-items: center;
  flex: 1;
  min-width: 0;
}
.md-slot > :deep(*) {
  flex: 1;
}
</style>
