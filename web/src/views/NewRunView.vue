<script setup lang="ts">
// The run form: presets on top (they layer), four numbered sections with the settings
// most runs touch, everything else under Advanced, and a sticky summary that says what
// you're about to do (and what looks off) next to the Start buttons. Model files, folders,
// samplers and schedulers come from ComfyUI's object_info; the rest is the harness's. The
// sections are in components/newrun/ and share the form through form.ts.
import { ZenButton, ZenMenuItem, ZenMenuSeparator, ZenPopover } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, provide, ref } from 'vue'
import { useRouter } from 'vue-router'
import { engine } from '@/api/engine'
import DatasetSection from '@/components/newrun/DatasetSection.vue'
import { RUN_FORM, useRunForm } from '@/components/newrun/form'
import MemorySection from '@/components/newrun/MemorySection.vue'
import ModelSection from '@/components/newrun/ModelSection.vue'
import RecipePicker from '@/components/newrun/RecipePicker.vue'
import SaveRecipe from '@/components/SaveRecipe.vue'
import RunSummary from '@/components/newrun/RunSummary.vue'
import SamplesSection from '@/components/newrun/SamplesSection.vue'
import SliderSection from '@/components/newrun/SliderSection.vue'
import TrainingSection from '@/components/newrun/TrainingSection.vue'
import { remembered } from '@/lib/remembered'
import { setSetting, settings } from '@/stores/settings'

const form = useRunForm()
provide(RUN_FORM, form)
const { spec, editing, retrying, resuming, resumeStep } = form
const { imported, exportSettings, copySettings, importSettings, recipeApplied, undoRecipe } = form
const recipesOpen = ref(false)
// Recipes opens from the model browser: on a recipe (its Details), or on the first (All recipes)
const recipeStart = ref<string | null>(null)
function openRecipe(id: string | null) {
  recipeStart.value = id
  recipesOpen.value = true
}

const saveOpen = ref(false) // these settings as one of your recipes (the recipes dialog lists them under Yours)
const openSave = () => (saveOpen.value = true)

// new here (no run has finished yet): a note that Simple mode exists, until it's closed
const router = useRouter()
const simpleHint = remembered('zentrainer.simpleHint', true)
const trainedOne = ref<boolean | null>(null) // null until the runs are in: no flash for people who have
onMounted(async () => {
  try {
    trainedOne.value = (await engine.runs()).some((r) => r.status === 'done')
  } catch {
    trainedOne.value = true // can't tell: don't nag
  }
})
const showSimpleHint = computed(() => simpleHint.value && trainedOne.value === false && !editing.value && settings.mode === 'advanced')
async function trySimple() {
  simpleHint.value = false
  await setSetting('mode', 'simple')
  await router.push('/simple')
}

// shared settings: a .zentrainer.json from Import, or its text pasted anywhere on the page
const picker = ref<HTMLInputElement | null>(null)
const copied = ref(false)
async function pick(e: Event) {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  input.value = ''
  if (f) await importSettings(await f.text())
}
async function copy() {
  copied.value = await copySettings()
  if (copied.value) setTimeout(() => (copied.value = false), 1500)
}
function onPaste(e: ClipboardEvent) {
  const t = e.target as HTMLElement | null
  if (resuming.value || (t && (t.closest('input, textarea, [contenteditable="true"]')))) return
  const text = e.clipboardData?.getData('text/plain')?.trim() ?? ''
  if (text.startsWith('{') && /"zentrainer-run"|"optim"\s*:/.test(text)) {
    e.preventDefault()
    void importSettings(text)
  }
}
onMounted(() => window.addEventListener('paste', onPaste))
onUnmounted(() => window.removeEventListener('paste', onPaste))
const missing = () => imported.value?.files.filter((f) => f.installed === false) ?? []
</script>

<template>
  <div class="page wide">
    <div class="page-head">
      <div class="titles">
        <h1>{{ editing ? (resuming ? 'Edit & resume' : retrying ? 'Edit & retry' : 'Edit queued run') : 'New run' }}</h1>
        <p v-if="editing" class="lede">
          {{ resuming ? `carries on from step ${resumeStep} with the new settings` : retrying ? 'goes back to the front of the queue' : 'keeps its place in the queue' }}
        </p>
        <p v-if="recipeApplied" class="lede from">
          From <b>{{ recipeApplied.title }}</b>
          <button class="undo" title="Put the settings back as they were before it" @click="undoRecipe"><i class="mdi mdi-undo" /> Undo</button>
        </p>
      </div>
      <span v-if="copied" class="copied small"><i class="mdi mdi-check" /> Copied</span>
      <ZenPopover placement="bottom-end">
        <template #trigger="{ toggle }">
          <button class="bar-btn icon" title="Save these settings as a recipe, share them, or load someone's" @click="toggle"><i class="mdi mdi-dots-horizontal" /></button>
        </template>
        <template #default="{ close }">
          <ZenMenuItem icon="mdi mdi-content-save-outline" @select="openSave(); close()">
            Save as recipe
            <template #hint>yours, in Recipes</template>
          </ZenMenuItem>
          <ZenMenuSeparator />
          <ZenMenuItem icon="mdi mdi-export-variant" @select="exportSettings(); close()">
            Export
            <template #hint>a file, without your dataset</template>
          </ZenMenuItem>
          <ZenMenuItem icon="mdi mdi-content-copy" @select="copy(); close()">
            Copy
            <template #hint>as text, for a chat</template>
          </ZenMenuItem>
          <ZenMenuItem v-if="!resuming" icon="mdi mdi-import" @select="picker?.click(); close()">
            Import
            <template #hint>a shared file</template>
          </ZenMenuItem>
        </template>
      </ZenPopover>
      <input ref="picker" type="file" accept=".json,application/json" hidden @change="pick" />
    </div>

    <div v-if="showSimpleHint" class="card simple-hint">
      <i class="mdi mdi-sprout-outline lead" />
      <div class="hint-text">
        <b>New to training?</b>
        <span class="muted">Simple mode trains from a recipe: pick what you're training, add your photos and a name. Settings switches back any time.</span>
      </div>
      <ZenButton size="sm" variant="primary" @click="trySimple">Try Simple mode</ZenButton>
      <button class="hint-close" title="Don't show this again" @click="simpleHint = false"><i class="mdi mdi-close" /></button>
    </div>

    <div v-if="imported" class="card imported small">
      <button class="close" title="Dismiss" @click="imported = null"><i class="mdi mdi-close" /></button>
      <div><i class="mdi mdi-import" /> Loaded <b>{{ imported.name || 'shared settings' }}</b>. Choose {{ imported.local.join(', ') }} here.</div>
      <div v-if="missing().length" class="warn">
        <i class="mdi mdi-alert-outline" /> Not installed here:
        <span v-for="f in missing()" :key="f.field" class="file">{{ f.name }} <span class="muted">({{ f.folder }})</span></span>
      </div>
      <div v-for="w in imported.warnings" :key="w" class="warn"><i class="mdi mdi-information-outline" /> {{ w }}</div>
    </div>

    <div class="layout">
      <div class="form">
        <div v-if="resuming" class="card locked-note small">
          <i class="mdi mdi-lock-outline" /> It has trained, so its model, dataset and LoRA shape (algorithm, rank, layers, blocks) and
          optimizer type stay as they were. Memory, samples, learning rate, schedule and length can change.
        </div>
        <ModelSection @recipe="openRecipe" />
        <DatasetSection />
        <!-- after the dataset: an image slider's other end is one of its picture sets -->
        <SliderSection />
        <TrainingSection />
        <SamplesSection />
        <MemorySection />
      </div>

      <RunSummary />
    </div>
    <RecipePicker v-model:open="recipesOpen" :start="recipeStart" />
    <SaveRecipe v-model:open="saveOpen" :spec="spec" />
  </div>
</template>

<style scoped>
.wide {
  max-width: 1280px;
}
.imported {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 12px;
  padding-right: 32px;
}
.imported .close {
  position: absolute;
  top: 6px;
  right: 6px;
  background: none;
  border: 0;
  color: inherit;
  cursor: pointer;
  opacity: 0.6;
}
.imported .warn {
  color: var(--zen-warn);
}
.imported .file {
  margin-left: 8px;
  font-family: var(--zen-mono);
}
/* the header's two buttons: one height, one look */
.bar-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 30px;
  padding: 0 12px;
  border-radius: var(--r-md);
  border: 1px solid var(--zen-border);
  background: var(--zen-surface);
  color: var(--zen-text);
  font: inherit;
  font-size: 12.5px;
  cursor: pointer;
}
.bar-btn.icon {
  width: 30px;
  padding: 0;
  font-size: 17px;
}
.bar-btn:hover {
  border-color: var(--zen-accent);
}
.from {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted);
}
.from b {
  color: var(--zen-text);
  font-weight: 600;
}
.copied {
  color: var(--zen-success, var(--zen-accent));
}
.from .undo {
  font: inherit;
  font-size: 11px;
  border: none;
  background: color-mix(in srgb, var(--zen-text) 10%, transparent);
  color: var(--zen-text);
  border-radius: 999px;
  padding: 1px 9px;
  cursor: pointer;
}
.from .undo:hover {
  background: color-mix(in srgb, var(--zen-text) 18%, transparent);
}
.warn {
  color: var(--zen-warn, #d9a400);
}
.simple-hint {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  margin: 12px 0 0;
  border-color: color-mix(in srgb, var(--zen-accent) 45%, var(--zen-border));
  background: color-mix(in srgb, var(--zen-accent) 7%, var(--zen-surface));
}
.simple-hint .lead {
  font-size: 22px;
  color: var(--zen-accent);
}
.hint-text {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 13px;
}
.hint-close {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
}
.hint-close:hover {
  color: var(--zen-text);
}
@media (max-width: 720px) {
  .simple-hint {
    flex-wrap: wrap;
  }
  .hint-text {
    flex-basis: calc(100% - 70px);
  }
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 18px;
  align-items: start;
}
.form {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.locked-note {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  color: var(--zen-muted);
  line-height: 1.5;
}
@media (max-width: 1000px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
