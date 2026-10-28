<script setup lang="ts">
// 1 Model: the family picker, its hand-picked recipe (applied when the family is picked, re-appliable
// after edits), and the file-by-file pick for a custom model.
import { ZenCombo, ZenField, ZenMenuItem, ZenPopover } from '@nynxz/zenkit-ui'
import { computed, ref, toRaw } from 'vue'
import type { ModelProfile, TrainSpec } from '@/api/engine'
import FamilyPicker from '@/components/FamilyPicker.vue'
import { RECIPES, applyTo, plan } from '@/lib/recipes'
import ModelBrowser from './ModelBrowser.vue'
import { items, merge, useRunFormContext } from './form'

const { spec, choices, modelMissing, sampleSuggestion, profile, familyId, resuming, applyRecipe } = useRunFormContext()

const customModel = ref(false) // the picker's custom (file-by-file) mode
function applyProfile(p: ModelProfile) {
  merge(spec as unknown as Record<string, unknown>, p.train)
  merge(spec.samples as unknown as Record<string, unknown>, p.samples as Record<string, unknown>)
}
function onProfile(p: ModelProfile | null, picked: boolean) {
  profile.value = p
  if (p && picked) applyProfile(p)
}
// does the form still hold every value the profile sets?
const profileApplied = computed(() => {
  const p = profile.value
  if (!p) return false
  const same = (want: Record<string, unknown>, have: Record<string, unknown>): boolean =>
    Object.entries(want).every(([k, v]) =>
      v && typeof v === 'object' && !Array.isArray(v)
        ? same(v as Record<string, unknown>, (have[k] ?? {}) as Record<string, unknown>)
        : JSON.stringify(v) === JSON.stringify(have[k]),
    )
  return same(p.train, spec as unknown as Record<string, unknown>) && same(p.samples as Record<string, unknown>, spec.samples as unknown as Record<string, unknown>)
})
const profileOpen = ref(false)
// the model browser: a family picked there goes through the picker (its files, its downloads); a
// recipe there is applied as Recipes applies it (its model with it, Undo to take it back). If its
// files can't be worked out here, Recipes opens on it to say why
const emit = defineEmits<{ recipe: [id: string | null] }>()
async function useRecipe(id: string) {
  const r = RECIPES.find((x) => x.id === id)
  if (!r) return
  try {
    applyRecipe({ id: r.id, title: r.title }, applyTo(r, await plan(r), toRaw(spec) as TrainSpec))
  } catch {
    emit('recipe', id)
  }
}
const picker = ref<InstanceType<typeof FamilyPicker> | null>(null)
const browsing = ref(false)
</script>

<template>
<section class="card sec" :class="{ locked: resuming }">
  <div class="head-row">
    <h2><span class="num">1</span> Model</h2>
    <span class="grow" />
    <!-- extras a run can add (a slider for now), so they don't sit in the form when unused -->
    <ZenPopover v-if="!resuming" placement="bottom-end">
      <template #trigger="{ toggle }">
        <button class="add small" title="Add something to this run" @click="toggle"><i class="mdi mdi-plus" /> Add</button>
      </template>
      <template #default="{ close }">
        <ZenMenuItem icon="mdi mdi-image-multiple-outline" :disabled="spec.slider.kind === 'pairs'" @select="spec.slider.kind = 'pairs'; close()">
          Slider from image pairs
          <template #hint>photo pairs that differ in one thing</template>
        </ZenMenuItem>
        <ZenMenuItem icon="mdi mdi-text-short" :disabled="spec.slider.kind === 'text'" @select="spec.slider.kind = 'text'; close()">
          Slider from prompts
          <template #hint>two prompts, no pairs needed</template>
        </ZenMenuItem>
      </template>
    </ZenPopover>
  </div>
  <ModelBrowser v-model:open="browsing" :current="familyId" @pick="picker?.pick($event)" @recipe="useRecipe" @details="emit('recipe', $event)" />
  <FamilyPicker
    ref="picker"
    v-model="spec.model"
    browse
    @browse="browsing = true"
    @missing="modelMissing = $event"
    @sample-suggestion="sampleSuggestion = $event"
    @custom="customModel = $event"
    @profile="onProfile"
    @family="familyId = $event"
  />
  <div v-if="profile" class="profile" :class="{ open: profileOpen }">
    <div class="profile-head">
      <i :class="profile.status === 'verified' ? 'mdi mdi-check-decagram' : 'mdi mdi-check-circle-outline'" />
      <span class="profile-sum">
        <b>{{ profile.status === 'verified' ? 'Tuned' : 'Recipe' }}</b> · {{ profile.summary }}
      </span>
      <span class="profile-state">
        <span v-if="profileApplied" class="muted small"><i class="mdi mdi-check" /> applied</span>
        <button v-else class="chip-btn" title="Set the form back to this model's recommended values" @click="applyProfile(profile)">Apply</button>
      </span>
      <button class="link small" @click="profileOpen = !profileOpen">{{ profileOpen ? 'Less' : 'Details' }}</button>
    </div>
    <div v-if="profileOpen" class="profile-body">
      <ul v-if="profile.notes.length" class="profile-notes">
        <li v-for="n in profile.notes" :key="n">{{ n }}</li>
      </ul>
      <div class="profile-src">
        <a v-for="src in profile.sources" :key="src.url" :href="src.url" target="_blank" rel="noopener" class="link small"><i class="mdi mdi-open-in-new" /> {{ src.label }}</a>
      </div>
    </div>
  </div>
  <div v-if="customModel" class="custom-model">
    <div class="seg">
      <button :class="{ on: spec.model.kind === 'checkpoint' }" @click="spec.model.kind = 'checkpoint'">One checkpoint file</button>
      <button :class="{ on: spec.model.kind === 'split' }" @click="spec.model.kind = 'split'">Diffusion model + text encoder + VAE</button>
    </div>
    <div class="grid-2">
      <ZenField v-if="spec.model.kind === 'checkpoint'" label="Checkpoint" stack class="wide-f">
        <ZenCombo v-model="spec.model.ckpt_name as string" :items="items(choices.ckpt)" placeholder="Search checkpoints…" :item-height="30" />
      </ZenField>
      <template v-else>
        <ZenField label="Diffusion model" stack class="wide-f">
          <ZenCombo v-model="spec.model.unet_name as string" :items="items(choices.unet)" placeholder="Search diffusion models…" :item-height="30" />
        </ZenField>
        <ZenField label="Text encoder" stack>
          <ZenCombo v-model="spec.model.clip_name as string" :items="items(choices.clip)" placeholder="Search text encoders…" :item-height="30" />
        </ZenField>
        <ZenField label="Text encoder type" stack tip="The model family">
          <ZenCombo v-model="spec.model.clip_type as string" :items="items(choices.clip_type)" placeholder="Family…" />
        </ZenField>
        <ZenField label="VAE" stack>
          <ZenCombo v-model="spec.model.vae_name as string" :items="items(choices.vae)" placeholder="Search VAEs…" :item-height="30" />
        </ZenField>
        <ZenField label="Weight dtype" stack tip="fp8 + quantized backward is the low-VRAM path">
          <ZenCombo v-model="spec.model.unet_weight_dtype as string" :items="items(choices.unet_dtype ?? ['default'])" :searchable="false" />
        </ZenField>
      </template>
    </div>
  </div>
</section>
</template>

<style scoped src="./form.css"></style>
<style scoped>
.head-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.head-row h2 {
  margin: 0;
}
.head-row .grow {
  flex: 1;
}
.add {
  background: none;
  border: 1px dashed var(--zen-border);
  border-radius: var(--zen-radius);
  color: var(--zen-muted);
  padding: 3px 10px;
  cursor: pointer;
}
.add:hover {
  color: var(--zen-accent);
  border-color: var(--zen-accent);
}
.custom-model {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 60%, transparent);
}
.link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: none;
  background: none;
  padding: 0;
  color: var(--zen-muted);
  font: inherit;
  cursor: pointer;
  text-decoration: none;
}
.link:hover {
  color: var(--zen-accent);
}
.profile {
  margin-top: 10px;
  padding: 8px 12px;
  border-radius: var(--r-lg);
  font-size: 12px;
  background: color-mix(in srgb, var(--zen-accent) 7%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 25%, transparent);
}
.profile-head {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 26px;
}
.profile-head > i {
  font-size: 16px;
  color: var(--zen-accent);
}
.profile-sum {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.open .profile-sum {
  white-space: normal;
}
.profile-state {
  min-width: 64px;
  display: inline-flex;
  justify-content: flex-end;
}
.profile-body {
  padding: 6px 0 2px 26px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.profile-notes {
  margin: 0;
  padding-left: 16px;
  line-height: 1.5;
  color: var(--zen-muted);
}
.profile-src {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}
</style>
