<script setup lang="ts">
// The LoRAs a run uses besides the one it trains, as one list: each a file and how it's used.
//   Continue it    the new LoRA starts from it and becomes it (network.init_from)
//   Underneath     it sits under the new one everywhere, which learns only what's left (base_lora)
//   Training only  it's on while training only, never in samples or the file (training_adapter)
// One of each (the spec has a setting per role); a row's role moves its file to that setting.
import { ZenCombo, ZenNumber } from '@nynxz/zenkit-ui'
import { ref, watch } from 'vue'
import AdapterPicker from '@/components/AdapterPicker.vue'
import { items, useRunFormContext } from './form'

const { spec, choices, familyId } = useRunFormContext()

type Role = 'continue' | 'base' | 'training'
const FIELD = { continue: 'init_from', base: 'base_lora', training: 'training_adapter' } as const
const ROLES: { id: Role; label: string; hint: string }[] = [
  { id: 'continue', label: 'Continue it', hint: 'The new LoRA starts from this one: train more on it. The saved file is the two together' },
  { id: 'base', label: 'Underneath', hint: 'On under the new LoRA in training and samples; the new one learns only what it adds, so load this one beside it when you use it' },
  { id: 'training', label: 'Training only', hint: "On while training, never in samples or the saved file: a de-distillation adapter, so a distilled model (MiniMax H3) trains on an undistilled one" },
]
const value = (r: Role) => spec.network[FIELD[r]] ?? ''
const setValue = (r: Role, v: string) => (spec.network[FIELD[r]] = v || null)

// the rows: the roles in use, and any just added with no file yet; a recipe, a duplicate or
// Train more that sets one shows it too
const rows = ref<Role[]>(ROLES.map((r) => r.id).filter((r) => value(r)))
watch(
  () => ROLES.map((r) => value(r.id)),
  () => {
    for (const r of ROLES) if (value(r.id) && !rows.value.includes(r.id)) rows.value.push(r.id)
  },
)
function add() {
  const free = ROLES.find((r) => !rows.value.includes(r.id))
  if (free) rows.value.push(free.id)
}
function setRole(i: number, to: Role) {
  const from = rows.value[i]!
  if (from === to || rows.value.includes(to)) return
  const v = value(from)
  setValue(from, '')
  setValue(to, v)
  rows.value[i] = to
}
function remove(i: number) {
  setValue(rows.value[i]!, '')
  rows.value.splice(i, 1)
}
const hint = (r: Role) => ROLES.find((x) => x.id === r)!.hint
</script>

<template>
  <div class="lora-uses">
    <div v-for="(r, i) in rows" :key="r" class="use">
      <div class="use-top">
        <div class="view-switch text" role="radiogroup" aria-label="How it's used">
          <button
            v-for="o in ROLES"
            :key="o.id"
            role="radio"
            :aria-checked="r === o.id"
            :class="{ on: r === o.id }"
            :disabled="o.id !== r && rows.includes(o.id)"
            :title="o.id !== r && rows.includes(o.id) ? `Another LoRA is ${o.label.toLowerCase()} already` : o.hint"
            @click="setRole(i, o.id)"
          >
            {{ o.label }}
          </button>
        </div>
        <span class="spacer" />
        <button type="button" class="remove" title="Don't use this LoRA" @click="remove(i)"><i class="mdi mdi-close" /></button>
      </div>
      <div class="use-pick">
        <ZenCombo
          v-if="r === 'continue'"
          :model-value="value(r)"
          :items="items(choices.loras)"
          placeholder="Pick a LoRA…"
          :item-height="30"
          @update:model-value="setValue(r, String($event))"
        />
        <AdapterPicker v-else :model-value="value(r)" :family="familyId" :kind="r === 'base' ? 'base' : 'training'" @update:model-value="setValue(r, $event)" />
        <label v-if="r === 'base'" class="strength" title="The strength it runs at, in training and samples: use it at the same strength">
          <span class="muted small">Strength</span>
          <ZenNumber :model-value="spec.network.base_strength ?? 1" :min="-4" :max="4" :step="0.05" :precision="2" @update:model-value="spec.network.base_strength = $event" />
        </label>
      </div>
      <p class="hint muted small">{{ hint(r) }}</p>
    </div>
    <button v-if="rows.length < ROLES.length" type="button" class="add-lora" @click="add">
      <i class="mdi mdi-plus" /> {{ rows.length ? 'Add another LoRA' : 'Use a LoRA' }}
    </button>
  </div>
</template>

<style scoped>
.lora-uses {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.use {
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.use-top {
  display: flex;
  align-items: center;
  gap: 8px;
}
.remove {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  padding: 2px;
  font-size: 16px;
}
.remove:hover {
  color: var(--zen-text);
}
.use-pick {
  display: flex;
  align-items: center;
  gap: 10px;
}
.use-pick > :first-child {
  flex: 1;
  min-width: 0;
}
.strength {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 170px;
  flex: none;
}
.hint {
  margin: 0;
  line-height: 1.4;
}
.add-lora {
  align-self: flex-start;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 5px 12px;
  border: 1px dashed var(--zen-border);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.add-lora:hover {
  color: var(--zen-text);
  border-color: color-mix(in srgb, var(--zen-accent) 50%, var(--zen-border));
}
.view-switch button:disabled {
  opacity: 0.4;
  cursor: default;
}
</style>
