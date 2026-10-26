// Your own recipes: run presets saved with the server (engine presets.py, kind "run"), listed under
// "Yours" in the recipes dialog. Saved from the run form or from a run's page, so the list is
// shared here rather than kept in either.

import { ref } from 'vue'
import { engine, type Preset, type TrainSpec } from '@/api/engine'

export type MyRecipe = Preset<Partial<TrainSpec>>

export const myRecipes = ref<MyRecipe[]>([])
let loaded: Promise<void> | null = null

/** the list from the server, once (later calls wait on the same load); `again` refetches */
export function loadMyRecipes(again = false): Promise<void> {
  if (!loaded || again) {
    loaded = engine.presets<Partial<TrainSpec>>('run').then(
      (r) => void (myRecipes.value = r),
      () => void (loaded = null), // engine unreachable: try again next time
    )
  }
  return loaded
}

/** what a recipe keeps of a spec: its settings and model, without what applying one leaves as it
 *  is (the run's name, dataset, trigger word and captions) */
export function recipeData(spec: TrainSpec): Partial<TrainSpec> {
  const { name: _n, dataset, ...rest } = JSON.parse(JSON.stringify(spec)) as TrainSpec
  const { folder: _f, trigger: _t, trigger_prepend: _tp, caption_set: _cs, caption_mix: _cm, ...ds } = dataset
  return { ...rest, dataset: ds } as unknown as Partial<TrainSpec> // a partial dataset: it merges over the form's
}

// the same JSON whatever order its keys were written in
function canonical(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`
  if (v && typeof v === 'object') {
    return `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}:${canonical((v as Record<string, unknown>)[k])}`).join(',')}}`
  }
  return JSON.stringify(v ?? null)
}
/** one of yours with exactly these settings, if there is one */
export function sameRecipe(data: Partial<TrainSpec>): MyRecipe | null {
  const want = canonical(data)
  return myRecipes.value.find((r) => canonical(r.data) === want) ?? null
}

export async function saveMyRecipe(name: string, description: string, data: Partial<TrainSpec>): Promise<MyRecipe> {
  const saved = (await engine.savePreset('run', name, data, description)).preset as MyRecipe
  myRecipes.value = [...myRecipes.value.filter((p) => p.id !== saved.id), saved]
  return saved
}

export async function deleteMyRecipe(id: string): Promise<void> {
  await engine.deletePreset('run', id)
  myRecipes.value = myRecipes.value.filter((p) => p.id !== id)
}
