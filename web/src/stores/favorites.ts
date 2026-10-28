// Starred models: the families someone actually uses, first in every model list (New run,
// Text to Image, Models); starred datasets, first in New run's dataset picker. Kept in ComfyUI's
// per-user settings, so they follow you around.

import { computed } from 'vue'
import { setSetting, settings } from '@/stores/settings'

export const starred = computed(() => new Set(settings.favoriteModels))
export const isStarred = (id: string) => starred.value.has(id)

export function toggleStar(id: string): void {
  const next = isStarred(id) ? settings.favoriteModels.filter((x) => x !== id) : [...settings.favoriteModels, id]
  setSetting('favoriteModels', next)
}

/** Starred first; otherwise the list keeps its order (newest first, for model families). */
export function starredFirst<T>(list: T[], id: (x: T) => string): T[] {
  return [...list].sort((a, b) => Number(isStarred(id(b))) - Number(isStarred(id(a))))
}

export const starredDatasets = computed(() => new Set(settings.favoriteDatasets))
export const isDatasetStarred = (folder: string) => starredDatasets.value.has(folder)

export function toggleDatasetStar(folder: string): void {
  const next = isDatasetStarred(folder) ? settings.favoriteDatasets.filter((x) => x !== folder) : [...settings.favoriteDatasets, folder]
  setSetting('favoriteDatasets', next)
}
