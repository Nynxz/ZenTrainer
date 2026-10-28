// The run page's rail cards that fold away (Time, Learning, Setup), remembered in this browser:
// a view preference, so it's local storage and not the settings file.
import { ref } from 'vue'

const KEY = 'zentrainer.runRailCollapsed'

function load(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? '[]')
    return Array.isArray(v) ? v.filter((x) => typeof x === 'string') : []
  } catch {
    return []
  }
}

const collapsed = ref<string[]>(load())

export function isCollapsed(card: string): boolean {
  return collapsed.value.includes(card)
}

export function toggleCollapsed(card: string): void {
  collapsed.value = isCollapsed(card) ? collapsed.value.filter((c) => c !== card) : [...collapsed.value, card]
  try {
    localStorage.setItem(KEY, JSON.stringify(collapsed.value))
  } catch {
    // private window or storage off: it folds for this visit only
  }
}
