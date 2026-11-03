// App settings, declared once here. Values are saved in ComfyUI's per-user settings
// (comfy.settings.json, as "ZenTrainer.<id>"), so they follow the user to any browser on this
// ComfyUI; localStorage is only a cache so the app starts with them before the fetch lands.
// The Settings page renders every entry, grouped by category.

import { reactive, watch } from 'vue'

export interface SettingOption<T> {
  value: T
  label: string
  icon?: string
}
export interface SettingDef<T = unknown> {
  category: string
  label: string
  description?: string
  default: T
  /** empty: any string (a path, an id): kept, but not drawn on the Settings page */
  options: SettingOption<T>[]
}

const def = <T>(d: SettingDef<T>) => d

export const SETTINGS = {
  mode: def<'simple' | 'advanced'>({
    category: 'General',
    label: 'Interface',
    description: 'Simple: train from ready-made recipes and see just the progress. Advanced: every setting, the loss, the captions.',
    default: 'advanced',
    options: [
      { value: 'simple', label: 'Simple', icon: 'mdi mdi-sprout-outline' },
      { value: 'advanced', label: 'Advanced', icon: 'mdi mdi-tune-variant' },
    ],
  }),
  uiSize: def<number>({
    category: 'Appearance',
    label: 'Text size',
    description: 'Scales the whole interface with the text, so nothing gets cramped. Saved with your ComfyUI user, so it follows you to other browsers.',
    default: 1,
    options: [
      { value: 0.9, label: '90%' },
      { value: 1, label: '100%' },
      { value: 1.1, label: '110%' },
      { value: 1.25, label: '125%' },
      { value: 1.5, label: '150%' },
    ],
  }),
  motion: def<'system' | 'reduce' | 'full'>({
    category: 'Appearance',
    label: 'Motion',
    description: 'Slides, zooms and folds. System follows your OS setting for reduced motion.',
    default: 'system',
    options: [
      { value: 'system', label: 'System', icon: 'mdi mdi-monitor' },
      { value: 'reduce', label: 'Reduced', icon: 'mdi mdi-motion-pause-outline' },
      { value: 'full', label: 'Full', icon: 'mdi mdi-motion-play-outline' },
    ],
  }),
  experimental: def<boolean>({
    category: 'General',
    label: 'Experimental pages',
    description: "Apps, Tools and Gallery: they work, but they're still being finished. A run's Try buttons open Apps either way.",
    default: false,
    options: [
      { value: false, label: 'Hidden', icon: 'mdi mdi-eye-off-outline' },
      { value: true, label: 'Shown', icon: 'mdi mdi-flask-outline' },
    ],
  }),
  keepStates: def<'latest' | 'all' | 'none'>({
    category: 'Training',
    label: 'Resume states',
    description: 'Optimizer state saved with checkpoints so a stopped run can resume. Often bigger than the LoRA itself. New runs start with this.',
    default: 'latest',
    options: [
      { value: 'latest', label: 'Latest only', icon: 'mdi mdi-history' },
      { value: 'all', label: 'Every checkpoint', icon: 'mdi mdi-layers-outline' },
      { value: 'none', label: 'None', icon: 'mdi mdi-close-circle-outline' },
    ],
  }),
  // a run page's noise chart (RunNoise): folded unless opened, remembered across runs
  noiseChartOpen: def<boolean>({ category: 'Training', label: 'Noise chart open', default: false, options: [] }),
  // the loras folder new runs save into ('' = inside this install); chosen in New run
  loraFolder: def<string>({ category: 'Training', label: 'LoRA folder', default: '', options: [] }),
  // starred model families: first in every model list (see favorites.ts)
  favoriteModels: def<string[]>({ category: 'Models', label: 'Starred models', default: [], options: [] }),
  // starred datasets: first in New run's dataset picker (see favorites.ts)
  favoriteDatasets: def<string[]>({ category: 'Datasets', label: 'Starred datasets', default: [], options: [] }),
  // your own auto-caption instructions, beside the built-in styles (see CaptionPanel)
  captionStyles: def<{ id: string; label: string; prompt: string; format?: 'text' | 'fibo_json' }[]>({ category: 'Datasets', label: 'Caption styles', default: [], options: [] }),
}

export type SettingId = keyof typeof SETTINGS
type Values = { [K in SettingId]: (typeof SETTINGS)[K]['default'] }

const PREFIX = 'ZenTrainer.'
const CACHE = 'zentrainer.settings'
const defaults = Object.fromEntries(Object.entries(SETTINGS).map(([k, d]) => [k, d.default])) as Values

function cached(): Partial<Values> {
  try {
    return JSON.parse(localStorage.getItem(CACHE) ?? '{}')
  } catch {
    return {}
  }
}
const valid = <K extends SettingId>(id: K, v: unknown): v is Values[K] => {
  const opts = SETTINGS[id].options as SettingOption<unknown>[]
  if (opts.length) return opts.some((o) => o.value === v)
  return Array.isArray(SETTINGS[id].default) ? Array.isArray(v) && v.every((x) => typeof x === 'string') : typeof v === 'string'
}

export const settings = reactive<Values>({ ...defaults, ...cached() })

function remember() {
  try {
    localStorage.setItem(CACHE, JSON.stringify(settings))
  } catch {
    // storage blocked: the server copy still holds it
  }
}

export async function setSetting<K extends SettingId>(id: K, value: Values[K]): Promise<void> {
  settings[id] = value
  remember()
  try {
    await fetch(`/settings/${encodeURIComponent(PREFIX + id)}`, { method: 'POST', body: JSON.stringify(value) })
  } catch {
    // offline: kept locally, saved next time
  }
}

/** Read the server copy (it wins over the cache) and start applying settings to the page. */
export async function initSettings(): Promise<void> {
  apply()
  watch(() => [settings.motion, settings.uiSize], apply)
  try {
    const all = (await (await fetch('/settings')).json()) as Record<string, unknown>
    for (const id of Object.keys(SETTINGS) as SettingId[]) {
      const v = all[PREFIX + id]
      if (v !== undefined && valid(id, v)) (settings as Record<SettingId, unknown>)[id] = v
    }
    remember()
  } catch {
    // ComfyUI unreachable: run on the cache
  }
}

/** whether to skip motion: the Motion setting, or the OS preference when it's System (for JS
 *  animations; CSS ones follow data-zen-motion through app.css) */
export function reducedMotion(): boolean {
  if (settings.motion !== 'system') return settings.motion === 'reduce'
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
}

// effects on the page
function apply() {
  const root = document.documentElement
  if (settings.motion === 'system') delete root.dataset.zenMotion
  else root.dataset.zenMotion = settings.motion // ZenKit and app.css read it
  // CSS zoom, not a root font-size: sizes here are px, and zoom scales the boxes around the text too.
  // It scales vw/vh as well, so lengths sized to the window divide by --zen-page-zoom (ZenKit's too).
  if (settings.uiSize === 1) {
    root.style.removeProperty('zoom')
    root.style.removeProperty('--zen-page-zoom')
  } else {
    root.style.zoom = String(settings.uiSize)
    root.style.setProperty('--zen-page-zoom', String(settings.uiSize))
  }
}
