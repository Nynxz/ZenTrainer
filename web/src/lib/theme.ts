// Theming: ZenKit's JSON packs, applied outside ComfyUI.
//
// Packs are shadcn-style semantic tokens (--background, --primary, ...). ZenKit's theme engine
// writes them to :root, and zenAliases derives the --zen-* tokens zenkit-ui reads.
// The engine serves the packs (/zentrainer/api/themes): the bundled set, a linked
// ZenKit's, and the shared user drop-in folder, so there's no dependency on ZenKit.

import { updateFavicon } from '@/lib/favicon'
import {
  BASE_TOKENS as BASE_DARK,
  applyPack,
  getPack,
  packModes,
  packs,
  parsePack,
  registerPacks,
  resolveTokens,
  zenAliases,
} from '@nynxz/zenkit-ui/theme'
import type { ThemeMode } from '@nynxz/zenkit-client'
import { computed, ref } from 'vue'

const SERVER_KEY = 'ZenTrainer.theme' // in ComfyUI's per-user settings
const STORAGE_KEY = 'zentrainer.theme'

// the theme engine's BASE_TOKENS are dark-only (inside ComfyUI the host palette fills the
// gaps). Standalone, a partial light pack would keep dark surfaces under dark text, so
// light mode gets a light base beneath the pack. UPSTREAM: belongs in ZenKit's theme engine.
const LIGHT_BASE: Record<string, string> = {
  '--background': '#fafafa',
  '--foreground': '#18181b',
  '--card': '#ffffff',
  '--card-foreground': '#18181b',
  '--popover': '#ffffff',
  '--popover-foreground': '#18181b',
  '--secondary': '#f1f1f4',
  '--secondary-foreground': '#18181b',
  '--muted': '#f1f1f4',
  '--muted-foreground': '#6b6b76',
  '--accent': '#ececf1',
  '--accent-foreground': '#18181b',
  '--border': '#e2e2e8',
  '--input': '#e2e2e8',
}

// Used only if the engine can't be reached; has both modes so the toggle always works.
const FALLBACK = {
  id: 'zen',
  name: 'Zen',
  tokens: {
    dark: {
      '--background': '#121216',
      '--foreground': '#e5e5ea',
      '--card': '#1a1a20',
      '--secondary': '#24242c',
      '--muted-foreground': '#9aa0aa',
      '--input': '#1b1b20',
      '--border': '#2e2e36',
      '--primary': '#8a4dd2',
      '--primary-foreground': '#ffffff',
      '--radius': '8px',
    },
    light: { ...LIGHT_BASE, '--primary': '#7c3aed', '--primary-foreground': '#ffffff', '--radius': '8px' },
  },
}

export interface PackCard {
  id: string
  name: string
  description?: string
  modes: ThemeMode[]
  /** a few resolved colours per mode, for drawing a preview without applying the pack */
  swatch: Record<ThemeMode, { bg: string; card: string; fg: string; muted: string; primary: string; border: string; radius: string }>
}

export const packCards = ref<PackCard[]>([])
/** `mode` is what's applied; `preferred` is what the user asked for. They differ while a
 *  one-mode pack is active, and the preference comes back with the next pack. */
export const currentTheme = ref<{ id: string; mode: ThemeMode; preferred: ThemeMode }>({
  id: 'zen',
  mode: 'dark',
  preferred: 'dark',
})
export const currentModes = computed(() => packCards.value.find((p) => p.id === currentTheme.value.id)?.modes ?? [])

export async function initTheme(): Promise<void> {
  let loaded: unknown[] = []
  try {
    const res = await fetch('/zentrainer/api/themes')
    if (res.ok) loaded = (await res.json()).themes ?? []
  } catch {
    // engine unreachable
  }
  if (!loaded.some((p) => (p as { id?: string }).id === 'zen')) loaded.unshift(FALLBACK)

  const raw = new Map(loaded.map((p) => [(p as { id?: string }).id, p as { description?: string; css?: string; has_css?: boolean }]))
  css.clear()
  hasCss.clear()
  for (const [id, p] of raw) {
    if (!id) continue
    if (typeof p.css === 'string') css.set(id, p.css) // a list that inlines it (an older engine)
    else if (p.has_css) hasCss.add(id) // fetched when it's the one shown
  }

  registerPacks(loaded.map(parsePack).filter((p) => p != null))
  packCards.value = packs().map((p) => {
    const sw = (mode: ThemeMode) => {
      const t = { ...(mode === 'light' ? LIGHT_BASE : BASE_DARK), ...resolveTokens(p, mode) }
      return {
        bg: t['--background'] ?? '',
        card: t['--card'] ?? t['--background'] ?? '',
        fg: t['--foreground'] ?? '',
        muted: t['--muted-foreground'] ?? '',
        primary: t['--primary'] ?? '',
        border: t['--border'] ?? '',
        radius: t['--radius'] ?? '8px',
      }
    }
    return {
      id: p.id,
      name: p.name,
      description: raw.get(p.id)?.description,
      modes: packModes(p),
      swatch: { light: sw('light'), dark: sw('dark') },
    }
  })

  let saved: { id?: string; mode?: ThemeMode; preferred?: ThemeMode } | null = null
  try {
    saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
  } catch {
    // storage blocked
  }
  // the copy saved with the ComfyUI user wins: the theme follows them to other browsers
  try {
    const server = await (await fetch(`/settings/${SERVER_KEY}`)).json()
    if (server && typeof server === 'object' && typeof server.id === 'string') saved = server
  } catch {
    // ComfyUI unreachable: the local copy will do
  }
  // ?theme=<id>&mode=<light|dark> wins: lets an embedding host (a ZenKit panel) pass its theme in.
  const q = new URLSearchParams(location.search)
  const qMode = q.get('mode')
  const prefersLight = matchMedia?.('(prefers-color-scheme: light)').matches
  const mode: ThemeMode =
    qMode === 'light' || qMode === 'dark' ? qMode : (saved?.preferred ?? saved?.mode ?? (prefersLight ? 'light' : 'dark'))
  const id = q.get('theme') ?? saved?.id
  const start = id && getPack(id) ? id : 'zen'
  await packCss(start) // before the first paint: its fonts don't swap in after
  // restoring isn't choosing: only a pick made here is saved back
  setTheme(start, mode, false)
}

// a pack's extra CSS (fonts, effects): only the shown pack's is fetched, once a session (the
// engine's ETag lets the browser keep it between loads)
const css = new Map<string, string>()
const hasCss = new Set<string>()
async function packCss(id: string): Promise<string> {
  if (css.has(id) || !hasCss.has(id)) return css.get(id) ?? ''
  try {
    const res = await fetch(`/zentrainer/api/themes/${encodeURIComponent(id)}/css`)
    if (res.ok) css.set(id, await res.text())
  } catch {
    // engine unreachable: the pack's tokens still apply
  }
  return css.get(id) ?? ''
}

export function setTheme(id: string, preferred: ThemeMode, save = true): void {
  const applied = applyPack(id, preferred)
  const root = document.documentElement
  const pack = getPack(id)
  if (applied === 'light') {
    const own = pack ? resolveTokens(pack, 'light') : {}
    for (const [k, v] of Object.entries(LIGHT_BASE)) if (!(k in own)) root.style.setProperty(k, v)
  }
  const cs = getComputedStyle(root)
  const tokens: Record<string, string> = {}
  for (const k of ['--background', '--foreground', '--card', '--secondary', '--muted', '--muted-foreground',
    '--input', '--border', '--primary', '--primary-foreground', '--destructive', '--radius',
    '--warning', '--success', '--info', '--font-mono']) {
    const v = cs.getPropertyValue(k).trim()
    if (v) tokens[k] = v
  }
  for (const [k, v] of Object.entries(zenAliases(tokens))) root.style.setProperty(k, v)
  for (const [k, v] of Object.entries(selectionTokens())) root.style.setProperty(k, v)
  root.style.colorScheme = applied
  root.dataset.mode = applied

  // a pack may carry extra CSS (fonts, effects); trusted local content, as in ZenKit
  let style = document.getElementById('zt-pack-css') as HTMLStyleElement | null
  if (!style) {
    style = document.createElement('style')
    style.id = 'zt-pack-css'
    document.head.append(style)
  }
  style.textContent = css.get(id) ?? ''
  if (!css.has(id) && hasCss.has(id)) {
    void packCss(id).then((text) => {
      if (currentTheme.value.id === id && style) style.textContent = text // still the one shown
    })
  }

  currentTheme.value = { id, mode: applied, preferred }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(currentTheme.value))
  } catch {
    // storage blocked
  }
  updateFavicon()
  if (save) fetch(`/settings/${SERVER_KEY}`, { method: 'POST', body: JSON.stringify(currentTheme.value) }).catch(() => {})
}

// What's selected has to read at a glance in every pack, but a pack's accent can sit almost on
// its card (Strawberry Milk's pale pink on white) and its second surface nearly match the card
// (Cute Hearts). So selection gets its own tokens: the accent moved toward the text colour until
// it stands 3:1 off the card (WCAG's contrast for UI parts), and a tint of that behind.
// UPSTREAM: belongs in zenAliases.
function rgbOf(color: string): [number, number, number] | null {
  const probe = document.createElement('span')
  probe.style.color = color
  document.body.append(probe)
  const resolved = getComputedStyle(probe).color // rgb(), or oklch() as written: the canvas reads either
  probe.remove()
  const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true })
  if (!ctx) return null
  ctx.fillStyle = '#000'
  ctx.fillStyle = resolved
  ctx.fillRect(0, 0, 1, 1)
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data
  return [r!, g!, b!]
}
const luminance = (c: number[]) =>
  c.map((v) => (v / 255 <= 0.03928 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4)).reduce((s, v, i) => s + v * [0.2126, 0.7152, 0.0722][i]!, 0)
const contrast = (a: number[], b: number[]) => {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p)
  return (x! + 0.05) / (y! + 0.05)
}
const hex = (c: number[]) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`

function selectionTokens(): Record<string, string> {
  const accent = rgbOf('var(--zen-accent)')
  const surface = rgbOf('var(--zen-surface)')
  const text = rgbOf('var(--zen-text)')
  if (!accent || !surface || !text) return {}
  let strong: number[] = accent
  for (let t = 0.05; contrast(strong, surface) < 3 && t <= 1; t += 0.05) strong = accent.map((v, i) => v + (text[i]! - v) * t)
  const bg = strong.map((v, i) => surface[i]! + (v - surface[i]!) * 0.24)
  return { '--zen-selected': hex(strong), '--zen-selected-bg': hex(bg), '--zen-selected-text': hex(text) }
}

/** Flip light/dark. Returns false when the active pack only has one mode. */
export function toggleMode(): boolean {
  if (currentModes.value.length < 2) return false
  setTheme(currentTheme.value.id, currentTheme.value.mode === 'dark' ? 'light' : 'dark')
  return true
}

export function setMode(mode: ThemeMode): void {
  setTheme(currentTheme.value.id, mode)
}
