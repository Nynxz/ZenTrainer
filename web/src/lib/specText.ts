// A run spec in words, for the dialogs that list its settings (Recipes' changes, a run's full spec):
// its parts' names, and each value as it reads, not its JSON.

/** a spec's top-level parts, as the run form names them */
export const SECTIONS: Record<string, string> = {
  model: 'Model', dataset: 'Dataset', network: 'LoRA', optim: 'Training', objective: 'Noise and loss',
  samples: 'Samples', validation: 'Validation', memory: 'Memory', output: 'Saving', slider: 'Slider',
}

// an object in words: a resolution stage as "768 px for the first 60%", anything else as "key value · key value"
function words(o: Record<string, unknown>): string {
  if (typeof o.resolution === 'number' && typeof o.until === 'number') return `${o.resolution} px for the first ${Math.round(o.until * 100)}%`
  return Object.entries(o).map(([k, x]) => `${k.replace(/_/g, ' ')} ${typeof x === 'object' && x ? JSON.stringify(x) : String(x)}`).join(' · ')
}

/** a setting's value, short: on/off, lists joined, objects in words, long text cut */
export function fmt(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  if (Array.isArray(v)) return v.length ? v.map((x) => (typeof x === 'object' && x ? words(x as Record<string, unknown>) : String(x))).join(', then ') : '—'
  if (typeof v === 'boolean') return v ? 'on' : 'off'
  if (typeof v === 'object') return words(v as Record<string, unknown>)
  const s = String(v)
  return s.length > 48 ? `${s.slice(0, 46)}…` : s
}

/** a setting's value in full, for a hover */
export const full = (v: unknown) => (v === null || v === undefined || v === '' ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v))

/** a setting's name from its key past the part: "lora_plus" -> "lora plus", "a.b" -> "a · b" */
export const label = (rest: string[], head: string) => (rest.length ? rest.join(' · ') : head).replace(/_/g, ' ')

/** every setting of a spec, by part: one row per value (lists and stages are one value) */
export function specRows(spec: Record<string, unknown>): { section: string; rows: { label: string; value: string; full: string }[] }[] {
  const out: { section: string; rows: { label: string; value: string; full: string }[] }[] = []
  const walk = (v: unknown, path: string[]) => {
    if (v && typeof v === 'object' && !Array.isArray(v) && path.length < 3) {
      for (const [k, x] of Object.entries(v as Record<string, unknown>)) walk(x, [...path, k])
      return
    }
    const [head, ...rest] = path
    const section = SECTIONS[head!] ?? 'Run'
    let g = out.find((x) => x.section === section)
    if (!g) out.push((g = { section, rows: [] }))
    g.rows.push({ label: label(rest, head!), value: fmt(v), full: `${path.join('.')}: ${full(v)}` })
  }
  for (const [k, v] of Object.entries(spec)) walk(v, [k])
  return out
}
