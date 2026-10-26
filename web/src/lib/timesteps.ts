// Where training's steps land on the noise schedule, worked out in the browser the way the
// engine draws them (engine/zentrainer/harness/objective.py: sample_percents, sigmas_for), for
// New run's Focus chart and bars. Flow models: σ is the noise fraction, 1 = full noise.
import { ref, watch, type Ref } from 'vue'
import { engine, type NoiseProfile, type TrainSpec } from '@/api/engine'

type Objective = Pick<TrainSpec['objective'], 'timesteps' | 'logit_mean' | 'logit_std' | 'uniform_mix' | 'shift' | 'bias' | 'bias_power' | 'timestep_scheduler' | 'timestep_scheduler_steps' | 'weighting'>

// Acklam's inverse normal CDF: logit-normal timesteps from evenly spaced quantiles
export function ndtri(p: number): number {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
  const lo = 0.02425
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p))
    return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
  }
  if (p > 1 - lo) return -ndtri(1 - p)
  const q = p - 0.5
  const r = q * q
  return ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) / (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1)
}

/** a scheduler's σ at `steps` steps, estimated from the profile's 200-step curve: only until
 * the exact ones (useSchedule) arrive, as schedulers like karras space few steps differently */
function scheduleAt(curve: number[], steps: number): number[] {
  return Array.from({ length: steps }, (_, i) => {
    const x = (i / steps) * (curve.length - 1)
    const j = Math.min(curve.length - 2, Math.floor(x))
    return curve[j]! + (curve[j + 1]! - curve[j]!) * (x - j)
  })
}

/** [σ, share of the steps] for `n` evenly spaced draws of the objective on this model */
export function draws(o: Objective, profile: NoiseProfile, n = 20000, exact: number[] | null = null): [number, number][] {
  const table = profile.sigmas ?? []
  const curve = o.timesteps === 'scheduler' ? profile.schedules?.[o.timestep_scheduler ?? 'simple'] : undefined
  const sched = o.timesteps !== 'scheduler' ? null : exact?.length ? exact : curve?.length ? scheduleAt(curve, o.timestep_scheduler_steps ?? 20) : null
  // percent of the schedule (0 = full noise) to σ: a scheduler's own steps, a shift, or the model's own
  const sigmaAt = (percent: number) => {
    if (sched) return sched[Math.min(sched.length - 1, Math.floor(percent * sched.length))]!
    const t = 1 - percent
    if (o.shift > 0) return (o.shift * t) / (1 + (o.shift - 1) * t)
    const x = percent * (table.length - 1)
    const i = Math.min(table.length - 2, Math.floor(x))
    return table[i]! + (table[i + 1]! - table[i]!) * (x - i)
  }
  const k = o.bias_power ?? 3
  const lean = (u: number) => (o.bias === 'high_noise' ? 1 - u ** k : o.bias === 'low_noise' ? u ** k : u)
  const mix = o.timesteps === 'logit_normal' ? (o.uniform_mix ?? 0) : 1
  const out: [number, number][] = []
  for (let i = 0; i < n; i++) {
    const q = Math.min(1 - 1e-6, Math.max(1e-6, lean((i + 0.5) / n))) // quantile, 1 = the most noise
    out.push([sigmaAt(1 - q), mix / n])
    if (mix < 1) out.push([sigmaAt(1 - 1 / (1 + Math.exp(-(o.logit_mean + o.logit_std * ndtri(q))))), (1 - mix) / n])
  }
  return out
}

/** the exact σ of the objective's scheduler on the profile's model, fetched when it changes
 * (null for other timesteps, or until it arrives) */
export function useSchedule(profile: Ref<NoiseProfile>, objective: () => Objective | undefined): Ref<number[] | null> {
  const out = ref<number[] | null>(null)
  watch(
    () => {
      const o = objective()
      const p = profile.value
      return o?.timesteps === 'scheduler' && p.folder && p.name ? `${p.folder}|${p.name}|${o.timestep_scheduler ?? 'simple'}|${o.timestep_scheduler_steps ?? 20}` : ''
    },
    async (key) => {
      out.value = null
      if (!key) return
      const [folder, name, scheduler, steps] = key.split('|') as [NoiseProfile['folder'] & string, string, string, string]
      try {
        const r = await engine.noiseSchedule(folder, name, scheduler, Number(steps))
        const o = objective()
        if (o && `${folder}|${name}|${o.timestep_scheduler ?? 'simple'}|${o.timestep_scheduler_steps ?? 20}` === key) out.value = r.sigmas
      } catch {
        /* the estimate stays */
      }
    },
    { immediate: true },
  )
  return out
}

/** [composition, form, detail] shares (σ above 2/3, between, below 1/3) */
export function bands(d: [number, number][]): number[] {
  const out = [0, 0, 0]
  for (const [s, w] of d) out[s > 2 / 3 ? 0 : s < 1 / 3 ? 2 : 1]! += w
  return out
}

/** shares in `bins` equal stretches of σ from 1 (first) down to 0, smoothed by `passes` of
 * [1 2 1] so a curve reads as its shape, not the bins' edges (a few exact levels want more) */
export function histogram(d: [number, number][], bins = 48, passes = 1): number[] {
  let out = new Array<number>(bins).fill(0)
  for (const [s, w] of d) out[Math.min(bins - 1, Math.max(0, Math.floor((1 - s) * bins)))]! += w
  for (let p = 0; p < passes; p++) {
    const prev = out
    out = prev.map((v, i) => (2 * v + (prev[i - 1] ?? v) + (prev[i + 1] ?? v)) / 4)
  }
  return out
}

/** a few exact noise levels (a scheduler's steps): [σ, share] for each */
export function levels(d: [number, number][]): [number, number][] {
  const m = new Map<number, number>()
  for (const [s, w] of d) {
    const k = Math.round(s * 1e4) / 1e4
    m.set(k, (m.get(k) ?? 0) + w)
  }
  return [...m.entries()].sort((a, b) => b[0] - a[0])
}

// AI Toolkit's "weighted" loss weights (objective.FLEX_WEIGHTS): σ = 1, 0.975, ... 0
export const FLEX_WEIGHTS = [
  0.986, 1.205, 1.369, 1.465, 1.503, 1.519, 1.509, 1.481, 1.443, 1.401, 1.357, 1.311, 1.265, 1.219,
  1.174, 1.128, 1.084, 1.042, 1.002, 0.963, 0.927, 0.893, 0.861, 0.832, 0.806, 0.783, 0.763, 0.746,
  0.732, 0.722, 0.716, 0.714, 0.716, 0.721, 0.728, 0.731, 0.716, 0.667, 0.586, 0.497, 0.449,
]
