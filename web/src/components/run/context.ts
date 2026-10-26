// One run as the run page and its parts see it: the stored run, kept live from the engine's
// websocket events, and what they all derive from it (progress, checkpoints, likeness, the
// commands). RunView makes it; its parts take it with useRunContext().
import { computed, inject, onMounted, onUnmounted, ref, shallowReactive, type InjectionKey, type Ref, watch } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { likeCeiling, likeShare } from '@/lib/likeness'
import { keepCompact } from '@/lib/resumed'
import {
  ACTIVE,
  EVENTS,
  engine,
  type CheckpointRow,
  type Command,
  type LikenessRow,
  type MetricRow,
  type RunDetail,
  type RunStatus,
  type SampleProgress,
  type SampleRow,
  type ValidationRow,
} from '@/api/engine'

export const dur = (s: number | null | undefined) => {
  if (s == null || !isFinite(s)) return '–'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = Math.floor(s % 60)
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : m ? `${m}m ${String(sec).padStart(2, '0')}s` : `${sec}s`
}
export const fmtLr = (v?: number | null) => (v == null ? '–' : v === 0 ? '0' : v.toExponential(1))
// why a checkpoint was saved, from its name: the schedule (null), by hand, or as the run stopped
export const ckKind = (c: { lora_name: string }) => /_(stopped|error|manual)-step\d+/.exec(c.lora_name)?.[1] ?? null
// one row per step, newest first: a run stopped right after a save (or resumed) can hold several
// files of one step, so the final, then the scheduled save, then a manual one, then a stop's stands
const ckRank = (c: { final?: boolean; lora_name: string }) => (c.final ? 0 : { null: 1, manual: 2, stopped: 3, error: 4 }[String(ckKind(c))] ?? 5)

export function useRun(id: Ref<string>) {
  const detail = ref<RunDetail | null>(null)
  const error = ref<string | null>(null)
  const busy = ref<string | null>(null)
  const lrInput = ref(0)
  const sampling = ref<SampleProgress | null>(null)
  const now = ref(Date.now() / 1000)
  let clock: ReturnType<typeof setInterval> | undefined

  async function load() {
    try {
      const d = await engine.run(id.value)
      // shallow: a long run's rows (thousands, each with its noise levels) stay plain objects
      d.metrics = shallowReactive(d.metrics)
      detail.value = d
      lrInput.value = detail.value.metrics.at(-1)?.lr ?? detail.value.spec.optim.learning_rate
    } catch (e) {
      error.value = String(e)
    }
  }

  // manual tries waiting their turn: the engine runs sample commands one after another at step
  // boundaries, so any number can be queued; each leaves this list when its render lands
  const queuedTries = ref<{ id: number; prompt: string; again?: number }[]>([])

  let off = () => {}
  onMounted(() => {
    load()
    clock = setInterval(() => (now.value = Date.now() / 1000), 1000)
    off = onComfyMessage((m) => {
      const d = detail.value
      if (!d || m.data.run_id !== id.value) return
      if (m.type === EVENTS.metrics) {
        const row = m.data as unknown as MetricRow
        if (!d.metrics.length || d.metrics.at(-1)!.step < row.step) {
          d.metrics.push(row)
          keepCompact(d.metrics)
        }
      } else if (m.type === EVENTS.checkpoint) {
        d.checkpoints.push(m.data as unknown as CheckpointRow)
      } else if (m.type === EVENTS.sample) {
        const row = m.data as unknown as SampleRow
        d.samples.push(row)
        if (row.manual && !row.base) queuedTries.value = queuedTries.value.slice(1) // its render landed
      } else if (m.type === EVENTS.validation) {
        d.validation.push(m.data as unknown as ValidationRow)
      } else if (m.type === EVENTS.likeness) {
        ;(d.likeness ??= []).push(m.data as unknown as LikenessRow)
      } else if (m.type === EVENTS.sampleProgress) {
        const p = m.data as unknown as SampleProgress
        if (p.phase === 'done') setTimeout(() => (sampling.value = null), 600)
        else
          sampling.value = {
            ...sampling.value,
            ...p,
            preview: p.preview ?? (p.index === sampling.value?.index ? sampling.value?.preview : undefined),
          }
      } else if (m.type === EVENTS.status) {
        d.run.status = m.data.status as RunStatus
        d.run.message = (m.data.message as string | null) ?? undefined
        d.live = ACTIVE.includes(d.run.status)
      }
    })
  })
  onUnmounted(() => {
    off()
    clearInterval(clock)
  })
  watch(id, load)

  // ---- progress -----------------------------------------------------------------------------
  const last = computed(() => detail.value?.metrics.at(-1))
  const total = computed(() => detail.value?.spec.optim.steps ?? 0)
  const step = computed(() => last.value?.step ?? 0)
  const progress = computed(() => (total.value ? step.value / total.value : 0))
  const status = computed(() => detail.value?.run.status)
  // queued to sample a finished run's checkpoint isn't training (no Pause or Stop)
  const queuedToSample = computed(() => status.value === 'queued' && !!detail.value?.run.message?.startsWith('sampling'))
  const running = computed(() => !!status.value && ACTIVE.includes(status.value) && !queuedToSample.value)
  const ckRows = computed(() => {
    const by = new Map<number, NonNullable<typeof detail.value>['checkpoints'][number]>()
    for (const c of detail.value?.checkpoints ?? []) {
      const had = by.get(c.step)
      if (!had || ckRank(c) < ckRank(had)) by.set(c.step, c)
    }
    return [...by.values()].sort((a, b) => b.step - a.step)
  })
  // finished, with a checkpoint file: Sample now and Manual tries run as a short job on its newest
  // checkpoint instead (engine: POST .../sample), and the results land at that step
  const sampleCkpt = computed(() => (running.value || sampleBusy.value ? null : (ckRows.value.find((c) => c.size !== null) ?? null)))
  const sampleBusy = computed(() => status.value === 'sampling' || queuedToSample.value)
  // a finished run's newest checkpoint with its file still there
  const newestCkpt = computed(() => (running.value ? null : (ckRows.value.find((c) => c.size !== null) ?? null)))
  async function sampleFinished(tries?: Record<string, unknown>[]) {
    if (!sampleCkpt.value) return
    try {
      const r = await engine.sampleRun(id.value, tries ? { checkpoint: sampleCkpt.value.path, run_prompts: false, tries } : { checkpoint: sampleCkpt.value.path })
      if (detail.value) detail.value.run.status = r.run.status
    } catch (e) {
      error.value = String((e as Error).message ?? e)
    }
  }
  const wall = computed(() => {
    const d = detail.value
    if (!d) return null
    return (running.value ? now.value : d.run.updated) - d.run.created
  })
  const marks = computed(() => {
    const d = detail.value
    if (!d) return []
    const out: { step: number; kind: 'ckpt' | 'sample'; done: boolean }[] = []
    const saved = new Set(d.checkpoints.map((c) => c.step))
    const sampled = new Set(d.samples.filter((x) => !x.manual).map((x) => x.step))
    const every = d.spec.output.save_every
    if (every) for (let s = every; s < total.value; s += every) out.push({ step: s, kind: 'ckpt', done: saved.has(s) })
    const se = d.spec.samples.every
    if (se) for (let s = se; s <= total.value; s += se) out.push({ step: s, kind: 'sample', done: sampled.has(s) })
    return out
  })
  const nextSave = computed(() => {
    const every = detail.value?.spec.output.save_every
    if (!every || !last.value?.it_s) return null
    const next = (Math.floor(step.value / every) + 1) * every
    if (next >= total.value) return null
    return { step: next, in: dur((next - step.value) / last.value.it_s) }
  })
  const nextSample = computed(() => {
    const every = detail.value?.spec.samples.every
    if (!every || !last.value?.it_s) return null
    const next = (Math.floor(step.value / every) + 1) * every
    if (next > total.value) return null
    return { step: next, in: dur((next - step.value) / last.value.it_s) }
  })

  // with faces on: each sampled step's mean likeness (run samples and Sample now; manual tries left out)
  const likeCeil = computed(() => likeCeiling(detail.value?.likeness))
  const likeSteps = computed(() => {
    const by = new Map<number, number[]>()
    for (const r of detail.value?.likeness ?? []) if (!r.manual && r.score != null) by.set(r.step, [...(by.get(r.step) ?? []), r.score])
    return [...by].sort((a, b) => a[0] - b[0]).map(([step, xs]) => {
      const score = xs.reduce((a, b) => a + b, 0) / xs.length
      return { step, score, share: likeShare(score, likeCeil.value) }
    })
  })
  const likeLast = computed(() => likeSteps.value.at(-1) ?? null)
  const likeFirst = computed(() => likeSteps.value[0] ?? null)
  // validation loss at a checkpoint: the latest evaluation at or before its step
  const ckptVal = (stepAt: number) => {
    let v: number | undefined
    for (const r of detail.value?.validation ?? []) if (r.step <= stepAt) v = r.val_loss
    return v ?? null
  }
  // with faces on: the run's samples' likeness at a step (their mean; manual tries left out)
  const likeAt = (stepAt: number): number | null => {
    const xs = (detail.value?.likeness ?? []).filter((r) => r.step === stepAt && !r.manual && r.score != null).map((r) => r.score!)
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null
  }

  let tryIds = 0
  async function queueTry(args: Record<string, unknown>, prompt: string, again?: number) {
    const tid = ++tryIds
    queuedTries.value = [...queuedTries.value, { id: tid, prompt, again }]
    try {
      await engine.control(id.value, 'sample', args)
    } catch (e) {
      queuedTries.value = queuedTries.value.filter((q) => q.id !== tid)
      error.value = String((e as Error).message ?? e)
    }
  }
  // finished: one job renders the tries on the newest checkpoint; each waits in the list till it lands
  function queueFinishedTries(prompts: string[]) {
    for (const prompt of prompts) queuedTries.value = [...queuedTries.value, { id: ++tryIds, prompt }]
  }

  // ---- actions ------------------------------------------------------------------------------
  async function send(cmd: Command, args?: Record<string, unknown>) {
    busy.value = cmd
    if (cmd === 'sample') sampling.value = { run_id: id.value, phase: 'requested' }
    try {
      await engine.control(id.value, cmd, args)
    } catch (e) {
      error.value = String(e)
    } finally {
      setTimeout(() => (busy.value = null), 400)
    }
  }
  const resumable = computed(() => ['error', 'stopped', 'interrupted', 'rejected'].includes(status.value ?? ''))
  // ended before it trained a step: its settings can still change (the backend requeues it)
  const retryable = computed(() => resumable.value && !detail.value?.metrics.length && !detail.value?.checkpoints.length)
  // the newest checkpoint that kept its training state: where Resume continues from
  const resumePoint = computed(() => {
    const c = (detail.value?.checkpoints ?? []).filter((x) => x.state)
    return c.length ? Math.max(...c.map((x) => x.step)) : 0
  })
  // trained, but no state left to continue from: only Duplicate starts it again
  const canResume = computed(() => resumable.value && (!!resumePoint.value || retryable.value))
  async function resumeFrom(c: CheckpointRow) {
    const later = (detail.value?.checkpoints ?? []).filter((x) => x.step > c.step).length
    if (later && !confirm(`Go back to step ${c.step} and train on from there? The ${later} checkpoint${later === 1 ? '' : 's'} after it, and their samples, move to a "replaced" folder beside them (kept, not deleted).`)) return
    await resume(c.path)
  }
  async function resume(checkpoint?: string) {
    busy.value = 'resume'
    error.value = null
    try {
      await engine.resume(id.value, checkpoint)
      await load()
    } catch (e) {
      error.value = String((e as Error).message ?? e)
    } finally {
      busy.value = null
    }
  }
  // a finished run whose final save kept its training state can train on from it
  const extendable = computed(() => status.value === 'done' && !!detail.value?.checkpoints.some((c) => c.final && c.state))
  async function extend(more: number) {
    busy.value = 'extend'
    error.value = null
    try {
      await engine.extend(id.value, total.value + more)
      await load()
    } catch (e) {
      error.value = String((e as Error).message ?? e)
    } finally {
      busy.value = null
    }
  }

  return {
    id, detail, error, busy, lrInput, sampling, now, load,
    last, total, step, progress, status, running, ckRows, sampleCkpt, sampleBusy, newestCkpt, sampleFinished,
    wall, marks, nextSave, nextSample,
    likeSteps, likeLast, likeFirst, likeCeil, ckptVal, likeAt,
    queuedTries, queueTry, queueFinishedTries,
    send, resumable, retryable, resumePoint, canResume, resumeFrom, resume, extendable, extend,
  }
}

export type RunContext = ReturnType<typeof useRun>
export const RUN: InjectionKey<RunContext> = Symbol('run')
export const useRunContext = () => inject(RUN)!
