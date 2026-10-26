// The run's samples as the samples card's views share them: the prompt × step grid, each
// prompt's newest render, the lightbox that pages through them, and videos played a row at a
// time. RunSamples makes it; Latest, Progression and Manual take it with useSamplesContext().
import { computed, inject, onUnmounted, ref, type InjectionKey } from 'vue'
import type { LightboxItem } from '@nynxz/zenkit-ui'
import { promptText as specPrompt, sampleInputUrl, viewUrl, type SampleRow } from '@/api/engine'
import { playTogether } from '@/lib/videoSync'
import type { RunContext } from './context'

export const isVideo = (x: SampleRow) => /\.(mp4|webm)$/i.test(x.filename)
export const sampleItem = (x: SampleRow): LightboxItem => ({
  src: viewUrl(x),
  kind: isVideo(x) ? 'video' : 'image',
  label: `${x.index + 1} · ${x.step === 0 ? 'baseline' : `step ${x.step}`}${x.model ? ` · ${x.model}` : ''}`,
  meta: x.prompt,
})
export const rowHasVideo = (xs: (SampleRow | null | undefined)[]) => xs.some((x) => x && isVideo(x))

export function useSamples(run: RunContext) {
  const { detail, sampling } = run
  // the run's own samples (its prompts at its steps) vs one-off "Try a prompt" renders
  const runSamples = computed(() => (detail.value?.samples ?? []).filter((x) => !x.manual))
  const steps = computed(() => [...new Set(runSamples.value.map((x) => x.step))].sort((a, b) => a - b))
  const prompts = computed(() => {
    const by = new Map<number, string>()
    for (const x of runSamples.value) by.set(x.index, x.prompt)
    return [...by.entries()].sort((a, b) => a[0] - b[0])
  })
  const cell = (index: number, s: number) => runSamples.value.find((x) => x.index === index && x.step === s)
  const latest = computed(() => {
    const s = steps.value.at(-1)
    if (s == null) return []
    const byIndex = new Map<number, SampleRow>()
    for (const x of runSamples.value) if (x.step === s) byIndex.set(x.index, x)
    return [...byIndex.values()].sort((a, b) => a.index - b.index)
  })
  const promptCount = computed(() => Math.max(detail.value?.spec.samples.prompts.length ?? 0, prompts.value.length))
  // each prompt's newest sample: mid-pass, the prompts already rendered show the new step
  // while the rest keep the previous one until their turn
  const latestByIndex = computed(() => {
    const by = new Map<number, SampleRow>()
    for (const x of runSamples.value) if ((by.get(x.index)?.step ?? -1) <= x.step) by.set(x.index, x)
    return by
  })
  const latestFor = (i: number) => latestByIndex.value.get(i)
  const promptText = (i: number) => latestFor(i)?.prompt ?? prompts.value.find(([k]) => k === i)?.[1] ?? (detail.value?.spec.samples.prompts[i] != null ? specPrompt(detail.value.spec.samples.prompts[i]!) : '')
  // edit runs: each prompt's input picture, and whether tiles wipe between it and the render. The
  // run gives it, found as training finds it (the dataset, ComfyUI's input, or the path as given:
  // a held-out folder, a picture on another drive); the input in the URL makes a changed one new
  const controlOf = (i: number) => {
    const sp = detail.value?.spec
    const p = sp?.dataset.edit ? sp.samples.prompts[i] : null
    return p && typeof p !== 'string' ? p.control || null : null
  }
  const inputFor = (i: number) => {
    const c = controlOf(i)
    return c ? sampleInputUrl(detail.value!.run.run_id, i, c) : null
  }
  const hasInputs = computed(() => (detail.value?.spec.samples.prompts ?? []).some((_p, i) => !!inputFor(i)))
  const compare = ref(true)
  const liveIndex = computed(() => (sampling.value?.phase === 'sampling' && !sampling.value.manual ? (sampling.value.index ?? -1) : -1))
  const liveManual = computed(() => (sampling.value?.manual ? sampling.value : null))

  const lb = ref<{ items: LightboxItem[]; index: number; grid?: number } | null>(null)
  // The lightbox walks the samples as a grid: a row per prompt, a column per step, so
  // left/right (A/D) is time and up/down (W/S) is the prompt. Only steps every prompt has
  // rendered make a column; a step still sampling opens as a lone row.
  function openGrid(index: number, s: number) {
    const rows = prompts.value.map(([i]) => i)
    const cols = steps.value.filter((st) => rows.every((i) => cell(i, st)))
    if (!cols.includes(s)) {
      const row = steps.value.map((st) => cell(index, st)).filter((x): x is SampleRow => !!x)
      lb.value = { items: row.map(sampleItem), index: Math.max(0, row.findIndex((x) => x.step === s)) }
      return
    }
    lb.value = {
      items: rows.flatMap((i) => cols.map((st) => sampleItem(cell(i, st)!))),
      index: rows.indexOf(index) * cols.length + cols.indexOf(s),
      grid: cols.length,
    }
  }

  // a row's videos playing together (Progression and Manual): one row at a time
  const syncing = ref<string | null>(null)
  let stopSync: (() => void) | null = null
  function toggleSync(key: string, row: Element | null | undefined) {
    stopSync?.()
    stopSync = null
    if (syncing.value === key || !row) {
      syncing.value = null
      return
    }
    syncing.value = key
    stopSync = playTogether([...row.querySelectorAll('video')])
  }
  onUnmounted(() => stopSync?.())
  const inSync = (v: HTMLVideoElement) => !!syncing.value && !!v.closest('[data-syncing]')
  const hoverPlay = (e: Event) => {
    const v = e.target as HTMLVideoElement
    if (!inSync(v)) void v.play().catch(() => {})
  }
  const hoverPause = (e: Event) => {
    const v = e.target as HTMLVideoElement
    if (!inSync(v)) v.pause()
  }

  return {
    steps, prompts, cell, latest, promptCount, latestFor, promptText, liveIndex, liveManual,
    lb, openGrid, syncing, toggleSync, hoverPlay, hoverPause, inputFor, hasInputs, compare,
  }
}

export type SamplesContext = ReturnType<typeof useSamples>
export const SAMPLES: InjectionKey<SamplesContext> = Symbol('samples')
export const useSamplesContext = () => inject(SAMPLES)!
