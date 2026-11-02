<script setup lang="ts">
// Simple mode's view of a run: what it's doing in plain words, a big progress bar with the time
// left, and the samples as they come (before, then each one after). No loss chart, no knobs;
// the full run page is one click away.
import { ZenButton, ZenCompare, ZenLightbox, type LightboxItem } from '@nynxz/zenkit-ui'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { onComfyMessage } from '@/api/comfy'
import { ACTIVE, EVENTS, engine, promptText, viewUrl as fileUrl, type CheckpointRow, type RunDetail, type SampleRow } from '@/api/engine'
import { downloadUrl } from '@/api/engine/runs'
import { likeCeiling, likeShare } from '@/lib/likeness'
import { eta } from '@/lib/recipes'

const props = defineProps<{ id: string }>()
const d = ref<RunDetail | null>(null)
const sampling = ref<{ index?: number; count?: number; s_step?: number; s_total?: number; preview?: string; manual?: boolean } | null>(null)
const now = ref(Date.now() / 1000)
let timer: ReturnType<typeof setInterval> | undefined
let off = () => {}

async function load() {
  d.value = await engine.run(props.id).catch(() => d.value)
}
onMounted(() => {
  load()
  timer = setInterval(() => {
    now.value = Date.now() / 1000
    if (d.value && !ACTIVE.includes(d.value.run.status)) return
    if (Math.round(now.value) % 5 === 0) load() // status and messages, in case an event was missed
  }, 1000)
  off = onComfyMessage((m) => {
    if (!d.value || m.data.run_id !== props.id) return
    if (m.type === EVENTS.metrics) d.value.metrics.push(m.data as never)
    else if (m.type === EVENTS.sample) {
      const row = m.data as unknown as SampleRow
      d.value.samples.push(row)
      if (row.manual && !row.base) trying.value = Math.max(0, trying.value - 1)
    }
    else if (m.type === EVENTS.checkpoint) d.value.checkpoints.push(m.data as unknown as CheckpointRow)
    else if (m.type === EVENTS.status) {
      d.value.run.status = m.data.status as never
      d.value.run.message = (m.data.message as string) ?? undefined
      if (!ACTIVE.includes(d.value.run.status)) load()
    } else if (m.type === EVENTS.sampleProgress) {
      const p = m.data as Record<string, unknown>
      sampling.value = p.phase === 'done' ? null : { ...sampling.value, ...(p as object) }
    }
  })
})
onUnmounted(() => {
  off()
  clearInterval(timer)
})

const status = computed(() => d.value?.run.status)
const total = computed(() => d.value?.spec.optim.steps ?? 0)
const last = computed(() => d.value?.metrics.at(-1))
const step = computed(() => last.value?.step ?? 0)
const pct = computed(() => (total.value ? Math.min(100, (step.value / total.value) * 100) : 0))
const left = computed(() => (last.value?.it_s && status.value === 'training' ? (total.value - step.value) / last.value.it_s : null))
const done = computed(() => status.value === 'done')
const failed = computed(() => status.value === 'error' || status.value === 'rejected')
// a finished run, in brief: what it trained on and how long it took
const modelName = computed(() => {
  const m = d.value?.spec.model
  const f = m?.unet_name ?? m?.ckpt_name
  return f ? f.split('/').pop()!.replace(/\.(safetensors|ckpt|gguf)$/i, '') : ''
})
// one go only: a resumed run's earlier goes aren't timed, and the gaps between them aren't training
const took = computed(() => {
  const r = d.value?.run
  const resumed = r?.first_started && r.first_started !== r.started
  return done.value && !resumed && r?.started && r.ended ? eta(r.ended - r.started).replace(/^about /, '') : ''
})

const headline = computed(() => {
  const s = status.value
  if (!s) return 'Loading…'
  if (s === 'queued') return 'Waiting for its turn'
  if (s === 'loading') return 'Getting ready'
  if (sampling.value?.manual) return 'Drawing your prompt'
  if (sampling.value) return step.value === 0 ? 'Making a "before" picture' : 'Making a new preview'
  if (s === 'training') return step.value ? 'Learning' : 'Starting'
  if (s === 'paused') return 'Paused'
  if (s === 'done') return 'Your LoRA is ready'
  if (s === 'stopped' || s === 'interrupted') return 'Stopped'
  return 'Something went wrong'
})
const detailLine = computed(() => {
  const s = status.value
  if (s === 'loading') return d.value?.run.message || 'Loading the model and your photos'
  if (s === 'paused') return d.value?.run.message || ''
  if (sampling.value?.s_total) return `${sampling.value.s_step ?? 0} of ${sampling.value.s_total}`
  if (s === 'training' && left.value != null) return `${eta(left.value)} left`
  if (failed.value) return d.value?.run.message ?? ''
  if (done.value) return d.value?.spec.dataset.trigger ? 'Use the word below in a prompt to bring it in.' : 'Slide across the picture to compare, then try it.'
  return ''
})

// the samples: the first prompt's, in order (before first); the newest big
const showPreviews = computed(() => !!d.value && (!done.value || rows.value.length <= 1 || !compare.value))
const viewUrl = (x: SampleRow) => fileUrl(x)
const isVideo = (x: SampleRow) => /\.(mp4|webm)$/i.test(x.filename)
const runSamples = computed(() => (d.value?.samples ?? []).filter((x) => !x.manual))
const strip = computed(() => runSamples.value.filter((x) => x.index === 0).sort((a, b) => a.step - b.step))
const pick = ref<number | null>(null)
const big = computed(() => (pick.value != null ? strip.value[pick.value] : strip.value.at(-1)) ?? null)
// while a preview draws, the stage shows it live, at the samples' own shape (ComfyUI's live frames
// are small: scaled up to fill the stage rather than sitting tiny in the middle)
const live = computed(() => (sampling.value && !sampling.value.manual && sampling.value.preview && pick.value == null ? sampling.value : null))
const sampleAspect = computed(() => {
  const sp = d.value?.spec.samples
  return sp?.width && sp?.height ? `${sp.width} / ${sp.height}` : '1 / 1'
})
const finalLora = computed(() => [...(d.value?.checkpoints ?? [])].reverse().find((c) => c.final)?.lora_name ?? null)
const tryLink = computed(() => (finalLora.value ? tryFor(finalLora.value) : null))
const tryFor = (lora: string) => `/apps/text-to-image?run=${props.id}&lora=${encodeURIComponent(lora)}`

// ---- every saved checkpoint with its samples: pick the one that looks best ------------------
// rows newest first, one column per sample prompt (the same prompt lines up down the page)
const prompts = computed(() => (d.value?.spec.samples.prompts ?? []).map(promptText))
const cols = computed(() => Math.max(1, prompts.value.length))
// the column labels without the word every prompt starts with (the trigger): what tells them apart
const promptLabels = computed(() => {
  const first = prompts.value.map((t) => /^\s*([^,\s]+),\s*/.exec(t))
  const lead = first[0]?.[1]
  const shared = prompts.value.length > 1 && lead && first.every((m) => m?.[1] === lead)
  return prompts.value.map((t, i) => (shared ? t.slice(first[i]![0].length) : t))
})
const rows = computed(() => {
  const r = d.value
  if (!r) return []
  const samples = runSamples.value
  const steps = [...new Set(samples.map((x) => x.step))].sort((a, b) => a - b)
  const at = (step: number) => {
    // the samples taken at this checkpoint, or the latest ones before it
    const s = [...steps].reverse().find((x) => x <= step)
    return Array.from({ length: cols.value }, (_, i) => samples.find((x) => x.step === s && x.index === i) ?? null)
  }
  const ck = [...r.checkpoints].sort((a, b) => b.step - a.step)
  const out = ck.map((c) => ({ key: `c${c.step}`, label: c.final ? 'Final' : `Step ${c.step}`, step: c.step, ckpt: c as CheckpointRow | null, recommended: c.step === best.value?.step, samples: at(c.step) }))
  if (steps[0] === 0) out.push({ key: 'base', label: 'Before training', step: 0, ckpt: null, recommended: false, samples: at(0) })
  return out
})
// the checkpoint to recommend: the one whose samples look most like the photos (faces on), else the newest
const likenessAt = (step: number) => {
  const xs = (d.value?.likeness ?? []).filter((x) => x.step === step && !x.manual && x.score != null).map((x) => x.score!)
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null
}
const best = computed(() => {
  const ck = [...(d.value?.checkpoints ?? [])].sort((a, b) => b.step - a.step)
  if (!ck.length) return null
  const scored = ck.map((c) => ({ c, like: likenessAt(c.step) })).filter((x) => x.like != null)
  if (scored.length) {
    const top = scored.reduce((a, b) => (b.like! > a.like! + 0.005 ? b : a))  // later wins a near tie (it's first)
    const ceil = likeCeiling(d.value?.likeness)
    const share = ceil ? `, ${likeShare(top.like!, ceil)} as alike as your photos are to each other` : ''
    return { step: top.c.step, why: `its samples look most like your photos (likeness ${top.like!.toFixed(2)}${share})` }
  }
  return { step: ck[0]!.step, why: 'the newest: compare them below, the one you like best is the one to keep' }
})
// done: the first prompt before training against the recommended checkpoint, slid across
const compare = computed(() => {
  if (!done.value || !best.value) return null
  const before = runSamples.value.find((x) => x.step === 0 && x.index === 0)
  const row = rows.value.find((r) => r.recommended)
  const after = row?.samples[0]
  if (!before || !after || isVideo(before) || isVideo(after)) return null
  // the widget fills its box: the box takes the sample's shape (square when it isn't known)
  return { before: viewUrl(before), after: viewUrl(after), label: row!.label, ratio: after.width && after.height ? `${after.width} / ${after.height}` : '1 / 1' }
})
// the run's own download route: the file's path says nothing about /view once a ZenTrainer folder is set
const download = (c: CheckpointRow) => (c.path ? downloadUrl(props.id, c.lora_name) : null)
const mb = (b?: number | null) => (b ? `${(b / 1e6).toFixed(0)} MB` : '')

// full screen: ←/→ across prompts, ↑/↓ the same prompt at other checkpoints
const lb = ref<number | null>(null)
const lbItems = computed<LightboxItem[]>(() =>
  rows.value.flatMap((r) =>
    r.samples.map((x, i) =>
      x
        ? { src: viewUrl(x), kind: isVideo(x) ? ('video' as const) : ('image' as const), label: `${r.label} · ${i + 1}`, meta: x.prompt }
        : { src: '', label: `${r.label} · ${i + 1}`, meta: 'no sample' },
    ),
  ),
)
const open = (row: number, col: number) => (lb.value = row * cols.value + col)

// ---- the bar's notches: each preview (a checkpoint and its samples), filled once it's taken ----
const marks = computed(() => {
  const every = d.value?.spec.samples.every
  if (!every) return []
  const taken = new Set(runSamples.value.map((x) => x.step))
  const out: { step: number; done: boolean }[] = []
  for (let s = every; s <= total.value; s += every) out.push({ step: s, done: taken.has(s) })
  return out
})
const nextPreview = computed(() => {
  const m = marks.value.find((x) => !x.done && x.step > step.value)
  if (!m || status.value !== 'training' || !last.value?.it_s) return null
  return { step: m.step, in: eta((m.step - step.value) / last.value.it_s) }
})

// ---- Try a prompt: drawn with what it has learned so far, while it trains ----------------------
const tryText = ref('')
const trying = ref(0) // asked for, not back yet
const tryError = ref('')
const tries = computed(() => (d.value?.samples ?? []).filter((x) => x.manual && !x.base).sort((a, b) => b.index - a.index))
async function tryIt() {
  const prompt = tryText.value.trim()
  if (!prompt) return
  tryText.value = ''
  tryError.value = ''
  trying.value++
  try {
    await engine.control(props.id, 'sample', { prompt })
  } catch (e) {
    trying.value--
    tryError.value = e instanceof Error ? e.message : String(e)
  }
}
</script>

<template>
  <div class="page srun">
    <!-- the before/after beside everything about it: what it is, which checkpoint to keep, what next -->
    <div v-if="d" class="hero card" :class="{ done, failed, 'with-media': compare || showPreviews }">
      <div v-if="compare" class="hero-compare" :style="{ aspectRatio: compare.ratio }">
        <ZenCompare :before="compare.before" :after="compare.after" before-label="Before training" :after-label="compare.label" />
      </div>
      <!-- while it trains (or with nothing to compare): the preview being drawn, else the newest, and the earlier ones under it -->
      <div v-else-if="showPreviews" class="previews">
        <div v-if="live" class="stage live" :style="{ aspectRatio: sampleAspect }">
          <img :src="live.preview" alt="" />
          <span class="tag small">Drawing{{ live.s_total ? ` ${live.s_step ?? 0}/${live.s_total}` : '…' }}</span>
          <div class="live-bar"><span :style="{ width: `${live.s_total ? ((live.s_step ?? 0) / live.s_total) * 100 : 0}%` }" /></div>
        </div>
        <div v-else-if="big" class="stage">
          <video v-if="isVideo(big)" :key="big.filename" :src="viewUrl(big)" autoplay loop muted playsinline controls />
          <img v-else :src="viewUrl(big)" alt="" />
          <span class="tag small">{{ big.step === 0 ? 'Before training' : `After ${big.step} steps` }}</span>
        </div>
        <div v-else class="muted small">The first preview shows up in a few minutes.</div>
        <div v-if="strip.length > 1" class="strip">
          <button v-for="(x, i) in strip" :key="x.filename" :class="{ on: big === x }" @click="pick = i === strip.length - 1 ? null : i">
            <video v-if="isVideo(x)" :src="viewUrl(x)" muted preload="metadata" />
            <img v-else :src="viewUrl(x)" loading="lazy" alt="" />
            <span class="small">{{ x.step === 0 ? 'Before' : x.step }}</span>
          </button>
        </div>
      </div>
      <div class="hero-side">
        <div class="top">
          <div>
            <div class="eyebrow">{{ d.spec.name }}</div>
            <h1>{{ headline }}</h1>
            <p class="muted">{{ detailLine }}</p>
          </div>
          <i v-if="!done" class="big-ic" :class="failed ? 'mdi mdi-alert-circle-outline' : 'mdi mdi-sprout-outline'" />
        </div>
        <div v-if="!done && !failed" class="track">
          <div class="bar"><span :style="{ width: `${pct}%` }" /></div>
          <span
            v-for="m in marks"
            :key="m.step"
            class="notch"
            :class="{ done: m.done }"
            :style="{ left: `${(m.step / total) * 100}%` }"
            :title="`Preview at step ${m.step}${m.done ? ' (taken)' : ''}`"
          />
        </div>
        <div v-if="done && best && rows.length > 2" class="why">
          <div class="eyebrow"><i class="mdi mdi-star-outline" /> Recommended</div>
          <b>{{ rows.find((r) => r.recommended)?.label }}</b>
          <span class="small muted">{{ best.why[0]?.toUpperCase() }}{{ best.why.slice(1) }}.</span>
        </div>
        <dl class="facts small">
          <template v-if="!done && !failed">
            <dt>Step</dt><dd class="mono">{{ step }} / {{ total }} <span class="muted">· {{ Math.round(pct) }}%</span></dd>
          </template>
          <template v-if="nextPreview"><dt>Next preview</dt><dd>step {{ nextPreview.step }}, {{ nextPreview.in }}</dd></template>
          <template v-if="d.spec.dataset.trigger"><dt>Trigger word</dt><dd><b class="mono">{{ d.spec.dataset.trigger }}</b></dd></template>
          <template v-if="done && modelName"><dt>Model</dt><dd class="mono">{{ modelName }}</dd></template>
          <template v-if="done"><dt>Trained</dt><dd>{{ total.toLocaleString() }} steps<template v-if="took">, {{ took }}</template></dd></template>
        </dl>
        <div class="acts">
          <template v-if="done">
            <RouterLink v-if="tryLink" :to="tryLink"><ZenButton variant="primary" icon="mdi mdi-creation-outline">Try it</ZenButton></RouterLink>
            <RouterLink to="/simple"><ZenButton icon="mdi mdi-plus">Train another</ZenButton></RouterLink>
          </template>
          <span class="spacer" />
          <RouterLink :to="`/runs/${id}`" class="small muted">All the details <i class="mdi mdi-chevron-right" /></RouterLink>
        </div>
      </div>
    </div>

    <section v-if="d && (status === 'training' || tries.length || trying)" class="card tries">
      <div class="picks-head">
        <div class="eyebrow">Try a prompt</div>
        <span class="muted small">Drawn with what it has learned so far. Training waits while it draws.</span>
      </div>
      <div v-if="status === 'training'" class="try">
        <input
          v-model="tryText"
          class="try-in"
          :placeholder="`e.g. ${d.spec.dataset.trigger || 'it'} reading a book in a cafe`"
          @keydown.enter="tryIt"
        />
        <ZenButton variant="primary" icon="mdi mdi-creation-outline" :disabled="!tryText.trim()" @click="tryIt">Try</ZenButton>
      </div>
      <div v-if="tryError" class="small danger">{{ tryError }}</div>
      <div v-if="tries.length || trying" class="try-grid">
        <div v-if="trying" class="tcell pending">
          <img v-if="sampling?.manual && sampling.preview" :src="sampling.preview" alt="" />
          <span class="small muted">{{ sampling?.manual ? `Drawing… ${sampling.s_step ?? 0}/${sampling.s_total ?? '?'}` : trying > 1 ? `${trying} waiting` : 'Waiting for the next step' }}</span>
        </div>
        <a v-for="x in tries" :key="x.filename" class="tcell" :href="viewUrl(x)" target="_blank" :title="x.prompt">
          <img :src="viewUrl(x)" loading="lazy" alt="" />
          <span class="small"><b>Step {{ x.step }}</b> <span class="muted ellipsis">{{ x.prompt }}</span></span>
        </a>
      </div>
    </section>

    <section v-if="d && rows.length > 1" class="card picks">
      <div class="picks-head">
        <div class="eyebrow">Checkpoints</div>
        <span class="muted small">{{ ACTIVE.includes(d.run.status) ? 'New ones appear as it trains.' : 'Compare them and keep the one you like.' }}</span>
      </div>
      <div class="ptable" :style="{ '--cols': cols }">
        <div class="phead">
          <span />
          <span v-for="(p, i) in promptLabels" :key="i" class="pp" :title="prompts[i]">{{ p }}</span>
        </div>
        <div v-for="(r, ri) in rows" :key="r.key" class="prow" :class="{ rec: r.recommended }">
          <div class="pinfo">
            <b>{{ r.label }}</b>
            <span v-if="r.recommended" class="rec-tag small">Recommended</span>
            <div v-if="r.ckpt" class="pacts">
              <a v-if="download(r.ckpt)" :href="download(r.ckpt)!" :download="r.ckpt.lora_name.split('/').pop()" class="mini" title="Download this LoRA"><i class="mdi mdi-download" /> {{ mb(r.ckpt.size) }}</a>
              <RouterLink :to="tryFor(r.ckpt.lora_name)" class="mini" title="Generate with this checkpoint"><i class="mdi mdi-creation-outline" /> Try</RouterLink>
              <RouterLink :to="`/runs/new?from=${id}&init=${encodeURIComponent(r.ckpt.lora_name)}`" class="mini" title="A new run that continues from this checkpoint"><i class="mdi mdi-plus" /> Train more</RouterLink>
            </div>
          </div>
          <button v-for="(x, ci) in r.samples" :key="ci" class="pcell" :disabled="!x" @click="open(ri, ci)">
            <template v-if="x">
              <video v-if="isVideo(x)" :src="viewUrl(x)" muted loop playsinline preload="metadata" @mouseenter="($event.target as HTMLVideoElement).play()" @mouseleave="($event.target as HTMLVideoElement).pause()" />
              <img v-else :src="viewUrl(x)" loading="lazy" alt="" />
            </template>
            <span v-else class="none small muted">No sample</span>
          </button>
        </div>
      </div>
      <ZenLightbox v-if="lb != null" v-model:index="lb" :items="lbItems" :grid="cols" :searchable="false" @close="lb = null" />
    </section>

  </div>
</template>

<style scoped>
.srun {
  max-width: 1120px;
}
.hero-compare {
  width: 100%;
  max-height: calc(70vh / var(--zen-page-zoom));
  border-radius: var(--r-lg, 12px);
  overflow: hidden;
}
.hero {
  display: flex;
  flex-direction: column;
  gap: 22px;
  padding: 22px;
}
/* the picture on the left, the rest beside it; stacked on a narrow screen */
.hero.with-media {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(280px, 1fr);
  align-items: start;
}
@media (max-width: 900px) {
  .hero.with-media {
    grid-template-columns: 1fr;
  }
}
.hero-side {
  display: flex;
  flex-direction: column;
  gap: 18px;
  min-width: 0;
}
.why {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 14px 16px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-accent) 35%, var(--zen-border));
  background: color-mix(in srgb, var(--zen-accent) 6%, transparent);
}
.why .eyebrow i {
  color: var(--zen-accent);
}
.why b {
  font-size: 18px;
}
.facts {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 16px;
  margin: 0;
}
.facts dt {
  color: var(--zen-muted);
}
.facts dd {
  margin: 0;
}
.facts:empty {
  display: none;
}
.top {
  display: flex;
  justify-content: space-between;
  gap: 16px;
}
.top h1 {
  margin: 4px 0;
  font-size: 26px;
}
.top p {
  margin: 0;
}
.big-ic {
  font-size: 44px;
  color: var(--zen-accent);
}
.done .big-ic {
  color: var(--zen-ok);
}
.failed .big-ic {
  color: var(--zen-danger);
}
.track {
  position: relative;
}
/* a dot at each preview: hollow until it's taken */
.notch {
  position: absolute;
  top: 2px;
  width: 8px;
  height: 8px;
  margin-left: -4px;
  border-radius: 50%;
  background: var(--zen-surface);
  border: 1.5px solid var(--zen-muted);
  box-sizing: border-box;
  pointer-events: auto;
}
.notch.done {
  background: var(--zen-text);
  border-color: var(--zen-text);
  opacity: 0.85;
}
.tries {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.try {
  display: flex;
  gap: 8px;
}
.try-in {
  flex: 1;
  min-width: 0;
  font: inherit;
  font-size: 13px;
  padding: 8px 10px;
  border-radius: var(--r-md);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: var(--zen-field-bg, var(--zen-input));
  color: var(--zen-text);
}
.try-in:focus {
  outline: none;
  border-color: var(--zen-accent);
}
.try-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 8px;
}
.tcell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  color: inherit;
  text-decoration: none;
}
.tcell img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
}
.tcell > span {
  display: flex;
  gap: 4px;
  min-width: 0;
}
.tcell.pending {
  justify-content: center;
}
.tcell.pending img + span {
  text-align: center;
}
.tcell.pending:not(:has(img))::before {
  content: '';
  display: block;
  aspect-ratio: 1;
  border-radius: var(--r-md);
  background: var(--zen-surface-2);
}
.danger {
  color: var(--zen-danger);
}
.bar {
  height: 12px;
  border-radius: 6px;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.bar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
.acts {
  display: flex;
  align-items: center;
  gap: 8px;
}
.previews {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.stage {
  position: relative;
  display: grid;
  place-items: center;
  background: var(--zen-surface-2);
  border-radius: var(--r-lg, 12px);
  overflow: hidden;
  max-height: calc(60vh / var(--zen-page-zoom));
}
.stage.live {
  width: 100%;
}
.stage.live img {
  width: 100%;
  height: 100%;
  max-width: none;
  max-height: none;
  object-fit: contain;
}
.live-bar {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 4px;
  background: rgb(0 0 0 / 0.35);
}
.live-bar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
.stage img,
.stage video {
  max-width: 100%;
  max-height: calc(60vh / var(--zen-page-zoom));
  object-fit: contain;
  display: block;
}
.tag {
  position: absolute;
  left: 10px;
  top: 10px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
}
.strip {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  scrollbar-width: thin;
}
.strip button {
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: center;
  border: 2px solid transparent;
  background: none;
  padding: 2px;
  border-radius: var(--r-md);
  color: var(--zen-muted);
  cursor: pointer;
}
.strip button.on {
  border-color: var(--zen-selected, var(--zen-accent));
  color: var(--zen-selected-text, var(--zen-text));
}
.strip img,
.strip video {
  width: 88px;
  height: 88px;
  object-fit: cover;
  border-radius: var(--r-sm);
  display: block;
}
.picks {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.picks-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.ptable {
  display: flex;
  flex-direction: column;
  align-items: flex-start; /* rows as wide as their pictures, not the card */
  gap: 8px;
  overflow-x: auto;
  scrollbar-width: thin;
}
.phead,
.prow {
  display: grid;
  grid-template-columns: 170px repeat(var(--cols), minmax(120px, 180px));
  gap: 8px;
  align-items: center;
}
/* the header lines up with the rows' cells: their padding and border */
.phead {
  padding: 0 9px;
  align-items: end;
}
.pp {
  font-size: 11px;
  line-height: 1.35;
  color: var(--zen-muted);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.prow {
  padding: 8px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 60%, transparent);
  border-radius: var(--r-md);
}
.prow.rec {
  border-color: color-mix(in srgb, var(--zen-accent) 60%, transparent);
  background: color-mix(in srgb, var(--zen-accent) 5%, transparent);
}
.pinfo {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: flex-start;
}
.rec-tag {
  color: var(--zen-accent);
}
.pacts {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.mini {
  font-size: 11px;
  padding: 2px 6px;
  border-radius: var(--r-sm);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  color: var(--zen-muted);
}
.mini:hover {
  color: var(--zen-text);
  border-color: var(--zen-accent);
}
.pcell {
  aspect-ratio: 1;
  border: none;
  padding: 0;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--zen-surface-2);
  cursor: pointer;
  display: grid;
  place-items: center;
}
.pcell:disabled {
  cursor: default;
  border: 1px dashed color-mix(in srgb, var(--zen-border) 80%, transparent);
  background: none;
}
.pcell img,
.pcell video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
</style>
