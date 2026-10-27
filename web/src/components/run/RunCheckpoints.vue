<script setup lang="ts">
// The rail's Checkpoints card: one line per saved step, newest first, with its likeness and
// validation loss; the best one marked; hover a line to test it, train more from it, download it
// or copy its name.
import { ZenTip } from '@nynxz/zenkit-ui'
import { computed, ref } from 'vue'
import { downloadUrl } from '@/api/engine/runs'
import { likeShare } from '@/lib/likeness'
import { fmtBytes } from '@/stores/downloads'
import { ckKind, useRunContext } from './context'

const { id, detail, total, running, ckRows, nextSave, likeAt, likeCeil, ckptVal, resumable, resumeFrom } = useRunContext()

// the best checkpoint: highest likeness when faces are scored (a character LoRA), else lowest validation loss
const bestBy = computed(() => {
  const cks = detail.value?.checkpoints ?? []
  return cks.filter((c) => likeAt(c.step) != null).length >= 2 ? 'likeness' : 'val'
})
const bestStep = computed(() => {
  const cks = (detail.value?.checkpoints ?? []).filter((c) => (bestBy.value === 'likeness' ? likeAt(c.step) : ckptVal(c.step)) != null)
  if (cks.length < 2) return null
  if (bestBy.value === 'likeness') return cks.reduce((a, b) => (likeAt(b.step)! > likeAt(a.step)! ? b : a)).step
  return cks.reduce((a, b) => (ckptVal(b.step)! < ckptVal(a.step)! ? b : a)).step
})
// every checkpoint of a run is the same size: said once, in the header
const ckSize = computed(() => {
  const sizes = new Set(ckRows.value.map((c) => c.size).filter((v): v is number => !!v))
  return sizes.size ? `${fmtBytes(Math.max(...sizes))} each` : ''
})
const saveLine = computed(() => {
  const n = ckRows.value.length
  const next = nextSave.value
  const plan = detail.value?.spec.output.save_every ? `every ${detail.value.spec.output.save_every}` : 'final only'
  if (running.value && next) return `${n ? `${n} · ` : ''}next ${next.step} · ${next.in}`
  return [n ? `${n} · ${plan}` : plan, ckSize.value].filter(Boolean).join(' · ')
})
// what output.keep removed along the way, newest first, each with why
const pruned = computed(() => [...(detail.value?.pruned ?? [])].sort((a, b) => b.step - a.step))
const keepLine = computed(() => {
  const o = detail.value?.spec.output
  if (!o || !o.keep || o.keep === 'all') return ''
  if (o.keep === 'likeness_new_best') return 'Keeping each new best likeness, the newest and the final'
  if (o.keep === 'val_loss_new_best') return 'Keeping each new lowest validation loss, the newest and the final'
  const what = { latest: 'the latest', likeness: 'the best likeness', val_loss: 'the lowest validation loss' }[o.keep]
  return `Keeping ${what} ${o.keep_count}, the newest and the final`
})
const copied = ref<string | null>(null)
async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    copied.value = text
    setTimeout(() => copied.value === text && (copied.value = null), 1500)
  } catch {
    // clipboard blocked
  }
}
</script>

<template>
  <section class="card">
    <header class="sec-head">
      <span class="sec-title"><ZenTip text="The LoRA saved along the way. Best: the highest likeness when faces are scored (a character), else the lowest validation loss. Hover a row to test it, train more from it, download it or copy its name">Checkpoints</ZenTip></span>
      <span class="sec-sub">{{ saveLine }}</span>
    </header>
    <div v-if="!ckRows.length" class="muted small">
      {{ running ? `The first save lands at step ${nextSave?.step ?? total}.` : 'None saved.' }}
    </div>
    <div v-if="keepLine" class="muted small keep-line">
      {{ keepLine }}<template v-if="pruned.length">
        ·
        <ZenTip :text="pruned.map((p) => `${p.step}: ${p.reason}`).join('\n')">{{ pruned.length }} removed</ZenTip>
      </template>
    </div>
    <div class="ckpts">
      <div
        v-for="c in ckRows"
        :key="c.path"
        class="ckpt"
        :class="{ best: c.step === bestStep }"
        :title="c.size === null ? `File missing: ${c.path}` : c.path"
        tabindex="-1"
      >
        <span class="ck-step mono" :class="{ gone: c.size === null }">{{ c.step }}</span>
        <span v-if="c.final" class="chip">final</span>
        <span v-else-if="ckKind(c)" class="chip muted" :title="`Saved when the run ${ckKind(c) === 'manual' ? 'was asked to' : ckKind(c)}`">{{ ckKind(c) }}</span>
        <span v-if="c.step === bestStep" class="chip best-chip" :title="bestBy === 'likeness' ? 'Its samples look most like the dataset\'s faces' : 'Lowest validation loss of the saved checkpoints'">best</span>
        <span class="spacer" />
        <span class="ck-val mono">
          <template v-if="likeAt(c.step) != null"><span :title="likeCeil ? `How much its samples' faces look like the dataset's (same person ≳ 0.5); ${likeShare(likeAt(c.step)!, likeCeil)} of how alike your photos are to each other (${likeCeil.toFixed(2)})` : 'How much its samples\' faces look like the dataset\'s (same person ≳ 0.5)'">♥ {{ likeAt(c.step)!.toFixed(2) }}<span v-if="likeCeil" class="share"> ({{ likeShare(likeAt(c.step)!, likeCeil) }})</span></span> · </template>
          <span title="Validation loss">{{ ckptVal(c.step) != null ? ckptVal(c.step)!.toFixed(4) : '—' }}</span>
        </span>
        <span class="ck-acts">
          <RouterLink class="ib" :to="`/apps/test-lora?run=${id}&lora=${encodeURIComponent(c.lora_name)}`" title="Test: try it in Test LoRA, on this run's model"><i class="mdi mdi-flask-outline" /></RouterLink>
          <button v-if="resumable && !c.final && c.state" class="ib" title="Resume this run from here, optimizer state and all" @click="resumeFrom(c)"><i class="mdi mdi-play-outline" /></button>
          <RouterLink v-else class="ib" :to="`/runs/new?from=${id}&init=${encodeURIComponent(c.lora_name)}`" title="Train more: a new run with these settings that starts from this LoRA"><i class="mdi mdi-fast-forward-outline" /></RouterLink>
          <a v-if="c.size !== null" class="ib" :href="downloadUrl(id, c.lora_name)" download :title="`Download ${c.lora_name.split('/').pop()}`"><i class="mdi mdi-download-outline" /></a>
          <button class="ib" :title="copied === c.lora_name ? 'Copied' : `Copy LoRA name: ${c.lora_name}`" @click="copy(c.lora_name)">
            <i :class="copied === c.lora_name ? 'mdi mdi-check' : 'mdi mdi-content-copy'" />
          </button>
        </span>
      </div>
    </div>
  </section>
</template>

<style scoped src="./section.css"></style>
<style scoped>
.ckpts {
  display: flex;
  flex-direction: column;
  gap: 1px;
}
/* one line per checkpoint; its actions take the numbers' place on hover */
.ckpts {
  container-type: inline-size;
}
/* a narrow rail: the likeness share goes first (it's in the hover too) */
@container (max-width: 290px) {
  .ckpts .share {
    display: none;
  }
}
.ckpts .ckpt {
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
  height: 32px;
  padding: 0 8px;
  margin: 0 -8px;
  border-radius: var(--r-md);
  font-size: 12px;
}
.ckpts .ckpt:hover,
.ckpts .ckpt:focus-within {
  background: color-mix(in srgb, var(--zen-surface-2) 60%, transparent);
}
.ckpts .ckpt.best {
  background: color-mix(in srgb, var(--zen-ok) 8%, transparent);
}
.ckpts .ck-step {
  font-weight: 650;
  font-size: 13px;
  min-width: 4ch;
}
.ckpts .ck-step.gone {
  color: var(--zen-warn);
  text-decoration: line-through;
}
.ckpts .ckpt .chip {
  font-size: 10px;
  padding: 0 6px;
  line-height: 16px;
  /* the chips give way, not the numbers: a row that's final and best still shows its loss */
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ckpts .ckpt .chip.muted {
  color: var(--zen-muted);
}
.ckpts .ck-val {
  font-size: 11px;
  color: var(--zen-muted);
  white-space: nowrap;
  flex-shrink: 0;
  transition: opacity 0.12s ease;
}
.ckpts .ck-acts {
  position: absolute;
  right: 4px;
  top: 50%;
  translate: 0 -50%;
  display: flex;
  gap: 2px;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
}
.ckpts .ckpt:hover .ck-acts,
.ckpts .ckpt:focus-within .ck-acts {
  opacity: 1;
  pointer-events: auto;
}
.ckpts .ckpt:hover .ck-val,
.ckpts .ckpt:focus-within .ck-val {
  opacity: 0;
}
.keep-line {
  margin-bottom: 4px;
}
.best-chip {
  color: var(--zen-ok);
}
.ib {
  width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
}
.ib:hover {
  color: var(--zen-text);
  background: var(--zen-surface-2);
}
</style>
