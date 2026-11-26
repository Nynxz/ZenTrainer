<script setup lang="ts">
// Dependencies box right below Start training: surfaces every auxiliary file this run needs
// (training adapter, base LoRA, sampling model/LoRA, face model, optional optimizer package)
// with install status and one-click download + live progress.
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { inputChoices } from '@/api/comfy'
import { EngineError, engine, type ExtraModel, type ModelDownload, type TrainingAdapter } from '@/api/engine'
import { downloadFor, finished, fmtBytes, startDownloadSync } from '@/stores/downloads'
import { useRunFormContext } from './form'

const { spec, choices, dataset, familyId, opt, sampleSuggestion } = useRunFormContext()

const trainAdapters = ref<TrainingAdapter[]>([])
const baseAdapters = ref<TrainingAdapter[]>([])
const extras = ref<ExtraModel[]>([])
const dlErrors = reactive<Record<string, string>>({})
const starting = reactive<Record<string, boolean>>({})

async function loadAll() {
  const [tr, ba, ex] = await Promise.all([
    engine.adapters(familyId.value, 'training').catch(() => []),
    engine.adapters(familyId.value, 'base').catch(() => []),
    engine.extraModels().catch(() => []),
  ])
  trainAdapters.value = tr
  baseAdapters.value = ba
  extras.value = ex
}

onMounted(() => {
  void startDownloadSync()
  void loadAll()
})
watch(familyId, () => void loadAll())
watch(finished, async () => {
  await loadAll()
  const [loras, unets, ckpts] = await Promise.all([
    inputChoices('LoraLoader', 'lora_name').catch(() => null),
    inputChoices('UNETLoader', 'unet_name').catch(() => null),
    inputChoices('CheckpointLoaderSimple', 'ckpt_name').catch(() => null),
  ])
  if (loras) choices.loras = loras
  if (unets) choices.unet = unets
  if (ckpts) choices.ckpt = ckpts
})

const basename = (p: string) => p.replace(/\\/g, '/').split('/').pop() ?? p
const hasLocal = (list: string[] | undefined, value: string) => {
  if (!list?.length) return false
  const norm = value.replace(/\\/g, '/')
  const base = basename(norm)
  return list.some((n) => {
    const x = n.replace(/\\/g, '/')
    return x === norm || basename(x) === base
  })
}

interface DepRow {
  key: string
  role: string
  label: string
  detail: string
  installed: boolean
  size?: number | null
  dl?: ModelDownload
  canDownload: boolean
  onDownload?: () => Promise<void>
  hint?: string
}

const deps = computed<DepRow[]>(() => {
  const rows: DepRow[] = []

  if (spec.network.training_adapter) {
    const want = spec.network.training_adapter
    const base = basename(want)
    const match = trainAdapters.value.find((a) => a.installed === want || a.name === base)
    const installed = Boolean(match?.installed || hasLocal(choices.loras, want))
    const dl = downloadFor('loras', match?.name ?? base)
    rows.push({
      key: `ta:${want}`,
      role: 'Training adapter',
      label: match?.label ?? base,
      detail: match?.name ?? want,
      installed,
      size: match?.size,
      dl,
      canDownload: !installed && Boolean(match && match.source !== 'local'),
      onDownload: match
        ? async () => {
            await engine.downloadAdapter(match.id)
          }
        : undefined,
      hint: !installed && !match ? 'Not found in your ComfyUI loras folder' : undefined,
    })
  }

  if (spec.network.base_lora) {
    const want = spec.network.base_lora
    const base = basename(want)
    const match = baseAdapters.value.find((a) => a.installed === want || a.name === base)
    const installed = Boolean(match?.installed || hasLocal(choices.loras, want))
    const dl = downloadFor('loras', match?.name ?? base)
    rows.push({
      key: `ba:${want}`,
      role: 'Base LoRA',
      label: match?.label ?? base,
      detail: match?.name ?? want,
      installed,
      size: match?.size,
      dl,
      canDownload: !installed && Boolean(match && match.source !== 'local'),
      onDownload: match
        ? async () => {
            await engine.downloadAdapter(match.id)
          }
        : undefined,
      hint: !installed && !match ? 'Not found in your ComfyUI loras folder' : undefined,
    })
  }

  if (spec.samples.model_name) {
    const want = spec.samples.model_name
    const list = spec.model.kind === 'split' ? choices.unet : choices.ckpt
    const folder = spec.model.kind === 'split' ? 'diffusion_models' : 'checkpoints'
    const installed = hasLocal(list, want)
    const sug = sampleSuggestion.value
    const matchSug = sug && sug.model === want ? sug : null
    const dl = downloadFor(folder, basename(want))
    if (!installed || dl) {
      rows.push({
        key: `sm:${want}`,
        role: 'Sampling model',
        label: basename(want),
        detail: folder,
        installed,
        size: matchSug?.size,
        dl,
        canDownload: !installed && Boolean(matchSug),
        onDownload: matchSug
          ? async () => {
              await engine.download(matchSug.family, matchSug.folder, matchSug.model)
            }
          : undefined,
        hint: !installed && !matchSug ? `Not found in ComfyUI ${folder}/` : undefined,
      })
    }
  }

  if (spec.samples.turbo_lora) {
    const want = spec.samples.turbo_lora
    const installed = hasLocal(choices.loras, want)
    if (!installed) {
      rows.push({
        key: `tl:${want}`,
        role: 'Sample LoRA',
        label: basename(want),
        detail: 'loras/',
        installed: false,
        canDownload: false,
        hint: 'Not found in your ComfyUI loras folder',
      })
    }
  }

  const needsFace = Boolean(dataset.value?.faces) && spec.samples.likeness !== false
  if (needsFace) {
    const fm = extras.value.find((e) => e.id === 'faces')
    if (fm) {
      const dl = fm.files.map((f) => downloadFor(fm.folder, f)).find((j) => j && ['queued', 'downloading', 'error'].includes(j.status))
        ?? downloadFor(fm.folder, fm.name)
      rows.push({
        key: 'extra:faces',
        role: 'Face likeness',
        label: fm.label,
        detail: `${fm.folder}/zentrainer/`,
        installed: fm.installed,
        size: fm.size,
        dl,
        canDownload: !fm.installed,
        onDownload: async () => {
          await engine.downloadExtraModel('faces')
        },
      })
    }
  }

  if (opt.value && !opt.value.available) {
    rows.push({
      key: `opt:${opt.value.id}`,
      role: 'Optimizer package',
      label: opt.value.label,
      detail: opt.value.pip || 'optional extra',
      installed: false,
      canDownload: false,
      hint: 'Run `scripts/zt extras optimizers` and restart ComfyUI',
    })
  }

  return rows
})

const missingCount = computed(() => deps.value.filter((d) => !d.installed).length)
const isActive = (dl?: ModelDownload) => Boolean(dl && (dl.status === 'queued' || dl.status === 'downloading'))
const pctOf = (dl?: ModelDownload) => (dl?.total ? Math.min(100, Math.floor((dl.done / dl.total) * 100)) : 0)

async function triggerDownload(row: DepRow) {
  if (!row.onDownload || starting[row.key]) return
  delete dlErrors[row.key]
  starting[row.key] = true
  try {
    await row.onDownload()
  } catch (e) {
    dlErrors[row.key] = e instanceof EngineError ? e.message : String(e)
  } finally {
    starting[row.key] = false
  }
}
const cancel = (dl?: ModelDownload) => dl && engine.cancelDownload(dl.id)
</script>

<template>
  <section v-if="deps.length" class="deps card" :class="{ missing: missingCount > 0 }">
    <div class="deps-head">
      <span class="deps-title"><i class="mdi mdi-package-variant-closed" /> Dependencies</span>
      <span v-if="missingCount" class="pill warn">{{ missingCount }} missing</span>
      <span v-else class="pill ok">Ready</span>
    </div>
    <ul class="dep-list">
      <li v-for="d in deps" :key="d.key" class="dep-row">
        <div class="dep-top">
          <i
            class="dep-ic mdi"
            :class="d.installed ? 'mdi-check-circle-outline ok' : isActive(d.dl) ? 'mdi-loading mdi-spin accent' : 'mdi-alert-circle-outline warn'"
          />
          <div class="dep-info">
            <div class="dep-name ellipsis" :title="d.detail">{{ d.label }}</div>
            <div class="dep-role muted">{{ d.role }}<template v-if="!d.installed && d.size"> · {{ fmtBytes(d.size) }}</template></div>
          </div>
          <button
            v-if="d.canDownload && !isActive(d.dl)"
            type="button"
            class="dl-btn"
            :disabled="starting[d.key]"
            @click="triggerDownload(d)"
          >
            <i class="mdi mdi-download" /> Download
          </button>
          <span v-else-if="d.installed" class="dep-tag ok">Installed</span>
        </div>
        <div v-if="isActive(d.dl)" class="dl-prog">
          <div class="dl-bar"><span :style="{ width: `${pctOf(d.dl)}%` }" /></div>
          <span class="mono">{{ d.dl?.status === 'queued' ? 'queued' : `${pctOf(d.dl)}%` }}</span>
          <span v-if="d.dl?.speed" class="mono muted">{{ fmtBytes(d.dl.speed) }}/s</span>
          <button type="button" class="dl-x" title="Cancel download" @click="cancel(d.dl)"><i class="mdi mdi-close" /></button>
        </div>
        <div v-if="d.hint && !d.installed" class="dep-hint warn">{{ d.hint }}</div>
        <div v-if="dlErrors[d.key] || d.dl?.status === 'error'" class="dep-err">
          {{ dlErrors[d.key] || d.dl?.error || 'Download failed' }}
        </div>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.deps {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px 16px;
}
.deps.missing {
  border-color: color-mix(in srgb, var(--zen-warn) 50%, var(--zen-border));
}
.deps-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.deps-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  color: var(--zen-text);
}
.pill {
  font-size: 10.5px;
  padding: 1px 7px;
  border-radius: 999px;
}
.pill.warn {
  background: color-mix(in srgb, var(--zen-warn) 16%, transparent);
  color: var(--zen-warn);
}
.pill.ok {
  background: color-mix(in srgb, var(--zen-success, var(--zen-accent)) 14%, transparent);
  color: var(--zen-success, var(--zen-accent));
}
.dep-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.dep-row {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.dep-top {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.dep-ic {
  font-size: 15px;
  flex: none;
}
.dep-ic.ok,
.dep-tag.ok {
  color: var(--zen-success, var(--zen-accent));
}
.dep-ic.warn,
.dep-hint.warn {
  color: var(--zen-warn);
}
.dep-ic.accent {
  color: var(--zen-accent);
}
.dep-info {
  flex: 1;
  min-width: 0;
}
.dep-name {
  font-size: 12px;
  color: var(--zen-text);
}
.dep-role {
  font-size: 10.5px;
}
.dep-tag {
  font-size: 10.5px;
  flex: none;
}
.dl-btn {
  flex: none;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 24px;
  padding: 0 8px;
  border: 1px solid color-mix(in srgb, var(--zen-accent) 55%, var(--zen-border));
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-accent) 12%, transparent);
  color: var(--zen-text);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.dl-btn:hover:not(:disabled) {
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 22%, transparent);
}
.dl-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.dl-prog {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10.5px;
  padding-left: 23px;
}
.dl-bar {
  flex: 1;
  height: 4px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--zen-text) 12%, transparent);
  overflow: hidden;
}
.dl-bar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
.dl-x {
  border: none;
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  padding: 0 2px;
  font-size: 13px;
}
.dl-x:hover {
  color: var(--zen-text);
}
.dep-hint,
.dep-err {
  font-size: 11px;
  padding-left: 23px;
  line-height: 1.35;
}
.dep-err {
  color: var(--zen-danger);
}
</style>
