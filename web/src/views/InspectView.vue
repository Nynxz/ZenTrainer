<script setup lang="ts">
// Inspect a .safetensors file, as Hugging Face's file viewer does: its tensors as a tree of
// layers (shape, precision, parameters), its metadata, and totals; read from the header alone,
// so a whole model opens at once. Save a copy beside it: other metadata, another precision,
// layers left out (engine/zentrainer/models/tensors.py).
import { ZenButton, ZenCombo, ZenEmpty, ZenIconButton, ZenInput, ZenToggleGroup } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { EngineError, engine, type TensorFileInfo, type TensorFolder, type TensorRow } from '@/api/engine'
import { fmtBytes } from '@/stores/downloads'

const route = useRoute()
const router = useRouter()

const FOLDER_LABELS: Record<TensorFolder, string> = {
  loras: 'LoRAs',
  diffusion_models: 'Diffusion models',
  checkpoints: 'Checkpoints',
  vae: 'VAEs',
  text_encoders: 'Text encoders',
}
const FOLDER_ICONS: Record<TensorFolder, string> = {
  loras: 'mdi mdi-puzzle-outline',
  diffusion_models: 'mdi mdi-cube-outline',
  checkpoints: 'mdi mdi-cube-outline',
  vae: 'mdi mdi-image-filter-center-focus-weak',
  text_encoders: 'mdi mdi-text-box-outline',
}
const files = ref<Partial<Record<TensorFolder, string[]>>>({})
const folder = ref<string>((route.query.folder as string) || 'loras')
const name = ref<string>((route.query.name as string) || '')
const info = ref<TensorFileInfo | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const folderOptions = computed(() =>
  (Object.keys(FOLDER_LABELS) as TensorFolder[])
    .filter((f) => files.value[f])
    .map((f) => ({ value: f, label: FOLDER_LABELS[f], count: files.value[f]?.length ?? 0, icon: FOLDER_ICONS[f] })),
)
const fileItems = computed(() => (files.value[folder.value as TensorFolder] ?? []).map((n) => ({ value: n, label: n })))

onMounted(async () => {
  files.value = await engine.tensorFiles().catch(() => ({}))
})

// ---- numbers ----------------------------------------------------------------------------------
const fmtParams = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n)
const kindLine = computed(() => {
  const i = info.value
  if (!i) return ''
  if (i.kind === 'lora') {
    const r = i.ranks?.length ? ` · rank ${i.ranks.length > 1 ? `${i.ranks[0]}–${i.ranks[i.ranks.length - 1]}` : i.ranks[0]}` : ''
    return `${i.algorithm} · ${i.layers} layers${r}`
  }
  return FOLDER_LABELS[folder.value as TensorFolder]?.replace(/s$/, '') ?? 'Model'
})
const fileBase = computed(() => name.value.split('/').pop() ?? '')
const fileDir = computed(() => name.value.split('/').slice(0, -1).join('/'))

// each precision its own colour, the same in the bar, its legend and the tree
const DT_COLORS: Record<string, string> = {
  F64: '#38bdf8',
  F32: '#60a5fa',
  BF16: '#a78bfa',
  F16: '#2dd4bf',
  F8_E4M3: '#fbbf24',
  F8_E5M2: '#f59e0b',
  I8: '#fb923c',
}
const dtColor = (d: string) => DT_COLORS[d] ?? '#94a3b8'
const precision = computed(() => {
  const i = info.value
  if (!i) return []
  const total = Object.values(i.by_dtype).reduce((a, d) => a + d.bytes, 0) || 1
  return Object.entries(i.by_dtype)
    .map(([k, d]) => ({ k, ...d, share: d.bytes / total }))
    .sort((a, b) => b.bytes - a.bytes)
})

// ---- the tensor tree --------------------------------------------------------------------------
interface Node {
  key: string
  label: string
  depth: number
  children: Map<string, Node>
  tensor?: TensorRow
  count: number
  params: number
  dtypes: Set<string>
}

const tree = computed<Node | null>(() => {
  const i = info.value
  if (!i) return null
  const root: Node = { key: '', label: '', depth: -1, children: new Map(), count: 0, params: 0, dtypes: new Set() }
  for (const t of i.tensors) {
    const parts = t.name.split('.')
    let node = root
    const path: Node[] = [root]
    parts.forEach((p, d) => {
      const key = parts.slice(0, d + 1).join('.')
      let next = node.children.get(p)
      if (!next) {
        next = { key, label: p, depth: d, children: new Map(), count: 0, params: 0, dtypes: new Set() }
        node.children.set(p, next)
      }
      node = next
      path.push(node)
    })
    node.tensor = t
    for (const n of path) {
      n.count += 1
      n.params += t.params
      n.dtypes.add(t.dtype)
    }
  }
  return root
})

const tab = ref<'tensors' | 'metadata'>('tensors')
const open = ref(new Set<string>())
const query = ref('')

watch(tree, (t) => {
  // open the levels that hold a single group, so the first screen shows something
  const s = new Set<string>()
  let n = t
  while (n && n.children.size === 1) {
    const only: Node = n.children.values().next().value!
    if (!only.tensor) s.add(only.key)
    n = only
  }
  open.value = s
})

const rows = computed<Node[]>(() => {
  const t = tree.value
  if (!t) return []
  const out: Node[] = []
  const walk = (n: Node) => {
    for (const c of n.children.values()) {
      out.push(c)
      if (c.children.size && open.value.has(c.key)) walk(c)
    }
  }
  walk(t)
  return out
})

const MATCH_CAP = 400
const matches = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q || !info.value) return null
  return info.value.tensors.filter((t) => t.name.toLowerCase().includes(q))
})
/** a row's share of the file's parameters, for the bar behind its count */
const share = (params: number) => `${(100 * params) / (info.value?.params || 1)}%`
/** a group's precisions, largest first: two shown, the rest as "+n" */
function dtypesOf(n: Node) {
  const by = info.value?.by_dtype ?? {}
  const all = n.tensor ? [n.tensor.dtype] : [...n.dtypes].sort((a, b) => (by[b]?.bytes ?? 0) - (by[a]?.bytes ?? 0))
  return { all, shown: all.slice(0, 2), more: Math.max(0, all.length - 2) }
}

function toggle(n: Node) {
  if (!n.children.size) return
  const s = new Set(open.value)
  if (s.has(n.key)) s.delete(n.key)
  else s.add(n.key)
  open.value = s
}

function expandAll(on: boolean) {
  const s = new Set<string>()
  if (on && tree.value) {
    const walk = (n: Node) => n.children.forEach((c) => c.children.size && (s.add(c.key), walk(c)))
    walk(tree.value)
  }
  open.value = s
}

// ---- leaving tensors out of the copy ------------------------------------------------------------
const dropped = ref(new Set<string>())
const isOut = (key: string) => [...dropped.value].some((p) => key === p || key.startsWith(p + '.'))
function toggleOut(key: string) {
  const s = new Set(dropped.value)
  if (s.has(key)) s.delete(key)
  else if (!isOut(key)) {
    for (const p of s) if (p.startsWith(key + '.')) s.delete(p) // the group now covers them
    s.add(key)
  } else return // inside a group that's out: bring the group back instead
  dropped.value = s
}

// ---- metadata -----------------------------------------------------------------------------------
const meta = ref<{ key: string; value: string }[]>([])
const metaEditing = ref(false)
const metaOpen = ref(new Set<string>())
function startMeta() {
  meta.value = Object.entries(info.value?.metadata ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => ({ key, value }))
}
const metaObject = computed(() => Object.fromEntries(meta.value.filter((m) => m.key.trim()).map((m) => [m.key.trim(), m.value])))
const metaChanged = computed(() => {
  const a = info.value?.metadata ?? {}
  const b = metaObject.value
  return Object.keys(a).length !== Object.keys(b).length || Object.entries(a).some(([k, v]) => b[k] !== v)
})
/** a JSON value shown pretty-printed; anything else as it is */
function pretty(v: string): { json: boolean; text: string } {
  const s = v.trim()
  if (s.startsWith('{') || s.startsWith('[')) {
    try {
      return { json: true, text: JSON.stringify(JSON.parse(s), null, 2) }
    } catch {
      /* not JSON after all */
    }
  }
  return { json: false, text: v }
}
const isLong = (v: string) => v.length > 240 || v.split('\n').length > 6
function toggleMeta(key: string) {
  const s = new Set(metaOpen.value)
  if (s.has(key)) s.delete(key)
  else s.add(key)
  metaOpen.value = s
}
const copied = ref<string | null>(null)
async function copyValue(m: { key: string; value: string }) {
  await navigator.clipboard?.writeText(pretty(m.value).text).catch(() => {})
  copied.value = m.key
  setTimeout(() => copied.value === m.key && (copied.value = null), 1200)
}

// ---- saving a copy ------------------------------------------------------------------------------
const outName = ref('')
const dtype = ref<string>('')
const DTYPE_OPTIONS = [
  { value: '', label: 'Keep', title: 'Keep each tensor’s precision' },
  { value: 'bf16', label: 'bf16' },
  { value: 'fp16', label: 'fp16' },
  { value: 'fp32', label: 'fp32' },
]
const plan = ref<{ tensors: number; bytes: number; converted: number; dropped: number } | null>(null)
const saving = ref(false)
const saveError = ref<string | null>(null)
const result = ref<{ folder: TensorFolder; name: string; bytes: number; tensors: number; converted: number; dropped: number } | null>(null)

const body = () => ({
  folder: folder.value as TensorFolder,
  name: name.value,
  out_name: outName.value.trim(),
  dtype: (dtype.value || null) as 'bf16' | 'fp16' | 'fp32' | null,
  drop: [...dropped.value],
  metadata: metaChanged.value ? metaObject.value : undefined,
})

let planTimer: ReturnType<typeof setTimeout> | undefined
watch([dtype, dropped, info], () => {
  clearTimeout(planTimer)
  plan.value = null
  if (!info.value || (!dtype.value && !dropped.value.size)) return
  planTimer = setTimeout(async () => {
    plan.value = await engine.planTensorCopy(body()).catch(() => null)
  }, 200)
})

/** where the copy goes: beside the original */
const outPath = computed(() => `${FOLDER_LABELS[folder.value as TensorFolder] ?? folder.value} / ${fileDir.value ? `${fileDir.value}/` : ''}${outName.value.trim() || '…'}.safetensors`)
const changeLine = computed(() => {
  const parts = [
    dtype.value && `float weights as ${dtype.value}${info.value?.quantized ? ' (quantized ones stay)' : ''}`,
    dropped.value.size && `${plan.value ? plan.value.dropped.toLocaleString() : '…'} tensors left out`,
    metaChanged.value && 'metadata as edited',
  ].filter(Boolean)
  return parts.length ? parts.join(', ') : 'an exact copy: untick layers, pick a precision or edit the metadata to change it'
})

/** a textarea as tall as its text (up to a limit, then it scrolls) */
const grow = (el: HTMLTextAreaElement) => {
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight + 2, 320)}px`
}
const vGrow = {
  mounted: (el: HTMLTextAreaElement) => {
    grow(el)
    el.addEventListener('input', () => grow(el))
  },
  updated: grow,
}

async function save() {
  saving.value = true
  saveError.value = null
  result.value = null
  try {
    result.value = await engine.copyTensors(body())
    files.value = await engine.tensorFiles().catch(() => files.value)
  } catch (e) {
    saveError.value = e instanceof EngineError ? e.message : String(e)
  } finally {
    saving.value = false
  }
}

// ---- the file picked (last: it resets everything above) ------------------------------------------
function reset() {
  open.value = new Set()
  dropped.value = new Set()
  query.value = ''
  tab.value = 'tensors'
  metaEditing.value = false
  metaOpen.value = new Set()
  dtype.value = ''
  result.value = null
  saveError.value = null
}

watch(
  [folder, name],
  async ([f, n], old) => {
    if (old && f !== old[0] && n === old[1]) {
      name.value = '' // a new folder: pick again
      return
    }
    router.replace({ query: n ? { folder: f, name: n } : { folder: f } })
    info.value = null
    error.value = null
    reset()
    if (!n) return
    loading.value = true
    try {
      info.value = await engine.inspectTensors(f as TensorFolder, n)
      startMeta()
      outName.value = `${n.split('/').pop()!.replace(/\.safetensors$/i, '')}_copy`
    } catch (e) {
      error.value = e instanceof EngineError ? e.message : String(e)
    } finally {
      loading.value = false
    }
  },
  { immediate: true },
)
</script>

<template>
  <div class="page inspect">
    <div class="page-head">
      <div class="titles">
        <h1><RouterLink to="/tools" class="crumb">Tools</RouterLink> Inspect a model file</h1>
        <p class="lede">Every layer, shape, precision and metadata entry in a .safetensors file, and a copy saved the way you want it.</p>
      </div>
    </div>

    <section class="card pick">
      <div class="folders">
        <button v-for="o in folderOptions" :key="o.value" class="fchip" :class="{ on: folder === o.value }" @click="folder = o.value">
          <i :class="o.icon" /> {{ o.label }} <span class="muted">{{ o.count }}</span>
        </button>
      </div>
      <ZenCombo v-model="name" :items="fileItems" placeholder="Search this folder for a .safetensors file…" :item-height="30" />
    </section>

    <div v-if="error" class="card err small"><i class="mdi mdi-alert-circle-outline" /> {{ error }}</div>
    <section v-else-if="loading" class="card skeleton"><div class="ghost w40" /><div class="ghost w20" /></section>
    <ZenEmpty v-else-if="!info" icon="mdi mdi-file-tree-outline" title="Pick a file">A LoRA, a diffusion model, a VAE or a text encoder: only its header is read, so even a large model opens at once.</ZenEmpty>

    <template v-if="info">
      <!-- the file, and a copy of it -->
      <section class="card file">
        <div class="fhead">
          <i class="ficon" :class="FOLDER_ICONS[folder as TensorFolder] ?? 'mdi mdi-file-outline'" />
          <span class="fname mono" :title="name">{{ fileBase }}</span>
          <span class="badge">{{ kindLine }}</span>
          <span v-if="info.quantized" class="badge warn">quantized</span>
          <span class="grow" />
          <span class="facts">
            <span><b>{{ fmtBytes(info.size) }}</b></span>
            <span><b>{{ fmtParams(info.params) }}</b> params</span>
            <span><b>{{ info.tensors.length.toLocaleString() }}</b> tensors</span>
          </span>
        </div>
        <div class="prec">
          <div class="pbar">
            <span
              v-for="p in precision"
              :key="p.k"
              :style="{ flexGrow: p.share, background: dtColor(p.k) }"
              :title="`${p.k}: ${fmtBytes(p.bytes)} · ${p.tensors} tensors · ${fmtParams(p.params)} parameters`"
            />
          </div>
          <div class="legend">
            <span v-for="p in precision" :key="p.k" class="lg">
              <i :style="{ background: dtColor(p.k) }" /><b class="mono">{{ p.k }}</b>
              <span class="muted">{{ fmtBytes(p.bytes) }} · {{ p.tensors.toLocaleString() }} tensors</span>
            </span>
          </div>
        </div>

        <div class="savebox">
          <div class="saverow">
            <span class="eyebrow">Save a copy</span>
            <div class="grow"><ZenInput v-model="outName" placeholder="e.g. my-lora_bf16" size="sm" /></div>
            <ZenToggleGroup v-model="dtype" :options="DTYPE_OPTIONS" size="sm" />
            <ZenButton variant="primary" size="sm" icon="mdi mdi-content-save-outline" :disabled="!outName.trim() || saving" @click="save">{{ saving ? 'Saving…' : 'Save copy' }}</ZenButton>
          </div>
          <div class="saveline small">
            <span class="muted">Saves as</span> <span class="mono out">{{ outPath }}</span>
            <span class="muted"> · {{ changeLine }}</span>
            <span v-if="plan" class="mono"> · {{ fmtBytes(info.size) }} → <b>{{ fmtBytes(plan.bytes) }}</b></span>
          </div>
          <div v-if="saveError" class="err small">{{ saveError }}</div>
          <div v-if="result" class="done">
            <i class="mdi mdi-check-circle" />
            <div class="grow">
              <div class="mono out">{{ result.name }}</div>
              <div class="small muted">{{ fmtBytes(result.bytes) }} · {{ result.tensors.toLocaleString() }} tensors</div>
            </div>
            <ZenButton size="sm" icon="mdi mdi-file-tree-outline" @click="folder = result.folder; name = result.name">Inspect it</ZenButton>
          </div>
        </div>
      </section>

      <section class="card body">
        <div class="tabs">
          <button :class="{ on: tab === 'tensors' }" @click="tab = 'tensors'">
            <i class="mdi mdi-file-tree-outline" /> Tensors <span class="count">{{ info.tensors.length.toLocaleString() }}</span>
          </button>
          <button :class="{ on: tab === 'metadata' }" @click="tab = 'metadata'">
            <i class="mdi mdi-tag-text-outline" /> Metadata <span class="count">{{ meta.length }}</span>
            <span v-if="metaChanged" class="dot" title="Edited" />
          </button>
        </div>

        <!-- tensors: a tree of layers, or a flat list while filtering -->
        <template v-if="tab === 'tensors'">
          <div class="bar">
            <div class="q"><ZenInput v-model="query" placeholder="Filter by name, e.g. layers.12 or lora_up" size="sm" /></div>
            <template v-if="!matches">
              <ZenIconButton icon="mdi mdi-unfold-more-horizontal" title="Expand all" @click="expandAll(true)" />
              <ZenIconButton icon="mdi mdi-unfold-less-horizontal" title="Collapse all" @click="expandAll(false)" />
            </template>
            <span v-else class="small muted">{{ matches.length.toLocaleString() }} found</span>
          </div>
          <div class="table">
            <div class="row th">
              <span />
              <span>Name</span>
              <span class="shape">Shape</span>
              <span class="dt">Precision</span>
              <span class="pr">Params</span>
            </div>
            <div class="list">
              <template v-if="matches">
                <div v-for="t in matches.slice(0, MATCH_CAP)" :key="t.name" class="row" :class="{ out: isOut(t.name) }">
                  <button class="tick" :title="isOut(t.name) ? 'Keep in the copy' : 'Leave out of the copy'" @click="toggleOut(t.name)">
                    <i :class="isOut(t.name) ? 'mdi mdi-checkbox-blank-outline' : 'mdi mdi-checkbox-marked'" />
                  </button>
                  <span class="nm mono" :title="t.name">{{ t.name }}</span>
                  <span class="shape mono">{{ t.shape.length ? t.shape.join(' × ') : 'scalar' }}</span>
                  <span class="dt"><span class="pill mono" :style="{ '--c': dtColor(t.dtype) }">{{ t.dtype }}</span></span>
                  <span class="pr mono">{{ fmtParams(t.params) }}</span>
                </div>
                <div v-if="matches.length > MATCH_CAP" class="small muted pad">and {{ (matches.length - MATCH_CAP).toLocaleString() }} more: narrow the filter</div>
                <div v-if="!matches.length" class="small muted pad">No tensor names contain that.</div>
              </template>
              <template v-else>
                <div
                  v-for="n in rows"
                  :key="n.key"
                  class="row"
                  :class="{ out: isOut(n.key), group: n.children.size, opened: open.has(n.key) }"
                  :style="{ '--d': n.depth }"
                  @click="toggle(n)"
                >
                  <button class="tick" :title="isOut(n.key) ? 'Keep in the copy' : 'Leave out of the copy'" @click.stop="toggleOut(n.key)">
                    <i :class="isOut(n.key) ? 'mdi mdi-checkbox-blank-outline' : 'mdi mdi-checkbox-marked'" />
                  </button>
                  <span class="nm mono" :title="n.key">
                    <i v-if="n.children.size" class="caret" :class="open.has(n.key) ? 'mdi mdi-chevron-down' : 'mdi mdi-chevron-right'" />
                    <span>{{ n.label }}</span>
                    <span v-if="n.children.size" class="muted cnt">{{ n.count.toLocaleString() }}</span>
                  </span>
                  <span class="shape mono">{{ n.tensor ? (n.tensor.shape.length ? n.tensor.shape.join(' × ') : 'scalar') : '' }}</span>
                  <span class="dt">
                    <span v-for="d in dtypesOf(n).shown" :key="d" class="pill mono" :style="{ '--c': dtColor(d) }">{{ d }}</span>
                    <span v-if="dtypesOf(n).more" class="pill more-dt mono" :title="dtypesOf(n).all.join(' ')">+{{ dtypesOf(n).more }}</span>
                  </span>
                  <span class="pr mono">
                    <span v-if="n.children.size && n.params / info.params >= 0.02" class="share" :style="{ width: share(n.params) }" />
                    <span class="v">{{ fmtParams(n.params) }}</span>
                  </span>
                </div>
              </template>
            </div>
          </div>
          <p class="small muted hint"><i class="mdi mdi-information-outline" /> Untick a layer or a whole group to leave it out of a saved copy.</p>
        </template>

        <!-- metadata -->
        <template v-else>
          <div class="bar">
            <span class="small muted">{{ meta.length ? `${meta.length} entries in the header` : 'This file has no metadata.' }}</span>
            <span class="grow" />
            <ZenButton v-if="metaChanged" size="sm" variant="ghost" icon="mdi mdi-undo" @click="startMeta()">Undo changes</ZenButton>
            <ZenButton size="sm" :icon="metaEditing ? 'mdi mdi-check' : 'mdi mdi-pencil-outline'" @click="metaEditing = !metaEditing">
              {{ metaEditing ? 'Done' : 'Edit' }}
            </ZenButton>
          </div>
          <div class="meta">
            <div v-for="(m, i) in meta" :key="i" class="mrow" :class="{ editing: metaEditing }">
              <template v-if="metaEditing">
                <textarea v-model="m.key" v-grow class="field mono" rows="1" placeholder="key" @keydown.enter.prevent />
                <textarea v-model="m.value" v-grow class="field mono" rows="1" placeholder="value" />
                <ZenIconButton icon="mdi mdi-close" size="sm" danger title="Remove" @click="meta.splice(i, 1)" />
              </template>
              <template v-else>
                <span class="k mono">{{ m.key }}</span>
                <div class="vwrap">
                  <pre class="v mono" :class="{ json: pretty(m.value).json, clamp: isLong(m.value) && !metaOpen.has(m.key) }">{{ pretty(m.value).text }}</pre>
                  <button v-if="isLong(m.value)" class="more small" @click="toggleMeta(m.key)">{{ metaOpen.has(m.key) ? 'Show less' : 'Show all' }}</button>
                </div>
                <ZenIconButton :icon="copied === m.key ? 'mdi mdi-check' : 'mdi mdi-content-copy'" size="sm" title="Copy the value" @click="copyValue(m)" />
              </template>
            </div>
          </div>
          <button v-if="metaEditing" class="add" @click="meta.push({ key: '', value: '' })"><i class="mdi mdi-plus" /> Add an entry</button>
        </template>
      </section>
    </template>
  </div>
</template>

<style scoped>
.inspect {
  max-width: 1400px;
}
.grow {
  flex: 1;
  min-width: 0;
}
.err {
  color: var(--zen-danger);
}

/* ---- picking --------------------------------------------------------------------------------- */
.pick {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.folders {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.fchip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 26px;
  padding: 0 10px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-text);
  font: inherit;
  font-size: 12px;
  white-space: nowrap;
  cursor: pointer;
}
.fchip:hover {
  background: var(--zen-surface-2);
}
.fchip.on {
  border-color: var(--zen-accent);
  background: color-mix(in srgb, var(--zen-accent) 12%, transparent);
}
.fchip i {
  color: var(--zen-muted);
}

/* ---- the file, and a copy of it -------------------------------------------------------------- */
.file {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.fhead {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.ficon {
  font-size: 18px;
  color: var(--zen-muted);
}
@media (max-width: 700px) {
  .ficon {
    display: none;
  }
}
.fname {
  font-size: 15px;
  font-weight: 600;
  overflow-wrap: anywhere;
  min-width: 0;
}
.badge {
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 99px;
  background: var(--zen-surface-2);
  border: 1px solid var(--zen-border);
  white-space: nowrap;
}
.badge.warn {
  color: var(--zen-warn);
  border-color: color-mix(in srgb, var(--zen-warn) 40%, transparent);
}
.facts {
  display: flex;
  gap: 14px;
  font-size: 12px;
  color: var(--zen-muted);
  white-space: nowrap;
}
.facts b {
  color: var(--zen-text);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.prec {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.pbar {
  display: flex;
  height: 6px;
  gap: 2px;
  border-radius: 99px;
  overflow: hidden;
  background: var(--zen-surface-2);
}
.pbar span {
  flex-basis: 0;
  min-width: 3px;
}
.legend {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  font-size: 11.5px;
}
.lg {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.lg i {
  width: 8px;
  height: 8px;
  border-radius: 2px;
}
.savebox {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid var(--zen-border);
}
.saverow {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.saverow .grow {
  min-width: 200px;
}
.saveline {
  line-height: 1.5;
}
.out {
  overflow-wrap: anywhere;
}
.skeleton {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.skeleton .ghost {
  height: 12px;
  border-radius: 6px;
  background: var(--zen-surface-2);
  animation: pulse 1.2s ease-in-out infinite;
}
.w40 {
  width: 40%;
}
.w20 {
  width: 20%;
}
@keyframes pulse {
  50% {
    opacity: 0.5;
  }
}

/* ---- the body: tabs, then the tree or the metadata, beside saving ------------------------------ */
.body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}
.tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--zen-border);
  margin: -4px -4px 0;
}
.tabs button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 13px;
  padding: 8px 12px 10px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}
.tabs button:hover {
  color: var(--zen-text);
}
.tabs button.on {
  color: var(--zen-text);
  border-bottom-color: var(--zen-accent);
}
.count {
  font-size: 11px;
  padding: 0 6px;
  border-radius: 99px;
  background: var(--zen-surface-2);
  font-variant-numeric: tabular-nums;
}
.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--zen-accent);
}
.bar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.q {
  flex: 1;
  min-width: 200px;
  position: relative;
}

.table {
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  overflow: hidden;
}
.list {
  max-height: calc(68vh / var(--zen-page-zoom));
  overflow: auto;
}
.row {
  display: grid;
  grid-template-columns: 26px minmax(0, 1fr) minmax(90px, auto) 112px 80px;
  gap: 10px;
  align-items: center;
  font-size: 12px;
  min-height: 30px;
  padding: 0 12px 0 8px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-border) 45%, transparent);
}
.list .row:last-child {
  border-bottom: none;
}
.row.th {
  min-height: 30px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--zen-muted);
  background: var(--zen-surface-2);
  border-bottom: 1px solid var(--zen-border);
}
.list .row:hover {
  background: color-mix(in srgb, var(--zen-surface-2) 70%, transparent);
}
.row.group {
  cursor: pointer;
}
.row.group.opened > .nm {
  font-weight: 600;
}
.row.out .nm,
.row.out .shape,
.row.out .dt,
.row.out .pr {
  opacity: 0.35;
}
.row.out .nm span:first-of-type {
  text-decoration: line-through;
}
.tick {
  border: none;
  background: none;
  color: var(--zen-muted);
  opacity: 0.45;
  cursor: pointer;
  padding: 0;
  font-size: 16px;
  line-height: 1;
}
.row:hover .tick {
  opacity: 1;
}
.row.out .tick {
  opacity: 1;
  color: var(--zen-warn);
}
.tick:hover {
  color: var(--zen-accent);
}
.nm {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  padding-left: calc(var(--d, 0) * 18px);
  /* a guide line for each level */
  background-image: repeating-linear-gradient(
    to right,
    color-mix(in srgb, var(--zen-border) 70%, transparent) 0 1px,
    transparent 1px 18px
  );
  background-size: calc(var(--d, 0) * 18px) 100%;
  background-repeat: no-repeat;
  background-position: 6px 0;
  align-self: stretch;
}
.nm > span:first-of-type {
  overflow: hidden;
  text-overflow: ellipsis;
}
.row:not(.group) .nm {
  padding-left: calc(var(--d, 0) * 18px + 18px);
}
.list .row:not([style]) .nm {
  padding-left: 0;
}
.caret {
  font-size: 16px;
  color: var(--zen-muted);
  width: 14px;
}
.cnt {
  font-size: 11px;
  margin-left: 4px;
}
.shape {
  color: var(--zen-muted);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.dt {
  display: flex;
  gap: 4px;
  justify-content: flex-end;
  white-space: nowrap;
}
.more-dt {
  --c: var(--zen-muted);
}
.pill {
  font-size: 10.5px;
  padding: 0 6px;
  border-radius: 4px;
  color: var(--c);
  background: color-mix(in srgb, var(--c) 14%, transparent);
  line-height: 18px;
}
.pr {
  position: relative;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--zen-muted);
  align-self: stretch;
  display: flex;
  align-items: center;
  justify-content: flex-end;
}
.th .pr,
.th .dt,
.th .shape {
  color: var(--zen-muted);
  font-family: inherit;
}
.share {
  position: absolute;
  right: 0;
  top: 7px;
  bottom: 7px;
  border-radius: 3px;
  max-width: 100%;
  background: color-mix(in srgb, var(--zen-text) 10%, transparent);
}
.pr .v {
  position: relative;
  padding-right: 4px;
}
.row.group .pr {
  color: var(--zen-text);
}
.pad {
  padding: 10px 12px;
}
.hint {
  margin: 0;
  display: flex;
  gap: 6px;
  align-items: center;
}
@media (max-width: 700px) {
  .row {
    grid-template-columns: 22px minmax(0, 1fr) auto 56px;
    gap: 6px;
    padding: 0 8px 0 6px;
  }
  .nm {
    padding-left: calc(var(--d, 0) * 10px);
    background-size: calc(var(--d, 0) * 10px) 100%;
  }
  .row:not(.group) .nm {
    padding-left: calc(var(--d, 0) * 10px + 18px);
  }
  .row .shape {
    display: none;
  }
}

/* ---- metadata -------------------------------------------------------------------------------- */
.meta {
  display: flex;
  flex-direction: column;
  max-height: calc(68vh / var(--zen-page-zoom));
  overflow: auto;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
}
.meta:empty {
  display: none;
}
.mrow {
  display: grid;
  grid-template-columns: minmax(140px, 1fr) minmax(0, 3fr) auto;
  gap: 12px;
  align-items: start;
  font-size: 12px;
  padding: 8px 10px 8px 12px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-border) 45%, transparent);
}
.mrow:last-child {
  border-bottom: none;
}
.mrow:hover {
  background: color-mix(in srgb, var(--zen-surface-2) 60%, transparent);
}
.k {
  word-break: break-all;
  color: var(--zen-muted);
  padding-top: 2px;
}
.vwrap {
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
}
.v {
  margin: 0;
  width: 100%;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  padding-top: 2px;
}
.v.json {
  background: var(--zen-surface-2);
  border-radius: var(--r-sm);
  padding: 6px 8px;
}
.v.clamp {
  max-height: 96px;
  overflow: hidden;
  mask-image: linear-gradient(to bottom, black 60%, transparent);
}
.more {
  border: none;
  background: none;
  color: var(--zen-accent);
  cursor: pointer;
  padding: 0;
  font: inherit;
  font-size: 12px;
}
.add {
  align-self: flex-start;
  border: 1px dashed var(--zen-border);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  padding: 6px 12px;
  border-radius: var(--r-md);
  cursor: pointer;
}
.add:hover {
  border-color: var(--zen-accent);
  color: var(--zen-text);
}
@media (max-width: 700px) {
  .mrow {
    grid-template-columns: minmax(0, 1fr) auto;
  }
  .mrow .k {
    grid-column: 1 / -1;
  }
  .mrow.editing {
    grid-template-columns: minmax(0, 1fr) auto;
  }
}

/* ---- the metadata editor's fields: zenkit's look, as tall as their text ----------------------- */
.field {
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  resize: none;
  overflow: auto;
  font-size: 12px;
  line-height: 1.45;
  padding: 6px 9px;
  color: var(--zen-text);
  background: var(--zen-input);
  border: 1px solid var(--zen-input);
  border-radius: var(--zen-radius, 8px);
  outline: none;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.field:focus {
  border-color: var(--zen-accent);
}
.done {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  padding: 10px;
  border-radius: var(--r-md);
  background: color-mix(in srgb, var(--zen-ok) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--zen-ok) 30%, transparent);
}
.done > i {
  color: var(--zen-ok);
  font-size: 20px;
}
.titles h1 {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
</style>
