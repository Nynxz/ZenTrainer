<script setup lang="ts">
// Settings, top to bottom: versions (what's running, what's newer: the first thing to check when
// something's off), appearance, training defaults, faces, then where things are stored.
// Theme previews are drawn from each pack's own tokens, so every ZenKit pack and user drop-in gets
// one without shipping banner images.
import { ZenButton, ZenInput } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { engine, type EngineSettings, type Health } from '@/api/engine'
import { SETTINGS, setSetting, settings, type SettingId } from '@/stores/settings'
import { checkingUpdates, checkUpdates, comfyVersion as comfy, webNewer, webStale, ztVersion as zt } from '@/stores/status'
import { currentModes, currentTheme, packCards, setMode, setTheme } from '@/lib/theme'

const mode = computed(() => currentTheme.value.mode)
// registry settings, by category (Appearance renders inside its own section)
const settingsIn = (category: string) => (Object.keys(SETTINGS) as SettingId[]).filter((id) => SETTINGS[id].category === category)
const currentName = computed(() => packCards.value.find((p) => p.id === currentTheme.value.id)?.name)
// keep the user's light/dark preference; a one-mode pack shows its own mode meanwhile
const pick = (id: string) => setTheme(id, currentTheme.value.preferred)
// ComfyUI's own palettes, first in the list
const COMFY: Record<string, string> = { 'maroon-volt': 'ComfyUI', 'comfyui-brand': 'ComfyUI' }
const packFilter = ref<'all' | 'dark' | 'light'>('all')
const shownPacks = computed(() =>
  packCards.value
    .filter((p) => packFilter.value === 'all' || p.modes.includes(packFilter.value))
    .sort((a, b) => Number(!!COMFY[b.id]) - Number(!!COMFY[a.id]) || (a.id === 'maroon-volt' ? -1 : b.id === 'maroon-volt' ? 1 : 0)),
)
// the swatch shows the pack in the current mode when it has it
const sw = (p: (typeof packCards.value)[number]) => (p.modes.includes(mode.value) ? p.swatch[mode.value] : p.swatch[p.modes[0]!])!

// Versions: ZenTrainer (the engine), this page's build and ComfyUI. The sidebar's check fills in
// what's upstream a little after launch; until then (or on a fresh tab) just what's running.
const WEB = __ZT_BUILD__
const health = ref<Health | null>(null)
onMounted(async () => {
  checkUpdates(false)
  health.value = await engine.health().catch(() => null)
})
/** `git describe` as people read it: "v0.3.0-45-g909bb94" → 0.3.0, 45 commits on, 909bb94 */
function described(d?: string, fallback = '') {
  const m = /^v?(.+?)-(\d+)-g([0-9a-f]+)$/.exec(d ?? '')
  if (m) return { release: m[1]!, after: Number(m[2]), sha: m[3]! }
  if (d && /^[0-9a-f]{7,}$/.test(d)) return { release: fallback, after: 0, sha: d }
  return { release: d?.replace(/^v/, '') || fallback, after: 0, sha: '' }
}
const ztD = computed(() => described(zt.value?.describe, zt.value?.version))
const webD = computed(() => described(WEB.describe, WEB.version))
// the page is the same code as the engine (or at least its web files haven't changed since)
const webMatches = computed(() => !webStale.value && !webNewer.value)
const showCommits = ref(false)
const copied = ref(false)
// one block to paste into an issue or a chat when asking for help
function versionText() {
  const z = zt.value
  const c = comfy.value
  const lines = [
    `ZenTrainer ${z?.version ?? '?'}${z?.describe ? ` (${z.describe}${z.dirty ? ', local changes' : ''}${z.branch && z.branch !== 'main' ? `, branch ${z.branch}` : ''})` : ''}`,
    `Web app ${WEB.version}${WEB.describe ? ` (${WEB.describe}${WEB.dirty ? ', local changes' : ''})` : ''}${webStale.value ? ', older than its source' : ''}`,
    `ComfyUI ${c?.version ?? '?'}${c?.describe ? ` (${c.describe})` : ''}${c?.pin ? (c.matches_pin ? ', the pin' : ', not the pin') : ''}`,
  ]
  const cap = health.value?.capabilities
  if (cap?.device_name || cap?.device) lines.push(`GPU ${cap.device_name ?? cap.device}`)
  return lines.join('\n')
}
async function copyVersions() {
  const text = versionText()
  // the clipboard API is only there on localhost or https: opened by LAN address, the old way
  if (navigator.clipboard) await navigator.clipboard.writeText(text)
  else {
    const t = Object.assign(document.createElement('textarea'), { value: text })
    document.body.append(t)
    t.select()
    document.execCommand('copy')
    t.remove()
  }
  copied.value = true
  setTimeout(() => (copied.value = false), 1500)
}
const reload = () => location.reload()

// Storage: the ZenTrainer folder (datasets and results out of ComfyUI's input/output). The engine
// reads it at start, so a saved change shows as pending until ComfyUI restarts.
const storage = ref<EngineSettings | null>(null)
const folderDraft = ref('')
const savingFolder = ref(false)
const folderNote = ref('')
const folderError = ref('')
onMounted(async () => {
  storage.value = await engine.settings().catch(() => null)
  folderDraft.value = storage.value?.settings.folder ?? ''
})
// Datasets folder: where datasets are listed and made. "" = as the folder above decides; "datasets"
// = input/datasets/, apart from everything else uploaded to ComfyUI's input; or an absolute path
const dsChoice = ref<'default' | 'input' | 'custom'>('default')
const dsCustom = ref('')
const savingDs = ref(false)
const dsNote = ref('')
const dsError = ref('')
function showDatasets(v: string) {
  dsChoice.value = !v ? 'default' : v === 'datasets' ? 'input' : 'custom'
  dsCustom.value = dsChoice.value === 'custom' ? v : ''
}
watch(storage, (s) => s && showDatasets(s.settings.datasets ?? ''), { immediate: true })
const dsValue = computed(() => (dsChoice.value === 'default' ? '' : dsChoice.value === 'input' ? 'datasets' : dsCustom.value.trim()))
async function saveDatasets() {
  savingDs.value = true
  dsError.value = dsNote.value = ''
  try {
    storage.value = await engine.putSettings({ datasets: dsValue.value })
    dsNote.value = storage.value.message || 'Saved'
  } catch (e) {
    dsError.value = String((e as Error).message ?? e)
  } finally {
    savingDs.value = false
  }
}
async function saveFolder() {
  savingFolder.value = true
  folderError.value = folderNote.value = ''
  try {
    storage.value = await engine.putSettings({ folder: folderDraft.value.trim() })
    folderDraft.value = storage.value.settings.folder
    folderNote.value = storage.value.message || 'Saved'
  } catch (e) {
    folderError.value = String((e as Error).message ?? e)
  } finally {
    savingFolder.value = false
  }
}

// Faces: which recognition model face reports and likeness use (applies at the next face check)
const savingFace = ref(false)
const faceError = ref('')
const faceModel = computed(() => storage.value?.face_models?.find((m) => m.id === storage.value?.settings.face_model))
async function pickFace(id: string) {
  if (!storage.value || id === storage.value.settings.face_model) return
  savingFace.value = true
  faceError.value = ''
  try {
    storage.value = await engine.putSettings({ face_model: id })
  } catch (e) {
    faceError.value = String((e as Error).message ?? e)
  } finally {
    savingFace.value = false
  }
}

</script>

<template>
  <div class="page narrow">
    <div class="page-head">
      <div class="titles">
        <h1>Settings</h1>
      </div>
    </div>

    <section class="card section">
      <div>
        <h2>Versions</h2>
        <p class="muted">What's running here, and whether anything newer is out. Asking for help? Copy these along.</p>
      </div>
      <dl class="kv versions">
        <dt>ZenTrainer</dt>
        <dd>
          <span class="mono strong">{{ ztD.release || zt?.version || '–' }}</span>
          <span v-if="ztD.after" class="muted">+ {{ ztD.after }} commit{{ ztD.after === 1 ? '' : 's' }}</span>
          <span v-if="ztD.sha" class="mono muted small-t">{{ ztD.sha }}</span>
          <span v-if="zt?.branch && zt.branch !== 'main'" class="tag-s mono">{{ zt.branch }}</span>
          <span v-if="zt?.dirty" class="tag-s" title="ZenTrainer's files have local changes: zt update leaves them alone and skips updating">local changes</span>
          <div class="status">
            <span v-if="zt?.on_disk" class="warn"><i class="mdi mdi-restart" /> {{ described(zt.on_disk.describe).sha || 'a newer version' }} is on disk: restart ComfyUI to load it (the sidebar's update button does)</span>
            <span v-else-if="zt && !zt.commit" class="muted">Not a git checkout, so it can't check for updates: compare with the releases on GitHub.</span>
            <span v-else-if="zt?.upstream_error" class="warn">Couldn't check: {{ zt.upstream_error }}</span>
            <template v-else-if="zt?.upstream?.behind">
              <span class="warn">
                <i class="mdi mdi-arrow-up-circle-outline" /> {{ zt.upstream.behind }} new commit{{ zt.upstream.behind === 1 ? '' : 's' }} on {{ zt.upstream.ref }}
                <button class="link" @click="showCommits = !showCommits">{{ showCommits ? 'hide' : 'what’s new' }}</button>
              </span>
              <ul v-if="showCommits" class="commits">
                <li v-for="c in zt.upstream.commits" :key="c"><span class="mono muted">{{ c.split(' ')[0] }}</span> {{ c.slice(c.indexOf(' ') + 1) }}</li>
                <li v-if="zt.upstream.behind > zt.upstream.commits.length" class="muted">… and {{ zt.upstream.behind - zt.upstream.commits.length }} more</li>
              </ul>
              <span class="muted">
                To update, run <span class="mono">zt update</span> (or start from the launcher, which offers it), then restart.
                <template v-if="zt.dirty"> Your local changes stop it: commit or stash them first.</template>
                <template v-else-if="zt.upstream.ahead"> This checkout has {{ zt.upstream.ahead }} commit{{ zt.upstream.ahead === 1 ? '' : 's' }} of its own, so it won't fast-forward: pull it yourself.</template>
              </span>
            </template>
            <span v-else-if="zt?.upstream" class="ok"><i class="mdi mdi-check-circle-outline" /> up to date with {{ zt.upstream.ref }}</span>
          </div>
        </dd>

        <dt>Web app</dt>
        <dd>
          <span class="mono strong">{{ webD.release || '–' }}</span>
          <span v-if="webD.after" class="muted">+ {{ webD.after }} commit{{ webD.after === 1 ? '' : 's' }}</span>
          <span v-if="webD.sha" class="mono muted small-t">{{ webD.sha }}</span>
          <span v-if="WEB.dirty" class="tag-s" title="Built from web files with local changes">local changes</span>
          <div class="status">
            <span v-if="webNewer" class="warn"><i class="mdi mdi-refresh" /> rebuilt since this tab opened: <button class="link" @click="reload">reload</button> to see it</span>
            <span v-else-if="webStale" class="warn"><i class="mdi mdi-hammer-wrench" /> its source is newer than this build (after an update): the sidebar's update button rebuilds it</span>
            <span v-else-if="webMatches" class="ok"><i class="mdi mdi-check-circle-outline" /> built from the code on disk</span>
          </div>
        </dd>

        <template v-if="comfy">
          <dt>ComfyUI</dt>
          <dd>
            <span class="mono strong">{{ comfy.version ?? '–' }}</span>
            <span class="mono muted small-t">{{ comfy.describe }}</span>
            <div class="status">
              <template v-if="comfy.pin">
                <span v-if="comfy.matches_pin" class="ok"><i class="mdi mdi-check-circle-outline" /> the version ZenTrainer is tested with (comfy.lock)</span>
                <span v-else class="warn"><i class="mdi mdi-alert-outline" /> not the tested version ({{ comfy.pin.describe }}): run <span class="mono">zt update</span></span>
              </template>
              <span v-if="comfy.upstream_error" class="warn">Couldn't check: {{ comfy.upstream_error }}</span>
              <span v-else-if="comfy.upstream?.behind" class="muted">
                Upstream is {{ comfy.upstream.behind }} commits ahead ({{ comfy.upstream.describe }}). Newer isn't automatically better here:
                ZenTrainer moves its pin once a version is tested.
              </span>
              <span class="mono muted small-t">{{ comfy.path }}</span>
            </div>
          </dd>
        </template>
      </dl>
      <div class="row-actions">
        <ZenButton size="sm" icon="mdi mdi-cloud-sync-outline" :disabled="checkingUpdates" @click="checkUpdates()">{{ checkingUpdates ? 'Checking…' : 'Check for updates' }}</ZenButton>
        <ZenButton size="sm" variant="ghost" :icon="copied ? 'mdi mdi-check' : 'mdi mdi-content-copy'" @click="copyVersions">{{ copied ? 'Copied' : 'Copy versions' }}</ZenButton>
      </div>
    </section>

    <section class="card section">
      <div>
        <h2>Appearance</h2>
        <p class="muted">Each pack brings its own palette and rounding. Some support light and dark; some are one-mode.</p>
      </div>

      <div class="stack">
        <div class="eyebrow">Mode</div>
        <div class="segmented">
          <button
            v-for="m in ['light', 'dark'] as const"
            :key="m"
            :class="{ on: mode === m }"
            :disabled="!currentModes.includes(m)"
            @click="setMode(m)"
          >
            <i :class="m === 'light' ? 'mdi mdi-white-balance-sunny' : 'mdi mdi-weather-night'" />
            {{ m === 'light' ? 'Light' : 'Dark' }}
          </button>
        </div>
        <p v-if="currentModes.length < 2" class="muted hint">
          {{ currentName }} is {{ mode }}-only. Packs with both modes follow your
          {{ currentTheme.preferred }} preference.
        </p>
      </div>

      <div v-for="id in settingsIn('Appearance')" :key="id" class="stack">
        <div class="eyebrow">{{ SETTINGS[id].label }}</div>
        <div class="segmented">
          <button
            v-for="o in SETTINGS[id].options"
            :key="String(o.value)"
            :class="{ on: settings[id] === o.value }"
            @click="setSetting(id, o.value)"
          >
            <i v-if="o.icon" :class="o.icon" /> {{ o.label }}
          </button>
        </div>
        <p v-if="SETTINGS[id].description" class="muted hint">{{ SETTINGS[id].description }}</p>
      </div>

      <div class="stack">
        <div class="pack-head">
          <span class="eyebrow">Theme pack</span>
          <span class="muted small">{{ shownPacks.length }} of {{ packCards.length }}</span>
          <span class="spacer" />
          <div class="segmented small-seg">
            <button v-for="f in (['all', 'dark', 'light'] as const)" :key="f" :class="{ on: packFilter === f }" @click="packFilter = f">
              {{ { all: 'All', dark: 'Dark', light: 'Light' }[f] }}
            </button>
          </div>
        </div>
        <div class="packs">
          <button
            v-for="p in shownPacks"
            :key="p.id"
            class="pack"
            :class="{ active: currentTheme.id === p.id }"
            :title="p.description || p.name"
            @click="pick(p.id)"
          >
            <span
              class="sw"
              :style="{
                background: sw(p).bg,
                borderColor: sw(p).border,
                '--c': sw(p).card,
                '--p': sw(p).primary,
                '--f': sw(p).fg,
                '--rr': sw(p).radius,
              }"
            >
              <span class="sw-card"><span class="sw-line" /><span class="sw-btn" /></span>
            </span>
            <span class="pack-text">
              <span class="name">{{ p.name }}</span>
              <span class="tags">
                <span v-if="COMFY[p.id]" class="tag">{{ COMFY[p.id] }}</span>
                <i v-if="p.modes.includes('light')" class="mdi mdi-white-balance-sunny" title="Light" />
                <i v-if="p.modes.includes('dark')" class="mdi mdi-weather-night" title="Dark" />
              </span>
            </span>
            <i v-if="currentTheme.id === p.id" class="mdi mdi-check check" />
          </button>
        </div>
      </div>
    </section>

    <section class="card section">
      <div>
        <h2>Training</h2>
        <p class="muted">Defaults for new runs. Each run keeps the values it started with; where LoRAs are saved is picked in New run.</p>
      </div>
      <div v-for="id in settingsIn('Training').filter((x) => SETTINGS[x].options.length)" :key="id" class="stack">
        <div class="eyebrow">{{ SETTINGS[id].label }}</div>
        <div class="segmented">
          <button
            v-for="o in SETTINGS[id].options"
            :key="String(o.value)"
            :class="{ on: settings[id] === o.value }"
            @click="setSetting(id, o.value as never)"
          >
            <i v-if="o.icon" :class="o.icon" /> {{ o.label }}
          </button>
        </div>
        <p v-if="SETTINGS[id].description" class="muted hint">{{ SETTINGS[id].description }}</p>
      </div>
    </section>

    <!-- an engine from before the face model setting sends no face_models: no section -->
    <section v-if="storage?.face_models" class="card section">
      <div>
        <h2>Faces</h2>
        <p class="muted">
          Face reports on datasets and the likeness score on samples use a face recognition model, on the CPU. A run keeps
          the model it started with; scores from different models don't compare.
        </p>
      </div>
      <div class="stack">
        <div class="eyebrow">Model</div>
        <div class="segmented">
          <button
            v-for="m in storage.face_models"
            :key="m.id"
            :class="{ on: storage.settings.face_model === m.id }"
            :disabled="savingFace"
            @click="pickFace(m.id)"
          >
            {{ m.label }}
          </button>
        </div>
        <p v-if="faceModel" class="muted hint">
          {{ faceModel.note }}<template v-if="!faceModel.downloaded"> Not downloaded yet: it downloads at the next face check.</template>
        </p>
        <p class="muted hint">InsightFace's models are for non-commercial research use only.</p>
        <p v-if="faceError" class="warn">{{ faceError }}</p>
      </div>
    </section>

    <section v-if="storage" class="card section">
      <div>
        <h2>Storage</h2>
        <p class="muted">
          By default datasets live in ComfyUI's input folder and LoRAs, samples and merges under its output/zentrainer.
          A ZenTrainer folder keeps them out of there: datasets in <span class="mono">datasets/</span>, results in
          <span class="mono">output/</span>. Nothing already there moves.
        </p>
      </div>
      <div class="stack">
        <div class="eyebrow">ZenTrainer folder</div>
        <div class="row-actions">
          <ZenInput v-model="folderDraft" class="grow mono" placeholder="Empty: ComfyUI's input and output folders" @keydown.enter="saveFolder" />
          <ZenButton size="sm" icon="mdi mdi-content-save-outline" :disabled="savingFolder || folderDraft.trim() === storage.settings.folder" @click="saveFolder">Save</ZenButton>
        </div>
        <p class="muted hint">An absolute path outside ComfyUI's input and output folders. Takes effect after restarting ComfyUI.</p>
        <p v-if="folderError" class="warn">{{ folderError }}</p>
        <p v-else-if="storage.restart_needed" class="warn"><i class="mdi mdi-restart" /> Saved: restart ComfyUI (System → Restart ComfyUI) to use it. Until then the folders below stay in use.</p>
        <p v-else-if="folderNote" class="ok">{{ folderNote }}</p>
      </div>
      <div class="stack">
        <div class="eyebrow">Datasets folder</div>
        <div class="segmented">
          <button :class="{ on: dsChoice === 'default' }" title="Datasets where the ZenTrainer folder puts them, or straight in ComfyUI's input folder" @click="dsChoice = 'default'">As above</button>
          <button :class="{ on: dsChoice === 'input' }" title="In input/datasets/: apart from everything else uploaded to ComfyUI's input folder" @click="dsChoice = 'input'">input/datasets</button>
          <button :class="{ on: dsChoice === 'custom' }" title="Any folder, e.g. on a bigger drive" @click="dsChoice = 'custom'">Custom</button>
        </div>
        <div class="row-actions">
          <ZenInput v-if="dsChoice === 'custom'" v-model="dsCustom" class="grow mono" placeholder="An absolute path, or one inside ComfyUI's input folder" @keydown.enter="saveDatasets" />
          <span v-else class="grow" />
          <ZenButton size="sm" icon="mdi mdi-content-save-outline" :disabled="savingDs || dsValue === (storage.settings.datasets ?? '') || (dsChoice === 'custom' && !dsValue)" @click="saveDatasets">Save</ZenButton>
        </div>
        <p class="muted hint">Only where ZenTrainer looks: nothing is moved. Runs still find datasets where they were. Takes effect after restarting ComfyUI.</p>
        <p v-if="dsError" class="warn">{{ dsError }}</p>
        <p v-else-if="dsNote && !storage.restart_needed" class="ok">{{ dsNote }}</p>
        <div v-if="storage.datasets_elsewhere?.length" class="warn small-t">
          <i class="mdi mdi-folder-move-outline" /> {{ storage.datasets_elsewhere.length }} dataset{{ storage.datasets_elsewhere.length === 1 ? '' : 's' }}
          won't be listed there until moved into it (training runs still find them):
          <ul class="mono"><li v-for="d in storage.datasets_elsewhere.slice(0, 8)" :key="d">{{ d }}</li><li v-if="storage.datasets_elsewhere.length > 8">…</li></ul>
        </div>
      </div>
      <div class="stack">
        <div class="eyebrow">In use now</div>
        <dl class="kv">
          <dt>Datasets</dt><dd class="mono small-t">{{ storage.active.datasets }}</dd>
          <dt>Outputs</dt><dd class="mono small-t">{{ storage.active.output }}</dd>
          <dt>LoRAs</dt><dd class="mono small-t">{{ storage.active.loras }}</dd>
        </dl>
      </div>
    </section>
  </div>
</template>

<style scoped>
.narrow {
  max-width: 880px;
}
.section {
  display: flex;
  flex-direction: column;
  gap: 22px;
  padding: 22px;
}
h2 {
  margin: 0 0 4px;
  font-size: 16px;
}
p {
  margin: 0;
}
.stack {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.kv {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 8px 16px;
  margin: 0;
  font-size: 13px;
}
.kv dt {
  color: var(--zen-muted);
}
.kv dd {
  margin: 0;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: baseline;
}
.versions {
  gap: 14px 20px;
}
.versions dt {
  padding-top: 1px;
}
.strong {
  font-weight: 600;
}
.status {
  flex-basis: 100%;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}
.tag-s {
  font-size: 10px;
  padding: 0 6px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-warn) 16%, transparent);
}
.link {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  color: var(--zen-accent);
  cursor: pointer;
  text-decoration: underline;
}
.commits {
  margin: 2px 0 4px;
  padding-left: 16px;
  font-size: 12px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.small-t {
  font-size: 11px;
  word-break: break-all;
}
.ok {
  color: var(--zen-ok);
  font-size: 12px;
}
.warn {
  color: var(--zen-warn);
  font-size: 12px;
}
.row-actions .grow {
  flex: 1;
  min-width: 240px;
}
.row-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.segmented {
  display: inline-flex;
  align-self: flex-start;
  flex-wrap: wrap;
  gap: 2px;
  padding: 2px;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 40%, transparent);
}
.segmented button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border: none;
  border-radius: var(--r-md);
  background: none;
  color: var(--zen-muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.segmented button:disabled {
  opacity: 0.4;
  cursor: default;
}
.hint {
  font-size: 12px;
}
.segmented button.on {
  background: var(--zen-selected-bg, var(--zen-surface-2));
  color: var(--zen-selected-text, var(--zen-text));
  box-shadow: inset 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
.pack-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.small-seg button {
  padding: 3px 10px;
  font-size: 12px;
}
.packs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 8px;
}
.pack {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 10px 7px 7px;
  text-align: left;
  font: inherit;
  color: inherit;
  cursor: pointer;
  border-radius: var(--r-lg);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
  background: color-mix(in srgb, var(--zen-surface) 40%, transparent);
  transition: border-color 0.15s ease;
  min-width: 0;
}
.pack:hover {
  border-color: var(--zen-muted);
}
.pack.active {
  border-color: var(--zen-selected, var(--zen-accent));
  box-shadow: 0 0 0 1px var(--zen-selected, var(--zen-accent));
}
/* the pack in miniature: its background, a card, a line of text, its primary button */
.sw {
  width: 46px;
  height: 36px;
  flex: none;
  box-sizing: border-box;
  padding: 5px;
  border: 1px solid;
  border-radius: calc(var(--rr) * 0.8);
}
.sw-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height: 100%;
  box-sizing: border-box;
  padding: 3px;
  border-radius: calc(var(--rr) * 0.5);
  background: var(--c);
}
.sw-line {
  width: 70%;
  height: 3px;
  border-radius: 2px;
  background: var(--f);
  opacity: 0.7;
}
.sw-btn {
  align-self: flex-end;
  width: 45%;
  height: 5px;
  border-radius: calc(var(--rr) * 0.4);
  background: var(--p);
}
.pack-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  flex: 1;
}
.name {
  font-weight: 600;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tags {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: var(--zen-muted);
}
.tag {
  white-space: nowrap;
  font-size: 10px;
  padding: 0 5px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-accent) 18%, transparent);
  color: var(--zen-text);
}
.check {
  width: 20px;
  height: 20px;
  flex: none;
  display: grid;
  place-items: center;
  border-radius: var(--r-md);
  background: var(--zen-accent);
  color: var(--zen-accent-text);
  font-size: 13px;
}
</style>
