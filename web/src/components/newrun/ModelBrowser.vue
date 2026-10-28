<script setup lang="ts">
// New run's model browser, laid out as the dataset picker: the model families on the left (starred
// first, a search, installed or all) and the one highlighted on the right: whether it's ready,
// where it comes from, and its recipes. Use picks the family (FamilyPicker then sorts out its
// files); a recipe card opens Recipes on it, to see what it changes before it's applied.
import { ZenButton, ZenInput } from '@nynxz/zenkit-ui'
import { computed, onMounted, ref, watch } from 'vue'
import { engine, type ModelFamily } from '@/api/engine'
import ScrollArea from '@/components/ScrollArea.vue'
import StarButton from '@/components/StarButton.vue'
import { isOther, makerLogo } from '@/lib/makers'
import { RECIPES } from '@/lib/recipes'
import { fmtBytes } from '@/stores/downloads'
import { isStarred } from '@/stores/favorites'
import { familiesSeen } from '@/stores/lists'

const props = defineProps<{ current: string | null }>()
/** recipe: apply it here; details: open Recipes on it (null: on them all, yours too) */
const emit = defineEmits<{ pick: [id: string]; recipe: [id: string]; details: [id: string | null] }>()
const open = defineModel<boolean>('open', { default: false })

const families = familiesSeen // the Models page's list too: shown at once, refreshed on open
async function load() {
  try {
    families.value = await engine.families()
  } catch {
    // the list as last seen (or empty): the combo beside the field still works
  }
}
onMounted(() => families.value.length || load())

// the model makers' Hugging Face repos listed again, for models added since the cache
const relisting = ref(false)
const relistNote = ref<string | null>(null)
async function relist() {
  relisting.value = true
  relistNote.value = null
  try {
    const r = await engine.relistFamilies()
    await load()
    relistNote.value = r.new.length ? `New: ${r.new.join(', ')}` : 'Up to date'
  } catch (e) {
    relistNote.value = String((e as Error).message ?? e)
  } finally {
    relisting.value = false
  }
}

const query = ref('')
const only = ref<'all' | 'ready'>('all')
const picked = ref<string>('')
// on opening, and as the view appears in the open dialog (a tab switch): after setup, so it
// reads what's declared further down
function reset(o: boolean) {
  if (!o) return
  query.value = ''
  picked.value = props.current || families.value[0]?.id || ''
  void load()
}
watch(open, reset)
onMounted(() => open.value && reset(true))

const ready = (f: ModelFamily) => !f.missing.length
const listed = computed(() => {
  const q = query.value.trim().toLowerCase()
  return families.value.filter((f) => (!q || f.name.toLowerCase().includes(q) || f.id.includes(q)) && (only.value === 'all' || ready(f)))
})
const groups = computed(() => {
  const starred = listed.value.filter((f) => isStarred(f.id))
  const main = listed.value.filter((f) => !isStarred(f.id) && !isOther(f.id))
  const other = listed.value.filter((f) => !isStarred(f.id) && isOther(f.id))
  return [
    ...(starred.length ? [{ key: 'starred', label: 'Starred', items: starred }] : []),
    ...(main.length ? [{ key: 'main', label: 'Models', items: main }] : []),
    ...(other.length ? [{ key: 'other', label: 'Other models', items: other }] : []),
  ]
})
const shown = computed(() => families.value.find((f) => f.id === picked.value) ?? null)
const recipes = computed(() => (shown.value ? RECIPES.filter((r) => r.family === shown.value!.id) : []))
const KINDS: Record<string, string> = { character: 'Character', style: 'Style', video: 'Video', general: 'Any subject' }
const missingSize = (f: ModelFamily) => f.missing.reduce((n, m) => n + (m.size ?? 0), 0)
const added = (f: ModelFamily) => (f.added ? new Date(f.added).toLocaleDateString([], { year: 'numeric', month: 'short' }) : null)

function use(id = picked.value) {
  if (!id) return
  emit('pick', id)
  open.value = false
}
// a recipe brings its own model: applied over the form, it picks the family too
function recipe(id: string) {
  emit('recipe', id)
  open.value = false
}
// the Recipes view, on it: the dialog stays open
function details(id: string | null) {
  emit('details', id)
}
</script>

<template>
  <!-- a view of ModelDialog: its controls and its buttons go in the dialog's header and footer -->
  <Teleport defer to="#zt-md-head">
      <div class="head">
        <div class="view-switch text" title="Show">
          <button :class="{ on: only === 'all' }" @click="only = 'all'">All</button>
          <button :class="{ on: only === 'ready' }" @click="only = 'ready'">Installed</button>
        </div>
        <div class="mb-search"><ZenInput v-model="query" size="sm" placeholder="Search models" /></div>
      </div>
  </Teleport>

    <div class="picker">
      <ScrollArea class="list">
        <nav class="list-in">
          <section v-for="g in groups" :key="g.key" class="group">
            <header class="g-head"><i v-if="g.key === 'starred'" class="mdi mdi-star" /> {{ g.label }}</header>
            <!-- a row, not a button: the star inside it is one -->
            <div
              v-for="f in g.items"
              :key="f.id"
              role="button"
              tabindex="0"
              class="item"
              :class="{ on: picked === f.id }"
              :aria-pressed="picked === f.id"
              @click="picked = f.id"
              @dblclick="use(f.id)"
              @keydown.enter.prevent="picked === f.id ? use(f.id) : (picked = f.id)"
            >
              <span class="logo">
                <img v-if="makerLogo(f.id)" :src="makerLogo(f.id)!.src" :class="{ mono: makerLogo(f.id)!.mono }" alt="" />
                <i v-else class="mdi mdi-cube-outline" />
              </span>
              <span class="item-text">
                <span class="item-title">{{ f.name }}<span v-if="f.id === current" class="here">in this run</span></span>
                <span class="item-blurb">
                  <template v-if="ready(f)"><i class="mdi mdi-check-circle ok" /> installed</template>
                  <template v-else>{{ f.missing.length }} to download</template>
                  <template v-if="RECIPES.some((r) => r.family === f.id)"> · {{ RECIPES.filter((r) => r.family === f.id).length }} recipe{{ RECIPES.filter((r) => r.family === f.id).length === 1 ? '' : 's' }}</template>
                </span>
              </span>
              <StarButton :id="f.id" :name="f.name" />
            </div>
          </section>
          <div v-if="!groups.length" class="empty muted small">{{ query ? `No model matches “${query}”.` : 'No models installed yet.' }}</div>
        </nav>
      </ScrollArea>

      <ScrollArea class="detail">
        <article v-if="shown" class="detail-in">
          <div class="title-row">
            <span class="logo big">
              <img v-if="makerLogo(shown.id)" :src="makerLogo(shown.id)!.src" :class="{ mono: makerLogo(shown.id)!.mono }" alt="" />
              <i v-else class="mdi mdi-cube-outline" />
            </span>
            <div>
              <h3 class="zt-title">{{ shown.name }}</h3>
              <div class="muted small">
                {{ makerLogo(shown.id)?.maker ?? '' }}{{ makerLogo(shown.id) && added(shown) ? ' · ' : '' }}{{ added(shown) ? `added ${added(shown)}` : '' }}
              </div>
            </div>
          </div>

          <dl class="facts">
            <div>
              <dt>Files</dt>
              <dd v-if="ready(shown)" class="ok"><i class="mdi mdi-check-circle" /> installed</dd>
              <dd v-else>{{ shown.missing.length }} to download{{ missingSize(shown) ? ` · ${fmtBytes(missingSize(shown))}` : '' }}</dd>
            </div>
            <div><dt>Loads as</dt><dd>{{ shown.kind === 'checkpoint' ? 'one checkpoint' : 'model + text encoder + VAE' }}</dd></div>
            <div v-if="shown.profile"><dt>Settings</dt><dd>{{ shown.profile.status === 'verified' ? 'tuned for it' : 'suggested' }}</dd></div>
          </dl>
          <p v-if="shown.gated" class="small warn"><i class="mdi mdi-lock-outline" /> Its repo asks you to accept a license (and set a Hugging Face token) before it downloads.</p>

          <h4>Recipes</h4>
          <div v-if="recipes.length" class="recipes">
            <!-- a row, not a button: its Details is one -->
            <div
              v-for="r in recipes"
              :key="r.id"
              role="button"
              tabindex="0"
              class="recipe"
              :title="`Use ${r.title}: its model and settings (Undo takes it back)`"
              @click="recipe(r.id)"
              @keydown.enter.prevent="recipe(r.id)"
            >
              <span class="cover" :style="{ background: `linear-gradient(135deg, ${r.cover.from}, ${r.cover.to})` }"><i :class="r.cover.icon" /></span>
              <span class="r-text">
                <span class="r-title">{{ KINDS[r.kind] ?? r.kind }}<span v-if="!r.validated" class="draft">draft</span></span>
                <span class="r-meta">{{ r.spec.network?.algorithm ?? 'LoRA' }} · {{ r.photos.ideal }}</span>
              </span>
              <button type="button" class="r-details" :title="`Every setting ${r.title} changes, before applying it`" @click.stop="details(r.id)">Details</button>
            </div>
          </div>
          <p v-else class="muted small">No recipe for it yet: its own settings are a start, and Save as recipe keeps yours.</p>
        </article>
        <div v-else class="empty muted small">Pick a model on the left.</div>
      </ScrollArea>
    </div>

    <Teleport defer to="#zt-md-foot">
      <div class="foot">
        <button type="button" class="foot-link" title="Pick the diffusion model, text encoder and VAE files yourself" @click="use('__custom')"><i class="mdi mdi-file-cog-outline" /> Pick the files yourself</button>
        <button type="button" class="foot-link" :disabled="relisting" title="List the model makers' Hugging Face repos again: models they've added show up without waiting for the cache" @click="relist">
          <i class="mdi mdi-refresh" :class="{ spin: relisting }" /> {{ relisting ? 'Re-listing…' : 'Re-list' }}
        </button>
        <span v-if="relistNote" class="muted small">{{ relistNote }}</span>
        <span class="spacer" />
        <ZenButton variant="primary" icon="mdi mdi-check" :disabled="!shown" @click="use()">{{ shown ? `Use ${shown.name}` : 'Use' }}</ZenButton>
      </div>
    </Teleport>
</template>

<style scoped>
.head,
.foot {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  min-width: 0;
}
.mb-search {
  width: 200px;
}
.picker {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  height: 100%;
  min-height: 0;
}
.list,
.detail {
  min-height: 0;
}
.list {
  border-right: 1px solid var(--zen-border);
  background: color-mix(in srgb, var(--zen-surface-2) 55%, transparent);
}
.list-in {
  padding: 10px 8px 16px;
}
.group + .group {
  margin-top: 10px;
}
.g-head {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted);
}
.g-head .mdi-star {
  color: var(--zen-accent);
}
.item {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 7px 8px;
  border-radius: var(--r-md);
  color: var(--zen-text);
  cursor: pointer;
  position: relative;
}
.item:hover {
  background: color-mix(in srgb, var(--zen-text) 6%, transparent);
}
.item.on {
  background: color-mix(in srgb, var(--zen-accent) 14%, transparent);
}
.item.on::before {
  content: '';
  position: absolute;
  left: -8px;
  top: 8px;
  bottom: 8px;
  width: 3px;
  border-radius: var(--r-sm);
  background: var(--zen-accent);
}
.logo {
  width: 30px;
  height: 30px;
  flex: none;
  border-radius: var(--r-md);
  display: grid;
  place-items: center;
  background: #fff;
  color: var(--zen-muted);
  overflow: hidden;
}
.logo img {
  width: 72%;
  height: 72%;
  object-fit: contain;
}
.logo i {
  color: #555;
}
.logo.big {
  width: 48px;
  height: 48px;
  border-radius: var(--r-lg);
}
.item-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}
.item-title {
  font-weight: 600;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.item-blurb {
  font-size: 11px;
  color: var(--zen-muted);
}
.ok {
  color: var(--zen-ok);
}
.here {
  font-size: 10px;
  font-weight: 500;
  color: var(--zen-accent);
}
.detail-in {
  padding: 18px 20px 20px;
}
.title-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}
.title-row h3 {
  margin: 0;
  font-size: 22px;
}
.facts {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
  gap: 8px;
  margin: 0;
}
.facts > div {
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  padding: 8px 10px;
}
.facts dt {
  font-size: 11px;
  color: var(--zen-muted);
}
.facts dd {
  margin: 2px 0 0;
  font-size: 13px;
}
.warn {
  color: var(--zen-warn, #d9a400);
  margin-top: 10px;
}
h4 {
  margin: 18px 0 8px;
  font-size: 13px;
}
.recipes {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px;
}
.recipe {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-lg);
  background: none;
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.recipe:hover {
  border-color: color-mix(in srgb, var(--zen-accent) 50%, var(--zen-border));
}
.cover {
  width: 40px;
  height: 40px;
  flex: none;
  border-radius: var(--r-md);
  display: grid;
  place-items: center;
  color: #fff;
  font-size: 20px;
}
.r-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}
.r-title {
  font-weight: 600;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.r-meta {
  font-size: 11px;
  color: var(--zen-muted);
}
.draft {
  font-size: 10px;
  font-weight: 500;
  padding: 0 5px;
  border: 1px dashed var(--zen-warn, #d9a400);
  color: var(--zen-warn, #d9a400);
  border-radius: var(--r-md);
}
.r-details {
  border: none;
  background: none;
  padding: 2px 4px;
  font: inherit;
  font-size: 11px;
  color: var(--zen-muted);
  cursor: pointer;
  flex: none;
}
.r-details:hover {
  color: var(--zen-text);
}
.empty {
  padding: 20px;
}
.foot-link {
  border: none;
  background: none;
  padding: 0;
  font: inherit;
  font-size: 12px;
  color: var(--zen-muted);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.foot-link:hover:not(:disabled) {
  color: var(--zen-text);
}
.spin {
  animation: spin 0.9s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@media (max-width: 720px) {
  .picker {
    grid-template-columns: 1fr;
  }
  .detail {
    display: none;
  }
}
</style>
