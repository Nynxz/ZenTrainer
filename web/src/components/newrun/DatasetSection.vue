<script setup lang="ts">
// 2 Dataset: the folder, captions and trigger, resolution (and starting lower), dropout, edit
// pairs, masks, and how videos turn into training clips.
import { ZenCombo, ZenField, ZenInput, ZenNumber, ZenSwitch } from '@nynxz/zenkit-ui'
import { computed, ref } from 'vue'
import { thumbUrl, type DatasetInfo, type TrainSpec } from '@/api/engine'
import DatasetPicker from './DatasetPicker.vue'
import { items, useRunFormContext } from './form'

const { spec, choices, datasets, dataset, folderWords, pairCount, stage0, resuming, controlCounts, chosenInputs, toggleInput, channelModel } = useRunFormContext()

const NK2E_TEMPLATE =
  'Make the following edit to the reference image, changing only what is requested and keeping everything else identical: {}'
// start lower: one earlier stage at a lower resolution (the engine takes several; the form edits one)
const LOWER = [256, 384, 512, 640, 768, 896]
const lowerChoices = computed(() => LOWER.filter((r) => r < spec.dataset.resolution))
const startLower = computed({
  get: () => !!spec.dataset.stages?.length,
  set: (on: boolean) => {
    const pick = lowerChoices.value.filter((r) => r <= spec.dataset.resolution * 0.75).at(-1) ?? lowerChoices.value.at(-1)
    spec.dataset.stages = on && pick ? [{ resolution: pick, until: 0.6 }] : []
    if (on) spec.dataset.stage_blend ??= 0.2
  },
})
// a step's cost goes roughly with its pixels (more, with attention): the ratio, to one decimal
const lowerSpeedup = computed(() => (stage0.value ? (spec.dataset.resolution / stage0.value.resolution) ** 2 : 1).toFixed(1))
// one set = caption_set; more than one = caption_mix (random per step)
function toggleSet(st: string) {
  const current = spec.dataset.caption_mix.length ? [...spec.dataset.caption_mix] : [spec.dataset.caption_set]
  const next = current.includes(st) ? current.filter((x) => x !== st) : [...current, st]
  if (!next.length) return
  spec.dataset.caption_set = next[0]!
  spec.dataset.caption_mix = next.length > 1 ? next : []
}
const pickerOpen = ref(false)
// masked training against the dataset's masks: says so when the two don't agree
const maskHint = computed(() => {
  const n = dataset.value?.masked ?? 0
  if (n && !spec.dataset.masked_loss) return `${n} of ${dataset.value!.images} pictures have masks: turn it on to train only where they're white`
  if (!n && spec.dataset.masked_loss) return 'This dataset has no masks: nothing to mask'
  return n ? `${n} of ${dataset.value!.images} pictures masked` : ''
})
// the dataset's name as a trigger word, one click away: lower case, spaces and dashes as _
const nameTrigger = computed(() => (spec.dataset.folder || '').toLowerCase().replace(/[\s-]+/g, '_').replace(/[^a-z0-9_]/g, ''))
// the dataset's cover: up to four pictures, from as many of its folders
const cover = (d: DatasetInfo) =>
  d.thumbs.slice(0, d.thumbs.length >= 4 ? 4 : 1).map((t) => thumbUrl(d.folder, t.path ?? [...t.subfolder.split('/').slice(1), t.filename].join('/'), 256))
// pictures outside the subfolders, which keep the one trigger word
const loose = computed(() => dataset.value?.folders?.find((f) => !f.name)?.images ?? 0)
// a folder's own word, or none: then its name's
function setWord(name: string, value: string) {
  const words = { ...(spec.dataset.folder_trigger_words ?? {}) }
  if (value.trim()) words[name] = value.trim()
  else delete words[name]
  spec.dataset.folder_trigger_words = words
}
const VIDEO_MODES: [TrainSpec['dataset']['video_mode'], string][] = [
  ['uniform', 'Spread'],
  ['head', 'Start'],
  ['chunks', 'Back to back'],
]
</script>

<template>
<section class="card sec" :class="{ locked: resuming }">
  <h2><span class="num">2</span> Dataset</h2>
  <ZenField label="Folder" stack>
    <button type="button" class="ds-field" :class="{ empty: !dataset }" @click="pickerOpen = true">
      <span class="ds-thumb" :class="{ blank: !dataset, mosaic: dataset && cover(dataset).length > 1 }"><template v-if="dataset && cover(dataset).length"><img v-for="src in cover(dataset)" :key="src" :src="src" alt="" /></template><i v-else :class="dataset ? 'mdi mdi-image-multiple-outline' : 'mdi mdi-plus'" /></span>
      <span class="ds-text">
        <b>{{ dataset?.folder ?? (spec.dataset.folder || 'Choose a dataset') }}</b>
        <span v-if="dataset" class="muted small">{{ dataset.images }} pictures · {{ dataset.captioned }} captioned</span>
        <span v-else class="muted small">A folder of pictures in ComfyUI's input</span>
      </span>
      <span class="ds-change">{{ dataset ? 'Change' : 'Pick' }} <i class="mdi mdi-chevron-right" /></span>
    </button>
  </ZenField>
  <DatasetPicker v-model:open="pickerOpen" :datasets="datasets" :current="spec.dataset.folder" @pick="spec.dataset.folder = $event" />
  <ZenField v-if="folderWords.length" label="Trigger per folder" stack tip="Each folder trains under a trigger word of its own: its name, or the word typed beside it. A number in front of a name sets repeats and stays out of the word (3_vegeta trains vegeta)">
    <div class="fold-block">
    <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.folder_triggers" @update:model-value="spec.dataset.folder_triggers = $event" /><span class="sw-state">{{ spec.dataset.folder_triggers ? `On: each of the ${folderWords.length} folders trains as the word in its box, or as its folder's name when the box is empty` : 'Off: every picture uses the one trigger word' }}</span></div>
    <div class="folds" :class="{ off: !spec.dataset.folder_triggers }">
      <div v-for="f in folderWords" :key="f.name" class="fold">
        <img :src="thumbUrl(spec.dataset.folder, f.path, 128)" alt="" loading="lazy" />
        <span class="fold-text"><b :title="f.name">{{ f.name }}</b><span class="muted small">{{ f.images }} picture{{ f.images === 1 ? '' : 's' }}</span></span>
        <ZenInput :model-value="spec.dataset.folder_trigger_words?.[f.name] ?? ''" :placeholder="f.trigger === f.name ? 'uses folder name' : `uses “${f.trigger}”`" :disabled="!spec.dataset.folder_triggers" size="sm" :title="`Its trigger word. Empty: “${f.trigger}”, the folder's name`" @update:model-value="setWord(f.name, String($event))" />
      </div>
    </div>
    <span v-if="loose && spec.dataset.folder_triggers" class="muted small">{{ loose }} picture{{ loose === 1 ? '' : 's' }} outside the folders use the trigger word below</span>
    </div>
  </ZenField>
  <div class="grid-2">
    <ZenField label="Captions" stack :hint="spec.dataset.caption_mix.length > 1 ? 'Each step picks one of these sets at random' : 'Which caption set to train on; click more to mix them'">
      <div class="chips">
        <button
          v-for="st in dataset?.sets ?? ['default']"
          :key="st"
          class="chip-btn"
          :class="{ on: spec.dataset.caption_mix.length ? spec.dataset.caption_mix.includes(st) : spec.dataset.caption_set === st }"
          @click="toggleSet(st)"
        >
          {{ st }}
        </button>
      </div>
    </ZenField>
    <!-- the dataset's name as a link on the label's line: the input keeps its width either way -->
    <div class="trig-field">
      <ZenField label="Trigger word" stack tip="Replaces [trigger] in captions and prompts">
        <ZenInput v-model="spec.dataset.trigger" placeholder="e.g. ohwx" />
      </ZenField>
      <button
        v-if="nameTrigger && spec.dataset.trigger !== nameTrigger"
        type="button"
        class="use-name"
        :title="`Use the dataset's name, ${nameTrigger}, as the trigger word`"
        @click="spec.dataset.trigger = nameTrigger"
      >
        <i class="mdi mdi-tag-outline" /> <span class="ellipsis">Use “{{ nameTrigger }}”</span>
      </button>
    </div>
    <ZenField label="Add if missing" stack tip="Puts the trigger word at the start of any caption that doesn't have it, so every picture teaches it. Only while training: your caption files stay as they are"><div class="sw-row"><ZenSwitch v-model="spec.dataset.trigger_prepend" :disabled="!spec.dataset.trigger && !spec.dataset.folder_triggers" /><span class="sw-state">{{ spec.dataset.trigger_prepend ? 'On' : 'Off' }}</span></div></ZenField>
    <ZenField label="Resolution" stack tip="Most models are made for about 1024. Lower has fewer pixels a step (256: a sixteenth), which on big models trains much faster and suits quick tests and coarse styles; on small models the saving is smaller. A LoRA trained low looks soft at full size">
      <div class="chips">
        <button v-for="r in [256, 512, 768, 1024, 1280]" :key="r" class="chip-btn" :class="{ on: spec.dataset.resolution === r }" @click="spec.dataset.resolution = r">
          {{ r }}
        </button>
      </div>
    </ZenField>
    <ZenField label="Start lower" stack tip="Train at a lower resolution first, then at the run's: most of what a LoRA learns comes early, fine detail last, and lower-resolution steps are much quicker. Each resolution is cached before training; the change blends over a share of the run so nothing switches at once">
      <div class="sw-row">
        <ZenSwitch v-model="startLower" :disabled="!lowerChoices.length" />
        <span class="sw-state">{{ stage0 ? `${stage0.resolution} px for ${Math.round(stage0.until * 100)}%, then ${spec.dataset.resolution}` : 'Off' }}</span>
      </div>
    </ZenField>
    <template v-if="stage0">
      <ZenField label="Lower resolution" stack :tip="`Steps at ${stage0.resolution} have 1/${lowerSpeedup} of the pixels of one at ${spec.dataset.resolution}: on big models they cost about that much less, on small ones the saving is smaller`">
        <div class="chips">
          <button v-for="r in lowerChoices" :key="r" class="chip-btn" :class="{ on: stage0.resolution === r }" @click="stage0.resolution = r">{{ r }}</button>
        </div>
      </ZenField>
      <ZenField label="Until / blend" stack tip="The share of the run at the lower resolution, then the share it blends over into the run's (the higher one grows from no steps to every step)">
        <div class="inline">
          <ZenNumber :model-value="Math.round(stage0.until * 100)" :min="5" :max="95" :step="5" :precision="0" @update:model-value="stage0.until = $event / 100" />
          <ZenNumber :model-value="Math.round((spec.dataset.stage_blend ?? 0.2) * 100)" :min="0" :max="50" :step="5" :precision="0" @update:model-value="spec.dataset.stage_blend = $event / 100" />
        </div>
      </ZenField>
      <p v-if="(spec.dataset.stages?.length ?? 0) > 1" class="muted small span-2">{{ spec.dataset.stages!.length }} stages from the preset; this edits the first.</p>
    </template>
    <ZenField label="Caption dropout" stack tip="Share of steps with an empty caption">
      <ZenNumber v-model="spec.dataset.caption_dropout" :min="0" :max="1" :step="0.05" :precision="2" />
    </ZenField>
    <ZenField
      label="Edit inputs"
      stack
      tip="Edit training: pick an input set (made on the dataset's page) and each picture's input there is what the model edits from; the caption is the instruction. Several sets give each picture several inputs, in the order picked. None: ordinary training. For models that take reference images (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit)"
    >
      <div class="input-chips">
        <button
          v-for="(n, c) in controlCounts"
          :key="c"
          class="chip-btn"
          :class="{ on: chosenInputs.includes(c) }"
          :title="`${n} picture${n === 1 ? '' : 's'} with an input in “${c}”`"
          @click="toggleInput(String(c))"
        >
          <span v-if="chosenInputs.length > 1 && chosenInputs.includes(c)" class="mono">{{ chosenInputs.indexOf(c) + 1 }}</span>
          {{ c }} · {{ n }}
        </button>
        <span v-if="!Object.keys(controlCounts).length" class="muted small">None: this dataset has no input sets</span>
        <span v-else-if="!chosenInputs.length" class="muted small">Off: ordinary training</span>
        <span v-else class="muted small">{{ pairCount }} pair{{ pairCount === 1 ? '' : 's' }}</span>
      </div>
    </ZenField>
    <template v-if="spec.dataset.edit">
      <ZenField
        label="Inputs join as"
        stack
        tip="References: each input picture joins the model's sequence beside the picture it makes, any size and framing (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit; MOSS Image through ZenTrainer's MOSS Edit nodes). Channels (MOSS Image): each input is stacked onto the picture pixel for pixel, so it must line up with it (same framing and shape); cheap, and the LoRA runs in plain ComfyUI (InstructPixToPixConditioning for one input)"
      >
        <div class="seg">
          <button :class="{ on: spec.dataset.edit_method !== 'concat' }" :disabled="resuming" @click="spec.dataset.edit_method = ''">References</button>
          <button :class="{ on: spec.dataset.edit_method === 'concat' }" :disabled="!channelModel || resuming" @click="spec.dataset.edit_method = 'concat'">Channels</button>
        </div>
      </ZenField>
      <ZenField
        v-if="spec.dataset.edit_method === 'concat'"
        label="Input dropout"
        stack
        tip="Share of steps each input is left out, on its own: the LoRA also learns to work with some or none of them. 0 for edits that always have their input"
      >
        <ZenNumber :model-value="spec.dataset.control_dropout ?? 0" :min="0" :max="0.9" :step="0.05" :precision="2" @update:model-value="spec.dataset.control_dropout = $event" />
      </ZenField>
      <ZenField label="Instruction template" stack tip="Wraps every instruction ({} is the caption), in training and samples alike. Training on top of NK2E? Use its template: its LoRA learned wrapped instructions">
        <div class="tpl-row">
          <ZenInput :model-value="spec.dataset.edit_template ?? ''" placeholder="None: the caption as it is" size="sm" @update:model-value="spec.dataset.edit_template = String($event)" />
          <button class="chip-btn" :class="{ on: spec.dataset.edit_template === NK2E_TEMPLATE }" title="NK2E's instruction template" @click="spec.dataset.edit_template = spec.dataset.edit_template === NK2E_TEMPLATE ? '' : NK2E_TEMPLATE">NK2E</button>
        </div>
      </ZenField>
      <ZenField v-if="spec.dataset.edit_method !== 'concat'" label="Encoder sees the input" stack tip="The text encoder looks at the input picture with the instruction (vision-language encoders: Krea 2, Qwen Image, MOSS Image), as krea2edit's identity edits do: better at keeping who and what is in the picture. Each pair's text is encoded on its own, so caching takes longer">
        <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.edit_grounded" @update:model-value="spec.dataset.edit_grounded = $event" /><span class="sw-state">{{ spec.dataset.edit_grounded ? 'On' : 'Off' }}</span></div>
      </ZenField>
    </template>
    <ZenField label="Cache on disk" stack tip="Each picture's latents (and its own conditioning) in a file of its own, read as training uses them, in bf16: for picture datasets bigger than RAM, such as thousands of edit pairs with the encoder seeing the input">
      <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.disk_cache" @update:model-value="spec.dataset.disk_cache = $event" /><span class="sw-state">{{ spec.dataset.disk_cache ? 'On' : 'Off' }}</span></div>
    </ZenField>
    <ZenField label="Masked training" stack tip="Masked images learn only where it's white" :hint="maskHint"><div class="sw-row"><ZenSwitch v-model="spec.dataset.masked_loss" /><span class="sw-state">{{ spec.dataset.masked_loss ? 'On' : 'Off' }}</span></div></ZenField>
    <ZenField v-if="spec.dataset.masked_loss" label="Unmasked weight" stack tip="0 ignores the rest; 0.1 learns it lightly">
      <ZenNumber v-model="spec.dataset.unmasked_weight" :min="0" :max="1" :step="0.05" :precision="2" />
    </ZenField>
  </div>
  <!-- only for datasets with videos: how they turn into training clips -->
  <div v-if="dataset?.videos" class="video-opts">
    <div class="sub-head"><i class="mdi mdi-filmstrip" /> Videos <span class="muted small">{{ dataset.videos }} in this dataset</span></div>
    <div class="grid-2">
      <ZenField label="Frames per clip" stack tip="The longest clip; it's cut to a length the model's VAE encodes exactly (49 → 39 for MiniMax H3, 49 for Wan)">
        <ZenNumber v-model="spec.dataset.video_frames" :min="1" :max="257" :step="8" :precision="0" />
      </ZenField>
      <ZenField label="Frame rate" stack tip="Resample clips to this many frames a second; 0 keeps each video's own">
        <ZenNumber v-model="spec.dataset.video_fps" :min="0" :max="60" :step="1" :precision="0" />
      </ZenField>
      <ZenField label="Which part" stack tip="Spread: clips spaced over the whole video. Start: the opening only. Back to back: consecutive clips from the start">
        <div class="chips">
          <button v-for="[m, l] in VIDEO_MODES" :key="m" class="chip-btn" :class="{ on: spec.dataset.video_mode === m }" @click="spec.dataset.video_mode = m">{{ l }}</button>
        </div>
      </ZenField>
      <ZenField label="Clips per video" stack tip="Each clip trains as its own sample">
        <ZenNumber v-model="spec.dataset.video_clips" :min="1" :max="32" :step="1" :precision="0" :disabled="spec.dataset.video_mode === 'head'" />
      </ZenField>
      <ZenField label="Video resolution" stack tip="A clip costs frames × pixels, so videos train smaller than images">
        <div class="chips">
          <button v-for="r in [256, 384, 512, 640, 768]" :key="r" class="chip-btn" :class="{ on: spec.dataset.video_resolution === r }" @click="spec.dataset.video_resolution = r">{{ r }}</button>
        </div>
      </ZenField>
      <ZenField label="Stay inside a shot" stack tip="Finds hard cuts and keeps every clip within one shot. Leave off for edits where the cutting is the point (music videos)">
        <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.video_avoid_cuts" @update:model-value="spec.dataset.video_avoid_cuts = $event" /><span class="sw-state">{{ spec.dataset.video_avoid_cuts ? 'Split at cuts' : 'Clips may cross cuts' }}</span></div>
      </ZenField>
      <ZenField label="Image to video" stack tip="MiniMax H3 fl2va: each clip is conditioned on its own first frame (or first and last), as the model is at inference when you give it a start image. Trains the LoRA for image-to-video; samples use their input picture as the first frame">
        <ZenCombo
          :model-value="spec.dataset.video_keyframes ?? ''"
          :items="items(['', 'first', 'first_last'], { '': 'Off: text to video', first: 'First frame', first_last: 'First and last frame' })"
          :searchable="false"
          :sorted="false"
          @update:model-value="spec.dataset.video_keyframes = String($event)"
        />
      </ZenField>
      <ZenField label="Clip stills" stack tip="Each clip also trains its sharpest frame as a photo, with the clip's caption. Stills run whole (never TREAD-routed) and anchor the LoRA's detail; Fizgig turns this on for H3">
        <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.video_stills" @update:model-value="spec.dataset.video_stills = $event" /><span class="sw-state">{{ spec.dataset.video_stills ? 'One per clip' : 'Off' }}</span></div>
      </ZenField>
      <ZenField label="Train on sound" stack tip="Joint audio-video models (MiniMax H3): each clip's soundtrack joins the loss, so motion and cuts can learn the beat. Needs the model's audio VAE">
        <div class="sw-row"><ZenSwitch :model-value="!!spec.dataset.video_audio" @update:model-value="spec.dataset.video_audio = $event" /><span class="sw-state">{{ spec.dataset.video_audio ? 'On' : 'Off' }}</span></div>
      </ZenField>
      <ZenField v-if="spec.dataset.video_audio" label="Audio VAE" stack tip="The model's audio VAE (MiniMax H3: minimax_h3_audio_vae)">
        <ZenCombo :model-value="spec.model.audio_vae_name ?? ''" :items="items(choices.vae ?? [])" placeholder="Pick the audio VAE…" :item-height="30" @update:model-value="spec.model.audio_vae_name = String($event) || null" />
      </ZenField>
    </div>
  </div>
</section>
</template>

<style scoped src="./form.css"></style>
<style scoped>
.input-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  min-height: 30px;
}

.video-opts {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.sub-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 13px;
}
/* the trigger word, with the dataset's name a click away on its label's line */
.trig-field {
  position: relative;
  min-width: 0;
}
.use-name {
  position: absolute;
  top: 0;
  right: 0;
  max-width: 60%;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 0;
  border: none;
  background: none;
  font: inherit;
  font-size: 11px;
  line-height: 18px;
  color: var(--zen-accent);
  white-space: nowrap;
  cursor: pointer;
}
.use-name:hover {
  text-decoration: underline;
}
.use-name .ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
}
/* the dataset field: what's picked, with a picture; it opens the picker */
.ds-field {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 6px 10px 6px 6px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  background: var(--zen-surface);
  color: var(--zen-text);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.ds-field:hover {
  border-color: color-mix(in srgb, var(--zen-accent) 50%, var(--zen-border));
}
.ds-thumb {
  width: 36px;
  height: 36px;
  flex: none;
  border-radius: var(--r-md);
  overflow: hidden;
  display: grid;
  place-items: center;
  background: var(--zen-surface-2);
  color: var(--zen-muted);
}
.ds-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
/* four pictures, from as many folders: big enough to tell them apart */
.ds-thumb.mosaic {
  width: 64px;
  height: 64px;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
}
.fold-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}
.folds {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 6px;
}
.folds.off {
  opacity: 0.6;
}
.fold {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-md);
  background: var(--zen-surface);
}
.fold img {
  width: 36px;
  height: 36px;
  flex: none;
  border-radius: var(--r-md);
  object-fit: cover;
}
.fold-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  width: 80px;
  flex: none;
}
.fold-text b {
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fold > :last-child {
  flex: 1;
  min-width: 0;
}
.fold :deep(input) {
  min-width: 0;
}
.ds-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}
.ds-text b {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ds-field.empty {
  border-style: dashed;
}
.ds-thumb.blank {
  color: var(--zen-accent);
  font-size: 18px;
}
.ds-change {
  font-size: 12px;
  color: var(--zen-accent);
  display: inline-flex;
  align-items: center;
}
</style>
