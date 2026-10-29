<script setup lang="ts">
// A card in the style of ComfyUI's template picker: a picture on top (one image, two that
// crossfade on hover, a before/after the pointer slides across, or an icon panel when there's
// no picture yet), then the title, a line of description and small tags. Simple mode's
// recipes, Apps and Tools all use it.
import { ZenCompare } from '@nynxz/zenkit-ui'

defineProps<{
  to: string
  title: string
  description?: string
  /** the picture; `after` makes it a pair (slider, or crossfade with mode "dissolve") */
  image?: string | null
  after?: string | null
  mode?: 'slider' | 'dissolve'
  beforeLabel?: string
  afterLabel?: string
  /** shown when there's no image */
  icon?: string
  /** the accent for the icon panel and hover */
  accent?: string
  /** no picture, no panel: the card is its text */
  pictureOnly?: boolean
  /** a small logo before the tags (a model maker's) */
  logo?: string | null
  tags?: string[]
  /** a corner tag on the picture (e.g. "Character") */
  badge?: string
  /** dim line at the bottom (photo count, time) */
  meta?: string[]
  /** a dashed warning tag (e.g. "draft") */
  flag?: string
  flagTitle?: string
}>()
</script>

<template>
  <RouterLink :to="to" class="tcard" :style="accent ? { '--a': accent } : undefined">
    <div v-if="!pictureOnly || image || $slots.thumb" class="thumb">
      <slot v-if="$slots.thumb" name="thumb" />
      <ZenCompare v-else-if="image && after" :before="image" :after="after" :mode="mode ?? 'slider'" :before-label="beforeLabel" :after-label="afterLabel" />
      <img v-else-if="image" :src="image" alt="" loading="lazy" />
      <div v-else class="thumb-none"><i :class="icon ?? 'mdi mdi-image-outline'" /></div>
      <span v-if="badge" class="badge">{{ badge }}</span>
    </div>
    <div class="body">
      <div v-if="logo || tags?.length || flag" class="tags">
        <img v-if="logo" class="logo" :src="logo" alt="" />
        <span v-for="t in tags ?? []" :key="t" class="tag">{{ t }}</span>
        <span v-if="flag" class="flag" :title="flagTitle">{{ flag }}</span>
      </div>
      <b class="title">{{ title }}</b>
      <p v-if="description" class="desc">{{ description }}</p>
      <div v-if="meta?.length" class="meta">
        <span v-for="m in meta" :key="m">{{ m }}</span>
      </div>
      <slot />
    </div>
  </RouterLink>
</template>

<style scoped>
.tcard {
  --a: var(--zen-accent);
  display: flex;
  flex-direction: column;
  border: 1px solid var(--zen-border);
  border-radius: var(--r-lg, 12px);
  overflow: hidden;
  background: var(--zen-surface);
  color: var(--zen-text);
  transition: border-color var(--zen-dur-fast) ease;
}
/* hover: the theme's primary colour on the border; nothing moves */
.tcard:hover,
.tcard:focus-visible {
  border-color: var(--zen-accent);
}
.thumb {
  position: relative;
  aspect-ratio: 4 / 3;
  background: var(--zen-surface-2);
  overflow: hidden;
}
.thumb > img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.4s ease;
}
.tcard:hover .thumb > img {
  transform: scale(1.04);
}
.thumb-none {
  height: 100%;
  display: grid;
  place-items: center;
  font-size: 42px;
  color: color-mix(in srgb, var(--a) 75%, var(--zen-muted));
  background: radial-gradient(circle at 50% 40%, color-mix(in srgb, var(--a) 16%, transparent), transparent 70%);
}
.tcard:hover .thumb-none i {
  transform: scale(1.08);
}
.thumb-none i {
  transition: transform var(--zen-dur) ease;
}
.badge {
  position: absolute;
  top: 10px;
  left: 10px;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  backdrop-filter: blur(6px);
}
.body {
  padding: 14px 16px 16px;
  display: flex;
  flex-direction: column;
  gap: 7px;
  flex: 1;
}
.tags {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.logo {
  width: 20px;
  height: 20px;
  border-radius: 5px;
  object-fit: cover;
  background: #fff;
}
.tag {
  font-size: 11px;
  font-weight: 600;
  padding: 1px 8px;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--zen-border) 80%, transparent);
}
.flag {
  font-size: 10px;
  padding: 0 6px;
  border-radius: 999px;
  color: var(--zen-warn);
  border: 1px dashed color-mix(in srgb, var(--zen-warn) 60%, transparent);
}
.title {
  font-size: 15px;
}
.desc {
  margin: 0;
  font-size: 13px;
  line-height: 1.5;
  color: var(--zen-muted);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.meta {
  display: flex;
  gap: 14px;
  margin-top: auto;
  padding-top: 2px;
  font-size: 12px;
  color: var(--zen-muted);
}
</style>
