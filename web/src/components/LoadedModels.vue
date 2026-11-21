<script setup lang="ts">
// The models ComfyUI holds right now and how much of each sits in VRAM: one strip per numbered
// list of blocks (blocks.N, double_blocks.N, layers.N, ...), one cell per block filled by its
// share on the GPU, then the weights outside any list (embeddings, heads). With offloading
// you see blocks stream in and out.
import { computed } from 'vue'
import { loadedModels as models, startLoadedSync } from '@/stores/loaded'
import { gb } from '@/stores/system'

startLoadedSync()

const BLOCK = /^(?:(.*)\.)?([^.]+)\.(\d+)$/ // "....visual.blocks.3" -> prefix, list, index
const rows = computed(() =>
  models.value
    .filter((m) => m.bytes > 0)
    .map((m) => {
      // the block lists, each in order; everything else is "other"
      const lists = new Map<string, { i: number; bytes: number; vram: number; name: string }[]>()
      let other = { bytes: 0, vram: 0 }
      for (const g of m.groups ?? []) {
        const hit = g.name.match(BLOCK)
        if (hit) {
          const key = hit[1] ? `${hit[1]}.${hit[2]}` : hit[2]!
          lists.set(key, [...(lists.get(key) ?? []), { i: Number(hit[3]), bytes: g.bytes, vram: g.vram, name: g.name }])
        } else other = { bytes: other.bytes + g.bytes, vram: other.vram + g.vram }
      }
      return {
        ...m,
        lists: [...lists.entries()]
          .map(([k, v]) => ({ name: label(k), cells: v.sort((a, b) => a.i - b.i) }))
          .sort((a, b) => sum(b.cells) - sum(a.cells)),
        other,
      }
    }),
)
const sum = (cells: { bytes: number }[]) => cells.reduce((n, c) => n + c.bytes, 0)
// the list's own name, plus what it belongs to when that tells lists apart
// (transformer.model.layers -> layers; visual.blocks -> visual blocks)
function label(key: string) {
  const parts = key.split('.').filter((p) => !/^(model|diffusion_model|transformer|clip_[lg]|encoder|decoder)$/.test(p))
  const last = parts.pop() ?? key
  const owner = parts.pop()
  return (owner && !/^\d+$/.test(owner) ? `${owner} ${last}` : last).replace(/_/g, ' ')
}
const pct = (v: number, b: number) => (b ? Math.round((v / b) * 100) : 0)
</script>

<template>
  <section class="lm">
    <div class="head">
      <span class="eyebrow">Loaded models</span>
      <span class="muted small">filled cells are in VRAM</span>
    </div>
    <div v-if="!rows.length" class="muted small">Nothing loaded.</div>
    <div v-for="m in rows" :key="m.name" class="model">
      <div class="line">
        <b class="ellipsis" :title="m.file ?? m.name">{{ m.file ?? m.name }}</b>
        <span class="muted small kind">{{ m.kind ?? '' }}{{ m.file ? ` · ${m.name}` : '' }}</span>
        <span v-if="m.dynamic" class="tag" title="Streams weights into VRAM as it needs them (dynamic VRAM)">streaming</span>
        <span class="spacer" />
        <span class="mono small">{{ gb(m.vram) }} / {{ gb(m.bytes) }} GB · {{ pct(m.vram, m.bytes) }}%</span>
      </div>
      <div v-for="l in m.lists" :key="l.name" class="list">
        <span class="lbl small muted ellipsis" :title="l.name">{{ l.name }} <span class="mono">{{ l.cells.length }}</span></span>
        <div class="strip">
          <span
            v-for="c in l.cells"
            :key="c.i"
            class="cell"
            :style="{ '--f': c.bytes ? c.vram / c.bytes : 0 }"
            :title="`${c.name}: ${pct(c.vram, c.bytes)}% in VRAM (${gb(c.bytes)} GB)`"
          />
        </div>
      </div>
      <div v-if="m.other.bytes" class="list" :title="`Weights outside the block lists: ${pct(m.other.vram, m.other.bytes)}% in VRAM`">
        <span class="lbl small muted">{{ m.lists.length ? 'other' : 'weights' }} <span class="mono">{{ gb(m.other.bytes) }} GB</span></span>
        <div class="bar"><span :style="{ width: `${pct(m.other.vram, m.other.bytes)}%` }" /></div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.lm {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.head,
.line {
  display: flex;
  align-items: center;
  gap: 8px;
}
.model {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.line b {
  font-size: 12px;
  min-width: 0;
}
.kind {
  white-space: nowrap;
}
.tag {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: var(--r-sm);
  color: var(--zen-muted);
  border: 1px solid color-mix(in srgb, var(--zen-border) 70%, transparent);
}
.list {
  display: grid;
  grid-template-columns: 120px 1fr;
  align-items: center;
  gap: 8px;
}
.lbl {
  min-width: 0;
}
.strip {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: minmax(3px, 1fr);
  gap: 2px;
  height: 14px;
}
.cell {
  border-radius: 2px;
  background: linear-gradient(to top, var(--zen-accent) calc(var(--f) * 100%), color-mix(in srgb, var(--zen-text) 10%, transparent) 0);
}
.bar {
  height: 6px;
  border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--zen-text) 10%, transparent);
  overflow: hidden;
}
.bar span {
  display: block;
  height: 100%;
  background: var(--zen-accent);
}
</style>
