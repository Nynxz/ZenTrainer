<script setup lang="ts">
// The dialog behind lib/ask (ask, askText, tell): mounted once, in App.vue. Enter says yes,
// Escape or closing it says no.
import { ZenButton, ZenInput, ZenModal } from '@nynxz/zenkit-ui'
import { computed, nextTick, ref, watch } from 'vue'
import { current } from '@/lib/ask'

const text = ref('')
const body = ref<HTMLElement | null>(null)
const problem = computed(() => (current.value?.input?.check && text.value.trim() ? current.value.input.check(text.value.trim()) : null))
const canSay = computed(() => !current.value?.input || (!!text.value.trim() && !problem.value))

watch(current, async (a) => {
  if (!a) return
  text.value = a.input?.value ?? ''
  await nextTick()
  // the text box when there is one, else the yes button: Enter answers either way
  ;(body.value?.querySelector('input') ?? document.querySelector<HTMLElement>('.ask-foot .yes'))?.focus()
})

function yes() {
  if (!current.value || !canSay.value) return
  current.value.answer(current.value.input ? text.value.trim() : true)
}
// not the Enter that picks a character in an input method (Japanese, Chinese...)
function onEnter(e: KeyboardEvent) {
  if (e.isComposing) return
  e.preventDefault()
  yes()
}
function no() {
  current.value?.answer(current.value.input ? null : false)
}
</script>

<template>
  <ZenModal :open="!!current" :title="current?.title" width="440px" height="auto" @update:open="(o) => !o && no()">
    <div v-if="current" ref="body" class="ask" @keydown.enter="onEnter">
      <p v-if="current.message" class="msg">{{ current.message }}</p>
      <template v-if="current.input">
        <ZenInput v-model="text" :placeholder="current.input.placeholder" />
        <p v-if="problem" class="err small"><i class="mdi mdi-alert-outline" /> {{ problem }}</p>
      </template>
    </div>
    <template #footer>
      <div v-if="current" class="ask-foot" @keydown.enter.stop>
        <ZenButton v-if="current.cancel !== false" @click="no">{{ current.cancel ?? 'Cancel' }}</ZenButton>
        <ZenButton class="yes" :variant="current.danger ? 'danger' : 'primary'" :disabled="!canSay" @click="yes">{{ current.ok ?? 'OK' }}</ZenButton>
      </div>
    </template>
  </ZenModal>
</template>

<style scoped>
.ask {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.msg {
  margin: 0;
  line-height: 1.5;
  white-space: pre-line;
  color: var(--zen-text);
}
.err {
  margin: 0;
  color: var(--zen-warn);
}
.ask-foot {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  width: 100%;
}
</style>
