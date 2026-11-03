// The app's own confirm / alert / prompt: one dialog in the app's look (AskDialog, mounted once in
// App.vue) instead of the browser's. Each returns a promise; asks while one is open wait their turn.

import { shallowRef } from 'vue'

export interface Asking {
  title: string
  /** more about it, under the title (newlines kept) */
  message?: string
  /** the yes button's label */
  ok?: string
  /** the no button's label; false: none (a notice) */
  cancel?: string | false
  /** the yes button in the danger colour: it deletes or replaces something */
  danger?: boolean
  /** a text box: what's typed is the answer */
  input?: {
    value?: string
    placeholder?: string
    /** what's wrong with it, said under the box (and the yes button waits), or null when it's fine */
    check?: (v: string) => string | null
  }
}

export const current = shallowRef<(Asking & { answer: (v: string | boolean | null) => void }) | null>(null)
const waiting: (() => void)[] = []

function open<T>(a: Asking): Promise<T> {
  return new Promise<T>((resolve) => {
    const show = () => {
      current.value = {
        ...a,
        answer: (v) => {
          current.value = null
          resolve(v as T)
          waiting.shift()?.()
        },
      }
    }
    if (current.value) waiting.push(show)
    else show()
  })
}

/** Yes or no: true when they said yes. */
export const ask = (a: Omit<Asking, 'input'>) => open<boolean>(a)

/** A word or a line: what they typed (trimmed), or null when they cancelled. */
export const askText = (a: Asking & { input: NonNullable<Asking['input']> }) => open<string | null>(a)

/** A notice with just an OK button: something didn't work. */
export const tell = (title: string, message?: string) => open<boolean>({ title, message, ok: 'OK', cancel: false }).then(() => {})

/** An error from a call, as a notice. */
export const tellError = (e: unknown, title = 'That didn’t work') => tell(title, String((e as Error)?.message ?? e))
