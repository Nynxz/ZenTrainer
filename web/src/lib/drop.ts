// Files dropped on a page from outside it (Explorer, Finder, another window): the whole page is the
// target, a cover says what a drop does while files are dragged over it, and a drop's folders count
// as their files.
import { onMounted, onUnmounted, ref, type Ref } from 'vue'

export interface Drop {
  files: File[]
  /** the folder's name when the drop was one folder: a dataset's name, say */
  folder: string | null
}

/** Files for a dataset's page to add as it opens: dropped on the Datasets page, which made the dataset. */
export const handoff: { name: string; files: File[] } = { name: '', files: [] }

// A drag that starts on this page (a thumbnail, selected text) carries this mark: only files from
// elsewhere are taken. The mark rides on the drag itself, so it can't go stale the way a flag set
// on dragstart did when the drag's dragend never came (the picture under it re-rendered): every
// drop after that was ignored until the page was opened again
const MINE = 'application/x-zentrainer-drag'
const fromOutside = (e: DragEvent) => !!e.dataTransfer?.types.includes('Files') && !e.dataTransfer.types.includes(MINE)

/** The page as a drop target: true while files from outside it are dragged across; `onDrop` gets what lands. */
export function useFileDrop(onDrop: (d: Drop) => void): Ref<boolean> {
  const over = ref(false)
  let depth = 0 // dragenter and dragleave come for every element crossed
  const on: Record<string, (e: DragEvent) => void> = {
    dragstart: (e) => e.dataTransfer?.setData(MINE, '1'),
    dragenter: (e) => {
      if (!fromOutside(e)) return
      depth++
      over.value = true
    },
    dragleave: () => {
      if (--depth <= 0) {
        depth = 0
        over.value = false
      }
    },
    dragover: (e) => e.preventDefault(),
    drop: (e) => {
      e.preventDefault() // never the browser opening the file instead
      depth = 0
      over.value = false
      if (fromOutside(e)) dropped(e.dataTransfer!).then(onDrop)
    },
  }
  onMounted(() => Object.entries(on).forEach(([k, f]) => addEventListener(k, f as EventListener)))
  onUnmounted(() => Object.entries(on).forEach(([k, f]) => removeEventListener(k, f as EventListener)))
  return over
}

// What a drop holds: its files, and those in any dropped folder (however deep) by their names,
// where the browser alone gives a folder as one empty, extensionless "file". A ZenTrainer dataset
// folder comes through as itself: _masks/<stem>.png is <stem>-masklabel.png again, which the
// upload puts back; _controls (an input set: those are added from the side panel) and hidden files
// (.DS_Store) stay out. The items and files are read before the first await: the browser empties
// them when the drop event ends
async function dropped(dt: DataTransfer): Promise<Drop> {
  const entries = [...dt.items].map((i) => i.webkitGetAsEntry?.() ?? null).filter((e): e is FileSystemEntry => !!e)
  if (!entries.some((e) => e.isDirectory)) return { files: [...dt.files], folder: null }
  const files: File[] = []
  const walk = async (e: FileSystemEntry) => {
    if (e.name.startsWith('.') || e.name === '_controls') return
    if (e.isFile) {
      const f = await new Promise<File>((ok, no) => (e as FileSystemFileEntry).file(ok, no))
      files.push(e.fullPath.includes('/_masks/') ? new File([f], f.name.replace(/\.[^.]+$/, '-masklabel$&'), { type: f.type }) : f)
      return
    }
    const reader = (e as FileSystemDirectoryEntry).createReader()
    for (;;) {
      const some = await new Promise<FileSystemEntry[]>((ok, no) => reader.readEntries(ok, no)) // a hundred at a time
      if (!some.length) break
      for (const c of some) await walk(c)
    }
  }
  for (const e of entries) await walk(e)
  return { files, folder: entries.length === 1 && entries[0]!.isDirectory ? entries[0]!.name : null }
}
