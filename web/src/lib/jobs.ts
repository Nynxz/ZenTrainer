// Two copies of a job (an event heard, the server's answer to a request) can arrive out of order:
// the one further along is the newer.

const STAGE: Record<string, number> = { queued: 0, running: 1, undone: 3 } // anything else: finished (2)

/** a is at least as far along as b: queued, then running (by pictures done), finished, undone */
export function asFarAlong(a: { status: string; done: number }, b: { status: string; done: number }): boolean {
  const sa = STAGE[a.status] ?? 2
  const sb = STAGE[b.status] ?? 2
  return sa !== sb ? sa > sb : sa !== 1 || a.done >= b.done
}
