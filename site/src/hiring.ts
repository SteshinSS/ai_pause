import { useSyncExternalStore } from 'react'

/**
 * Whether the result tables (ResultTable.tsx) show the lab with unlimited hiring or with only the staff it had at the
 * start of the pause. One switch for every table on the page, off when the page loads.
 */
let on = false
const listeners = new Set<() => void>()

export function setHiring(value: boolean) {
  on = value
  listeners.forEach((listener) => listener())
}

export function useHiring() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    () => on,
    () => on, // at build time (src/prerender.ts)
  )
}
