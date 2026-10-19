import { useLayoutEffect, useRef, useSyncExternalStore, type ReactNode } from 'react'

/**
 * Which derivations are open, by id. A note and its companion (content.ts) each have a `<Derivation>` with the same
 * id, the formulas and their explanation: opening one opens the other. All closed when the page loads.
 */
const open = new Set<string>()
const listeners = new Set<() => void>()

/** Where the derivations that open or close with the one the reader toggled were on screen, until they have. */
const were = new Map<HTMLDetailsElement, DOMRect>()

function setOpen(id: string, value: boolean, from: HTMLDetailsElement) {
  if (open.has(id) === value) return
  if (value) open.add(id)
  else open.delete(id)
  for (const el of document.querySelectorAll<HTMLDetailsElement>('details.derivation')) {
    if (el !== from && el.dataset.id === id) were.set(el, el.getBoundingClientRect())
  }
  listeners.forEach((listener) => listener())
}

function useOpen(id: string) {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    () => open.has(id),
    () => open.has(id), // at build time (src/prerender.ts)
  )
}

/**
 * Scrolls the column of derivation `el`, which opened or closed with its twin, so that what the reader saw there stays:
 * above the view, the text after it stays where it was; else its label stays, or comes to the top of the view when the
 * view was inside the steps.
 */
function keep(el: HTMLDetailsElement, was: DOMRect) {
  const scroller = el.closest<HTMLElement>('.pane-scroll')
  if (!scroller) return
  const view = scroller.getBoundingClientRect().top
  const now = el.getBoundingClientRect()
  scroller.scrollTop += was.bottom <= view ? now.bottom - was.bottom : now.top - Math.max(was.top, view)
}

/**
 * `<Derivation id="…">` in an .mdx file around the steps behind a formula: closed, only a small label shows under the
 * formula; a click opens the steps, here and in the companion's `<Derivation>` with the same id.
 */
export function Derivation({ id, label = 'Derivation', children }: { id: string; label?: string; children: ReactNode }) {
  const isOpen = useOpen(id)
  const ref = useRef<HTMLDetailsElement>(null)
  useLayoutEffect(() => {
    const el = ref.current!
    const was = were.get(el)
    were.delete(el)
    if (was) keep(el, was)
  }, [isOpen])
  return (
    <details
      ref={ref}
      className="derivation"
      data-id={id}
      open={isOpen}
      onToggle={(e) => setOpen(id, e.currentTarget.open, e.currentTarget)}
    >
      <summary>{label}</summary>
      {children}
    </details>
  )
}
