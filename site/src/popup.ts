// Timing and placement of popups: a term's (Gloss.tsx) and a formula symbol's (Symbols.tsx).

import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'

/** A popup rendered into `into`, or into <body>. At build time (src/prerender.ts) there is no page, so nothing. */
export const portal = (popup: ReactNode, into: HTMLElement | null) =>
  typeof document === 'undefined' ? null : createPortal(popup, into ?? document.body)

export const SHOW_MS = 100 // the mouse rests this long on a term before its popup shows
export const HIDE_MS = 200 // time to move the mouse from the term into its popup
const GAP = 6 // px between the term and its popup
const EDGE = 8 // px the popup keeps from the edges of the window

/** Puts popup `pop` under `term`, or over it if there is no room below; inside the window. */
export function place(term: Element, pop: HTMLElement) {
  const t = term.getBoundingClientRect()
  const { width, height } = pop.getBoundingClientRect()
  const { clientWidth, clientHeight } = document.documentElement
  const below = t.bottom + GAP
  const above = t.top - GAP - height
  pop.style.left = `${Math.max(EDGE, Math.min(t.left, clientWidth - width - EDGE))}px`
  pop.style.top = `${below + height > clientHeight - EDGE && above >= EDGE ? above : below}px`
  pop.dataset.placed = ''
}

/** Puts popup `pop` left of `box` (the list `term` sits in), level with `term`, so that it covers none of the box;
 *  under or over `term` when there is no room on the left (place). */
export function placeLeft(term: Element, pop: HTMLElement, box: Element) {
  const t = term.getBoundingClientRect()
  const left = box.getBoundingClientRect().left - GAP - pop.getBoundingClientRect().width
  if (left < EDGE) return place(term, pop)
  const { height } = pop.getBoundingClientRect()
  const top = t.top + t.height / 2 - 22 // its title level with the term
  pop.style.left = `${left}px`
  pop.style.top = `${Math.max(EDGE, Math.min(top, document.documentElement.clientHeight - height - EDGE))}px`
  pop.dataset.placed = ''
}
