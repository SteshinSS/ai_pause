import { LEAD } from './sync'

/** The heading with address `section` (headings.ts) in the text of `scope`, or null. */
export const findSection = (scope: Element, section: string) =>
  scope.querySelector<HTMLElement>(`.pane-body [data-section="${CSS.escape(section)}"]`)

/** The scroller of heading `h`'s column and the position that puts the heading just below the column's bar. A column
 *  scrolls inside itself, except the main column on a phone, which scrolls with the page. Space that lines the column
 *  up with its companion (sync.ts) stays above the top. */
function target(h: HTMLElement) {
  const pane = h.closest<HTMLElement>('.pane-scroll')!
  const own = pane.scrollHeight > pane.clientHeight && getComputedStyle(pane).overflowY !== 'visible'
  const scroller = own ? pane : document.scrollingElement!
  const top = own ? pane.getBoundingClientRect().top : 0
  const bar = pane.querySelector<HTMLElement>('.pane-bar')?.offsetHeight ?? 0
  const space = parseFloat(h.style.paddingTop) || 0
  const y = scroller.scrollTop + h.getBoundingClientRect().top + space - top - bar - 16
  return { scroller, y: Math.max(0, Math.min(y, scroller.scrollHeight - scroller.clientHeight)) }
}

/** Scrolls the column of heading `h` to it, just below the column's bar. */
export function scrollToHeading(h: HTMLElement) {
  const { scroller, y } = target(h)
  scroller.scrollTo({ top: y })
}

/** Scrolls the column of heading `h` to it smoothly, longer the farther it goes (GLIDE), and calls `done` once there.
 *  The heading may move as the text settles: each frame closes in on where it is then. Returns the function that stops
 *  it where it is. */
function glide(h: HTMLElement, done: () => void) {
  const { scroller, y } = target(h)
  const gap = y - scroller.scrollTop // the way to go, at the start
  const duration = Math.min(GLIDE.max, GLIDE.min + GLIDE.per * Math.log2(1 + Math.abs(gap) / GLIDE.px))
  let start = NaN
  let frame = requestAnimationFrame(function step(now) {
    if (isNaN(start)) start = now
    const t = Math.min(1, (now - start) / duration)
    scroller.scrollTop = target(h).y - gap * (1 - ease(t))
    if (t < 1) frame = requestAnimationFrame(step)
    else done()
  })
  return () => cancelAnimationFrame(frame)
}

/** Glide times, in ms: `min` plus `per` for each doubling of 1 + distance / `px`, at most `max`. */
const GLIDE = { min: 400, per: 160, px: 500, max: 1200 }

/** Slow at both ends: cubic ease-in-out. */
const ease = (t: number) => (t < 0.5 ? 4 * t ** 3 : 1 - (2 - 2 * t) ** 3 / 2)

/** Scrolls to heading `h`, at once or, with `smooth`, gliding there (glide()), and keeps it there as the text settles
 *  (fonts arrive, a companion lines up), until the reader scrolls, clicks or types, or HOLD ms pass. Its column leads
 *  its companion (sync.ts). Returns the function that stops keeping it there. */
export function showSection(h: HTMLElement, smooth = false) {
  h.closest('.pane-scroll')?.dispatchEvent(new Event(LEAD))
  let gliding = smooth
  const halt = smooth ? glide(h, () => (gliding = false)) : () => {}
  if (!smooth) scrollToHeading(h)
  const observer = new ResizeObserver(() => gliding || scrollToHeading(h))
  observer.observe(h.closest('.pane-body')!)
  const inputs = ['wheel', 'pointerdown', 'touchstart', 'keydown']
  const stop = () => {
    halt()
    observer.disconnect()
    clearTimeout(timer)
    for (const type of inputs) removeEventListener(type, stop, true)
  }
  for (const type of inputs) addEventListener(type, stop, { capture: true, passive: true })
  const timer = setTimeout(stop, HOLD)
  return stop
}

const HOLD = 10_000
