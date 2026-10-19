import { use, useEffect, useId, useRef, useState, type MouseEvent, type PointerEvent, type ReactNode } from 'react'
import { Inside } from './context'
import { HIDE_MS, place, placeLeft, portal, SHOW_MS } from './popup'

type Props = {
  term: ReactNode
  title?: ReactNode
  /** Name of the term for screen readers, when the term is not text. */
  label?: string
  /** More classes of the term. */
  className?: string
  /** 'left': the popup goes left of the popover the term sits in (a list), level with the term, not under it. */
  side?: 'left'
  children: ReactNode
}

/**
 * A term with a popup that explains it. With a mouse the popup shows while the pointer rests on the term or on the
 * popup; a tap, or Enter on the focused term, toggles it. A click elsewhere or Esc closes it. It is a popover, so it
 * sits above everything and no scrolling column clips it. It is rendered in <body>, so it can hold paragraphs even when
 * the term sits in one. A popup opened from a term in another popup (a citation in a footnote) is rendered inside
 * that popup: the browser then keeps both open, and moving the mouse into it doesn't leave the other.
 */
export function Gloss({ term, title, label, className, side, children }: Props) {
  const id = useId()
  const termRef = useRef<HTMLButtonElement>(null)
  const popRef = useRef<HTMLDivElement>(null)
  const timer = useRef(0)
  const press = useRef('') // pointer type of the press before a click on the term; '' for a key
  const parent = use(Inside)
  const [inside, setInside] = useState<HTMLElement | null>(null)

  useEffect(() => {
    const term = termRef.current!
    const pop = popRef.current!
    const box = side === 'left' ? term.closest('[popover]') : null
    const put = () => (box ? placeLeft(term, pop, box) : place(term, pop))
    const onMove = put
    const follow = (on: boolean) => {
      const method = on ? addEventListener : removeEventListener
      method('scroll', onMove, { capture: true, passive: true }) // scrolls of any column
      method('resize', onMove)
    }
    // The toggle event comes after the popup is shown, so it stays hidden (CSS) until it is placed.
    const onBeforeToggle = (e: Event) => {
      if ((e as ToggleEvent).newState === 'open') delete pop.dataset.placed
    }
    const onToggle = (e: Event) => {
      const open = (e as ToggleEvent).newState === 'open'
      if (open) put()
      term.toggleAttribute('data-open', open)
      follow(open)
    }
    pop.addEventListener('beforetoggle', onBeforeToggle)
    pop.addEventListener('toggle', onToggle)
    return () => {
      clearTimeout(timer.current)
      pop.removeEventListener('beforetoggle', onBeforeToggle)
      pop.removeEventListener('toggle', onToggle)
      follow(false)
    }
  }, [side])

  const setOpen = (open: boolean) => {
    const pop = popRef.current
    if (pop && pop.matches(':popover-open') !== open) pop.togglePopover(open)
  }
  const later = (open: boolean, ms: number) => {
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setOpen(open), ms)
  }

  // Mouse only: touch and pen tap instead.
  const onEnter = (e: PointerEvent) => {
    if (e.pointerType === 'mouse') later(true, popRef.current!.matches(':popover-open') ? 0 : SHOW_MS)
  }
  const onLeave = (e: PointerEvent) => {
    if (e.pointerType === 'mouse') later(false, HIDE_MS)
  }

  // The button toggles its popup, but a mouse click would close what hovering opened: it only shows.
  const onClick = (e: MouseEvent) => {
    if (press.current === 'mouse') {
      e.preventDefault()
      later(true, 0)
    }
    press.current = ''
  }

  // A link in the popup opens a note or a new tab: close the popup.
  const onPopupClick = (e: MouseEvent) => {
    if ((e.target as Element).closest('a')) setOpen(false)
  }

  return (
    <>
      <button
        ref={termRef}
        type="button"
        className={className ? `gloss-term ${className}` : 'gloss-term'}
        aria-label={label}
        popoverTarget={id}
        onPointerDown={(e) => (press.current = e.pointerType)}
        onPointerEnter={onEnter}
        onPointerLeave={onLeave}
        onClick={onClick}
      >
        {term}
      </button>
      {portal(
        <div
          ref={popRef}
          id={id}
          popover="auto"
          className="gloss-pop"
          onPointerEnter={onEnter}
          onPointerLeave={onLeave}
          onClick={onPopupClick}
        >
          {/* First, so that the last paragraph stays the popup's last child. Terms inside render once it exists. */}
          <div ref={setInside} className="gloss-inside" />
          {title && <p className="gloss-title">{title}</p>}
          {inside && <Inside value={inside}>{children}</Inside>}
        </div>,
        parent,
      )}
    </>
  )
}
