import {
  createContext,
  use,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
  type RefObject,
} from 'react'
import sensitivity from '../content/sensitivity.json'
import supplement from '../content/supplement.json'
import table from '../content/table.json'
import { Cite } from './Cite'
import { SYMBOLS as Entries } from './content'
import { Inside, PaneContext, StackContext } from './context'
import { Footnote } from './Footnote'
import { MdxLink, NoteLink } from './NoteLink'
import { HIDE_MS, place, portal, SHOW_MS } from './popup'

/** The id of the entry the popup shows. */
const Shown = createContext<string | undefined>(undefined)

const INPUTS: Record<string, unknown> = table.inputs
const AIFM: Record<string, unknown> = table.aifm_parameters
/** Titles of the headings of AIFM's supplementary materials, by id (scripts/supplement.mjs). */
const SECTIONS: Record<string, string> = supplement.headings
/** The note with the estimates of our inputs, content/notes/parameters.mdx: their popups link to it. */
const PARAMETERS = 'parameters'

/** Three significant digits, whole numbers in full, with a minus sign. */
function format(v: number) {
  const s = Number.isInteger(v) ? v.toLocaleString('en-US') : v.toLocaleString('en-US', { maximumSignificantDigits: 3 })
  return s.replace('-', '−')
}

/**
 * A value of content/table.json formatted as in the popups, `<Value input="m_C" />` in .mdx files with no import
 * (Pane.tsx provides it): an input of pause_inputs.csv, or else one of AIFM's parameters.
 */
export function Value({ input }: { input: string }) {
  const raw = Object.hasOwn(INPUTS, input) ? INPUTS[input] : AIFM[input]
  return typeof raw === 'number' ? format(raw) : String(raw)
}

/** The p10 and p90 of our inputs, by name: pause_inputs.csv's, as sensitivity.py writes them to sensitivity.json. */
const RANGES: Record<string, [number, number]> = Object.fromEntries(
  sensitivity.ours.map((it) => [it.name, [it.p10, it.p90]]),
)

/**
 * An input's `[p10; p90]` of pause_inputs.csv, `<Range input="m_C" />` in .mdx files with no import (Pane.tsx provides
 * it), in the units of its value; nothing for an input with none.
 */
export function Range({ input }: { input: string }) {
  const range = RANGES[input]
  if (!range) return null
  return <span className="range">[{format(range[0])}; {format(range[1])}]</span>
}

type EntryProps = {
  id: string
  unit?: string
  input?: string
  value?: string
  supplement?: string
  children?: ReactNode
}

/**
 * An entry of content/symbols.mdx, `<Symbol tex="m_C" unit="…" input="m_C">what it is</Symbol>`, of which a popup
 * shows only the one it is for. symbols.ts gave it its id and set its symbol in maths above its text. `input` names a
 * value of content/table.json: an input of pause_inputs.csv, whose popup links to their estimates (PARAMETERS) unless
 * it is open there, or else one of AIFM's parameters. `supplement` lists headings of AIFM's supplementary materials by
 * id, which the popup links to, titled as in content/supplement.json.
 */
function SymbolEntry({ id, unit, input, value, supplement: headings = '', children }: EntryProps) {
  const pane = use(PaneContext)
  const { ids } = use(StackContext)
  if (use(Shown) !== id) return null
  const ours = input !== undefined && Object.hasOwn(INPUTS, input)
  const aifm = input !== undefined && !ours
  const raw = input === undefined ? value : aifm ? AIFM[input] : INPUTS[input]
  const shown = typeof raw === 'number' ? format(raw) : raw === undefined ? undefined : String(raw)
  const sections = headings.split(/\s+/).filter(Boolean)
  return (
    <>
      {children}
      {(unit || shown !== undefined || sections.length > 0) && (
        <dl className="symbol-facts">
          {unit && (
            <div>
              <dt>Units</dt>
              <dd>{unit}</dd>
            </div>
          )}
          {shown !== undefined && (
            <div>
              <dt>Value</dt>
              <dd>
                {shown}
                {aifm && <span className="symbol-source">AIFM</span>}
                {ours && ids[pane - 1] !== PARAMETERS && (
                  <span className="symbol-estimate">
                    <NoteLink id={PARAMETERS}>our estimate</NoteLink>
                  </span>
                )}
              </dd>
            </div>
          )}
          {sections.length > 0 && (
            <div>
              <dt>AIFM</dt>
              <dd className="symbol-sections">
                {sections.map((h) => (
                  <a
                    key={h}
                    href={`${supplement.url}#heading=${h}`}
                    target="_blank"
                    rel="noreferrer"
                    title="Opens AIFM's supplementary materials, a Google Doc, at this section"
                  >
                    {SECTIONS[h] ?? h}
                  </a>
                ))}
              </dd>
            </div>
          )}
        </dl>
      )}
    </>
  )
}

const components = { a: MdxLink, Cite, Footnote, Symbol: SymbolEntry }

/**
 * The popup of the symbols in the formulas of `scope`, a note's text, or, without `scope`, of the popup it is rendered
 * in: a footnote's, whose text is the note's (Pane.tsx). symbols.ts marks each symbol that has an entry in
 * content/symbols.mdx with `data-sym`. With a mouse the popup shows while the pointer rests on a symbol or on the
 * popup, and moves along to the next symbol; a tap toggles it. A click elsewhere or Esc closes it. One popover in
 * <body> serves the whole text, like Gloss's popups but without a button per symbol: it only listens to the text. In
 * a popup it renders inside that popup instead, as Gloss's do, so that opening it leaves that one open.
 */
export function SymbolPopup({ scope }: { scope?: RefObject<HTMLElement | null> }) {
  const parent = use(Inside)
  const [shown, setShown] = useState<HTMLElement | null>(null) // the symbol whose entry shows
  const shownRef = useRef<HTMLElement | null>(null) // the same, for the event listeners
  const popRef = useRef<HTMLDivElement>(null)
  const timer = useRef(0)
  const [inside, setInside] = useState<HTMLElement | null>(null)

  const later = (f: () => void, ms: number) => {
    clearTimeout(timer.current)
    timer.current = setTimeout(f, ms)
  }
  const hide = () => {
    const pop = popRef.current
    if (pop?.matches(':popover-open')) pop.hidePopover()
  }

  useEffect(() => {
    const root = scope ? scope.current : parent?.closest<HTMLElement>('.gloss-pop')
    const pop = popRef.current
    if (!root || !pop) return
    const isOpen = () => pop.matches(':popover-open')
    const symbolOf = (target: EventTarget | null) =>
      target instanceof Element ? target.closest<HTMLElement>('[data-sym]') : null
    let press = '' // pointer type of the press before a click
    let pressedOpen: Element | null = null // the symbol pressed while its popup was open

    // Mouse only: touch and pen tap instead. In a footnote the popup itself sits in `root`: moving in it isn't leaving.
    const onOver = (e: globalThis.PointerEvent) => {
      if (e.pointerType !== 'mouse' || pop.contains(e.target as Node)) return
      const symbol = symbolOf(e.target)
      if (symbol) later(() => setShown(symbol), isOpen() ? 0 : SHOW_MS)
      else if (isOpen()) later(hide, HIDE_MS)
      else clearTimeout(timer.current)
    }
    const onOut = (e: globalThis.PointerEvent) => {
      if (e.pointerType !== 'mouse' || root.contains(e.relatedTarget as Node | null)) return
      if (isOpen()) later(hide, HIDE_MS)
      else clearTimeout(timer.current)
    }
    const onDown = (e: globalThis.PointerEvent) => {
      press = e.pointerType
      const symbol = symbolOf(e.target)
      pressedOpen = symbol && symbol === shownRef.current && isOpen() ? symbol : null
    }
    // A press outside the popup closes it before the click (light dismiss). A tap on the open symbol leaves it
    // closed; a mouse click only shows, as hovering already opened it.
    const onClick = (e: globalThis.MouseEvent) => {
      const symbol = symbolOf(e.target)
      if (!symbol || (press !== 'mouse' && symbol === pressedOpen)) return
      clearTimeout(timer.current)
      if (symbol !== shownRef.current) setShown(symbol)
      else if (!isOpen()) {
        pop.showPopover()
        place(symbol, pop)
      }
    }
    const onBeforeToggle = (e: Event) => {
      if ((e as ToggleEvent).newState === 'open') delete pop.dataset.placed
    }
    // Toggle events of a close and a reopen in one click merge into one: ask the popup, not the event.
    const onToggle = () => {
      if (!isOpen()) setShown(null)
    }
    root.addEventListener('pointerover', onOver)
    root.addEventListener('pointerout', onOut)
    root.addEventListener('pointerdown', onDown)
    root.addEventListener('click', onClick)
    pop.addEventListener('beforetoggle', onBeforeToggle)
    pop.addEventListener('toggle', onToggle)
    return () => {
      clearTimeout(timer.current)
      root.removeEventListener('pointerover', onOver)
      root.removeEventListener('pointerout', onOut)
      root.removeEventListener('pointerdown', onDown)
      root.removeEventListener('click', onClick)
      pop.removeEventListener('beforetoggle', onBeforeToggle)
      pop.removeEventListener('toggle', onToggle)
    }
  }, [scope, parent])

  // Once the entry is rendered: open, place under the symbol, follow it as columns scroll, tint it.
  useLayoutEffect(() => {
    shownRef.current = shown
    const pop = popRef.current
    if (!shown || !pop) return
    if (!pop.matches(':popover-open')) pop.showPopover()
    const onMove = () => place(shown, pop)
    onMove()
    shown.toggleAttribute('data-open', true)
    addEventListener('scroll', onMove, { capture: true, passive: true })
    addEventListener('resize', onMove)
    return () => {
      shown.removeAttribute('data-open')
      removeEventListener('scroll', onMove, { capture: true })
      removeEventListener('resize', onMove)
    }
  }, [shown])

  const onEnter = (e: PointerEvent) => {
    if (e.pointerType === 'mouse') clearTimeout(timer.current)
  }
  const onLeave = (e: PointerEvent) => {
    if (e.pointerType === 'mouse') later(hide, HIDE_MS)
  }
  // A link in the popup opens a note or a new tab: close the popup.
  const onPopupClick = (e: MouseEvent) => {
    if ((e.target as Element).closest('a')) hide()
  }

  return portal(
    <div
      ref={popRef}
      popover="auto"
      className="gloss-pop symbol-pop"
      onPointerEnter={onEnter}
      onPointerLeave={onLeave}
      onClick={onPopupClick}
    >
      {/* As in Gloss.tsx: popups of citations in an entry render in here, so opening them leaves this one open. */}
      <div ref={setInside} className="gloss-inside" />
      {shown && inside && Entries && (
        <Inside value={inside}>
          <Shown value={shown.dataset.sym}>
            <Entries components={components} />
          </Shown>
        </Inside>
      )}
    </div>,
    parent,
  )
}
