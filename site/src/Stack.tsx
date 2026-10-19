import {
  useCallback,
  useEffect,
  useEffectEvent,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from 'react'
import { isNote, keptBy, MAIN, NOTES, withCompanion, type Doc } from './content'
import { StackContext } from './context'
import { findSection, showSection } from './section'
import { ASIDE_MARGIN, columns, MOBILE_QUERY, PLATE_MARGIN, type Columns } from './layout'
import { Pane } from './Pane'
import { syncScroll } from './sync'
import { countVisit, hrefFor, setCanonical } from './url'
import { same, useStack } from './useStack'

const still = () => matchMedia('(prefers-reduced-motion: reduce)').matches
const motion = (): ScrollBehavior => (still() ? 'instant' : 'smooth')

/** The main text and the open notes, left to right. */
const docsOf = (ids: string[]): Doc[] => [MAIN, ...ids.map((id) => NOTES[id])]

/** A frame longer than this, in ms, holds up an animation visibly (Stack's arrive). */
const LONG_FRAME = 40

/** A CSS time in ms. The build minifies 380ms in styles.css to .38s. */
const ms = (time: string) => parseFloat(time) * (time.trim().endsWith('ms') ? 1 : 1000)

/** A column on screen: pane `index` of the stack `docs`, an open note or one leaving, which keeps what it had. */
type Column = {
  doc: Doc
  index: number
  docs: Doc[]
  layout: Columns
  collapsed: boolean
  leaving?: boolean
  returning?: boolean
}

/**
 * Main column plus open notes.
 * Desktop: columns side by side in a horizontal scroller. Spare width goes to the outer columns, so a
 * lone main column is centred. Columns that don't fit fold into spines via sticky positioning, as on
 * notes.andymatuschak.org.
 * Phone: the main text scrolls with the page; each note is a full-screen sheet on top.
 * A note opened slides in, and one closed slides out, as the columns move to their new places.
 */
export function Stack() {
  const { ids, seen, navigate, close } = useStack()
  const docs = useMemo(() => docsOf(ids), [ids])
  const n = docs.length
  const ref = useRef<HTMLDivElement>(null)

  const [mobile, setMobile] = useState(() => matchMedia(MOBILE_QUERY).matches)
  const [width, setWidth] = useState(() => document.documentElement.clientWidth)
  const carried = useRef<Anchor | null>(null) // reading position across a layout switch

  useEffect(() => {
    const mq = matchMedia(MOBILE_QUERY)
    const sync = () => {
      setWidth(document.documentElement.clientWidth)
      const el = ref.current
      if (el && mq.matches !== (el.dataset.mode === 'mobile')) {
        carried.current = readMain(el) // the old layout is still on screen here
        setMobile(mq.matches)
      }
    }
    mq.addEventListener('change', sync)
    addEventListener('resize', sync)
    return () => {
      mq.removeEventListener('change', sync)
      removeEventListener('resize', sync)
    }
  }, [])

  useLayoutEffect(() => {
    if (!carried.current || !ref.current) return
    writeMain(ref.current, carried.current)
    carried.current = null
  }, [mobile])

  // Column width (layout.ts).
  const layout = columns(docs, width, mobile)
  const { paneW, spineW } = layout
  const paneWRef = useRef(paneW) // for reveal() and measure(), which stay the same functions as the widths change
  const spineWRef = useRef(spineW)
  const pairRef = useRef(layout.pair)
  useLayoutEffect(() => {
    paneWRef.current = paneW
    spineWRef.current = spineW
    pairRef.current = layout.pair
  }, [paneW, spineW, layout.pair])

  const fits = n * paneW <= width
  const style = { '--width': `${width}px` } as CSSProperties

  // Which panes are folded to a strip, from where the sticky layout actually put them.
  const [collapsed, setCollapsed] = useState<boolean[]>([])
  const measure = useCallback(() => {
    const el = ref.current
    if (!el) return
    let next: boolean[] = []
    if (el.dataset.mode === 'desktop' && !el.hasAttribute('data-fits')) {
      const lefts = panesOf(el).map((p) => p.getBoundingClientRect().left)
      const end = el.getBoundingClientRect().right
      next = lefts.map((left, i) => (lefts[i + 1] ?? end) - left < 2 * spineWRef.current)
    }
    setCollapsed((prev) => (sameFlags(prev, next) ? prev : next))
  }, [])

  useEffect(() => {
    const el = ref.current!
    let frame = 0
    const schedule = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(measure)
    }
    const events = ['scroll', 'transitionend', 'animationend'] as const
    events.forEach((e) => el.addEventListener(e, schedule, { passive: true }))
    addEventListener('resize', schedule)
    return () => {
      cancelAnimationFrame(frame)
      events.forEach((e) => el.removeEventListener(e, schedule))
      removeEventListener('resize', schedule)
    }
  }, [measure])

  // A change of the stack moves the columns (`moving`: styles.css animates their widths). Notes closed, or dropped for
  // a note opened from a column before them, stay a while as they were (`leaving`): their columns narrow to nothing and
  // fade as the others move into place, so nothing jumps. They sit after the notes kept (`at`), before any opened,
  // which slide in from behind them. A note opened again on its way out widens back (`returning`). Not when the browser
  // already showed the change, or the reader wants no motion.
  const [shift, setShift] = useState({
    ids,
    leaving: [] as Column[],
    returning: [] as string[],
    at: n,
    moving: false,
    key: 0,
  })
  if (!same(shift.ids, ids)) {
    const before = docsOf(shift.ids)
    const was = columns(before, width, mobile)
    const shown = before.map(
      (doc, index): Column => ({ doc, index, docs: before, layout: was, collapsed: !!collapsed[index], leaving: true }),
    )
    shown.splice(shift.at, 0, ...shift.leaving)
    const back = [...shift.returning, ...shift.leaving.map((c) => c.doc.id)].filter((id) => ids.includes(id))
    let at = 1
    while (at < n && docs[at] === before[at]) at++
    const animate = !seen && !still()
    setShift({
      ids,
      leaving: animate ? shown.filter((c) => c.index > 0 && !ids.includes(c.doc.id)) : [],
      returning: animate ? back : [],
      at,
      moving: animate,
      key: shift.key + 1,
    })
  }
  const live = docs.map((doc, index): Column => {
    const returning = shift.returning.includes(doc.id)
    return { doc, index, docs, layout, collapsed: !!collapsed[index], returning }
  })
  const shown = [...live.slice(0, shift.at), ...shift.leaving, ...live.slice(shift.at)]

  /** Scroll sideways just enough to show pane `i` in full. */
  const reveal = useCallback((i: number, behavior: ScrollBehavior = motion()) => {
    const el = ref.current
    if (!el || el.dataset.mode !== 'desktop') return
    let left = 0
    if (!el.hasAttribute('data-fits')) {
      const w = paneWRef.current
      const min = (i + 1) * w - el.clientWidth // right edge on screen
      const max = i * (w - spineWRef.current) // left edge not under the next pane
      left = Math.max(0, Math.min(Math.max(el.scrollLeft, min), max))
    }
    el.scrollTo({ left, behavior })
  }, [])

  const flash = useCallback((i: number) => {
    const pane = ref.current && panesOf(ref.current)[i]
    if (!pane) return
    pane.classList.remove('flash')
    void pane.offsetWidth // restart the animation
    pane.classList.add('flash')
  }, [])

  // A section shown (below), kept where it is as the text settles until the reader acts: the function that stops that.
  const held = useRef(() => {})
  // A link to a section of a note (NoteLink.tsx) that opens the note: the section, shown once the note is there (below).
  const wanted = useRef<{ id: string; section: string } | null>(null)

  const open = useCallback(
    (from: number, id: string, section?: string) => {
      const at = ids.indexOf(id) + 1 // its pane, if already open
      const opened = withCompanion(id, mobile)
      const apart = opened[1] === id && ids[at - 2] !== opened[0] // a companion open without the note it goes beside
      const next = !at || apart
        ? [...keptBy(ids, from).filter((x) => !opened.includes(x)), ...opened]
        : mobile
          ? ids.slice(0, at) // back down to it
          : opened[0] === id && opened.length > 1 && at === ids.length
            ? [...ids, opened[1]] // its companion back beside it
            : ids
      if (!same(next, ids)) {
        wanted.current = section ? { id, section } : null
        navigate(next)
        return
      }
      if (!mobile) {
        reveal(at)
        flash(at)
      }
      if (section) hold(held, showIn(ref.current!, ids, id, section, !still()))
    },
    [ids, mobile, navigate, reveal, flash],
  )

  // The newest pane comes into view and gets keyboard focus (arrows / space scroll it): the columns end flush right, or
  // centred when they all fit. While they move, the scroll glides with them, on their curve (styles.css: --dur, --ease),
  // so that each column moves smoothly to its place; an animation without keyframes is its clock. Once the reader
  // scrolls sideways, the scroll is theirs. The motion's animations start with the first frame that shows them (the
  // browser would date them to the start of the frame they were made in, which can be long before), and start over
  // after a frame that comes late before they have got going: the first frames lay out a note just opened and line it
  // up with its companion, which takes a while for a long note. When it's over, the notes leaving go. Once for each
  // new stack, and on a layout switch.
  const arrive = useEffectEvent(() => {
    const el = ref.current!
    panesOf(el)[n - 1]?.querySelector<HTMLElement>('.pane-scroll')?.focus({ preventScroll: true })
    const desktop = el.dataset.mode === 'desktop'
    const left = Math.max(0, n * paneW - el.clientWidth)
    if (!shift.moving) {
      if (desktop) el.scrollLeft = left
      return
    }
    const css = getComputedStyle(el)
    const clock = el.animate([], {
      duration: ms(css.getPropertyValue('--dur')),
      easing: css.getPropertyValue('--ease'),
      fill: 'both',
    })
    const begun = Number(document.timeline.currentTime) // the motion's animations start from here on
    const from = el.scrollLeft
    let at = desktop && Math.abs(left - from) >= 1 ? from : NaN // where the glide put the scroll, NaN once it's not
    let last = NaN // when the last frame came
    let shown = 0 // how far the motion had got then
    let frame = 0
    const step = () => {
      const now = performance.now()
      if (isNaN(last) || (now - last > LONG_FRAME && shown < 0.05)) {
        for (const a of el.getAnimations({ subtree: true })) {
          if (a.startTime === null || Number(a.startTime) >= begun) a.startTime = now
        }
      }
      last = now
      shown = clock.effect?.getComputedTiming().progress ?? 1
      // Not where the glide put it (short of the end, as the columns narrow, is still): the reader scrolled.
      if (Math.abs(el.scrollLeft - Math.min(at, el.scrollWidth - el.clientWidth)) > 1) at = NaN
      if (!isNaN(at)) el.scrollLeft = at = from + (left - from) * shown
      frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    clock.finished.then(
      () => {
        cancelAnimationFrame(frame)
        clock.cancel()
        if (!isNaN(at)) el.scrollLeft = left
        setShift((s) => ({ ...s, leaving: [], returning: [], moving: false }))
      },
      () => {}, // cancelled by the next change
    )
    return () => {
      cancelAnimationFrame(frame)
      clock.cancel()
    }
  })
  useLayoutEffect(() => arrive(), [shift.key, mobile])

  useLayoutEffect(measure, [measure, ids, mobile, width])

  // Resizing the window resizes a shared pair: keep the last column in full view. (A new stack is revealed above.)
  const revealedN = useRef(n)
  useLayoutEffect(() => {
    const resized = revealedN.current === n
    revealedN.current = n
    if (resized && pairRef.current) reveal(n - 1, 'instant')
  }, [width, n, reveal])

  // A note and its companion open to its right scroll together (sync.ts), unless the reader turns that off with the
  // switch at the top of the companion (Pane.tsx); the choice is remembered. Turned off, both stay where they are;
  // turned on, the companion moves to the note. Keyed by column and note, whose elements stay. A pair on its way out
  // stays together until it's gone: taking the space out lays both notes out again, which would hold up the columns
  // while they move.
  const pairs = shown
    .flatMap(({ doc, index }, k) =>
      index > 0 && doc.companion && shown[k + 1]?.doc.id === doc.companion ? [`${k}:${doc.id}`] : [],
    )
    .join(' ')
  const [together, setTogether] = useState(readTogether)
  const toggleTogether = () => {
    writeTogether(!together)
    setTogether(!together)
  }
  useEffect(() => {
    if (!pairs || !together) return
    const scrollers = ref.current!.querySelectorAll<HTMLElement>('.pane-scroll')
    const stops = pairs.split(' ').map((pair) => {
      const k = Number(pair.split(':')[0])
      return syncScroll(scrollers[k], scrollers[k + 1])
    })
    return () => stops.forEach((stop) => stop())
  }, [pairs, together])

  // Once the note has slid in at its top, it scrolls down to the section, so the reader sees where it is in the note.
  // After the effect above, so that a pair opened by a link to a section of one of them follows that one (sync.ts).
  useEffect(() => {
    const want = wanted.current
    if (!want || shift.moving) return
    wanted.current = null
    hold(held, showIn(ref.current!, ids, want.id, want.section, !still()))
  })

  // A link to a section (#address, Heading.tsx) shows the last column that has the heading, scrolled to it: as the page
  // opens, and when only the address's # changes (a link pasted over the same notes). Back never brings one back
  // (useStack.ts).
  useEffect(() => {
    let stop = () => {}
    const show = () => {
      const section = decodeURIComponent(location.hash.slice(1))
      const panes = section ? panesOf(ref.current!) : []
      const i = panes.findLastIndex((pane) => findSection(pane, section))
      if (i < 0) return
      reveal(i, 'instant')
      stop = showSection(findSection(panes[i], section)!)
      hold(held, stop)
    }
    show()
    addEventListener('hashchange', show)
    return () => {
      removeEventListener('hashchange', show)
      stop()
    }
  }, [reveal])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // An open popup takes Esc for itself (the browser closes it); the next Esc closes the last note.
      if (e.key === 'Escape' && ids.length && !document.querySelector(':popover-open')) close(ids.length)
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [ids, close])

  useEffect(() => {
    document.title = n > 1 ? `${docs[n - 1].title} · ${MAIN.title}` : MAIN.title
  }, [docs, n])

  useEffect(() => setCanonical(ids), [ids])

  // Each note opened counts as a visit to its page; the page the reader arrived on counts itself (index.html).
  const counted = useRef(ids)
  useEffect(() => {
    for (const doc of docs.slice(1))
      if (!counted.current.includes(doc.id)) countVisit(doc.id, `${doc.title} · ${MAIN.title}`)
    counted.current = ids
  }, [docs, ids])

  const api = useMemo(() => ({ ids, mobile, open }), [ids, mobile, open])

  return (
    <StackContext value={api}>
      <div
        ref={ref}
        className="stack"
        data-mode={mobile ? 'mobile' : 'desktop'}
        data-fits={fits || undefined}
        data-animate={shift.moving || undefined}
        style={style}
      >
        <div className="track">
          {shown.map(({ doc, index: i, ...c }) => (
            <Pane
              key={i ? `note:${doc.id}` : 'main'}
              index={i}
              doc={doc}
              layout={c.layout}
              leaving={!!c.leaving}
              returning={!!c.returning}
              back={i ? c.docs[i - 1].title : undefined}
              collapsed={c.collapsed}
              inert={!!c.leaving || (mobile && i < n - 1)}
              plates={!mobile && (width - (i + 1) * c.layout.paneW) / 2 >= PLATE_MARGIN}
              aside={!mobile && i === c.layout.n - 1 && c.layout.extra >= ASIDE_MARGIN}
              companion={
                i && doc.companion && isNote(doc.companion)
                  ? { label: shortTitle(NOTES[doc.companion].title, doc.title), open: () => open(i, doc.companion!) }
                  : undefined
              }
              together={
                i > 1 && c.docs[i - 1].companion === doc.id ? { on: together, toggle: toggleTogether } : undefined
              }
              onClose={() => close(i)}
              onReveal={() => reveal(i)}
            />
          ))}
        </div>
      </div>
    </StackContext>
  )
}

/** Whether a note and its companion scroll together (on unless the reader turned it off), in this browser. */
const TOGETHER_KEY = 'scroll-together'

function readTogether() {
  try {
    return localStorage.getItem(TOGETHER_KEY) !== 'off'
  } catch {
    return true
  }
}

function writeTogether(on: boolean) {
  try {
    if (on) localStorage.removeItem(TOGETHER_KEY)
    else localStorage.setItem(TOGETHER_KEY, 'off')
  } catch {
    // storage blocked: the choice lasts until the page reloads
  }
}

/** The panes of the open notes, not of those leaving. */
const panesOf = (el: HTMLElement) => Array.from(el.querySelectorAll<HTMLElement>('.pane:not([data-leaving])'))

/** Makes `stop` the function that stops the section shown (`held`), after stopping the one before. */
function hold(held: { current: () => void }, stop?: () => void) {
  if (!stop) return
  held.current()
  held.current = stop
}

/** Shows the heading with address `section` of open note `id` in stack `el`, as a link to the section does (Stack's
 *  effect for #address), scrolling there smoothly with `smooth`, and puts the address in the address bar, as a click on
 *  the heading does (Heading.tsx). Returns the function that stops keeping it in place (section.ts), or nothing when
 *  the note has no such heading. */
function showIn(el: HTMLElement, ids: string[], id: string, section: string, smooth: boolean) {
  const at = ids.indexOf(id) + 1
  const heading = at ? findSection(panesOf(el)[at], section) : null
  if (!heading) return
  history.replaceState(history.state, '', `${hrefFor(ids)}#${section}`)
  return showSection(heading, smooth)
}

const sameFlags = (a: boolean[], b: boolean[]) => a.length === b.length && a.every((x, i) => x === b[i])

/** A title without the words it starts with in common with another: "AI Pause Model Explanations" beside
 *  "AI Pause Model" is "Explanations". */
function shortTitle(title: string, beside: string) {
  const words = title.split(' ')
  const other = beside.split(' ')
  let k = 0
  while (k < words.length - 1 && words[k] === other[k]) k++
  return words.slice(k).join(' ')
}

/** A block of the main text and its distance from the top of the screen. */
type Anchor = { el: Element; offset: number }

/** The first block of the main text still on screen. Both scrollers start at the top of the screen. */
function readMain(stack: HTMLElement): Anchor | null {
  const blocks = Array.from(stack.querySelectorAll('.pane:first-child .pane-body > *'))
  const el = blocks.find((b) => b.getBoundingClientRect().bottom > 0)
  return el ? { el, offset: el.getBoundingClientRect().top } : null
}

/** Scroll the main text so the anchor block is back where it was. The main column scrolls with
 *  the page on a phone and inside its pane on desktop. */
function writeMain(stack: HTMLElement, { el, offset }: Anchor) {
  const scroller =
    stack.dataset.mode === 'mobile' ? document.scrollingElement! : stack.querySelector('.pane-scroll')!
  scroller.scrollTop += el.getBoundingClientRect().top - offset
}
