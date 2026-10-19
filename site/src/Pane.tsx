import { memo, useRef, type ComponentProps, type CSSProperties, type ReactNode } from 'react'
import { AuditorCurve } from './AuditorCurve'
import { Authors } from './Authors'
import { Backdrop } from './Backdrop'
import { BlindPlot } from './BlindPlot'
import { Cite, References } from './Cite'
import type { Doc } from './content'
import { PaneContext } from './context'
import { Derivation } from './Derivation'
import { Doodle } from './Doodle'
import { Flow } from './Flow'
import { Footnote } from './Footnote'
import { Heading } from './Heading'
import { HiringCurve } from './HiringCurve'
import { HiringPlot } from './HiringPlot'
import { Incident, Said } from './Incident'
import type { Columns } from './layout'
import { Mark } from './Mark'
import { MdxLink } from './NoteLink'
import { Night } from './Night'
import { PaperCountPlot } from './PaperCountPlot'
import { Parameter } from './Parameter'
import { PilePlot } from './PilePlot'
import { Plate } from './Plate'
import { RandomChecks } from './RandomChecks'
import { Milestone, ResultTable } from './ResultTable'
import { SlowScaling } from './SlowScaling'
import { Summary } from './Summary'
import { Term } from './Term'
import { Tornado } from './Tornado'
import { Range, SymbolPopup, Value } from './Symbols'
import { AifmOnly, NotInModel } from './Unused'

/** Headings, each a link to its section. */
const headings = Object.fromEntries(
  [1, 2, 3, 4, 5, 6].map((level) => [`h${level}`, (props: ComponentProps<'h1'>) => <Heading level={level} {...props} />]),
)

// Every .mdx file can use these without importing them. An import in an .mdx file of a module that imports
// content.ts would make an import cycle, and editing that .mdx file would then fail to hot-reload.
const components = {
  ...headings,
  a: MdxLink,
  AifmOnly,
  AuditorCurve,
  Authors,
  Backdrop,
  BlindPlot,
  Cite,
  Derivation,
  Doodle,
  Flow,
  Footnote: NoteFootnote,
  HiringCurve,
  HiringPlot,
  Incident,
  Mark,
  Milestone,
  Night,
  NotInModel,
  PaperCountPlot,
  Parameter,
  PilePlot,
  Plate,
  Range,
  RandomChecks,
  References,
  ResultTable,
  Said,
  SlowScaling,
  Summary,
  Term,
  Tornado,
  Value,
}

/** A footnote of the note: its text is the note's, so symbols.ts marks its symbols too, and it gets their popup. */
function NoteFootnote({ children }: { children: ReactNode }) {
  return (
    <Footnote>
      {children}
      <SymbolPopup />
    </Footnote>
  )
}

type Props = {
  index: number
  doc: Doc
  /** The columns it is one of (layout.ts), for styles.css. */
  layout: Columns
  /** Closed: it narrows to nothing and fades as the other columns move into place (Stack.tsx), keeping the columns,
   *  index and look it had. */
  leaving: boolean
  /** Opened again while leaving: it widens back. */
  returning: boolean
  /** Title of the pane to the left, shown on the phone's back button. Absent for the main column. */
  back?: string
  collapsed: boolean
  inert: boolean
  /** Show the column's drawings (Plate.tsx): its side margin, when it opened as the last column, is wide enough. */
  plates: boolean
  /** Its right margin holds the result table's menu (ResultTable.tsx): it is the last column, with room beside it. */
  aside: boolean
  /** The note's companion (content.ts): the label and action of the phone bar's button that opens it. */
  companion?: { label: string; open: () => void }
  /** The companion of the note before it: the switch that makes the two scroll together (Stack.tsx) or not. */
  together?: { on: boolean; toggle: () => void }
  onClose: () => void
  onReveal: () => void
}

export function Pane({
  index,
  doc,
  layout,
  leaving,
  returning,
  back,
  collapsed,
  inert,
  plates,
  aside,
  companion,
  together,
  onClose,
  onReveal,
}: Props) {
  const { n, paneW, spineW, extra, pair } = layout
  const style = {
    '--i': index,
    '--n': n,
    '--pane-w': `${paneW}px`,
    '--spine-w': `${spineW}px`,
    '--extra': `${extra}px`,
  } as CSSProperties
  return (
    <section
      className="pane"
      style={style}
      data-last={index === n - 1 || undefined}
      data-behind={(pair && index < n - 2) || undefined}
      data-leaving={leaving || undefined}
      data-returning={returning || undefined}
      data-collapsed={collapsed || undefined}
      data-plates={plates || undefined}
      data-aside={aside || undefined}
      inert={inert}
      aria-label={doc.title}
    >
      <div className="pane-scroll" tabIndex={-1}>
        {back !== undefined && (
          <div className="pane-bar">
            <button className="pane-back" onClick={onClose} aria-label={`Back to ${back}`}>
              <span className="pane-back-arrow" aria-hidden="true">‹</span>
              <span className="pane-back-label">{back}</span>
            </button>
            {/* Phone only: on desktop the companion opens beside the note. */}
            {companion && (
              <button className="pane-companion" onClick={companion.open}>
                <span className="pane-companion-label">{companion.label}</span>
                <span className="pane-companion-arrow" aria-hidden="true">›</span>
              </button>
            )}
            {together && (
              <button className="pane-together" role="switch" aria-checked={together.on} onClick={together.toggle}>
                <span className="pane-together-label">Scroll together</span>
                <span className="pane-together-track" aria-hidden="true" />
              </button>
            )}
            <button className="pane-close" onClick={onClose} aria-label={`Close ${doc.title}`}>
              <svg className="pane-close-icon" viewBox="0 0 16 16" aria-hidden="true">
                <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />
              </svg>
            </button>
          </div>
        )}
        <Body doc={doc} index={index} />
      </div>
      {/* Visible only while the pane is folded to a strip; click to bring it back. */}
      <button
        className="pane-spine"
        onClick={onReveal}
        tabIndex={collapsed ? 0 : -1}
        aria-hidden={!collapsed}
        aria-label={`Show ${doc.title}`}
      >
        <span>{doc.title}</span>
      </button>
    </section>
  )
}

/** The note's text. The stack renders its panes again whenever it changes or scrolls, and a long note takes a while to
 *  render, so the text renders again only when its pane moves along the stack. */
export const Body = memo(function Body({ doc, index }: { doc: Doc; index: number }) {
  const { Content } = doc
  const body = useRef<HTMLElement>(null)
  return (
    <article className="pane-body" ref={body}>
      <h1>{doc.title}</h1>
      <PaneContext value={index}>
        <Content components={components} />
        {/* The popup of the symbols in its formulas (symbols.ts); it renders in <body>. */}
        <SymbolPopup scope={body} />
      </PaneContext>
    </article>
  )
})
