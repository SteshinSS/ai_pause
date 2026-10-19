import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import table from '../content/table.json'
import { MARKS, MILESTONES } from './content'
import { fmt, people } from './format'
import { Gloss } from './Gloss'
import { setHiring, useHiring } from './hiring'
import { MdxLink } from './NoteLink'
import { ProtocolMenu } from './ProtocolMenu'

const components = { a: MdxLink }
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
/** '2028-05' -> 'May 2028' */
function monthYear(month: string) {
  const [y, m] = month.split('-').map(Number)
  return `${MONTHS[m - 1]} ${y}`
}

/** A pause length in years, as pause.py writes it: '9 months', '1.5 years'. */
const duration = (years: number) => (years < 1 ? `${years * 12} months` : `${years} years`)

type Row = (typeof table.rows)[number]
type Model = keyof Row['cells']
type Hiring = keyof Row['hired']
const hires = (model: Model): model is Hiring => model in table.rows[0].hired
/** The trust models in the menu, in table.json's order (the notes'). */
const MODELS = Object.keys(table.rows[0].cells) as Model[]

/** The rows in table.json's order, by name. */
const ROWS = table.rows.map((row) => row.name)
const TRUST_FROM: Record<string, string> = table.trust_from

/** Whether row `i` of trust model `model`'s table is a simpler model's: it comes before the row the model starts at
 *  (table.json's trust_from), as before AC there is nothing to verify and before SAR nothing to compartmentalize. */
const fallback = (model: Model, i: number) => model in TRUST_FROM && i < ROWS.indexOf(TRUST_FROM[model])

/** Where the tables with the switch were on screen when the reader flipped it, until they have changed. */
const were = new Map<HTMLElement, DOMRect>()

/** Flips the switch of every table on the page (hiring.ts), noting first where the tables that change are. */
function flip(value: boolean) {
  for (const el of document.querySelectorAll<HTMLElement>('.result-table[data-switch]')) {
    were.set(el, el.getBoundingClientRect())
  }
  setHiring(value)
}

/**
 * Scrolls the column of table `el`, which the switch grew or shrank, so that what the reader sees there stays, as
 * Derivation.tsx does for its twins. A table changes below its top, so only one above the view moves the text on
 * screen: its column scrolls by as much. In a browser with scroll anchoring that is already done, and this adds nothing.
 */
function keep(el: HTMLElement, was: DOMRect) {
  const pane = el.closest<HTMLElement>('.pane-scroll')
  if (!pane) return
  const own = getComputedStyle(pane).overflowY !== 'visible' // a phone's main text scrolls with the page
  const view = own ? pane.getBoundingClientRect().top : 0
  if (was.bottom > view) return
  const scroller = own ? pane : document.scrollingElement!
  scroller.scrollTop += el.getBoundingClientRect().bottom - was.bottom
}

/**
 * A result table from content/table.json, which `python pause.py --json site/content/table.json` writes with a table
 * per trust model: `model` is 'full trust', 'verified', 'compartmentalized' or 'approved', by default the main one
 * (table.main). With `menu` a menu picks the trust model shown, `model` first (ProtocolMenu.tsx): in the right
 * margin beside the table's head when the column has room there (Pane.tsx's data-aside), else above the table. The
 * verified, compartmentalized and approved tables show the lab with only the staff it had at the start of the pause.
 * With
 * `toggle` they get a switch to unlimited hiring, where the lab hires the people it lacks, and a second table shows
 * how many; the switch flips every table on the page (hiring.ts) and keeps the text on screen in place (keep()).
 * A row's name opens its popup, content/milestones/<name in lower case>.mdx. The children are the caption. In .mdx
 * files it is `<ResultTable model="verified" toggle>caption</ResultTable>`, with no import (Pane.tsx provides it).
 */
export function ResultTable({
  model = table.main as Model,
  menu = false,
  toggle = false,
  children,
}: {
  model?: Model
  menu?: boolean
  toggle?: boolean
  children?: ReactNode
}) {
  const [shown, setShown] = useState(model)
  const hiring = useHiring()
  const switchable = toggle && hires(shown) ? shown : undefined
  const unlimited = hiring ? switchable : undefined
  const ref = useRef<HTMLElement>(null)
  useLayoutEffect(() => {
    const el = ref.current!
    const was = were.get(el)
    were.delete(el)
    if (was) keep(el, was)
  }, [unlimited])
  // The menu moves between the right margin and above the table as notes open and close beside the column or the
  // window is resized (Pane.tsx's data-aside), which changes the table's height: keep() holds the text on screen in
  // place, from where the table was on screen, noted at each scroll of its column or of the page (a phone's main text).
  // Browsers' own scroll anchoring doesn't, as the columns move. When the table's width or the layout (the stack's
  // data-mode) changed too, the text was laid out anew, and it only notes.
  useLayoutEffect(() => {
    if (!menu) return
    const el = ref.current!
    const pane = el.closest('.pane-scroll')
    const stack = el.closest<HTMLElement>('.stack')
    let was = el.getBoundingClientRect()
    let mode = stack?.dataset.mode
    const note = () => {
      was = el.getBoundingClientRect()
      mode = stack?.dataset.mode
    }
    const resized = new ResizeObserver(() => {
      if (el.getBoundingClientRect().width === was.width && stack?.dataset.mode === mode) keep(el, was)
      note()
    })
    resized.observe(el)
    pane?.addEventListener('scroll', note, { passive: true })
    addEventListener('scroll', note, { passive: true })
    return () => {
      resized.disconnect()
      pane?.removeEventListener('scroll', note)
      removeEventListener('scroll', note)
    }
  }, [menu])
  return (
    <figure ref={ref} className="result-table" data-switch={!!switchable || undefined}>
      {menu && <ProtocolMenu models={MODELS} value={shown} onChange={setShown} className="result-table-pick" />}
      {switchable && (
        <div className="result-table-bar">
          <button
            className="result-table-hiring"
            role="switch"
            aria-checked={hiring}
            onClick={() => flip(!hiring)}
          >
            <span className="result-table-hiring-label">Unlimited hiring</span>
            <span className="result-table-hiring-track" aria-hidden="true" />
          </button>
        </div>
      )}
      <Grid
        model={shown}
        values={(row) => (unlimited ? row.unlimited[unlimited] : row.cells[shown])}
        format={fmt}
        marked
      />
      {unlimited && (
        <>
          <div className="result-table-title">People to hire</div>
          <Grid model={shown} values={(row) => row.hired[unlimited]} format={people} />
          <div className="result-table-note">
            On top of the labs' own staff: the most they need at any time in the pause.
          </div>
        </>
      )}
      {children && <figcaption>{children}</figcaption>}
    </figure>
  )
}

/**
 * The table itself: a row per pause start, a column per duration. The rows where `model` falls back to a simpler model
 * are gray (fallback()). With `marked`, the cells content/marks/ names get a † with its popup (content.ts, MARKS).
 */
export function Grid({
  model,
  values,
  format,
  marked = false,
}: {
  model: Model
  values: (row: Row) => number[]
  format: (v: number) => string
  marked?: boolean
}) {
  return (
    <div className="result-table-scroll">
      <table>
        <thead>
          <tr>
            <th scope="col" rowSpan={2}>
              AI capability at the start
            </th>
            <th scope="colgroup" colSpan={table.durations_years.length} className="result-table-group">
              Pause duration
            </th>
          </tr>
          <tr>
            {table.durations_years.map((d) => (
              <th scope="col" key={d}>
                {duration(d)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, r) => (
            <tr key={row.name} className={fallback(model, r) ? 'result-table-fallback' : undefined}>
              <th scope="row">
                <Milestone>{row.name}</Milestone>
                <span className="result-table-date">{monthYear(row.month)}</span>
              </th>
              {values(row).map((v, i) => {
                const years = table.durations_years[i]
                const mark =
                  marked && MARKS.find((m) => m.model === model && m.row === row.name && m.years.includes(years))
                return (
                  <td key={i}>
                    {format(v)}
                    {mark && (
                      <Gloss term="†" title={mark.title} label={mark.title} className="result-table-mark">
                        <mark.Content components={components} />
                      </Gloss>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/**
 * A milestone's name with its popup, content/milestones/<name in lower case>.mdx, as in the table's rows; just the
 * name when there is none or with `plain`. In .mdx files it is `<Milestone>SAR</Milestone>`, with no import (Pane.tsx
 * provides it); milestones.ts adds `plain` to the uses that get no popup.
 */
export function Milestone({ children: name, plain }: { children: string; plain?: boolean }) {
  const popup = MILESTONES[name.toLowerCase()]
  if (!popup || plain) return name
  return (
    <Gloss term={name} title={popup.title}>
      <popup.Content components={components} />
    </Gloss>
  )
}
