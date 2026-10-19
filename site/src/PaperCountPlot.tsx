import {
  Children,
  isValidElement,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent,
  type ReactNode,
} from 'react'
import table from '../content/table.json'
import { count, month } from './plot'
import { useWidth } from './useWidth'

/** A pause's papers a year n at the dates t. */
type Series = { name: string; t: number[]; n: number[] }
/** Under the pointer: a year of the dataset, or a date in the pauses. */
type Hover = { year: number } | { t: number }

// ---------- the paper count n(t) (29) of content/table.json: our dataset's, then the lab's in a pause from each row ----------

const { n_s: nS, dataset, rows } = table.papers
const pauses: Series[] = rows.map((r) => ({ name: r.name, t: r.t, n: r['n / n_s'].map((v) => v * nS) }))
const last = dataset[dataset.length - 1]

// Years on x, from the dataset's first to the end of the last pause; papers a year on a log scale on y, whole decades
// around every count. Ticks every TICK_YEARS years and every decade.
const T0 = dataset[0].year
const T1 = Math.ceil(Math.max(last.year, ...pauses.map((p) => p.t[p.t.length - 1])))
const counts = [...dataset.map((d) => d.n), ...pauses.flatMap((p) => p.n)].filter((v) => v > 0)
const LO = Math.floor(Math.log10(Math.min(...counts)))
const HI = Math.ceil(Math.log10(Math.max(...counts)))
const TICK_YEARS = 5
const YEARS = Array.from({ length: Math.floor(T1 / TICK_YEARS) - Math.ceil(T0 / TICK_YEARS) + 1 }, (_, i) =>
  (Math.ceil(T0 / TICK_YEARS) + i) * TICK_YEARS,
)
const DECADES = Array.from({ length: HI - LO + 1 }, (_, i) => LO + i)
const GAP = 12 // px: the least distance between the pauses' labels

/** The pause's papers a year at t, log-interpolated between its months; undefined outside the pause. */
function at(p: Series, t: number) {
  const k = p.t.length - 1
  if (t < p.t[0] || t > p.t[k]) return undefined
  const i = Math.min(k - 1, Math.floor(((t - p.t[0]) / (p.t[k] - p.t[0])) * k))
  const f = (t - p.t[i]) / (p.t[i + 1] - p.t[i])
  return Math.exp((1 - f) * Math.log(p.n[i]) + f * Math.log(p.n[i + 1]))
}

/**
 * `<PaperCountPlot>` in an .mdx file: papers a year, our AI safety dataset's to 2025 and then the lab's paper count n
 * (29) in a pause from each row on, with the inputs of content/table.json (`python pause.py --json` writes them: the
 * trust model and staffing it names, n_s the dataset's 2025 count). Monochrome, on a log scale. Hovering or tapping
 * shows the counts at that date. The children are labels, by data attribute, then the caption:
 *
 *   <span data-axis>papers a year $n$</span>    the y axis's title
 *   <span data-dataset>our dataset</span>    the dataset's label, by its last point
 *
 * In .mdx files it needs no import (Pane.tsx provides it).
 */
export function PaperCountPlot({ children }: { children?: ReactNode }) {
  let axis: ReactNode
  let named: ReactNode
  const caption: ReactNode[] = []
  type Label = { 'data-axis'?: unknown; 'data-dataset'?: unknown; children?: ReactNode }
  Children.forEach(children, (child) => {
    if (isValidElement<Label>(child)) {
      if (child.props['data-axis'] !== undefined) return void (axis = child.props.children)
      if (child.props['data-dataset'] !== undefined) return void (named = child.props.children)
    }
    caption.push(child)
  })

  const [box, width] = useWidth()
  const [hover, setHover] = useState<Hover>()
  const m = { l: 36, r: 50, t: 8, b: 24 }
  const w = Math.max(0, width - m.l - m.r)
  const h = Math.round(Math.min(320, Math.max(200, w * 0.6)))
  const X = (t: number) => m.l + ((t - T0) / (T1 - T0)) * w
  const Y = (n: number) => m.t + h - ((Math.log10(n) - LO) / (HI - LO)) * h
  const line = (pts: [number, number][]) =>
    pts.map(([t, n], i) => `${i ? 'L' : 'M'}${X(t).toFixed(1)},${Y(n).toFixed(1)}`).join('')

  // Each pause's name at its end, moved apart where they would touch.
  const ends = pauses
    .map((p) => ({ name: p.name, x: X(p.t[p.t.length - 1]) + 6, y: Y(p.n[p.n.length - 1]) }))
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + GAP)

  // The dataset's year nearest the pointer up to half a year past its last; later, the date, if a pause runs then.
  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const t = T0 + ((e.clientX - e.currentTarget.getBoundingClientRect().left - m.l) / w) * (T1 - T0)
    if (t <= last.year + 0.5) return setHover({ year: Math.max(T0, Math.round(t)) })
    const s = Math.min(t, T1)
    setHover(pauses.some((p) => at(p, s) !== undefined) ? { t: s } : undefined)
  }
  const point = hover && 'year' in hover ? dataset.find((d) => d.year === hover.year) : undefined
  const date = hover && 't' in hover ? hover.t : undefined
  const lit: { name: string; n: number }[] = []
  for (const p of pauses) {
    const n = date === undefined ? undefined : at(p, date)
    if (n !== undefined) lit.push({ name: p.name, n })
  }
  const cross = point ? X(point.year) : date !== undefined ? X(date) : undefined

  const span = pauses.length ? ` and the labs' in a pause from ${pauses.map((p) => p.name).join(', ')} on` : ''
  return (
    <figure className="paper-plot">
      {axis && <div className="paper-plot-axis">{axis}</div>}
      <div className="paper-plot-box" ref={box}>
        {w > 0 && (
          <svg
            width={width}
            height={m.t + h + m.b}
            role="img"
            aria-label={`Papers a year on a log scale: our dataset's ${T0}–${last.year}${span}`}
            onPointerMove={(e) => e.pointerType !== 'touch' && pick(e)}
            onPointerDown={(e) => e.pointerType === 'touch' && pick(e)}
            onPointerLeave={(e) => e.pointerType !== 'touch' && setHover(undefined)}
          >
            {DECADES.map((e) => (
              <g key={e}>
                <line className="pp-grid" x1={m.l} x2={m.l + w} y1={Y(10 ** e)} y2={Y(10 ** e)} />
                <text className="pp-tick" x={m.l - 6} y={Y(10 ** e) + 3.5} textAnchor="end">
                  {tick(e)}
                </text>
              </g>
            ))}
            {YEARS.map((y) => (
              <text key={y} className="pp-tick" x={X(y)} y={m.t + h + 16} textAnchor="middle">
                {y}
              </text>
            ))}
            {cross !== undefined && <line className="pp-cross" x1={cross} x2={cross} y1={m.t} y2={m.t + h} />}
            <path className="pp-data" d={line(dataset.map((d) => [d.year, d.n]))} />
            {pauses.map((p) => (
              <path key={p.name} className="pp-line" d={line(p.t.map((t, i) => [t, p.n[i]]))} />
            ))}
            {dataset.map((d) => (
              <circle key={d.year} className="pp-dot" cx={X(d.year)} cy={Y(d.n)} r={d === point ? 5.5 : 4} />
            ))}
            {date !== undefined &&
              lit.map((l) => <circle key={l.name} className="pp-dot" cx={X(date)} cy={Y(l.n)} r={4} />)}
            {ends.map((l) => (
              <text key={l.name} className="pp-label" x={l.x} y={l.y + 3.5}>
                {l.name}
              </text>
            ))}
          </svg>
        )}
        {w > 0 && named && (
          <Named right={X(last.year) + 2} left={m.l} bottom={Y(Math.max(...dataset.map((d) => d.n))) - 8}>
            {named}
          </Named>
        )}
        {cross !== undefined && (point || lit.length > 0) && (
          <Tip x={cross} top={m.t} width={width}>
            <div className="paper-plot-tip-date">{point ? point.year : month(date!)}</div>
            <table>
              <tbody>
                {point ? (
                  <tr>
                    <th scope="row">{named ?? 'Dataset'}</th>
                    <td>{point.n.toLocaleString('en-US')}</td>
                    <td>×{ratio(point.n / nS)}</td>
                  </tr>
                ) : (
                  lit.map((l) => (
                    <tr key={l.name}>
                      <th scope="row">{l.name}</th>
                      <td>{count(l.n)}</td>
                      <td>×{ratio(l.n / nS)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </Tip>
        )}
      </div>
      {caption.length > 0 && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

/** The dataset's label, its bottom at `bottom` and its right side at `right`, moved right where it would cross `left`:
 *  above every dataset point, inside the plot. */
function Named({ right, left, bottom, children }: { right: number; left: number; bottom: number; children: ReactNode }) {
  const label = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = label.current
    if (!el) return
    el.style.left = `${Math.max(left, right - el.offsetWidth)}px`
    el.style.top = `${bottom - el.offsetHeight}px`
    el.style.visibility = 'visible'
  })
  return (
    <div className="paper-plot-named" ref={label} style={{ visibility: 'hidden' }}>
      {children}
    </div>
  )
}

/** The counts' card beside the crosshair at x, flipped to its left where it would leave the plot. HiringPlot.tsx uses
 *  it too. */
export function Tip({ x, top, width, children }: { x: number; top: number; width: number; children: ReactNode }) {
  const tip = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const t = tip.current
    if (!t) return
    const gap = 12
    t.style.left = `${x + gap + t.offsetWidth <= width ? x + gap : Math.max(0, x - gap - t.offsetWidth)}px`
    t.style.visibility = 'visible'
  })
  return (
    <div className="paper-plot-tip" ref={tip} style={{ top, visibility: 'hidden' }}>
      {children}
    </div>
  )
}

/** A tick at 10^e papers: 1k, 10k, 1M. */
const tick = (e: number) => (e < 3 ? String(10 ** e) : e < 6 ? `${10 ** (e - 3)}k` : `${10 ** (e - 6)}M`)

const ratio = (r: number) => (r >= 100 ? count(r) : r.toPrecision(3))
