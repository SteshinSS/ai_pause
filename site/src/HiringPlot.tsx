import { Children, isValidElement, useState, type PointerEvent, type ReactNode } from 'react'
import table from '../content/table.json'
import { Tip } from './PaperCountPlot'
import { axis, count, month, people, steps } from './plot'
import { useWidth } from './useWidth'

type Model = 'verified' | 'compartmentalized' | 'approved'
/** A pause's people hired N_h and the lab's own staff L_H at the dates t. */
type Series = { name: string; t: number[]; hired: number[]; staff: number[] }

// ---------- the people the lab hires with unlimited hiring (36), (49) of content/table.json, in a pause from each row on ----------

// Years on x, from the plot's first pause's start to the end of its last, a tick every TICK_YEARS years. People on a
// linear scale on y, as the lines start at 0: from 0 to a round number above the plot's most (plot.ts).
const { rows } = table.hiring
const TICK_YEARS = 2
const GAP = 12 // px: the least distance between the pauses' labels
const FLOOR = 5 // px: a label at 0 sits this far above the axis, off its line
const CHAR = 6.6 // px: a label's character, 10px IBM Plex Mono with its letter spacing (styles.css .pp-label)
const CAP = 6 // px: a line this close above or below a label's middle runs through it

/** The pause's people hired and staff at t, linear between its months; undefined outside the pause. */
function at(p: Series, t: number) {
  const k = p.t.length - 1
  if (t < p.t[0] || t > p.t[k]) return undefined
  const i = Math.min(k - 1, Math.floor(((t - p.t[0]) / (p.t[k] - p.t[0])) * k))
  const f = (t - p.t[i]) / (p.t[i + 1] - p.t[i])
  const mix = (v: number[]) => (1 - f) * v[i] + f * v[i + 1]
  return { hired: mix(p.hired), staff: mix(p.staff) }
}

/**
 * `<HiringPlot model="verified">` in an .mdx file: the people the lab hires on top of its own staff with unlimited
 * hiring, by date, in a pause from each row on where the trust model applies, with `model` 'verified',
 * 'compartmentalized' or 'approved'. The data is content/table.json's `hiring` (`python pause.py --json` writes it: a point a month,
 * the most the lab has needed so far, as the result tables' "People to hire"). Monochrome, as PaperCountPlot.tsx;
 * hovering or tapping shows the people hired at that date, and per person of the lab's own staff. The children are the
 * y axis's title, `<span data-axis>people to hire $N_h$</span>`, then the caption. In .mdx files it needs no import
 * (Pane.tsx provides it).
 */
export function HiringPlot({ model, children }: { model: Model; children?: ReactNode }) {
  let title: ReactNode
  const caption: ReactNode[] = []
  type Label = { 'data-axis'?: unknown; children?: ReactNode }
  Children.forEach(children, (child) => {
    if (isValidElement<Label>(child) && child.props['data-axis'] !== undefined) return void (title = child.props.children)
    caption.push(child)
  })

  // the rows where the trust model applies: table.json has its hires only there
  const pauses: Series[] = rows.flatMap((r) => {
    const hired = (r.hired as Partial<Record<Model, number[]>>)[model]
    return hired ? [{ name: r.name, t: r.t, hired, staff: r.L_H }] : []
  })
  const T0 = Math.floor(Math.min(...pauses.map((p) => p.t[0])))
  const T1 = Math.ceil(Math.max(...pauses.map((p) => p.t[p.t.length - 1])))
  const years = Array.from({ length: Math.floor(T1 / TICK_YEARS) - Math.ceil(T0 / TICK_YEARS) + 1 }, (_, i) =>
    (Math.ceil(T0 / TICK_YEARS) + i) * TICK_YEARS,
  )
  const y = axis(Math.max(...pauses.flatMap((p) => p.hired)))
  const [box, width] = useWidth()
  const [date, setDate] = useState<number>()
  const m = { l: 36, r: 50, t: 8, b: 24 }
  const w = Math.max(0, width - m.l - m.r)
  const h = Math.round(Math.min(320, Math.max(200, w * 0.6)))
  const X = (t: number) => m.l + ((t - T0) / (T1 - T0)) * w
  const Y = (v: number) => m.t + h - (v / y.top) * h
  const line = (p: Series) => p.t.map((t, i) => `${i ? 'L' : 'M'}${X(t).toFixed(1)},${Y(p.hired[i]).toFixed(1)}`).join('')

  // Each pause's name at its end, moved apart where they would touch: down from the top, then up from just above the
  // axis, so that a line ending at 0 keeps its name off the axis. A name another pause's line runs through goes on the
  // left of its line's end instead.
  const end = (p: Series, dx: number) => ({ name: p.name, x: X(p.t[p.t.length - 1]) + dx, y: Y(p.hired[p.hired.length - 1]) })
  const spread = (ps: Series[]) => {
    const ends = ps.map((p) => end(p, 6)).sort((a, b) => a.y - b.y)
    for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + GAP)
    for (let i = ends.length - 1; i >= 0; i--) {
      ends[i].y = Math.min(ends[i].y, i === ends.length - 1 ? m.t + h - FLOOR : ends[i + 1].y - GAP)
    }
    return ends
  }
  const struck = ({ name, x, y }: { name: string; x: number; y: number }) =>
    pauses.some(
      (q) =>
        q.name !== name &&
        Array.from({ length: Math.ceil(name.length * CHAR) + 1 }, (_, i) => at(q, T0 + ((x + i - m.l) / w) * (T1 - T0))).some(
          (v) => v !== undefined && Math.abs(Y(v.hired) - y) < CAP,
        ),
    )
  const moved = new Set(spread(pauses).filter(struck).map((l) => l.name))
  const ends = spread(pauses.filter((p) => !moved.has(p.name)))
  const left = pauses.filter((p) => moved.has(p.name)).map((p) => end(p, -6))

  // The date under the pointer, if a pause runs then.
  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const x = (e.clientX - e.currentTarget.getBoundingClientRect().left - m.l) / w
    const t = Math.min(T1, Math.max(T0, T0 + x * (T1 - T0)))
    setDate(pauses.some((p) => at(p, t) !== undefined) ? t : undefined)
  }
  const lit: { name: string; hired: number; staff: number }[] = []
  for (const p of pauses) {
    const v = date === undefined ? undefined : at(p, date)
    if (v !== undefined) lit.push({ name: p.name, ...v })
  }

  return (
    <figure className="paper-plot">
      {title && <div className="paper-plot-axis">{title}</div>}
      <div className="paper-plot-box" ref={box}>
        {w > 0 && (
          <svg
            width={width}
            height={m.t + h + m.b}
            role="img"
            aria-label={`People the labs hire on top of their own staff, by date, in a pause from ${pauses.map((p) => p.name).join(', ')} on`}
            onPointerMove={(e) => e.pointerType !== 'touch' && pick(e)}
            onPointerDown={(e) => e.pointerType === 'touch' && pick(e)}
            onPointerLeave={(e) => e.pointerType !== 'touch' && setDate(undefined)}
          >
            {steps(y).map((v) => (
              <g key={v}>
                <line className="pp-grid" x1={m.l} x2={m.l + w} y1={Y(v)} y2={Y(v)} />
                <text className="pp-tick" x={m.l - 6} y={Y(v) + 3.5} textAnchor="end">
                  {people(v)}
                </text>
              </g>
            ))}
            {years.map((year) => (
              <text key={year} className="pp-tick" x={X(year)} y={m.t + h + 16} textAnchor="middle">
                {year}
              </text>
            ))}
            {date !== undefined && <line className="pp-cross" x1={X(date)} x2={X(date)} y1={m.t} y2={m.t + h} />}
            {pauses.map((p) => (
              <path key={p.name} className="pp-line" d={line(p)} />
            ))}
            {date !== undefined &&
              lit.map((l) => <circle key={l.name} className="pp-dot" cx={X(date)} cy={Y(l.hired)} r={4} />)}
            {ends.map((l) => (
              <text key={l.name} className="pp-label" x={l.x} y={l.y + 3.5}>
                {l.name}
              </text>
            ))}
            {left.map((l) => (
              <text key={l.name} className="pp-label" x={l.x} y={l.y + 3.5} textAnchor="end">
                {l.name}
              </text>
            ))}
          </svg>
        )}
        {date !== undefined && lit.length > 0 && (
          <Tip x={X(date)} top={m.t} width={width}>
            <div className="paper-plot-tip-date">{month(date)}</div>
            <table>
              <tbody>
                {lit.map((l) => (
                  <tr key={l.name}>
                    <th scope="row">{l.name}</th>
                    <td>{count(Math.round(l.hired))}</td>
                    <td>{Math.round(l.hired) > 0 && `×${share(l.hired / l.staff)} staff`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Tip>
        )}
      </div>
      {caption.length > 0 && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

/** People hired per person of the staff: two decimals. */
const share = (r: number) => (r < 0.01 ? '<0.01' : r.toFixed(2))
