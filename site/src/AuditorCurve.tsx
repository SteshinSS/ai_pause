import { Children, isValidElement, useState, type PointerEvent, type ReactNode } from 'react'
import table from '../content/table.json'
import { useHiring } from './hiring'
import { Tip } from './PaperCountPlot'
import { axis, count, duration, people, steps } from './plot'
import { useWidth } from './useWidth'

// ---------- AI safety years against the auditors N_a, of content/table.json (pause.py's auditor_curves()) ----------

// Auditors on a log scale on x, whole decades from the fewest to the most; AI safety years on a linear scale on y, from 0
// to a round number above the most of the pause length shown (plot.ts).
const { N_a: AUDITORS, rows } = table.auditor_curves
const DURATIONS = table.durations_years
const LAST = AUDITORS.length - 1
const LO = Math.floor(Math.log10(AUDITORS[0]))
const HI = Math.ceil(Math.log10(AUDITORS[LAST]))
const DECADES = Array.from({ length: HI - LO + 1 }, (_, i) => LO + i)
/** The table's auditors, pause_inputs.csv's N_a: a dot on each line. */
const TABLE = AUDITORS.indexOf(table.inputs.N_a)
/** Shown first: a three-year pause, as on the main page. */
const FIRST = Math.max(0, DURATIONS.indexOf(3))
const GAP = 12 // px: the least distance between the rows' labels

/**
 * `<AuditorCurve>` in an .mdx file: the AI safety years of the approved model (Workload Approval) when N_a auditors
 * approve the code, from the fewest to the most that pause.py's auditor_curves() solves for (it writes them to
 * content/table.json), a line per row where the model applies, for the pause length picked on the slider above it (a
 * stop per length of the result tables). Dots mark the table's N_a. The lab has its own staff, or unlimited hiring while
 * the result tables' switch is on (hiring.ts). Monochrome, as PaperCountPlot.tsx; hovering or tapping shows the nearest
 * computed N_a. The children are the axes' titles, `<span data-axis>` for y and `<span data-x>` for x, then the caption.
 * In .mdx files it needs no import (Pane.tsx provides it).
 */
export function AuditorCurve({ children }: { children?: ReactNode }) {
  let yTitle: ReactNode
  let xTitle: ReactNode
  const caption: ReactNode[] = []
  type Label = { 'data-axis'?: unknown; 'data-x'?: unknown; children?: ReactNode }
  Children.forEach(children, (child) => {
    if (isValidElement<Label>(child)) {
      if (child.props['data-axis'] !== undefined) return void (yTitle = child.props.children)
      if (child.props['data-x'] !== undefined) return void (xTitle = child.props.children)
    }
    caption.push(child)
  })

  const hiring = useHiring()
  const [chosen, setChosen] = useState(FIRST)
  const [point, setPoint] = useState<number>() // the computed N_a under the pointer
  const lines = rows.map((r) => ({ name: r.name, cells: (hiring ? r.unlimited : r.cells).map((c) => c[chosen]) }))
  const y = axis(Math.max(...lines.flatMap((l) => l.cells)))
  const [box, width] = useWidth()
  const m = { l: 36, r: 50, t: 8, b: 24 } // as the other plots of the page, so that their plot areas line up
  const w = Math.max(0, width - m.l - m.r)
  const h = Math.round(Math.min(320, Math.max(200, w * 0.6)))
  const X = (n: number) => m.l + ((Math.log10(n) - LO) / (HI - LO)) * w
  const Y = (v: number) => m.t + h - (v / y.top) * h
  const path = (cells: number[]) =>
    AUDITORS.map((n, i) => `${i ? 'L' : 'M'}${X(n).toFixed(1)},${Y(cells[i]).toFixed(1)}`).join('')

  // Each row's name at its line's end, moved apart where they would touch.
  const ends = lines
    .map((l) => ({ name: l.name, x: X(AUDITORS[LAST]) + 6, y: Y(l.cells[LAST]) }))
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + GAP)

  // The computed N_a nearest the pointer on the log scale.
  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const u = LO + ((e.clientX - e.currentTarget.getBoundingClientRect().left - m.l) / w) * (HI - LO)
    const near = AUDITORS.map((n) => Math.abs(Math.log10(n) - u))
    setPoint(near.indexOf(Math.min(...near)))
  }

  // dots at the table's N_a and at the one under the pointer
  const dots = [...new Set([TABLE, point])].filter((i): i is number => i !== undefined && i >= 0)
  const range = lines.map((l) => `${l.name} ${count(l.cells[0])} to ${count(l.cells[LAST])}`).join(', ')
  return (
    <figure className="paper-plot auditor-curve">
      <label className="plot-slider">
        <span>Pause</span>
        <input
          type="range"
          min={0}
          max={DURATIONS.length - 1}
          step={1}
          value={chosen}
          onChange={(e) => {
            setChosen(Number(e.target.value))
            setPoint(undefined)
          }}
          aria-valuetext={duration(DURATIONS[chosen])}
        />
        <span className="plot-slider-value">{duration(DURATIONS[chosen])}</span>
      </label>
      {yTitle && <div className="paper-plot-axis">{yTitle}</div>}
      <div className="paper-plot-box" ref={box}>
        {w > 0 && (
          <svg
            width={width}
            height={m.t + h + m.b}
            role="img"
            aria-label={`AI safety years in a pause of ${duration(DURATIONS[chosen])} against auditors, ${count(AUDITORS[0])} to ${count(AUDITORS[LAST])}: ${range}`}
            onPointerMove={(e) => e.pointerType !== 'touch' && pick(e)}
            onPointerDown={(e) => e.pointerType === 'touch' && pick(e)}
            onPointerLeave={(e) => e.pointerType !== 'touch' && setPoint(undefined)}
          >
            {steps(y).map((v) => (
              <g key={v}>
                <line className="pp-grid" x1={m.l} x2={m.l + w} y1={Y(v)} y2={Y(v)} />
                <text className="pp-tick" x={m.l - 6} y={Y(v) + 3.5} textAnchor="end">
                  {people(v)}
                </text>
              </g>
            ))}
            {DECADES.map((e) => (
              <text key={e} className="pp-tick" x={X(10 ** e)} y={m.t + h + 16} textAnchor="middle">
                {people(10 ** e)}
              </text>
            ))}
            {point !== undefined && (
              <line className="pp-cross" x1={X(AUDITORS[point])} x2={X(AUDITORS[point])} y1={m.t} y2={m.t + h} />
            )}
            {lines.map((l) => (
              <path key={l.name} className="pp-line" d={path(l.cells)} />
            ))}
            {dots.flatMap((i) =>
              lines.map((l) => (
                <circle key={`${i} ${l.name}`} className="pp-dot" cx={X(AUDITORS[i])} cy={Y(l.cells[i])} r={4} />
              )),
            )}
            {ends.map((l) => (
              <text key={l.name} className="pp-label" x={l.x} y={l.y + 3.5}>
                {l.name}
              </text>
            ))}
          </svg>
        )}
        {point !== undefined && w > 0 && (
          <Tip x={X(AUDITORS[point])} top={m.t} width={width}>
            <div className="paper-plot-tip-date">
              {count(AUDITORS[point])} auditors{point === TABLE && ', as in the table'}
            </div>
            <table>
              <tbody>
                {lines.map((l) => (
                  <tr key={l.name}>
                    <th scope="row">{l.name}</th>
                    <td>{count(l.cells[point])}</td>
                    <td>{point !== TABLE && TABLE >= 0 && change(l.cells[point] / l.cells[TABLE] - 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Tip>
        )}
      </div>
      {xTitle && <div className="paper-plot-axis hiring-curve-x">{xTitle}</div>}
      {caption.length > 0 && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

/** The change from the table's cell, as a percentage: −68%, +34%, +2.6%. */
function change(r: number) {
  const p = Math.abs(r) * 100
  return `${r < 0 ? '−' : '+'}${p >= 10 ? Math.round(p) : +p.toFixed(1)}%`
}
