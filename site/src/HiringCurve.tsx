import { Children, isValidElement, useState, type PointerEvent, type ReactNode } from 'react'
import table from '../content/table.json'
import { Tip } from './PaperCountPlot'
import { axis, count, duration, people, steps } from './plot'
import { useWidth } from './useWidth'

type Scenario = (typeof table.hiring_curves.scenarios)[number]

// ---------- AI safety years against people hired, of content/table.json (pause.py's hiring_curves()) ----------

const { scenarios } = table.hiring_curves
/** The trust models' names on the website. */
const MODELS: Record<string, string> = {
  verified: 'Results Verification',
  compartmentalized: 'Compartmentalization',
  approved: 'Workload Approval',
}
const name =(s: Scenario) => `${MODELS[s.model] ?? s.model}, ${s.row}, ${duration(s.years)}`
/** What hiring adds to the cell, as a share of the cell with only the lab's own staff. */
const gain = (s: Scenario, v = s.cells[s.cells.length - 1]) => v / s.cells[0] - 1
/** Shown first: the scenario where hiring adds the most. */
const FIRST = scenarios.reduce((best, s, i) => (gain(s) > gain(scenarios[best]) ? i : best), 0)

/**
 * `<HiringCurve>` in an .mdx file: the AI safety years a pause produces when the lab may hire at most so many people
 * on top of its own staff, from none (the result tables with own staff) to as many as unlimited hiring hires, for one
 * scenario at a time, picked in the menu above it: the trust model, the row and the pause length, of those where hiring
 * changes the result (pause.py's hiring_curves() writes them to content/table.json). Monochrome, as PaperCountPlot.tsx;
 * hovering or tapping shows the nearest computed point. The children are the axes' titles, `<span data-axis>` for y
 * and `<span data-x>` for x, then the caption. In .mdx files it needs no import (Pane.tsx provides it).
 */
export function HiringCurve({ children }: { children?: ReactNode }) {
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

  const [chosen, setChosen] = useState(FIRST)
  const [point, setPoint] = useState<number>() // the computed point under the pointer
  const s = scenarios[chosen]
  const last = s.hired.length - 1
  const x = axis(s.hired[last])
  const y = axis(Math.max(...s.cells))
  const [box, width] = useWidth()
  const m = { l: 36, r: 50, t: 8, b: 24 } // as the other plots of the page, so that their plot areas line up
  const w = Math.max(0, width - m.l - m.r)
  const h = Math.round(Math.min(320, Math.max(200, w * 0.6)))
  const X = (v: number) => m.l + (v / x.top) * w
  const Y = (v: number) => m.t + h - (v / y.top) * h
  const line = s.hired.map((v, i) => `${i ? 'L' : 'M'}${X(v).toFixed(1)},${Y(s.cells[i]).toFixed(1)}`).join('')

  // The computed point nearest the pointer: they are equally spaced from none hired to the last.
  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const v = ((e.clientX - e.currentTarget.getBoundingClientRect().left - m.l) / w) * x.top
    setPoint(Math.min(last, Math.max(0, Math.round((v / s.hired[last]) * last))))
  }

  const ends = [
    { label: 'own staff', x: X(0) + 8, y: Y(s.cells[0]) + 16, anchor: 'start' },
    { label: 'unlimited hiring', x: X(s.hired[last]), y: Y(s.cells[last]) + 16, anchor: 'end' },
  ] as const
  return (
    <figure className="paper-plot hiring-curve">
      <label className="hiring-curve-pick">
        <span>Scenario</span>
        <span className="hiring-curve-select">
          <select
            value={chosen}
            onChange={(e) => {
              setChosen(Number(e.target.value))
              setPoint(undefined)
            }}
          >
            {scenarios.map((sc, i) => (
              <option key={i} value={i}>
                {name(sc)}
              </option>
            ))}
          </select>
        </span>
      </label>
      {yTitle && <div className="paper-plot-axis">{yTitle}</div>}
      <div className="paper-plot-box" ref={box}>
        {w > 0 && (
          <svg
            width={width}
            height={m.t + h + m.b}
            role="img"
            aria-label={`AI safety years against people hired, ${name(s)}: ${count(s.cells[0])} with the labs' own staff, ${count(s.cells[last])} with ${count(s.hired[last])} hired`}
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
            {steps(x).map((v) => (
              <text key={v} className="pp-tick" x={X(v)} y={m.t + h + 16} textAnchor="middle">
                {people(v)}
              </text>
            ))}
            {point !== undefined && (
              <line className="pp-cross" x1={X(s.hired[point])} x2={X(s.hired[point])} y1={m.t} y2={m.t + h} />
            )}
            <path className="pp-line" d={line} />
            {[0, last].map((i) => (
              <circle key={i} className="pp-dot" cx={X(s.hired[i])} cy={Y(s.cells[i])} r={4} />
            ))}
            {point !== undefined && (
              <circle className="pp-dot" cx={X(s.hired[point])} cy={Y(s.cells[point])} r={4} />
            )}
            {ends.map((e) => (
              <text key={e.label} className="pp-end" x={e.x} y={e.y} textAnchor={e.anchor}>
                {e.label}
              </text>
            ))}
          </svg>
        )}
        {point !== undefined && w > 0 && (
          <Tip x={X(s.hired[point])} top={m.t} width={width}>
            <div className="paper-plot-tip-date">
              {point ? `${count(Math.round(s.hired[point]))} hired` : 'own staff only'}
            </div>
            <table>
              <tbody>
                <tr>
                  <th scope="row">AI safety years</th>
                  <td>{count(s.cells[point])}</td>
                  <td>{point > 0 && percent(gain(s, s.cells[point]))}</td>
                </tr>
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

/** A gain as a percentage: +54%, +2.6%. */
const percent = (r: number) => `+${r >= 0.1 ? Math.round(r * 100) : +(r * 100).toFixed(1)}%`
