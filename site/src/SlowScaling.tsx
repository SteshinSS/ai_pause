import { Children, isValidElement, useState, type PointerEvent, type ReactNode } from 'react'
import data from '../content/slow_scaling.json'
import table from '../content/table.json'
import { PROTOCOLS } from './content'
import { Tip } from './PaperCountPlot'
import { axis, count, month, people, steps } from './plot'
import { useWidth } from './useWidth'

// ---------- the Plan A labs' handoff delay against the Full Speed labs' years at TED-AI, of content/slow_scaling.json ----------

// The Full Speed lab's years at TED-AI k on x, a point a month (slow_scaling.py); the handoff delay Delta t in years on y,
// from 0 to the longest delay (a line starts at its delay). The worked example draws both labs on one calendar, from
// AC, where the Plan A labs' pause starts, and the research a year at TED-AI, a point a month from the Full Speed labs' TED-AI.
const { lags: LAGS, k: K, t_ted: T0, t_ac: T_AC, d: DAYS, stages: STAGES } = data
const D_FAST = T0 - T_AC // years: the Full Speed labs' race from AC to TED-AI, AIFM's
const DELTA = data.delta_t as Record<string, number[][]>
const RATE = data.rate as Record<string, number[]>
const SAFETY = data.safety_years as Record<string, number[]>
const MODELS = Object.keys(DELTA)
const LAST = K.length - 1
const K1 = K[LAST]
/** Shown first in both plots: Workload Approval, else the main page's trust model. */
const FIRST = MODELS.includes('approved') ? MODELS.indexOf('approved') : Math.max(0, MODELS.indexOf(table.main))
const GAP = 12 // px: the least distance between the lines' labels
const m = { l: 36, r: 50, t: 8, b: 24 } // as the other plots of the page, so that their plot areas line up

/** A trust model's name on the website: its menu popup's title (content/protocols/). */
const name = (model: string) => PROTOCOLS[model.replaceAll(' ', '-')]?.title ?? model
const years = (L: number) => `${L} year${L === 1 ? '' : 's'}`
/** A monthly series of the example at d years after the Full Speed labs' TED-AI, interpolated. */
const at = (v: number[], d: number) => {
  const i = Math.min(v.length - 2, Math.max(0, Math.floor(d * 12)))
  return v[i] + (v[i + 1] - v[i]) * (d * 12 - i)
}
/** The d at which a rising monthly series reaches v, interpolated. */
const inverse = (series: number[], v: number) => {
  let i = 0
  while (i < series.length - 2 && series[i + 1] < v) i++
  return (i + (v - series[i]) / (series[i + 1] - series[i])) / 12
}
/** Width in px of `text` in the plots' 10px monospace. */
const textWidth = (text: string) => text.length * 6.4

/** The children: `<span data-axis>` the y axis's title, `<span data-x>` the x axis's, the rest the caption. */
function parts(children: ReactNode) {
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
  return { yTitle, xTitle, caption }
}

/** A menu above a plot, as HiringCurve's. */
function Pick<T extends string | number>(props: {
  label: string
  value: T
  options: readonly T[]
  show: (v: T) => string
  set: (v: T) => void
}) {
  return (
    <label className="hiring-curve-pick">
      <span>{props.label}</span>
      <span className="hiring-curve-select">
        <select
          value={props.options.indexOf(props.value)}
          onChange={(e) => props.set(props.options[Number(e.target.value)])}
        >
          {props.options.map((v, i) => (
            <option key={String(v)} value={i}>
              {props.show(v)}
            </option>
          ))}
        </select>
      </span>
    </label>
  )
}

/**
 * `<SlowScaling>` in an .mdx file: an upper bound on how much later the Plan A lab hands off, Delta t, when it reaches
 * TED-AI 1 to 5 years after the Full Speed lab, against the years the Full Speed lab spends at TED-AI, a line per delay, for the
 * trust model picked in the menu above it, Workload Approval first (slow_scaling.py writes them to
 * content/slow_scaling.json). The lab has its own staff. Monochrome, as PaperCountPlot.tsx; hovering or tapping shows
 * the nearest month. The children are the axes' titles, `<span data-axis>` for y and `<span data-x>` for x, then the
 * caption. `<SlowScaling example>`: one Full Speed and one Plan A lab on one calendar, from AC, and the research a year at
 * TED-AI, shaded where only one of them researches; a menu picks the trust model and sliders the Plan A labs' Stage 1
 * (AC to TED-AI) and Stage 2 (at TED-AI), and the Full Speed lab hands off when it has done as much AI safety research; the caption under it says when each hands off. In .mdx files it needs no import (Pane.tsx provides it).
 */
export function SlowScaling({ example, children }: { example?: boolean; children?: ReactNode }) {
  return example ? <Example /> : <Delays>{children}</Delays>
}

function Delays({ children }: { children?: ReactNode }) {
  const { yTitle, xTitle, caption } = parts(children)
  const [chosen, setChosen] = useState(MODELS[FIRST])
  const [point, setPoint] = useState<number>() // the month under the pointer, an index of K
  const lines = LAGS.map((L, i) => ({ L, v: DELTA[chosen][i] }))
  const y = axis(Math.max(...LAGS))
  const [box, width] = useWidth()
  const w = Math.max(0, width - m.l - m.r)
  const h = Math.round(Math.min(320, Math.max(200, w * 0.6)))
  const X = (k: number) => m.l + (k / K1) * w
  const Y = (v: number) => m.t + h - (v / y.top) * h
  const path = (v: number[]) => K.map((k, i) => `${i ? 'L' : 'M'}${X(k).toFixed(1)},${Y(v[i]).toFixed(1)}`).join('')
  const every = w < 300 ? 5 : w < 560 ? 2 : 1 // years between the ticks
  const ticks = Array.from({ length: Math.floor(K1 / every) + 1 }, (_, i) => i * every)

  // Each delay at its line's end, moved apart where they would touch.
  const ends = lines.map((l) => ({ L: l.L, x: X(K1) + 6, y: Y(l.v[LAST]) })).sort((a, b) => b.y - a.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.min(ends[i].y, ends[i - 1].y - GAP)

  // The month nearest the pointer.
  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const k = ((e.clientX - e.currentTarget.getBoundingClientRect().left - m.l) / w) * K1
    setPoint(Math.min(LAST, Math.max(0, Math.round(k * 12))))
  }

  const range = lines.map((l) => `${years(l.L)} later, ${l.v[0].toFixed(1)} to ${l.v[LAST].toFixed(2)}`).join('; ')
  return (
    <figure className="paper-plot slow-scaling">
      <Pick label="Verification level" value={chosen} options={MODELS} show={name} set={setChosen} />
      {yTitle && <div className="paper-plot-axis">{yTitle}</div>}
      <div className="paper-plot-box" ref={box}>
        {w > 0 && (
          <svg
            width={width}
            height={m.t + h + m.b}
            role="img"
            aria-label={`${name(chosen)}: how much later the Plan A labs hand off, in years, when the Full Speed labs spend 0 to ${K1} years at TED-AI, by how much later the Plan A labs reach TED-AI: ${range}`}
            onPointerMove={(e) => e.pointerType !== 'touch' && pick(e)}
            onPointerDown={(e) => e.pointerType === 'touch' && pick(e)}
            onPointerLeave={(e) => e.pointerType !== 'touch' && setPoint(undefined)}
          >
            {steps(y).map((v) => (
              <g key={v}>
                <line className="pp-grid" x1={m.l} x2={m.l + w} y1={Y(v)} y2={Y(v)} />
                <text className="pp-tick" x={m.l - 6} y={Y(v) + 3.5} textAnchor="end">
                  {v}
                </text>
              </g>
            ))}
            {ticks.map((k) => (
              <text key={k} className="pp-tick" x={X(k)} y={m.t + h + 16} textAnchor="middle">
                {k}
              </text>
            ))}
            {point !== undefined && <line className="pp-cross" x1={X(K[point])} x2={X(K[point])} y1={m.t} y2={m.t + h} />}
            {lines.map((l) => (
              <path key={l.L} className="pp-line" d={path(l.v)} />
            ))}
            {point !== undefined &&
              lines.map((l) => <circle key={l.L} className="pp-dot" cx={X(K[point])} cy={Y(l.v[point])} r={4} />)}
            {ends.map((l) => (
              <text key={l.L} className="pp-label" x={l.x} y={l.y + 3.5}>
                {l.L} yr
              </text>
            ))}
          </svg>
        )}
        {point !== undefined && w > 0 && (
          <Tip x={X(K[point])} top={m.t} width={width}>
            <div className="paper-plot-tip-date">
              {years(+K[point].toFixed(2))} at TED-AI, handoff {month(T0 + K[point])}
            </div>
            <table>
              <tbody>
                {lines.map((l) => (
                  <tr key={l.L}>
                    <th scope="row">{years(l.L)} later</th>
                    <td>{l.v[point].toFixed(2)}</td>
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

/** A slider above a plot from `min` to `max` in steps of `step` years, its value shown after it. */
function Slider(props: {
  label: string
  value: number
  min: number
  max: number
  step: number
  set: (v: number) => void
}) {
  return (
    <label className="plot-slider">
      <span>{props.label}</span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.set(Number(e.target.value))}
        aria-valuetext={years(props.value)}
      />
      <span className="plot-slider-value">{years(props.value)}</span>
    </label>
  )
}

const EXAMPLE = { stage1: 5, stage2: 5 } // the example's first view

function Example() {
  const [chosen, setChosen] = useState(MODELS[FIRST])
  const [stage1, setStage1] = useState(Math.min(EXAMPLE.stage1, STAGES[0])) // the Plan A labs' years from AC to TED-AI
  const [stage2, setStage2] = useState(Math.min(EXAMPLE.stage2, STAGES[1])) // and at TED-AI
  const [box, width] = useWidth()
  const rate = RATE[chosen]
  // The Plan A lab reaches TED-AI L years after the Full Speed lab; the Full Speed lab hands off when it has done as much AI safety
  // research, k years after its TED-AI: C(k) = C(L + Stage 2) - C(L). Delta t = L + Stage 2 - k, how much earlier.
  const L = stage1 - D_FAST
  const C = SAFETY[chosen]
  const k = inverse(C, at(C, L + stage2) - at(C, L))
  const dt = L + stage2 - k
  // Only the Full Speed lab researches until the Plan A lab reaches TED-AI or the Full Speed lab hands off, whichever comes first;
  // only the Plan A lab from the later of the two: the same AI safety years, C(min(L, k)) = C(k + Delta t) - C(max(L, k)).
  const first = Math.min(L, k)
  const last = Math.max(L, k)
  const lost = at(SAFETY[chosen], first)
  // Handing off together, at the Full Speed labs' date: the Plan A lab lacks the shaded end, C(min(L, k)) / C(k) of the research.
  const share = Math.min(1, lost / at(SAFETY[chosen], k))

  // One calendar: from just before AC to just past the Plan A labs' handoff.
  const w = Math.max(0, width - m.l - m.r)
  const x0 = T_AC - 0.3
  const x1 = T0 + L + stage2 + 0.6
  const X = (t: number) => m.l + ((t - x0) / (x1 - x0)) * w
  const bar = 22
  const fastY = 40
  const slowY = 88
  const top = slowY + bar + 46 // the research a year, below the bars
  const h = Math.round(Math.min(210, Math.max(140, w * 0.36)))
  const bottom = top + h
  const y = axis(at(rate, k + dt + 0.3) * 1.15, 4)
  const Y = (v: number) => bottom - (v / y.top) * h
  const span = (x1 - x0) / Math.max(1, w)
  const every = [1, 2, 5, 10].find((e) => e / span >= 44) ?? 10 // years between the date ticks
  const ticks = Array.from({ length: Math.floor(x1 / every) - Math.ceil(x0 / every) + 1 }, (_, i) =>
    (Math.ceil(x0 / every) + i) * every,
  )

  // The research a year from a to b years after the Full Speed labs' TED-AI: a point a month, and both ends.
  const curve = (a: number, b: number) => {
    const ds = [a, ...DAYS.filter((d) => d > a && d < b), b]
    return ds.map((d, i) => `${i ? 'L' : 'M'}${X(T0 + d).toFixed(1)},${Y(at(rate, d)).toFixed(1)}`).join('')
  }
  const area = (a: number, b: number) =>
    `${curve(a, b)}L${X(T0 + b).toFixed(1)},${bottom}L${X(T0 + a).toFixed(1)},${bottom}Z`

  // A segment of a lab's bar, its label inside it if one fits, in the wider side of a guide that crosses the bar.
  const segment = (a: number, b: number, y0: number, kind: string, labels: string[], guide?: number) => {
    const [p, q] = guide !== undefined && guide > a && guide < b ? (guide - a > b - guide ? [a, guide] : [guide, b]) : [a, b]
    const label = labels.find((t) => textWidth(t) + 14 <= X(q) - X(p))
    return (
      <g>
        <rect className={`ss-${kind}`} x={X(a)} y={y0} width={Math.max(0, X(b) - X(a))} height={bar} rx={3} />
        {label && (
          <text className="ss-bar-text" x={(X(p) + X(q)) / 2} y={y0 + bar / 2 + 3.5} textAnchor="middle">
            {label}
          </text>
        )}
      </g>
    )
  }

  const th = T0 + k // the Full Speed labs' handoff
  const end = th + dt // the Plan A labs'
  const stay = stage2 // the Plan A labs' years at TED-AI, k - L + Delta t
  const delayLabel = `Δt ${dt.toFixed(2)} yr`
  const delayRight = X(end) + 5 + textWidth(delayLabel) <= width
  const one = (v: number) => +v.toFixed(1)
  // A lab's time at TED-AI: its arrival, dated like AC above the bars, and its years there; shorter for a narrow bar.
  const atTed = (from: number, yr: number) => [
    `TED-AI ${month(from)} · ${one(yr)} yr`,
    `TED-AI ${month(from)}`,
    `${one(yr)} yr`,
  ]
  const pct = `${share < 0.1 ? (100 * share).toFixed(1) : Math.round(100 * share)}%`
  const endLabel = `${count(lost)} · ${pct}`
  const endRight = X(end) + 5 + textWidth(endLabel) <= width
  return (
    <figure className="paper-plot slow-scaling slow-scaling-example">
      <div className="slow-scaling-picks">
        <Pick label="Verification level" value={chosen} options={MODELS} show={name} set={setChosen} />
      </div>
      <div className="slow-scaling-sliders">
        <Slider label="Plan A labs' Stage 1" value={stage1} min={1.5} max={STAGES[0]} step={0.5} set={setStage1} />
        <Slider label="Plan A labs' Stage 2" value={stage2} min={0.5} max={STAGES[1]} step={0.5} set={setStage2} />
      </div>
      <div className="paper-plot-box" ref={box}>
        {w > 0 && (
          <svg
            width={width}
            height={bottom + m.b}
            role="img"
            aria-label={`${name(chosen)}: the Full Speed labs race from AC, ${month(T_AC)}, to TED-AI, ${month(T0)}, and pause there for ${years(one(k))}, until ${month(th)}. The Plan A labs pause at AC, scale carefully, reach TED-AI ${one(L)} years after the Full Speed labs, and research there for ${years(one(stay))}, until ${month(end)}. The AI safety research the Full Speed labs do in the Plan A labs' missing years equals what only the Plan A labs do, after the Full Speed labs have handed off: ${pct} of each scenario's total.`}
          >
            <line className="ss-guide" x1={X(T_AC)} x2={X(T_AC)} y1={2} y2={slowY + bar} />
            <text className="pp-tick" x={X(T_AC) + 4} y={10}>
              AC {month(T_AC)}
            </text>
            <text className="ss-row" x={X(T_AC) + 4} y={fastY - 6}>
              Full Speed labs
            </text>
            <text className="ss-row" x={X(T_AC) + 4} y={slowY - 6}>
              Plan A labs
            </text>
            {segment(T_AC, T0, fastY, 'race', ['racing'])}
            {segment(T0, th, fastY, 'fast', atTed(T0, k))}
            {segment(T_AC, T0 + L, slowY, 'scale', [`careful scaling · ${one(T0 + L - T_AC)} yr`, `scaling · ${one(T0 + L - T_AC)} yr`, 'scaling'], th)}
            {segment(T0 + L, end, slowY, 'slow', atTed(T0 + L, stay), th)}
            {[
              [th, fastY],
              [end, slowY],
            ].map(([t, y0]) => (
              <g key={y0}>
                <line className="ss-guide" x1={X(t)} x2={X(t)} y1={y0} y2={slowY + bar + 22} />
                <line className="ss-guide" x1={X(t)} x2={X(t)} y1={top} y2={bottom} />
              </g>
            ))}
            <text
              className="pp-label"
              x={delayRight ? X(end) + 5 : X(th) - 5}
              y={slowY + bar + 16}
              textAnchor={delayRight ? 'start' : 'end'}
            >
              {delayLabel}
            </text>

            <text className="ss-title" x={m.l} y={top - 14}>
              AI safety research a year (2025 field = 1)
            </text>
            {steps(y).map((v) => (
              <g key={v}>
                <line className="pp-grid" x1={m.l} x2={m.l + w} y1={Y(v)} y2={Y(v)} />
                <text className="pp-tick" x={m.l - 6} y={Y(v) + 3.5} textAnchor="end">
                  {people(v)}
                </text>
              </g>
            ))}
            {ticks.map((t) => (
              <text key={t} className="pp-tick" x={X(t)} y={bottom + 16} textAnchor="middle">
                {t}
              </text>
            ))}
            <path className="ss-only-fast" d={area(0, first)} />
            {L < k && <path className="ss-both" d={area(L, k)} />}
            <path className="ss-only-slow" d={area(last, k + dt)} />
            <path className="pp-data" d={curve(0, Math.min(DAYS[DAYS.length - 1], k + dt + 0.3))} />
            <text className="pp-label" x={X(T0 + first / 2)} y={Y(at(rate, first)) - 8} textAnchor="middle">
              {count(lost)}
            </text>
            <text
              className="pp-label"
              x={endRight ? X(end) + 5 : X(last + T0) - 5}
              y={endRight ? (Y(at(rate, k + dt)) + bottom) / 2 + 3.5 : bottom - 10}
              textAnchor={endRight ? 'start' : 'end'}
            >
              {endLabel}
            </text>
          </svg>
        )}
      </div>
      <div className="slow-scaling-key" aria-hidden="true">
        <span>
          <i className="ss-only-fast" />
          only the Full Speed labs research
        </span>
        <span>
          <i className="ss-both" />
          both, the same
        </span>
        <span>
          <i className="ss-only-slow" />
          only the Plan A labs research
        </span>
      </div>
      <figcaption>
        The Plan A labs reach TED-AI in {month(T0 + L)} and hand off in {month(end)}. The Full Speed labs do as much AI safety
        research after {years(one(k))} at TED-AI and hand off in {month(th)}, {dt.toFixed(2)} years earlier.{' '}
        {L < k
          ? `If both hand off in ${month(th)} instead, the Plan A labs end up with ${pct} less AI safety research than the Full Speed labs.`
          : `If both hand off in ${month(th)} instead, the Plan A labs haven't reached TED-AI yet and have done no AI safety research.`}
      </figcaption>
    </figure>
  )
}
