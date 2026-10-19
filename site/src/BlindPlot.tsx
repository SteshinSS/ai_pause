import { useEffect, useLayoutEffect, useRef, useState, type PointerEvent, type ReactNode, type RefObject } from 'react'
import { useWidth } from './useWidth'

type Level = 'all' | 'part' | 'project'
type Model = 'sonnet' | 'opus'
/** A paper of the blind test: the authors' figure (`true`) and each model's estimate, in H100e-years. */
type Row = { id: string; year: number; level: Level; title: string; covers: string; true: number } & Record<Model, number>
/** The paper under the pointer, at its dot's centre in the figure's coordinates. */
type Hover = { row: Row; x: number; y: number }

/** What the authors' figure covers, in the legend's order; the colours are --scope-<level> in styles.css. */
const LEVELS: [Level, string][] = [
  ['all', 'All experiments'],
  ['part', 'A named part'],
  ['project', 'Whole project'],
]
const MODELS: Record<Model, string> = { sonnet: 'Sonnet 5.5', opus: 'Opus 5.5' }
const YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
const HOURS = 8766 // in a year

// Both axes in H100e-hours on a log scale, 10^LO to 10^HI: it holds every figure and estimate (0.0002 to 200k). Grid
// lines at the edges and at the labelled TICKS.
const LO = -4
const HI = 6
const TICKS = [-4, -2, 0, 2, 4, 6]
const BAND = Math.log10(3)
const REACH = 14 // px: how far from a dot the pointer still picks it

// The data is a chunk of its own (~190 kB), loaded when the first plot mounts.
let loading: Promise<Row[]> | undefined
const load = () => (loading ??= import('../content/blind_test.json').then((m) => m.default.rows as Row[]))

const lg = (h100eYears: number) => Math.log10(h100eYears * HOURS)

/**
 * `<BlindPlot>caption</BlindPlot>`: the blind test of the compute estimates, content/blind_test.json (written by
 * safety_papers/export.py). A log-log scatter of each model's estimate against the authors' figure, one panel per model;
 * with `by="year"`, Sonnet's estimates, one panel per year. Buttons above hide papers by what the authors' figure
 * covers; hovering or tapping a dot shows the paper. In .mdx files it needs no import (Pane.tsx provides it).
 */
export function BlindPlot({ by = 'model', children }: { by?: 'model' | 'year'; children?: ReactNode }) {
  const [rows, setRows] = useState<Row[]>()
  const [shown, setShown] = useState<Level[]>(LEVELS.map(([l]) => l))
  const [hover, setHover] = useState<Hover>()
  const frame = useRef<HTMLElement>(null)
  useEffect(() => {
    let live = true
    load().then((r) => live && setRows(r))
    return () => {
      live = false
    }
  }, [])

  const visible = rows?.filter((r) => shown.includes(r.level)) ?? []
  const panels: { title: string; model: Model; rows: Row[] }[] =
    by === 'year'
      ? YEARS.map((y) => ({ title: String(y), model: 'sonnet', rows: visible.filter((r) => r.year === y) }))
      : (['sonnet', 'opus'] as const).map((m) => ({ title: MODELS[m], model: m, rows: visible }))
  const toggle = (level: Level) =>
    setShown((s) => (s.includes(level) ? s.filter((l) => l !== level) : [...s, level]))

  return (
    <figure className="blind-plot" data-by={by} ref={frame}>
      <div className="blind-plot-legend" role="group" aria-label="Papers by what the authors' figure covers">
        {LEVELS.map(([level, label]) => (
          <button key={level} type="button" aria-pressed={shown.includes(level)} onClick={() => toggle(level)}>
            <span className={`blind-plot-key bp-${level}`} aria-hidden="true" />
            {label}
            <span className="blind-plot-count">{rows ? rows.filter((r) => r.level === level).length : ''}</span>
          </button>
        ))}
      </div>
      <div className="blind-plot-panels">
        {panels.map((p) => (
          <Panel key={p.title} {...p} axes={by === 'model'} hover={hover} setHover={setHover} frame={frame} />
        ))}
      </div>
      {hover && <Tip hover={hover} frame={frame} />}
      {children && <figcaption>{children}</figcaption>}
    </figure>
  )
}

function Panel({
  title,
  model,
  rows,
  axes,
  hover,
  setHover,
  frame,
}: {
  title: string
  model: Model
  rows: Row[]
  axes: boolean
  hover: Hover | undefined
  setHover: (h: Hover | undefined) => void
  frame: RefObject<HTMLElement | null>
}) {
  const [box, width] = useWidth()
  const m = { l: 38, r: 6, t: 6, b: axes ? 40 : 22 }
  const side = Math.max(0, width - m.l - m.r)
  const at = (e: number) => ((e - LO) / (HI - LO)) * side
  const X = (v: number) => m.l + at(lg(v))
  const Y = (v: number) => m.t + side - at(lg(v))
  const pt = (x: number, y: number) => `${m.l + at(x)},${m.t + side - at(y)}`

  // The paper whose dot is nearest the pointer, if it is within REACH.
  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const f = frame.current?.getBoundingClientRect()
    const px = e.clientX - r.left
    const py = e.clientY - r.top
    let best: Row | undefined
    let dist = REACH * REACH
    for (const d of rows) {
      const dd = (X(d.true) - px) ** 2 + (Y(d[model]) - py) ** 2
      if (dd < dist) [best, dist] = [d, dd]
    }
    setHover(best && f ? { row: best, x: r.left - f.left + X(best.true), y: r.top - f.top + Y(best[model]) } : undefined)
  }

  const s = stats(rows, model)
  const fit = s && rows.length > 2 ? s.fit : undefined
  const lit = hover && rows.find((d) => d.id === hover.row.id)
  return (
    <div className="blind-plot-panel">
      <div className="blind-plot-title">{title}</div>
      <div className="blind-plot-stats">
        {s ? (
          <>
            <span>
              {rows.length} paper{rows.length === 1 ? '' : 's'} · median ×{ratio(s.median)}
            </span>
            <span>
              {s.within}% within ×3{axes && ` · slope ${s.fit.slope.toFixed(2)}`}
            </span>
          </>
        ) : (
          <span>no papers shown</span>
        )}
      </div>
      <div ref={box}>
        {side > 0 && (
          <svg
            width={width}
            height={side + m.t + m.b}
            role="img"
            aria-label={`${title}: estimate against the authors' figure, H100e-hours, log scales`}
            onPointerMove={(e) => e.pointerType !== 'touch' && pick(e)}
            onPointerDown={(e) => e.pointerType === 'touch' && pick(e)}
            onPointerLeave={(e) => e.pointerType !== 'touch' && setHover(undefined)}
          >
            {[...new Set([LO, ...TICKS, HI])].map((e) => (
              <g key={e}>
                <line className="bp-grid" x1={m.l + at(e)} x2={m.l + at(e)} y1={m.t} y2={m.t + side} />
                <line className="bp-grid" x1={m.l} x2={m.l + side} y1={m.t + side - at(e)} y2={m.t + side - at(e)} />
                {TICKS.includes(e) && (
                  <>
                    <text className="bp-tick" x={m.l + at(e)} y={m.t + side + 14} textAnchor="middle">
                      {tick(e)}
                    </text>
                    <text className="bp-tick" x={m.l - 6} y={m.t + side - at(e) + 3.5} textAnchor="end">
                      {tick(e)}
                    </text>
                  </>
                )}
              </g>
            ))}
            <polygon
              className="bp-band"
              points={[
                pt(LO, LO + BAND),
                pt(HI - BAND, HI),
                pt(HI, HI),
                pt(HI, HI - BAND),
                pt(LO + BAND, LO),
                pt(LO, LO),
              ].join(' ')}
            />
            <line className="bp-diag" x1={m.l} y1={m.t + side} x2={m.l + side} y2={m.t} />
            {axes && (
              <>
                <text className="bp-axis" x={m.l + side / 2} y={m.t + side + 34} textAnchor="middle">
                  Authors' figure, H100e-hours
                </text>
                <text
                  className="bp-axis"
                  transform={`translate(10 ${m.t + side / 2}) rotate(-90)`}
                  textAnchor="middle"
                >
                  Estimate, H100e-hours
                </text>
              </>
            )}
            {LEVELS.map(([level]) => (
              <g key={level} className={`bp-${level}`}>
                {rows
                  .filter((d) => d.level === level)
                  .map((d) => (
                    <circle key={d.id} className="bp-dot" cx={X(d.true)} cy={Y(d[model])} r={4} />
                  ))}
              </g>
            ))}
            {fit && (
              <line
                className="bp-fit"
                x1={m.l + at(fit.x0)}
                y1={m.t + side - at(fit.y0)}
                x2={m.l + at(fit.x1)}
                y2={m.t + side - at(fit.y1)}
              />
            )}
            {lit && (
              <circle className={`bp-lit bp-${lit.level}`} cx={X(lit.true)} cy={Y(lit[model])} r={5.5} />
            )}
          </svg>
        )}
      </div>
    </div>
  )
}

/** The paper's card next to its dot, kept inside the figure. */
function Tip({ hover, frame }: { hover: Hover; frame: RefObject<HTMLElement | null> }) {
  const tip = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const t = tip.current
    const f = frame.current
    if (!t || !f) return
    const [w, h, gap] = [t.offsetWidth, t.offsetHeight, 14]
    const right = hover.x + gap
    t.style.left = `${right + w <= f.clientWidth ? right : Math.max(0, Math.min(hover.x - gap - w, f.clientWidth - w))}px`
    t.style.top = `${hover.y + gap + h > f.clientHeight && hover.y - gap - h >= 0 ? hover.y - gap - h : hover.y + gap}px`
    t.style.visibility = 'visible'
  }, [hover, frame])
  const d = hover.row
  const scope = LEVELS.find(([l]) => l === d.level)?.[1]
  return (
    <div className="blind-plot-tip" ref={tip} style={{ visibility: 'hidden' }}>
      <div className="blind-plot-tip-id">
        {d.id} · {d.year}
      </div>
      <div className="blind-plot-tip-title">{d.title}</div>
      <div className="blind-plot-tip-scope">
        {scope}
        {d.covers && `: ${d.covers}`}
      </div>
      <table>
        <tbody>
          <tr>
            <th scope="row">Authors</th>
            <td>{hours(d.true)}</td>
            <td />
          </tr>
          {(['sonnet', 'opus'] as const).map((m) => (
            <tr key={m}>
              <th scope="row">{MODELS[m]}</th>
              <td>{hours(d[m])}</td>
              <td>×{ratio(d[m] / d.true)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Median of estimate / figure, the share within ×3 (percent) and the least-squares line of the log estimate on the
 *  log figure, in log10 H100e-hours; as in safety_papers/results/blind_test.md. */
function stats(rows: Row[], model: Model) {
  if (!rows.length) return undefined
  const xs = rows.map((d) => lg(d.true))
  const ys = rows.map((d) => lg(d[model]))
  const err = ys.map((y, i) => y - xs[i]).sort((a, b) => a - b)
  const n = err.length
  const median = 10 ** (n % 2 ? err[(n - 1) / 2] : (err[n / 2 - 1] + err[n / 2]) / 2)
  const within = Math.round((100 * err.filter((e) => Math.abs(e) <= BAND + 1e-12).length) / n)
  const mx = xs.reduce((a, b) => a + b) / n
  const my = ys.reduce((a, b) => a + b) / n
  let sxy = 0
  let sxx = 0
  xs.forEach((x, i) => {
    sxy += (x - mx) * (ys[i] - my)
    sxx += (x - mx) ** 2
  })
  const slope = sxx ? sxy / sxx : 0
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  return { median, within, fit: { slope, x0, x1, y0: my + slope * (x0 - mx), y1: my + slope * (x1 - mx) } }
}

/** A tick at 10^e H100e-hours: 0.0001, 0.01, 1, 100, 10k, 1M. */
const tick = (e: number) => (e < 3 ? String(10 ** e) : e < 6 ? `${10 ** (e - 3)}k` : `${10 ** (e - 6)}M`)

const ratio = (r: number) => (r >= 10 ? Math.round(r).toLocaleString('en-US') : r.toFixed(2))

function hours(h100eYears: number) {
  const h = h100eYears * HOURS
  const v = h >= 100 ? Math.round(h).toLocaleString('en-US') : h >= 1 ? h.toPrecision(2) : h.toPrecision(1)
  return `${v} H100e-h`
}
