import { Children, isValidElement, useState, type PointerEvent, type ReactNode } from 'react'
import data from '../content/sensitivity.json'
import { Tip } from './PaperCountPlot'
import { count, month } from './plot'
import { useWidth } from './useWidth'

// ---------- tornado plots of one scenario, of content/sensitivity.json (sensitivity.py) ----------

type Output = 'years' | 'hired'
type Item = {
  name: string
  label?: string
  p10: number
  p90: number
  years: number[]
  hired: number[]
  t_pause?: number[]
  moves: Record<Output, boolean>
}
type Label = { symbol?: ReactNode; name: string }

const SETS: Record<'ours' | 'aifm', Item[]> = { ours: data.ours, aifm: data.aifm }
/** The staffings of pause.py as the site names them. */
const STAFFING: Record<string, string> = { 'own staff': 'frozen hiring', 'unlimited hiring': 'unlimited hiring' }
const OUTPUTS: { key: Output; title: string; unit: string }[] = [
  { key: 'years', title: `AI safety years, ${STAFFING[data.scenario.staffing.years]}`, unit: 'AI safety years' },
  { key: 'hired', title: `People to hire, ${STAFFING[data.scenario.staffing.hired]}`, unit: 'people to hire' },
]
const MIN_GAP = 44 // px: the least distance between two ticks' centres
const ZERO = 34 // px: the slot of 0 at the left of a log axis that has to show it, and the break after it

/**
 * `<Tornado set="ours">` (or `"aifm"`) in an .mdx file: tornado plots of one scenario, its AI safety years and its people
 * to hire, each with its staffing, as sensitivity.py writes them to content/sensitivity.json. A row per input that moves
 * the result: a bar from the result with every input at its value, the line in the middle, to the result with the input
 * at its 10th percentile (pale) and at its 90th (ink), on a log scale; the rows in order of how far they reach. Inputs
 * that move nothing are listed under the plot. Hovering a row's bars, tapping them or focusing the row shows the numbers. The
 * children name the inputs, `<span data-input="m_C" data-name="merged lab's compute">$m_C$</span>`: the symbol and a
 * short name. Ours need them all; AIFM's parameters take AIFM's names unless given. Then comes the caption. In .mdx files
 * it needs no import (Pane.tsx provides it).
 */
export function Tornado({ set, children }: { set: 'ours' | 'aifm'; children?: ReactNode }) {
  const items = SETS[set]
  const labels = new Map<string, Label>()
  const caption: ReactNode[] = []
  type Props = { 'data-input'?: string; 'data-name'?: string; children?: ReactNode }
  Children.forEach(children, (child) => {
    const input = isValidElement<Props>(child) ? child.props['data-input'] : undefined
    if (input === undefined) return void caption.push(child)
    const { props } = child as { props: Props }
    labels.set(input, { symbol: props.children, name: props['data-name'] ?? '' })
  })
  const labelOf = (item: Item): Label => labels.get(item.name) ?? { name: item.label ?? item.name }
  return (
    <figure className="paper-plot tornado">
      <div className="tornado-key">
        <span>
          <i className="tornado-swatch" data-end="p10" />
          input at its 10th percentile
        </span>
        <span>
          <i className="tornado-swatch" data-end="p90" />
          at its 90th
        </span>
      </div>
      {OUTPUTS.map((output) => (
        <Plot key={output.key} items={items} output={output} labelOf={labelOf} aifm={set === 'aifm'} />
      ))}
      {caption.length > 0 && <figcaption>{caption}</figcaption>}
    </figure>
  )
}

/** One output's plot: its title, the rows, the ticks and the inputs that don't move it. */
function Plot({
  items,
  output,
  labelOf,
  aifm,
}: {
  items: Item[]
  output: (typeof OUTPUTS)[number]
  labelOf: (item: Item) => Label
  aifm: boolean
}) {
  const [box, width] = useWidth()
  const [shown, setShown] = useState<{ name: string; top: number }>() // the row whose numbers show, where it is
  const middle = data.middle[output.key]
  const moving = items.filter((it) => it.moves[output.key])
  const still = items.filter((it) => !it.moves[output.key])

  // Wide: the inputs' names left of the bars; narrow: above them.
  const stacked = width < 520
  const labelW = stacked ? 0 : Math.round(Math.min(240, width * 0.38))
  const gap = stacked ? 0 : 16
  const barW = Math.max(0, width - labelW - gap)
  const s = logScale([middle, ...moving.flatMap((it) => it[output.key])], barW)
  const ends = (it: Item) => it[output.key].map(s.x)
  const reach = (it: Item) => Math.max(s.x(middle), ...ends(it)) - Math.min(s.x(middle), ...ends(it))
  const rows = [...moving].sort((a, b) => reach(b) - reach(a))

  const open = (it: Item, row: HTMLElement) => setShown({ name: it.name, top: row.offsetTop + row.offsetHeight })
  const tip = shown && rows.find((it) => it.name === shown.name)
  return (
    <div className="tornado-plot">
      <div className="paper-plot-axis">{output.title}</div>
      <div className="paper-plot-box" ref={box}>
        {width > 0 && (
          <>
            <div className="tornado-head" style={{ marginLeft: labelW + gap, width: barW }}>
              <span style={{ left: clamp(s.x(middle), 24, barW - 24) }}>{count(middle)}</span>
            </div>
            <div role="list" aria-label={`${output.title}: one input at a time at its 10th and 90th percentiles`}>
              {rows.map((it) => {
                const label = labelOf(it)
                const [lo, hi] = it[output.key]
                const toggle = (e: PointerEvent<HTMLElement>) =>
                  shown?.name === it.name ? setShown(undefined) : open(it, e.currentTarget.parentElement!)
                return (
                  <div
                    key={it.name}
                    role="listitem"
                    tabIndex={0}
                    className="tornado-row"
                    data-on={shown?.name === it.name || undefined}
                    style={{ gridTemplateColumns: stacked ? '1fr' : `${labelW}px ${barW}px`, columnGap: gap }}
                    aria-label={`${label.name || it.name}: ${count(lo)} ${output.unit} at the 10th percentile, ${num(it.p10)}; ${count(hi)} at the 90th, ${num(it.p90)}`}
                    onFocus={(e) => open(it, e.currentTarget)}
                    onBlur={() => setShown(undefined)}
                  >
                    <div className="tornado-label">
                      <span className="tornado-name">
                        {label.symbol}
                        {label.symbol !== undefined && ' '}
                        {label.name}
                      </span>
                      <span className="tornado-range">
                        {num(it.p10)} – {num(it.p90)}
                      </span>
                    </div>
                    <div
                      className="tornado-bars"
                      onPointerEnter={(e) => e.pointerType === 'mouse' && open(it, e.currentTarget.parentElement!)}
                      onPointerLeave={(e) => e.pointerType === 'mouse' && setShown(undefined)}
                      onPointerDown={(e) => e.pointerType !== 'mouse' && toggle(e)}
                    >
                      {s.ticks.map((t) => (
                        <i key={t.v} className="tornado-grid" style={{ left: s.x(t.v) }} />
                      ))}
                      <Bars middle={s.x(middle)} ends={ends(it)} />
                      <i className="tornado-grid" data-middle style={{ left: s.x(middle) }} />
                    </div>
                  </div>
                )
              })}
            </div>
            <div className="tornado-axis" style={{ marginLeft: labelW + gap, width: barW }} aria-hidden="true">
              {s.ticks.map((t) => (
                <span key={t.v} style={{ left: s.x(t.v) }}>
                  {t.label}
                </span>
              ))}
              {s.zero && <i className="tornado-break" style={{ left: ZERO - 9 }} />}
            </div>
            {tip && (
              <Tip x={labelW + gap + Math.max(s.x(middle), ...ends(tip))} top={shown!.top} width={width}>
                <div className="paper-plot-tip-date">
                  {labelOf(tip).symbol}
                  {labelOf(tip).symbol !== undefined && ' '}
                  {labelOf(tip).name}
                </div>
                <table>
                  <tbody>
                    {(['p10', 'p90'] as const).map((end, i) => (
                      <tr key={end}>
                        <th scope="row">
                          <i className="tornado-tip-key" data-end={end} />
                          {num(tip[end])}
                        </th>
                        <td>{count(tip[output.key][i])}</td>
                        <td>{change(tip[output.key][i] / middle - 1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {aifm && tip.t_pause && (
                  <div className="tornado-tip-note">
                    {data.scenario.row} in {month(tip.t_pause[0])} and {month(tip.t_pause[1])}, instead of{' '}
                    {month(data.scenario.t_pause)}
                  </div>
                )}
              </Tip>
            )}
          </>
        )}
      </div>
      {still.length > 0 && (
        <p className="tornado-still">
          No effect:{' '}
          {still.map((it, i) => (
            <span key={it.name}>
              {i > 0 && ', '}
              {labelOf(it).symbol ?? labelOf(it).name}
            </span>
          ))}
          .
        </p>
      )}
    </div>
  )
}

/** A row's two bars from the middle, at x px, to the ends: the input at its 10th percentile, then at its 90th. When both
 *  reach to the same side, the shorter lies on the longer, thinner. */
function Bars({ middle, ends }: { middle: number; ends: number[] }) {
  const same = Math.sign(ends[0] - middle) === Math.sign(ends[1] - middle)
  const order = [0, 1].sort((a, b) => Math.abs(ends[b] - middle) - Math.abs(ends[a] - middle)) // the longer first
  return order.map((i, k) => {
    const left = Math.min(middle, ends[i])
    return (
      <i
        key={i}
        className="tornado-bar"
        data-end={i ? 'p90' : 'p10'}
        data-side={ends[i] < middle ? 'left' : 'right'}
        data-inner={(same && k === 1) || undefined}
        style={{ left, width: Math.max(1, Math.abs(ends[i] - middle)) }}
      />
    )
  })
}

/**
 * A log scale over `width` px for these values, padded a little at both ends; 0, which a log scale can't show, gets a
 * slot of its own at the left, ZERO px with a break after it. Ticks: the most of decades, 1-2-5 or 1-9 per decade, or a
 * linear step, at most 8, no two closer than MIN_GAP px.
 */
function logScale(values: number[], width: number) {
  const positive = values.filter((v) => v > 0)
  if (!positive.length) positive.push(1) // nothing but 0: any scale
  const zero = values.some((v) => v <= 0)
  let lo = Math.log10(Math.min(...positive))
  let hi = Math.log10(Math.max(...positive))
  const pad = Math.max((hi - lo) * 0.04, 0.004)
  lo -= pad
  hi += pad
  const left = zero ? ZERO : 0
  const x = (v: number) => (v <= 0 ? ZERO / 2 - 8 : left + ((Math.log10(v) - lo) / (hi - lo)) * (width - left))

  const inside = (v: number) => Math.log10(v) >= lo && Math.log10(v) <= hi
  const sets: number[][] = []
  for (const stride of [1, 2, 3]) {
    sets.push(range(Math.ceil(lo), Math.floor(hi)).filter((e) => e % stride === 0).map((e) => 10 ** e))
  }
  for (const mults of [[1, 2, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9]]) {
    sets.push(range(Math.floor(lo), Math.ceil(hi)).flatMap((e) => mults.map((m) => m * 10 ** e)).filter(inside))
  }
  const [a, b] = [10 ** lo, 10 ** hi]
  for (const e of range(Math.floor(Math.log10(b - a)) - 2, Math.ceil(Math.log10(b - a)))) {
    for (const m of [1, 2, 5]) {
      const step = m * 10 ** e
      const n = Math.floor(b / step) - Math.ceil(a / step) + 1
      if (n >= 2 && n <= 8) sets.push(range(Math.ceil(a / step), Math.floor(b / step)).map((k) => +(k * step).toPrecision(12)))
    }
  }
  const spread = (set: number[]) => Math.min(...set.slice(1).map((v, i) => x(v) - x(set[i])))
  const fits = sets.filter((set) => set.length >= 2 && set.length <= 8 && spread(set) >= MIN_GAP)
  const ticks = fits.reduce<number[]>((best, set) => (set.length > best.length ? set : best), [])
  return {
    x,
    zero,
    ticks: [...(zero ? [{ v: 0, label: '0' }] : []), ...ticks.map((v) => ({ v, label: compact(v) }))],
  }
}

/** The integers from a to b. */
const range = (a: number, b: number) => Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i)

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), Math.max(lo, hi))

/** A tick: 500, 20k, 1.5M, 10B. */
function compact(v: number) {
  const [d, unit] = v >= 1e9 ? [1e9, 'B'] : v >= 1e6 ? [1e6, 'M'] : v >= 1e3 ? [1e3, 'k'] : [1, '']
  return `${+(v / d).toPrecision(3)}${unit}`
}

const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹'
/** An input's value: three significant digits, a power of ten from a million on: 0.0005, 2,000, 1.88×10¹⁰. */
function num(v: number) {
  const a = Math.abs(v)
  if (a >= 1e6) {
    const e = Math.floor(Math.log10(a))
    const m = +(v / 10 ** e).toPrecision(3)
    const power = `10${[...String(e)].map((d) => SUPERSCRIPT[+d]).join('')}`
    return m === 1 ? power : `${m}×${power}`
  }
  return v.toLocaleString('en-US', { maximumSignificantDigits: 3 }).replace('-', '−')
}

/** A result's change from the middle, as a percentage: −68%, +2.6%, +34,155%. */
function change(r: number) {
  const p = Math.abs(r) * 100
  const shown = p >= 10 ? Math.round(p).toLocaleString('en-US') : +p.toFixed(1)
  return `${r < 0 ? '−' : '+'}${shown}%`
}
