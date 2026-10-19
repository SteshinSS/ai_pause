// Numbers and axes of the line plots: PaperCountPlot.tsx, HiringPlot.tsx, HiringCurve.tsx, AuditorCurve.tsx.

/** Three significant digits, thousands separators. */
export const count = (n: number) => Number(n.toPrecision(3)).toLocaleString('en-US')

/** A pause length as the result tables write it: '9 months', '1.5 years'. */
export const duration = (years: number) => (years < 1 ? `${years * 12} months` : `${years} years`)

/** The month a date falls in, as the result tables write it: 05.2031. */
export function month(t: number) {
  const m = Math.floor(t * 12)
  return `${String((m % 12) + 1).padStart(2, '0')}.${Math.floor(m / 12)}`
}

/** A tick of a linear axis: 0, 500, 20k, 1.5M. */
export const people = (v: number) =>
  v < 1e3 ? String(v) : v < 1e6 ? `${+(v / 1e3).toPrecision(3)}k` : `${+(v / 1e6).toPrecision(3)}M`

/** A linear axis from 0: its top and step, the step 1, 2 or 5 times a power of ten, the fewest of at most `ticks`
 *  steps over `most`. */
export function axis(most: number, ticks = 5) {
  if (!Number.isFinite(most) || most <= 0) return { top: 1, step: 1 }
  for (let e = Math.floor(Math.log10(most / ticks)); ; e++) {
    for (const m of [1, 2, 5]) {
      const step = m * 10 ** e
      const steps = Math.max(1, Math.ceil(most / step))
      if (steps <= ticks) return { top: steps * step, step }
    }
  }
}

/** The ticks of a linear axis from 0 to `top`, a `step` apart. */
export const steps = ({ top, step }: { top: number; step: number }) =>
  Array.from({ length: Math.round(top / step) + 1 }, (_, i) => i * step)
