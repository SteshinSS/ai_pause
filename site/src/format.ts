// The result tables' numbers (ResultTable.tsx, and the PDF's tables, paper.tsx).

/** Three significant digits with thousands separators, as fmt() in pause.py. */
export function fmt(v: number) {
  const decimals = Math.max(0, 2 - Math.floor(Math.log10(Math.abs(v))))
  return Number(v.toPrecision(3)).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

/** People, three significant digits: 34.4k, 1.61M. */
export const people = (v: number) => (v === 0 ? '0' : v < 1e6 ? `${fmt(v / 1e3)}k` : `${fmt(v / 1e6)}M`)
