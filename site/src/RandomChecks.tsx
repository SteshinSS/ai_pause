import type { ReactNode } from 'react'
import data from '../content/random_checks.json'
import { count, duration } from './plot'

// ---------- verifiers who check the best papers first against a random sample, of content/random_checks.json
// (random_checks.py) ----------

const { years } = data.scenario
const best = data.cells['best first']
const random = data.cells.random
const OWN = 'own staff'
const HIRING = 'unlimited hiring'

/** A change as a percentage, to a tenth: −29.5%, +2.6%; none is 0%. */
function change(r: number) {
  const p = Math.abs(r) * 100
  if (p < 0.05) return '0%'
  return `${r < 0 ? '−' : '+'}${+p.toFixed(1)}%`
}

/** A share as a percentage: 100%, 18%. */
const share = (v: number) => `${Math.round(v * 100)}%`

/**
 * `<RandomChecks table="cells">` in an .mdx file: Compartmentalization's AI safety years when the verifiers check the
 * papers worth the most first and when they check a random sample, a column per pause length of random_checks.py
 * (content/random_checks.json), with frozen hiring and with unlimited hiring, then the people to hire. With
 * `table="years"`, the shares of the papers and of their research effort that the lab's own staff check, a row per
 * year of the longest pause. The children are the caption. In .mdx files it needs no import (Pane.tsx provides it).
 */
export function RandomChecks({ table, children }: { table: 'cells' | 'years'; children?: ReactNode }) {
  return (
    <figure className="result-table random-checks">
      {table === 'cells' ? <Cells /> : <Years />}
      {children && <figcaption>{children}</figcaption>}
    </figure>
  )
}

/** AI safety years and people to hire, a row per pause length. */
function Cells() {
  return (
    <>
      <div className="result-table-title">AI safety years</div>
      <div className="result-table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>
                Pause duration
              </th>
              <th scope="colgroup" colSpan={3} className="result-table-group">
                Frozen hiring
              </th>
              <th scope="colgroup" className="result-table-group">
                Unlimited hiring
              </th>
            </tr>
            <tr>
              <th scope="col">Best papers first</th>
              <th scope="col">Random sample</th>
              <th scope="col">Change</th>
              <th scope="col">Every paper</th>
            </tr>
          </thead>
          <tbody>
            {years.map((D, i) => (
              <tr key={D}>
                <th scope="row">{duration(D)}</th>
                <td>{count(best[OWN][i])}</td>
                <td>{count(random[OWN][i])}</td>
                <td>{change(random[OWN][i] / best[OWN][i] - 1)}</td>
                <td>{count(best[HIRING][i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="result-table-title">People to hire</div>
      <div className="result-table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col" rowSpan={1}>
                Pause duration
              </th>
              <th scope="col">To check every paper</th>
              <th scope="col">For a random sample to match the best papers first</th>
            </tr>
          </thead>
          <tbody>
            {years.map((D, i) => (
              <tr key={D}>
                <th scope="row">{duration(D)}</th>
                <td>{count(data.hired[i])}</td>
                <td>{count(data.match[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

/** The shares the lab's own staff check, a row per year of the longest pause. */
function Years() {
  const { t } = data.yearly
  const ranked = data.yearly['best first']
  const sample = data.yearly.random
  return (
    <div className="result-table-scroll">
      <table>
        <thead>
          <tr>
            <th scope="col" rowSpan={2}>
              Years into the pause
            </th>
            <th scope="colgroup" colSpan={2} className="result-table-group">
              Papers checked
            </th>
            <th scope="colgroup" colSpan={2} className="result-table-group">
              Research effort checked
            </th>
          </tr>
          <tr>
            <th scope="col">Best papers first</th>
            <th scope="col">Random sample</th>
            <th scope="col">Best papers first</th>
            <th scope="col">Random sample</th>
          </tr>
        </thead>
        <tbody>
          {t.map((year, i) => (
            <tr key={year}>
              <th scope="row">{year}</th>
              <td>{share(ranked.q[i])}</td>
              <td>{share(sample.q[i])}</td>
              <td>{share(ranked.counted[i])}</td>
              <td>{share(sample.counted[i])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
