// The headings of AIFM's supplementary materials, a Google Doc, for the links in the symbol popups: writes
// content/supplement.json, {url, fetched, headings: {id: title}}. A link to a heading is <url>#heading=<id>, the link
// Google Docs copies for a heading. No dependencies beyond Node.
//
//   npm run supplement      (node scripts/supplement.mjs)
//
// Some headings share an id (the doc's tables have rows copied from other rows): Google Docs goes to the first, so
// only the first is kept, and a link to a later one shows the title of where it lands. A title loses its trailing
// symbol, "(η_init)" or "λ": the popup shows the symbol already, and the doc sets it with subscripts or an image.
import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const DOC = '1ru6Okbxb6XuH18Cz8439sdQJazMV39hNxsWDokh97r0'
const URL_ = `https://docs.google.com/document/d/${DOC}/edit?tab=t.0`
const OUT = fileURLToPath(new URL('../content/supplement.json', import.meta.url))

const GREEK = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu',
  'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega']
const ENTITIES = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', lsquo: '‘', rsquo: '’', ldquo: '“', rdquo: '”',
  ndash: '–', mdash: '—', hellip: '…', times: '×', minus: '−',
  ...Object.fromEntries(GREEK.flatMap((name, i) => {
    const skip = i >= 17 ? 1 : 0 // no capital final sigma: Σ is U+03A3, σ U+03C3
    return [[name, String.fromCodePoint(0x391 + i + skip)], [name.toLowerCase(), String.fromCodePoint(0x3b1 + i + skip)]]
  })),
}
const MARK = '￼' // stands for maths the text lost: an image, or the start of a subscript or superscript

function decode(html) {
  return html.replace(/&(#x[0-9a-f]+|#\d+|\w+);/gi, (all, name) => {
    if (name[0] === '#') return String.fromCodePoint(Number(name[1] === 'x' ? `0${name.slice(1)}` : name.slice(1)))
    if (!Object.hasOwn(ENTITIES, name)) throw new Error(`unknown entity ${all}: add it to ENTITIES`)
    return ENTITIES[name]
  })
}

/** Mathematical italic, bold etc. Greek (𝜆, U+1D706) as plain Greek (λ). */
const plainGreek = (text) =>
  text.replace(/[\u{1D6A8}-\u{1D7C9}]/gu, (c) => {
    // Five alphabets of 58: capitals Α–Ω with ϴ after Ρ, ∇, small α–ω with ς, ∂, six variant letters.
    const k = (c.codePointAt(0) - 0x1d6a8) % 58
    if (k < 25) return k === 17 ? 'ϴ' : String.fromCodePoint(0x391 + k)
    if (k > 25 && k < 51) return String.fromCodePoint(0x3b1 + k - 26)
    return c
  })

/** A symbol as a word of a title: a Greek letter or lost maths, then letters. */
const SYMBOL = new RegExp(`^[\\u0370-\\u03FF${MARK}][A-Za-z${MARK}]*$`)

function title(inner, scripted) {
  const text = decode(
    inner
      .replace(/<img\b[^>]*>/g, MARK)
      .replace(/<span class="([^"]*)">/g, (all, cls) => (cls.split(' ').some((c) => scripted.has(c)) ? MARK : ''))
      .replace(/<[^>]*>/g, ''),
  )
  return plainGreek(text)
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/ \(([^()]*)\)$/, (all, symbol) => (SYMBOL.test(symbol) ? '' : all))
    .replace(/ (\S+)$/, (all, symbol) => (SYMBOL.test(symbol) ? '' : all))
    .replaceAll(MARK, '')
}

const res = await fetch(`https://docs.google.com/document/d/${DOC}/export?format=html`)
if (!res.ok) throw new Error(`export failed: ${res.status} ${res.statusText}`)
const html = await res.text()

// Classes of subscripts and superscripts, from the export's stylesheet.
const scripted = new Set([...html.matchAll(/\.(c\d+)\{[^}]*vertical-align:(?:sub|super)/g)].map((m) => m[1]))
const headings = {}
const seen = new Set()
let shared = 0
for (const m of html.matchAll(/<h([1-6])\b[^>]*\bid="(h\.[^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g)) {
  const id = m[2].replace(/-\d+$/, '') // the export numbers repeated ids: h.e1zpd6153zmb-3
  if (seen.has(id)) {
    shared++
    continue
  }
  seen.add(id)
  const text = title(m[3], scripted)
  if (text) headings[id] = text // an empty heading or a lone image: nothing to link to
}
if (!Object.keys(headings).length) throw new Error('no headings found: has the export changed?')

const fetched = new Date().toISOString().slice(0, 10)
writeFileSync(OUT, `${JSON.stringify({ url: URL_, fetched, headings }, null, 2)}\n`)
console.log(`content/supplement.json: ${Object.keys(headings).length} headings (${shared} more share an id with one before)`)
