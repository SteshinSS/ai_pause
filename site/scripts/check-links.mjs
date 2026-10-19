// Checks content/ before a build: every file has a title, note and popup ids are URL-safe, every
// [text](note:id) link points to an existing note (and [text](note:id#address) to a heading of it, headings.ts), every
// row of the result table (content/table.json) has its popup, content/milestones/<row name in lower case>.mdx, and
// every trust model has its menu's, content/protocols/<model with a dash for each space>.mdx, and every † of
// content/marks/ names in its frontmatter a trust model, a row and pause durations of content/table.json. Fails the
// build on any of these.
// Every <Plate id="…" />, <Mark id="…" />, <Backdrop id="…" />, <Doodle id="…" /> and <Night id="…"> has its drawing,
// src/art/<id>.webp or .svg. Every <ResultTable model="…"> names a table of content/table.json, every <HiringPlot>
// has a model="…" of its hiring, every <HiringCurve> has its hiring_curves, every <AuditorCurve> its auditor_curves,
// every <Tornado set="…"> a set of content/sensitivity.json whose inputs its <span data-input="…"> name (all of them
// for set="ours"), every <RandomChecks table="…"> a table it draws from content/random_checks.json, every
// <SlowScaling> its content/slow_scaling.json, every <Value input="…" /> an input or AIFM parameter of it, every
// <Range input="…" /> an input with a p10 and a p90 in content/sensitivity.json, and every
// `companion:` in a frontmatter names a note. In content/symbols.mdx, every input="…" is an input or AIFM parameter of
// content/table.json, every under="…" a heading's text and every id in supplement="…" a heading of
// content/supplement.json.
// Notes that nothing links to, popups of no row or model and entries of content/references.bib that nothing cites are
// only reported, as are headings of a note that its companion lacks. (Citations of keys the .bib doesn't have fail in
// citations.ts.)
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { basename, extname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const site = fileURLToPath(new URL('..', import.meta.url))
const content = join(site, 'content')
/** The .mdx files of content/<dir>/ by id, the file name. */
const folder = (dir) =>
  Object.fromEntries(
    readdirSync(join(content, dir))
      .filter((f) => f.endsWith('.mdx'))
      .map((f) => [basename(f, '.mdx'), join(content, dir, f)]),
  )
const notes = folder('notes')
const milestones = folder('milestones')
const protocols = folder('protocols')
const marks = folder('marks')
const files = [join(content, 'main.mdx'), ...[notes, milestones, protocols, marks].flatMap(Object.values)]

const artDir = join(site, 'src/art')
const art = new Set(existsSync(artDir) ? readdirSync(artDir).map((f) => basename(f, extname(f))) : [])

const table = JSON.parse(readFileSync(join(content, 'table.json'), 'utf8'))
const rows = table.rows.map((r) => r.name)
const models = Object.keys(table.rows[0].cells)
const hiring = [...new Set(table.hiring.rows.flatMap((r) => Object.keys(r.hired)))]
const values = { ...table.inputs, ...table.aifm_parameters }
const sensitivityFile = join(content, 'sensitivity.json')
const sensitivity = existsSync(sensitivityFile) ? JSON.parse(readFileSync(sensitivityFile, 'utf8')) : {}
const randomChecks = existsSync(join(content, 'random_checks.json'))
const slowScaling = existsSync(join(content, 'slow_scaling.json'))

/** The addresses of the headings of a file, as headings.ts gives them (it can't be imported here): each heading's text
 *  as a slug, with the slugs of the headings it sits under in front, as many as tell it apart from the file's other
 *  headings. Markdown headings outside code only, read with their links' and footnotes' syntax dropped. */
function addressesOf(file) {
  const slug = (text) =>
    text
      .toLowerCase()
      .replace(/['’]/g, '')
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  const src = readFileSync(file, 'utf8')
    .replace(/^---[\s\S]*?\n---/, '')
    .replace(/^(```|~~~)[\s\S]*?^\1/gm, '') // a # in code is no heading
  const found = [] // each heading's slug and those of the headings it sits under, its own last
  const outline = []
  for (const m of src.matchAll(/^(#{1,6})[ \t]+(.+?)[ \t]*$/gm)) {
    const text = m[2].replace(/\[\^[^\]]*\]/g, '').replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    while (outline.length && outline.at(-1).depth >= m[1].length) outline.pop()
    outline.push({ depth: m[1].length, slug: slug(text) })
    found.push(outline.map((h) => h.slug))
  }
  const end = (path, k) => path.slice(-k).join('\n')
  const used = new Set()
  for (const path of found) {
    let k = 1
    while (k < path.length && found.some((f) => f !== path && end(f, k) === end(path, k))) k++
    const base = path.slice(-k).join('-') || 'section'
    let address = base
    for (let i = 2; used.has(address); i++) address = `${base}-${i}`
    used.add(address)
  }
  return used
}

const PLATE = /<(Plate|Mark|Backdrop|Doodle|Night)\s[^>]*\bid=["']([^"']*)["']/g
const MODEL = /<ResultTable\s[^>]*\bmodel=["']([^"']*)["']/g
const HIRING = /<HiringPlot\b[^>]*>/g
const CURVE = /<HiringCurve\b/g
const AUDITORS = /<AuditorCurve\b/g
const TORNADO = /<Tornado\b([^>]*?)(?:\/>|>([\s\S]*?)<\/Tornado>)/g
const RANDOM = /<RandomChecks\b([^>]*)>/g
const SLOW = /<SlowScaling\b/g
const VALUE = /<Value\s[^>]*\binput=["']([^"']*)["']/g
const RANGE = /<Range\s[^>]*\binput=["']([^"']*)["']/g
const LINK = /\]\(note:([^)\s]*)\)|href=["']note:([^"']*)["']/g
const CITE = /(?<![\w@\\])@(\w(?:[\w:.#$%&+?<>~/-]*\w)?)/g // as in citations.ts
const errors = []
const linked = new Set()
const cited = new Set()
const companions = [] // [file of a note, file of its companion]

for (const [dir, docs] of Object.entries({ notes, milestones, protocols, marks })) {
  for (const id of Object.keys(docs)) {
    if (!/^[a-z0-9-]+$/.test(id)) errors.push(`content/${dir}/${id}.mdx: use a-z, 0-9 and - in file names`)
  }
}

for (const file of files) {
  const where = relative(site, file)
  const src = readFileSync(file, 'utf8')
  const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---/.exec(src)?.[1] ?? ''
  if (!/^title:\s*\S/m.test(frontmatter)) errors.push(`${where}: no "title:" in the frontmatter`)
  for (const m of src.matchAll(CITE)) cited.add(m[1])
  for (const m of src.matchAll(PLATE)) {
    const line = src.slice(0, m.index).split('\n').length
    if (!art.has(m[2])) errors.push(`${where}:${line}: <${m[1]} id="${m[2]}"> has no src/art/${m[2]}.webp or .svg`)
  }
  for (const m of src.matchAll(MODEL)) {
    const line = src.slice(0, m.index).split('\n').length
    if (!models.includes(m[1])) errors.push(`${where}:${line}: <ResultTable model="${m[1]}">: table.json has ${models.join(', ')}`)
  }
  for (const m of src.matchAll(HIRING)) {
    const line = src.slice(0, m.index).split('\n').length
    const model = /\bmodel=["']([^"']*)["']/.exec(m[0])?.[1]
    if (!hiring.includes(model)) errors.push(`${where}:${line}: <HiringPlot model="${model ?? ''}">: table.json's hiring has ${hiring.join(', ')}`)
  }
  for (const m of src.matchAll(CURVE)) {
    const line = src.slice(0, m.index).split('\n').length
    if (!table.hiring_curves?.scenarios?.length) errors.push(`${where}:${line}: <HiringCurve>: table.json has no hiring_curves`)
  }
  for (const m of src.matchAll(AUDITORS)) {
    const line = src.slice(0, m.index).split('\n').length
    if (!table.auditor_curves?.rows?.length) errors.push(`${where}:${line}: <AuditorCurve>: table.json has no auditor_curves`)
  }
  for (const m of src.matchAll(TORNADO)) {
    const line = src.slice(0, m.index).split('\n').length
    const set = /\bset=["']([^"']*)["']/.exec(m[1])?.[1]
    const items = ['ours', 'aifm'].includes(set) ? sensitivity[set] : undefined
    if (!Array.isArray(items)) {
      errors.push(`${where}:${line}: <Tornado set="${set ?? ''}">: content/sensitivity.json has no such set (ours, aifm)`)
      continue
    }
    const names = items.map((it) => it.name)
    const given = [...(m[2] ?? '').matchAll(/\bdata-input=["']([^"']*)["']/g)].map((g) => g[1])
    for (const name of given) {
      if (!names.includes(name)) errors.push(`${where}:${line}: <Tornado set="${set}">: ${name} is no input of its set`)
    }
    for (const name of set === 'ours' ? names : []) {
      if (!given.includes(name)) errors.push(`${where}:${line}: <Tornado set="ours">: no <span data-input="${name}">`)
    }
  }
  for (const m of src.matchAll(RANDOM)) {
    const line = src.slice(0, m.index).split('\n').length
    const kind = /\btable=["']([^"']*)["']/.exec(m[1])?.[1]
    if (!['cells', 'years'].includes(kind)) {
      errors.push(`${where}:${line}: <RandomChecks table="${kind ?? ''}">: it draws table="cells" or table="years"`)
    } else if (!randomChecks) {
      errors.push(`${where}:${line}: <RandomChecks>: no content/random_checks.json (python random_checks.py --json)`)
    }
  }
  for (const m of src.matchAll(SLOW)) {
    const line = src.slice(0, m.index).split('\n').length
    if (!slowScaling) errors.push(`${where}:${line}: <SlowScaling>: no content/slow_scaling.json (python slow_scaling.py --json)`)
  }
  for (const m of src.matchAll(VALUE)) {
    const line = src.slice(0, m.index).split('\n').length
    if (!Object.hasOwn(values, m[1])) {
      errors.push(`${where}:${line}: <Value input="${m[1]}">: not in content/table.json (inputs, aifm_parameters)`)
    }
  }
  for (const m of src.matchAll(RANGE)) {
    const line = src.slice(0, m.index).split('\n').length
    if (!(sensitivity.ours ?? []).some((it) => it.name === m[1])) {
      errors.push(`${where}:${line}: <Range input="${m[1]}">: no p10 and p90 in content/sensitivity.json (ours)`)
    }
  }
  for (const m of src.matchAll(LINK)) {
    const [id, section] = (m[1] ?? m[2]).split('#')
    const line = src.slice(0, m.index).split('\n').length
    if (!Object.hasOwn(notes, id)) errors.push(`${where}:${line}: link to missing note "${id}"`)
    else {
      linked.add(id)
      if (section && !addressesOf(notes[id]).has(section)) {
        errors.push(`${where}:${line}: link to note:${id}#${section}: no heading of the note has that address`)
      }
    }
  }
  const companion = /^companion:\s*(\S+)/m.exec(frontmatter)?.[1]
  if (companion && Object.hasOwn(notes, companion)) {
    linked.add(companion)
    companions.push([file, notes[companion]])
  } else if (companion) errors.push(`${where}: companion "${companion}" is not a note`)
}

// A note and its companion scroll side by side, lined up by headings with the same level and text in the same order
// (src/sync.ts). Report the note's headings that don't find theirs.
const headingsOf = (file) =>
  [...readFileSync(file, 'utf8').replace(/^---[\s\S]*?\n---/, '').matchAll(/^(#{1,6})[ \t]+(.+?)[ \t]*$/gm)].map(
    (m) => `${m[1]} ${m[2]}`,
  )
/** Items of a outside a longest common subsequence of a and b. */
function unmatched(a, b) {
  const len = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0))
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      len[i][j] = a[i] === b[j] ? len[i + 1][j + 1] + 1 : Math.max(len[i + 1][j], len[i][j + 1])
    }
  }
  const out = []
  for (let i = 0, j = 0; i < a.length; ) {
    if (j < b.length && a[i] === b[j]) {
      i++
      j++
    } else if (j < b.length && len[i + 1][j] < len[i][j + 1]) j++
    else out.push(a[i++])
  }
  return out
}
for (const [file, other] of companions) {
  for (const h of unmatched(headingsOf(file), headingsOf(other))) {
    console.warn(`warning: ${relative(site, file)}: "${h}" has no match in its companion ${relative(site, other)}`)
  }
}

// The symbol popups (symbols.ts): values come from table.json, and `under` limits an entry to formulas under a heading.
// Links go to headings of AIFM's supplementary materials, content/supplement.json.
const symbols = join(content, 'symbols.mdx')
if (existsSync(symbols)) {
  const headings = new Set(files.flatMap((f) => headingsOf(f).map((h) => h.replace(/^#+ /, ''))))
  const sections = JSON.parse(readFileSync(join(content, 'supplement.json'), 'utf8')).headings
  const src = readFileSync(symbols, 'utf8')
  for (const m of src.matchAll(/<Symbol\s[^>]*>/g)) {
    const line = src.slice(0, m.index).split('\n').length
    const input = /\binput=["']([^"']*)["']/.exec(m[0])?.[1]
    const under = /\bunder=["']([^"']*)["']/.exec(m[0])?.[1]
    const supplement = /\bsupplement=["']([^"']*)["']/.exec(m[0])?.[1]
    if (input !== undefined && !Object.hasOwn(values, input)) {
      errors.push(`content/symbols.mdx:${line}: input="${input}" is not in content/table.json (inputs, aifm_parameters)`)
    }
    if (under !== undefined && !headings.has(under)) {
      errors.push(`content/symbols.mdx:${line}: under="${under}": no heading has this text`)
    }
    for (const id of supplement?.split(/\s+/).filter(Boolean) ?? []) {
      if (!Object.hasOwn(sections, id)) {
        errors.push(`content/symbols.mdx:${line}: supplement: ${id} is not a heading of content/supplement.json `
          + '(npm run supplement refreshes it)')
      }
    }
  }
}

for (const row of rows) {
  const id = row.toLowerCase()
  if (!Object.hasOwn(milestones, id)) errors.push(`content/table.json: row "${row}" has no content/milestones/${id}.mdx`)
}

for (const model of models) {
  const id = model.replaceAll(' ', '-')
  if (!Object.hasOwn(protocols, id)) {
    errors.push(`content/table.json: trust model "${model}" has no content/protocols/${id}.mdx`)
  }
}

for (const [id, file] of Object.entries(marks)) {
  const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---/.exec(readFileSync(file, 'utf8'))?.[1] ?? ''
  const field = (name) => new RegExp(`^${name}:\\s*(.*?)\\s*$`, 'm').exec(frontmatter)?.[1]
  const where = `content/marks/${id}.mdx`
  const model = field('model')
  const row = field('row')
  const years = /^\[(.*)\]$/.exec(field('years') ?? '')?.[1].split(',').map(Number) ?? []
  if (!models.includes(model)) errors.push(`${where}: model "${model ?? ''}": table.json has ${models.join(', ')}`)
  if (!rows.includes(row)) errors.push(`${where}: row "${row ?? ''}": table.json has ${rows.join(', ')}`)
  if (!years.length || years.some((y) => !table.durations_years.includes(y))) {
    errors.push(`${where}: years [${years.join(', ')}]: table.json has ${table.durations_years.join(', ')}`)
  }
}

for (const id of Object.keys(notes)) {
  if (!linked.has(id)) console.warn(`warning: nothing links to content/notes/${id}.mdx`)
}
for (const id of Object.keys(protocols)) {
  if (!models.some((model) => model.replaceAll(' ', '-') === id)) {
    console.warn(`warning: content/protocols/${id}.mdx is the popup of no trust model of content/table.json`)
  }
}
for (const id of Object.keys(milestones)) {
  if (!rows.some((row) => row.toLowerCase() === id)) {
    console.warn(`warning: content/milestones/${id}.mdx is the popup of no row of content/table.json`)
  }
}

const bib = readFileSync(join(content, 'references.bib'), 'utf8')
for (const [, key] of bib.matchAll(/^\s*@(?!comment|string|preamble)\w+\s*\{\s*([^,\s]+)\s*,/gim)) {
  if (!cited.has(key)) console.warn(`warning: nothing cites ${key} of content/references.bib`)
}

if (errors.length) {
  console.error(errors.join('\n'))
  process.exit(1)
}
console.log(`content ok: main + ${Object.keys(notes).length} notes + ${rows.length} table rows with popups`)
