import type { Element, Root as HastRoot, Text } from 'hast'
import katex, { type KatexOptions } from 'katex'
import type { Nodes, Paragraph, Root, RootContent } from 'mdast'
import type { MdxJsxAttribute, MdxJsxFlowElement, MdxJsxTextElement } from 'mdast-util-mdx-jsx'
import { readFileSync, statSync } from 'node:fs'
import { relative } from 'node:path'
import type { Processor } from 'unified'
import { VFile } from 'vfile'

/** KaTeX options of the site: remarkSymbols marks symbols with \htmlData, the one HTML extension it allows. */
export const KATEX_OPTIONS: KatexOptions = {
  trust: (context) => context.command === '\\htmlData',
  strict: (code: string) => (code === 'htmlExtension' ? 'ignore' : 'warn'),
}

/**
 * Popups for the symbols of formulas. In a file with `symbols: true` in its frontmatter, every symbol of its maths
 * that has an entry in the symbols file (content/symbols.mdx) is marked: `C_{train}` becomes
 * `\htmlData{sym=C-train}{C_{train}}`, which KaTeX renders as a span with `data-sym="C-train"` around the same
 * glyphs, and src/Symbols.tsx shows the entry's popup. A symbol whose `input` is one of ours (pause_inputs.csv, the
 * `inputs` of content/table.json) gets `ours=1` too, `data-ours="1"`, which styles.css colours. The formulas stay
 * plain LaTeX. A formula whose marked version KaTeX would draw differently in any way keeps no marks, with a warning;
 * letters without an entry get a warning too. Maths in a link's text keeps no marks: a click there follows the link.
 * Maths in a footnote is marked like the rest, as footnotes.ts has put its text where it is referred to.
 *
 * The symbols file itself becomes the popups' content: its <Symbol> entries with an `id` and the symbol set in
 * maths on top; everything else in it (headings, notes to self) is left out.
 */
export function remarkSymbols(this: Processor, { symbols, values }: { symbols: string; values: string }) {
  const name = relative(process.cwd(), symbols) // for messages
  return (tree: Root, file: VFile) => {
    if (file.path === symbols) return entryPopups(tree, load(this, symbols, values, file), file)
    if (!/^symbols:\s*true\s*$/m.test(frontmatter(tree))) return
    const table = load(this, symbols, values, file)
    const where = relative(process.cwd(), file.path)
    const sources: Record<string, string> = ((file.data as { symbolSources?: Record<string, string> }).symbolSources =
      {})
    const unknown = new Map<string, string>() // name -> source of atoms without an entry
    const headings: string[] = [] // texts of the headings the node is under, by depth

    const visit = (node: Nodes) => {
      if (node.type === 'heading') {
        headings.length = node.depth - 1
        headings[node.depth - 1] = toString(node)
        return
      }
      if (node.type === 'link' || node.type === 'linkReference') return // a click there follows the link
      if (node.type === 'math' || node.type === 'inlineMath') {
        const marked = markFormula(node.value, node.type === 'math', finder(table, headings.filter(Boolean)), unknown)
        if (marked === undefined) {
          console.warn(`warning: ${where}: no symbol popups in $${node.value.replace(/\s+/g, ' ')}$: KaTeX would draw `
            + 'it differently with the marks. Braces around one-letter arguments help: \\frac{1}{\\alpha}')
        } else if (marked !== node.value) {
          sources[marked] = node.value
          const text: Text = { type: 'text', value: marked }
          const data = node.data as { hChildren: Element[] }
          if (node.type === 'math') data.hChildren[0].children = [text]
          else data.hChildren = [text as unknown as Element]
          node.value = marked
        }
        return
      }
      if ('children' in node) node.children.forEach(visit)
    }
    visit(tree)
    for (const [symbol, source] of unknown) {
      if (!table.skip.has(symbol)) console.warn(`warning: ${where}: no entry in ${name} for $${source}$`)
    }
  }
}

/**
 * KaTeX writes each formula's LaTeX into its MathML, for screen readers and copying: put back the formula as written,
 * without the marks of remarkSymbols. Runs after rehype-katex.
 */
export function rehypeSymbolSources() {
  return (tree: HastRoot, file: VFile) => {
    const sources = (file.data as { symbolSources?: Record<string, string> }).symbolSources
    if (!sources) return
    // Any node with children: formulas also sit in MDX elements, such as <div className="unused">.
    const visit = (node: { children?: unknown[] }) => {
      for (const child of (node.children ?? []) as (Element | Text)[]) {
        const text = child.type === 'element' ? child.children[0] : undefined
        if (child.type === 'element' && child.tagName === 'annotation' && text?.type === 'text') {
          if (Object.hasOwn(sources, text.value)) text.value = sources[text.value]
        } else if ('children' in child) visit(child)
      }
    }
    visit(tree)
  }
}

// ---------- the symbols file ----------

type Entry = {
  /** `data-sym` of the entry's symbols in formulas and `id` of its <Symbol>: 'C-train', 'k.automation'. */
  id: string
  tex: string
  /** Only for formulas under a heading with this text. */
  under?: string
  /** Names of its atoms (formulaNames): one, or more for a name of several letters such as RS_0. */
  names: string[]
  /** Offset of its <Symbol> in the symbols file. */
  offset: number
  /** Its `input` is one of ours, an input of pause_inputs.csv, not one of AIFM's parameters. */
  ours: boolean
}

type Table = {
  entries: Entry[]
  /** Names that are not symbols (`skip:` in the frontmatter): no warning for them. */
  skip: Set<string>
}

type SymbolNode = MdxJsxFlowElement | MdxJsxTextElement

const isSymbol = (node: Nodes): node is SymbolNode =>
  (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') && node.name === 'Symbol'

/** The <Symbol> elements of a tree, in order, wherever they are. */
function symbolNodes(node: Nodes, out: SymbolNode[] = []): SymbolNode[] {
  if (isSymbol(node)) out.push(node)
  else if ('children' in node) for (const child of node.children) symbolNodes(child, out)
  return out
}

const frontmatter = (tree: Root) => {
  const node = tree.children.find((n) => n.type === 'yaml')
  return node?.type === 'yaml' ? node.value : ''
}

let cache: { key: string; table: Table } | undefined

/**
 * The entries of the symbols file, parsed as the build parses it (so offsets match), re-read when it or `values`
 * (content/table.json, whose `inputs` are ours) changes.
 */
function load(processor: Processor, path: string, values: string, file: VFile): Table {
  const name = relative(process.cwd(), path)
  const key = `${path} ${statSync(path).mtimeMs} ${statSync(values).mtimeMs}`
  if (cache?.key === key) return cache.table
  const ours = new Set(Object.keys((JSON.parse(readFileSync(values, 'utf8')) as { inputs: object }).inputs))
  const tree = processor.parse(new VFile({ path, value: readFileSync(path, 'utf8') })) as Root
  const entries: Entry[] = []
  const seen = new Set<string>()
  for (const node of symbolNodes(tree)) {
    const at = `${name}:${node.position?.start.line}`
    const tex = attribute(node, 'tex', at, file)
    const under = attribute(node, 'under', at, file)
    const input = attribute(node, 'input', at, file)
    if (tex === undefined) file.fail(`${at}: <Symbol> without tex="…"`, { source: 'symbols' })
    const names = formulaNames(tex)
    if (!names) file.fail(`${at}: tex="${tex}" is not a symbol: letters, Greek letters, accents, _ and ^`, { source: 'symbols' })
    const key = `${names.join(' ')} @${under ?? ''}`
    if (seen.has(key)) file.fail(`${at}: a second entry for ${tex}${under ? ` under "${under}"` : ''}`, { source: 'symbols' })
    seen.add(key)
    let id = slug(tex) + (under ? `.${slug(under).toLowerCase()}` : '')
    while (entries.some((e) => e.id === id)) id += '_'
    entries.push({ id, tex, under, names, offset: node.position!.start.offset!, ours: ours.has(input ?? '') })
  }
  const skipped = /^skip:\s*(.*)$/m.exec(frontmatter(tree))?.[1] ?? ''
  const skip = new Set(skipped.split(/[\s,[\]]+/).filter(Boolean).map((s) => formulaNames(s)?.join('') ?? s))
  cache = { key, table: { entries, skip } }
  return cache.table
}

function attribute(node: SymbolNode, key: string, at: string, file: VFile): string | undefined {
  const attr = node.attributes.find((a): a is MdxJsxAttribute => a.type === 'mdxJsxAttribute' && a.name === key)
  if (attr && typeof attr.value !== 'string') file.fail(`${at}: write ${key}="…" as a string`, { source: 'symbols' })
  return attr?.value as string | undefined
}

const slug = (tex: string) => tex.replace(/\\/g, '').replace(/[^A-Za-z0-9]+/g, '-').replace(/^-|-$/g, '')

/**
 * The symbols file as popups: only its entries, each with an `id` and a first paragraph (class symbol-tex) that sets
 * its symbol in maths. Entries on one line (text elements) become blocks, so no empty paragraph is left behind.
 */
function entryPopups(tree: Root, table: Table, file: VFile) {
  const byOffset = new Map(table.entries.map((e) => [e.offset, e]))
  const entries = symbolNodes(tree).map((node): MdxJsxFlowElement => {
    const entry = byOffset.get(node.position!.start.offset!)
    if (!entry) return file.fail('a <Symbol> the symbols table lacks', { place: node.position, source: 'symbols' })
    const title: Paragraph = {
      type: 'paragraph',
      data: { hProperties: { className: entry.ours ? ['symbol-tex', 'symbol-ours'] : ['symbol-tex'] } },
      children: [
        {
          type: 'inlineMath',
          value: entry.tex,
          data: {
            hName: 'code',
            hProperties: { className: ['language-math', 'math-inline'] },
            hChildren: [{ type: 'text', value: entry.tex }],
          },
        },
      ],
    }
    const body = node.type === 'mdxJsxTextElement'
      ? node.children.length ? [{ type: 'paragraph', children: node.children } as Paragraph] : []
      : node.children
    return {
      type: 'mdxJsxFlowElement',
      name: 'Symbol',
      attributes: [
        ...node.attributes.filter((a) => !(a.type === 'mdxJsxAttribute' && a.name === 'id')),
        { type: 'mdxJsxAttribute', name: 'id', value: entry.id },
      ],
      children: [title, ...body],
    }
  })
  tree.children = [...tree.children.filter((n) => n.type === 'yaml' || n.type === 'mdxjsEsm'), ...entries] as RootContent[]
}

function toString(node: Nodes): string {
  if ('value' in node && typeof node.value === 'string') return node.value
  return 'children' in node ? node.children.map(toString).join('') : ''
}

// ---------- LaTeX: just enough to find the symbols of a formula ----------

type Token = { text: string; start: number; end: number }

/** A group's inside or a single token: the argument of a command, _ or ^. */
type Arg = { from: number; to: number; next: number; bare: boolean }

/**
 * A letter, Greek letter or accented letter with its subscript, superscript and primes: tokens from..next, the
 * letter itself from..baseNext.
 */
type Atom = { from: number; baseNext: number; next: number; base: string; sub?: Arg; sup?: Arg; primes: number }

const GREEK = new Set(
  ('alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho varrho '
    + 'sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega '
    + 'ell').split(' ').map((n) => `\\${n}`),
)
const ACCENTS = new Set(['\\tilde', '\\bar', '\\hat', '\\check', '\\breve', '\\dot', '\\ddot', '\\vec', '\\widetilde', '\\widehat', '\\overline'])
/**
 * Commands whose arguments are words, names or lengths, never symbols, by how many: skipped whole (so the E of
 * \mathbb{E} is no symbol). \begin also skips the column spec of an array.
 */
const OPAQUE = new Map([
  ...['\\text', '\\textrm', '\\textit', '\\textbf', '\\mathrm', '\\mathit', '\\mathbf', '\\mathsf', '\\mathtt',
    '\\mathbb', '\\mathcal', '\\mathfrak', '\\mathscr', '\\operatorname', '\\end', '\\tag', '\\label', '\\color',
    '\\hspace', '\\vspace', '\\htmlData', '\\htmlClass', '\\htmlId', '\\htmlStyle'].map((c) => [c, 1] as const),
  ['\\begin', 1],
  ['\\rule', 2],
  ['\\raisebox', 2],
])
const ARRAYS = /^(array|darray|subarray|alignedat|alignat)\*?$/
const isLetter = (text: string) => /^[A-Za-z]$/.test(text) || GREEK.has(text)

class Formula {
  readonly tex: string
  readonly tokens: Token[] = []
  private readonly close: number[] = [] // index of the } of each {

  constructor(tex: string) {
    this.tex = tex
    // Control words, control symbols, characters; spaces and % comments dropped.
    for (const m of tex.matchAll(/\\[A-Za-z]+|\\[^A-Za-z]|%[^\n]*|\s+|[^]/g)) {
      if (!/^[\s%]/.test(m[0])) this.tokens.push({ text: m[0], start: m.index, end: m.index + m[0].length })
    }
    const open: number[] = []
    this.tokens.forEach((t, i) => {
      if (t.text === '{') open.push(i)
      else if (t.text === '}' && open.length) this.close[open.pop()!] = i
    })
  }

  arg(i: number, to: number): Arg | undefined {
    const text = this.tokens[i]?.text
    if (i >= to || text === undefined || ['}', '_', '^', "'"].includes(text)) return
    if (text !== '{') return { from: i, to: i + 1, next: i + 1, bare: true }
    const close = this.close[i]
    return close === undefined || close >= to ? undefined : { from: i + 1, to: close, next: close + 1, bare: false }
  }

  atom(i: number, to: number): Atom | undefined {
    const text = this.tokens[i]?.text
    if (i >= to || text === undefined) return
    let base: string
    let next: number
    if (isLetter(text)) {
      base = text
      next = i + 1
    } else if (ACCENTS.has(text)) {
      const arg = this.arg(i + 1, to)
      if (!arg || arg.to - arg.from !== 1 || !isLetter(this.tokens[arg.from].text)) return
      base = `${text}{${this.tokens[arg.from].text}}`
      next = arg.next
    } else return
    const atom: Atom = { from: i, baseNext: next, next, base, primes: 0 }
    for (;;) {
      const script = next < to ? this.tokens[next].text : undefined
      if (script === "'") {
        atom.primes++
        next++
      } else if ((script === '_' && !atom.sub) || (script === '^' && !atom.sup)) {
        const arg = this.arg(next + 1, to)
        if (!arg) break
        atom[script === '_' ? 'sub' : 'sup'] = arg
        next = arg.next
      } else break
    }
    atom.next = next
    return atom
  }

  /** 'C_{t r a i n}': tokens joined by spaces, so that spacing and optional braces in the source don't matter. */
  name(atom: Atom, withSuperscript: boolean): string {
    const text = (arg: Arg) => this.tokens.slice(arg.from, arg.to).map((t) => t.text).join(' ')
    let name = atom.base
    if (atom.sub) name += `_{${text(atom.sub)}}`
    if (withSuperscript && atom.sup) name += `^{${text(atom.sup)}}`
    return withSuperscript ? name + "'".repeat(atom.primes) : name
  }

  /** The source of an atom without its superscript, for messages. */
  source(atom: Atom): string {
    const last = (atom.sub?.next ?? atom.baseNext) - 1
    return this.tex.slice(this.tokens[atom.from].start, this.tokens[last].end)
  }

  /** The token after command `i` and its arguments, if it is one whose arguments hold no symbols. */
  skipOpaque(i: number, to: number): number | undefined {
    const text = this.tokens[i].text
    if (text === '\\\\' && this.tokens[i + 1]?.text === '[') {
      // \\[4pt]: a line break with extra space
      const end = this.tokens.findIndex((t, j) => j > i && t.text === ']')
      return end < 0 || end >= to ? undefined : end + 1
    }
    const count = OPAQUE.get(text)
    if (count === undefined) return
    let next = i + 1
    for (let k = 0; k < count; k++) next = this.arg(next, to)?.next ?? next
    if (text === '\\begin' && ARRAYS.test(this.tex.slice(this.tokens[i + 2]?.start, this.tokens[next - 2]?.end))) {
      next = this.arg(next, to)?.next ?? next
    }
    return next
  }
}

/** The atom names of a symbol written in LaTeX ('RS_0' is R and S_{0}); undefined if it isn't only atoms. */
function formulaNames(tex: string | undefined): string[] | undefined {
  if (!tex) return
  const f = new Formula(tex)
  const names: string[] = []
  for (let i = 0; i < f.tokens.length; ) {
    const atom = f.atom(i, f.tokens.length)
    if (!atom) return
    names.push(f.name(atom, true))
    i = atom.next
  }
  return names.length ? names : undefined
}

type Finder = {
  /** The entry of a one-atom name under the current headings. */
  one: (name: string) => Entry | undefined
  /** Entries of several atoms under the current headings, longest first. */
  many: Entry[]
}

/** Entries that apply under these headings: one with `under` naming one of them (the innermost wins), else one without. */
function finder(table: Table, headings: string[]): Finder {
  const applies = (e: Entry) => !e.under || headings.includes(e.under)
  const rank = (e: Entry) => (e.under ? headings.lastIndexOf(e.under) : -1)
  const best = (list: Entry[]) => list.filter(applies).sort((a, b) => rank(b) - rank(a))[0]
  const one = new Map<string, Entry[]>()
  for (const e of table.entries) if (e.names.length === 1) one.set(e.names[0], [...(one.get(e.names[0]) ?? []), e])
  const many = table.entries.filter((e) => e.names.length > 1 && applies(e)).sort((a, b) => b.names.length - a.names.length)
  return { one: (name) => best(one.get(name) ?? []), many }
}

type Edit = { at: number; open: boolean; marked: string; plain: string }

/**
 * The formula with its symbols marked, the same if it has none, undefined if KaTeX would draw the marked version
 * differently. A symbol is an atom whose name, with or without its superscript, has an entry: with, as in
 * T_{AI}^{SD}, the whole atom is the symbol; without, as in C_{xpm}^{\zeta}, the superscript is an exponent, and its
 * own symbols are marked inside the mark. Subscripts are part of the name and never searched. Atoms without an
 * entry go to `unknown`, by their source.
 */
function markFormula(tex: string, display: boolean, find: Finder, unknown: Map<string, string>): string | undefined {
  const f = new Formula(tex)
  const edits: Edit[] = []
  const wrap = (start: number, end: number, entry: Entry, bare: boolean) => {
    // A single-token argument (x^\alpha) gets braces: \htmlData can't be one unbraced.
    const data = `sym=${entry.id}${entry.ours ? ', ours=1' : ''}`
    edits.push(
      { at: start, open: true, marked: `${bare ? '{' : ''}\\htmlData{${data}}{`, plain: bare ? '{' : '' },
      { at: end, open: false, marked: bare ? '}}' : '}', plain: bare ? '}' : '' },
    )
  }
  const script = (arg: Arg | undefined) => arg && scan(arg.from, arg.to, arg.bare)

  const scan = (from: number, to: number, bare: boolean) => {
    for (let i = from; i < to; ) {
      const text = f.tokens[i].text
      if (text === '{') {
        const arg = f.arg(i, to)
        if (!arg) return
        scan(arg.from, arg.to, false)
        i = arg.next
        continue
      }
      const skip = f.skipOpaque(i, to)
      if (skip !== undefined) {
        i = skip
        continue
      }
      if (text === '_' || text === '^') {
        const arg = f.arg(i + 1, to)
        if (arg?.bare) script(arg)
        i = arg?.bare ? arg.next : i + 1
        continue
      }
      const atom = f.atom(i, to)
      if (!atom) {
        i++
        continue
      }
      const start = f.tokens[atom.from].start
      const several = severalAtoms(f, atom, to, find)
      if (several) {
        wrap(start, f.tokens[several.last.next - 1].end, several.entry, false)
        if (several.exponent) script(several.last.sup)
        i = several.last.next
        continue
      }
      const whole = find.one(f.name(atom, true))
      const entry = whole ?? (atom.sup || atom.primes ? find.one(f.name(atom, false)) : undefined)
      if (entry) wrap(start, f.tokens[atom.next - 1].end, entry, bare && atom.from === from && atom.next === to)
      else unknown.set(f.name(atom, false), f.source(atom))
      if (!whole) script(atom.sup)
      i = atom.next
    }
  }
  scan(0, f.tokens.length, false)
  if (!edits.length) return tex

  // Closing marks before opening ones at the same place: `}` of R, then `\htmlData{…}{` of S in RS.
  edits.sort((a, b) => a.at - b.at || Number(a.open) - Number(b.open))
  const apply = (key: 'marked' | 'plain') => {
    let out = ''
    let last = 0
    for (const e of edits) {
      out += tex.slice(last, e.at) + e[key]
      last = e.at
    }
    return out + tex.slice(last)
  }
  const marked = apply('marked')
  const drawn = render(marked, display)
  const plain = render(apply('plain'), display) // the formula with the braces the marks add, if any
  if (drawn !== undefined && plain !== undefined) return unmarked(drawn) === plain ? marked : undefined
  return render(tex, display) === undefined ? tex : undefined // a broken formula: rehype-katex reports it
}

/**
 * An entry of several atoms (RS_0) that starts with `first`; the last atom may carry an exponent. Of entries as long,
 * one with the superscript (RE^v) beats one without it (RE, with v an exponent).
 */
function severalAtoms(f: Formula, first: Atom, to: number, find: Finder) {
  let withExponent: { entry: Entry; last: Atom; exponent: boolean } | undefined
  for (const entry of find.many) {
    if (withExponent && entry.names.length < withExponent.entry.names.length) break
    let atom: Atom | undefined = first
    let exponent = false
    for (let k = 0; atom && k < entry.names.length; k++) {
      const last = k === entry.names.length - 1
      if (f.name(atom, true) !== entry.names[k]) {
        if (!last || !(atom.sup || atom.primes) || f.name(atom, false) !== entry.names[k]) break
        exponent = true
      }
      if (last) {
        if (!exponent) return { entry, last: atom, exponent }
        withExponent ??= { entry, last: atom, exponent }
      } else atom = f.atom(atom.next, to)
    }
  }
  return withExponent
}

/** KaTeX's HTML of a formula (without its MathML, which holds the LaTeX); undefined if KaTeX can't draw it. */
function render(tex: string, displayMode: boolean): string | undefined {
  try {
    const html = katex.renderToString(tex, { ...KATEX_OPTIONS, strict: 'ignore', displayMode, throwOnError: true })
    return html.slice(html.indexOf('<span class="katex-html"'))
  } catch {
    return undefined
  }
}

/** The HTML without the spans of the marks, to compare with the unmarked formula. */
function unmarked(html: string): string {
  let out = ''
  let last = 0
  const marks: boolean[] = []
  for (const m of html.matchAll(/<span\b[^>]*>|<\/span>/g)) {
    out += html.slice(last, m.index)
    last = m.index + m[0].length
    const mark = m[0] === '</span>' ? marks.pop() : m[0].includes(' data-sym="')
    if (m[0] !== '</span>') marks.push(mark!)
    if (!mark) out += m[0]
  }
  return out + html.slice(last)
}
