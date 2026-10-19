import { plugins, type CSL, type CSLName } from '@citation-js/core'
import '@citation-js/plugin-bibtex'
import type { Parent, PhrasingContent, RootContent } from 'mdast'
import type { MdxJsxTextElement } from 'mdast-util-mdx-jsx'
import { readFileSync, statSync } from 'node:fs'
import { relative } from 'node:path'
import type { VFile } from 'vfile'

type Ref = {
  /** "Lifland et al." as in "(Lifland et al., 2025)"; the title when there are no authors. */
  names: string
  /** "Kokotajlo and Lifland" as in "Kokotajlo and Lifland (2025)". */
  narrative: string
  /** "2025"; "2025a" when another reference has the same names and year; "n.d." without a date. */
  year: string
  /** The full reference. */
  text: PhrasingContent[]
  sortKey: string
}

// A citation key: a letter, digit or _, then also :.#$%&+?<>~/-, ending in a letter, digit or _ (as in pandoc).
const KEY = String.raw`\w(?:[\w:.#$%&+?<>~/-]*\w)?`
// `[...]` without brackets inside, or `@key` not after a letter, digit, @ or backslash (not in an e-mail address).
const CITATION = new RegExp(String.raw`\[([^[\]]*)\]|(?<![\w@\\])@(${KEY})`, 'g')
// One citation in brackets: prefix, `-` that leaves the names out, key, suffix. The `@` starts it or follows a space.
const ITEM = new RegExp(String.raw`^(.*?)(?<=^|\s)(-?)@(${KEY})(.*)$`, 's')

/**
 * Citations as in pandoc, from a BibTeX file:
 * - `[@key]` is "(Lifland et al., 2025)", `[@a; @b]` is "(Davidson, 2023; Lifland et al., 2025)". Words around a key
 *   stay: `[see @key, p. 5]` is "(see Lifland et al., 2025, p. 5)". `[-@key]` is "(2025)".
 * - `@key` is "Lifland et al. (2025)".
 * - `<References />` becomes the list of all entries of the file.
 * Each "Lifland et al., 2025" becomes `<Cite label="Lifland et al., 2025">` (src/Cite.tsx), whose popup holds the full
 * reference. The build fails on a bracketed key the file doesn't have; an unknown `@key` stays text, with a warning.
 * Run it before footnotes.ts, so that footnotes can hold citations.
 */
export function remarkCitations({ bibliography }: { bibliography: string }) {
  const name = relative(process.cwd(), bibliography) // for messages
  return (tree: Parent, file: VFile) => {
    const refs = load(bibliography, file)

    const cite = (key: string, label: string): MdxJsxTextElement => ({
      type: 'mdxJsxTextElement',
      name: 'Cite',
      attributes: [{ type: 'mdxJsxAttribute', name: 'label', value: label }],
      // A paragraph in a text element: MDX passes it on as it is (as in footnotes.ts).
      children: [{ type: 'paragraph', children: structuredClone(refs.get(key)!.text) }] as unknown as PhrasingContent[],
    })

    // A text as texts and citations; `node` places errors.
    const split = (value: string, node: RootContent): PhrasingContent[] => {
      const out: PhrasingContent[] = []
      const text = (value: string) => {
        const prev = out.at(-1)
        if (prev?.type === 'text') prev.value += value
        else if (value) out.push({ type: 'text', value })
      }
      let last = 0
      for (const m of value.matchAll(CITATION)) {
        const [whole, inside, key] = m
        if (key !== undefined && !refs.has(key)) {
          file.message(`@${key} is not in ${name}, left as text`, { place: node.position, source: 'citations' })
          continue
        }
        text(value.slice(last, m.index))
        last = m.index + whole.length
        if (key !== undefined) {
          const ref = refs.get(key)!
          out.push(cite(key, `${ref.narrative} (${ref.year})`))
          continue
        }
        const items = inside.split(';').map((item) => ITEM.exec(item.trim()))
        if (items.some((item) => !item)) {
          // Brackets, not a citation; an `@key` may be inside.
          text('[')
          for (const part of split(inside, node)) {
            if (part.type === 'text') text(part.value)
            else out.push(part)
          }
          text(']')
          continue
        }
        // Each citation with its words, bracket or semicolon in a span that keeps them on one line: a line may break on
        // either side of a popup's term (a button), even after "(".
        items.forEach((item, i) => {
          const [, prefix, bare, key, suffix] = item!
          const ref = refs.get(key)
          if (!ref) file.fail(`citation @${key} is not in ${name}`, { place: node.position, source: 'citations' })
          const before = (i ? '' : '(') + prefix
          const after = suffix + (i < items.length - 1 ? ';' : ')')
          if (i) text(' ')
          out.push({
            type: 'mdxJsxTextElement',
            name: 'span',
            attributes: [{ type: 'mdxJsxAttribute', name: 'className', value: 'cite-item' }],
            children: [
              ...(before ? [{ type: 'text', value: before } as const] : []),
              cite(key, bare ? ref.year : `${ref.names}, ${ref.year}`),
              { type: 'text', value: after },
            ],
          })
        })
      }
      text(value.slice(last))
      return out
    }

    const walk = (parent: Parent) => {
      const children: RootContent[] = []
      for (const node of parent.children) {
        if (node.type === 'text') {
          children.push(...split(node.value, node))
          continue
        }
        if (node.type === 'mdxJsxFlowElement' && node.name === 'References') {
          const all = [...refs.values()].sort((a, b) => a.sortKey.localeCompare(b.sortKey))
          node.children = all.map((ref) => ({ type: 'paragraph', children: structuredClone(ref.text) }))
        } else if ('children' in node && node.type !== 'link' && node.type !== 'linkReference') {
          walk(node) // not in a link's text: a popup's term can't sit inside a link
        }
        children.push(node)
      }
      parent.children = children
    }
    walk(tree)
  }
}

type Options = Parameters<typeof plugins.input.chain>[1]

let cache: { path: string; mtime: number; refs: Map<string, Ref> } | undefined

/** The references in the BibTeX file by key; read again when the file changes. */
function load(path: string, file: VFile): Map<string, Ref> {
  try {
    const mtime = statSync(path).mtimeMs
    if (cache?.path !== path || cache.mtime !== mtime) cache = { path, mtime, refs: parse(readFileSync(path, 'utf8')) }
    return cache.refs
  } catch (e) {
    return file.fail(`${relative(process.cwd(), path)}: ${e instanceof Error ? e.message : e}`, { source: 'citations' })
  }
}

function parse(bibtex: string): Map<string, Ref> {
  if (!/@\w+\s*\{/.test(bibtex)) return new Map() // no entries: citation-js would not take the text for BibTeX
  const raw = plugins.input.chain(bibtex, { target: '@biblatex/entries+list', generateGraph: false } as Options) as unknown as {
    label: string
    properties: Record<string, string>
  }[]
  const keys = new Set<string>()
  for (const { label } of raw) {
    if (keys.has(label)) throw new Error(`two entries have the key ${label}`)
    keys.add(label)
  }
  const entries = plugins.input.chain(raw, { generateGraph: false } as Options)

  const refs = new Map(entries.map((entry) => [String(entry.id), describe(entry)]))
  // Same names and year: 2025a, 2025b, ... by title.
  const groups = new Map<string, Ref[]>()
  for (const ref of refs.values()) {
    const same = `${ref.names} ${ref.year}`
    groups.set(same, [...(groups.get(same) ?? []), ref])
  }
  for (const group of groups.values()) {
    if (group.length < 2) continue
    group.sort((a, b) => a.sortKey.localeCompare(b.sortKey))
    group.forEach((ref, i) => (ref.year += String.fromCharCode(97 + i)))
  }
  entries.forEach((entry, i) => {
    // citation-js keeps only PubMed ids of the eprint fields, so arXiv ids come from the fields as written.
    const { eprint, archiveprefix, eprinttype } = raw[i].properties
    const arxiv = eprint && /^arxiv$/i.test(archiveprefix ?? eprinttype ?? '') ? eprint : undefined
    const ref = refs.get(String(entry.id))!
    ref.text = reference(entry, arxiv, ref.year)
  })
  return refs
}

function describe(entry: CSL): Ref {
  const families = (entry.author ?? []).map(family)
  const title = entry.title ?? String(entry.id)
  const year = String(entry.issued?.['date-parts']?.[0]?.[0] ?? 'n.d.')
  const short = (and: string) => (families.length > 2 ? `${families[0]} et al.` : join(families, and))
  return {
    names: families.length ? short('&') : title,
    narrative: families.length ? short('and') : title,
    year,
    text: [],
    sortKey: `${families.length ? (entry.author ?? []).map(inverted).join(' ') : title} ${year} ${title}`.toLowerCase(),
  }
}

/**
 * APA-like: "Jones, C. I. (1995). R&D-based models of economic growth. *Journal of Political Economy*, 103(4),
 * 759–784." The title links to the DOI, the URL or the arXiv page, whichever comes first.
 */
function reference(entry: CSL, arxiv: string | undefined, year: string): PhrasingContent[] {
  const out: PhrasingContent[] = []
  const add = (node: PhrasingContent | string) => {
    const prev = out.at(-1)
    if (typeof node !== 'string') out.push(node)
    else if (prev?.type === 'text') prev.value += node
    else out.push({ type: 'text', value: node })
  }
  const italic = (value: string): PhrasingContent => ({ type: 'emphasis', children: [{ type: 'text', value }] })
  const stop = (s: string) => (/[.?!]$/.test(s) ? '' : '.')

  const title = entry.title ?? String(entry.id)
  const container = entry['container-title']
  const url = entry.DOI ? `https://doi.org/${entry.DOI}` : (entry.URL ?? (arxiv && `https://arxiv.org/abs/${arxiv}`))
  // A work inside another (an article in a journal) has an upright title and the other's title in italics.
  const name = container ? ({ type: 'text', value: title } as const) : italic(title)

  const authors = (entry.author ?? []).map(inverted)
  const byline = authors.length < 2 ? authors.join('') : `${authors.slice(0, -1).join(', ')}, & ${authors.at(-1)}`
  if (byline) add(`${byline}${stop(byline)} (${year}). `)
  add(url ? { type: 'link', url, children: [name] } : name)
  add(stop(title))
  if (!byline) add(` (${year}).`)
  if (container) {
    const book = entry.type === 'paper-conference' || entry.type === 'chapter'
    const pages = entry.page && String(entry.page).replace(/-+/g, '–')
    add(book ? ' In ' : ' ')
    add(italic(container))
    if (entry.volume) add(`, ${entry.volume}`)
    if (entry.issue) add(`(${entry.issue})`)
    if (pages) add(book ? ` (pp. ${pages})` : `, ${pages}`)
    add('.')
  }
  if (entry.publisher && entry.publisher !== container) add(` ${entry.publisher}${stop(entry.publisher)}`)
  if (arxiv && !container?.includes(arxiv)) add(` arXiv:${arxiv}.`)
  return out
}

/** "van der Hubinger"; a group's name as it is. */
const family = (n: CSLName) => n.literal ?? [n['non-dropping-particle'], n.family].filter(Boolean).join(' ')

/** "Hubinger, E." as in a reference list; "Charles I." -> "C. I.", "Jean-Paul" -> "J.-P.". */
const inverted = (n: CSLName) => {
  const initials = n.given
    ?.split(/\s+/)
    .map((word) => word.split('-').map((part) => (part.endsWith('.') ? part : `${[...part][0]}.`)).join('-'))
    .join(' ')
  return n.literal ?? [family(n), initials, n.suffix].filter(Boolean).join(', ')
}

/** "A", "A and B", "A, B, and C" (`and` may be "&"). */
const join = (names: string[], and: string) =>
  names.length < 3 ? names.join(` ${and} `) : `${names.slice(0, -1).join(', ')}, ${and} ${names.at(-1)}`
