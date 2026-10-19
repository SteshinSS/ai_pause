import type { MDXContent } from 'mdx/types'

type Module = { default: MDXContent; frontmatter?: { title?: string; companion?: string; description?: string } }
/** `companion`: the id of a note read side by side with this one (see withCompanion). `description`: what search
 *  engines and link previews say of a note's own page (scripts/prerender.mjs), else its first paragraph. */
export type Doc = { id: string; title: string; companion?: string; description?: string; Content: MDXContent }

const modules = import.meta.glob<Module>('../content/**/*.mdx', { eager: true })

const doc = (id: string, m: Module): Doc => ({
  id,
  title: m.frontmatter?.title ?? id,
  companion: m.frontmatter?.companion,
  description: m.frontmatter?.description,
  Content: m.default,
})

/** The docs in content/<dir>/ by id; the id is the file name. */
const folder = (dir: string): Record<string, Doc> =>
  Object.fromEntries(
    Object.entries(modules).flatMap(([path, m]) => {
      const id = new RegExp(`/${dir}/([^/]+)\\.mdx$`).exec(path)?.[1]
      return id ? [[id, doc(id, m)]] : []
    }),
  )

export const MAIN = doc('main', modules['../content/main.mdx'])

/** Side notes by id. */
export const NOTES = folder('notes')

/** Popups of the result table's rows, by row name in lower case. */
export const MILESTONES = folder('milestones')

/** Popups of the result table's trust models in its menu, by model (table.json) with a dash for each space; the title
 *  is the model's name on the site. */
export const PROTOCOLS = folder('protocols')

/** A † on result-table cells, content/marks/<id>.mdx: its frontmatter names the trust model (table.json), the row and
 *  the pause durations in years of the cells it marks; the file is the popup, under its title. */
export type CellMark = Doc & { model: string; row: string; years: number[] }
export const MARKS: CellMark[] = Object.values(folder('marks')).map((d) => {
  const { model, row, years } = modules[`../content/marks/${d.id}.mdx`].frontmatter as unknown as CellMark
  return { ...d, model, row, years }
})

/** The entries of the symbol popups (Symbols.tsx), content/symbols.mdx. */
export const SYMBOLS: MDXContent | undefined = modules['../content/symbols.mdx']?.default

export const isNote = (id: string) => Object.hasOwn(NOTES, id)

/** The notes a link to note `id` opens: the note and, on desktop, its companion beside it (frontmatter
 *  `companion: <id>`), with which it scrolls (sync.ts); or, for a note that is another's companion, that note before
 *  it, as the model's formulas before their explanations. */
export function withCompanion(id: string, mobile: boolean): string[] {
  if (mobile) return [id]
  const companion = NOTES[id].companion
  if (companion && companion !== id && isNote(companion)) return [id, companion]
  const lead = Object.values(NOTES).find((note) => note.companion === id && note.id !== id)
  return lead ? [lead.id, id] : [id]
}

/** The open notes a link in pane `from` (0 = main column) keeps when it opens another: those up to its own and, when
 *  that one is read beside its companion, the companion too, so that the pair stays open. */
export function keptBy(ids: string[], from: number): string[] {
  const companion = from > 0 ? NOTES[ids[from - 1]]?.companion : undefined
  return ids.slice(0, companion && ids[from] === companion ? from + 1 : from)
}
