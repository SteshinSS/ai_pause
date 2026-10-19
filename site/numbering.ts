import type { Root } from 'mdast'

/**
 * Section numbers for the headings of a file with `numbered: true` in its frontmatter: 1, 1.2, 1.2.1, counted from the
 * file's highest heading level, so a note that starts at `##` numbers its `##` headings 1, 2, 3. Only the note's own
 * headings count, not those inside components. The number goes in `data-num` and styles.css draws it, so the heading's
 * text, which links (NoteLink.tsx) and the side-by-side sync (sync.ts) match on, stays as it is.
 */
export function remarkNumbering() {
  return (tree: Root) => {
    const yaml = tree.children.find((n) => n.type === 'yaml')
    if (!(yaml?.type === 'yaml' && /^numbered:\s*true\s*$/m.test(yaml.value))) return
    const headings = tree.children.filter((n) => n.type === 'heading')
    const top = Math.min(...headings.map((h) => h.depth))
    const counts: number[] = []
    for (const h of headings) {
      const level = h.depth - top
      counts.length = level + 1
      counts[level] = (counts[level] ?? 0) + 1
      const num = Array.from(counts, (c) => c ?? 1).join('.') // a level skipped counts as 1
      h.data = { ...h.data, hProperties: { ...h.data?.hProperties, dataNum: num } }
    }
  }
}
