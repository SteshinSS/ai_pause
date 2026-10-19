import type { Nodes, Root } from 'mdast'
import type { MdxJsxFlowElement, MdxJsxTextElement } from 'mdast-util-mdx-jsx'

/** Sections a milestone's popup covers: the next one with a popup is at least this many sections later. */
const EVERY = 3

/**
 * Popups of `<Milestone>` names, not on every use. In each file a milestone's name gets its popup on its first use and
 * then on its first use at least three sections after the last one with a popup; the uses between get `plain`, which
 * src/ResultTable.tsx renders as just the name. A section starts at a `#` or `##` heading with text after it, so a
 * part's heading and its first `##` heading are one section. Names count apart: AC and pre-AC are two milestones.
 * Runs after remarkFootnotes, so a footnote's milestones count where it is referred to.
 */
export function remarkMilestones() {
  return (tree: Root) => {
    const last = new Map<string, number>() // name -> section of its last popup
    let section = 0
    let empty = true // nothing but headings since the section started

    const visit = (node: Nodes) => {
      if (node.type === 'heading') {
        if (node.depth <= 2 && !empty) {
          section++
          empty = true
        }
        return
      }
      if (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') {
        if (node.name === 'Milestone') return milestone(node)
      }
      if (node.type !== 'root') empty = false
      if ('children' in node) node.children.forEach(visit)
    }

    const milestone = (node: MdxJsxFlowElement | MdxJsxTextElement) => {
      empty = false
      const name = node.children.map((c) => ('value' in c ? c.value : '')).join('').toLowerCase()
      const at = last.get(name)
      if (at === undefined || section - at >= EVERY) last.set(name, section)
      else node.attributes.push({ type: 'mdxJsxAttribute', name: 'plain', value: null })
    }

    visit(tree)
  }
}
