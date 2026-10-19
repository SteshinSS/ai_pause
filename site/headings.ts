import type { Heading, Link, Nodes, Root } from 'mdast'
import type { VFile } from 'vfile'

/**
 * The address of each heading, in `data-section`, for links to its section (src/Heading.tsx): its text in lower case
 * without apostrophes, a-z and 0-9 with a dash for each run of anything else, so "Full Trust" is `full-trust`. A
 * heading whose text another heading of the file has too gets the headings it sits under in front, as many as tell the
 * two apart: "Splits" under "# Compartmentalization" is `compartmentalization-splits`. Headings inside components count
 * too. A note and its companion with the same headings give them the same addresses. The build fails on a link
 * `[text](#address)` to no heading of its own file.
 */
export function remarkHeadings() {
  return (tree: Root, file: VFile) => {
    const found: { node: Heading; path: string[] }[] = [] // each heading and the slugs of those it sits under, its own last
    const links: Link[] = []
    const outline: Heading[] = []
    const walk = (node: Nodes) => {
      if (node.type === 'heading') {
        while (outline.length && outline.at(-1)!.depth >= node.depth) outline.pop()
        outline.push(node)
        found.push({ node, path: outline.map((h) => slug(text(h))) })
      }
      if (node.type === 'link' && node.url.startsWith('#')) links.push(node)
      if ('children' in node) for (const child of node.children) walk(child)
    }
    walk(tree)

    const end = (path: string[], k: number) => path.slice(-k).join('\n')
    const used = new Set<string>()
    for (const { node, path } of found) {
      let k = 1
      while (k < path.length && found.some((f) => f.node !== node && end(f.path, k) === end(path, k))) k++
      const base = path.slice(-k).join('-') || 'section'
      let address = base
      for (let i = 2; used.has(address); i++) address = `${base}-${i}` // same text under the same headings
      used.add(address)
      node.data = { ...node.data, hProperties: { ...node.data?.hProperties, dataSection: address } }
    }

    for (const link of links) {
      const address = decodeURIComponent(link.url.slice(1))
      if (!used.has(address)) {
        file.fail(`link to #${address}: no heading of this file has that address (headings.ts)`, {
          place: link.position,
          source: 'headings',
        })
      }
    }
  }
}

const slug = (text: string) =>
  text
    .toLowerCase()
    .replace(/['’]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')

/** The words of a heading: its text, code and maths source. */
const text = (node: Nodes): string =>
  node.type === 'text' || node.type === 'inlineCode' || node.type === 'inlineMath'
    ? node.value
    : 'children' in node
      ? (node.children as Nodes[]).map(text).join('')
      : ''
