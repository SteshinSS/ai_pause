import type { FootnoteDefinition, Parent, PhrasingContent, RootContent } from 'mdast'
import { gfmFootnoteFromMarkdown } from 'mdast-util-gfm-footnote'
import type { MdxJsxTextElement } from 'mdast-util-mdx-jsx'
import { gfmFootnote } from 'micromark-extension-gfm-footnote'
import type { Processor } from 'unified'
import type { VFile } from 'vfile'

const REFERENCE = /\[\^[^\]\s]*\]/ // [^id] left as text: no definition has that id

/**
 * Markdown footnotes as popups. A reference `word[^id]` and its definition `[^id]: text` (anywhere in the file, from
 * the start of a line; more paragraphs indented by four spaces) become
 * `<span className="footnote-end">word<Footnote>text</Footnote></span>`: the definition leaves the page, and the span
 * keeps the word and the footnote's mark on one line. `Footnote` is src/Footnote.tsx. The build fails on a reference
 * without a definition and on a definition without a reference.
 */
export function remarkFootnotes(this: Processor) {
  const data = this.data() as { micromarkExtensions?: unknown[]; fromMarkdownExtensions?: unknown[] } // remark-parse's
  ;(data.micromarkExtensions ??= []).push(gfmFootnote())
  ;(data.fromMarkdownExtensions ??= []).push(gfmFootnoteFromMarkdown())

  return (tree: Parent, file: VFile) => {
    const notes = new Map<string, FootnoteDefinition>()
    const used = new Set<string>()

    const takeDefinitions = (parent: Parent) => {
      parent.children = parent.children.filter((node) => {
        if (node.type === 'footnoteDefinition') {
          notes.set(node.identifier, node)
          return false
        }
        if ('children' in node) takeDefinitions(node)
        return true
      })
    }

    const placeNotes = (parent: Parent) => {
      const out: RootContent[] = []
      for (const node of parent.children) {
        if (node.type === 'footnoteReference') {
          used.add(node.identifier)
          const note: MdxJsxTextElement = {
            type: 'mdxJsxTextElement',
            name: 'Footnote',
            attributes: [],
            // Paragraphs and display maths in a text element: MDX passes them on as they are.
            children: structuredClone(notes.get(node.identifier)!.children) as unknown as PhrasingContent[],
          }
          out.push({
            type: 'mdxJsxTextElement',
            name: 'span',
            attributes: [{ type: 'mdxJsxAttribute', name: 'className', value: 'footnote-end' }],
            children: [...lastWord(out), note],
          })
          continue
        }
        if (node.type === 'text' && REFERENCE.test(node.value)) {
          file.fail(`footnote ${REFERENCE.exec(node.value)![0]} has no definition`, {
            place: node.position,
            source: 'footnotes',
          })
        }
        if ('children' in node) placeNotes(node)
        out.push(node)
      }
      parent.children = out
    }

    takeDefinitions(tree)
    placeNotes(tree)
    for (const [id, node] of notes) {
      if (!used.has(id)) file.fail(`nothing refers to footnote [^${node.label}]`, { place: node.position, source: 'footnotes' })
    }
  }
}

/** Takes the word before a footnote off the end of `out`: the end of a text, or a whole node such as inline maths. */
function lastWord(out: RootContent[]): PhrasingContent[] {
  const prev = out.at(-1)
  if (!prev || prev.type === 'break') return []
  if (prev.type !== 'text') return [out.pop() as PhrasingContent]
  const word = /\S+$/.exec(prev.value)?.[0]
  if (!word) return []
  prev.value = prev.value.slice(0, -word.length)
  if (!prev.value) out.pop()
  return [{ type: 'text', value: word }]
}
