import type { Element, ElementContent, Root, Text } from 'hast'

const AFTER = /^[.,;:!?)\]}’”]+/ // punctuation that never starts a line
const BEFORE = /[([{‘“]+$/ // brackets that never end one

const isInlineMath = (node: ElementContent | undefined): node is Element =>
  node?.type === 'element' && Array.isArray(node.properties.className) && node.properties.className.includes('katex')

/**
 * Inline maths and the punctuation right after it, or the bracket right before it, never part at a line break:
 * Chrome breaks lines on either side of KaTeX's boxes, even before a full stop. Runs after rehype-katex and wraps them
 * in `<span class="math-end">`, which styles.css keeps on one line.
 */
export function rehypeMathPunctuation() {
  const visit = (parent: { children: unknown }) => {
    const children = parent.children as ElementContent[]
    children.forEach((node, i) => {
      if (!isInlineMath(node)) {
        if ('children' in node) visit(node)
        return
      }
      const prev = children[i - 1]
      const next = children[i + 1]
      const before = prev?.type === 'text' ? BEFORE.exec(prev.value)?.[0] : undefined
      const after = next?.type === 'text' ? AFTER.exec(next.value)?.[0] : undefined
      if (!before && !after) return
      const text = (value: string): Text => ({ type: 'text', value })
      children[i] = {
        type: 'element',
        tagName: 'span',
        properties: { className: ['math-end'] },
        children: [...(before ? [text(before)] : []), node, ...(after ? [text(after)] : [])],
      }
      if (before) (prev as Text).value = (prev as Text).value.slice(0, -before.length)
      if (after) (next as Text).value = (next as Text).value.slice(after.length)
    })
  }
  return (tree: Root) => visit(tree)
}
