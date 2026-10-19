import type { ReactNode } from 'react'

/**
 * `<Summary>` around markdown in an .mdx file: the executive summary, set apart from the text in a framed block
 * (styles.css). A `##` heading titles it, `###` headings start its parts. A flat list shows as numbered headline
 * points; a nested list as a tree, its top items as headings of their branches. In .mdx files it needs no import
 * (Pane.tsx provides it).
 */
export function Summary({ children }: { children: ReactNode }) {
  return <aside className="summary">{children}</aside>
}
