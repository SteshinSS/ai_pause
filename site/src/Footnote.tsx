import type { ReactNode } from 'react'
import { Gloss } from './Gloss'

/**
 * A footnote: a mark after a word, with a popup of things the text doesn't need. The mark is a magnifying
 * glass, src/magnifier.svg, tinted by styles.css. In .mdx files it is a markdown footnote, `word[^id]` and
 * `[^id]: text`; footnotes.ts turns it into this (Pane.tsx provides it).
 */
export function Footnote({ children }: { children: ReactNode }) {
  return (
    <Gloss term={<span className="footnote-mark" />} label="Footnote" className="footnote-ref">
      {children}
    </Gloss>
  )
}
