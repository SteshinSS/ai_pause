import type { ReactNode } from 'react'
import { Gloss } from './Gloss'

/**
 * `<Term term="moonshot benchmarks">…</Term>` in an .mdx file: a term of the text with a popup that explains it. The
 * popup is inline JSX, since MDX can't hold block markdown in a paragraph: write it on one line, with `<p>` and
 * `<ul><li>` for its paragraphs and lists. In .mdx files it needs no import (Pane.tsx provides it).
 */
export function Term({ term, children }: { term: string; children: ReactNode }) {
  return <Gloss term={term}>{children}</Gloss>
}
