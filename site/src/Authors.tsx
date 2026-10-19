import type { ReactNode } from 'react'

/**
 * The authors under a heading, centred side by side as on a paper's title page; the heading above is centred with
 * them (styles.css). The children are a markdown list of names; an affiliation follows a name in *italics* and
 * shows as a line under it. In .mdx files it is `<Authors>` around the list, with no import (Pane.tsx provides it).
 */
export function Authors({ children }: { children: ReactNode }) {
  return <div className="authors">{children}</div>
}
