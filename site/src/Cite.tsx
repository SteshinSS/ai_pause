import type { ReactNode } from 'react'
import { Gloss } from './Gloss'

/**
 * A citation: "Lifland et al., 2025" with the full reference in a popup. In .mdx files it is `[@key]` or `@key`;
 * citations.ts turns it into this, with the reference from content/references.bib (Pane.tsx provides it).
 */
export function Cite({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Gloss term={label} className="cite-ref">
      {children}
    </Gloss>
  )
}

/** All references of content/references.bib, filled in by citations.ts. In .mdx files it is `<References />`. */
export function References({ children }: { children?: ReactNode }) {
  return <div className="references">{children}</div>
}
