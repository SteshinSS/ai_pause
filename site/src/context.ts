import { createContext } from 'react'

/** Index of the pane the content is rendered in (0 = main column). */
export const PaneContext = createContext(0)

/** Where the popups of terms in a popup go (Gloss.tsx): inside that popup, so that opening them leaves it open. */
export const Inside = createContext<HTMLElement | null>(null)

export type StackApi = {
  ids: string[]
  /** Phone layout: notes are full-screen sheets. */
  mobile: boolean
  /** Open note `id` from a link in pane `from`, scrolled to its heading with address `section` if given. */
  open: (from: number, id: string, section?: string) => void
}

export const StackContext = createContext<StackApi>({ ids: [], mobile: false, open: () => {} })
