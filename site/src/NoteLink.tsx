import { use, type ComponentProps, type MouseEvent, type ReactNode } from 'react'
import { isNote, keptBy, NOTES, withCompanion } from './content'
import { PaneContext, StackContext } from './context'
import { findSection, scrollToHeading } from './section'
import { hrefFor } from './url'

/** Opens note `id` to the right of the pane the link sits in, or of its companion open beside it; with `section`, at
 *  its heading with that address (headings.ts). */
export function NoteLink({ id, section, children }: { id: string; section?: string; children?: ReactNode }) {
  const pane = use(PaneContext)
  const { ids, mobile, open } = use(StackContext)
  if (!isNote(id)) {
    return <span className="note-link-missing" title={`Missing note: ${id}`}>{children}</span>
  }

  const onClick = (e: MouseEvent) => {
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return // new tab etc.
    e.preventDefault()
    open(pane, id, section)
  }
  // A real href, so "open in new tab" and copy-link give the stack this click would show.
  const opened = withCompanion(id, mobile)
  const notes = [...keptBy(ids, pane).filter((x) => !opened.includes(x)), ...opened]
  const href = hrefFor(notes) + (section ? `#${section}` : '')
  return (
    <a
      className="note-link"
      href={href}
      title={`Opens a side note: ${NOTES[id].title}`}
      data-open={ids.includes(id) || undefined}
      onClick={onClick}
    >
      {withArrow(children)}
    </a>
  )
}

const ARROW = (
  <svg className="note-link-arrow" viewBox="0 0 16 16" aria-hidden="true">
    <path d="M2.5 8h10M8.5 3.5 13 8l-4.5 4.5" />
  </svg>
)

/** The text with the arrow after it. The last word and the arrow never part at a line break. */
function withArrow(children: ReactNode) {
  if (typeof children !== 'string') {
    return (
      <>
        {children}
        {ARROW}
      </>
    )
  }
  const cut = children.lastIndexOf(' ') + 1
  return (
    <>
      {children.slice(0, cut)}
      <span className="note-link-end">
        {children.slice(cut)}
        {ARROW}
      </span>
    </>
  )
}

/** Markdown links: `[text](note:id)` opens a note, `[text](note:id#full-trust)` opens it at its heading "Full Trust",
 *  `[text](#full-trust)` scrolls its note to that heading (its address, headings.ts); web links open in a new tab. */
export function MdxLink({ href = '', children, ...rest }: ComponentProps<'a'>) {
  if (href.startsWith('note:')) {
    const [id, section] = href.slice(5).split('#')
    return (
      <NoteLink id={id} section={section || undefined}>
        {children}
      </NoteLink>
    )
  }
  if (href.startsWith('#')) {
    return (
      <a href={href} className="section-link" onClick={(e) => toSection(e, href.slice(1))} {...rest}>
        {children}
      </a>
    )
  }
  const web = /^https?:/.test(href)
  return (
    <a href={href} target={web ? '_blank' : undefined} rel={web ? 'noreferrer' : undefined} {...rest}>
      {children}
    </a>
  )
}

/** Scroll the note the link sits in to its heading with address `id` (headings.ts). */
function toSection(e: MouseEvent<HTMLAnchorElement>, id: string) {
  const pane = e.currentTarget.closest('.pane')
  const heading = pane && findSection(pane, id)
  if (!heading) return
  e.preventDefault()
  scrollToHeading(heading)
}
