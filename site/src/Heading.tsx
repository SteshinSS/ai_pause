import { use, useEffect, useState, type ComponentProps, type MouseEvent } from 'react'
import { keptBy } from './content'
import { PaneContext, StackContext } from './context'
import { hrefFor } from './url'
import { same } from './useStack'

type Props = ComponentProps<'h1'> & { 'data-section'?: string }

/**
 * A heading of a note's text, a link to its section: the notes open up to its column (with its companion, when open
 * beside it), then #address (headings.ts). A click copies that link and says so after the heading; when those are the
 * notes open, the address bar shows it too. Opened, the link scrolls the column to the heading (Stack.tsx).
 */
export function Heading({ level, 'data-section': section, children, ...rest }: Props & { level: number }) {
  const pane = use(PaneContext)
  const { ids } = use(StackContext)
  const [copied, setCopied] = useState(0) // clicks that copied the link; back to 0 a while after the last
  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(0), 1600)
    return () => clearTimeout(timer)
  }, [copied])

  const H = `h${level}` as 'h1'
  if (!section) return <H {...rest}>{children}</H>
  const notes = keptBy(ids, pane)
  const href = `${hrefFor(notes)}#${section}`
  const onClick = (e: MouseEvent) => {
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return // new tab etc.
    e.preventDefault()
    if (same(notes, ids)) history.replaceState(history.state, '', href)
    navigator.clipboard?.writeText(new URL(href, location.href).href).then(
      () => setCopied((n) => n + 1),
      () => {}, // not allowed: the address bar may still have it
    )
  }
  return (
    <H {...rest} data-section={section}>
      <a className="heading-link" href={href} onClick={onClick} data-copied={copied > 0 || undefined}>
        {children}
      </a>
    </H>
  )
}
