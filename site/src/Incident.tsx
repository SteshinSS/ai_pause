import { Fragment, type ReactNode } from 'react'

/**
 * `<Incident title year source>` around markdown in an .mdx file: a card with one case of a model misbehaving, framed
 * like a plate. `year` is lettered beside the title. `source` is a URL or a list of them, shown by host name at its
 * foot. In .mdx files it needs no import (Pane.tsx provides it).
 */
export function Incident({
  title,
  year,
  source = [],
  children,
}: {
  title: string
  year?: number
  source?: string | string[]
  children?: ReactNode
}) {
  const urls = typeof source === 'string' ? [source] : source
  return (
    <article className="incident">
      <h3>
        {title}
        {year !== undefined && <span className="incident-year">{year}</span>}
      </h3>
      {children}
      {urls.length > 0 && (
        <p className="incident-source">
          {urls.length > 1 ? 'Sources: ' : 'Source: '}
          {urls.map((url, i) => (
            <Fragment key={url}>
              {i > 0 && ', '}
              <a href={url} target="_blank" rel="noreferrer">
                {new URL(url).hostname.replace(/^www\./, '')}
              </a>
            </Fragment>
          ))}
        </p>
      )}
    </article>
  )
}

/**
 * `<Said by="Its reasoning">…</Said>` in an <Incident>: the model's own words, the evidence that it knew what it was
 * doing, in pale ink on a black mass, the face behind the mask. `by` is lettered under them, as a plate's title is
 * under its picture.
 */
export function Said({ by = 'In its own words', children }: { by?: string; children?: ReactNode }) {
  return (
    <figure className="said">
      <blockquote>{children}</blockquote>
      <figcaption>{by}</figcaption>
    </figure>
  )
}
