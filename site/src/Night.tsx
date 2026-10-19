import type { CSSProperties, ReactNode } from 'react'
import { PLATES } from './plates'

/**
 * `<Night id="stairs">…</Night>` in an .mdx file: the end of a note on black. A wide drawing of src/art/ crosses the
 * column's whole sheet, side margins included, and its black foot runs on as the background of everything inside,
 * down to the note's end. On paper that black is the ink and the text inside is set in the dark palette; in dark mode
 * the page is already dark, so the drawing comes as src/art/<id>-dark, whose foot is the page. A mask painted in the
 * ink colour, like a plate. scripts/check-links.mjs fails the build on an id without a drawing.
 */
export function Night({ id, children }: { id: string; children: ReactNode }) {
  const src = PLATES[id]
  const dark = PLATES[`${id}-dark`] ?? src
  const style = src ? ({ '--mask': `url(${src})`, '--mask-dark': `url(${dark})` } as CSSProperties) : undefined
  return (
    <div className="night">
      {src && <div className="night-art" style={style} aria-hidden="true" />}
      <div className="night-body">{children}</div>
    </div>
  )
}
