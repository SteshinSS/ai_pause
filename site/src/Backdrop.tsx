import type { CSSProperties } from 'react'
import { PLATES } from './plates'

/**
 * `<Backdrop id="full-trust" />` in an .mdx file: a drawing of src/art/ under the text, hung from its place and
 * scrolling with it, as the background of a section. It spans the column's whole sheet, side margins included, at
 * 2:3, and fades out along the bottom. `lift` is the share of its height that sits above the anchor, so a detail near
 * its top can sit beside a heading at any column width. `top` hangs it from the top of the note instead, above the
 * title, as the background of the whole note. Faint (--backdrop, styles.css), so the formulas over it stay
 * readable. A mask painted in the ink colour, so it follows the theme; like a plate, it may come with
 * src/art/<id>-dark for dark mode. Decoration only. scripts/check-links.mjs fails the build on an id without a drawing.
 */
export function Backdrop({ id, lift = 0, top = false }: { id: string; lift?: number; top?: boolean }) {
  const src = PLATES[id]
  if (!src) return null
  const mask = (url: string) => `url(${url}), linear-gradient(to bottom, #000 70%, transparent)`
  const dark = PLATES[`${id}-dark`] ?? src
  const style = {
    '--mask': mask(src),
    '--mask-dark': mask(dark),
    transform: `translateY(${-100 * lift}%)`,
  } as CSSProperties
  return (
    <div className="backdrop" data-top={top || undefined} aria-hidden="true">
      <div className="backdrop-art" style={style} />
    </div>
  )
}
