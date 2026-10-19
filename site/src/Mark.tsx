import type { CSSProperties } from 'react'
import { PLATES } from './plates'

/**
 * `<Mark id="disk-1" />` in an .mdx file, before a paragraph: a small drawing of src/art/ at the screen edge, level
 * with that paragraph, like the stray suns and specks Fomenko scatters around his plates. `size` is
 * its width in px, `out` the share of it beyond the screen edge, `dy` a shift down in px, so it doesn't line up with
 * the paragraph's first line and seem to point at it; it is half transparent. Left by default,
 * `side="right"`. Shown, like plates, only on a desktop with a wide enough margin, and moves with the text.
 * scripts/check-links.mjs fails the build on an id without a drawing.
 */
export function Mark({
  id,
  side = 'left',
  size = 120,
  out = 0.4,
  dy = 0,
}: {
  id: string
  side?: 'left' | 'right'
  size?: number
  out?: number
  dy?: number
}) {
  const src = PLATES[id]
  if (!src) return null
  const mask = `url(${src})`
  const style = { '--size': `${size}px`, '--out': out, top: dy, WebkitMaskImage: mask, maskImage: mask } as CSSProperties
  return (
    <div className={`plate mark-${side}`} aria-hidden="true">
      <div className="mark-art" style={style} />
    </div>
  )
}
