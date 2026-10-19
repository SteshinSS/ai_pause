import type { CSSProperties } from 'react'
import { PLATES } from './plates'

/**
 * `<Doodle id="equations" />` in an .mdx file, just before a heading: a drawing of src/art/ in the big space above
 * that heading that lines the note up with its companion (sync.ts, which sets --space). So, like the coffee stains,
 * it shows only while the companion is open beside the note. `x` is the share of the sheet's width (side margins
 * included) at which its middle sits, `y` the share of the space, `width` its width as a share of the sheet, so it
 * may reach past the sheet's edges, which cut it; `turn` turns it, in degrees. It fades out to the left and right and
 * is barely visible (--doodle, styles.css). A mask painted in the ink colour, like a plate. Decoration only.
 * scripts/check-links.mjs fails the build on an id without a drawing.
 */
export function Doodle({
  id,
  x = 0.5,
  y = 0.5,
  width = 1,
  turn = 0,
}: {
  id: string
  x?: number
  y?: number
  width?: number
  turn?: number
}) {
  const src = PLATES[id]
  if (!src) return null
  const style = { '--mask': `url(${src})`, '--x': x, '--y': y, '--w': width, '--turn': `${turn}deg` } as CSSProperties
  return (
    <div className="doodle" aria-hidden="true">
      <div className="doodle-art" style={style} />
    </div>
  )
}
