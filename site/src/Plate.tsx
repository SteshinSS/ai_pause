import type { CSSProperties } from 'react'
import { PLATES } from './plates'

/**
 * `<Plate id="main" />` in an .mdx file: a drawing of src/art/ in the side margin, hung from its place in the text
 * and scrolling with it, so several down a column tell a story. It runs from the screen edge to a little under the
 * text, and moves with the text when later columns squeeze the margin. Left of the column by default;
 * `side="right"` for a note, whose spare width is on the right. `under` is the drawing's opacity at the text's edge,
 * from which it fades out under the text: lower it for a drawing that reaches far under the text. `flip` mirrors a
 * drawing made for the other side, so its blank third still faces the text. `scale` shrinks it, for a drawing that
 * fills its whole sheet, so its subject clears the fade; `fadeTop` fades it in along the top, for a drawing that runs
 * to its upper edge. A mask painted in the ink colour, so it follows the theme. Decoration only. The CSS shows it
 * only on a desktop with a wide enough margin (Pane.tsx).
 * scripts/check-links.mjs fails the build on an id without a drawing.
 */
export function Plate({
  id,
  side = 'left',
  under = 0.3,
  flip = false,
  scale = 1,
  fadeTop = false,
}: {
  id: string
  side?: 'left' | 'right'
  under?: number
  flip?: boolean
  scale?: number
  fadeTop?: boolean
}) {
  const src = PLATES[id]
  if (!src) return null
  // Faded out along the bottom, so a drawing that runs to its lower edge doesn't end in a line across the margin,
  // and towards the text (--lead, --fade, --reach: styles.css): from full ink --fade before the text's edge to `under` at
  // it, and out 3 × --reach under the text, so the text reads over plain paper. A flipped drawing is mirrored with its
  // box (styles.css), these gradients included, so its fade towards the text is laid out the other way round.
  const toText = (side === 'left') !== flip ? 'to right' : 'to left'
  const mask = (url: string) =>
    [
      `url(${url})`,
      `linear-gradient(to bottom, ${fadeTop ? 'transparent, #000 20%, ' : ''}#000 80%, transparent)`,
      `linear-gradient(${toText}, #000 calc(var(--lead) - var(--fade)), rgb(0 0 0 / ${under}) var(--lead), transparent calc(var(--lead) + 3 * var(--reach)))`,
    ].join(', ')
  // A drawing on a black ground would turn into a large pale mass in dark mode, so it may come with src/art/<id>-dark:
  // its negative, whose white lines are the ink and whose black ground is the page. styles.css picks it in dark mode.
  const dark = PLATES[`${id}-dark`] ?? src
  const style = { '--mask': mask(src), '--mask-dark': mask(dark), '--scale': scale } as CSSProperties
  return (
    <div className={`plate plate-${side}${flip ? ' plate-flip' : ''}`} aria-hidden="true">
      <div className="plate-art" style={style} />
    </div>
  )
}
