import type { Doc } from './content'

/** Column geometry in px, used by the CSS (as custom properties) and by the scroll maths. */
export const PANE_W = 720
export const SPINE_W = 40

/** A note and its companion, the last two columns, share the screen (Stack.tsx), down to this width each: room for
 *  the widest formula and the result table. Narrower, they overlap. */
export const PAIR_MIN_W = 560

/** Beside such a pair, the columns before it show this many times less width than beside two full columns; under
 *  two spines' worth they fold into spines PAIR_SPINE_W wide. styles.css shades them. */
export const PAIR_SHRINK = 1.5
export const PAIR_SPINE_W = 24

/** The narrowest side margin, when a column opens as the last one, that shows its drawings (Plate.tsx); in a
 *  narrower one they'd be smudges. */
export const PLATE_MARGIN = 160

/** The narrowest spare width beside the last column, the main one when alone, that holds a control beside its text:
 *  the result table's menu (ResultTable.tsx) with room to the screen edge. Narrower, the menu goes above the table. */
export const ASIDE_MARGIN = 240

/** Phone layout: notes open as full-screen sheets. Also catches a phone held sideways. */
export const MOBILE_QUERY = '(max-width: 899px), (pointer: coarse) and (max-height: 500px)'

/** The columns of a stack on desktop: `n` of them `paneW` wide, folding into spines `spineW` wide, and `extra` spare
 *  width beside each outer one when they all fit. `pair`: the last two are a note and its companion. */
export type Columns = { n: number; paneW: number; spineW: number; extra: number; pair: boolean }

/** The columns of `docs` (the main text first) on a screen `width` px wide. A note and its companion as the last two
 *  columns take the screen but for what the columns before them show: 1 / PAIR_SHRINK of what they'd show beside two
 *  full columns, or narrower spines. The pair goes down to PAIR_MIN_W each; wider than PANE_W, its text keeps its
 *  width and the margins grow (styles.css). */
export function columns(docs: Doc[], width: number, mobile: boolean): Columns {
  const n = docs.length
  const pair = !mobile && n > 2 && docs[n - 2].companion === docs[n - 1].id
  const spineW = pair ? PAIR_SPINE_W : SPINE_W
  const peek = (width - 2 * PANE_W) / PAIR_SHRINK
  const before = peek < 2 * SPINE_W ? (n - 2) * spineW : peek
  const paneW = pair ? Math.max(PAIR_MIN_W, Math.floor((width - before) / 2)) : PANE_W
  return { n, paneW, spineW, extra: Math.max(0, (width - n * paneW) / 2), pair }
}
