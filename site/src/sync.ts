/**
 * A note and its companion (content.ts), open side by side, scroll together: scrolling either column moves the other
 * so that their matching headings cross the reading line together. Between two such headings the other column moves
 * in proportion. Headings match when they have the same level and text and come in the same order (a longest common
 * subsequence), so either note may have headings the other lacks. Side by side, space before the headings lines the
 * columns up (align), so a short formula waits beside its long explanation and the two move as one. On a phone the
 * covered sheet follows too, so it opens, or shows again, at the matching place. The reader can turn it off with a
 * switch at the top of the companion (Stack.tsx). Three of the note's big spaces, spread down it, get a faint coffee
 * stain each (stain()).
 */

/** An event that makes a column the one the reader scrolled last, which the other follows: Heading.tsx sends it when it
 *  scrolls a column to a section. */
export const LEAD = 'lead'

/** The reading line, as a share of the column's height from its top. */
const LINE = 0.3

/** Coffee stains: in a space that fits one of at least STAIN_MIN px, STAIN_PAD px clear of the text above and below. */
const STAIN_MIN = 160
const STAIN_MAX = 340
const STAIN_PAD = 48
const STAINS = 3 // styles.css: [data-stain='0'] … [data-stain='2']

type Pair = [number, number]

/** The headings of a note's own text, not those inside components such as cards. */
const headings = (scroller: HTMLElement) =>
  Array.from(scroller.querySelectorAll<HTMLElement>('.pane-body > :is(h1, h2, h3, h4, h5, h6)'))

const key = (h: HTMLElement) => `${h.tagName} ${(h.textContent ?? '').replace(/\s+/g, ' ').trim()}`

/** Index pairs (i, j), a[i] = b[j], of a longest common subsequence of a and b. */
function matches(a: string[], b: string[]): Pair[] {
  // len[i][j]: length of a longest common subsequence of a.slice(i) and b.slice(j)
  const len = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0))
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      len[i][j] = a[i] === b[j] ? len[i + 1][j + 1] + 1 : Math.max(len[i + 1][j], len[i][j + 1])
    }
  }
  const out: Pair[] = []
  for (let i = 0, j = 0; i < a.length && j < b.length; ) {
    if (a[i] === b[j]) out.push([i++, j++])
    else if (len[i + 1][j] >= len[i][j + 1]) i++
    else j++
  }
  return out
}

/** The space align() put above heading `h`. */
const added = (h: HTMLElement) => parseFloat(h.style.paddingTop) || 0

/** How far down the scrolled content of `scroller` the text of heading `h` starts, below the space above it. */
const top = (scroller: HTMLElement, h: HTMLElement) =>
  h.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop + added(h)

/** The scroll position that puts heading `h` on the reading line of `scroller`. */
const onLine = (scroller: HTMLElement, h: HTMLElement) => top(scroller, h) - LINE * scroller.clientHeight

const bodyOf = (scroller: HTMLElement) => scroller.querySelector<HTMLElement>('.pane-body')!

/** The space align() put below the text of `scroller`. */
const tail = (scroller: HTMLElement) => parseFloat(bodyOf(scroller).style.getPropertyValue('--tail')) || 0

/**
 * Space that lines up columns a and b side by side: above each pair of matching headings, in the column that reaches
 * it first, so both start at the same height; below the text of the shorter column, so both end together. Then the
 * columns scroll as one. Whole pixels, and left as it is when nothing would move by more than one, so the resizes it
 * causes settle. On a phone, where one sheet covers the other, the space goes.
 */
function align(a: HTMLElement, b: HTMLElement) {
  if (a.closest<HTMLElement>('.stack')?.dataset.mode !== 'desktop') {
    place(a)
    place(b)
    return
  }
  const cols = [a, b].map((scroller) => {
    const hs = headings(scroller)
    let sum = 0
    const now = hs.map((h) => (sum += added(h))) // space down to each heading, its own included
    const bare = hs.map((h, i) => top(scroller, h) - now[i]) // where it would start without any
    const end = scroller.scrollHeight - sum - tail(scroller)
    return { scroller, hs, now, bare, end, shift: new Array<number>(hs.length).fill(NaN), tail: 0 }
  })
  const [A, B] = cols
  let [sa, sb] = [0, 0] // space so far
  for (const [i, j] of matches(A.hs.map(key), B.hs.map(key))) {
    const y = Math.max(A.bare[i] + sa, B.bare[j] + sb)
    A.shift[i] = sa = Math.round(y - A.bare[i])
    B.shift[j] = sb = Math.round(y - B.bare[j])
  }
  for (const c of cols) {
    let s = 0
    c.shift = c.shift.map((v) => (s = isNaN(v) ? s : v)) // a heading without a match gets no space
  }
  const ends = cols.map((c) => c.end + (c.shift.at(-1) ?? 0))
  for (const [k, c] of cols.entries()) c.tail = Math.round(Math.max(...ends) - ends[k])

  const same = (c: (typeof cols)[number]) =>
    Math.abs(c.tail - tail(c.scroller)) <= 1 && c.shift.every((s, i) => Math.abs(s - c.now[i]) <= 1)
  if (!cols.every(same)) for (const c of cols) place(c.scroller, c.shift, c.tail, c.scroller === a)
}

/**
 * Sets the space in `scroller`: above its headings, so much in all down to each (`shift`), and below its text, with
 * `stains`, each once, in big spaces spread down the note that no backdrop (Backdrop.tsx) already fills, and any
 * drawing just before a heading (Doodle.tsx) in the space above it. A column at its very top stays there; otherwise the
 * text on the reading line stays where it is on screen: the last block whose text starts on or above the line, so when the
 * line is in the space above a heading, the text above the space stays and the heading moves.
 */
function place(scroller: HTMLElement, shift: number[] = [], below = 0, stains = false) {
  const body = bodyOf(scroller)
  const fold = scroller.getBoundingClientRect().top + LINE * scroller.clientHeight
  const start = (el: HTMLElement) => el.getBoundingClientRect().top + added(el) // where its text starts
  const blocks = Array.from(body.children as HTMLCollectionOf<HTMLElement>)
  // A column at its very top, such as a note just opened, stays there.
  const anchor = scroller.scrollTop < 1 ? undefined : (blocks.findLast((el) => start(el) <= fold) ?? blocks[0])
  const at = () => (anchor ? start(anchor) : 0)
  const before = at()
  const hs = headings(scroller)
  const spaces = hs.map((_, i) => (shift[i] ?? 0) - (shift[i - 1] ?? 0))
  hs.forEach((h, i) => (h.style.paddingTop = spaces[i] ? `${spaces[i]}px` : ''))
  hs.forEach((h, i) => doodle(h.previousElementSibling, spaces[i]))
  const drawn = stains ? Array.from(body.querySelectorAll('.backdrop-art'), (el) => el.getBoundingClientRect()) : []
  const tops = hs.map((h) => h.getBoundingClientRect().top)
  const fits = hs.flatMap((_, i) => {
    const free = !drawn.some((r) => r.top < tops[i] + spaces[i] && r.bottom > tops[i])
    return stains && free && spaces[i] - 2 * STAIN_PAD >= STAIN_MIN ? [i] : []
  })
  // Each drawing once, spread down the whole note: where the spaces that fit one sit, as shares of its height.
  const box = body.getBoundingClientRect()
  const ys = fits.map((i) => (tops[i] + spaces[i] / 2 - box.top) / box.height)
  const chosen = spread(ys, STAINS).map((j) => fits[j])
  hs.forEach((h, i) => stain(h, chosen.indexOf(i), spaces[i], i))
  if (below) body.style.setProperty('--tail', `${below}px`)
  else body.style.removeProperty('--tail')
  if (anchor) scroller.scrollTop += at() - before
}

/**
 * Indices of n of the increasing shares `ys` (all of them when there are no more than n) that come closest, in sum, to
 * n evenly spread shares, (k + 1/2) / n: the k-th pick goes with the k-th share.
 */
function spread(ys: number[], n: number): number[] {
  if (ys.length <= n) return ys.map((_, i) => i)
  const far = (k: number, i: number) => Math.abs(ys[i] - (k + 0.5) / n)
  let cost = ys.map((_, i) => far(0, i)) // least sum with the k-th pick at i
  const back: number[][] = [] // the pick before it
  for (let k = 1; k < n; k++) {
    const prev = cost
    const from: number[] = []
    let j = -1 // the best earlier pick so far
    cost = ys.map((_, i) => {
      const c = j < 0 ? Infinity : prev[j] + far(k, i)
      from.push(j)
      if (j < 0 || prev[i] < prev[j]) j = i
      return c
    })
    back.push(from)
  }
  let i = cost.indexOf(Math.min(...cost))
  const out = [i]
  for (let k = n - 1; k > 0; k--) out.unshift((i = back[k - 1][i]))
  return out
}

/**
 * Puts drawing k (none when k < 0) in the `space` above heading `h`, the i-th, painted by styles.css. Its size, place
 * and turn look random but are the same for the same heading every time, and keep it clear of the text.
 */
function stain(h: HTMLElement, k: number, space: number, i: number) {
  const vars = ['--stain-size', '--stain-top', '--stain-x', '--stain-turn']
  if (k < 0) {
    delete h.dataset.stain
    for (const v of vars) h.style.removeProperty(v)
    return
  }
  const r = random(`${i} ${key(h)}`)
  const room = space - 2 * STAIN_PAD
  const size = Math.round(Math.min(room, STAIN_MAX * (0.75 + 0.25 * r())))
  const y = Math.round(STAIN_PAD + r() * (room - size))
  const values = [`${size}px`, `${y}px`, r().toFixed(3), `${Math.round(360 * r())}deg`]
  h.dataset.stain = String(k)
  vars.forEach((v, j) => h.style.setProperty(v, values[j]))
}

/** Puts a drawing just before a heading (Doodle.tsx), when `el` is one, in the `space` above it: none hides it. */
function doodle(el: Element | null, space: number) {
  if (!(el instanceof HTMLElement) || !el.classList.contains('doodle')) return
  if (space) {
    el.dataset.space = String(space)
    el.style.setProperty('--space', `${space}px`)
  } else {
    delete el.dataset.space
    el.style.removeProperty('--space')
  }
}

/** Numbers in [0, 1) that look random, the same for the same `seed`: mulberry32 seeded by an FNV-1a hash. */
function random(seed: string) {
  let s = 2166136261
  for (let i = 0; i < seed.length; i++) s = Math.imul(s ^ seed.charCodeAt(i), 16777619)
  return () => {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Scroll positions of columns a and b that go together: both tops, every pair of matching headings on the reading
 * line, both bottoms. A pair is left out when a heading can't reach the line or the pair would break the order, so
 * both coordinates increase. Null when a column doesn't scroll.
 */
function positions(a: HTMLElement, b: HTMLElement): Pair[] | null {
  const end: Pair = [a.scrollHeight - a.clientHeight, b.scrollHeight - b.clientHeight]
  if (end[0] <= 0 || end[1] <= 0) return null
  const ha = headings(a)
  const hb = headings(b)
  const out: Pair[] = [[0, 0]]
  for (const [i, j] of matches(ha.map(key), hb.map(key))) {
    const p: Pair = [onLine(a, ha[i]), onLine(b, hb[j])]
    const last = out.at(-1)!
    if (p[0] > last[0] && p[1] > last[1] && p[0] < end[0] && p[1] < end[1]) out.push(p)
  }
  out.push(end)
  return out
}

/** The position in column `to` (0 = a, 1 = b) that goes with position x of the other column: linear between pairs. */
function along(pairs: Pair[], x: number, to: 0 | 1): number {
  const from = 1 - to
  let k = 1
  while (k < pairs.length - 1 && pairs[k][from] < x) k++
  const [p, q] = [pairs[k - 1], pairs[k]]
  const t = Math.min(1, Math.max(0, (x - p[from]) / (q[from] - p[from])))
  return p[to] + t * (q[to] - p[to])
}

/**
 * Makes scrollers a and b scroll together, b starting where a is. Returns the function that stops it, which takes the
 * space out and leaves both where they are.
 */
export function syncScroll(a: HTMLElement, b: HTMLElement): () => void {
  // The column the reader scrolled, clicked or typed in last. Only its scrolling moves the other: the other also
  // scrolls when it follows, when align() keeps its text in place, or when the browser or Derivation.tsx keep its
  // text in place as a derivation opens or closes, and none of that is the reader.
  let lead = a

  const follow = (from: HTMLElement) => {
    const pairs = positions(a, b)
    if (!pairs) return
    const to = from === a ? b : a
    const y = along(pairs, from.scrollTop, from === a ? 1 : 0)
    if (Math.abs(to.scrollTop - y) >= 1) to.scrollTop = y
  }

  const onInput = (e: Event) => {
    lead = e.currentTarget as HTMLElement
  }
  const onScroll = (e: Event) => {
    if (e.currentTarget !== lead) return
    if (heights() === lined) follow(lead)
    else line()
  }
  const onResize = () => follow(lead)

  // Text that reflows (a new width, a font loaded, the tables of the hiring switch) moves the headings: line them up
  // again, a frame later, since that resizes the columns once more. The lead may scroll before then, as the browser or
  // ResultTable.tsx keep its text in place; onScroll then lines them up first, as following from headings that are
  // still to move would move the other column twice, now and again once they are lined up.
  let frame = 0
  let lined = '' // the columns' heights when they were last lined up
  const heights = () => `${a.scrollHeight} ${b.scrollHeight}`
  const observer = new ResizeObserver(() => {
    cancelAnimationFrame(frame)
    frame = requestAnimationFrame(line)
  })
  const line = () => {
    align(a, b)
    lined = heights()
    follow(lead)
  }

  const inputs = ['wheel', 'pointerdown', 'touchstart', 'keydown', LEAD]
  for (const el of [a, b]) {
    el.addEventListener('scroll', onScroll, { passive: true })
    for (const type of inputs) el.addEventListener(type, onInput, { passive: true })
  }
  addEventListener('resize', onResize)
  line()
  for (const el of [a, b]) observer.observe(bodyOf(el))
  return () => {
    for (const el of [a, b]) {
      el.removeEventListener('scroll', onScroll)
      for (const type of inputs) el.removeEventListener(type, onInput)
    }
    removeEventListener('resize', onResize)
    observer.disconnect()
    cancelAnimationFrame(frame)
    place(a)
    place(b)
  }
}
