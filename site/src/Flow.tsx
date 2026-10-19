import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'

/** A box's edges, relative to the diagram's top left corner. */
type Box = { left: number; right: number; top: number; bottom: number }

type Arrow = { d: string; back: boolean }

const middle = (box: Box) => (box.top + box.bottom) / 2

const center = (box: Box) => (box.left + box.right) / 2

/** Whether the box crosses the horizontal line at y between x0 and x1. */
const crosses = (box: Box, y: number, x0: number, x1: number) =>
  box.top < y && y < box.bottom && box.left < x1 && box.right > x0

/**
 * `<Flow label="…">` in an .mdx file: a diagram of boxes and the arrows between them. The boxes are the elements
 * inside it with `data-node="id"`, laid out by CSS (styles.css, .flow), and `data-to="id id"` on a box names the
 * boxes its arrows point to. The arrows are measured from the boxes and drawn over them, so they follow any width:
 * sideways to a box beside, straight down to a box below, along the left margin when other boxes are in the way
 * (and under the diagram when boxes block the way in from the left), and back up along the right margin.
 *
 * `<Flow ledger>` lays the boxes out in three columns: `.flow-left` and `.flow-right` boxes at the sides, and a
 * `.flow-card` in the middle whose rows (its children) line up with the side boxes' rows. In .mdx files it needs no
 * import (Pane.tsx provides it).
 */
export function Flow({ label, ledger, children }: { label: string; ledger?: boolean; children: ReactNode }) {
  const figure = useRef<HTMLElement>(null)
  const [arrows, setArrows] = useState<Arrow[]>([])
  const head = `flow-head${useId().replace(/[^\w-]/g, '')}`

  useLayoutEffect(() => {
    const root = figure.current
    if (!root) return
    // A card spans as many of the diagram's rows as it has rows of its own (styles.css, .flow-card).
    root.querySelectorAll<HTMLElement>(':scope > .flow-card').forEach((card) => {
      card.style.setProperty('--rows', `${card.children.length}`)
      card.style.setProperty('--rows-2', `${2 * card.children.length}`)
    })
    const draw = () => {
      const origin = root.getBoundingClientRect()
      const boxes = new Map<string, Box>()
      const links: [string, string][] = []
      for (const el of root.querySelectorAll<HTMLElement>('[data-node]')) {
        const r = el.getBoundingClientRect()
        const id = el.dataset.node ?? ''
        boxes.set(id, {
          left: r.left - origin.left,
          right: r.right - origin.left,
          top: r.top - origin.top,
          bottom: r.bottom - origin.top,
        })
        for (const to of (el.dataset.to ?? '').split(/\s+/).filter(Boolean)) links.push([id, to])
      }
      const style = getComputedStyle(root)
      const laneLeft = parseFloat(style.paddingLeft) / 2
      const laneRight = origin.width - parseFloat(style.paddingRight) / 2
      const laneBottom = Math.max(...[...boxes.values()].map((box) => box.bottom)) + laneLeft
      const next = links.flatMap(([from, to]): Arrow[] => {
        const a = boxes.get(from)
        const b = boxes.get(to)
        if (!a || !b) return []
        const others = [...boxes].filter(([id]) => id !== from && id !== to).map(([, c]) => c)
        if (b.top < a.bottom && a.top < b.bottom) {
          const y = (Math.max(a.top, b.top) + Math.min(a.bottom, b.bottom)) / 2
          if (b.left >= a.right) return [{ d: `M${a.right} ${y}H${b.left}`, back: false }]
          if (b.right <= a.left) return [{ d: `M${a.left} ${y}H${b.right}`, back: false }]
        }
        if (b.top < a.bottom) return [{ d: `M${a.right} ${middle(a)}H${laneRight}V${middle(b)}H${b.right}`, back: true }]
        const overlap = Math.max(a.left, b.left) < Math.min(a.right, b.right)
        const x = (Math.max(a.left, b.left) + Math.min(a.right, b.right)) / 2
        const inTheWay = others.some((c) => c.left < x && x < c.right && c.top < b.top && c.bottom > a.bottom)
        if (overlap && !inTheWay) return [{ d: `M${x} ${a.bottom}V${b.top}`, back: false }]
        const down = `M${a.left} ${middle(a)}H${laneLeft}`
        if (!others.some((c) => crosses(c, middle(b), laneLeft, b.left)))
          return [{ d: `${down}V${middle(b)}H${b.left}`, back: false }]
        // Under the diagram and up into the box, or, with boxes below it, on round to the right margin.
        const under = `${down}V${laneBottom}`
        if (!others.some((c) => c.left < center(b) && center(b) < c.right && c.top >= b.bottom))
          return [{ d: `${under}H${center(b)}V${b.bottom}`, back: false }]
        return [{ d: `${under}H${laneRight}V${middle(b)}H${b.right}`, back: false }]
      })
      setArrows((old) => (JSON.stringify(old) === JSON.stringify(next) ? old : next))
    }
    draw()
    // Fonts, KaTeX and the column's width all move the boxes.
    const observer = new ResizeObserver(draw)
    observer.observe(root)
    root.querySelectorAll('[data-node]').forEach((el) => observer.observe(el))
    return () => observer.disconnect()
  }, [])

  return (
    // The frame is the container whose width a ledger's layout queries (styles.css, .flow-frame).
    <div className="flow-frame">
      <figure className={ledger ? 'flow flow-ledger' : 'flow'} ref={figure} aria-label={label}>
        {children}
        <svg className="flow-arrows" aria-hidden="true">
          <defs>
            <marker id={head} viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0 0.5L8 4L0 7.5z" />
            </marker>
          </defs>
          {arrows.map(({ d, back }) => (
            <path key={d} d={d} className={back ? 'flow-back' : undefined} markerEnd={`url(#${head})`} />
          ))}
        </svg>
      </figure>
    </div>
  )
}
