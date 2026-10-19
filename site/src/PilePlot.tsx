import {
  Children,
  isValidElement,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from 'react'
import table from '../content/table.json'

type V3 = [number, number, number]
type Axis = 'v' | 'c' | 'h'
type Share = 'q' | 'phi' | 'psi'
type View = { az: number; el: number }
/** A paper: log10 of its idea quality, of its compute in H100e-years and of its research effort in average 2025 papers. */
type Paper = { lv: number; lc: number; lh: number }
/** Where an axis title goes: a point in the canvas and the way out of the box there, the title's nearest side
 *  touching the point; or, with `start`, the point where the title starts, level with its middle. */
type Spot = { x: number; y: number; nx: number; ny: number; start?: boolean }

// ---------- the papers of (24), (30), (31): pause.py's pile_sample, in content/table.json ----------

const { alpha, zeta_p: zeta, sigma_c: sigmaC } = table.inputs
const vFloor = 1 - 1 / alpha // v_m: idea quality averages 1
const lnMeanY = (-zeta * (1 - zeta) * sigmaC ** 2) / 2 // ln E[y_i], y_i = (c_i / c̄)^zeta_p (27c)
const { c_bar: cBar, RE_bar: reBar } = table.pile

/** log10 of a paper's research effort, in average 2025 papers, from log10 of its idea quality and of its compute in
 *  H100e-years: the year's average paper's times v_i y_i / E[y_i] (32a). */
const effort = (lv: number, lc: number) =>
  Math.log10(reBar) + lv + zeta * (lc - Math.log10(cBar)) - lnMeanY / Math.LN10

/** The papers drawn, worth the most first: papers the lab writes in a year of a pause, drawn from (24), (30). */
const PAPERS = [...table.pile.papers].sort((a, b) => b.RE - a.RE)
const SAMPLE: Paper[] = PAPERS.map(({ v, c, RE }) => ({ lv: Math.log10(v), lc: Math.log10(c), lh: Math.log10(RE) }))
const N = SAMPLE.length
/** Sums over the first k papers, k = 0, ..., N. */
const running = (xs: number[]) => xs.reduce((sums, x) => [...sums, sums[sums.length - 1] + x], [0])
const RE_SUM = running(PAPERS.map((p) => p.RE))
const C_SUM = running(PAPERS.map((p) => p.c))

/** When the verifiers check the k papers worth the most, k = 1, ..., N: the bar, as log10 of research effort, halfway
 *  (in logarithms) between the last paper checked and the first one left, and the checked papers' shares of the
 *  research effort, phi_h, and of the compute, psi (31). */
function shares(k: number) {
  return {
    bar: k >= N ? -Infinity : (SAMPLE[k - 1].lh + SAMPLE[k].lh) / 2,
    phi: RE_SUM[k] / RE_SUM[N],
    psi: C_SUM[k] / C_SUM[N],
  }
}

// ---------- the scene: the floor is idea quality x compute, both in logarithms; the height is research effort ----------

const V0 = Math.log10(vFloor)

// Idea quality from its floor v_m, and compute, to a little past the papers' least and most; research effort from the
// floor's least to its most, in decades.
const V1 = Math.max(...SAMPLE.map((p) => p.lv)) + 0.2
const C0 = Math.min(...SAMPLE.map((p) => p.lc)) - 0.2
const C1 = Math.max(...SAMPLE.map((p) => p.lc)) + 0.2
const H0 = Math.floor(effort(V0, C0))
const H1 = Math.ceil(effort(V1, C1) * 2) / 2

const V_TICKS = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].filter(
  (v) => Math.log10(v) > V0 - 1e-9 && Math.log10(v) < V1,
)
const decades = (lo: number, hi: number) =>
  Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, i) => Math.ceil(lo) + i)
const C_TICKS = decades(C0, C1)
const H_TICKS = decades(H0, H1)

/** The box's height, against its sides of 1. */
const ZH = 0.8
const wx = (lv: number) => (lv - V0) / (V1 - V0)
const wy = (lc: number) => (lc - C0) / (C1 - C0)
const wz = (lh: number) => ((lh - H0) / (H1 - H0)) * ZH
/** The height of the surface of research effort over the floor point (x, y). */
const roof = (x: number, y: number) => wz(effort(V0 + x * (V1 - V0), C0 + y * (C1 - C0)))

/** The eye's distance from the box's centre, in the box's sides: a mild perspective. */
const DIST = 3.6
const START: View = { az: (300 * Math.PI) / 180, el: (24 * Math.PI) / 180 }
const EL_MIN = (9 * Math.PI) / 180
const EL_MAX = (72 * Math.PI) / 180
/** Room around the box for the tick labels and titles, px. */
const PAD = { side: 50, top: 30, bottom: 44 }
const CORNERS: V3[] = [0, 1].flatMap((x) => [0, 1].flatMap((y) => [0, ZH].map((z): V3 => [x, y, z])))

/** The eye at the view, looking at the box's centre: a point's place on the picture plane, at unit scale. */
function rig({ az, el }: View) {
  const back: V3 = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)]
  const eye: V3 = [0.5 + DIST * back[0], 0.5 + DIST * back[1], ZH / 2 + DIST * back[2]]
  const f: V3 = [-back[0], -back[1], -back[2]]
  const len = Math.hypot(f[0], f[1])
  const r: V3 = [f[1] / len, -f[0] / len, 0]
  const u: V3 = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
  const flat = ([x, y, z]: V3): [number, number, number] => {
    const d = [x - eye[0], y - eye[1], z - eye[2]]
    const depth = d[0] * f[0] + d[1] * f[1] + d[2] * f[2]
    return [(DIST * (d[0] * r[0] + d[1] * r[1])) / depth, (DIST * (d[0] * u[0] + d[1] * u[1] + d[2] * u[2])) / depth, depth]
  }
  return { eye, f, r, flat }
}

/** The view's projection into a w x h canvas, scaled so the box fits at this elevation whatever the azimuth: it keeps
 *  its size as it turns. */
function camera(view: View, w: number, h: number) {
  const { eye, f, r, flat } = rig(view)
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity]
  for (let k = 0; k < 24; k++) {
    const turned = rig({ az: (k * Math.PI) / 12, el: view.el })
    for (const corner of CORNERS) {
      const [x, y] = turned.flat(corner)
      ;[x0, x1, y0, y1] = [Math.min(x0, x), Math.max(x1, x), Math.min(y0, y), Math.max(y1, y)]
    }
  }
  const s = Math.min((w - 2 * PAD.side) / (x1 - x0), (h - PAD.top - PAD.bottom) / (y1 - y0))
  const [ox, oy] = [w / 2 - (s * (x0 + x1)) / 2, PAD.top + s * y1 + (h - PAD.top - PAD.bottom - s * (y1 - y0)) / 2]
  const project = (p: V3): [number, number, number] => {
    const [x, y, depth] = flat(p)
    return [ox + s * x, oy - s * y, depth]
  }
  return { eye, f, r, project }
}

/** The theme, read from the figure's CSS: colours as rgb(), where the paper is the page under the text, --bg under
 *  --veil, and the sea is --pile-sea (styles.css); and the monospaced face of the labels. */
type Ink = { ink: string; muted: string; paper: string; sea: string; mono: string }

function readInk(el: HTMLElement): Ink {
  const probe = document.createElement('span')
  el.append(probe)
  const dot = document.createElement('canvas').getContext('2d', { willReadFrequently: true })
  /** The colour of the CSS colours painted over one another, as rgb(). */
  const paint = (...colors: string[]) => {
    if (!dot) return colors[0]
    dot.clearRect(0, 0, 1, 1)
    for (const color of colors) {
      probe.style.color = color
      dot.fillStyle = '#000'
      dot.fillStyle = getComputedStyle(probe).color
      dot.fillRect(0, 0, 1, 1)
    }
    const [r, g, b] = dot.getImageData(0, 0, 1, 1).data
    return `rgb(${r} ${g} ${b})`
  }
  const ink = {
    ink: paint('var(--ink)'),
    muted: paint('var(--muted)'),
    paper: paint('var(--bg)', 'var(--veil)'),
    sea: paint('var(--pile-sea)'),
    mono: getComputedStyle(el).getPropertyValue('--mono') || 'monospace',
  }
  probe.remove()
  return ink
}

/** Clips the line p + t d to the unit square: the t range inside it, or undefined (Liang–Barsky). */
function clip(px: number, py: number, dx: number, dy: number): [number, number] | undefined {
  let [t0, t1] = [-Infinity, Infinity]
  for (const [p, d] of [
    [px, dx],
    [py, dy],
  ]) {
    if (Math.abs(d) < 1e-12) {
      if (p < 0 || p > 1) return undefined
      continue
    }
    const [a, b] = [(0 - p) / d, (1 - p) / d]
    t0 = Math.max(t0, Math.min(a, b))
    t1 = Math.min(t1, Math.max(a, b))
  }
  return t0 < t1 ? [t0, t1] : undefined
}

type Pt = [number, number]
const SQUARE: Pt[] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
]

/** The part of a convex polygon where a + b x + c y >= 0 (Sutherland–Hodgman, one half-plane). */
function cut(poly: Pt[], a: number, b: number, c: number): Pt[] {
  const out: Pt[] = []
  poly.forEach((p, i) => {
    const q = poly[(i + 1) % poly.length]
    const [fp, fq] = [a + b * p[0] + c * p[1], a + b * q[0] + c * q[1]]
    if (fp >= 0) out.push(p)
    if (fp >= 0 !== fq >= 0) {
      const t = fp / (fp - fq)
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])])
    }
  })
  return out
}

/** A seeded random number generator (mulberry32), for the ripples' dashes. */
function random(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const tick = (e: number) => (e >= 0 ? (10 ** e).toLocaleString('en-US') : `0.${'0'.repeat(-e - 1)}1`)

/** The surface of research effort is a plane in logarithms: roof(x, y) = A + B x + C y. */
const A = roof(0, 0)
const B = roof(1, 0) - A
const C = roof(0, 1) - A
/** Contours on the island, every CONTOUR decades of research effort. */
const CONTOUR = 0.08

/** Draws the scene; returns where the axis titles go. */
function draw(canvas: HTMLCanvasElement, view: View, bar: number, ink: Ink, sea?: string): Partial<Record<Axis, Spot>> {
  const dpr = window.devicePixelRatio || 1
  const [w, h] = [canvas.clientWidth, canvas.clientHeight]
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
  }
  const ctx = canvas.getContext('2d')
  if (!ctx || w === 0) return {}
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  const { eye, f, r, project: P } = camera(view, w, h)
  const zb = Math.min(ZH, Math.max(0, wz(bar)))
  const path = (pts: V3[], close = false) => {
    ctx.beginPath()
    pts.forEach((p, i) => {
      const [x, y] = P(p)
      if (i) ctx.lineTo(x, y)
      else ctx.moveTo(x, y)
    })
    if (close) ctx.closePath()
  }
  const stroke = (color: string, width: number, alpha = 1) => {
    ctx.strokeStyle = color
    ctx.lineWidth = width
    ctx.globalAlpha = alpha
    ctx.stroke()
    ctx.globalAlpha = 1
  }
  const fill = (color: string, alpha = 1) => {
    ctx.fillStyle = color
    ctx.globalAlpha = alpha
    ctx.fill()
    ctx.globalAlpha = 1
  }
  const onRoof = ([x, y]: Pt): V3 => [x, y, A + B * x + C * y]
  /** The contour of the surface at height z, as a segment across the floor, if it crosses it. */
  const contour = (z: number): [V3, V3] | undefined => {
    const n = B * B + C * C
    const [px, py] = [((z - A) * B) / n, ((z - A) * C) / n]
    const span = clip(px, py, C, -B)
    return span && [onRoof([px + span[0] * C, py - span[0] * B]), onRoof([px + span[1] * C, py - span[1] * B])]
  }
  const dot = (p: V3, color: string, radius: number, halo: number, alpha = 1) => {
    const [sx, sy] = P(p)
    if (halo) {
      ctx.beginPath()
      ctx.arc(sx, sy, radius + halo, 0, 2 * Math.PI)
      fill(halo > 0 ? ink.paper : ink.sea)
    }
    ctx.beginPath()
    ctx.arc(sx, sy, radius, 0, 2 * Math.PI)
    fill(color, alpha)
  }

  /** Writes on the sea's surface, in perspective, along its near edge that runs closer to level on the screen. */
  const write = (text: string) => {
    let best: { s: V3; dir: V3; down: V3; slope: number } | undefined
    for (const [a, b, out] of [
      [[0, yNear, zb], [1, yNear, zb], [0, yNear ? 1 : -1, 0]],
      [[xNear, 0, zb], [xNear, 1, zb], [xNear ? 1 : -1, 0, 0]],
    ] as [V3, V3, V3][]) {
      const [s, e] = P(a)[0] < P(b)[0] ? [a, b] : [b, a]
      const slope = Math.abs((P(e)[1] - P(s)[1]) / (P(e)[0] - P(s)[0] || 1e-9))
      if (!best || slope < best.slope)
        best = { s, dir: [e[0] - s[0], e[1] - s[1], 0], down: out, slope }
    }
    if (!best) return
    const { s, dir, down } = best
    // 1 px of text is UNIT of the box's side; the text starts in from the edge's left end and its baseline sits in
    // from the edge
    const UNIT = 0.0032
    const at = (u: number, v: number): V3 => [
      s[0] + 0.1 * dir[0] + u * UNIT * dir[0] - (0.045 - v * UNIT) * down[0],
      s[1] + 0.1 * dir[1] + u * UNIT * dir[1] - (0.045 - v * UNIT) * down[1],
      zb,
    ]
    const [ox, oy] = P(at(0, 0))
    const [ux, uy] = P(at(100, 0))
    const [vx, vy] = P(at(0, 100))
    ctx.save()
    ctx.setTransform(
      (dpr * (ux - ox)) / 100,
      (dpr * (uy - oy)) / 100,
      (dpr * (vx - ox)) / 100,
      (dpr * (vy - oy)) / 100,
      dpr * ox,
      dpr * oy,
    )
    ctx.font = `500 11px ${ink.mono}`
    ctx.letterSpacing = '1.5px'
    const width = ctx.measureText(text).width
    ctx.fillStyle = ink.sea
    ctx.fillRect(-6, -11, width + 12, 15)
    ctx.fillStyle = ink.paper
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    ctx.fillText(text, 0, 0)
    ctx.restore()
  }

  // the box's walls and floor edges nearest the eye, and the ones away from it
  const xNear = eye[0] < 0.5 ? 0 : 1
  const yNear = eye[1] < 0.5 ? 0 : 1
  const [xFar, yFar] = [1 - xNear, 1 - yNear]

  // Back walls, above the sea: the decades of research effort, ruled like a facade.
  for (const e of H_TICKS) {
    if (wz(e) < zb) continue
    path([
      [xNear, yFar, wz(e)],
      [xFar, yFar, wz(e)],
      [xFar, yNear, wz(e)],
    ])
    stroke(ink.muted, 0.75, 0.5)
  }
  for (const [x, y] of [
    [xNear, yFar],
    [xFar, yFar],
    [xFar, yNear],
  ]) {
    path([
      [x, y, zb],
      [x, y, ZH],
    ])
    stroke(ink.muted, 1, 0.7)
  }

  // The floor, seen only when every paper is checked.
  for (const v of V_TICKS) {
    const x = wx(Math.log10(v))
    path([
      [x, 0, 0],
      [x, 1, 0],
    ])
    stroke(ink.muted, 0.75, 0.5)
  }
  for (const e of C_TICKS) {
    path([
      [0, wy(e), 0],
      [1, wy(e), 0],
    ])
    stroke(ink.muted, 0.75, 0.5)
  }

  const papers = SAMPLE.map((p) => {
    const at: V3 = [wx(p.lv), wy(p.lc), wz(p.lh)]
    return { at, depth: P(at)[2], up: p.lh >= bar }
  }).sort((a, b) => b.depth - a.depth)
  const [near, far] = papers.length ? [papers[papers.length - 1].depth, papers[0].depth] : [0, 1]
  const fade = (depth: number) => 0.95 - 0.55 * ((depth - near) / (far - near || 1))

  if (zb > 0) {
    // The sea: every paper below the bar, under black water. Its sides facing the eye, then its surface.
    const sides: V3[][] = []
    if (eye[0] < 0 || eye[0] > 1)
      sides.push([
        [xNear, 0, 0],
        [xNear, 1, 0],
        [xNear, 1, zb],
        [xNear, 0, zb],
      ])
    if (eye[1] < 0 || eye[1] > 1)
      sides.push([
        [0, yNear, 0],
        [1, yNear, 0],
        [1, yNear, zb],
        [0, yNear, zb],
      ])
    const surface: V3[] = SQUARE.map(([x, y]) => [x, y, zb])
    for (const side of sides) {
      path(side, true)
      fill(ink.sea)
    }
    path(surface, true)
    fill(ink.sea)
    // faint level lines on its sides at the decades of research effort, to read depths by
    for (const e of H_TICKS) {
      const z = wz(e)
      if (z >= zb || z <= 0) continue
      for (const side of sides) {
        path([
          [side[0][0], side[0][1], z],
          [side[1][0], side[1][1], z],
        ])
        stroke(ink.paper, 0.75, 0.3)
      }
    }

    // Under the sea, seen through it: the surface of research effort, a shade lighter than the water, and the
    // papers on it.
    const deep = cut(cut(SQUARE, zb - A, -B, -C), A, B, C)
    if (deep.length > 2) {
      path(deep.map(onRoof), true)
      fill(ink.paper, 0.1)
      stroke(ink.paper, 0.75, 0.3)
    }
    for (const p of papers) {
      if (p.up) continue
      dot(p.at, ink.paper, 1.5, 0, fade(p.depth))
    }

    // Ripples, level across the view, in broken strokes.
    const fl = Math.hypot(f[0], f[1])
    const [fx, fy] = [f[0] / fl, f[1] / fl]
    const rand = random(7)
    for (let s = -0.75; s <= 0.75; s += 1 / 30) {
      const [px, py] = [0.5 + s * fx, 0.5 + s * fy]
      const span = clip(px, py, r[0], r[1])
      if (!span) continue
      ctx.setLineDash([4 + 14 * rand(), 3 + 6 * rand(), 2 + 8 * rand(), 3 + 5 * rand()])
      ctx.lineDashOffset = 20 * rand()
      path([
        [px + span[0] * r[0], py + span[0] * r[1], zb],
        [px + span[1] * r[0], py + span[1] * r[1], zb],
      ])
      stroke(ink.paper, 0.8, 0.55)
    }
    ctx.setLineDash([])
    if (sea) write(sea.toUpperCase())

    // White edges where the sea's faces meet, as on a cut black form.
    for (const side of sides) {
      path([side[3], side[2]])
      stroke(ink.paper, 1.25)
    }
    if (sides.length === 2) {
      path([
        [xNear, yNear, 0],
        [xNear, yNear, zb],
      ])
      stroke(ink.paper, 1.25)
    }
  }

  // The island: the surface of research effort above the bar, where the checked papers are, hatched along its
  // contours like a map, with the shore's wash on the sea around it.
  const island = cut(cut(SQUARE, A - zb, B, C), ZH - A, -B, -C)
  if (island.length > 2) {
    if (zb > 0) {
      for (const [k, lift] of [
        [1, 0.012],
        [2, 0.026],
      ]) {
        const seg = contour(zb - lift)
        if (!seg) continue
        path([
          [seg[0][0], seg[0][1], zb],
          [seg[1][0], seg[1][1], zb],
        ])
        stroke(ink.paper, 1.2 / k, 0.8)
      }
    }
    path(island.map(onRoof), true)
    fill(ink.paper)
    // heavier near the shore and finer toward the top, as an engraver shades a form
    const step = (CONTOUR / (H1 - H0)) * ZH
    const top = Math.max(...island.map((p) => onRoof(p)[2]))
    for (let z = zb + step; z < top; z += step) {
      const seg = contour(z)
      if (!seg) continue
      path(seg)
      stroke(ink.ink, 1.5 - (0.9 * (z - zb)) / (top - zb || 1), 0.9)
    }
    path(island.map(onRoof), true)
    stroke(ink.ink, 1.5)
    // where the box's walls cut it, a heavy edge, as on a carved block
    island.forEach((p, i) => {
      const q = island[(i + 1) % island.length]
      const onWall = (k: 0 | 1) => (Math.abs(p[k] - q[k]) < 1e-9 && (p[k] < 1e-9 || p[k] > 1 - 1e-9))
      if (!onWall(0) && !onWall(1)) return
      path([onRoof(p), onRoof(q)])
      stroke(ink.ink, 3)
    })
  }
  for (const p of papers) if (p.up) dot(p.at, ink.ink, 2.8, 1.5)

  // Tick labels along the floor's near edges and up the side edge further left on the screen; each floor axis's title
  // goes past its labels, across the middle of its edge.
  ctx.font = `400 10px ${ink.mono}`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.strokeStyle = ink.paper
  ctx.lineWidth = 3
  /** A tick label, with a rim of paper where it crosses a line. */
  const label3 = (text: string, x: number, y: number) => {
    ctx.strokeText(text, x, y)
    ctx.fillStyle = ink.muted
    ctx.fillText(text, x, y)
  }
  const [mx, my] = P([0.5, 0.5, 0])
  const row = (a: V3, b: V3, ticks: [number, string][], nearEnd: number, gap: number): Spot => {
    const [ax, ay] = P(a)
    const [bx, by] = P(b)
    let [nx, ny] = [by - ay, ax - bx]
    const len = Math.hypot(nx, ny) || 1
    ;[nx, ny] = [nx / len, ny / len]
    if (nx * ((ax + bx) / 2 - mx) + ny * ((ay + by) / 2 - my) < 0) [nx, ny] = [-nx, -ny]
    const one = Math.max(0, ticks.findIndex(([, label]) => label === '1'))
    const labels = ticks
      .map(([t, label], i) => {
        const [x, y] = [ax + t * (bx - ax) + gap * nx, ay + t * (by - ay) + gap * ny]
        return { t, label, i: i - one, x, y, w: ctx.measureText(label).width }
      })
      // none at the near corner, where the two rows meet
      .filter(({ t }) => Math.abs(t - nearEnd) >= 0.07)
    // every label, or every second, third...: the fewest skipped so that none touch, always with the one at 1
    const every = (k: number) => labels.filter(({ i }) => i % k === 0)
    const touch = (p: (typeof labels)[number], q: (typeof labels)[number]) =>
      Math.abs(p.x - q.x) < (p.w + q.w) / 2 + 4 && Math.abs(p.y - q.y) < 12
    let k = 1
    while (k < labels.length && every(k).some((p, j, all) => j > 0 && touch(all[j - 1], p))) k++
    let reach = gap
    for (const { label, x, y, w } of every(k)) {
      label3(label, x, y)
      reach = Math.max(reach, gap + (Math.abs(nx) * w) / 2 + Math.abs(ny) * 6)
    }
    return { x: (ax + bx) / 2 + (reach + 6) * nx, y: (ay + by) / 2 + (reach + 6) * ny, nx, ny }
  }
  const v = row([0, yNear, 0], [1, yNear, 0], V_TICKS.map((x) => [wx(Math.log10(x)), String(x)]), xNear, 12)
  const c = row([xNear, 0, 0], [xNear, 1, 0], C_TICKS.map((e) => [wy(e), tick(e)]), yNear, 14)
  const corners: V3[] = [
    [xNear, yFar, 0],
    [xFar, yNear, 0],
  ]
  const side = P(corners[0])[0] < P(corners[1])[0] ? corners[0] : corners[1]
  ctx.textAlign = 'right'
  for (const e of H_TICKS) {
    const [x, y] = P([side[0], side[1], wz(e)])
    label3(tick(e), x - 8, y)
  }
  // the height's title above its top tick, starting where the tick labels start
  const [tx, ty] = P([side[0], side[1], ZH])
  const lead = Math.max(...H_TICKS.map((e) => ctx.measureText(tick(e)).width))
  return { v, c, h: { x: tx - 8 - lead, y: ty - 18, nx: 0, ny: -1, start: true } }
}

const AXES: Axis[] = ['v', 'c', 'h']
const METERS: Share[] = ['phi', 'psi']

function percent(p: number) {
  const x = 100 * p
  if (x > 99 && x < 100) return `${Math.min(+x.toFixed(2), 99.99)}%` // not 100% until it is
  return `${x < 1 ? +x.toFixed(2) : Math.round(x)}%`
}

/**
 * `<PilePlot>` in an .mdx file: the surface of a paper's research effort over its idea quality and compute (32a), and
 * pause.py's sample of the lab's papers as dots on it, drawn as an ink plate. A sea fills the box up to the bar: the
 * papers under it are left unchecked and show through it in white; the island above it holds the checked papers. The
 * reader sets the share of the papers checked, q, on a meter with a handle, a paper at a time; meters under it show
 * the checked papers' shares of the dots' research effort and compute (31). Drag (or arrow keys) to turn it. The
 * children are labels, by data attribute, then the caption:
 *
 *   <span data-axis="v">idea quality $v_i$</span>    an axis title: v, c (compute) or h (research effort)
 *   <span data-sea>the bar</span>    written on the sea, plain text
 *   <span data-share="phi">research effort $\varphi_h$</span>    a meter's label: q (the slider's), phi or psi
 *
 * In .mdx files it needs no import (Pane.tsx provides it).
 */
export function PilePlot({ children }: { children?: ReactNode }) {
  const titles: Partial<Record<Axis, ReactNode>> = {}
  const names: Partial<Record<Share, ReactNode>> = {}
  let sea: string | undefined
  const caption: ReactNode[] = []
  type Label = {
    'data-axis'?: Axis
    'data-share'?: Share
    'data-sea'?: unknown
    children?: ReactNode
  }
  Children.forEach(children, (child) => {
    if (isValidElement<Label>(child)) {
      const { 'data-axis': axis, 'data-share': share, children: label } = child.props
      if (axis) return void (titles[axis] = label)
      if (share) return void (names[share] = label)
      if (child.props['data-sea'] !== undefined) return void (sea = typeof label === 'string' ? label : undefined)
    }
    caption.push(child)
  })

  const [k, setK] = useState(Math.round(0.2 * N)) // the papers checked
  const q = k / N
  const { bar, phi, psi } = shares(k)
  const values: Record<Share, number> = { q, phi, psi }

  const figure = useRef<HTMLElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const view = useRef<View>(START)
  const ink = useRef<Ink>(undefined)
  // what the last render showed, for redraws between renders (turning, resizing)
  const shown = useRef({ bar, sea })

  const render = () => {
    const c = canvas.current
    const fig = figure.current
    if (!c || !fig) return
    ink.current ??= readInk(fig)
    const spots = draw(c, view.current, shown.current.bar, ink.current, shown.current.sea)
    for (const axis of AXES) {
      const el = fig.querySelector<HTMLElement>(`.pile-plot-title[data-axis="${axis}"]`)
      const at = spots[axis]
      if (!el || !at) continue
      // out from the spot, so that the title's nearest side touches it
      const [w, h] = [el.offsetWidth, el.offsetHeight]
      const out = at.start ? 0 : (Math.abs(at.nx) * w) / 2 + (Math.abs(at.ny) * h) / 2
      const left = at.start ? at.x : at.x + at.nx * out - w / 2
      el.style.transform = `translate(${left}px, ${at.y + at.ny * out - h / 2}px)`
    }
  }

  useLayoutEffect(() => {
    shown.current = { bar, sea }
    render()
  })

  useEffect(() => {
    const fig = figure.current
    if (!fig) return
    const observer = new ResizeObserver(() => render())
    observer.observe(fig)
    const scheme = matchMedia('(prefers-color-scheme: dark)')
    const recolor = () => {
      ink.current = undefined
      render()
    }
    scheme.addEventListener('change', recolor)
    document.fonts.ready.then(() => render())
    return () => {
      observer.disconnect()
      scheme.removeEventListener('change', recolor)
    }
  }, [])

  const drag = useRef<{ x: number; y: number; from: View }>(undefined)
  const turn = (to: View) => {
    view.current = { az: to.az, el: Math.min(EL_MAX, Math.max(EL_MIN, to.el)) }
    render()
  }
  const onDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return
    drag.current = { x: e.clientX, y: e.clientY, from: view.current }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const onMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current
    if (!d) return
    // on a touch screen, an upward drag scrolls the page (touch-action), so only sideways drags turn it
    const tilt = e.pointerType === 'touch' ? 0 : e.clientY - d.y
    turn({ az: d.from.az - (e.clientX - d.x) * 0.008, el: d.from.el + tilt * 0.006 })
  }
  const onUp = () => (drag.current = undefined)
  const onKey = (e: KeyboardEvent<HTMLCanvasElement>) => {
    const step = { ArrowLeft: [0.08, 0], ArrowRight: [-0.08, 0], ArrowUp: [0, 0.05], ArrowDown: [0, -0.05] }[e.key]
    if (!step) return
    e.preventDefault()
    turn({ az: view.current.az + step[0], el: view.current.el + step[1] })
  }

  return (
    <figure className="pile-plot" ref={figure}>
      <div className="pile-plot-stage">
        <canvas
          ref={canvas}
          tabIndex={0}
          role="img"
          aria-label="The surface of a paper's research effort over its idea quality and compute, with the year's papers on it; a sea fills the box up to the bar, and the papers above it are checked. Drag or use the arrow keys to turn it."
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onKeyDown={onKey}
        />
        {AXES.map(
          (axis) =>
            titles[axis] && (
              <div key={axis} className="pile-plot-title" data-axis={axis} aria-hidden="true">
                {titles[axis]}
              </div>
            ),
        )}
      </div>
      <div className="pile-plot-controls">
        <label className="pile-plot-meter" data-share="q">
          <span className="pile-plot-name">{names.q}</span>
          <span className="pile-plot-bar">
            <span className="pile-plot-fill" style={{ width: `${100 * q}%` }} />
            <input
              type="range"
              min={1}
              max={N}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              aria-valuetext={`${k} of the ${N} papers checked, ${percent(q)}`}
            />
          </span>
          <span className="pile-plot-value">{percent(q)}</span>
        </label>
        {METERS.map((share) => (
          <div key={share} className="pile-plot-meter" data-share={share}>
            <span className="pile-plot-name">{names[share]}</span>
            <span
              className="pile-plot-bar"
              role="meter"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(100 * values[share])}
            >
              <span className="pile-plot-fill" style={{ width: `${100 * values[share]}%` }} />
            </span>
            <span className="pile-plot-value">{percent(values[share])}</span>
          </div>
        ))}
      </div>
      {caption.length > 0 && <figcaption>{caption}</figcaption>}
    </figure>
  )
}
