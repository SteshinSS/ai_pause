// The link-preview picture and the favicons, from two drawings. Needs ImageMagick 7 (magick).
//
//   node scripts/icons.mjs <social.png> <favicon.png>
//
// - public/og.jpg: the preview picture (scripts/prerender.mjs tags it), 1200 × 630, the shape previews use, cut from
//   the middle. A JPEG of about 200 kB: some apps (WhatsApp) show no picture when it's big.
// - public/favicon.ico (16, 32 and 48 px) and public/favicon-96.png: the drawing cut to its outline and centred in a
//   square, with a thin edge in the ink colour, so that the white mask shows on a white tab bar too.
// - public/apple-touch-icon.png (180 px), for iPhone home screens: on the dark mode's background (iOS fills
//   transparency with black anyway).
// The originals live in art/raw/share/ (not in git).
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const [social, favicon] = process.argv.slice(2)
if (!social || !favicon) {
  console.error('usage: node scripts/icons.mjs <social.png> <favicon.png>')
  process.exit(1)
}
const PUBLIC = new URL('../public/', import.meta.url).pathname
const INK = '#161616' // --ink in src/styles.css, light mode
const NIGHT = '#131313' // --bg in src/styles.css, dark mode
const magick = (...args) => execFileSync('magick', args, { stdio: 'inherit' })

magick(social, '-resize', '1200x630^', '-gravity', 'center', '-extent', '1200x630', '-strip', '-interlace', 'JPEG',
  '-sampling-factor', '4:2:0', '-quality', '86', '-type', 'TrueColor', `${PUBLIC}og.jpg`)

// A square of the drawing: trimmed to what it draws (near-invisible specks cleared first), centred, transparent around it.
const tmp = mkdtempSync(join(tmpdir(), 'icons-'))
const square = join(tmp, 'square.png')
magick(favicon, '-channel', 'A', '-fx', 'a<0.02?0:a', '+channel', '-trim', '+repage', '-background', 'none', '-gravity', 'center', '-extent', '%[fx:max(w,h)*1.03]x%[fx:max(w,h)*1.03]', square)

// At `size` px, with an edge `edge` px wide.
const edged = (size, edge, out) =>
  magick(square, '-resize', `${size}x${size}`, '(', '+clone', '-alpha', 'extract', '-morphology', 'Dilate',
    `Disk:${edge}`, '-background', INK, '-alpha', 'shape', ')', '-compose', 'DstOver', '-composite', out)
const sizes = [[16, 1], [32, 1.5], [48, 2]].map(([size, edge]) => {
  const out = join(tmp, `${size}.png`)
  edged(size, edge, out)
  return out
})
magick(...sizes, `${PUBLIC}favicon.ico`)
edged(96, 3, `${PUBLIC}favicon-96.png`)
magick(square, '-resize', '144x144', '-background', NIGHT, '-gravity', 'center', '-extent', '180x180', '-alpha', 'remove',
  `${PUBLIC}apple-touch-icon.png`)
rmSync(tmp, { recursive: true })
