// After `vite build` (npm run build), what search engines, link previews and AI tools read, as they don't run the
// page's JavaScript:
// - the main text in dist/index.html's #root, rendered by src/prerender.ts. A browser that runs the page's JavaScript
//   hides it (index.html), and React replaces it, so readers see the page as before.
// - a page for each note, dist/n/<id>/index.html (src/url.ts): index.html with the note's title, its description for
//   search results (its frontmatter's, else its first paragraph) and its text in #root. Link previews keep the site's
//   description. The page opens the note beside the main text.
// - link-preview tags that need the site's address (vite.config.ts): og:url and the canonical address, and og:image
//   when public/og.jpg (or og.png; scripts/icons.mjs) exists. The image's address ends in a hash of the file, so
//   previews pick up a new one.
// - robots.txt, which lets every crawler in, and sitemap.xml: the main text, each note's page and the PDF
//   (scripts/pdf.mjs), each with the date git last changed its file.
// - Google Scholar's citation_ tags (index.html), which describe the PDF, only on the main page: on the note pages,
//   Scholar would take each note for another copy of the work.
import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'vite'

const root = fileURLToPath(new URL('..', import.meta.url))
const dist = `${root}dist/`
const out = `${root}node_modules/.cache/prerender/`

await build({
  root,
  logLevel: 'warn',
  build: { ssr: 'src/prerender.ts', outDir: out, emptyOutDir: true, copyPublicDir: false },
})
const { SITE_URL, SITE_NAME, mainHtml, notePages } = await import(pathToFileURL(`${out}prerender.js`).href)

/** Width and height of a PNG (from its header) or a JPEG (from its frame marker). */
function sizeOf(file) {
  if (file.toString('latin1', 12, 16) === 'IHDR') return { width: file.readUInt32BE(16), height: file.readUInt32BE(20) }
  for (let at = 2; at + 9 < file.length; at += 2 + file.readUInt16BE(at + 2)) {
    const marker = file[at + 1]
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker))
      return { width: file.readUInt16BE(at + 7), height: file.readUInt16BE(at + 5) }
  }
  throw new Error('prerender: the preview picture is neither a PNG nor a JPEG')
}

const escape = (text) =>
  text.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ' }
const unescape = (text) =>
  text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (all, name) =>
    name[0] === '#' ? String.fromCodePoint(Number(`0${name.slice(1)}`)) : (ENTITIES[name] ?? all),
  )

const VOID = new Set(['area', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr'])

/** The text of rendered HTML as it reads, a formula as its TeX. */
function textOf(html) {
  let text = ''
  let math = 0 // depth inside a formula (KaTeX), whose text is only its TeX, in <annotation>
  let tex = false
  for (const [, close, tag, attrs, self, chars] of html.matchAll(
    /<!--[\s\S]*?-->|<(\/?)([a-zA-Z][\w-]*)([^>]*?)(\/?)>|([^<]+)/g,
  )) {
    if (chars !== undefined) {
      if (!math || tex) text += chars
    } else if (!tag || self || VOID.has(tag)) {
      // a comment, or an element without content
    } else if (close) {
      if (tag === 'annotation') tex = false
      if (math) math--
    } else if (math || /\bclass="katex"/.test(attrs)) {
      math++
      if (tag === 'annotation') tex = true
    }
  }
  return unescape(text).replace(/\s+/g, ' ').trim()
}

/** A note page's description: its first paragraph long enough to say what the note is about (not "Open it side by
 *  side with ..."), its first sentences up to at least 100 characters; a sentence too long for a search result is cut
 *  to its words within 160. */
function describe(html) {
  const paragraphs = [...html.matchAll(/<p(?:\s[^>]*)?>([\s\S]*?)<\/p>/g)].map(([, p]) => textOf(p))
  const longest = paragraphs.reduce((a, b) => (b.length > a.length ? b : a), '')
  const text = paragraphs.find((p) => p.length >= 80) ?? longest
  const end = [...text.matchAll(/[.!?]['")\]]*(?=\s)/g)].map((m) => m.index + m[0].length).find((at) => at >= 100)
  const cut = text.slice(0, end)
  return cut.length <= 300 ? cut : `${text.slice(0, text.lastIndexOf(' ', 159))}…`
}

/** The date (YYYY-MM-DD) git last changed `file`. None from a shallow clone, where that would be its last commit's. */
function changed(file) {
  const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: 'pipe' }).trim()
  try {
    if (git('rev-parse', '--is-shallow-repository') !== 'false') return undefined
    return git('log', '-1', '--format=%cs', '--', file) || undefined
  } catch {
    return undefined
  }
}

const images = [] // the preview picture's tags, as parts for page()
const image = ['og.jpg', 'og.png'].find((name) => existsSync(dist + name))
if (image) {
  const file = readFileSync(dist + image)
  const hash = createHash('sha256').update(file).digest('hex').slice(0, 8)
  const { width, height } = sizeOf(file)
  // before index.html's og:image:alt, which belongs to the og:image above it
  images.push([
    '<meta property="og:image:alt"',
    `<meta property="og:image" content="${escape(SITE_URL)}/${image}?v=${hash}" />
    <meta property="og:image:width" content="${width}" />
    <meta property="og:image:height" content="${height}" />
    <meta property="og:image:alt"`,
  ])
} else console.warn('prerender: no public/og.jpg, so link previews show no picture')

const template = readFileSync(`${dist}index.html`, 'utf8')

/** index.html as the page at `path`, with `text` in #root and each [from, to] of `parts` replaced (from: a string or a
 *  regular expression, to: a string or, for a regular expression with groups, a function of the match). Fails when
 *  index.html has no `from`. */
function page(path, text, parts = []) {
  const url = escape(SITE_URL + path)
  let html = template
  for (const [from, to] of [
    ...parts,
    ...images,
    ['</head>', `  <link rel="canonical" href="${url}" />\n    <meta property="og:url" content="${url}" />\n  </head>`],
    ['<div id="root"></div>', `<div id="root"><div class="prerendered">${text}</div></div>`],
  ]) {
    if (!(typeof from === 'string' ? html.includes(from) : from.test(html)))
      throw new Error(`prerender: dist/index.html has no ${from}`)
    html = html.replace(from, typeof to === 'function' ? to : () => to)
  }
  return html
}

const main = mainHtml()
writeFileSync(`${dist}index.html`, page('/', main))

const LD = /(<script type="application\/ld\+json">)([\s\S]*?)(<\/script>)/
const SCHOLAR = /\s*<!-- Google Scholar:[\s\S]*?<meta name="citation_pdf_url"[^>]*>/
const ld = JSON.parse(LD.exec(template)?.[2] ?? 'null')
const notes = notePages()
for (const note of notes) {
  const description = note.description ?? describe(note.html)
  if (!description) throw new Error(`prerender: note ${note.id} has no text for its description`)
  // The structured data says what the page is: the note, a part of the work the main text is.
  const data = ld && {
    ...ld,
    headline: note.name,
    description,
    url: SITE_URL + note.path,
    isPartOf: { '@type': ld['@type'], headline: ld.headline, url: `${SITE_URL}/` },
  }
  const json = JSON.stringify(data, null, 2).replaceAll('<', '\\u003c').replaceAll('\n', '\n      ')
  const html = page(
    note.path,
    `<nav><a href="/">${escape(SITE_NAME)}</a></nav>${note.html}`,
    [
      [/<title>[^<]*<\/title>/, `<title>${escape(note.title)}</title>`],
      [/(<meta name="description" content=")[^"]*/, (_, tag) => tag + escape(description)],
      [/(<meta property="og:title" content=")[^"]*/, (_, tag) => tag + escape(note.name)],
      ...(data ? [[LD, (_, open, _old, close) => `${open}\n      ${json}\n    ${close}`]] : []),
      ...(SCHOLAR.test(template) ? [[SCHOLAR, '']] : []),
    ],
  )
  mkdirSync(dist + note.path, { recursive: true })
  writeFileSync(`${dist + note.path}index.html`, html)
}

writeFileSync(`${dist}robots.txt`, `User-agent: *\nAllow: /\n\nSitemap: ${SITE_URL}/sitemap.xml\n`)
const pages = [
  { path: '/', file: 'content/main.mdx' },
  ...notes.map(({ id, path }) => ({ path, file: `content/notes/${id}.mdx` })),
  ...(existsSync(`${dist}1000-years-of-ai-safety.pdf`)
    ? [{ path: '/1000-years-of-ai-safety.pdf', file: 'public/1000-years-of-ai-safety.pdf' }]
    : []),
]
writeFileSync(
  `${dist}sitemap.xml`,
  [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...pages.map(({ path, file }) => {
      const date = changed(file)
      return `  <url><loc>${escape(SITE_URL + path)}</loc>${date ? `<lastmod>${date}</lastmod>` : ''}</url>`
    }),
    '</urlset>',
    '',
  ].join('\n'),
)
console.log(
  `prerender: ${SITE_URL}, main text ${Math.round(main.length / 1024)} kB, ${notes.length} note pages, ` +
    `${pages.length} pages in sitemap.xml`,
)
