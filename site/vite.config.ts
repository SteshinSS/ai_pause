import mdx from '@mdx-js/rollup'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import rehypeKatex from 'rehype-katex'
import remarkFrontmatter from 'remark-frontmatter'
import remarkMath from 'remark-math'
import remarkMdxFrontmatter from 'remark-mdx-frontmatter'
import type { VFile } from 'vfile'
import { defineConfig, type Plugin } from 'vite'
import { remarkCitations } from './citations.ts'
import { remarkFootnotes } from './footnotes.ts'
import { remarkHeadings } from './headings.ts'
import { rehypeMathPunctuation } from './math.ts'
import { remarkMilestones } from './milestones.ts'
import { remarkNumbering } from './numbering.ts'
import { KATEX_OPTIONS, rehypeSymbolSources, remarkSymbols } from './symbols.ts'

// rehype-katex only warns about bad math and shows its source in red; fail the build (and the dev overlay) instead.
const failOnMathErrors = () => (_tree: unknown, file: VFile) => {
  const bad = file.messages.find((m) => m.source === 'rehype-katex')
  if (bad) file.fail(bad.cause instanceof Error ? bad.cause.message : bad.reason, { place: bad.place, source: bad.source })
}

const BIBLIOGRAPHY = fileURLToPath(new URL('content/references.bib', import.meta.url))
const SYMBOLS = fileURLToPath(new URL('content/symbols.mdx', import.meta.url))
const TABLE = fileURLToPath(new URL('content/table.json', import.meta.url))

// The .mdx files read references.bib, the symbol entries and table.json (which symbols are our inputs) themselves, so
// Vite doesn't know they depend on them: rebuild them all when any of these changes.
const reloadOnBibliography = (): Plugin => ({
  name: 'reload-on-bibliography',
  configureServer(server) {
    server.watcher.add(BIBLIOGRAPHY)
    server.watcher.on('change', (path) => {
      if (path !== BIBLIOGRAPHY && path !== SYMBOLS && path !== TABLE) return
      for (const mod of server.moduleGraph.idToModuleMap.values()) {
        if (mod.id?.endsWith('.mdx')) server.moduleGraph.invalidateModule(mod)
      }
      server.ws.send({ type: 'full-reload' })
    })
  },
})

// The site's address, for link previews, canonical links and the sitemap (src/url.ts, scripts/prerender.mjs): the
// environment variable SITE_URL (set it in Render once the site has its own domain), else the onrender.com address
// Render builds it for.
const SITE_URL = (process.env.SITE_URL || process.env.RENDER_EXTERNAL_URL || 'http://localhost:4173').replace(/\/+$/, '')

export default defineConfig({
  define: { 'import.meta.env.SITE_URL': JSON.stringify(SITE_URL) },
  plugins: [
    // `frontmatter` export from each .mdx file carries its title. $...$ and $$...$$ are rendered by KaTeX at build time,
    // and inline maths keeps the punctuation next to it on its line (math.ts). Citations [@key] and footnotes [^id]
    // become popups (citations.ts, footnotes.ts), and so do the symbols of notes with `symbols: true` (symbols.ts).
    // A milestone's name gets its popup once every three sections (milestones.ts). Notes with `numbered: true` number
    // their headings (numbering.ts). Each heading gets its address for links to its section, from its text as written,
    // before the plugins after it change that (headings.ts).
    {
      enforce: 'pre',
      ...mdx({
        remarkPlugins: [
          remarkFrontmatter,
          remarkMdxFrontmatter,
          remarkMath,
          remarkHeadings,
          [remarkCitations, { bibliography: BIBLIOGRAPHY }],
          remarkFootnotes,
          remarkMilestones,
          remarkNumbering,
          [remarkSymbols, { symbols: SYMBOLS, values: TABLE }],
        ],
        rehypePlugins: [[rehypeKatex, KATEX_OPTIONS], failOnMathErrors, rehypeMathPunctuation, rehypeSymbolSources],
      }),
    },
    react({ include: /\.(mdx|js|jsx|ts|tsx)$/ }),
    reloadOnBibliography(),
  ],
})
