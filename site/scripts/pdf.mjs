// npm run pdf: paper/paper.mdx as public/1000-years-of-ai-safety.pdf, the version of the work to cite, which Google
// Scholar indexes (index.html's citation_ tags point to it).
//
// It serves src/paper.tsx from Vite's dev server, opens it in headless Chrome over the DevTools protocol, prints it on
// Letter pages with 1 in margins and page numbers, and gives the file its title, authors, abstract and keywords
// (pypdf). Needs Google Chrome (or CHROME=<its binary>) and python3 with pypdf. Google Scholar takes PDFs up to 5 MB.
import { execFileSync, spawn } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const root = fileURLToPath(new URL('..', import.meta.url))
const OUT = `${root}public/1000-years-of-ai-safety.pdf`
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const MAX_BYTES = 5 * 1024 * 1024

const sleep = (ms) => new Promise((done) => setTimeout(done, ms))

/** A small DevTools protocol client over Node's WebSocket. */
function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url)
    const pending = new Map()
    const listeners = new Map()
    let id = 0
    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data)
      if (msg.id === undefined) return listeners.get(msg.method)?.forEach((listener) => listener(msg.params))
      const call = pending.get(msg.id)
      pending.delete(msg.id)
      if (msg.error) call.reject(new Error(`${call.method}: ${msg.error.message}`))
      else call.resolve(msg.result)
    }
    ws.onerror = () => reject(new Error('pdf: no connection to Chrome'))
    ws.onopen = () =>
      resolve({
        send: (method, params = {}) =>
          new Promise((resolve, reject) => {
            pending.set(++id, { resolve, reject, method })
            ws.send(JSON.stringify({ id, method, params }))
          }),
        on: (method, listener) => listeners.set(method, [...(listeners.get(method) ?? []), listener]),
        close: () => ws.close(),
      })
  })
}

/** Headless Chrome with a fresh profile, and a page in it. */
async function launchChrome() {
  if (!existsSync(CHROME)) throw new Error(`pdf: no Chrome at ${CHROME}; set CHROME to its binary`)
  const profile = mkdtempSync(join(tmpdir(), 'pdf-chrome-'))
  const args = ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run']
  const chrome = spawn(CHROME, [...args, '--no-default-browser-check', '--hide-scrollbars', '--mute-audio'], {
    stdio: 'ignore',
  })
  const portFile = join(profile, 'DevToolsActivePort')
  let port = ''
  for (const deadline = Date.now() + 30_000; !port; await sleep(100)) {
    if (Date.now() > deadline) throw new Error('pdf: Chrome did not start')
    port = existsSync(portFile) ? readFileSync(portFile, 'utf8').split('\n')[0] : ''
  }
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json()
  const page = await connect(target.webSocketDebuggerUrl)
  return {
    page,
    async close() {
      page.close()
      const exited = new Promise((done) => chrome.once('exit', done))
      chrome.kill()
      await exited // else Chrome still writes to its profile as it goes
      rmSync(profile, { recursive: true, force: true, maxRetries: 5 })
    },
  }
}

async function evaluate(page, expression) {
  const { result, exceptionDetails } = await page.send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  })
  if (exceptionDetails) {
    throw new Error(`pdf: in the page: ${exceptionDetails.exception?.description ?? exceptionDetails.text}`)
  }
  return result.value
}

// The footer of every page: the citation, as Google Scholar likes on the first page, and the page number. Chrome
// draws it apart from the page, so in a font every Mac has.
const FOOTER = `
  <div style="width: 100%; margin: 0 1in; display: flex; justify-content: space-between;
              font: 8.5px Georgia, serif; color: #6b6b68;">
    <span>Steshin and Reznikov (2026). 1000 Years of AI Safety: A Model of AI Safety Research during an AI Pause.</span>
    <span class="pageNumber"></span>
  </div>`

const PAGE =
  '<!doctype html><html lang="en"><head><meta charset="UTF-8" /><title>paper</title></head>' +
  '<body><div id="root"></div><script type="module" src="/src/paper.tsx"></script></body></html>'
const server = await createServer({
  root,
  logLevel: 'error',
  server: { host: '127.0.0.1', port: 5198, hmr: false, forwardConsole: false },
  plugins: [
    {
      name: 'paper-page',
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          if (!req.url?.startsWith('/paper.html')) return next()
          res.setHeader('Content-Type', 'text/html')
          res.end(await server.transformIndexHtml(req.url, PAGE))
        })
      },
    },
  ],
})
await server.listen()
const chrome = await launchChrome()
try {
  const { page } = chrome
  page.on('Runtime.exceptionThrown', ({ exceptionDetails: e }) =>
    console.error(`pdf: page error: ${e.exception?.description ?? e.text}`),
  )
  await page.send('Runtime.enable')
  // Letter's width at 96 px per inch: the 6.5 in text in the middle, as in print.
  await page.send('Emulation.setDeviceMetricsOverride', { width: 816, height: 1056, deviceScaleFactor: 2, mobile: false })
  await page.send('Emulation.setEmulatedMedia', {
    features: [
      { name: 'prefers-color-scheme', value: 'light' },
      { name: 'prefers-reduced-motion', value: 'reduce' },
    ],
  })
  await page.send('Page.navigate', { url: `${server.resolvedUrls.local[0]}paper.html` })
  for (const deadline = Date.now() + 300_000; !(await evaluate(page, 'window.paper !== undefined').catch(() => false)); ) {
    if (Date.now() > deadline) throw new Error('pdf: the page never loaded')
    await sleep(250)
  }
  await evaluate(page, 'window.paper.ready')
  const title = await evaluate(page, 'document.title')
  const { data } = await page.send('Page.printToPDF', {
    paperWidth: 8.5,
    paperHeight: 11,
    marginTop: 1,
    marginBottom: 1,
    marginLeft: 1,
    marginRight: 1,
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: FOOTER,
    generateTaggedPDF: true,
    generateDocumentOutline: true,
  })
  writeFileSync(OUT, Buffer.from(data, 'base64'))

  // Title, authors, abstract and keywords in the file's own metadata, for search engines and readers.
  const meta = {
    '/Title': title,
    '/Author': 'Simon Steshin; Grigory Reznikov',
    '/Subject': JSON.parse(await evaluate(page, 'JSON.stringify(document.querySelector(".pdf-abstract p").textContent)')),
    '/Keywords': 'AI pause; AI safety; AI safety research; automated AI research; AI Futures Model; verification',
  }
  execFileSync('python3', [
    '-I',
    '-c',
    `import json, sys
from pypdf import PdfWriter
w = PdfWriter(clone_from=sys.argv[1])
w.add_metadata(json.loads(sys.argv[2]))
w.write(sys.argv[1])`,
    OUT,
    JSON.stringify(meta),
  ])
} finally {
  await chrome.close()
  await server.close()
}

const bytes = statSync(OUT).size
console.log(`pdf: ${OUT.slice(root.length)}, ${(bytes / 1024 / 1024).toFixed(2)} MB`)
if (bytes > MAX_BYTES) throw new Error('pdf: over Google Scholar\'s 5 MB')
