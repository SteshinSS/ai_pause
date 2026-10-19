// The page scripts/pdf.mjs opens in headless Chrome and prints to public/1000-years-of-ai-safety.pdf: paper/paper.mdx
// with the site's components, on a white page at the PDF's text width (6.5 in, 624 px), under a title page made from
// its frontmatter. Popups become plain text, footnotes notes at the end, citations a list of the works cited, and links
// to notes that the PDF holds go to their place in it. Not part of the site: `vite build` doesn't build it.
import '@fontsource-variable/newsreader/opsz.css'
import '@fontsource-variable/newsreader/opsz-italic.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import 'katex/dist/katex.min.css'
import type { MDXContent } from 'mdx/types'
import { Children, isValidElement, type ComponentProps, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import table from '../content/table.json'
import { AuditorCurve } from './AuditorCurve'
import { MARKS, MILESTONES } from './content'
import { Flow } from './Flow'
import { fmt, people } from './format'
import { HiringCurve } from './HiringCurve'
import { HiringPlot } from './HiringPlot'
import { PaperCountPlot } from './PaperCountPlot'
import { Grid } from './ResultTable'
import { Range, Value } from './Symbols'
import { Tornado } from './Tornado'
import './styles.css'
import './paper.css'

type Author = { name: string; url?: string; affiliation?: string }
type Front = { title: string; authors: Author[]; date: string; abstract: string }
const [paper] = Object.values(
  import.meta.glob<{ default: MDXContent; frontmatter: Front }>('../paper/paper.mdx', { eager: true }),
)
const front = paper.frontmatter

const SITE = 'https://1000yearsofaisafety.com'
/** Notes whose text the PDF holds, with the address of the heading it starts at: their links stay in the PDF. */
const IN_PDF: Record<string, string> = {
  model: 'ai-pause-model',
  'model-explanations': 'ai-pause-model',
  sensitivity: 'sensitivity-analysis',
  parameters: 'model-parameters',
}

type Model = ComponentProps<typeof Grid>['model']

/** A link: to a note's page on the site, or to its place in the PDF when the PDF holds it (resolve(), below). */
export function Link({ href = '', children }: ComponentProps<'a'>) {
  if (!href.startsWith('note:')) return <a href={href}>{children}</a>
  const [id, section] = href.slice(5).split('#')
  return (
    <a href={`${SITE}/n/${id}/${section ? `#${section}` : ''}`} data-note={id} data-to={section || undefined}>
      {children}
    </a>
  )
}

/** A heading with its address as its id, for the links within the PDF; numbered: true gives it data-num. */
const headings = Object.fromEntries(
  [1, 2, 3, 4, 5, 6].map((level) => {
    const H = `h${level}` as 'h1'
    return [`h${level}`, ({ 'data-section': section, ...rest }: ComponentProps<'h1'> & { 'data-section'?: string }) => (
      <H id={section} data-section={section} {...rest} />
    )]
  }),
)

/** A footnote: its text stays hidden here, and resolve() numbers it and copies it to the notes at the end. */
export function Footnote({ children }: { children: ReactNode }) {
  return (
    <span className="pdf-fn">
      <span className="pdf-fn-text">{children}</span>
    </span>
  )
}

/** A citation: its label, with the full reference hidden for the list of works cited. */
export function Cite({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="pdf-cite">
      {label}
      <span className="pdf-ref">{children}</span>
    </span>
  )
}

/** The works cited, filled in by resolve(); citations.ts fills its children with all of references.bib. */
export function References() {
  return (
    <section className="pdf-references">
      <h1>References</h1>
      <div className="pdf-reflist" />
    </section>
  )
}

export function Notes() {
  return (
    <section className="pdf-notes">
      <h1>Notes</h1>
      <ol className="pdf-notelist" />
    </section>
  )
}

export function Appendix({ children }: { children: ReactNode }) {
  return <section className="pdf-appendix">{children}</section>
}

/** A parameter's row of the parameters page with its rationale unfolded below it. */
export function Parameter({ children }: { children: ReactNode }) {
  const [name, estimate, rationale] = Children.toArray(children)
  const text = isValidElement<{ children?: ReactNode }>(rationale) ? rationale.props.children : undefined
  return (
    <>
      <tr className="pdf-param">
        {name}
        {estimate}
      </tr>
      {Children.count(text) > 0 && (
        <tr className="pdf-param-why">
          <td colSpan={2}>{text}</td>
        </tr>
      )}
    </>
  )
}

export function Unused({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="unused">
      <div className="unused-label">{label}</div>
      {children}
    </div>
  )
}

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const monthYear = (month: string) => {
  const [y, m] = month.split('-').map(Number)
  return `${MONTHS[m - 1]} ${y}`
}

/**
 * A result table, ResultTable.tsx's grid, in both states of the site's switch: the labs' own staff and, for the trust
 * models that hire, unlimited hiring with the people it hires. `menu`: the site's menu picks the trust model shown;
 * here it is the main one. A † marks the cells content/marks/ explains, under the table.
 */
export function PaperTable({ model = table.main as Model, toggle = false, children }: { model?: Model; menu?: boolean; toggle?: boolean; children?: ReactNode }) {
  const hires = toggle && model in table.rows[0].hired
  const unlimited = model as keyof (typeof table.rows)[number]['hired']
  const marks = MARKS.filter((m) => m.model === model)
  return (
    <figure className="result-table pdf-table">
      {hires && <div className="result-table-title">Own staff</div>}
      <Grid model={model} values={(row) => row.cells[model]} format={fmt} marked />
      {hires && (
        <>
          <div className="result-table-title">Unlimited hiring</div>
          <Grid model={model} values={(row) => row.unlimited[unlimited]} format={fmt} />
          <div className="result-table-title">People to hire</div>
          <Grid model={model} values={(row) => row.hired[unlimited]} format={people} />
          <div className="result-table-note">On top of the labs' own staff: the most they need at any time in the pause.</div>
        </>
      )}
      {marks.map((m) => (
        <div className="result-table-note pdf-mark" key={m.id}>
          † <m.Content components={{ a: Link, p: ({ children }) => <>{children} </> }} />
        </div>
      ))}
      {children && <figcaption>{children}</figcaption>}
    </figure>
  )
}

/** The rows of the result tables: each capability level, its date in the model and its popup on the site. */
export function CapabilityLevels() {
  return (
    <figure className="result-table pdf-levels">
      <table>
        <thead>
          <tr>
            <th scope="col">AI capability at the start</th>
            <th scope="col">Date</th>
            <th scope="col">Definition</th>
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row) => {
            const popup = MILESTONES[row.name.toLowerCase()]
            return (
              <tr key={row.name}>
                <th scope="row">{popup.title}</th>
                <td>{monthYear(row.month)}</td>
                <td>
                  <popup.Content components={{ a: Link, p: ({ children }) => <>{children}</> }} />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <figcaption>The rows of the result tables: the AI capability when the pause starts, and its date in our model.</figcaption>
    </figure>
  )
}

const none = () => null
const components = {
  ...headings,
  a: Link,
  AifmOnly: ({ children }: { children: ReactNode }) => <Unused label="AIFM only">{children}</Unused>,
  Appendix,
  AuditorCurve,
  Backdrop: none,
  CapabilityLevels,
  Cite,
  Derivation: none,
  Doodle: none,
  Flow,
  Footnote,
  HiringCurve,
  HiringPlot,
  Mark: none,
  Milestone: ({ children }: { children: string }) => children,
  Night: ({ children }: { children: ReactNode }) => children,
  NotInModel: ({ children }: { children: ReactNode }) => <Unused label="not in the model">{children}</Unused>,
  Notes,
  PaperCountPlot,
  Parameter,
  PilePlot: none,
  Plate: none,
  Range,
  References,
  ResultTable: PaperTable,
  Term: ({ term }: { term: string }) => term,
  Tornado,
  Value,
}

export function Paper() {
  const Content = paper.default
  return (
    <>
      <header className="pdf-title">
        {/* Not a heading: the PDF's bookmarks are the sections', and Chrome's would lose the space at its line break. */}
        <div className="pdf-title-text">{front.title}</div>
        <div className="pdf-authors">
          {front.authors.map((a) => (
            <div className="pdf-author" key={a.name}>
              <div className="pdf-author-name">{a.url ? <a href={a.url}>{a.name}</a> : a.name}</div>
              {a.affiliation && <div className="pdf-author-affiliation">{a.affiliation}</div>}
            </div>
          ))}
        </div>
        <div className="pdf-date">{front.date}</div>
        <section className="pdf-abstract">
          <div className="pdf-abstract-label">Abstract</div>
          <p>{front.abstract}</p>
        </section>
        <p className="pdf-site">
          Interactive version: <a href={`${SITE}/`}>{SITE.replace('https://', '')}</a>
        </p>
      </header>
      <article className="pane-body pdf-body">
        <Content components={components} />
      </article>
    </>
  )
}

/** Numbers the footnotes and copies them to the notes, lists the works cited, and points links into the PDF. */
function resolve() {
  const list = document.querySelector('.pdf-notelist')!
  document.querySelectorAll<HTMLElement>('.pdf-fn').forEach((fn, i) => {
    const n = i + 1
    const ref = document.createElement('sup')
    ref.className = 'pdf-fn-ref'
    ref.innerHTML = `<a href="#note-${n}" id="ref-${n}">${n}</a>`
    fn.prepend(ref)
    const item = document.createElement('li')
    item.id = `note-${n}`
    item.append(...[...fn.querySelector('.pdf-fn-text')!.childNodes].map((node) => node.cloneNode(true)))
    list.append(item)
  })

  const refs = new Map<string, Element>()
  document.querySelectorAll('.pdf-ref').forEach((ref) => {
    const text = ref.textContent!.trim()
    if (!refs.has(text)) refs.set(text, ref)
  })
  const works = document.querySelector('.pdf-reflist')!
  for (const [, ref] of [...refs].sort(([a], [b]) => a.localeCompare(b))) {
    works.append(...[...ref.childNodes].map((node) => node.cloneNode(true)))
  }

  document.querySelectorAll<HTMLAnchorElement>('a[data-note]').forEach((a) => {
    const start = IN_PDF[a.dataset.note!]
    if (!start) return
    const to = a.dataset.to
    a.href = `#${to && document.getElementById(to) ? to : start}`
  })
}

const frames = () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))
const FACES = [
  '15px "Newsreader Variable"',
  'italic 15px "Newsreader Variable"',
  '600 15px "Newsreader Variable"',
  '400 15px "IBM Plex Mono"',
  '500 15px "IBM Plex Mono"',
  ...['AMS', 'Caligraphic', 'Main', 'Math', 'Size1', 'Size2', 'Size3', 'Size4'].map((f) => `15px KaTeX_${f}`),
  'bold 15px KaTeX_Main',
  'italic 15px KaTeX_Main',
  'italic 15px KaTeX_Math',
]

declare global {
  interface Window {
    paper?: { ready: Promise<void> }
  }
}

// The fonts first, so plots draw their labels in them.
await Promise.all(FACES.map((face) => document.fonts.load(face).catch(() => [])))
document.title = front.title
createRoot(document.getElementById('root')!).render(<Paper />)
window.paper = {
  ready: (async () => {
    await frames()
    await document.fonts.ready
    await new Promise((done) => setTimeout(done, 300)) // plots measure their width, diagrams draw their arrows
    await frames()
    resolve()
    await frames()
  })(),
}
