// What scripts/prerender.mjs writes into dist/ at build time, rendered by React in Node (not part of the page).

import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { MAIN, NOTES, type Doc } from './content'
import { StackContext } from './context'
import { Body } from './Pane'
import { notePath } from './url'

/** The site's address (vite.config.ts). */
export const SITE_URL: string = import.meta.env.SITE_URL

/** The main text's title, the site's name. */
export const SITE_NAME = MAIN.title

/** Doc `doc` as the page shows it in pane `index` when it is the only one open, laid out for a phone: a link opens just
 *  its note, not beside its companion. A link that would open more than one note (`/?n=a,b`) points to the page of the
 *  last, which it opens. Without popups (footnotes, citations, a row's milestone): they render only in a browser. */
function render(doc: Doc, index: number) {
  const stack = { ids: index ? [doc.id] : [], mobile: true, open: () => {} }
  const html = renderToString(createElement(StackContext.Provider, { value: stack }, createElement(Body, { doc, index })))
  return html.replace(/href="\/\?n=(?:[^",#]+,)*([^",#]+)(#[^"]*)?"/g, (_, id, hash = '') => `href="${notePath(id)}${hash}"`)
}

/** The main text as the page shows it. */
export const mainHtml = () => render(MAIN, 0)

/** Each note's own page (url.ts): its address, its title and the page's, its description from its frontmatter if it
 *  has one, and its text as the page shows it there. */
export const notePages = () =>
  Object.values(NOTES).map((doc) => ({
    id: doc.id,
    path: notePath(doc.id),
    name: doc.title,
    title: `${doc.title} · ${MAIN.title}`, // as Stack.tsx's document.title
    description: doc.description,
    html: render(doc, 1),
  }))
