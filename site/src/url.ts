import { isNote } from './content'

const KEY = 'n'

/** A note's own page, which scripts/prerender.mjs writes for each note: the address of the note open alone. */
export const notePath = (id: string) => `/n/${id}/`
const NOTE_PATH = /^\/n\/([^/]+)\/?$/

/** Open notes from an address: `/n/a/` -> ['a'], `?n=a,b` -> ['a', 'b']. Unknown and repeated ids are dropped. */
export function idsIn({ pathname, search }: { pathname: string; search: string }): string[] {
  const raw = new URLSearchParams(search).get(KEY) ?? NOTE_PATH.exec(pathname)?.[1] ?? ''
  return [...new Set(raw.split(','))].filter(isNote)
}

/** Open notes from the address bar. */
export const readIds = () => idsIn(location)

/** Address of a stack of notes: `/`, a note's own page for one, `/?n=a,b` for more. Other query params are kept;
 *  commas stay unescaped. */
export function hrefFor(ids: string[]): string {
  // At build time (src/prerender.ts) there is no address bar.
  const params = new URLSearchParams(typeof location === 'undefined' ? '' : location.search)
  params.delete(KEY)
  const query = [params.toString(), ids.length > 1 ? `${KEY}=${ids.join(',')}` : ''].filter(Boolean).join('&')
  return pathOf(ids) + (query ? `?${query}` : '')
}

const pathOf = (ids: string[]) => (ids.length === 1 ? notePath(ids[0]) : '/')

/** Tells search engines the address of what is open, on the site's own address (SITE_URL, vite.config.ts) and without
 *  query params, so the page under another address (onrender.com, a link with ?utm=) counts as this one: a note's own
 *  page for one note, else the main text's. The same as in the page's HTML (scripts/prerender.mjs), except on an old
 *  link `?n=a`. */
export function setCanonical(ids: string[]) {
  let link = document.querySelector<HTMLLinkElement>('link[rel="canonical"]')
  if (!link) {
    link = document.createElement('link')
    link.rel = 'canonical'
    document.head.append(link)
  }
  const href = import.meta.env.SITE_URL + pathOf(ids)
  if (link.href !== href) link.href = href
}

declare global {
  interface Window {
    /** GoatCounter's script (index.html), once it has loaded. */
    goatcounter?: { count?: (vars: { path: string; title: string }) => void }
  }
}

/** Counts a visit to a note's page, as opening a note changes the address without loading a page. The script skips
 *  localhost, and a reader whose ad-blocker stops it isn't counted. */
export function countVisit(id: string, title: string) {
  window.goatcounter?.count?.({ path: notePath(id), title })
}
