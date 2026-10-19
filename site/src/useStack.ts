import { useCallback, useEffect, useRef, useState } from 'react'
import { hrefFor, readIds } from './url'

/** What we store in history.state: the stack this entry was opened from. */
type Entry = { from?: string[] }

export const same = (a?: string[], b?: string[]) =>
  !!a && !!b && a.length === b.length && a.every((x, i) => x === b[i])

/** Open notes, left to right. Kept in the URL, so a view can be shared and Back closes the last note. `seen`: the
 *  browser already showed the last change with a transition of its own (a swipe back), so the page shows it at once. */
export function useStack() {
  const [{ ids, seen }, setStack] = useState(() => ({ ids: readIds(), seen: false }))

  // The page shows a new stack at once and the address bar follows. Closing goes Back when that gives the same stack,
  // so history doesn't fill up with open/close pairs. Until that Back's popstate comes, the address bar and
  // history.state are still the old entry's, so the address bar follows what is asked for meanwhile only then: closing
  // fast, as with Esc held down, doesn't go Back further than it closes.
  const wanted = useRef(ids) // the stack asked for last
  const closing = useRef(false) // by closing notes
  const backs = useRef(0) // our Backs whose popstate hasn't come yet

  const follow = useCallback(() => {
    const now = readIds()
    const next = wanted.current
    if (backs.current || same(next, now)) return
    if (closing.current && same((history.state as Entry | null)?.from, next)) {
      backs.current++
      history.back()
    } else {
      // The entry left keeps its notes but not a section's #, so that Back doesn't scroll to it again (Stack.tsx).
      if (location.hash) history.replaceState(history.state, '', hrefFor(now))
      history.pushState({ from: now } satisfies Entry, '', hrefFor(next))
    }
  }, [])

  useEffect(() => {
    history.replaceState(history.state, '', hrefFor(readIds()) + location.hash) // drop unknown ids from a pasted link
    const onPop = (e: PopStateEvent) => {
      if (backs.current) {
        backs.current--
        follow()
        return
      }
      wanted.current = readIds()
      setStack({ ids: wanted.current, seen: e.hasUAVisualTransition === true })
    }
    addEventListener('popstate', onPop)
    return () => removeEventListener('popstate', onPop)
  }, [follow])

  const show = useCallback(
    (next: string[], close: boolean) => {
      if (same(next, wanted.current)) return
      wanted.current = next
      closing.current = close
      setStack({ ids: next, seen: false })
      follow()
    },
    [follow],
  )

  const navigate = useCallback((next: string[]) => show(next, false), [show])

  /** Close pane `pane` (1 = first note) and everything to its right. */
  const close = useCallback((pane: number) => show(wanted.current.slice(0, pane - 1), true), [show])

  return { ids, seen, navigate, close }
}
