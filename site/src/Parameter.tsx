import {
  Children,
  isValidElement,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type MouseEvent,
  type ReactNode,
} from 'react'

/** Clicks on these open something of their own: a symbol's popup, a link, a term's popup. */
const OWN_CLICK = '[data-sym], a, .gloss-term'

/**
 * A row of the parameter table in content/notes/parameters.mdx, `<Parameter>` in place of `<tr>`, with three cells:
 * the symbol and name, the estimate, the rationale. The rationale is folded; a click on the row or its button unfolds
 * it, except on a symbol or a link, which open their own popups. A row with an empty rationale doesn't fold. Folded,
 * the rationale is `hidden="until-found"`: the browser's find in page still reaches it and unfolds it.
 */
export function Parameter({ children }: { children: ReactNode }) {
  const [name, estimate, rationale] = Children.toArray(children)
  const text = isValidElement<{ children?: ReactNode }>(rationale) ? rationale.props.children : undefined
  const folds = Children.count(text) > 0
  const [open, setOpen] = useState(false)
  const id = useId()
  const cell = useRef<HTMLTableCellElement>(null)

  // React only knows `hidden` as true or false.
  useLayoutEffect(() => {
    const el = cell.current
    if (!el) return
    if (open) el.removeAttribute('hidden')
    else el.setAttribute('hidden', 'until-found')
    const onMatch = () => setOpen(true)
    el.addEventListener('beforematch', onMatch)
    return () => el.removeEventListener('beforematch', onMatch)
  }, [open])

  const onClick = (e: MouseEvent) => {
    const target = e.target as Element
    if (target.closest(OWN_CLICK) || cell.current?.contains(target)) return
    setOpen((o) => !o)
  }

  return (
    <tr
      className="parameter"
      data-folds={folds || undefined}
      data-open={open || undefined}
      onClick={folds ? onClick : undefined}
    >
      {name}
      {estimate}
      <td className="parameter-toggle">
        {folds && (
          <button type="button" aria-expanded={open} aria-controls={id} aria-label="Rationale">
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M4 6l4 4 4-4" />
            </svg>
          </button>
        )}
      </td>
      {folds && (
        <td ref={cell} id={id} className="parameter-rationale">
          {text}
        </td>
      )}
    </tr>
  )
}
