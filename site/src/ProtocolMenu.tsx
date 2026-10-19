import { use, useEffect, useId, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react'
import { PROTOCOLS } from './content'
import { Inside } from './context'
import { Gloss } from './Gloss'
import { MdxLink } from './NoteLink'
import { place, portal } from './popup'

const components = { a: MdxLink }

/** A trust model's popup, content/protocols/<model with a dash for each space>.mdx: its title is the model's name on
 *  the site (scripts/check-links.mjs fails the build on a model without one). */
const protocol = (model: string) => PROTOCOLS[model.replaceAll(' ', '-')]

/**
 * The menu of trust models of a result table (ResultTable.tsx's `menu`), labelled "How much do you trust AI": the name of
 * the model shown, on a rule, that opens the list of models, each after a "?" whose popup (Gloss.tsx) explains it and links to its
 * section of the model explanations. The list is a popover under the name, like a popup: a click elsewhere or Esc
 * closes it, and the popups of its "?"s go inside it, so that opening one leaves it open. They open left of the list,
 * so none covers another "?" and the mouse can run down them. The arrow keys move between the models.
 * `className` is the class of the label and button's box.
 */
export function ProtocolMenu<M extends string>({
  models,
  value,
  onChange,
  className,
}: {
  models: M[]
  value: M
  onChange: (model: M) => void
  className?: string
}) {
  const id = useId()
  const current = useRef<HTMLButtonElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const parent = use(Inside)
  const [inside, setInside] = useState<HTMLElement | null>(null)
  const [open, setOpen] = useState(false)

  // Placed under the name (or over it), as a popup is, and kept there as columns scroll; the model shown has the focus.
  useEffect(() => {
    const button = current.current!
    const pop = list.current!
    const onMove = () => place(button, pop)
    const follow = (on: boolean) => {
      const method = on ? addEventListener : removeEventListener
      method('scroll', onMove, { capture: true, passive: true })
      method('resize', onMove)
    }
    const onBeforeToggle = (e: Event) => {
      if ((e as ToggleEvent).newState === 'open') delete pop.dataset.placed
    }
    const onToggle = (e: Event) => {
      const shown = (e as ToggleEvent).newState === 'open'
      if (shown) {
        pop.style.minWidth = `${button.offsetWidth}px`
        place(button, pop)
        pop.querySelector<HTMLElement>('[aria-checked="true"]')?.focus({ preventScroll: true })
      }
      setOpen(shown)
      follow(shown)
    }
    pop.addEventListener('beforetoggle', onBeforeToggle)
    pop.addEventListener('toggle', onToggle)
    return () => {
      pop.removeEventListener('beforetoggle', onBeforeToggle)
      pop.removeEventListener('toggle', onToggle)
      follow(false)
    }
  }, [])

  const choose = (model: M) => {
    onChange(model)
    list.current?.hidePopover()
    current.current?.focus({ preventScroll: true })
  }

  // Up and down move between the models, from the name too.
  const step = (e: KeyboardEvent, by: number) => {
    const rows = [...list.current!.querySelectorAll<HTMLElement>('.protocol-option')]
    const at = rows.findIndex((row) => row.contains(document.activeElement))
    rows[(at + by + rows.length) % rows.length].querySelector<HTMLElement>('[role="radio"]')!.focus()
    e.preventDefault()
  }
  const onListKey = (e: KeyboardEvent) => {
    if (!(e.target as Element).closest('.protocol-option')) return // in a "?" popup
    if (e.key === 'ArrowDown') step(e, 1)
    if (e.key === 'ArrowUp') step(e, -1)
  }
  const onPillKey = (e: KeyboardEvent) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      list.current!.showPopover()
    }
  }

  // A link in a "?" popup opens a note: the list closes too.
  const onListClick = (e: MouseEvent) => {
    if ((e.target as Element).closest('a')) list.current?.hidePopover()
  }

  const name = (model: M) => protocol(model)?.title ?? model
  return (
    <div className={className}>
      <span id={`${id}label`}>How much do you trust AI</span>
      <button
        ref={current}
        id={`${id}current`}
        type="button"
        className="protocol-current"
        popoverTarget={`${id}list`}
        aria-expanded={open}
        aria-labelledby={`${id}label ${id}current`}
        onKeyDown={onPillKey}
      >
        {/* Every name in one cell, so the button is as wide as the longest, whichever shows. */}
        <span className="protocol-names">
          {models.map((model) => (
            <span key={model} data-shown={model === value || undefined}>
              {name(model)}
            </span>
          ))}
        </span>
      </button>
      {portal(
        <div
          ref={list}
          id={`${id}list`}
          popover="auto"
          className="protocol-list"
          role="radiogroup"
          aria-label="How much do you trust AI"
          onKeyDown={onListKey}
          onClick={onListClick}
        >
          <div ref={setInside} className="gloss-inside" />
          {inside && (
            <Inside value={inside}>
              {models.map((model) => {
                const popup = protocol(model)
                return (
                  <div key={model} className="protocol-option">
                    {popup && (
                      <Gloss
                        term="?"
                        title={popup.title}
                        label={`About ${popup.title}`}
                        className="protocol-help"
                        side="left"
                      >
                        <popup.Content components={components} />
                      </Gloss>
                    )}
                    <button type="button" role="radio" aria-checked={model === value} onClick={() => choose(model)}>
                      {name(model)}
                    </button>
                  </div>
                )
              })}
            </Inside>
          )}
        </div>,
        parent,
      )}
    </div>
  )
}
