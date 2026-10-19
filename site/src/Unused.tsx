import type { ReactNode } from 'react'
import { Gloss } from './Gloss'

/**
 * `<AifmOnly>` in an .mdx file around a formula of AIFM that our model doesn't use (model.mdx): the formula in grey,
 * like a side remark, under a small AIFM label whose popup says so, since the grey alone is easy to miss.
 */
export function AifmOnly({ children }: { children: ReactNode }) {
  return (
    <Unused label="AIFM" note="The gray formula below is true only in AIFM, not in our model.">
      {children}
    </Unused>
  )
}

/** `<NotInModel>`: a formula that isn't in our model but shows the logic behind one that is, set like AifmOnly. */
export function NotInModel({ children }: { children: ReactNode }) {
  return (
    <Unused
      label="not in the model"
      note="The gray formula below isn't in our model: it shows the logic behind one that is."
    >
      {children}
    </Unused>
  )
}

function Unused({ label, note, children }: { label: string; note: string; children: ReactNode }) {
  return (
    <div className="unused">
      <div className="unused-label">
        <Gloss term={label}>
          <p>{note}</p>
        </Gloss>
      </div>
      {children}
    </div>
  )
}
