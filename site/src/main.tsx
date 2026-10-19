import '@fontsource-variable/newsreader/opsz.css'
import '@fontsource-variable/newsreader/opsz-italic.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import 'katex/dist/katex.min.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Stack } from './Stack'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Stack />
  </StrictMode>,
)

// The KaTeX faces the notes' formulas use, fetched once the page is idle: fetched when a note first shows them, they
// would arrive as it opens and lay it out again. A face not listed here still loads when first shown.
const KATEX_FACES = [
  '16px KaTeX_AMS',
  '16px KaTeX_Caligraphic',
  '16px KaTeX_Main',
  'bold 16px KaTeX_Main',
  'italic 16px KaTeX_Math',
  '16px KaTeX_Size1',
  '16px KaTeX_Size2',
  '16px KaTeX_Size3',
  '16px KaTeX_Size4',
]
const idle = window.requestIdleCallback ?? ((then: () => void) => setTimeout(then, 1000))
idle(() => KATEX_FACES.forEach((face) => document.fonts.load(face).catch(() => {})))
