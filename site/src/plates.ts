/** The drawings of src/art/ by id, the file name: `<Plate id="main" />` shows src/art/main.webp. */
const files = import.meta.glob<string>('./art/*.{svg,webp}', { eager: true, query: '?url', import: 'default' })

export const PLATES: Record<string, string> = Object.fromEntries(
  Object.entries(files).map(([path, url]) => [path.slice('./art/'.length).replace(/\.\w+$/, ''), url]),
)
