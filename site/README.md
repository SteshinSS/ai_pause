# Site

The website at [1000yearsofaisafety.com](https://1000yearsofaisafety.com). Vite, React and MDX; a static build.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # -> dist/
npm run pdf      # -> public/1000-years-of-ai-safety.pdf
```

The text is in `content/`: `main.mdx` is the main page and `notes/` holds the side notes. The JSON files there are
written by the Python scripts in the repo root.

The PDF is the version to cite: `paper/paper.mdx`, the notes' text frozen for it, printed by headless Chrome
(`scripts/pdf.mjs`, `src/paper.tsx`). `index.html`'s `citation_` tags point Google Scholar to it; keep their title,
authors and date the PDF's.
