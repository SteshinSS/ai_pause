#!/usr/bin/env python3
"""The tracked results (results/, not gitignored) from the working files in data/.

    python export.py

- results/safety_papers_compute.csv: every AI-safety paper (2020-2025) except those whose experiments ran only on CPU
  cores (cpu_only in compute.csv), with its metadata, organizations, top-3-lab
  authors (labs.py), Semantic Scholar citations (citations.py) and its compute in H100e-years (estimate_compute.py
  --all --run census): with dead ends (h100e_years), the reported experiments and a stated whole-project figure; the
  authors' own figure where the blind test has one.
- results/blind_test.csv: the blind test (blind.py dataset): the authors' figure vs Sonnet's and Opus's estimates.
- results/blind_test.md: how good the estimates are (blind.py score): before calibration, the calibration, after it.
- results/blind_plot.html, blind_plot_years.html: the scatter plots (blind.py plot).
- ../site/content/blind_test.json: the scored papers of the blind test for the site's scatter plots (BlindPlot.tsx).
"""
import csv
import json
import shutil

from common import DATA, HERE, RUNS
from h100e import DEAD_ENDS

OUT = HERE / "results"
BLIND = DATA / "blind"
SITE = HERE.parent / "site" / "content" / "blind_test.json"


def papers() -> None:
    if not (DATA / "compute_estimates" / "census" / "compute.csv").exists():
        print("no census run yet (estimate_compute.py run --all --run census): safety_papers_compute.csv not written")
        return
    meta = list(csv.DictReader(open(RUNS / "census" / "safety_papers.csv")))
    est = {r["id"]: r for r in csv.DictReader(open(DATA / "compute_estimates" / "census" / "compute.csv"))}
    s2 = json.loads((DATA / "s2.json").read_text()) if (DATA / "s2.json").exists() else {}
    stated = {r["id"]: r for r in csv.DictReader(open(BLIND / "targets.csv")) if r["use"] == "1"}
    labs = {r["id"]: r for r in csv.DictReader(open(DATA / "labs.csv"))} if (DATA / "labs.csv").exists() else {}
    cols = ["id", "year", "month", "created", "topic", "topic_name", "title", "authors", "n_authors", "primary",
            "cats", "journal_ref", "doi", "url", "venue", "organizations", "lab_author", "labs", "anthropic_fellows",
            "citations", "influential_citations",
            "h100e_years", "reported_h100e_years", "project_h100e_years", "states_compute", "accelerator", "route",
            "authors_h100e_years", "authors_figure_covers"]
    with open(OUT / "safety_papers_compute.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for m in meta:
            if est.get(m["id"], {}).get("cpu_only") == "1":
                continue
            e, c, t, lab = est.get(m["id"], {}), s2.get(m["id"]) or {}, stated.get(m["id"], {}), labs.get(m["id"], {})
            w.writerow({**{k: m[k] for k in cols if k in m}, "venue": c.get("venue") or "",
                        "organizations": e.get("organizations", ""), "lab_author": lab.get("lab", ""),
                        "labs": ";".join(k for k in ("openai", "anthropic", "google") if lab.get(k) == "1"),
                        "anthropic_fellows": lab.get("anthropic_fellows", ""),
                        "citations": "" if c.get("citationCount") is None else c["citationCount"],
                        "influential_citations": "" if c.get("influentialCitationCount") is None
                        else c["influentialCitationCount"],
                        **{k: e.get(k, "") for k in ("h100e_years", "reported_h100e_years", "project_h100e_years",
                                                     "states_compute", "accelerator", "route")},
                        "authors_h100e_years": t.get("true_h100e_years", ""),
                        "authors_figure_covers": {"all": "all reported experiments", "project": "the whole project"}
                        .get(t.get("level"), t.get("covers", ""))})
    cpu = sum(e.get("cpu_only") == "1" for e in est.values())
    print(f"{len(meta) - cpu:,} papers ({cpu:,} that ran only on CPUs left out) -> {OUT / 'safety_papers_compute.csv'}"
          f" (citations as of {s2.get('_fetched', '-')})")


def blind() -> None:
    """results/blind_test.csv (dataset.csv with titles), blind_test.md (blind.py score), the plots."""
    title = {r["id"]: r["title"] for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))}
    rows = list(csv.DictReader(open(BLIND / "dataset.csv")))
    with open(OUT / "blind_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]) + ["title"])
        w.writeheader()
        w.writerows({**r, "title": title.get(r["id"], "")} for r in rows)
    shutil.copy(BLIND / "score.md", OUT / "blind_test.md")
    for name in ("plot.html", "plot_years.html"):
        shutil.copy(BLIND / name, OUT / f"blind_{name}")
    print(f"{len(rows)} blind-test papers -> {OUT / 'blind_test.csv'}, blind_test.md, blind_plot*.html")


def site() -> None:
    """The papers both models estimated blind without seeing the compute, for the site's scatter plots: the authors'
    figure and each model's estimate of the reported experiments before calibration (the model's hours, FLOP at
    40%; x DEAD_ENDS where the authors' figure is the whole project), in H100e-years."""
    title = {r["id"]: r["title"] for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))}
    arms = {"sonnet": "sonnet-blind", "opus": "opus-blind"}
    rows = [r for r in csv.DictReader(open(BLIND / "dataset.csv"))
            if all(r[f"{a}_seen"] == "no" and r[f"{a}_raw"] and float(r[f"{a}_raw"]) > 0 for a in arms.values())]
    out = {"made_by": "safety_papers/export.py from data/blind/dataset.csv", "unit": "H100e-years (AI 2027, TPP)",
           "estimates": "reported experiments, before calibration",
           "rows": [{"id": r["id"], "year": int(r["year"]), "level": r["level"], "title": title.get(r["id"], ""),
                     "covers": r["covers"] if r["level"] == "part" else "", "true": float(r["true_h100e_years"]),
                     **{m: float(r[f"{a}_raw"]) * (DEAD_ENDS if r["level"] == "project" else 1)
                        for m, a in arms.items()}} for r in rows]}
    SITE.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"{len(rows)} blind-test papers -> {SITE}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    papers()
    blind()
    site()


if __name__ == "__main__":
    main()
