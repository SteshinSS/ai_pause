#!/usr/bin/env python3
"""The list of in-scope papers and counts per year. In scope: topic IN, SD, AV, JB, UL, AL or EV with kind E.

    python report.py [--run census]   # -> data/runs/<run>/safety_papers.csv, labels.csv, report.md

safety_papers.csv: one row per in-scope paper with its label and arXiv metadata (authors, categories, v1 date,
comments, journal ref, DOI, abstract). labels.csv: every labelled paper, label and title only.
"""
import argparse
import csv
from collections import Counter

from classify import dollars
from common import GAVE_UP, IN_TOPICS, OUT_TOPICS, RUNS, TOPIC_NAMES, in_scope, load_labels, load_papers, read_json


def table(header: list, rows: list) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
                     + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="census")
    args = ap.parse_args()
    rd = RUNS / args.run
    cfg = read_json(rd / "run.json", None)
    ids = (rd / "ids.txt").read_text().split()
    every = load_papers(cfg["papers"], fields=["id", "year", "month", "primary", "title"])
    papers = {i: every[i] for i in ids}
    labels = {i: lab for i, lab in load_labels(rd).items() if i in papers}
    listed = sorted(i for i, lab in labels.items() if in_scope(lab))
    years = sorted({p["year"] for p in papers.values()})

    rows, rest = [], []
    for y in years:
        n = sum(p["year"] == y for p in papers.values())
        mine = [i for i in labels if papers[i]["year"] == y]
        t = Counter(labels[i]["topic"] for i in listed if papers[i]["year"] == y)
        rows.append([y, f"{n:,}", f"{len(mine):,}", f"{sum(t.values()):,}", f"{100 * sum(t.values()) / n:.2f}%"]
                    + [f"{t[x]:,}" for x in IN_TOPICS])
        e = Counter(labels[i]["topic"] for i in mine if labels[i]["kind"] == "E")
        k = Counter(labels[i]["kind"] for i in mine if labels[i]["topic"] in IN_TOPICS)
        rest.append([y] + [f"{e[x]:,}" for x in OUT_TOPICS] + [f"{k['T']:,}", f"{k['S']:,}"])
    md = [f"# Safety papers, run `{args.run}`", "", f"{len(papers):,} papers from `{cfg['papers']}`, "
          f"labelled by {cfg['model']}.", "",
          table(["year", "papers", "labelled", "in scope", "share"] + IN_TOPICS, rows), "",
          "Out of scope: neighbours of safety (kind E), and in-scope topics that are theory (T) or surveys and "
          "position papers (S).", "", table(["year"] + OUT_TOPICS + ["T", "S"], rest), ""]
    gave = [i for i in sorted(labels) if labels[i]["topic"] in GAVE_UP]
    if gave:
        md += ["Not labelled (read by hand):", ""] + [f"- {i} {GAVE_UP[labels[i]['topic']]}: {papers[i]['title']}"
                                                      for i in gave] + [""]
    md.append(f"Cost: ${dollars(read_json(rd / 'usage.json', {})):,.2f}")

    with open(rd / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "month", "primary", "topic", "kind", "conf", "in_scope", "why", "title"])
        for i in sorted(labels):
            lab, p = labels[i], papers[i]
            w.writerow([i, p["year"], p["month"], p["primary"], lab["topic"], lab["kind"], lab["conf"],
                        int(in_scope(lab)), lab["why"], p["title"]])
    # the list itself, with the paper's metadata (arXiv gives no affiliations: they are empty for every paper)
    full = load_papers(cfg["papers"], ids=set(listed))
    with open(rd / "safety_papers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "month", "created", "topic", "topic_name", "conf", "why", "title", "n_authors",
                    "authors", "primary", "cats", "comments", "journal_ref", "doi", "url", "abstract"])
        for i in listed:
            lab, p = labels[i], full[i]
            w.writerow([i, p["year"], p["month"], p.get("created", ""), lab["topic"], TOPIC_NAMES[lab["topic"]],
                        lab["conf"], lab["why"], p["title"], len(p.get("authors", [])),
                        "; ".join(" ".join(x for x in a[1::-1] if x) for a in p.get("authors", [])), p["primary"],
                        p["cats"], p.get("comments", ""), p.get("journal_ref", ""), p.get("doi", ""),
                        f"https://arxiv.org/abs/{i}", p["abstract"]])
    (rd / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> {rd / 'safety_papers.csv'} ({len(listed):,} papers), {rd / 'labels.csv'}, {rd / 'report.md'}")


if __name__ == "__main__":
    main()
