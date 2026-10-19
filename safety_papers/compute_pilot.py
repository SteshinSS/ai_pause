#!/usr/bin/env python3
"""Pilot of flag_compute.py: a set of safety papers with independent evidence about stated compute, and checks.

    python compute_pilot.py ids                  # -> data/compute_flags/pilot_set.csv, pilot_ids.txt, regex_hits.csv
    python flag_compute.py run --run pilot --ids data/compute_flags/pilot_ids.txt --effort medium
    python flag_compute.py run --run pilot-opus --ids data/compute_flags/pilot_ids.txt --model claude-opus-5-5
    python compute_pilot.py check --run pilot --vs pilot-opus   # -> data/compute_flags/<run>/check.csv, ...

The pilot of 3 Oct 2026 ran these two with the flag instructions of that day (levels all / most / some / hardware /
none, no number); `check` reads those levels.

The set has three groups:
- earlier: every safety paper read by the compute readers of outputs/paper_compute and outputs/rest_compute_2025
  (paper-reader agents with each folder's compute_rubric.md, on arXiv HTML text; mostly 2025). Their
  `stated_compute` quotes and the share of their H100-hours that came from stated figures are a second reading.
- random: 20 random safety papers a year, 2020-2024 (the earlier ones are mostly 2025).
- regex: 17 a year, 2020-2025, among the papers whose text matches a compute-amount pattern (GPU-hours, N GPUs for
  T hours, $ for API or cloud, kWh / CO2, FLOP): likely positives from every year.

`check` compares the flags with the earlier readings, tests whether every quote is in the text (exactly after
collapsing whitespace, or word for word ignoring punctuation), lists papers flagged hardware / none whose text
matches a compute-amount pattern (recall.csv: read them), and earlier quotes that no new quote covers (missed.csv).
"""
import argparse
import csv
import glob
import json
import random
import re
from collections import Counter
from multiprocessing import Pool

from common import DATA, HERE, RUNS, read_jsonl

OUT = DATA / "compute_flags"
TEXTS = DATA / "fulltext"
EARLIER = ["outputs/paper_compute/compute/*.jsonl", "outputs/rest_compute_2025/compute/*.jsonl"]
SEED = 20261003
YEARS = range(2020, 2026)

HW = (r"(?:gpus?|tpus?|a100s?|h100s?|h800s?|h200s?|a800s?|v100s?|p100s?|t4s?|a6000s?|a5000s?|a40s?|l40s?|"
      r"rtx\s?\d{4}(?:\s?ti)?|titan\s?(?:xp?|v|rtx)|gtx\s?\d{3,4}(?:\s?ti)?|tpu\s?v\d\w*|mi\d{3}x?|b200s?|gh200)")
DUR = r"\d[\d,.]*\s*(?:hours?|hrs?|h|days?|minutes?|mins?|weeks?)\b"
PATTERNS = {
    "gpu_hours": rf"\b(?:gpu|tpu|accelerator|chip|node)[\s-]*(?:hours?|hrs?|days?|weeks?|months?|years?)\b"
                 rf"|\b{HW}[\s-]*(?:gpu[\s-]*)?(?:hours?|hrs?|days?)\b",
    "hw_for_time": rf"\b{HW}\b[^.;]{{0,80}}?\b(?:for|in|within|took|takes|take|requires?|required|lasts?|about|"
                   rf"approximately|around|roughly)\s+(?:about\s+|approximately\s+|around\s+|roughly\s+|~\s?)?{DUR}",
    "time_on_hw": rf"\b{DUR}[^.;]{{0,60}}?\bon\s+(?:a\s+single\s+|one\s+|\d+\s*(?:x\s*)?)?(?:nvidia\s+)?{HW}\b",
    "dollars": r"\$\s?\d[\d,.]*\s*(?:k|thousand|million)?\b[^.;]{0,80}?\b(?:api|openai|anthropic|credits?|cloud|aws|"
               r"azure|gcp|compute)\b|\b(?:api|openai|anthropic|credits?|cloud|aws|azure|compute|spent|spend)\b"
               r"[^.;]{0,80}?\$\s?\d[\d,.]*",
    "energy": r"\b\d[\d,.]*\s*(?:kwh|mwh)\b|\bco2\s?-?(?:eq|e)\b|\bcarbon (?:footprint|emissions?)\b",
    "flop": r"\b(?:petaflops?|pf)[\s/-]*(?:s[\s-]*)?days?\b|\b\d(?:\.\d+)?\s?(?:x|×|\*)\s?10\s?\^?\s?\{?\s?\d{2}\s?\}?"
            r"\s*flops?\b|\b\d(?:\.\d+)?\s?e\s?\+?\d{2}\s*flops?\b",
}
RX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}


def body(pid: str) -> str:
    """The paper's text without the header line and END OF PAPER, line breaks as spaces."""
    lines = (TEXTS / f"{pid}.txt").read_text().split("\n")
    return " ".join(lines[2:-2])


def hits(pid: str) -> dict:
    text = body(pid)
    row, snips = {"id": pid}, []
    for k, rx in RX.items():
        found = list(rx.finditer(text))
        row[k] = len(found)
        snips += [text[max(0, m.start() - 80):m.end() + 80] for m in found[:2]]
    row["n"] = sum(row[k] for k in RX)
    row["snippets"] = " || ".join(snips[:6])
    return row


def earlier_readings(safety: set) -> dict:
    """{id: (source, reading)} of the earlier compute readers, safety papers only; the last reading wins."""
    out = {}
    for pattern in EARLIER:
        for fn in sorted(glob.glob(str(HERE.parent / pattern))):
            for line in open(fn):
                line = line.strip()
                try:
                    d = json.loads(line) if line.startswith("{") else None
                except json.JSONDecodeError:
                    continue
                if d and "stated_compute" in d and d.get("id") in safety:
                    out[d["id"]] = (fn.split("/outputs/")[1].split("/")[0], d)
    return out


def earlier_class(d: dict) -> tuple:
    """(class, share of the reading's H100-hours that came from stated figures)."""
    total = d.get("h100_hours") or 0
    stated = sum(c.get("h100_hours") or 0 for c in d.get("components", []) if c.get("kind") == "stated")
    share = stated / total if total else 0.0
    if not d["stated_compute"]:
        return "no quotes", share
    if share >= 0.9:
        return "stated >= 90%", share
    if share >= 0.6:
        return "stated 60-90%", share
    if share > 0:
        return "stated < 60%", share
    return "quotes, no stated amount", share


def ids(args) -> None:
    safety = {r["id"]: r for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))}
    have = sorted(r["id"] for r in csv.DictReader(open(TEXTS / "index.csv")) if r["status"] == "ok")
    print(f"scanning {len(have):,} texts for compute-amount patterns ...", flush=True)
    with Pool(10) as pool:
        rows = pool.map(hits, have, chunksize=200)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "regex_hits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "year", "n", *RX, "snippets"])
        w.writeheader()
        for r in rows:
            w.writerow({"year": safety[r["id"]]["year"], **r})
    regex = {r["id"]: r for r in rows}
    for y in YEARS:
        ys = [r for r in rows if safety[r["id"]]["year"] == str(y)]
        print(f"  {y}: {sum(r['n'] > 0 for r in ys) / len(ys):.0%} of {len(ys):,} papers match; "
              + ", ".join(f"{k} {sum(r[k] > 0 for r in ys) / len(ys):.0%}" for k in RX))

    early = {i: v for i, v in earlier_readings(set(safety)).items() if i in regex}
    rnd = random.Random(SEED)
    chosen = {i: "earlier" for i in early}
    for y in YEARS[:-1]:
        pool_y = [i for i in have if safety[i]["year"] == str(y) and i not in chosen]
        chosen.update({i: "random" for i in rnd.sample(pool_y, 20)})
    for y in YEARS:
        pool_y = [i for i in have if safety[i]["year"] == str(y) and i not in chosen and regex[i]["n"] > 0]
        chosen.update({i: "regex" for i in rnd.sample(pool_y, 17)})
    with open(OUT / "pilot_set.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "group", "earlier_source", "earlier_class", "earlier_stated_share", "earlier_h100",
                    "earlier_quotes", "regex_n", "title"])
        for i in sorted(chosen):
            src, d = early.get(i, ("", {}))
            cls, share = earlier_class(d) if d else ("", "")
            w.writerow([i, safety[i]["year"], chosen[i], src, cls, f"{share:.2f}" if d else "",
                        d.get("h100_hours", ""), " || ".join(d.get("stated_compute", [])), regex[i]["n"],
                        safety[i]["title"]])
    (OUT / "pilot_ids.txt").write_text("\n".join(sorted(chosen)) + "\n")
    print(f"{len(chosen)} papers: " + ", ".join(f"{v} {k}" for k, v in Counter(chosen.values()).items())
          + "; earlier classes: " + ", ".join(f"{v} {k}" for k, v in
                                               Counter(earlier_class(d)[0] for _, d in early.values()).items())
          + f" -> {OUT / 'pilot_set.csv'}, {OUT / 'pilot_ids.txt'}")


def words(s: str) -> str:
    return " ".join(re.findall(r"\w+", s.lower()))


def crosstab(pairs: list, rows: list, cols: list) -> None:
    c = Counter(pairs)
    print(f"{'':26}" + "".join(f"{x:>10}" for x in cols))
    for r in rows:
        if any(c[r, x] for x in cols):
            print(f"{r:26}" + "".join(f"{c[r, x]:10}" for x in cols))


def check(args) -> None:
    levels = ["all", "most", "some", "hardware", "none", "RF", "ER"]
    pilot = {r["id"]: r for r in csv.DictReader(open(OUT / "pilot_set.csv"))}
    flags = {r["id"]: r for r in read_jsonl(OUT / args.run / "flags.jsonl") if r["id"] in pilot}
    other = {r["id"]: r for r in read_jsonl(OUT / args.vs / "flags.jsonl")} if args.vs else {}
    print(f"run {args.run}: {len(flags)} of {len(pilot)} pilot papers flagged\n")

    print("level by group")
    crosstab([(pilot[i]["group"], f["level"]) for i, f in flags.items()], ["earlier", "random", "regex"], levels)
    print("\nearlier reading (rows) vs level (columns)")
    crosstab([(pilot[i]["earlier_class"], f["level"]) for i, f in flags.items() if pilot[i]["group"] == "earlier"],
             ["stated >= 90%", "stated 60-90%", "stated < 60%", "quotes, no stated amount", "no quotes"], levels)
    if other:
        print(f"\n{args.run} (rows) vs {args.vs} (columns)")
        crosstab([(f["level"], other[i]["level"]) for i, f in flags.items() if i in other], levels, levels)

    rows, recall, missed, quotes = [], [], [], Counter()
    for i, f in sorted(flags.items()):
        text = body(i)
        flat, toks = " ".join(text.split()), words(text)
        kinds = Counter()
        for q in f.get("quotes", []):
            qt = " ".join(q["text"].split())
            how = "exact" if qt and qt in flat else "words" if words(qt) and words(qt) in toks else "missing"
            quotes[how] += 1
            kinds[how] += 1
        new = [words(q["text"]) for q in f.get("quotes", [])]
        lost = []
        for e in filter(None, pilot[i]["earlier_quotes"].split(" || ")):
            ew = words(e).split()
            if ew and not any(sum(w in n.split() for w in ew) / len(ew) >= 0.6 for n in new):
                lost.append(e)
        missed += [{"id": i, "level": f["level"], "earlier_quote": e} for e in lost]
        h = hits(i)
        if f["level"] in ("hardware", "none") and h["n"]:
            recall.append({"id": i, "group": pilot[i]["group"], "level": f["level"], "regex_n": h["n"],
                           "snippets": h["snippets"]})
        rows.append({"id": i, "year": pilot[i]["year"], "group": pilot[i]["group"],
                     "earlier_class": pilot[i]["earlier_class"], "level": f["level"],
                     "vs_level": other.get(i, {}).get("level", ""), "stated": f.get("stated", ""),
                     "n_quotes": len(f.get("quotes", [])), "quotes_exact": kinds["exact"],
                     "quotes_words": kinds["words"], "quotes_missing": kinds["missing"], "regex_n": h["n"],
                     "earlier_quotes_missed": len(lost), "why": f.get("why", ""), "title": pilot[i]["title"]})
    rd = OUT / args.run
    for name, data in [("check.csv", rows), ("recall.csv", recall), ("missed.csv", missed)]:
        with open(rd / name, "w", newline="") as fh:
            if data:
                w = csv.DictWriter(fh, list(data[0]))
                w.writeheader()
                w.writerows(data)
    n = sum(quotes.values())
    print(f"\nquotes: {n:,}; in the text exactly {quotes['exact'] / n:.0%}, word for word {quotes['words'] / n:.0%}, "
          f"not found {quotes['missing'] / n:.0%}" if n else "\nno quotes")
    print(f"hardware / none with a compute-amount pattern in the text: {len(recall)} -> {rd / 'recall.csv'}")
    print(f"earlier quotes no new quote covers: {len(missed)} in {len({m['id'] for m in missed})} papers "
          f"-> {rd / 'missed.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["ids", "check"])
    ap.add_argument("--run", default="pilot", help="check: run folder in data/compute_flags (default pilot)")
    ap.add_argument("--vs", help="check: a second run on the same papers, to compare levels")
    args = ap.parse_args()
    {"ids": ids, "check": check}[args.action](args)


if __name__ == "__main__":
    main()
