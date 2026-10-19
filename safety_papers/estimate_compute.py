#!/usr/bin/env python3
"""Compute estimates: one request per paper through the Batches API (batch.py), instructions estimate_prompt.md.

The answer lists the authors' organizations and components: accelerator-hours on a named accelerator, or FLOP and the kind of work; scope reported
(the reported experiments) or project (a whole-project figure the paper states). h100e.py converts them to
H100e-years as AI 2027 counts them (TPP) and adds dead ends: the stated whole-project figure, else 2.5 x reported.

    python estimate_compute.py run --run NAME --model M --effort E   # blind targets (texts without the compute)
    python estimate_compute.py run --full --run NAME ...             # the same targets, whole text
    python estimate_compute.py run --all --run census --model claude-sonnet-5-5 --effort medium
        # every safety paper, whole text -> data/compute_estimates/census/compute.csv
    python estimate_compute.py send | collect [--wait] | status | report --run NAME [--all | --full]
    (--dry-run: cost only; the API key is asked once per command)

blind.py estimate runs the blind and whole-text pilot arms; blind.py dataset / score compare them with the authors'
figures.
"""
import argparse
import csv
import json
from collections import Counter

import batch
import h100e
from common import DATA, HERE, RUNS, read_jsonl

BLIND = DATA / "blind"
COMPONENT = {"type": "object", "additionalProperties": False,
             "required": ["what", "scope", "source", "unit", "amount", "accelerator", "accelerator_named", "kind"],
             "properties": {"what": {"type": "string"}, "scope": {"type": "string", "enum": ["reported", "project"]},
                            "source": {"type": "string", "enum": ["stated", "estimated"]},
                            "unit": {"type": "string", "enum": ["hours", "FLOP"]}, "amount": {"type": "string"},
                            "accelerator": {"type": "string", "enum": h100e.ACCELERATORS + ["none"]},
                            "accelerator_named": {"type": "string", "enum": ["yes", "no"]},
                            "kind": {"type": "string", "enum": ["train", "forward", "generate", "mixed"]}}}
FIELDS = {"components": {"type": "array", "items": COMPONENT},
          "organizations": {"type": "array", "items": {"type": "string"}}, "note": {"type": "string"},
          "seen": {"type": "string", "enum": ["no", "yes"]}}
SCHEMA = {"type": "object", "properties": FIELDS, "required": list(FIELDS), "additionalProperties": False}
FILL = {"ACCELERATORS": "; ".join(h100e.ACCELERATORS)}
PROJECT = ", and of the whole project if the paper states a figure for it"


def targets() -> dict:
    path = BLIND / "targets.csv"
    return {r["id"]: r for r in csv.DictReader(open(path)) if r["use"] == "1"} if path.exists() else {}


def full_text(pid: str) -> str:
    """The paper's text as the blind test gave it: without the header line and END OF PAPER."""
    return "\n".join((DATA / "fulltext" / f"{pid}.txt").read_text().split("\n")[2:-2])


def size_of(path):
    return path.stat().st_size + 400 if path.exists() else None


def work(pid: str) -> str:
    t = TARGETS[pid]
    return t["covers"] if t["level"] == "part" else "all experiments the paper reports"


def ask(text: str, year, what: str) -> str:
    return (f"<paper>\n{text}\n</paper>\n\nThe paper was first posted to arXiv in {year}. Report the accelerator "
            f"compute of: {what}.")


TARGETS = targets()
META = {r["id"]: r for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))} \
    if (RUNS / "census" / "safety_papers.csv").exists() else {}
COMMON = dict(out=DATA / "compute_estimates", prompt=HERE / "estimate_prompt.md", schema=SCHEMA,
              results="estimates.jsonl", key="seen", max_tokens=32000, answer_tokens=700, think_tokens=0, fill=FILL)

# blind targets, text without the compute passages: the work the authors' figure covers
TASK = batch.Task(user=lambda pid, cfg: ask((BLIND / "texts" / f"{pid}.txt").read_text(), TARGETS[pid]["year"],
                                            work(pid)),
                  size=lambda pid: size_of(BLIND / "texts" / f"{pid}.txt") if pid in TARGETS else None,
                  ids=lambda: list(TARGETS), **COMMON)
# the same targets, whole text: does the model read stated figures (and whole-project figures) right?
FULL = batch.Task(user=lambda pid, cfg: ask(full_text(pid), TARGETS[pid]["year"], work(pid) + PROJECT),
                  size=lambda pid: size_of(DATA / "fulltext" / f"{pid}.txt") if pid in TARGETS else None,
                  ids=lambda: list(TARGETS), **COMMON)
# every safety paper, whole text
ALL = batch.Task(user=lambda pid, cfg: ask(full_text(pid), META[pid]["year"],
                                           "all experiments the paper reports" + PROJECT),
                 size=lambda pid: size_of(DATA / "fulltext" / f"{pid}.txt"), ids=lambda: list(META), **COMMON)


def summary(answer: dict, model: str = h100e.MODEL, raw: bool = False) -> dict:
    """H100e-years (reported, project, with dead ends) of one answer; route: how most of the reported compute was
    reached (FLOP, hours on a named accelerator, hours on an assumed one); accelerator: the one with the most hours;
    stated_share: share of the reported compute from the authors' figures; cpu_only: every part of the reported
    experiments ran on CPU cores (such papers are left out of the results, export.py)."""
    by, accel, stated = Counter(), Counter(), 0.0
    for c in answer.get("components") or []:
        if c.get("scope") != "reported":
            continue
        v = h100e.component(c, model, raw) or 0
        by["FLOP" if c.get("unit") == "FLOP" else "hours named" if c.get("accelerator_named") == "yes"
           else "hours assumed"] += v
        if c.get("unit") == "hours":
            accel[c.get("accelerator", "")] += v
        stated += v if c.get("source") == "stated" else 0
    total = sum(by.values())
    rep = [c for c in answer.get("components") or [] if c.get("scope") == "reported"]
    cpu_only = bool(rep) and all(c.get("unit") == "hours" and c.get("accelerator") == "CPU core" for c in rep)
    return {**h100e.totals(answer, model, raw), "cpu_only": cpu_only, "route": max(by, key=by.get) if total else "",
            "accelerator": max(accel, key=accel.get) if accel else "", "stated_share": stated / total if total else 0.0}


def report(args) -> None:
    """-> data/compute_estimates/<run>/compute.csv: one row per paper of the census."""
    rd = ALL.out / args.run
    est = {r["id"]: r for r in read_jsonl(rd / ALL.results)}
    model = json.loads((rd / "run.json").read_text())["model"]
    fmt = lambda v: "" if v is None else f"{v:.6g}"  # noqa: E731
    n = Counter()
    with open(rd / "compute.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "month", "topic", "h100e_years", "reported_h100e_years", "project_h100e_years",
                    "states_compute", "route", "accelerator", "cpu_only", "stated_share", "organizations", "note",
                    "title"])
        for pid, m in META.items():
            e = est.get(pid, {})
            if e.get("seen") in ("RF", "ER") or not e:
                w.writerow([pid, m["year"], m["month"], m["topic"], "", "", "", e.get("seen", ""), "", "", "", "", "",
                            "", m["title"]])
                n[e.get("seen", "missing")] += 1
                continue
            s = summary(e, model)
            n["no valid amount" if s["with_dead_ends"] is None else "zero" if s["with_dead_ends"] == 0 else
              "CPU only" if s["cpu_only"] else "project figure" if s["project"] is not None else "estimate"] += 1
            w.writerow([pid, m["year"], m["month"], m["topic"], fmt(s["with_dead_ends"]), fmt(s["reported"]),
                        fmt(s["project"]), e["seen"], s["route"], s["accelerator"], int(s["cpu_only"]),
                        f"{s['stated_share']:.2f}",
                        "; ".join(e.get("organizations") or []), e.get("note", ""), m["title"]])
    print(f"{len(META):,} papers: " + ", ".join(f"{v:,} {k}" for k, v in n.most_common()) + f" -> {rd / 'compute.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "send", "collect", "status", "report"])
    ap.add_argument("--all", action="store_true", help="every safety paper, whole text")
    ap.add_argument("--full", action="store_true", help="the blind targets, whole text")
    batch.add_arguments(ap, "")
    args = ap.parse_args()
    if not args.run:
        ap.error("--run is required")
    task = ALL if args.all else FULL if args.full else TASK
    if args.action == "report":
        report(args)
        return
    getattr(batch, args.action)(task, args)
    if args.all and args.action == "run" and not args.dry_run:
        report(args)


if __name__ == "__main__":
    main()
