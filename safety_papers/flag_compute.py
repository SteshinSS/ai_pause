#!/usr/bin/env python3
"""The compute the authors state for their own experiments, as one number, with every passage that reveals it.

One request per paper (full text from data/fulltext/<id>.txt, instructions compute_prompt.md) through the Batches API
(batch.py). The answer: quotes (amount / speed / hardware passages, verbatim, for deleting them later), stated (the
figures in the authors' units), api (none / minor / main), level (all / part / project / other / hardware / none),
covers (the work the number covers), calc (arithmetic giving H100e-years; evaluated here, not by the model),
project_calc (a whole-project figure besides), why.

    python flag_compute.py run --dry-run --run NAME --ids FILE   # cost; sends nothing
    python flag_compute.py run --run NAME --ids FILE             # send, wait, collect; asks for the API key once
    python flag_compute.py report --run NAME                     # -> data/compute_flags/NAME/flags.csv
    python flag_compute.py send | collect [--wait] | status      # the steps one by one

Without --ids a run takes every paper of data/runs/census/safety_papers.csv. --model / --effort on the first send.
"""
import argparse
import ast
import csv
import operator
import re
from collections import Counter

import batch
from common import DATA, HERE, RUNS, read_jsonl

TEXTS = DATA / "fulltext"
LEVELS = ["all", "part", "project", "other", "hardware", "none"]
QUOTE = {"type": "object", "additionalProperties": False, "required": ["kind", "text"],
         "properties": {"kind": {"type": "string", "enum": ["amount", "speed", "hardware"]},
                        "text": {"type": "string"}}}
FIELDS = {"quotes": {"type": "array", "items": QUOTE}, "stated": {"type": "string"},
          "api": {"type": "string", "enum": ["none", "minor", "main"]}, "level": {"type": "string", "enum": LEVELS},
          "covers": {"type": "string"}, "calc": {"type": "string"}, "project_calc": {"type": "string"},
          "why": {"type": "string"}}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate(expr: str, zero: bool = False):
    """Value of an arithmetic expression (numbers, + - * / ** and parentheses; ^ read as a power and "(4+7)e9" as
    "(4+7)*1e9"); None if it is anything else, or if it is not positive (0 allowed with zero=True)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        expr = re.sub(r"\)\s*[eE]\+?(\d+)", r")*1e\1", expr.replace("×", "*").replace(",", "").replace("^", "**"))
        v = ev(ast.parse(expr.strip(), mode="eval").body)
        return float(v) if v > 0 or (zero and v == 0) else None
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def text_size(pid: str):
    p = TEXTS / f"{pid}.txt"
    return p.stat().st_size if p.exists() else None


TASK = batch.Task(
    out=DATA / "compute_flags", prompt=HERE / "compute_prompt.md",
    schema={"type": "object", "properties": FIELDS, "required": list(FIELDS), "additionalProperties": False},
    results="flags.jsonl", key="level",
    user=lambda pid, cfg: (f"<paper>\n{(TEXTS / f'{pid}.txt').read_text()}\n</paper>\n\n"
                           "Report the compute the authors state, as the instructions say."),
    size=text_size, ids=lambda: [r["id"] for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))],
    answer_tokens=400)


def report(args) -> None:
    rd = TASK.out / args.run
    flags = {r["id"]: r for r in read_jsonl(rd / TASK.results)}
    meta = {r["id"]: r for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))}
    order = {k: n for n, k in enumerate(LEVELS + ["RF", "ER"])}
    with open(rd / "flags.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "topic", "level", "api", "h100e_years", "covers", "calc", "project_h100e_years",
                    "stated", "why", "n_quotes", "title", "quotes"])
        for i in sorted(flags, key=lambda i: (order[flags[i]["level"]], i)):
            r, m = flags[i], meta.get(i, {})
            w.writerow([i, m.get("year", ""), m.get("topic", ""), r["level"], r.get("api", ""),
                        evaluate(r.get("calc", "")) or "", r.get("covers", ""), r.get("calc", ""),
                        evaluate(r.get("project_calc", "")) or "", r.get("stated", ""), r.get("why", ""),
                        len(r.get("quotes", [])), m.get("title", ""),
                        " || ".join(f"{q['kind']}: {q['text']}" for q in r.get("quotes", []))])
    c = Counter(r["level"] for r in flags.values())
    bad = [i for i, r in flags.items() if r["level"] in ("all", "part", "project") and not evaluate(r.get("calc", ""))]
    print(", ".join(f"{c[k]:,} {k}" for k in LEVELS + ["RF", "ER"] if c[k])
          + f"; {sum(r.get('api') == 'main' for r in flags.values())} mostly API; "
          f"{len(bad)} with a number but no valid calc {bad[:5]} -> {rd / 'flags.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "send", "collect", "status", "report"])
    batch.add_arguments(ap, "census")
    args = ap.parse_args()
    if args.action == "report":
        report(args)
    else:
        getattr(batch, args.action)(TASK, args)
        if args.action == "run" and not args.dry_run:
            report(args)


if __name__ == "__main__":
    main()
