#!/usr/bin/env python3
"""Blind test of compute estimates: papers whose authors state their compute, that compute deleted from the text, a
model's estimate from what remains -> data/blind/dataset.csv (one row per paper: the authors' figure and each
estimate, in H100e-years as AI 2027 counts them, h100e.py).

    python blind.py ids        # papers to read: every text-search hit + 2,000 random others (weights for those)
    python blind.py extract    # Sonnet reads them all (flag_compute.py, run `blind`), Opus reads Sonnet's
                               # candidates (run `blind-opus`), then `redact`; asks for the API key once
    python blind.py redact     # -> data/blind/targets.csv, texts/<id>.txt (compute passages deleted)
    python blind.py estimate   # the pilot: Sonnet and Opus estimate every target blind (runs sonnet-blind,
                               # opus-blind) and from its whole text (sonnet-full, opus-full); --arms picks
                               # some; estimate_compute.py; then `dataset` and `score`
    python blind.py dataset    # -> data/blind/dataset.csv
    python blind.py score      # -> data/blind/score.md: accuracy by year, accelerator, route, size; dead ends
    python blind.py plot       # -> data/blind/plot.html: estimate vs authors' figure, log-log
    python blind.py plot --by-year   # -> plot_years.html: one panel per year
    python blind.py extract --dry-run / estimate --dry-run   # costs; nothing sent

Which papers: the text search (compute_pilot.py: GPU-hours, N GPUs for T hours, $ for cloud or API, kWh / CO2,
FLOP) hits 1,694 of the 25,700 safety papers with a full text; in the pilot 22-29% of hits and 1-2.5% of the others
stated a usable figure. All hits are read (weight 1) and 2,000 random others (weight = others / 2,000).

A paper is a target (`use` 1) when both readings give a figure (all reported experiments, a named part, or the
whole project) within a factor of 1.5, the work is not mostly API calls, and the arithmetic evaluates; the authors'
figure is Sonnet's. When the two readings give the same figure different scopes, the narrower one is taken (a part
over all experiments; the whole project over all experiments), noted in `note`. Other candidates get `use` 0 and a
reason in `check`; data/blind/overrides.csv (id, use, level, covers, calc; blank = keep) changes a target by hand.

Deletion: every `amount` and `speed` quote of either reading is cut from the text (exactly; else word for word;
else sentence by sentence; else word for word with gaps of up to 40 words, if that stretch is not much longer than
the quote), then every sentence that still matches the text search or states a run time ("takes 36.9 s", "24h
training") is cut. Hardware passages stay.

Units: the readings' arithmetic (compute_prompt.md) used per-accelerator factors from dense 16-bit throughput with
FP32 accumulate, half the TPP for GeForce cards and cards without tensor cores. `tpp` puts each factor a reading
used back to the TPP factor (h100e.py) of the accelerator its figures name.
"""
import argparse
import csv
import math
import random
import re
import statistics as st
from collections import Counter

import batch
import flag_compute
import h100e
from common import DATA, HERE, RUNS, read_jsonl
from compute_pilot import RX, hits

BLIND = DATA / "blind"
SAMPLE = 2000
SEED = 20261003
USE = ("all", "part", "project")
ARMS = {"sonnet-blind": ("claude-sonnet-5-5", "medium", "blind"), "opus-blind": ("claude-opus-5-5", "medium", "blind"),
        "sonnet-full": ("claude-sonnet-5-5", "medium", "full"), "opus-full": ("claude-opus-5-5", "medium", "full")}
OLD = DATA / "compute_estimates" / "sonnet"  # the first blind run (old instructions: H100e-years by the model)

# (accelerator named in a reading's figures, the factor compute_prompt.md gave it, h100e.TPP key); an accelerator
# the table lacked got "the closest one", so some appear with several factors
READING_FACTORS = [(re.compile(rx), old, key) for rx, old, key in [
    (r"H100.{0,15}PCIe|PCIe.{0,15}H100", 1.0, "H100 PCIe / H800 PCIe"),
    (r"H100|H800|H200", 1.0, "H100 / H800 / H200 / GH200"), (r"\bH20\b", 0.15, "H20"),
    (r"GB200", 2.5, "GB200"), (r"(?<!G)B200", 2.3, "B200"), (r"A100|A800", 0.32, "A100 / A800"),
    (r"6000 ?Ada", 0.37, "RTX 6000 Ada"), (r"A6000", 0.16, "RTX A6000"), (r"L40S", 0.37, "L40S"),
    (r"\bL40\b", 0.18, "L40"), (r"\bL4\b", 0.07, "L4"), (r"\bL4\b", 0.13, "L4"), (r"\bA40\b", 0.15, "A40"),
    (r"\bA10G\b", 0.13, "A10G"), (r"\bA10\b", 0.13, "A10"), (r"\bA30\b", 0.13, "A30"), (r"\bA30\b", 0.15, "A30"),
    (r"A5000|A5500", 0.11, "RTX A5000 / A5500"), (r"A4000|A4500", 0.08, "RTX A4000 / A4500"),
    (r"5090", 0.21, "RTX 5090"), (r"4090", 0.17, "RTX 4090"), (r"4080", 0.10, "RTX 4080"),
    (r"4070 ?Ti", 0.10, "RTX 4070 Ti"), (r"4070(?! ?Ti)", 0.10, "RTX 4070"), (r"4070(?! ?Ti)", 0.07, "RTX 4070"),
    (r"4060", 0.07, "RTX 4060 / 4060 Ti"), (r"3090", 0.07, "RTX 3090 / 3090 Ti"), (r"3080", 0.06, "RTX 3080 / 3080 Ti"),
    (r"3070", 0.06, "RTX 3070 / 3070 Ti"), (r"3070", 0.04, "RTX 3070 / 3070 Ti"),
    (r"3060", 0.04, "RTX 3060 / 3060 Ti"), (r"3060", 0.06, "RTX 3060 / 3060 Ti"),
    (r"2080 ?(?i:ti)", 0.06, "RTX 2080 Ti"), (r"(?i:titan ?rtx)", 0.06, "Titan RTX"),
    (r"(?i:rtx ?2080)(?! ?(?i:ti))", 0.04, "RTX 2080 / 2080 Super"), (r"(?i:rtx ?2070)", 0.04, "RTX 2070 / 2070 Super"),
    (r"(?i:rtx ?2060)", 0.04, "RTX 2060"), (r"(?i:gtx ?16[56]0)", 0.007, "GTX 1660 / 1650"),
    (r"(?i:gtx ?16[56]0)", 0.006, "GTX 1660 / 1650"), (r"V100", 0.13, "V100"), (r"P100", 0.02, "P100"),
    (r"\bT4\b", 0.07, "T4"), (r"(?i:tpu ?-?v2)|v2-\d", 0.045, "TPU v2 (per chip; a v2-8 has 4)"),
    (r"(?i:tpu ?-?v3)|v3-\d", 0.12, "TPU v3 (per chip; a v3-8 has 4)"),
    (r"(?i:tpu ?-?v4)|v4-\d", 0.28, "TPU v4 (per chip; a v4-8 has 4)"), (r"v5e|v5 ?lite", 0.20, "TPU v5e (per chip)"),
    (r"v5p", 0.46, "TPU v5p (per chip; a v5p-8 has 4)"), (r"v6e|Trillium", 0.93, "TPU v6e (per chip)"),
    (r"(?i:ascend)", 0.32, "Ascend 910B"), (r"MI300|MI325", 1.3, "MI300X / MI325X"), (r"MI250", 0.39, "MI250X"),
    (r"MI210", 0.18, "MI210"), (r"(?i:gaudi ?2)", 0.43, "Gaudi 2"), (r"(?i:gaudi ?3)", 1.85, "Gaudi 3"),
    (r"(?i:titan ?v)\b|GV100", 0.11, "Titan V / Quadro GV100"),
    (r"(?i:quadro rtx ?[86]000|rtx ?8000)", 0.13, "Quadro RTX 8000 / 6000"), (r"1080 ?(?i:ti)", 0.011, "GTX 1080 Ti"),
    (r"(?i:gtx ?1080)(?! ?(?i:ti))", 0.011, "GTX 1080 / 1070 Ti"),
    (r"(?i:titan ?xp?)\b", 0.011, "Titan Xp / Titan X (Pascal)"),
    (r"(?i:titan ?x)\b|980 ?Ti", 0.006, "GTX Titan X (Maxwell) / 980 Ti"),
    (r"(?i:gtx ?9[78]0)(?! ?(?i:ti))", 0.006, "GTX 980 / 970"), (r"(?i:gtx ?1070)", 0.007, "GTX 1070"),
    (r"(?i:gtx ?10[56]0)", 0.007, "GTX 1060 / 1050 Ti"), (r"\bP40\b", 0.012, "P40"), (r"\bP4\b", 0.006, "P4"),
    (r"\bM40\b", 0.007, "M40"), (r"\bK40[cm]?\b", 0.005, "K40"), (r"\bK80\b", 0.005, "K80 (per GPU; a board has two)"),
    (r"Apple|\bM[1-4] ?(?:Pro|Max|Ultra)?\b|(?i:mac ?(?:studio|book|mini))", 0.01, "Apple silicon (M1-M4)"),
    (r"(?i:rtx ?pro ?6000)", 0.37, "RTX PRO 6000 Blackwell"),
    (r"(?i:quadro rtx ?5000)|RTX ?5000(?! ?Ada)", 0.11, "Quadro RTX 5000"),
    (r"(?i:quadro rtx ?5000)|RTX ?5000(?! ?Ada)", 0.13, "Quadro RTX 5000"),
    (r"\bCPUs?\b|CPU[- ]core|core[- ](?:hours|years)", 0.01, "CPU core")]]
DECIMAL = re.compile(r"(?<![\w.])\d*\.\d+(?![\w.])")


def reading_text(r: dict) -> str:
    return r.get("stated", "") + " || " + " || ".join(q["text"] for q in r.get("quotes", []))


def tpp(expr: str, *readings) -> tuple:
    """(expr with each factor of an accelerator the readings name replaced by its TPP factor, the accelerators)."""
    text = " || ".join(reading_text(r) for r in readings if r)
    named = [(old, key) for rx, old, key in READING_FACTORS if rx.search(text)]

    old_of = {}
    for old, key in named:
        old_of.setdefault(key, old)
    # a factor no named accelerator had (an average over a mixed fleet, a near neighbour's): scaled by the named
    # accelerators' mean TPP / old factor; only a small decimal in a factor's place (after *, before / + or ")") in
    # hours-based arithmetic, never 0.01 (CPU)
    gpus = {k: o for k, o in old_of.items() if k != "CPU core"}
    scale = math.exp(st.mean(math.log(h100e.factor(k) / o) for k, o in gpus.items())) if gpus else 1

    def sub(m):
        v = float(m.group(0))
        keys = {key for old, key in named if abs(old - v) < 1e-9}
        if keys:
            return f"{math.exp(st.mean(math.log(h100e.factor(k)) for k in keys)):.6g}"
        before, after = m.string[:m.start()].rstrip(), m.string[m.end():].lstrip()
        if gpus and "8766" in expr and 0.004 <= v < 0.1 and v != 0.01 and before.endswith("*") \
                and (not after or after[0] in "/+)"):
            return f"{v * scale:.6g}"
        return m.group(0)
    return DECIMAL.sub(sub, expr or ""), sorted(old_of)


def converted(r: dict, other: dict = None, field: str = "calc"):
    """A reading's figure in H100e-years (TPP), with the accelerators either reading names."""
    return flag_compute.evaluate(tpp(r.get(field, ""), r, other)[0])
DUR = r"\d[\d.,]*\s*(?:hours?|hrs?|h|days?|minutes?|mins?|weeks?|seconds?|s)\b"
SWEEP = list(RX.values()) + [  # the text search, plus run times said without the hardware
    re.compile(rf"\b(?:took|takes|take|taking|lasted|lasts|required|requires|require|completed|finished|trained for|"
               rf"training time of|run ?time of|wall[- ]clock(?: time)? of)\s+(?:about\s+|approximately\s+|around\s+|"
               rf"roughly\s+|~\s?|less than\s+|under\s+|over\s+|nearly\s+|up to\s+)?{DUR}", re.I),
    re.compile(rf"\b{DUR}\s+(?:of\s+)?(?:training|compute|computation|gpu|tpu|wall[- ]clock|runtime|run time)", re.I)]


def run_args(run, ids=None, model=None, effort=None, dry_run=False):
    return argparse.Namespace(run=run, ids=ids, limit=None, model=model, effort=effort, dry_run=dry_run, wait=True)


# --- papers -------------------------------------------------------------------------------------------------------

def ids(args) -> None:
    path = DATA / "compute_flags" / "regex_hits.csv"
    if path.exists():
        n = {r["id"]: int(r["n"]) for r in csv.DictReader(open(path))}
    else:
        have = [r["id"] for r in csv.DictReader(open(DATA / "fulltext" / "index.csv")) if r["status"] == "ok"]
        n = {i: hits(i)["n"] for i in have}
    meta = {r["id"]: r for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))}
    pos = sorted(i for i, k in n.items() if k > 0)
    neg = sorted(i for i, k in n.items() if k == 0)
    drawn = sorted(random.Random(SEED).sample(neg, SAMPLE))
    BLIND.mkdir(parents=True, exist_ok=True)
    with open(BLIND / "sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "topic", "stratum", "weight"])
        w.writerows([i, meta[i]["year"], meta[i]["topic"], "hit", 1] for i in pos)
        w.writerows([i, meta[i]["year"], meta[i]["topic"], "other", f"{len(neg) / SAMPLE:.4f}"] for i in drawn)
    (BLIND / "ids.txt").write_text("\n".join(pos + drawn) + "\n")
    print(f"{len(pos):,} text-search hits + {SAMPLE:,} of {len(neg):,} others -> {BLIND / 'ids.txt'}, sample.csv")


def candidates(run: str) -> list:
    flags = read_jsonl(DATA / "compute_flags" / run / "flags.jsonl")
    return sorted(r["id"] for r in flags if r["level"] in USE and r.get("api") != "main")


def extract(args) -> None:
    first = run_args("blind", str(BLIND / "ids.txt"), "claude-sonnet-5-5", "medium", args.dry_run)
    if args.dry_run:
        batch.send(flag_compute.TASK, first)
        print("  then Opus (effort low) reads Sonnet's candidates, ~600 papers in the pilot's proportions: ~$50")
        return
    batch.run(flag_compute.TASK, first)
    flag_compute.report(first)
    (BLIND / "candidates.txt").write_text("\n".join(candidates("blind")) + "\n")
    second = run_args("blind-opus", str(BLIND / "candidates.txt"), "claude-opus-5-5", "low")
    batch.run(flag_compute.TASK, second)
    flag_compute.report(second)
    redact(args)


# --- deletion -----------------------------------------------------------------------------------------------------

def collapsed(text: str) -> tuple:
    """text with whitespace runs as one space, and the position in `text` of every character of it."""
    out, pos, space = [], [], False
    for k, ch in enumerate(text):
        if ch.isspace():
            if space:
                continue
            ch, space = " ", True
        else:
            space = False
        out.append(ch)
        pos.append(k)
    return "".join(out), pos


def place(quote: str, text: str, flat: str, pos: list, spans: list, toks: list) -> list:
    """Stretches of `text` to delete for one quote; [] if it cannot be placed."""
    q = " ".join(quote.split())
    k = flat.find(q) if q else -1
    if k >= 0:
        return [(pos[k], pos[k + len(q) - 1] + 1)]
    qw = re.findall(r"\w+", quote.lower())
    if not qw:
        return []
    for s in (s for s, t in enumerate(toks) if t == qw[0]):
        if toks[s:s + len(qw)] == qw:
            return [(spans[s][0], spans[s + len(qw) - 1][1])]
    pieces = [p for p in re.split(r"(?<=[.!?;])\s+", quote) if len(p.split()) >= 3]
    if len(pieces) > 1:
        got = [place(p, text, flat, pos, spans, toks) for p in pieces]
        if all(got):
            return [x for g in got for x in g]
    if len(qw) >= 3:  # word for word with gaps (a footnote or page break inside the sentence, scattered cells)
        best = None
        for s in (s for s, t in enumerate(toks) if t == qw[0]):
            k, p, found, last = 0, s, 0, s
            while k < len(qw) and p < len(toks):
                j = next((x for x in range(p, min(len(toks), p + 41)) if toks[x] == qw[k]), None)
                if j is not None:
                    found, last, p = found + 1, j, j + 1
                k += 1
            if found >= 0.85 * len(qw) and (best is None or found > best[0]):
                best = (found, s, last)
        if best:
            a, b = spans[best[1]][0], spans[best[2]][1]
            if b - a <= 3 * len(quote) + 200:
                return [(a, b)]
    return []


def cut(text: str, stretches: list) -> str:
    out, last = [], 0
    for a, b in sorted(stretches):
        if b <= last:
            continue
        out.append(text[last:max(a, last)])
        out.append(" ")
        last = b
    return "".join(out) + text[last:]


def sweep(text: str) -> tuple:
    """Cut every sentence that matches SWEEP; (text, number of sentences cut)."""
    stretches = []
    for rx in SWEEP:
        for m in rx.finditer(text):
            a = max(text.rfind(". ", 0, m.start()), text.rfind(".\n", 0, m.start()), m.start() - 400) + 1
            ends = [e for e in (text.find(". ", m.end()), text.find(".\n", m.end())) if e >= 0]
            b = min(min(ends) + 1 if ends else len(text), m.end() + 400)
            stretches.append((a, b))
    return cut(text, stretches), len(stretches)


def blind_text(pid: str, quotes: list) -> tuple:
    """(text without the compute passages, quotes cut, quotes not placed, sentences cut by the sweep)."""
    lines = (DATA / "fulltext" / f"{pid}.txt").read_text().split("\n")
    text = "\n".join(lines[2:-2])  # without the header line and END OF PAPER
    flat, pos = collapsed(text)
    spans = [(m.start(), m.end()) for m in re.finditer(r"\w+", text)]
    toks = [text[a:b].lower() for a, b in spans]
    stretches, lost = [], 0
    for q in {q["text"] for q in quotes if q["kind"] in ("amount", "speed")}:
        got = place(q, text, flat, pos, spans, toks)
        stretches += got
        lost += not got
    text, swept = sweep(cut(text, stretches))
    return text, len({q["text"] for q in quotes if q["kind"] in ("amount", "speed")}) - lost, lost, swept


def redact(args) -> None:
    s = {r["id"]: r for r in read_jsonl(DATA / "compute_flags" / "blind" / "flags.jsonl")}
    o = {r["id"]: r for r in read_jsonl(DATA / "compute_flags" / "blind-opus" / "flags.jsonl")}
    sample = {r["id"]: r for r in csv.DictReader(open(BLIND / "sample.csv"))}
    path = BLIND / "overrides.csv"
    over = {r["id"]: r for r in csv.DictReader(open(path))} if path.exists() else {}
    (BLIND / "texts").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in candidates("blind"):
        a, b = s[i], o.get(i, {})
        va, vb = flag_compute.evaluate(a.get("calc", "")), flag_compute.evaluate(b.get("calc", ""))  # as read
        level, covers, true, note, reading = a["level"], a.get("covers", ""), converted(a, b), "", a
        if not b:
            check = "no second reading"
        elif b.get("api") == "main":
            check = "Opus: mostly API"
        elif b["level"] not in USE:
            check = f"Opus: {b['level']}"
        elif not va or not vb:
            check = "calc does not evaluate"
        elif max(va, vb) / min(va, vb) > 1.5:
            check = f"figures differ x{max(va, vb) / min(va, vb):.1f}"
        else:
            check = ""
            if b["level"] != level:  # the same figure read with two scopes: take the narrower reading
                narrow = next(x for k in ("part", "project") for x in (a, b) if x["level"] == k)
                level, covers, true, reading = narrow["level"], narrow.get("covers", ""), converted(narrow, a), narrow
                note = f"scope Sonnet {a['level']}, Opus {b['level']}: {level}"
        use = int(not check)
        if i in over:
            v = over[i]
            use = int(v["use"]) if v.get("use") else use
            level = v.get("level") or level
            covers = v.get("covers") or covers
            true = flag_compute.evaluate(v["calc"]) if v.get("calc") else true
            note = (note + "; " if note else "") + "override"
        project = converted(a, b, "project_calc")
        text, placed, lost, swept = blind_text(i, a.get("quotes", []) + b.get("quotes", []))
        (BLIND / "texts" / f"{i}.txt").write_text(text)
        rows.append({"id": i, "year": sample[i]["year"], "topic": sample[i]["topic"], "stratum": sample[i]["stratum"],
                     "weight": sample[i]["weight"], "use": use, "check": check, "note": note, "level": level,
                     "covers": covers,
                     "true_h100e_years": f"{true:.6g}" if true else "",
                     "opus_h100e_years": f"{converted(b, a):.6g}" if vb else "",
                     "project_h100e_years": f"{project:.6g}" if project else "",
                     "accelerators": "; ".join(tpp(reading.get("calc", ""), a, b)[1]),
                     "quotes_cut": placed, "quotes_not_placed": lost, "sentences_swept": swept,
                     "stated": a.get("stated", ""), "why": a.get("why", "")})
    with open(BLIND / "targets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    used = [r for r in rows if r["use"] == 1]
    print(f"{len(rows)} candidates, {len(used)} targets ("
          + ", ".join(f"{v} {k}" for k, v in Counter(r["level"] for r in used).items()) + "); not used: "
          + ", ".join(f"{v} {k}" for k, v in Counter(r["check"].split(" (")[0].split(" x")[0]
                                                     for r in rows if not r["use"]).items())
          + f"; quotes not placed in {sum(r['quotes_not_placed'] > 0 for r in used)} targets"
          f" -> {BLIND / 'targets.csv'}")


# --- estimates ----------------------------------------------------------------------------------------------------

def estimate(args) -> None:
    import estimate_compute
    tasks = {"blind": estimate_compute.TASK, "full": estimate_compute.FULL}
    runs = [(tasks[ARMS[a][2]], run_args(a, model=ARMS[a][0], effort=ARMS[a][1], dry_run=args.dry_run))
            for a in args.arms]
    for task, r in runs:  # every arm's batches first, so they are processed at the same time
        batch.send(task, r)
    if not args.dry_run:
        for task, r in runs:
            batch.run(task, r)
        dataset(args)
        score(args)


def answers(arm: str) -> dict:
    return {r["id"]: r for r in read_jsonl(DATA / "compute_estimates" / arm / "estimates.jsonl")}


def dataset(args) -> None:
    """-> data/blind/dataset.csv: per target the authors' figure (TPP), the old instructions' estimate and, per arm,
    the reported experiments before calibration (raw: the model's hours, FLOP at 40%) and after, the whole-project
    and with-dead-ends H100e-years (calibrated), the leak flag, route (raw) and main accelerator."""
    from estimate_compute import summary
    rows = [r for r in csv.DictReader(open(BLIND / "targets.csv")) if r["use"] == "1"]
    arms = [a for a in ARMS if (DATA / "compute_estimates" / a / "estimates.jsonl").exists()]
    est = {a: answers(a) for a in arms}
    old = {r["id"]: r for r in read_jsonl(OLD / "estimates.jsonl")}
    fmt = lambda v: "" if v is None else f"{v:.6g}"  # noqa: E731
    cols = ["raw", "reported", "project", "dead_ends", "seen", "route", "accelerator", "stated_share", "assumed_share"]
    with open(BLIND / "dataset.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "year", "topic", "stratum", "weight", "level", "covers", "accelerators", "true_h100e_years",
                    "project_h100e_years", "old_estimate", "old_seen"] + [f"{a}_{c}" for a in arms for c in cols])
        for r in rows:
            o = old.get(r["id"], {})
            out = [r["id"], r["year"], r["topic"], r["stratum"], r["weight"], r["level"], r["covers"],
                   r["accelerators"], r["true_h100e_years"], r["project_h100e_years"],
                   fmt(flag_compute.evaluate(o.get("calc", ""))), o.get("seen", "")]
            for a in arms:
                e = est[a].get(r["id"], {})
                if e.get("seen") not in ("no", "yes"):
                    out += ["", "", "", "", e.get("seen", ""), "", "", "", ""]
                    continue
                m, r0 = summary(e, ARMS[a][0]), summary(e, ARMS[a][0], raw=True)
                hrs = [c for c in e["components"] if c["scope"] == "reported" and c["unit"] == "hours"]
                v = [h100e.component(c, ARMS[a][0]) or 0 for c in hrs]
                assumed = sum(x for x, c in zip(v, hrs) if c["accelerator_named"] == "no") / sum(v) if sum(v) else 0
                out += [fmt(r0["reported"]), fmt(m["reported"]), fmt(m["project"]), fmt(m["with_dead_ends"]), e["seen"],
                        r0["route"],
                        m["accelerator"], f"{m['stated_share']:.2f}", f"{assumed:.2f}"]
            w.writerow(out)
    print(f"{len(rows)} papers x {', '.join(arms) or 'no arm yet'} -> {BLIND / 'dataset.csv'}")


# --- scoring ------------------------------------------------------------------------------------------------------

def family(accelerators: str) -> str:
    """The class of the accelerators an authors' figure names (the fastest one)."""
    keys = [k for k in accelerators.split("; ") if k in h100e.TPP]
    if not keys:
        return "none named"
    k = max(keys, key=h100e.factor)
    if h100e.factor(k) >= 0.75:
        return "H100 or newer"
    if re.search(r"^(RTX [345]0|RTX 4060)", k):
        return "GeForce RTX 30-50"
    if re.search(r"^(RTX 20|GTX|Titan X|Titan RTX)", k):
        return "GeForce RTX 20, GTX"
    if re.search(r"^(A100|A30|L40S|TPU v[456]|Gaudi|Ascend|MI2|MI3|Trainium)", k):
        return "A100 class"
    return "V100, T4, TPU v2-3, workstation"


def metrics(pairs: list) -> list:
    """[papers, median est/true, within x1.5, within x3, within x10, typical error (10^mean |log10 error|),
    correlation, slope]"""
    x = [math.log10(t) for t, e in pairs]
    y = [math.log10(e) for t, e in pairs]
    err = [b - a for a, b in zip(x, y)]
    out = [len(pairs), f"x{10 ** st.median(err):.2f}", f"{sum(abs(v) <= math.log10(1.5) for v in err) / len(err):.0%}",
           f"{sum(abs(v) <= math.log10(3) for v in err) / len(err):.0%}",
           f"{sum(abs(v) <= 1 for v in err) / len(err):.0%}", f"x{10 ** st.mean(abs(v) for v in err):.1f}"]
    if len(pairs) > 2 and st.pstdev(x) > 0 and st.pstdev(y) > 0:
        mx, my = st.mean(x), st.mean(y)
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
        out += [f"{sxy / math.sqrt(sxx * syy):.2f}", f"{sxy / sxx:.2f}"]
    else:
        out += ["", ""]
    return out


def table(groups: list) -> list:
    head = ["", "papers", "median est / true", "within x1.5", "within x3", "within x10", "typical error",
            "correlation of logs", "slope"]
    out = ["| " + " | ".join(head) + " |", "|---" + "|---:" * (len(head) - 1) + "|"]
    for name, pairs in groups:
        if pairs:
            out.append("| " + " | ".join([str(name)] + [str(v) for v in metrics(pairs)]) + " |")
    return out


def arms_of(rows: list, kind: str) -> list:
    """The arms of this kind (blind / full) that dataset.csv has."""
    return [a for a in ARMS if ARMS[a][2] == kind and rows and f"{a}_reported" in rows[0]]


def flop_kinds(answer: dict, model: str) -> tuple:
    """(share of the reported H100e-years from FLOP, the kind of work holding most of that FLOP compute), before
    calibration: grouping papers by calibrated values would make the fit depend on itself."""
    flop, total = Counter(), 0.0
    for c in answer.get("components") or []:
        if c.get("scope") == "reported":
            v = h100e.component(c, model, raw=True) or 0
            total += v
            if c.get("unit") == "FLOP":
                flop[c.get("kind") or "train"] += v
    return (sum(flop.values()) / total if total else 0.0), (max(flop, key=flop.get) if flop else "")


def score(args) -> None:
    """-> data/blind/score.md from dataset.csv and the arms' answers: accuracy before calibration (the model's hours,
    FLOP at 40%), the calibration that follows from it, and the results after it."""
    rows = list(csv.DictReader(open(BLIND / "dataset.csv")))
    num = lambda v: float(v) if v not in ("", None) else None  # noqa: E731
    size = lambda h: "<1" if h < 1 else "1-30" if h < 30 else "30-1000" if h < 1000 else ">1000"  # noqa: E731
    true = lambda r: num(r["true_h100e_years"])  # noqa: E731
    md = ["# Blind test: compute estimates in H100e-years (TPP)", "",
          "Authors' figures: the blind test's readings, factors put back to TPP (blind.tpp). Estimates: the components "
          "of each answer converted by h100e.py, before calibration (the model's accelerator-hours as given, FLOP at "
          "40% utilization) unless a section says otherwise. Scored: papers where the arm answered and did not see the "
          "compute. Typical error: 10^(mean |log10 estimate / authors' figure|). Sizes in H100e-hours; dataset.csv holds "
          "H100e-years.", ""]
    for n, a in enumerate(arms_of(rows, "blind")):
        model, ans = ARMS[a][0], answers(a)
        ok = [r for r in rows if r[f"{a}_seen"] == "no" and num(r[f"{a}_raw"])]
        rep = [r for r in ok if r["level"] in ("all", "part")]
        pair = lambda r: (true(r), num(r[f"{a}_raw"]))  # noqa: E731
        cal = lambda r: (true(r), num(r[f"{a}_reported"]))  # noqa: E731
        old = [r for r in rows if r["level"] in ("all", "part") and r["old_seen"] == "no" and num(r["old_estimate"])]
        opair = lambda r: (true(r), num(r["old_estimate"]))  # noqa: E731
        md += [f"## {a}: the reported experiments ({len(rep)} papers whose authors state all or part of their compute; "
               f"{sum(r[f'{a}_seen'] == 'yes' for r in rows)} saw it, "
               f"{sum(r[f'{a}_seen'] in ('RF', 'ER') for r in rows)} refused or failed)", ""]
        md += table([("before calibration", [pair(r) for r in rep])]
                    + ([("old instructions (H100e by the model)", [opair(r) for r in old])] if n == 0 else []))
        md += ["", "### By year", ""]
        md += table([(y, [pair(r) for r in rep if r["year"] == str(y)]) for y in range(2020, 2026)])
        if n == 0:
            md += ["", "Old instructions (Sonnet), same authors' figures:", ""]
            md += table([(y, [opair(r) for r in old if r["year"] == str(y)]) for y in range(2020, 2026)])
        md += ["", "### By the accelerator the authors name", ""]
        md += table([(f, [pair(r) for r in rep if family(r["accelerators"]) == f])
                     for f in sorted({family(r["accelerators"]) for r in rep})])
        md += ["", "### By route (how most of the estimate was reached)", ""]
        md += table([(k, [pair(r) for r in rep if r[f"{a}_route"] == k])
                     for k in ("hours named", "hours assumed", "FLOP")])
        md += ["", "### By size of the authors' figure, H100e-hours", ""]
        md += table([(k, [pair(r) for r in rep if size(true(r) * 8766) == k])
                     for k in ("<1", "1-30", "30-1000", ">1000")])
        c = h100e.CALIBRATION[model]
        hrs = [pair(r) for r in rep if r[f"{a}_route"] in ("hours named", "hours assumed")]
        md += ["", "### Calibration", "",
               f"Hours: on papers whose estimate is mostly accelerator-hours the authors' figure is a median "
               f"x{10 ** st.median(math.log10(t / e) for t, e in hrs):.2f} the estimate; h100e.py multiplies "
               f"estimated hours by {c['hours']:g} (stated hours stay).", "",
               "FLOP: papers whose estimate is at least 80% FLOP, by the kind of work holding most of it; the "
               "utilization that makes the median estimate equal the authors' figure, and the one h100e.py uses:", "",
               "| kind | papers | median est / true at 40% | utilization for x1 | h100e.py |", "|---|---:|---:|---:|---:|"]
        for k in ("train", "forward", "generate", "mixed"):
            g = [pair(r) for r in rep if flop_kinds(ans.get(r["id"], {}), model)[0] >= 0.8
                 and flop_kinds(ans.get(r["id"], {}), model)[1] == k]
            if g:
                m = 10 ** st.median(math.log10(e / t) for t, e in g)
                md.append(f"| {k} | {len(g)} | x{m:.2f} | {0.4 * m:.3f} | {c['utilization'][k]:g} |")
        md += ["", "After calibration:", ""]
        md += table([("every paper", [cal(r) for r in rep if num(r[f"{a}_reported"])])]
                    + [(y, [cal(r) for r in rep if r["year"] == str(y) and num(r[f"{a}_reported"])])
                       for y in range(2020, 2026)])
        proj = [r for r in ok if r["level"] == "project" and num(r[f"{a}_reported"])]
        if proj:
            ratio = sorted(true(r) / num(r[f"{a}_reported"]) for r in proj)
            md += ["", "### Dead ends (calibrated estimates)", "",
                   f"{len(proj)} papers state only a whole-project figure (deleted): whole project / estimate of the "
                   f"reported experiments: median x{st.median(ratio):.2f}, quartiles x{ratio[len(ratio) // 4]:.2f}-"
                   f"x{ratio[3 * len(ratio) // 4]:.2f}. With x{h100e.DEAD_ENDS}:", ""]
            md += table([(f"x{h100e.DEAD_ENDS} x estimate", [(true(r), num(r[f"{a}_dead_ends"])) for r in proj])])
        both = [r for r in rows if r["level"] == "all" and num(r["project_h100e_years"])]
        if both and n == 0:
            ratio = sorted(num(r["project_h100e_years"]) / true(r) for r in both)
            md += ["", f"{len(both)} targets state both all reported experiments and the whole project: whole / "
                       f"reported median x{st.median(ratio):.2f}, quartiles x{ratio[len(ratio) // 4]:.2f}-"
                       f"x{ratio[3 * len(ratio) // 4]:.2f}."]
        md += ["", "### Assumed accelerators", "",
               "| year | papers | mean share of hours-based compute on an assumed accelerator |", "|---|---:|---:|"]
        for y in range(2020, 2026):
            v = [num(r[f"{a}_assumed_share"]) for r in ok if r["year"] == str(y)]
            if v:
                md.append(f"| {y} | {len(v)} | {st.mean(v):.0%} |")
        md.append("")
    blind_arms = arms_of(rows, "blind")
    if len(blind_arms) > 1:
        sets = [{r["id"]: math.log10(num(r[f"{a}_raw"]) / true(r)) for r in rows if r["level"] in ("all", "part")
                 and r[f"{a}_seen"] == "no" and num(r[f"{a}_raw"])} for a in blind_arms[:2]]
        ids = sorted(set(sets[0]) & set(sets[1]))
        x, y = [sets[0][i] for i in ids], [sets[1][i] for i in ids]
        mx, my = st.mean(x), st.mean(y)
        corr = sum((p - mx) * (q - my) for p, q in zip(x, y)) / math.sqrt(sum((p - mx) ** 2 for p in x)
                                                                         * sum((q - my) ** 2 for q in y))
        md += [f"Errors of {blind_arms[0]} and {blind_arms[1]} on the {len(ids)} papers both scored: correlation of "
               f"log errors {corr:.2f}.", ""]
    for a in arms_of(rows, "full"):
        ok = [r for r in rows if r[f"{a}_seen"] in ("no", "yes") and num(r[f"{a}_reported"]) is not None]
        md += [f"## {a}: whole text, reading the stated figures", ""]
        md += table([(lv, [(true(r), num(r[f"{a}_reported"])) for r in ok
                           if r["level"] == lv and num(r[f"{a}_reported"])]) for lv in ("all", "part")])
        proj = [r for r in ok if r["level"] == "project"]
        got = [r for r in proj if num(r[f"{a}_project"])]
        md += ["", f"Whole-project figures: {len(got)} of {len(proj)} papers that state only one got a project "
                   "component:", ""]
        md += table([("project component vs authors", [(true(r), num(r[f"{a}_project"])) for r in got])])
        both = [r for r in ok if r["level"] in ("all", "part") and num(r["project_h100e_years"])]
        extra = sum(bool(num(r[f"{a}_project"])) for r in ok
                    if r["level"] != "project" and not num(r["project_h100e_years"]))
        md += ["", f"{sum(bool(num(r[f'{a}_project'])) for r in both)} of {len(both)} targets with both figures got a "
                   f"project component; {extra} others got one without a known whole-project figure.",
               f"Stated share of the reported compute (mean): "
               f"{st.mean(num(r[f'{a}_stated_share']) for r in ok) if ok else 0:.0%}.", ""]
    (BLIND / "score.md").write_text("\n".join(md))
    print("\n".join(md))


def plot(args) -> None:
    """data/blind/plot.html: estimate vs the authors' figure in H100e-hours, log-log (template blind_plot.html), one
    panel per blind arm; estimates of the reported experiments before calibration, x DEAD_ENDS where the authors'
    figure is the whole project. Papers where an arm refused or saw the compute are left out."""
    import json
    title = {r["id"]: r["title"] for r in csv.DictReader(open(RUNS / "census" / "safety_papers.csv"))}
    rows = list(csv.DictReader(open(BLIND / "dataset.csv")))
    arms = arms_of(rows, "blind")
    rows = [r for r in rows if arms and all(r[f"{a}_seen"] == "no" and r[f"{a}_raw"] and float(r[f"{a}_raw"]) > 0
                                            for a in arms)]
    est = lambda r, a: (float(r[f"{a}_raw"]) * h100e.HOURS  # noqa: E731  H100e-hours, as plotted
                        * (h100e.DEAD_ENDS if r["level"] == "project" else 1))
    data = [{"id": r["id"], "year": r["year"], "level": r["level"], "covers": r["covers"][:200],
             "title": title.get(r["id"], "")[:160], "true": float(r["true_h100e_years"]) * h100e.HOURS,
             **{f"est_{a}": est(r, a) for a in arms}} for r in rows]
    name = {a: ARMS[a][0].replace("claude-", "").replace("-5-5", " 5.5").title() for a in arms}
    labels = {f"est_{a}": name[a] for a in arms}
    if args.by_year:
        panels = [[f"{name[a]}, {y}", f"est_{a}", [str(y)]] for a in arms[:1] for y in range(2020, 2026)]
        sub, out = f"{name[arms[0]]} (medium effort), before calibration, by year.", BLIND / "plot_years.html"
    else:
        panels = [[name[a], f"est_{a}", None] for a in arms]
        sub, out = "Medium effort; before calibration (the model's hours, FLOP at 40%).", BLIND / "plot.html"
    html = (HERE / "blind_plot.html").read_text().replace("/*DATA*/", json.dumps(data).replace("</", "<\\/"))
    out.write_text(html.replace("/*PANELS*/", json.dumps(panels)).replace("/*LABELS*/", json.dumps(labels))
                   .replace("/*SUB*/", sub))
    print(f"{len(data)} papers -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["ids", "extract", "redact", "estimate", "dataset", "score", "plot"])
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS), help="estimate: which arms")
    ap.add_argument("--dry-run", action="store_true", help="extract / estimate: costs only, send nothing")
    ap.add_argument("--by-year", action="store_true", help="plot: one panel per year -> plot_years.html")
    args = ap.parse_args()
    {"ids": ids, "extract": extract, "redact": redact, "estimate": estimate, "dataset": dataset, "score": score,
     "plot": plot}[args.action](args)


if __name__ == "__main__":
    main()
