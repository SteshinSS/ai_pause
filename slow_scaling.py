#!/usr/bin/env python3
"""Slow scaling: an upper bound on how much later the lab has to hand off when it reaches TED-AI later.

    python slow_scaling.py [--preset eli-08-16-26] [--json FILE]

AI 2040's Capability Scaling Strategy compares scaling fast to TED-AI with scaling slowly. Two labs, both the merged lab
of the TED-AI row: the fast lab pauses at TED-AI at t_TED, the TED-AI row's start, and hands off k years later, at
t_h = t_TED + k; the slow lab reaches TED-AI L years later (LAGS) and pauses from then on. Upper bound: the slow lab
does no safety research before TED-AI. From TED-AI on, the two labs research at the same rate on the same date, the
TED-AI row's, so with C(d) = int_0^d RE dt / RS_safety_2025, the TED-AI row's AI safety years after d years, the slow
lab matches the fast lab's research Delta t after t_h, the handoff delay:

    C(k + Delta t) - C(L) = C(k)

Delta t < L, and it shrinks as k grows, since RE grows with the lab's compute. The slow lab spends k - L + Delta t
years at TED-AI. Every trust model with the lab's own staff, as the website's tables show by default. --json also
writes the AC row's start, where the slow lab's pause starts, and RE / RS_safety_2025 and C a month, for the website's
worked example: its sliders set the slow lab's Stage 1 (AC to TED-AI, up to STAGES[0]) and Stage 2 (at TED-AI), and the
fast lab hands off when it has done as much, C(k) = C(L + Stage 2) - C(L). If both labs hand off at t_h instead, the slow lab lacks the share C(min(L, k)) / C(k) of the fast
lab's research (the tables' last section).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

import pause as P

ROW = "TED-AI"
LAGS = (1.0, 2.0, 3.0, 4.0, 5.0)    # years by which the slow lab reaches TED-AI after the fast one
HANDOFF = 15.0                      # years: the latest handoff, k, after the fast lab reaches TED-AI
SHOWN = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0)      # the tables' k
STAGES = (7.0, 15.0)                # the website's worked example: the slow lab's longest Stage 1 (from AC to TED-AI)
                                    # and Stage 2 (at TED-AI), in years; --json writes C far enough for both
TABLE = P.ROOT / "site/content/table.json"              # pause.py --json: C at the tables' durations must be its cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default=P.PRESETS[0], choices=P.PRESETS, help="baseline parameters (default: %(default)s)")
    ap.add_argument("--json", metavar="FILE", help="also write the results to FILE as JSON, for the website: "
                    "site/content/slow_scaling.json")
    args = ap.parse_args()
    preset = args.preset
    command = " ".join(["python slow_scaling.py"] + sys.argv[1:])

    inp = P.load_inputs()
    base = P.Baseline(preset)
    base.check()
    staffing = P.STAFFINGS[0]
    t_p, E_p = P.pause_start(base, P.ROWS[ROW])
    t_ac = P.pause_start(base, P.ROWS["AC"])[0]                    # the slow lab's pause starts here (the website)
    D = math.ceil(max(HANDOFF + max(LAGS), STAGES[0] - (t_p - t_ac) + STAGES[1]))   # years: k + Delta t < k + L, and
                                    # the example's slow lab; whole years keep the grid's steps at 1 / STEPS_PER_YEAR
    d = np.linspace(0.0, D, round(D * P.STEPS_PER_YEAR) + 1)       # years at TED-AI
    lab = P.merged_lab(base, inp, t_p, d)
    RE = {m: f(base, lab, E_p, inp, staffing)["RE"] for m, f in P.TRUST_MODELS.items()}
    C = {m: cumulative_trapezoid(r, d, initial=0.0) / inp["RS_safety_2025"] for m, r in RE.items()}
    for m, r in RE.items():         # research a year rises, so Delta t < L and it shrinks with k
        assert np.all(np.diff(r) > 0), f"{m}: research a year falls at TED-AI"

    table = json.loads(TABLE.read_text())
    if table["preset"] == preset:       # C is the website's TED-AI row
        r = next(r for r in table["rows"] if r["name"] == ROW)
        for m, c in C.items():
            if not np.allclose(np.interp(P.DURATIONS, d, c), r["cells"][m], rtol=1e-12, atol=0.0):
                sys.exit(f"{TABLE.relative_to(P.ROOT)} is stale: rerun python pause.py --json {TABLE.relative_to(P.ROOT)}")

    k = np.arange(round(HANDOFF * 12) + 1) / 12                  # a handoff a month
    dt = {m: {L: np.interp(np.interp(k, d, c) + np.interp(L, d, c), c, d) - k for L in LAGS} for m, c in C.items()}
    year, month = divmod(math.floor(t_p * 12), 12)                 # the month t_p falls in

    print(f"# Slow scaling ({preset})\n\n"
          f"Output of `{command}`. The fast lab pauses at {ROW} ({month + 1:02d}.{year}) and hands off k years later; "
          "the slow lab reaches TED-AI L years later\nand does no safety research before it. Delta t: how much later "
          "the slow lab has to hand off to match the fast lab's research,\nC(k + Delta t) - C(L) = C(k), C the "
          f"{ROW} row's AI safety years ({staffing}). In years.")
    at = lambda m, L, x: float(np.interp(x, k, dt[m][L]))
    for m in C:
        print(f"\n## {m}\n\n| k | handoff | " + " | ".join(f"L = {L:g}" for L in LAGS) + " | research a year, x the first |"
              "\n|---:|---:|" + "---:|" * (len(LAGS) + 1))
        for x in SHOWN:
            th = math.floor((t_p + x) * 12)
            print(f"| {x:g} | {th % 12 + 1:02d}.{th // 12} | " + " | ".join(f"{at(m, L, x):.2f}" for L in LAGS)
                  + f" | {np.interp(x, d, RE[m]) / RE[m][0]:.1f} |")
    print("\n## The research the slow lab lacks if both hand off at t_h\n\nC(min(L, k)) / C(k): the share of the fast lab's AI "
          "safety years the slow lab hasn't done when both hand off k years after the fast lab's TED-AI, in %.")
    for m, c in C.items():
        print(f"\n### {m}\n\n| k | handoff | " + " | ".join(f"L = {L:g}" for L in LAGS) + " |\n|---:|---:|"
              + "---:|" * len(LAGS))
        for x in SHOWN[1:]:
            th = math.floor((t_p + x) * 12)
            print(f"| {x:g} | {th % 12 + 1:02d}.{th // 12} | " + " | ".join(
                f"{100 * np.interp(min(L, x), d, c) / np.interp(x, d, c):.1f}" for L in LAGS) + " |")

    if args.json:
        at_k = np.arange(round(D * 12) + 1) / 12                     # a month, from TED-AI to the slow lab's last handoff
        Path(args.json).write_text(json.dumps({
            "command": command, "preset": preset, "row": ROW, "staffing": staffing,
            "t_ted": t_p, "month": f"{year}-{month + 1:02d}", "t_ac": t_ac,
            "units": {"k": "years from the fast lab's TED-AI to its handoff, a point a month",
                      "lags": "years by which the slow lab reaches TED-AI after the fast one",
                      "delta_t": "years: how much later the slow lab hands off, an upper bound",
                      "t_ac": "year: the AC row's start, where the slow lab's pause starts",
                      "d": "years after the fast lab reaches TED-AI, a point a month",
                      "rate": "AI safety years a year at TED-AI, RE / RS_safety_2025, at d",
                      "safety_years": "AI safety years since TED-AI, C(d), at d",
                      "stages": "years: the example's longest Stage 1 and Stage 2 of the slow lab"},
            "lags": list(LAGS), "k": [round(x, 6) for x in k.tolist()], "stages": list(STAGES),
            "delta_t": {m: [[round(x, 4) for x in v.tolist()] for v in by_lag.values()] for m, by_lag in dt.items()},
            "d": [round(x, 6) for x in at_k.tolist()],
            "rate": {m: [round(x, 1) for x in (np.interp(at_k, d, r) / inp["RS_safety_2025"]).tolist()]
                     for m, r in RE.items()},
            "safety_years": {m: [round(x, 1) for x in np.interp(at_k, d, c).tolist()] for m, c in C.items()},
        }, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
