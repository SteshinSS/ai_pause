#!/usr/bin/env python3
"""Random checks: the compartmentalized model when the verifiers can't tell which papers are worth the most and check a
random sample of them, against Table 3's verifiers, who check the papers worth the most first.

    python random_checks.py [--preset eli-08-16-26] [--json FILE]

The scenario is SCENARIO: pauses of 3 and 9 years from SAR. A random share of the planned papers holds, on average,
the same share of their research effort and of their compute, phi_h = psi = q / q_p (pause.random_papers()). A planned
paper nobody checks then adds nothing, so the lab plans only the papers its verifiers can check
(pause.compartmentalized(ranked=False)). With unlimited hiring every paper is checked, in any order: those cells and
hires are Tables 3b and 3c with either. Hiring: the lab with its own staff and at most H hired people, who plan and
check as the staff do, as pause.hiring_curves() computes it, from none to unlimited hiring's hires, and the H with
which random checks match best-first checks with the lab's own staff.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

import pause as P

SCENARIO = {"model": "compartmentalized", "row": "SAR", "years": (3.0, 9.0)}
TABLE = P.ROOT / "site/content/table.json"                     # pause.py --json: best-first cells must be its cells
ORDERS = {"best first": True, "random": False}                 # the verifiers' order: pause.compartmentalized(ranked=)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default=P.PRESETS[0], choices=P.PRESETS, help="baseline parameters (default: %(default)s)")
    ap.add_argument("--json", metavar="FILE", help="also write the results to FILE as JSON, for the website: "
                    "site/content/random_checks.json")
    args = ap.parse_args()
    preset = args.preset

    inp = P.load_inputs()
    base = P.Baseline(preset)
    base.check()
    model, row, years = SCENARIO["model"], SCENARIO["row"], list(SCENARIO["years"])
    solve = P.TRUST_MODELS[model]
    t_p, E_p = P.pause_start(base, P.ROWS[row])
    d = np.linspace(0.0, years[-1], round(years[-1] * P.STEPS_PER_YEAR) + 1)          # years into the pause
    lab = P.merged_lab(base, inp, t_p, d)
    RS_s = inp["RS_safety_2025"]
    own, hiring = P.STAFFINGS
    out = {(o, s): solve(base, lab, E_p, inp, s, ranked=ranked) for o, ranked in ORDERS.items() for s in P.STAFFINGS}
    cells = {k: np.interp(years, d, cumulative_trapezoid(o["RE"], d, initial=0.0)) / RS_s for k, o in out.items()}
    hired = np.interp(years, d, np.maximum.accumulate(out["best first", hiring]["hired"]))

    table = json.loads(TABLE.read_text())
    if table["preset"] == preset:       # best-first checks are the website's Tables 3a-3c
        r = next(r for r in table["rows"] if r["name"] == row)
        i = [P.DURATIONS.index(D) for D in years]
        for got, want in ((cells["best first", own], r["cells"][model]),
                          (cells["best first", hiring], r["unlimited"][model]), (hired, r["hired"][model])):
            if not np.allclose(got, np.take(want, i), rtol=1e-12, atol=0.0):
                sys.exit(f"{TABLE.relative_to(P.ROOT)} is stale: rerun python pause.py --json {TABLE.relative_to(P.ROOT)}")
    # every paper checked: the order doesn't matter
    np.testing.assert_allclose(out["random", hiring]["RE"], out["best first", hiring]["RE"], rtol=1e-12)
    np.testing.assert_allclose(out["random", hiring]["hired"], out["best first", hiring]["hired"], rtol=1e-12)
    q = {o: out[o, own]["q"] for o in ORDERS}
    assert np.array_equal(q["best first"] < 1, q["random"] < 1), "the staff run short at different times"
    short = float(d[np.argmax(q["best first"] < 1)]) if np.any(q["best first"] < 1) else None   # can't check all
    # the share of the lab's research effort, T_w X_e, that counts: q_p phi_h(q / q_p) (46)
    counted = {o: out[o, own]["q_p"] * out[o, own]["phi_h"] for o in ORDERS}

    # hiring: the lab with its own staff and at most H hired people, as pause.hiring_curves(): the gain over own staff
    # is integrated a month at a time and added to the own-staff cell
    def with_hired(order, D, H):
        """The cell of a pause of D years with at most H (array) hired people, the verifiers' order `order`."""
        dm = d[::P.STEPS_PER_YEAR // 12][:round(D * 12) + 1]           # a date a month up to D
        sparse = P.merged_lab(base, inp, t_p, dm)
        RE = lambda h: solve(base, sparse, E_p, inp, own, hired=h, ranked=ORDERS[order])["RE"]
        RE_own = RE(0.0)
        gain = [np.trapezoid(RE(h) - RE_own, dm) if h > 0 else 0.0 for h in np.atleast_1d(H)]
        return cells[order, own][years.index(D)] + np.array(gain) / RS_s

    D = years[-1]
    H = hired[-1] * np.arange(P.CURVE_STEPS + 1) / P.CURVE_STEPS
    curves = {o: with_hired(o, D, H) for o in ORDERS}
    for o, c in curves.items():
        assert np.all(np.diff(c) >= -1e-9 * c[0]), f"{o}: hiring loses research"
        assert abs(c[-1] / cells[o, hiring][-1] - 1) < 1e-3, f"{o}: unlimited hiring's hires don't check everything"
    # the people with whom random checks match best-first checks with the lab's own staff, by bisection on H
    match = []
    for D_, goal, got, most in zip(years, cells["best first", own], cells["random", own], hired):
        if got >= goal * (1 - 1e-12):           # the staff check every paper: nothing to make up
            match.append(0.0)
            continue
        lo, hi = 0.0, most
        while hi - lo > 1e-6 * most:
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if with_hired("random", D_, mid)[0] < goal else (lo, mid)
        match.append((lo + hi) / 2)

    command = " ".join(["python random_checks.py"] + sys.argv[1:])
    year, month = divmod(math.floor(t_p * 12), 12)   # the month t_p falls in
    pct = lambda a, b: f"{a / b - 1:+.1%}"
    head = "| | " + " | ".join(f"{D:g} years" for D in years) + " |\n|---|" + "---:|" * len(years)
    print(f"# Random checks ({preset})\n\n"
          f"Output of `{command}`. The {model} model in pauses from {row} ({month + 1:02d}.{year}): verifiers who "
          "check the papers worth the most\nfirst (best first, Tables 3a-3c) against verifiers who can't tell which "
          "papers are worth the most and check a random\nsample (random), phi_h = psi = q / q_p, and the lab plans "
          "only the papers they can check. AI safety years: the research\nstock the lab builds over RS_safety_2025; "
          "hired: the most at any time of the pause, on top of the lab's own staff.")
    print(f"\n## AI safety years\n\n{head}")
    for s in P.STAFFINGS:
        for o in ORDERS:
            print(f"| {o}, {s} | " + " | ".join(P.fmt(v) for v in cells[o, s]) + " |")
    print("| random / best first, own staff | "
          + " | ".join(pct(a, b) for a, b in zip(cells["random", own], cells["best first", own])) + " |")
    print(f"| people to hire, {hiring}, either order | " + " | ".join(P.fmt(v) for v in hired) + " |")
    print("| people to hire for random checks to match best first, own staff | "
          + " | ".join(P.fmt(v) for v in match) + " |")
    print(f"\nWith its own staff, the lab can plan and check every paper until {short:.2f} years into the pause."
          if short is not None else "\nWith its own staff, the lab plans and checks every paper all pause.")
    print(f"\n## The lab with its own staff, year by year\n\nq: the share of the lab's papers checked; counted: the "
          "share of its research effort T_w X_e that counts, q_p phi_h(q / q_p);\nN_p / L_H: the staff who plan.\n\n"
          "| year | " + " | ".join(f"{k}, {o}" for o in ORDERS for k in ("q", "counted", "N_p / L_H"))
          + " |\n|---|" + "---:|" * 6)
    for t in range(round(D) + 1):
        j = round(t * P.STEPS_PER_YEAR)
        print(f"| {t} | " + " | ".join(f"{v:.3f}" for o in ORDERS
                                        for v in (q[o][j], counted[o][j], out[o, own]["N_p / L_H"][j])) + " |")
    print(f"\n## Hiring, {D:g} years\n\nThe cell with the lab's own staff and at most H hired people, who plan and "
          "check as the staff do (pause.hiring_curves()).\n\n| H | " + " | ".join(ORDERS) + " |\n|---:|---:|---:|")
    for h, a, b in zip(H, *curves.values()):
        print(f"| {P.fmt(h)} | {P.fmt(a)} | {P.fmt(b)} |")
    print(f"\nRandom checks match best-first checks with the lab's own staff ({P.fmt(cells['best first', own][-1])}) "
          f"with {P.fmt(match[-1])} hired, {match[-1] / hired[-1]:.0%} of unlimited hiring's {P.fmt(hired[-1])} and "
          f"{match[-1] / lab['L_H'][0]:.2f} times the lab's own staff, {P.fmt(lab['L_H'][0])}.")

    if args.json:
        every = slice(None, None, P.STEPS_PER_YEAR)                     # a point a year
        Path(args.json).write_text(json.dumps({
            "command": command, "preset": preset,
            "scenario": {"model": model, "row": row, "t_pause": t_p, "month": f"{year}-{month + 1:02d}",
                         "years": years},
            "units": {"cells": "AI safety years: the research stock the lab builds in the pause / RS_safety_2025",
                      "hired": "people hired on top of the lab's own staff: the most at any time of the pause (FTE)"},
            "cells": {o: {s: cells[o, s].tolist() for s in P.STAFFINGS} for o in ORDERS},
            "hired": hired.tolist(), "match": match,
            "staff": float(lab["L_H"][0]), "short": short,
            "yearly": {"t": d[every].tolist(),
                       **{o: {"q": q[o][every].tolist(), "counted": counted[o][every].tolist(),
                              "N_p / L_H": out[o, own]["N_p / L_H"][every].tolist()} for o in ORDERS}},
            "hiring": {"years": D, "hired": H.tolist(), **{o: c.tolist() for o, c in curves.items()}},
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
