#!/usr/bin/env python3
"""Tornado plots: how one scenario of pause.py moves when one input at a time moves to its 10th and then its 90th
percentile, the others at their values, as AIFM's own tornado plots do.

    python sensitivity.py [--preset eli-08-16-26] [--json FILE]

The scenario is SCENARIO: the compartmentalized model in a pause of 9 years from SAR. Its two results, each with its
staffing (OUTPUTS), are those of pause.py's Tables 3a and 3c: AI safety years with the lab's own staff, frozen at the
start of the pause, the research stock it builds over RS_safety_2025; and people to hire with unlimited hiring, the
most the lab has hired at any time of the pause. With every input at its value they are the tables' cells, the middle
of every plot.

Our inputs: those of pause_inputs.csv with a p10 and a p90. RS_safety_2025 is not one of them: it is rs_safety() of
C_safety_2025 and L_safety_2025 (16), so it moves with them, in proportion, and is the CSV's at their values.

AIFM's parameters: those the preset's forecast draws from a distribution over numbers
(aifm/app/forecast/data/simulations/<preset>/sampling_config.yaml), at that distribution's 10th and 90th percentiles as
AIFM's tornado plots take them (percentile()). AIFM runs again with that one parameter changed from the preset's value
and the others at the preset's: the pause starts at that run's SAR, and RS_safety_2025 moves with rs_safety() of that
run. Distributions over versions of the model, such as how AC is anchored, stay at the preset's version.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import cumulative_trapezoid
from scipy.special import betaincinv, ndtri

import pause as P

SCENARIO = {"model": "compartmentalized", "row": "SAR", "years": 9.0}
FORECASTS = P.ROOT / "aifm/app/forecast/data/simulations"     # a published forecast per preset: sampling_config.yaml
TABLE = P.ROOT / "site/content/table.json"                     # pause.py --json: the middle must be its cells
NAMES = {**json.loads((P.ROOT / "aifm/config/parameter_display_names.json").read_text()),
         # AIFM's display names lack these
         "excess_uplift_doubling_time": "Doubling Time of the Excess Coding Uplift",
         "present_revenue_growth_rate": "Present Revenue Growth Rate",
         "revenue_required_for_ac": "Revenue Required for AC",
         "serial_uplift_at_reference_date": "Serial Coding Uplift at the Reference Date"}
NO_EFFECT = 1e-9            # a result within this share of the middle's hasn't moved
OUTPUTS = {"years": ("AI safety years", "own staff"), "hired": ("people to hire", "unlimited hiring")}   # staffing


def scenario(base, inp):
    """(AI safety years, people to hire, t_p) of SCENARIO with the inputs inp, each with its staffing of OUTPUTS, as
    pause.py's main() computes the cells of its tables."""
    milestone = P.ROWS[SCENARIO["row"]]
    if "time" not in base.r["milestones"][milestone]:
        sys.exit(f"{milestone} is not reached in a run of AIFM")
    t_p, E_p = P.pause_start(base, milestone)
    d = np.linspace(0.0, P.DURATIONS[-1], round(P.DURATIONS[-1] * P.STEPS_PER_YEAR) + 1)    # years into the pause
    if t_p + d[-1] > base.data.time[-1]:
        sys.exit(f"a pause from {t_p:.2f} outlasts AIFM's series")
    lab = P.merged_lab(base, inp, t_p, d)
    out = {k: P.TRUST_MODELS[SCENARIO["model"]](base, lab, E_p, inp, staffing) for k, (_, staffing) in OUTPUTS.items()}
    D = SCENARIO["years"]
    stock = np.interp(D, d, cumulative_trapezoid(out["years"]["RE"], d, initial=0.0))
    hired = np.interp(D, d, np.maximum.accumulate(out["hired"].get("hired", 0.0 * d)))
    return float(stock / inp["RS_safety_2025"]), float(hired), float(t_p)


def percentile(name, spec, q):
    """The q-quantile of AIFM's parameter `name` under `spec`, its entry of a sampling_config.yaml, as AIFM's tornado
    plots take it (scripts/analysis/tornado_plot.py of aifm-public, not vendored: sample_at_quantile(), clip_to_bounds()):
    a normal or lognormal with 10th and 90th percentiles ci80, plus its shift, clipped to min and max when the entry
    has both; a beta stretched over min to max; a uniform from min to max; then clipped to AIFM's PARAMETER_BOUNDS. A
    choice among numbers, which AIFM's plots skip, gives the least whose cumulative probability reaches q. None for a
    choice among anything else."""
    kind = spec["dist"]
    if kind in ("normal", "lognormal", "shifted_lognormal"):
        lo, hi = sorted(float(v) for v in spec["ci80"])
        z = ndtri(q) / ndtri(0.9)                                   # -1 at the 10th percentile, 1 at the 90th
        x = ((lo + hi) / 2 + z * (hi - lo) / 2 if kind == "normal"
             else float(spec.get("shift", 0.0)) + math.sqrt(lo * hi) * (hi / lo) ** (z / 2))
        if "min" in spec and "max" in spec:
            x = min(max(x, float(spec["min"])), float(spec["max"]))
    elif kind == "beta":
        lo, hi = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
        x = lo + (hi - lo) * betaincinv(float(spec["alpha"]), float(spec["beta"]), q)
    elif kind == "uniform":
        x = float(spec["min"]) + q * (float(spec["max"]) - float(spec["min"]))
    elif kind == "choice" and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in spec["values"]):
        n = len(spec["values"])
        values, p = zip(*sorted(zip(spec["values"], spec.get("p") or [1 / n] * n)))
        x = values[min(int(np.searchsorted(np.cumsum(p), q - 1e-12)), n - 1)]
    elif kind == "choice":
        return None
    else:
        raise ValueError(f"{name}: unknown distribution {kind}")
    bounds = P.cfg.PARAMETER_BOUNDS.get(name)
    return float(np.clip(x, *bounds) if bounds else x)


def moves(ends, middle):
    """Whether either end is off the middle by more than NO_EFFECT of it."""
    return any(abs(v - middle) > NO_EFFECT * abs(middle) for v in ends)


def month(t):
    """The month t falls in, as the tables' rows write it: 02.2029."""
    year, m = divmod(math.floor(t * 12), 12)
    return f"{m + 1:02d}.{year}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default=P.PRESETS[0], choices=P.PRESETS, help="baseline parameters (default: %(default)s)")
    ap.add_argument("--json", metavar="FILE", help="also write the results to FILE as JSON, for the website: "
                    "site/content/sensitivity.json")
    args = ap.parse_args()
    preset = args.preset

    inp = P.load_inputs()
    with open(P.ROOT / "pause_inputs.csv", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    base = P.Baseline(preset)
    base.check()
    # RS_safety_2025 moves with rs_safety() of the 2025 field's people and compute and of AIFM's run, in proportion
    ref = P.rs_safety(base, inp["C_safety_2025"], inp["L_safety_2025"])
    scaled = lambda b, x: {**x, "RS_safety_2025": inp["RS_safety_2025"] *
                           P.rs_safety(b, x["C_safety_2025"], x["L_safety_2025"]) / ref}
    years, hired, t_p = scenario(base, inp)
    middle = {"years": years, "hired": hired}

    table = json.loads(TABLE.read_text())
    if table["preset"] == preset:       # the middle of every plot is the website's cells
        row = next(r for r in table["rows"] if r["name"] == SCENARIO["row"])
        i = P.DURATIONS.index(SCENARIO["years"])
        model, (_, own), (_, hiring) = SCENARIO["model"], OUTPUTS["years"], OUTPUTS["hired"]
        cells = (row["cells" if own == P.STAFFINGS[0] else "unlimited"][model][i],
                 row["hired"][model][i] if hiring == P.STAFFINGS[1] else 0.0)
        if not np.allclose((years, hired), cells, rtol=1e-12, atol=0.0):
            sys.exit(f"{TABLE.relative_to(P.ROOT)} is stale: rerun python pause.py --json {TABLE.relative_to(P.ROOT)}")

    ours, fixed = [], []
    for r in csv_rows:
        if not r["p10"] and not r["p90"]:
            fixed.append(r["name"])
            continue
        assert r["p10"] and r["p90"], f"pause_inputs.csv: {r['name']} needs both p10 and p90"
        ends = []
        for q in ("p10", "p90"):
            x = {**inp, r["name"]: float(r[q])}
            P.check_inputs(x)
            ends.append(scenario(base, scaled(base, x)))
        ours.append({"name": r["name"], "unit": r["unit"], "value": inp[r["name"]],
                     "p10": float(r["p10"]), "p90": float(r["p90"]),
                     **{k: [e[j] for e in ends] for j, k in enumerate(OUTPUTS)}})

    spec = yaml.safe_load((FORECASTS / preset / "sampling_config.yaml").read_text())["parameters"]
    values = P.load_preset(preset)
    aifm, kept = [], []
    for name, s in spec.items():
        if s["dist"] == "fixed":
            continue
        p10, p90 = percentile(name, s, 0.1), percentile(name, s, 0.9)
        if p10 is None:                                                 # a choice of the model's version
            kept.append({"name": name, "value": values[name]})
            continue
        assert name in values, f"{name} is not a parameter of the preset"
        ends = []
        for v in (p10, p90):
            run = P.Baseline(preset, **{name: v})
            run.check()
            ends.append(scenario(run, scaled(run, inp)))
        aifm.append({"name": name, "label": NAMES.get(name, name), "dist": s["dist"], "value": float(values[name]),
                     "median": percentile(name, s, 0.5), "p10": p10, "p90": p90,
                     **{k: [e[j] for e in ends] for j, k in enumerate(OUTPUTS)}, "t_pause": [e[2] for e in ends]})
    for item in ours + aifm:
        item["moves"] = {k: moves(item[k], middle[k]) for k in OUTPUTS}

    command = " ".join(["python sensitivity.py"] + sys.argv[1:])
    D, model, row = (SCENARIO[k] for k in ("years", "model", "row"))
    change = lambda v, k: f"{P.fmt(v)} ({v / middle[k] - 1:+.1%})" if middle[k] else P.fmt(v)
    print(f"# Sensitivity analysis ({preset})\n\n"
          f"Output of `{command}`. Tornado plots of one scenario: the {model} model in a pause of {D:g} years from {row} "
          f"({month(t_p)}),\nwhose AI safety years with {OUTPUTS['years'][1]} (Table 3a) are {P.fmt(years)} and people "
          f"to hire with {OUTPUTS['hired'][1]} (Table 3c) {P.fmt(hired)}, with every input at its value.\n"
          "One input at a time moves to its 10th and then its 90th percentile, the others at their values. "
          "RS_safety_2025 moves with rs_safety()\nof C_safety_2025 and L_safety_2025 and of AIFM's run, in proportion.")
    head = ("".join(f"| {title}, {staffing}, at p10 | at p90 " for title, staffing in OUTPUTS.values()) + "|",
            "---:|" * 4)
    print(f"\n## Our inputs (pause_inputs.csv)\n\n| input | value | p10 | p90 {head[0]}\n|---|" + "---:|" * 3 + head[1])
    for it in ours:
        print(f"| {it['name']} | {it['value']:g} | {it['p10']:g} | {it['p90']:g} | "
              + " | ".join(change(v, k) for k in OUTPUTS for v in it[k]) + " |")
    print(f"\nNot varied, no p10 and p90: {', '.join(fixed)} (RS_safety_2025 moves with C_safety_2025 and L_safety_2025).")
    print(f"\n## AIFM's parameters (the distributions of the {preset} forecast)\n\n"
          f"| parameter | distribution | preset | median | p10 | p90 | {row} at p10 | at p90 {head[0]}\n|---|---|"
          + "---:|" * 6 + head[1])
    for it in aifm:
        print(f"| {it['name']} | {it['dist']} | {it['value']:g} | {it['median']:g} | {it['p10']:g} | {it['p90']:g} | "
              + " | ".join(month(t) for t in it["t_pause"]) + " | "
              + " | ".join(change(v, k) for k in OUTPUTS for v in it[k]) + " |")
    print("\nKept at the preset's, as distributions over versions of the model: "
          + ", ".join(f"{k['name']} = {k['value']}" for k in kept) + ".")
    for k, (title, staffing) in OUTPUTS.items():
        none = [it["name"] for it in ours + aifm if not it["moves"][k]]
        print(f"\nNo effect on {title} ({staffing}): {', '.join(none) or 'none'}.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "command": command, "preset": preset,
            "scenario": {**SCENARIO, "staffing": {k: staffing for k, (_, staffing) in OUTPUTS.items()},
                         "t_pause": t_p},
            "units": {"years": "AI safety years: the research stock the lab builds in the pause / RS_safety_2025",
                      "hired": "people to hire on top of the lab's own staff: the most at any time of the pause (FTE)"},
            "middle": middle,
            "ours": ours, "fixed": fixed,
            "aifm": aifm, "kept": kept,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
