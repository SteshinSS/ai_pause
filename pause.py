#!/usr/bin/env python3
"""A pause on the AI Futures Model: the safety research a merged US lab does while effective compute is frozen.

    python pause.py [--preset PRESET] [--headcount staff.csv] [--json FILE]

AIFM (aifm/, unmodified) is run once as published. Each row of the tables freezes effective compute E from the start
of the pause on, so AIFM's own functions can be evaluated on the merged lab's inputs (pause_inputs.csv) with no ODE. A
trust model says how much of the lab's research counts. A cell is the research it does in a pause of D years, in years
of the world's 2025 technical safety research. The model is written up in site/content/notes/model.mdx; (n) in the
comments is its formula n.
"""
import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import log_ndtr, ndtr, ndtri, wrightomega

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "aifm"))
from progress_model import Parameters, ProgressModel, load_time_series_data  # noqa: E402
from progress_model import config as cfg  # noqa: E402
from progress_model.progress_rate import (compute_aggregate_research_taste, compute_ai_research_taste,  # noqa: E402
                                          compute_research_effort)

logging.disable(logging.WARNING)    # AIFM logs every solver step

PRESETS = ("eli-08-16-26", "daniel-08-16-26", "brendan-08-16-26")   # AIFM's website presets; the first is ours
ROWS = {                            # when the pause starts: a date, or an AIFM milestone the baseline crosses
    "pre-AC": 2027 + 5 / 12,
    "AC": "AC",
    "SAR": "SAR-level-experiment-selection-skill",
    "TED-AI": "TED-AI",
}
DURATIONS = (0.25, 0.75, 1.5, 3.0, 9.0)   # years
STAFFINGS = ("own staff", "unlimited hiring")
TRUST_FROM = {"verified": "AC", "compartmentalized": "SAR", "approved": "SAR"}   # the row each trust model starts at;
                                    # before it, the simpler model (38), (51), (54)
ENGINEER = "AC"                     # compartmentalized: the engineer's E is this row's
T_SAFETY = 2025.5                   # RS_safety_2025: AI assistance as in the leading lab at mid-2025
STEPS_PER_YEAR = 48                 # trapezoid rule; 480 moves no cell by more than 2e-5
STAFF_FLOOR = 1e-3                  # the grids of N_e and N_p start at this share of the staff
STAFF_GRID = 481                    # log-spaced points of those grids; 4x as many move no cell by more than 2e-4
RESEARCH_TOL = 0.01                 # unlimited hiring: research effort the lab may give up to hire fewer people
BAR_GRID = 4001                     # TopPapers: points of each half of the grid of the bar u
TASTE_GRID = 12001                  # Baseline.taste_advantage: points from the AI's taste rank to 12 SD above it
PAPERS_DATASET = ROOT / "safety_papers/results/safety_papers_compute.csv"   # its 2025 count is n_s
STAFF = "m_L x L_HUMAN(t_p), AIFM's engineers at the start of the pause, the same all pause"   # default headcount

# --json only
SITE_TABLE = "compartmentalized"    # the trust model of the website's main table
PAPER_COUNT = "verified"            # the trust model whose paper count n(t) the website plots
PILE_ROW, PILE_T, PILE_SIZE = "SAR", 2029.5, 400    # pile_sample()
CURVE_STEPS = 12                    # hiring_curves(): steps from nobody hired to unlimited hiring's hires
AUDITORS = (100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)   # auditor_curves()


def load_preset(name):
    """A website preset (aifm/constants/parameters.ts) as kwargs of Parameters, converted as the website's API does."""
    ts = (ROOT / "aifm/constants/parameters.ts").read_text()
    block = ts.split(f"'{name}': {{", 1)[1].split("} as ParametersType", 1)[0]
    raw = {}
    for line in block.splitlines():
        key, colon, value = line.split("//")[0].strip().rstrip(",").partition(":")
        if colon:
            raw[key.strip()] = json.loads(value.strip().replace("'", '"'))
    raw["ac_time_horizon_minutes"] = 10.0 ** raw["ac_time_horizon_minutes"]   # the website stores log10(minutes)
    raw["parallel_penalty"] = raw.pop("coding_labor_exponent")
    raw["include_gap"] = "gap" if raw.pop("benchmarks_and_gaps_mode") else "no gap"
    init = {f.name for f in fields(Parameters) if f.init}
    return {k: v for k, v in raw.items() if k in init}


class Baseline:
    """One run of AIFM as published, or with `changes` to the preset (sensitivity.py). Afterwards L_C, X and T can be
    evaluated at any inputs and any E. E is passed as AIFM's state `progress` = log10 E, which is the milestones'
    `effective_compute_ooms` plus ooms_offset."""

    def __init__(self, preset, **changes):
        self.data = load_time_series_data(str(ROOT / "aifm/inputs/input_data.csv"))
        model = ProgressModel(Parameters(**{**load_preset(preset), **changes}), self.data)
        model.compute_progress_trajectory([2017.0, 2200.0], initial_progress=0.0)   # the website's time range
        self.p, self.r = model.params, model.results
        self.times, self.progress = np.asarray(self.r["times"]), np.asarray(self.r["progress"])
        self.ooms_offset = self.progress[0] - self.r["effective_compute"][0]

    def series(self, name, t):
        """An exogenous series of AIFM at t, log-interpolated as in the model."""
        return np.exp(np.interp(t, self.data.time, np.log(getattr(self.data, name))))

    def coding_labor(self, L_H, C_aut, progress):
        """Parallel coding labor L_C of L_H humans and AI coders on C_aut H100e (6)-(10). L_H = 0 gives the limit
        L_H -> 0, AIFM's AI-only coding labor, positive from AC on. Arrays broadcast."""
        L_H, C_aut, progress = np.broadcast_arrays(L_H, C_aut, progress)
        ln_E = math.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress
        automation = self.p.automation_model
        L_C = automation.coding_labor_optimal_ces(L_H.ravel(), C_aut.ravel(), ln_E.ravel(), self.p)
        if L_C is None or not np.all(np.isfinite(L_C)):   # AIFM logs the error and returns None
            raise RuntimeError("coding_labor_optimal_ces failed")
        nobody = L_H.ravel() <= 0
        if np.any(nobody):
            A = automation.get_automation_fraction(progress.ravel()[nobody])
            L_C[nobody] = automation.compute_ai_only_coding_labor(C_aut.ravel()[nobody], ln_E.ravel()[nobody],
                                                                  self.p, A)
        return np.reshape(L_C, L_H.shape)

    def X(self, C_xpm, L_C):
        """Experiment throughput of C_xpm H100e and parallel coding labor L_C (11)."""
        p = self.p
        serial_coding_labor = L_C ** p.parallel_penalty * p.coding_labor_normalization
        return compute_research_effort(C_xpm, serial_coding_labor, p.alpha_experiment_capacity,
                                       p.rho_experiment_capacity, p.experiment_compute_exponent)

    def taste(self, progress):
        """(AI research taste T_AI, aggregate research taste T) (12)."""
        T_AI = compute_ai_research_taste(progress, self.p)
        return T_AI, compute_aggregate_research_taste(T_AI, self.p.taste_distribution)

    def taste_advantage(self, progress):
        """(T_AI, T, z, gain): gain(z) is the taste advantage over the AI of the staff ranked between the AI and z SD,
        on a grid of z. T_AI + gain(inf) is AIFM's T (12)."""
        T_AI, taste = self.taste(progress)
        dist = self.p.taste_distribution
        z = dist.get_sd_of_taste(T_AI) + np.linspace(0.0, 12.0, TASTE_GRID)
        integrand = np.maximum(dist.get_taste_at_sd(z) - T_AI, 0.0) * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        return T_AI, taste, z, cumulative_trapezoid(integrand, z, initial=0.0)

    def staff_taste(self, progress, share):
        """T(E, N_e) (20): the taste of the research when the best `share` = N_e / L_H of the staff keep researching
        and the AI chooses the rest. Share 1 gives AIFM's T, share 0 the AI's."""
        T_AI, taste, z, gain = self.taste_advantage(progress)
        if gain[-1] <= 0:                                       # nobody's taste beats the AI's
            return np.full(np.shape(share), taste)
        cut = -ndtri(np.clip(share, 0.0, 1.0))                  # Phi^-1(1 - share)
        lost = np.interp(cut, z, gain) / gain[-1]               # the people below the cut, ranked above the AI
        return taste - lost * (taste - T_AI)

    def check(self):
        """On the baseline path, with AIFM's own allocation, the methods return AIFM's series. Where AIFM caps the
        research effort (only with extreme parameters, decades into takeoff), research effort and experiment capacity
        are not compared."""
        t = self.times
        C_aut = self.series("inference_H100e", t)
        L_C = self.coding_labor(self.series("L_HUMAN", t), C_aut, self.progress)
        ai_only = self.coding_labor(0.0, C_aut, self.progress)
        X = self.X(self.series("experiment_H100e", t), L_C)
        taste = self.taste(self.progress)[1]
        capped = np.asarray(self.r["research_efforts"]) >= cfg.MAX_RESEARCH_EFFORT
        for ours, key in ((L_C, "coding_labors"), (ai_only, "ai_only_coding_labors"), (X, "experiment_capacity"),
                          (taste, "aggregate_research_taste"), (taste * X, "research_efforts")):
            kept = ~capped if key in ("experiment_capacity", "research_efforts") else slice(None)
            np.testing.assert_allclose(ours[kept], np.asarray(self.r[key])[kept], rtol=1e-12, err_msg=key)

    def parameters(self):
        """AIFM's parameters as this run used them, after calibration, plus the fitted automation logistic. For the
        website's symbol popups."""
        out = {}
        for f in fields(Parameters):
            v = getattr(self.p, f.name)
            if isinstance(v, (bool, str)) or (isinstance(v, (int, float)) and math.isfinite(v)):
                out[f.name] = v if isinstance(v, (bool, str, int)) else float(v)
        automation = self.p.automation_model
        out["automation_logistic_k"] = float(automation.logistic_k)
        out["automation_logistic_x0"] = float(automation.logistic_x0)
        return out


def pause_start(base, start):
    """(t_p, E_p as progress) of a row of ROWS (2)."""
    if isinstance(start, str):
        milestone = base.r["milestones"][start]
        if "time" not in milestone:
            sys.exit(f"milestone {start} is not reached in the baseline run")
        return milestone["time"], milestone["effective_compute_ooms"] + base.ooms_offset
    return start, float(np.interp(start, base.times, base.progress))


def applies(base, model, E_p):
    """Whether a trust model's checks run at E_p: from its row of TRUST_FROM on."""
    return E_p >= pause_start(base, ROWS[TRUST_FROM[model]])[1]


def merged_lab(base, inp, t_p, d, headcount=None):
    """The merged lab's resources d years (array) into a pause that started at t_p. headcount: its staff L_H as a
    function of the date (default STAFF)."""
    assert min(inp["s_aut"], inp["s_ext"]) >= 0 and inp["s_aut"] + inp["s_ext"] < 1, "needs s_aut, s_ext >= 0, sum < 1"
    C = inp["m_C"] * base.series("total_ai_company_H100e", t_p + d)                                # (3)
    C_aut = inp["s_aut"] * C                                                                         # (5)
    C_ext = inp["s_ext"] * inp["m_C"] * base.series("total_ai_company_H100e", t_p)                   # (5)
    L_H = (np.full(np.shape(d), inp["m_L"] * base.series("L_HUMAN", t_p)) if headcount is None       # (3)
           else headcount(t_p + d))
    return {"d": d, "C": C, "C_aut": C_aut, "C_xpm": C - C_ext - C_aut, "L_H": L_H}                 # (4), (5)


def load_headcount(path):
    """L_H(t) from a CSV with columns year, L_H (FTE), log-interpolated."""
    with open(path, newline="") as f:
        rows = sorted((float(r["year"]), float(r["L_H"])) for r in csv.DictReader(f))
    years, L_H = np.array(rows).T

    def headcount(t):
        if np.min(t) < years[0] or np.max(t) > years[-1]:
            sys.exit(f"{path}: covers {years[0]:g}-{years[-1]:g}, the pause needs {np.min(t):.2f}-{np.max(t):.2f}")
        return np.exp(np.interp(t, years, np.log(L_H)))
    return headcount


# Papers (22)-(30). The lab writes its research up as n papers a year. A paper's research grows with its compute c_i as
# c_i^zeta_p, ln c_i is normal with sd sigma_c, and its idea quality is Pareto with tail alpha and mean 1. n is given
# relative to the 2025 field's count n_s.

def papers(X_e, C_e, inp):
    """n / n_s at research X_e and exploration compute C_e (29)."""
    z = inp["zeta_p"]
    k_X, k_C = X_e / inp["RS_safety_2025"], C_e / inp["C_safety_2025"]     # (27), (28)
    return (k_X / (inp["l_inf"] * k_C ** z)) ** (1 / (1 - z))


def per_paper(kappa, p, n, inp):
    """A check's (kappa_h, p_h) or a plan's (kappa_p, p_p) time over t_s = N_s / n_s: kappa, plus p FTE-years of
    prerequisites for every tenfold more papers than the 2025 field's (25), (42)."""
    t_s = inp["L_safety_2025"] / inp["n_s"]
    with np.errstate(divide="ignore"):                                      # n = 0: nothing to study
        return kappa + p / t_s * np.where(n > 0, np.log10(n), 0.0)


def review_work(n, inp):
    """W (26): the verifiers, in FTE, needed to check every paper, m_h of them each."""
    return inp["m_h"] * per_paper(inp["kappa_h"], inp["p_h"], n, inp) * inp["L_safety_2025"] * n


def planning_work(n, inp):
    """W_p (44): the staff, in FTE, needed to plan every round."""
    return per_paper(inp["kappa_p"], inp["p_p"], n, inp) * inp["L_safety_2025"] * n


def growth(kappa, p, n, inp):
    """A check's or a plan's time over its time at the 2025 field's count (25), (42)."""
    return per_paper(kappa, p, n, inp) / kappa if kappa > 0 else np.ones(np.shape(n))


def planned(N_p, inp):
    """F / n_s: the papers N_p FTE of planning make (42). f (kappa_p + c log10 f) = y with y = N_p / N_s, c = p_p / t_s,
    solved with Lambert's W as z = W(a y e^(a kappa_p)) / a, a = ln 10 / c, f = y / z. W(x) is taken as omega(ln x)
    so it can't overflow."""
    t_s = inp["L_safety_2025"] / inp["n_s"]
    k, c, y = inp["kappa_p"], inp["p_p"] / t_s, N_p / inp["L_safety_2025"]
    if c == 0:
        return y / k
    a = math.log(10) / c
    with np.errstate(divide="ignore", invalid="ignore"):                    # N_p = 0: no papers
        return np.where(y > 0, y / (wrightomega(np.log(a * y) + a * k) / a), 0.0)


class TopPapers:
    """Verifiers check the papers worth the most (31)-(33). Called with the share q of the papers checked, it returns
    the shares of the research effort, phi_h, and of the compute, psi, those papers hold. Tabulated once on a grid of
    the bar u and interpolated in ln q; below the grid only idea quality decides, phi_h ~ q^(1 - 1/alpha), psi ~ q."""

    def __init__(self, inp):
        assert inp["alpha"] > 1 and 0 <= inp["zeta_p"] < 1 and inp["sigma_c"] >= 0, \
            "papers need alpha > 1, 0 <= zeta_p < 1 and sigma_c >= 0"
        self.alpha, self.sigma_c, self.sigma_y = inp["alpha"], inp["sigma_c"], inp["zeta_p"] * inp["sigma_c"]
        a, s_c, s_y = self.alpha, self.sigma_c, self.sigma_y
        if s_y == 0:                    # every paper has the same compute factor
            return
        # u from nearly every paper checked to q ~ e^-60: fine where compute decides, coarse in the tail
        mid = s_y * (12 + a * s_y + s_c)
        u = np.concatenate([np.linspace(-12 * s_y, mid, BAR_GRID), np.linspace(mid, mid + 60 / a, BAR_GRID)[1:]])

        def G(x, rate):                 # (33)
            ln_low = -rate * x + rate ** 2 * s_y ** 2 / 2 + log_ndtr(x / s_y - rate * s_y)
            return ndtr(-x / s_y) + np.exp(ln_low)
        q = G(u, a)
        phi = G(u - s_y ** 2, a - 1)                                 # weighted by research effort
        psi = G(u - s_y * s_c, a)                                    # weighted by compute
        keep = q < 1 - 1e-9
        self.ln_q, self.ln_phi, self.ln_psi = (np.log(v[keep])[::-1] for v in (q, phi, psi))
        assert np.all(np.diff(self.ln_q) > 0), "q not monotone in u"

    def __call__(self, q):
        """(phi_h, psi) at 0 <= q <= 1 (32)."""
        a = self.alpha
        if self.sigma_y == 0:
            return q ** (1 - 1 / a), q
        with np.errstate(divide="ignore"):                    # q = 0: nothing is checked
            ln_q = np.log(q)
        below = np.minimum(ln_q - self.ln_q[0], 0.0)
        phi_h = np.exp(np.interp(ln_q, self.ln_q, self.ln_phi) + (1 - 1 / a) * below)
        psi = np.exp(np.interp(ln_q, self.ln_q, self.ln_psi) + below)
        return np.where(q >= 1, 1.0, phi_h), np.where(q >= 1, 1.0, psi)


def check_top_papers(inp):
    """Largest relative error of TopPapers against quadrature over the papers' compute (32)."""
    top = TopPapers(inp)
    if top.sigma_y == 0:
        return 0.0
    a, s_c, s_y = top.alpha, top.sigma_c, top.sigma_y
    t = np.linspace(-12.0, 12.0, 200_001)                 # ln(c / c_bar) = s_c t - s_c^2 / 2, t standard normal
    weight = np.exp(-t * t / 2) / np.exp(-t * t / 2).sum()
    c, ln_y = np.exp(s_c * t - s_c ** 2 / 2), s_y * t - s_y * s_c / 2
    worst = 0.0
    for ln_b in np.linspace(-3 * s_y - s_y * s_c / 2, 14 / a, 9):
        checked = np.minimum(1.0, np.exp(a * (ln_y - ln_b)))
        q = weight @ checked
        phi_h = weight @ (np.exp(ln_y) * np.minimum(1.0, np.exp((a - 1) * (ln_y - ln_b)))) / (weight @ np.exp(ln_y))
        got = top(np.array(q))
        worst = max(worst, abs(got[0] / phi_h - 1), abs(got[1] / (weight @ (c * checked)) - 1))
    return worst


def random_papers(q):
    """(phi_h, psi) when the verifiers check a random share q of the papers (random_checks.py)."""
    return q, q


# Trust models. Each takes the baseline, the merged lab, E_p, the inputs and a staffing, and returns the research effort
# RE that counts at every d, with anything else worth reporting. The docstring's first line is the table title.

def full_trust(base, lab, E_p, inp, staffing=None):
    """All of the lab's research counts, as in AIFM."""
    L_C = base.coding_labor(lab["L_H"], lab["C_aut"], E_p)
    return {"RE": base.taste(E_p)[1] * base.X(lab["C_xpm"], L_C)}     # (13)


def share_grid():
    """Shares of the staff the lab tries, log-spaced from STAFF_FLOOR to 1, as a column."""
    return np.logspace(math.log10(STAFF_FLOOR), 0.0, STAFF_GRID)[:, None]


def unlimited(effort, N_e, need, L_H):
    """Unlimited hiring (36). Of the options (N_e, splits x d) within RESEARCH_TOL of the most research effort, the lab
    takes the one that needs the fewest hires, then the most effort. Returns (the option, N_v, N_h) at every d."""
    near = effort >= (1 - RESEARCH_TOL) * effort.max(axis=0)
    N_h = np.where(near, np.maximum(0.0, N_e + need - L_H), np.inf)
    pick = np.where(N_h == N_h.min(axis=0), effort, -np.inf).argmax(axis=0), np.arange(len(L_H))
    return pick, np.minimum(need[pick], L_H - N_e[pick]), N_h[pick]


def bisect_exploration(C_xpm, rerun_cost, rerun_share):
    """C_e with C_e (1 + rerun_cost * rerun_share(C_e)) = C_xpm, elementwise (34), (47); the left side rises with C_e."""
    lo, hi = C_xpm / (1 + rerun_cost), C_xpm
    for _ in range(40):
        C_e = (lo + hi) / 2
        over = C_e * (1 + rerun_cost * rerun_share(C_e)) > C_xpm
        lo, hi = np.where(over, lo, C_e), np.where(over, C_e, hi)
    return (lo + hi) / 2


def verified(base, lab, E_p, inp, staffing, staff=None, hired=0.0, ranked=True):
    """Only research that humans reviewed and reproduced counts.

    N_e of the staff, those with the best taste, keep researching; the rest check the papers worth the most, and part
    of the experiment compute re-runs them. At every d the lab takes the N_e that gives the most checked research.
    With unlimited hiring it checks everything and hires the verifiers it lacks. Before TRUST_FROM it is full trust.
    staff: the N_e to choose from (default the grid; zeros give N_e -> 0). hired: extra verifiers with own staff.
    ranked: False, the verifiers check a random sample.
    """
    if not applies(base, "verified", E_p):                                          # (38)
        return full_trust(base, lab, E_p, inp)
    L_H = lab["L_H"]
    N_e = L_H * share_grid() if staff is None else staff                             # (18)
    taste, T_Ep = base.staff_taste(E_p, N_e / L_H), base.taste(E_p)[1]               # T(E_p, N_e) (20), T(E_p)
    L_C = base.coding_labor(N_e, lab["C_aut"], E_p)                                  # L_C(N_e, C_aut, E_p) (19)
    rerun = inp["kappa_c"] / inp["M"]         # a re-run repeats a paper's own runs, not its failed projects (34)
    if staffing == "unlimited hiring":
        C_e = lab["C_xpm"] / (1 + rerun)                                             # (37)
        X_e = base.X(C_e, L_C)                                                       # (19)
        n = papers(X_e, C_e, inp)                                                    # (29)
        pick, N_v, N_h = unlimited(taste * X_e, N_e, review_work(n, inp), L_H)       # (36)
        return {"RE": taste[pick] * X_e[pick], "hired": N_h,                         # (38)
                "N_e / L_H": N_e[pick] / L_H, "N_v / L_H": N_v / L_H, "hired / L_H": N_h / L_H, "n / n_s": n[pick],
                "tau / kappa_h t_s": growth(inp["kappa_h"], inp["p_h"], n[pick], inp),
                "C_r / C_xpm": np.full(len(L_H), 1 - 1 / (1 + rerun)), "taste / T(E_p)": taste[pick] / T_Ep}
    top = TopPapers(inp) if ranked else random_papers
    N_v = L_H - N_e                                                                  # (18)

    def checked(C_e):
        """(X_e, n / n_s, q) at exploration compute C_e."""
        X_e = base.X(C_e, L_C)                                                       # (19)
        n = papers(X_e, C_e, inp)                                                    # (29)
        W = review_work(n, inp)                                                      # (26)
        with np.errstate(divide="ignore", invalid="ignore"):                         # W = 0: nothing to check
            return X_e, n, np.where(N_v + hired >= W, 1.0, (N_v + hired) / W)        # (30)

    C_e = bisect_exploration(np.broadcast_to(lab["C_xpm"], N_e.shape), rerun,       # C_r = rerun psi C_e (34)
                             lambda C_e: top(checked(C_e)[2])[1])
    X_e, n, q = checked(C_e)
    phi_h, psi = top(q)
    effort = taste * phi_h * X_e                                                     # RE^v (21), (38)
    pick = effort.argmax(axis=0), np.arange(len(L_H))                                # (35)
    return {"RE": effort[pick],                                                      # (38)
            "N_e / L_H": N_e[pick] / L_H, "N_v / L_H": N_v[pick] / L_H, "n / n_s": n[pick],
            "tau / kappa_h t_s": growth(inp["kappa_h"], inp["p_h"], n[pick], inp), "q": q[pick],
            "phi_h": phi_h[pick], "psi": psi[pick], "C_r / C_xpm": 1 - C_e[pick] / lab["C_xpm"],
            "taste / T(E_p)": taste[pick] / T_Ep}


def compartmentalized(base, lab, E_p, inp, staffing, E_a=None, hired=0.0, approval=False, ranked=True):
    """Only research that humans planned, reviewed and reproduced counts.

    The frontier model only writes plans, and an engineer at E_a (default ENGINEER's E) codes them alone. The staff
    only plan and check: N_p plan rounds, N_v check the planned papers worth the most. At every d the lab takes the
    split that gives the most checked research. Plans carry the share w of the frontier model's taste advantage over
    the engineer. With unlimited hiring it plans and checks everything and hires the staff it lacks. Before TRUST_FROM
    it is the verified model. hired: extra staff with own staff. approval: only code the auditors approve runs
    (approved()). ranked: False, the verifiers check a random sample, and the lab also tries planning only what they
    can check, F = V.
    """
    if not applies(base, "compartmentalized", E_p):                                 # (51)
        return verified(base, lab, E_p, inp, staffing, hired=hired, ranked=ranked)
    E_a = pause_start(base, ROWS[ENGINEER])[1] if E_a is None else E_a               # E_AC (40)
    assert base.p.automation_model.get_automation_fraction(E_a) >= 1 - 1e-9, \
        "the engineer codes alone, so it must do every task: E_a >= E_AC"
    L_H = lab["L_H"]
    T_w = (1 - inp["w"]) * base.taste(E_a)[0] + inp["w"] * base.taste(E_p)[0]       # (41)
    L_C = base.coding_labor(0.0, lab["C_aut"], E_a)                                  # L_C(0, C_aut, E_a) (40)
    approved_share = {}
    if approval:                                                                     # L_C^a (52)
        approvable = inp["N_a"] / inp["kappa_a"] if inp["kappa_a"] > 0 else math.inf
        approved_share = {"L_C^a / L_C": np.minimum(1.0, approvable / L_C)}
        L_C = np.minimum(L_C, approvable)
    nobody, taste = np.zeros(len(L_H)), np.full(len(L_H), T_w / base.taste(E_p)[1])  # N_e / L_H, T_w / T(E_p)
    rerun = inp["kappa_c"] / inp["M"]                                                # (47)
    if staffing == "unlimited hiring":
        C_e = lab["C_xpm"] / (1 + rerun)                                             # (50)
        X_e = base.X(C_e, L_C)                                                       # (40), (53)
        n = papers(X_e, C_e, inp)
        W, W_p = review_work(n, inp), planning_work(n, inp)                          # (26), (44)
        need = W_p + W
        N_h = np.maximum(0.0, need - L_H)                                            # (49)
        busy = np.minimum(need, L_H) / L_H                                           # N_p + N_v (50)
        with np.errstate(divide="ignore", invalid="ignore"):                         # nothing to plan or check
            plan = np.where(need > 0, W_p / need, 0.0)
        return {"RE": T_w * X_e, "hired": N_h,                                       # (51), (54)
                "N_e / L_H": nobody, "N_v / L_H": (1 - plan) * busy, "hired / L_H": N_h / L_H, "n / n_s": n,
                "N_p / L_H": plan * busy, "tau / kappa_h t_s": growth(inp["kappa_h"], inp["p_h"], n, inp),
                "m_p tau_p / kappa_p t_s": growth(inp["kappa_p"], inp["p_p"], n, inp),
                "C_r / C_xpm": np.full(len(L_H), 1 - 1 / (1 + rerun)), "taste / T(E_p)": taste, **approved_share}
    top = TopPapers(inp) if ranked else random_papers
    share = share_grid()                                                             # N_p / (L_H + hired)
    people = L_H + hired
    if not ranked:    # plan only what the verifiers check, F = V: F (m_p tau_p + m_h tau) = N_p + N_v (42)
        F = planned(people, {**inp, "kappa_p": inp["kappa_p"] + inp["m_h"] * inp["kappa_h"],
                             "p_p": inp["p_p"] + inp["m_h"] * inp["p_h"]})
        share = np.vstack([np.broadcast_to(share, (STAFF_GRID, len(L_H))), planning_work(F, inp) / people])

    def checked(C_e):
        """(X_e, n / n_s, N_p, q_p, q, F / n_s, W_F) at exploration compute C_e: q_p and q the shares of the papers
        planned and checked, W_F the verifiers needed to check all F planned papers."""
        X_e = base.X(C_e, L_C)                                                       # (40), (53)
        n = papers(X_e, C_e, inp)
        W_p = planning_work(n, inp)                                                  # (44)
        N_p = np.minimum(W_p, share * people)                                        # (45)
        N_v = people - N_p                                                           # (39)
        with np.errstate(divide="ignore", invalid="ignore"):   # W_p = 0: every round is planned; W_F = 0: all checked
            F = np.where(N_p >= W_p, n, planned(N_p, inp))                           # (42), (45)
            q_p, W_F = F / n, review_work(F, inp)                                    # (45)
            return X_e, n, N_p, q_p, np.where(N_v >= W_F, q_p, q_p * N_v / W_F), F, W_F

    def rerun_share(C_e):                                       # C_r = rerun q_p psi(q / q_p) C_e (47)
        q_p, q = checked(C_e)[3:5]
        return q_p * top(q / q_p)[1]

    C_e = bisect_exploration(np.broadcast_to(lab["C_xpm"], (len(share), len(L_H))), rerun, rerun_share)
    X_e, n, N_p, q_p, q, F, W_F = checked(C_e)
    phi_h, psi = top(q / q_p)                                                        # (46), (47)
    effort = T_w * q_p * phi_h * X_e                                                 # RE^v (46)
    pick = effort.argmax(axis=0), np.arange(len(L_H))                                # (48)
    N_v = np.minimum(people - N_p, W_F)                                              # time spent checking (42)
    return {"RE": effort[pick],                                                      # (51), (54)
            "N_e / L_H": nobody, "N_v / L_H": N_v[pick] / L_H, "n / n_s": n[pick], "N_p / L_H": N_p[pick] / L_H,
            "tau / kappa_h t_s": growth(inp["kappa_h"], inp["p_h"], F[pick], inp),
            "m_p tau_p / kappa_p t_s": growth(inp["kappa_p"], inp["p_p"], F[pick], inp),
            "q_p": q_p[pick], "q": q[pick], "phi_h": phi_h[pick], "psi": psi[pick],
            "C_r / C_xpm": 1 - C_e[pick] / lab["C_xpm"], "taste / T(E_p)": taste, **approved_share}


def approved(base, lab, E_p, inp, staffing, hired=0.0):
    """Only research that humans planned, approved, reviewed and reproduced counts.

    The compartmentalized model, with N_a auditors outside the staff approving the engineer's code before it runs, so
    the coding labor that runs is min(L_C, N_a / kappa_a) (52). Before TRUST_FROM it is the compartmentalized model
    (54).
    """
    return compartmentalized(base, lab, E_p, inp, staffing, hired=hired, approval=applies(base, "approved", E_p))


TRUST_MODELS = {"full trust": full_trust, "verified": verified, "compartmentalized": compartmentalized,
                "approved": approved}
SIMPLER = {"compartmentalized": "verified", "approved": "compartmentalized"}   # what a model is before TRUST_FROM
TABLES = {("full trust", None): "1", **{(m, s): f"{i}{'ab'[j]}" for i, m in enumerate(TRUST_FROM, 2)
                                         for j, s in enumerate(STAFFINGS)}}   # (trust model, staffing): table number


def rs_safety(base, C_s, N_s, t=T_SAFETY):
    """RS_safety_2025 (16): the research one lab with C_s H100e and N_s people does in a year, X with T = 1, its AI
    coders with the leading lab's compute per engineer and capability at t."""
    C_aut = N_s * base.series("inference_H100e", t) / base.series("L_HUMAN", t)
    L_C = base.coding_labor(N_s, C_aut, float(np.interp(t, base.times, base.progress)))
    return float(base.X(C_s, L_C))


def dataset_papers():
    """{year: papers} of PAPERS_DATASET."""
    with open(PAPERS_DATASET, newline="") as f:
        years = [int(r["year"]) for r in csv.DictReader(f)]
    return {y: years.count(y) for y in sorted(set(years))}


def check_inputs(inp):
    """Fails on inputs the model can't take."""
    assert inp["l_inf"] >= 1, "papers need l_inf >= 1: labor can't make a paper worse"
    assert inp["M"] >= 1, "M >= 1: all projects take at least the compute of those that became papers"
    assert min(inp["p_h"], inp["p_p"]) >= 0, "needs p_h, p_p >= 0"


def load_inputs():
    """The inputs of pause_inputs.csv, {name: value}."""
    with open(ROOT / "pause_inputs.csv", newline="") as f:
        inp = {row["name"]: float(row["value"]) for row in csv.DictReader(f)}
    check_inputs(inp)
    return inp


def solve_row(base, inp, name, start, d, headcount):
    """A row of the tables: every trust model and staffing in a pause from `start`, at the dates d."""
    t_p, E_p = pause_start(base, start)
    lab = merged_lab(base, inp, t_p, d, headcount)
    out = {}
    for m, s in TABLES:
        if m in SIMPLER and not applies(base, m, E_p):
            out[m, s] = out[SIMPLER[m], s]                    # the same model, solved once
        else:
            out[m, s] = TRUST_MODELS[m](base, lab, E_p, inp, s)
    T_AI, taste = base.taste(E_p)
    taste_sd = base.p.taste_distribution.get_sd_of_taste(T_AI)
    year, month = divmod(math.floor(t_p * 12), 12)            # the month t_p falls in
    return {"name": name, "label": f"{name} ({month + 1:02d}.{year})", "month": f"{year}-{month + 1:02d}",
            "t_p": t_p, "E_p": E_p, "out": out,
            "cells": {k: np.interp(DURATIONS, d, cumulative_trapezoid(o["RE"], d, initial=0.0)) / inp["RS_safety_2025"]
                      for k, o in out.items()},                                                        # (17)
            "hired": {k: np.maximum.accumulate(o.get("hired", 0.0 * d)) for k, o in out.items()},   # the most so far
            "ooms": E_p - base.ooms_offset, "taste": taste, "taste_sd": taste_sd, "above_ai": ndtr(-taste_sd),
            "A": base.p.automation_model.get_automation_fraction(E_p),
            "C": lab["C"][0], "L_H": lab["L_H"][0], "world_H100e": base.series("world_H100e", t_p), "staff": lab["L_H"]}


def check_row(base, inp, r, d, headcount):
    """Limits of a row, on every twelfth date: with its checks off the verified model is full trust; with free planning,
    w = 1 and the engineer at E_p the compartmentalized model is the verified one with N_e -> 0; with free approval the
    approved model is the compartmentalized one; the staff's taste advantage adds up to AIFM's T. Returns the largest
    deviations of the first and the last. (AIFM tabulates L_C, which is monotone in staff only to ~1e-3 near AC.)"""
    E_p = r["E_p"]
    lab = merged_lab(base, inp, r["t_p"], d[::12], headcount)
    full = full_trust(base, lab, E_p, inp)["RE"]
    checks_off = {**inp, "kappa_h": 0.0, "p_h": 0.0, "kappa_c": 0.0}
    nobody = np.zeros((1, len(lab["d"]))) if applies(base, "compartmentalized", E_p) else None
    checks_off_dev = 0.0
    for s in STAFFINGS:
        dev = verified(base, lab, E_p, checks_off, s)["RE"] / full - 1
        assert np.all((dev > -1e-12) & (dev < 1e-3)), f"{r['name']}, {s}: verified with checks off is not full trust"
        checks_off_dev = max(checks_off_dev, np.abs(dev).max())
        plain = compartmentalized(base, lab, E_p, {**inp, "kappa_p": 0.0, "p_p": 0.0, "w": 1.0}, s, E_a=E_p)["RE"]
        np.testing.assert_allclose(plain, verified(base, lab, E_p, inp, s, staff=nobody)["RE"], rtol=1e-12)
        free = approved(base, lab, E_p, {**inp, "kappa_a": 0.0}, s)["RE"]
        np.testing.assert_allclose(free, compartmentalized(base, lab, E_p, inp, s)["RE"], rtol=1e-12)
    T_AI, taste, _, gain = base.taste_advantage(E_p)
    taste_dev = abs((T_AI + gain[-1]) / taste - 1)
    assert taste_dev < 0.01, f"{r['name']}: the staff's taste advantage over the AI doesn't add up to AIFM's T"
    return checks_off_dev, taste_dev


def hiring_curves(base, inp, rows, d, headcount):
    """For the website: the cell with own staff plus at most H hired people, for H from 0 to what unlimited hiring hires
    in CURVE_STEPS steps, wherever unlimited hiring beats own staff by more than RESEARCH_TOL. The gain over own staff
    is integrated a month at a time, so H = 0 gives the table's cell."""
    month = STEPS_PER_YEAR // 12
    curves = []
    for m in TRUST_FROM:
        for r in rows:
            if not applies(base, m, r["E_p"]):
                continue
            for i, D in enumerate(DURATIONS):
                own, unlimited = (r["cells"][m, s][i] for s in STAFFINGS)
                most = float(np.interp(D, d, r["hired"][m, STAFFINGS[1]]))
                if most <= 0 or unlimited <= (1 + RESEARCH_TOL) * own:
                    continue
                dm = d[::month][:round(D * 12) + 1]                    # a date a month up to D
                lab = merged_lab(base, inp, r["t_p"], dm, headcount)
                solve = lambda h: TRUST_MODELS[m](base, lab, r["E_p"], inp, STAFFINGS[0], hired=h)["RE"]
                RE_own = solve(0.0)
                H = most * np.arange(CURVE_STEPS + 1) / CURVE_STEPS
                gain = [0.0] + [np.trapezoid(solve(h) - RE_own, dm) for h in H[1:]]
                cells = own + np.array(gain) / inp["RS_safety_2025"]
                assert np.all(np.diff(cells) >= -1e-9 * own), f"{m}, {r['name']}, {D:g} years: hiring loses research"
                curves.append({"model": m, "row": r["name"], "years": D, "hired": H.tolist(), "cells": cells.tolist()})
    return curves


def auditor_curves(base, inp, rows, d, headcount):
    """For the website: the approved model's cells with N_a of AUDITORS and pause_inputs.csv's, both staffings."""
    N_a = sorted({*AUDITORS, inp["N_a"]})
    curves = []
    for r in rows:
        if not applies(base, "approved", r["E_p"]):
            continue
        lab = merged_lab(base, inp, r["t_p"], d, headcount)
        cells = {}
        for s in STAFFINGS:
            stock = [cumulative_trapezoid(approved(base, lab, r["E_p"], {**inp, "N_a": n}, s)["RE"], d, initial=0.0)
                     for n in N_a]
            cells[s] = np.array([np.interp(DURATIONS, d, v) for v in stock]) / inp["RS_safety_2025"]    # (17)
            np.testing.assert_allclose(cells[s][N_a.index(inp["N_a"])], r["cells"]["approved", s], rtol=1e-12)
            assert np.all(np.diff(cells[s], axis=0) >= -1e-9 * cells[s][1:]), \
                f"{r['name']}, {s}: more auditors lose research"
        curves.append({"name": r["name"], "cells": cells[STAFFINGS[0]].tolist(),
                       "unlimited": cells[STAFFINGS[1]].tolist()})
    return {"N_a": [float(n) for n in N_a], "rows": curves}


def pile_sample(base, inp, headcount, row=PILE_ROW, t=PILE_T, size=PILE_SIZE):
    """For the website: `size` random papers the lab writes at t in a pause from `row`, verified model, own staff (24),
    (30). Compute in H100e-years, research effort in units of the 2025 field's average paper."""
    t_p, E_p = pause_start(base, ROWS[row])
    lab = merged_lab(base, inp, t_p, np.array([t - t_p]), headcount)
    out = verified(base, lab, E_p, inp, STAFFINGS[0])
    z, s_c, a, M = inp["zeta_p"], inp["sigma_c"], inp["alpha"], inp["M"]
    n_s = inp["n_s"]
    unit = inp["RS_safety_2025"] / n_s                                   # the 2025 field's average paper
    T = out["taste / T(E_p)"][0] * base.taste(E_p)[1]                    # T(E_p, N_e) (20)
    C_e = lab["C_xpm"][0] * (1 - out["C_r / C_xpm"][0])                  # (18)
    c_bar, c_bar_s = C_e / (M * out["n / n_s"][0] * n_s), inp["C_safety_2025"] / (M * n_s)   # (22)
    r_s = unit / math.exp(-z * (1 - z) * s_c ** 2 / 2)                   # (27b), (27c)
    rng = np.random.default_rng(math.floor(t))
    v = (1 - 1 / a) * (1 - rng.random(size)) ** (-1 / a)                  # Pareto, floor 1 - 1 / alpha (30)
    c = c_bar * np.exp(s_c * rng.standard_normal(size) - s_c ** 2 / 2)    # (24)
    RE = T * v * r_s * inp["l_inf"] * (c / c_bar_s) ** z                  # (24), (30)
    sig = lambda x: float(f"{x:.4g}")
    return {"row": row, "t": t, "c_bar": sig(c_bar), "RE_bar": sig(T * inp["l_inf"] * (c_bar / c_bar_s) ** z),
            "papers": [{"v": sig(v[i]), "c": sig(c[i]), "RE": sig(RE[i] / unit)} for i in range(size)]}


def fmt(v):
    """Three significant digits, thousands separators."""
    if v == 0:
        return "0"
    digits = math.floor(math.log10(abs(v)))
    return f"{round(v, 2 - digits):,.{max(0, 2 - digits)}f}"


def print_tables(command, preset, staff, inp, rows, d):
    """The tables as markdown."""
    RS_s = inp["RS_safety_2025"]
    print(f"# Pause tables ({preset})\n\n"
          f"Output of `{command}`. Staff L_H: {staff}.\n"
          f"A cell is int RE dt / RS_safety_2025 = {RS_s:g}: years of the world's 2025 technical safety research.\n"
          "Inputs (`pause_inputs.csv`): " + ", ".join(f"{k} = {v:g}" for k, v in inp.items()) + ".\n"
          "Rows before a trust model's start (" + ", ".join(f"{m} {r}" for m, r in TRUST_FROM.items())
          + ") are the simpler model's.")
    header = "| pause starts at | " + " | ".join(f"{D * 12:g} months" if D < 1 else f"{D:g} years" for D in DURATIONS)
    rule = " |\n|---|" + "---:|" * len(DURATIONS)
    for (m, s), number in TABLES.items():
        title = TRUST_MODELS[m].__doc__.splitlines()[0]
        print(f"\n## Table {number} ({m}{', ' + s if s else ''}). {title}\n\n{header}{rule}")
        for r in rows:
            print(f"| {r['label']} | " + " | ".join(fmt(v) for v in r["cells"][m, s]) + " |")
        if s == "unlimited hiring":
            print(f"\n## Table {number[0]}c ({m}, unlimited hiring): people hired for Table {number}, thousand FTE\n\n"
                  "The most the lab has hired at any time of the pause, on top of its own staff.\n\n"
                  f"{header}{rule}")
            for r in rows:
                print(f"| {r['label']} | " + " | ".join(fmt(v / 1e3) for v in np.interp(DURATIONS, d, r["hired"][m, s]))
                      + " |")
    ratio = lambda v: f"{v:.2f}" if not abs(v) >= 10 else fmt(v)    # three significant digits from 10 on
    for (m, s), number in TABLES.items():
        # the last row has every key; earlier rows may be a simpler model's, whose missing keys print as -
        keys = list(dict.fromkeys(k for r in reversed(rows) for k in r["out"][m, s] if k not in ("RE", "hired")))
        if not keys:
            continue
        print(f"\n## Table {number} ({m}{', ' + s if s else ''}): the lab at the start of the pause -> "
              f"{DURATIONS[-1]:g} years in\n\n"
              "| row | " + " | ".join(keys) + " | RE / RE of full trust |\n|---|" + "---:|" * (len(keys) + 1))
        for r in rows:
            o = r["out"][m, s]
            cols = [o.get(k) for k in keys] + [o["RE"] / r["out"]["full trust", None]["RE"]]
            print(f"| {r['name']} | " + " | ".join("-" if c is None else f"{ratio(c[0])} -> {ratio(c[-1])}"
                                                   for c in cols) + " |")
    years = np.arange(DURATIONS[-1] + 1)
    for m in TRUST_FROM:
        print(f"\n## Table {TABLES[m, 'unlimited hiring']} ({m}, unlimited hiring): headcount in year t of the pause, "
              "thousand FTE\n\nStaff L_H + hired (in brackets: verifiers and planners, from staff and hired).\n\n"
              "| row | " + " | ".join(f"{t:g}" for t in years) + " |\n|---|" + "---:|" * len(years))
        for r in rows:
            o = r["out"][m, "unlimited hiring"]
            L_H, N_h = r["staff"], o.get("hired", 0.0 * d)
            checking = (o.get("N_v / L_H", 0.0 * d) + o.get("N_p / L_H", 0.0 * d)) * L_H + N_h
            print(f"| {r['name']} | " + " | ".join(f"{fmt(a / 1e3)} ({fmt(b / 1e3)})" for a, b in
                                                   zip(np.interp(years, d, L_H + N_h),
                                                       np.interp(years, d, checking))) + " |")
    print("\n## Rows: what is frozen, and the merged lab at the start of the pause\n\n"
          "| row | t_pause | log10 E_pause (AIFM OOMs) | A | AI taste, human SDs | staff with taste above the AI's | T "
          f"| C, H100e | share of world compute | L_H, FTE | RE(t_pause) / {RS_s:g}, full trust |\n|---|"
          + "---:|" * 10)
    for r in rows:
        print(f"| {r['name']} | {r['t_p']:.2f} | {r['ooms']:.2f} | {r['A']:.2f} | {r['taste_sd']:.2f} "
              f"| {r['above_ai']:.1%} | {fmt(r['taste'])} | {r['C'] / 1e6:.1f}M | {r['C'] / r['world_H100e']:.0%} "
              f"| {fmt(r['L_H'])} | {fmt(r['out']['full trust', None]['RE'][0] / RS_s)} |")


def write_json(path, command, preset, staff, base, inp, rows, d, headcount):
    """The tables and the website's plots, unrounded, for site/content/table.json."""
    default, hiring = STAFFINGS
    month = STEPS_PER_YEAR // 12
    table = {"command": command, "preset": preset, "headcount": staff,
             "main": SITE_TABLE, "staffings": list(STAFFINGS),
             "unit": "years of the world's 2025 technical safety research: int_0^D RE dt / RS_safety_2025",
             "hired_unit": "people hired on top of the lab's own: the most at any time of the pause (FTE)",
             "inputs": inp, "aifm_parameters": base.parameters(), "durations_years": DURATIONS,
             "trust_from": dict(TRUST_FROM),
             "rows": [{"name": r["name"], "t_pause": float(r["t_p"]), "month": r["month"],
                       "cells": {m: r["cells"][m, None if m == "full trust" else default].tolist()
                                 for m in TRUST_MODELS},
                       "unlimited": {m: r["cells"][m, hiring].tolist() for m in TRUST_FROM},
                       "hired": {m: np.interp(DURATIONS, d, r["hired"][m, hiring]).tolist() for m in TRUST_FROM}}
                      for r in rows]}
    counts = dataset_papers()
    table["papers"] = {                                       # the paper count n(t) (29), a point a month
        "model": PAPER_COUNT, "staffing": default, "n_s": counts[2025],
        "dataset": [{"year": y, "n": c} for y, c in counts.items()],
        "rows": [{"name": r["name"], "t": (r["t_p"] + d[::month]).tolist(),
                  "n / n_s": r["out"][PAPER_COUNT, default]["n / n_s"][::month].tolist()}
                 for r in rows if "n / n_s" in r["out"][PAPER_COUNT, default]]}
    table["hiring"] = {                                       # the people hired (36), (49), a point a month
        "staffing": hiring,
        "unit": "people hired on top of the lab's own staff L_H by date t: the most it has needed so far (FTE)",
        "rows": [{"name": r["name"], "t": (r["t_p"] + d[::month]).tolist(), "L_H": r["staff"][::month].tolist(),
                  "hired": {m: r["hired"][m, hiring][::month].tolist()
                            for m in TRUST_FROM if applies(base, m, r["E_p"])}}
                 for r in rows if any(applies(base, m, r["E_p"]) for m in TRUST_FROM)]}
    table["hiring_curves"] = {
        "unit": "the cell of a pause D years long (years of the world's 2025 technical safety research) when the "
                "lab may hire at most `hired` people on top of its own staff",
        "scenarios": hiring_curves(base, inp, rows, d, headcount)}
    table["auditor_curves"] = {
        "unit": "the approved model's cells (years of the world's 2025 technical safety research) with N_a[i] "
                "auditors, a cell per duration: rows[].cells[i] with own staff, rows[].unlimited[i] with unlimited "
                "hiring",
        **auditor_curves(base, inp, rows, d, headcount)}
    table["pile"] = pile_sample(base, inp, headcount)
    Path(path).write_text(json.dumps(table, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default=PRESETS[0], choices=PRESETS, help="baseline parameters (default: %(default)s)")
    ap.add_argument("--headcount", metavar="CSV", help="the merged lab's staff L_H: columns year, L_H (FTE), "
                    f"log-interpolated (default: {STAFF})")
    ap.add_argument("--json", metavar="FILE", help="also write the tables to FILE as JSON, for the website: "
                    "site/content/table.json")
    args = ap.parse_args()
    headcount = load_headcount(args.headcount) if args.headcount else None
    staff = args.headcount or STAFF
    command = " ".join(["python pause.py"] + sys.argv[1:])

    inp = load_inputs()
    n_s = dataset_papers()[2025]
    if inp["n_s"] != n_s:
        sys.exit(f"pause_inputs.csv: n_s = {inp['n_s']:g} is stale, {PAPERS_DATASET.name} has {n_s} papers of 2025")
    base = Baseline(args.preset)
    base.check()
    RS_s = rs_safety(base, inp["C_safety_2025"], inp["L_safety_2025"])
    if abs(RS_s / inp["RS_safety_2025"] - 1) > 0.01:
        sys.exit(f"pause_inputs.csv: RS_safety_2025 = {inp['RS_safety_2025']:g} is stale, "
                 f"C_safety_2025 and L_safety_2025 give {RS_s:.4g}")
    shares_err = check_top_papers(inp)
    assert shares_err < 1e-5, f"TopPapers is off quadrature by {shares_err:.1e}"

    d = np.linspace(0.0, DURATIONS[-1], round(DURATIONS[-1] * STEPS_PER_YEAR) + 1)   # years into the pause
    rows = [solve_row(base, inp, name, start, d, headcount) for name, start in ROWS.items()]
    checks_off_dev, taste_dev = np.max([check_row(base, inp, r, d, headcount) for r in rows], axis=0)

    print_tables(command, args.preset, staff, inp, rows, d)
    print(f"\nChecks passed: AIFM reproduced to 1e-12; verified with its checks off = full trust to "
          f"{max(checks_off_dev, 1e-16):.0e}; compartmentalized limits to 1e-12; staff taste = AIFM's T to "
          f"{taste_dev:.0e}; TopPapers = quadrature to {max(shares_err, 1e-16):.0e}; RS_safety_2025 recomputed "
          f"{RS_s:.1f}.")
    if args.json:
        write_json(args.json, command, args.preset, staff, base, inp, rows, d, headcount)


if __name__ == "__main__":
    main()
