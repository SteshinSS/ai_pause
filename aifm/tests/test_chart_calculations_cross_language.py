"""
Cross-language golden test for utils/chartCalculations.ts.

utils/chartCalculations.ts re-implements, in TypeScript, closed-form algebra that
also exists in Python. It powers the slider-hover preview charts
(components/SliderPopover.tsx), so it never integrates the ODE — it evaluates the
algebra on a synthetic effective-compute axis, with constants that its own header
says are "kept in step with Python progress_model/config.py by hand". Nothing else
in the repo detects the two languages drifting apart.

The golden fixture is produced by running the REAL exported frontend functions
(see tests/gen_chart_calculations_golden.ts). Here we feed the same inputs to the
real Python functions and compare values. Neither side reimplements the other.

Pairs pinned here
-----------------
1. computeHorizonFromProgress
     <-> progress_model.ac_anchor._make_decaying_horizon_observable(...).predict
2. generateSCHorizonData's automationFraction channel (getAutomationFraction)
     <-> progress_model.progress_rate.compute_automation_fraction, i.e.
         AutomationModel.get_automation_fraction on a linear schedule
3. generateSCHorizonData's progressAtSc (calculateProgressAtSC)
     <-> the `invert` half of the same decaying-doubling-time observable

Pair deliberately NOT pinned: TasteDistribution
-----------------------------------------------
The TS TasteDistribution (chartCalculations.ts ~line 182) is a plain log-normal:
sigma = ln(median_to_top_gap) / z_p, mu = ln(baseline) - sigma^2/2, and
getTasteAtSD(n) = exp(mu + n*sigma). Python's TasteDistribution
(progress_model/taste_distribution.py) applies a bounded transform
T = (A^rho + (1 - A^rho) * exp(v*x)^rho)^(1/rho) with A = median_to_top_gap ** (1 +
taste_limit_m2b), and solves for (mu, sigma) by least squares against a ratio
constraint and a mean constraint. The TS formulas are exactly Python's *initial
guess* (mu_init/sigma_init), not its fitted answer, and the TS transform is the
identity rather than the bounded one. They are different models, not two
implementations of one model, so there is no tolerance at which an equality
assertion would be meaningful. See the accompanying report for the measured gap.

Known drift, documented rather than pinned
-----------------------------------------
Two things the pinned pairs above cannot cover, found while writing this test:

  * Schedule form. getAutomationFraction only implements the LINEAR schedule and
    ignores automation_interp_type, but every shipping preset in
    constants/parameters.ts sets automation_interp_type: 'logistic', and so does
    progress_model/config.py. So the test below pins TS against Python's linear
    branch (a genuine, exact counterpart), while the live preview chart is
    drawing a different functional form from the one the model integrates.
  * T_0 units. TS passes uiParameters.present_doubling_time straight in as T_0,
    whereas progress_model/ac_anchor.py converts it first
    (doubling_time_in_progress_units = present_doubling_time * anchor_rate).
    The cases here therefore feed BOTH sides the same numeric T_0: the algebra is
    what is being pinned, not the unit conversion in front of it.

Known branch divergences, documented rather than pinned
------------------------------------------------------
Both horizon implementations guard their parameters, but disagree on the fallback:
  * A_0 == 0: TS returns H_0 * 2 ** (progress / T_0) (a pure exponential, taken
    BEFORE its validation block); Python treats A_0 == 0 as invalid and returns 1e12.
  * A_0 >= 1, T_0 <= 0, H_0 <= 0: TS returns H_0; Python returns 1e12.
  * non-finite/non-positive result: TS returns H_0; Python returns 1e12.
The fixture stays inside the shared main branch so that a real algebra change
fails this test, instead of the test freezing fallback behavior that neither side
intends to be load-bearing.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import assert_arrays_close, assert_scalar_close
from progress_model import Parameters
from progress_model.ac_anchor import _make_decaying_horizon_observable
from progress_model.automation_model import AutomationModel
from progress_model.progress_rate import compute_automation_fraction

GOLDEN_PATH = Path(__file__).parent / "golden_data" / "chart_calculations_golden.json"

# Tolerances.
#
# The automation fraction is +, -, *, / and a clip only. IEEE-754 makes those
# exact and the two implementations apply them in the same order, so the two
# languages must agree bit-for-bit: rtol=0.
AUTOMATION_RTOL = 0.0
# The horizon and its inverse additionally use log() and pow(). Those are libm
# routines rather than IEEE-754-mandated operations, so V8 and numpy can differ
# by a fraction of an ULP, and pow() compounds that through an exponent of
# ln(2)/ln(1-A_0) (up to ~-13.5 here). 1e-12 is ~4 orders of magnitude tighter
# than any parameter-level drift this test exists to catch (the smallest
# meaningful constant change would move a horizon by >1e-6 relative), while
# leaving room for libm differences across platforms.
HORIZON_RTOL = 1e-12


def _load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


# Loaded at import time so the cases can name the parametrization ids.
_GOLDEN = _load_golden()


@pytest.mark.parametrize(
    "case", _GOLDEN["horizon_cases"], ids=[c["name"] for c in _GOLDEN["horizon_cases"]]
)
def test_horizon_from_progress_matches_frontend(case):
    """Python decaying-doubling-time predict == frontend computeHorizonFromProgress."""
    # Pin the A_0 derivation too: both sides compute it as 1 - growth factor.
    a0 = 1.0 - case["doubling_difficulty_growth_factor"]
    assert a0 == case["A_0"], (
        f"[{case['name']}] A_0 derivation differs: Python {a0!r} vs frontend {case['A_0']!r}"
    )

    observable = _make_decaying_horizon_observable(
        H_0=case["H_0"],
        A_0=a0,
        T_0=case["T_0"],
        anchor_progress=case["anchor_progress"],
    )
    progress = np.array(case["progress"], dtype=float)
    actual = observable.predict(progress)

    # Guard against silently comparing against Python's 1e12 fallback: the
    # fixture is built to stay on the main analytic branch.
    assert not np.any(actual == 1e12), (
        f"[{case['name']}] Python took its 1e12 fallback branch; this case is "
        f"outside the shared analytic domain and should not be in the fixture"
    )

    assert_arrays_close(
        actual,
        case["horizon"],
        rtol=HORIZON_RTOL,
        msg=f"[{case['name']}] horizon(progress) mismatch vs frontend "
        f"computeHorizonFromProgress (H_0={case['H_0']}, A_0={a0}, T_0={case['T_0']}, "
        f"anchor={case['anchor_progress']})",
    )


@pytest.mark.parametrize(
    "case", _GOLDEN["horizon_cases"], ids=[c["name"] for c in _GOLDEN["horizon_cases"]]
)
def test_horizon_is_increasing_in_progress(case):
    """Invariant both implementations must share: with A_0 < 1 the horizon is
    non-decreasing in progress (doubling times shrink but never reverse)."""
    horizons = np.array(case["horizon"], dtype=float)
    progress = np.array(case["progress"], dtype=float)
    order = np.argsort(progress)
    ordered = horizons[order]
    assert np.all(np.diff(ordered) >= 0), (
        f"[{case['name']}] frontend horizon is not monotone in progress: {ordered}"
    )


@pytest.mark.parametrize(
    "case", _GOLDEN["sc_cases"], ids=[c["name"] for c in _GOLDEN["sc_cases"]]
)
def test_automation_fraction_matches_frontend(case):
    """Python compute_automation_fraction == frontend getAutomationFraction.

    Both are the linear schedule through two anchors, extrapolated and clipped
    to [0, 1]. The anchors are the ones createDefaultSCHorizonConfig built on the
    TS side (lower anchor hardcoded there, upper anchor = progress_at_aa from the
    horizon inversion), replayed here so the same line is fitted.

    automation_interp_type is pinned to "linear" here because that is the only
    schedule the TS side implements — see the module docstring on the live
    presets selecting "logistic".
    """
    anchors = {
        float(case["lower_anchor_progress"]): float(case["lower_anchor_automation"]),
        float(case["progress_at_sc"]): float(
            case["automation_fraction_at_coding_automation_anchor"]
        ),
    }
    params = Parameters(
        automation_interp_type="linear",
        automation_anchors=anchors,
    )
    params.automation_model = AutomationModel(params)

    actual = [
        compute_automation_fraction(float(p), params) for p in case["progress"]
    ]

    assert_arrays_close(
        actual,
        case["automation_fraction"],
        rtol=AUTOMATION_RTOL,
        msg=f"[{case['name']}] automation fraction mismatch vs frontend "
        f"getAutomationFraction (anchors={anchors})",
    )

    # The grid is chosen to exercise all three regimes, so a case that only ever
    # returns clipped values would be silently weak.
    interior = [a for a in actual if 0.0 < a < 1.0]
    assert interior, (
        f"[{case['name']}] every automation value is clipped to 0 or 1; the "
        f"progress grid no longer exercises the interpolating branch"
    )


@pytest.mark.parametrize(
    "case", _GOLDEN["sc_cases"], ids=[c["name"] for c in _GOLDEN["sc_cases"]]
)
def test_progress_at_ac_inversion_matches_frontend(case):
    """Python observable.invert(target) == frontend calculateProgressAtSC.

    This is the inverse of the horizon relation: given the AC time horizon in
    minutes, solve for the progress at which it is reached.
    """
    observable = _make_decaying_horizon_observable(
        H_0=case["H_0"],
        A_0=1.0 - case["doubling_difficulty_growth_factor"],
        T_0=case["T_0"],
        anchor_progress=case["anchor_progress"],
    )
    target = case["ac_horizon_minutes"]
    actual = observable.invert(target)
    assert actual is not None, f"[{case['name']}] Python invert returned None"

    assert_scalar_close(
        actual,
        case["progress_at_sc"],
        rtol=HORIZON_RTOL,
        msg=f"[{case['name']}] progress_at_aa mismatch vs frontend "
        f"calculateProgressAtSC (target={target} min)",
    )


@pytest.mark.parametrize(
    "case", _GOLDEN["sc_cases"], ids=[c["name"] for c in _GOLDEN["sc_cases"]]
)
def test_inversion_round_trips_through_python_predict(case):
    """The frontend's progress_at_aa, fed back through the Python forward
    horizon, must reproduce the AC horizon it was derived from. This catches a
    sign or shift error that agreed on both sides of the inversion."""
    observable = _make_decaying_horizon_observable(
        H_0=case["H_0"],
        A_0=1.0 - case["doubling_difficulty_growth_factor"],
        T_0=case["T_0"],
        anchor_progress=case["anchor_progress"],
    )
    round_tripped = observable.predict(float(case["progress_at_sc"]))
    assert_scalar_close(
        round_tripped,
        case["ac_horizon_minutes"],
        # Round-tripping through pow() and its inverse loses a few ULP more than
        # a single evaluation does.
        rtol=1e-9,
        msg=f"[{case['name']}] predict(frontend progress_at_aa) != AC horizon",
    )
