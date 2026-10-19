"""
Tests for the COMBINATION of the two uplift features:

    ac_anchor_mode = 'uplift-solved'   (solve the automation schedule so the
                     model's OWN endogenous serial uplift matches an elicited
                     level M* and doubling time D at a reference date t*)

    the uplift reconstruction          (every date-anchored quantity is derived
                     against a one-shot reconstruction of the REALIZED with-AI
                     trajectory, not the human-only counterfactual -- the
                     unconditional behavior, not a mode)

Each has its own tests (test_uplift_solved_anchor.py,
test_reconstruction_anchoring.py, test_uplift_reconstruction.py). What is only
testable here is that they compose, and what composing BUYS.

THE WRINKLE THEY CLOSE
----------------------
Anchored on the HUMAN-ONLY mapping, 'uplift-solved' imposes M* at the
human-only progress level p* at t*, because the automation schedule has to
exist before the real trajectory can be integrated. The realized trajectory
then reaches a HIGHER progress level at t*, so the realized uplift is read
further along the solved schedule and lands above M* -- the chart's
curve-above-pin gap, and why the single-feature e2e test can only afford a 10%
tolerance.

With the reconstruction, it is built FIRST (its two statistics are exogenous
in time -- the excess doubles on the clock, never reading the trajectory back,
so there is no circularity), and the solve then imposes M* at the
RECONSTRUCTED p*, converting D at the RECONSTRUCTED rate(t*). Since the
reconstruction approximates the trajectory that will actually be integrated,
the imposed and realized uplift agree to the reconstruction's accuracy instead
of to the historical human-only/realized gap.

The tolerances below were MEASURED and then pinned with headroom; each one
also asserts the human-only-mapping run against the same tolerance to show the
tolerance is not vacuous.

MEASURED WRINKLE COLLAPSE
-------------------------
The evaluation headline, recorded here so it survives next to the assertions
that pin it. All rows: ac_anchor_mode='uplift-solved', DEFAULT_TIME_RANGE
([2015, 2050]), inputs/input_data.csv, everything else at defaults. "reconstruction"
is the shipped behavior; "human-only" is what the code did before it, reached
through the documented no-uplift fallback (see _run_on_the_human_only_mapping).

                                        human-only   reconstruction   factor
  realized serial uplift at t* vs M*     +4.874573%    +0.340199%        14x
  realized present-day progress vs p0    +0.080312     +0.003205         25x
    (OOMs of logE at present_day)
  AC time                                 2027.6435     2027.7824
  progress_at_aa                          37.5581       37.7524
  r_software                              3.1952        3.1855
  rate(t*) the solve converted D at       1.50853       1.61841
    (results['uplift_solve']['reference_progress_rate'] of each run)

The human-only mapping's error is DIRECTIONAL: its counterfactual has no
accumulated uplift, so it under-states present-day progress and both columns are
positive by construction. What the reconstruction leaves is small in MAGNITUDE
rather than reliably opposite in sign -- these rows used to read -0.005877% and
-0.000759, i.e. a slight overshoot, and at the Q2 belief they undershoot instead
while shrinking further. The sign of the remnant is noise; its size is the claim.

Off-default beliefs do at least as well -- the collapse is not an artifact of
the default numbers:

                                        human-only   reconstruction   factor
  M*=1.8, D=9mo    uplift error at t*    +3.082431%    +0.110691%        28x
                   progress residual     +0.107290     +0.002445         44x
  M*=1.1, D=20mo   uplift error at t*    +0.062502%    +0.000516%       121x
                   progress residual     +0.024262     -0.000370         66x
  t*=2027.0        uplift error at t*         --       -0.064716%
                   progress residual          --       -0.000232

And with the uplift solve OFF (ac_anchor_mode='time-horizon'), the baseline is no
longer reachable: the T_d correction pass rebuilds the reconstruction from the
derived schedule's MEASURED statistics through reconstruct_uplifted_trajectory,
which _run_on_the_human_only_mapping does not close off, and in that mode the
believed T_d misses by ~43% so the pass fires in both runs. Both end up on a
reconstruction and agree to noise:

  progress residual                      +0.002951     +0.005668
  AC time                                 2028.3809     2028.3797

(that row and the other AC-anchor modes are pinned in
test_reconstruction_anchoring.py, not here)

The BASELINE for every assertion below is the human-only column, i.e. the
behavior this branch had before the mode existed. There is no exact yardstick
to compare against -- solving the mismatch exactly would mean iterating full
integrations to a fixed point, which is precisely the cost the one-shot
reconstruction exists to avoid, and nothing here implements it. So "how good
is the approximation" is answered against the trajectory it approximates
(test_uplift_reconstruction.py) and "what does it buy" against the mapping it
replaces (here).

These are recorded as prose, not asserted as golden values -- what IS asserted
is the tolerance each column has to stay inside, plus the improvement factor
over the human-only column. So the input series or the present-day parameters
can move without this table having to be regenerated; a drift big enough to
matter shows up as a tolerance failure rather than silently.

These tests call production code and assert on values.
"""

import numpy as np
import pytest

import progress_model._impl as _impl
from progress_model import Parameters, ProgressModel
from progress_model.uplift_reconstruction import uplift_statistics

from utils import DEFAULT_TIME_RANGE, load_time_series_data


# --- Pinned tolerances -------------------------------------------------------
# Measured on this branch, at DEFAULT_TIME_RANGE, then given headroom.
#
#   realized serial uplift at t* vs M*      human-only   +4.874573%
#                                           reconstruct  +0.340199%  -> pin 5e-3
#   realized present-day progress vs p0     human-only   +0.080312 OOM
#                                           reconstruct  +0.003205 OOM -> pin 4e-3
#
# The uplift budget was 3e-4, measured against a present-day belief of 1.34
# serial. At the Q2 belief (2.0) there is an order of magnitude more excess for
# the one-shot linearization to get wrong between t* and the integration, and the
# residual is 0.34%. That is the accuracy of the scheme at this belief, not slack:
# closing it further needs the fixed-point loop the one-shot design exists to avoid.
#
# For scale: the single-mode e2e test (test_uplift_solved_anchor.py) can only
# assert rel=0.10 on the first of these. 5e-3 is still 20x tighter.
COMBINED_UPLIFT_RTOL = 5e-3
COMBINED_PROGRESS_TOL = 4e-3    # OOMs of logE at present_day

# A t* away from present_day exercises a second np.interp on the mapping (p* and
# rate(t*) both move). This constant was once 2e-3, tightened on a measurement of
# -0.0647% at t* = 2027.0 -- but that number was taken while the T_d correction pass
# was silently firing inside uplift-solved: the solve pins T_d at t*, the pass
# measured it as a secant around present_day, and the 6.8% cross-date mismatch
# cleared the old 5% epsilon. The pass is now decided by mode and never applies in
# uplift-solved, so the offset path is the plain one-shot reconstruction and
# measures +0.213%. That is the same scheme as the base case, so it gets the base
# case's budget rather than a tighter one it only ever met by accident.
OFFSET_REFERENCE_UPLIFT_RTOL = COMBINED_UPLIFT_RTOL

# Same story for the offset path's present-day anchor residual. With the pass firing
# it sat inside COMBINED_PROGRESS_TOL; as the plain one-shot reconstruction it
# measures +0.00457 OOM at t* = 2027.0. Budgeted the way HORIZON_PROGRESS_TOL is,
# measured value plus ~30% headroom.
OFFSET_REFERENCE_PROGRESS_TOL = 6e-3    # OOMs of logE at present_day

# Off-default beliefs: a stronger uplift with a faster doubling time gives the
# linearization more to get wrong. Measured +0.110691% at M*=1.8 / D=9mo, against
# +0.000516% at M*=1.1 / D=20mo.
OFF_DEFAULT_UPLIFT_RTOL = 2e-3

# Horizon mode + reconstruction: no uplift is being solved for, so only the
# anchor residual is meaningful. Measured +0.005668 OOM.
HORIZON_PROGRESS_TOL = 8e-3

# How much better the reconstruction has to be than the human-only mapping for
# the improvement to count as real rather than noise. Measured 14.3x on the
# uplift error and 25.1x on the progress residual at the defaults; the uplift
# factor shrank from 122x only because the reconstruction's own error grew with
# the belief, so the two quantities no longer share one threshold.
MIN_IMPROVEMENT_FACTOR = 20.0
MIN_UPLIFT_IMPROVEMENT_FACTOR = 10.0


@pytest.fixture(scope="module")
def time_series():
    return load_time_series_data()


def _run(time_series, **overrides):
    model = ProgressModel(Parameters(**overrides), time_series)
    model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
    return model


def _run_on_the_human_only_mapping(time_series, **overrides):
    """The pre-reconstruction behavior, for measuring what the correction buys.

    There is no mode to switch off, so the baseline is reached through
    production's OWN degenerate path: when reconstruct_from_params returns None
    both the r_software calibration and the anchored derivation read the
    human-only run, which is exactly what they did before the reconstruction
    existed. Not a reimplementation -- the real fallback, exercised on purpose.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_impl, 'reconstruct_from_params', lambda *a, **k: None)
        return _run(time_series, **overrides)


def _resolved_reference_date(params):
    if params.uplift_reference_date is not None:
        return float(params.uplift_reference_date)
    return float(params.present_day)


def _realized_serial_uplift_at_reference_date(model):
    """The serial coding-labor uplift the INTEGRATED trajectory actually has at t*.

    ai_coding_labor_mult_ref_present_day is the PARALLEL-space multiplier;
    ** parallel_penalty converts it to the serial one the solve targets. Same
    convention as test_uplift_solved_anchor.py, deliberately, so the two files
    are measuring the same quantity.
    """
    params = model.params
    times = np.asarray(model.results['times'], dtype=float)
    parallel = float(np.interp(
        _resolved_reference_date(params), times,
        np.asarray(model.results['ai_coding_labor_mult_ref_present_day'], dtype=float),
    ))
    return parallel ** float(params.parallel_penalty)


def _uplift_relative_error(model):
    """(realized uplift at t*) / M* - 1. Signed: positive means above the pin."""
    target = model.params.serial_uplift_at_reference_date
    return _realized_serial_uplift_at_reference_date(model) / target - 1.0


def _present_day_anchor_residual(model):
    """(realized progress at present_day) - (the p0 the anchors were built from).

    _anchor_stats_used is the anchor_stats dict compute_progress_trajectory
    actually derived everything from, so this is the exact self-consistency
    defect the reconstruction exists to shrink.
    """
    realized = float(np.interp(
        model.params.present_day,
        np.asarray(model.results['times'], dtype=float),
        np.asarray(model.results['progress'], dtype=float),
    ))
    return realized - float(model._anchor_stats_used['progress'])


# =============================================================================
# The statistics the two modes share
# =============================================================================


class TestSharedStatisticSource:
    """The reconstruction must consume the SAME uplift belief the solve imposes.

    If these ever diverge the composition is meaningless: the reconstruction
    would be approximating a trajectory built from a different uplift path than
    the one being solved for, and the agreement at t* would be a coincidence.

    The sourcing lives in one place -- uplift_reconstruction.uplift_statistics
    -- and is pinned here rather than in test_uplift_reconstruction.py because
    what makes it correct is agreement with the solve, not anything internal to
    the reconstruction.
    """

    def test_statistics_are_the_uplift_solved_targets(self):
        params = Parameters(ac_anchor_mode='uplift-solved',
                            serial_uplift_at_reference_date=1.8,
                            excess_uplift_doubling_time=9.0,
                            uplift_reference_date=2026.4)
        stats = uplift_statistics(params)
        assert stats['serial_multiplier'] == pytest.approx(1.8, rel=1e-12)
        # the parameter is in anchor-MONTHS
        assert stats['doubling_time_years'] == pytest.approx(9.0 / 12.0, rel=1e-12)
        assert stats['eval_date'] == pytest.approx(2026.4, rel=1e-12)
        # ...and the level is exactly the resolver's, not a parallel-space
        # number the solve would then reinterpret. This is the surviving form
        # of "M_0 must not silently disagree with the other description of
        # present-day uplift": the resolver IS that other description now.
        assert stats['serial_multiplier'] == pytest.approx(
            params.serial_uplift_at_reference_date, rel=1e-12)

    def test_the_none_conventions_match_the_resolver(self):
        """None means 'derive from the present-day belief' / 'use present_day',
        and the reconstruction must apply both conventions the same way the
        solve does."""
        params = Parameters(ac_anchor_mode='uplift-solved',
                            serial_uplift_at_reference_date=1.9331820449317627,
                            uplift_reference_date=None,
                            parallel_penalty=0.6)
        stats = uplift_statistics(params)
        assert stats['serial_multiplier'] == pytest.approx(
            params.serial_uplift_at_reference_date, rel=1e-12)
        assert stats['serial_multiplier'] == pytest.approx(1.9331820449317627, rel=1e-12)
        assert stats['eval_date'] == pytest.approx(params.present_day, rel=1e-12)

    def test_the_run_records_those_statistics(self, time_series):
        model = _run(time_series, ac_anchor_mode='uplift-solved',
                     serial_uplift_at_reference_date=1.6,
                     excess_uplift_doubling_time=15.0)
        used = model._uplift_reconstruction_info['statistics_used']
        assert used['serial_multiplier'] == pytest.approx(1.6, rel=1e-12)
        assert used['doubling_time_years'] == pytest.approx(15.0 / 12.0, rel=1e-12)
        assert used['eval_date'] == pytest.approx(model.params.present_day, rel=1e-12)

    def test_the_statistics_are_recorded_without_the_uplift_solve(self, time_series):
        """The reconstruction must source its statistics the same way when the AC
        anchor is NOT the uplift solve -- it is built from the uplift belief
        regardless of what determines AC. 'uplift-solved' is the default mode now, so
        the observable path has to be asked for by name."""
        model = _run(time_series, ac_anchor_mode='time-horizon')
        params = model.params
        assert params.ac_anchor_mode == 'time-horizon'
        used = model._uplift_reconstruction_info['statistics_used']
        assert used['serial_multiplier'] == pytest.approx(
            params.serial_uplift_at_reference_date, rel=1e-12)
        assert used['doubling_time_years'] == pytest.approx(
            params.excess_uplift_doubling_time / 12.0, rel=1e-12)
        assert used['eval_date'] == pytest.approx(params.present_day, rel=1e-12)


# =============================================================================
# The solve reads the reconstruction, not the human-only run
# =============================================================================


class TestTheSolveReadsTheReconstruction:

    def test_rate_at_t_star_comes_off_the_reconstruction(self, time_series):
        """rate(t*) converts the doubling time from months to progress units.
        It must be the RECONSTRUCTED rate; taking the human-only one would
        impose the right doubling time in the wrong coordinates."""
        model = _run(time_series, ac_anchor_mode='uplift-solved')
        t_star = _resolved_reference_date(model.params)
        reported = float(model.results['uplift_solve']['reference_progress_rate'])

        reconstructed = float(np.interp(
            t_star,
            np.asarray(model._uplift_reconstruction['times'], dtype=float),
            np.asarray(model._uplift_reconstruction['progress_rates'], dtype=float),
        ))
        human_only = float(np.interp(
            t_star,
            np.asarray(model.human_only_results['times'], dtype=float),
            np.asarray(model.human_only_results['progress_rates'], dtype=float),
        ))

        assert reported == pytest.approx(reconstructed, rel=1e-12)
        # and the two really are different, so the assertion above has teeth
        assert abs(reconstructed - human_only) > 1e-3

    def test_rate_at_t_star_is_human_only_without_the_reconstruction(self, time_series):
        model = _run_on_the_human_only_mapping(
            time_series, ac_anchor_mode='uplift-solved')
        t_star = _resolved_reference_date(model.params)
        reported = float(model.results['uplift_solve']['reference_progress_rate'])
        human_only = float(np.interp(
            t_star,
            np.asarray(model.human_only_results['times'], dtype=float),
            np.asarray(model.human_only_results['progress_rates'], dtype=float),
        ))
        assert reported == pytest.approx(human_only, rel=1e-12)


# =============================================================================
# What the combination buys: the wrinkle collapses
# =============================================================================


class TestTheWrinkleCollapses:

    @pytest.fixture(scope="class")
    def pair(self, time_series):
        return {
            'human-only': _run_on_the_human_only_mapping(
                time_series, ac_anchor_mode='uplift-solved'),
            'uplift-reconstruction': _run(time_series, ac_anchor_mode='uplift-solved'),
        }

    def test_realized_uplift_at_t_star_hits_the_target(self, pair):
        error = _uplift_relative_error(pair['uplift-reconstruction'])
        assert abs(error) < COMBINED_UPLIFT_RTOL, (
            f"realized serial uplift at t* is {error:+.6%} from M*, "
            f"outside the pinned {COMBINED_UPLIFT_RTOL:.1e}")

    def test_the_tolerance_is_not_vacuous_for_uplift(self, pair):
        """The same assertion must FAIL on the human-only mapping, or the
        tolerance above is measuring nothing."""
        baseline = abs(_uplift_relative_error(pair['human-only']))
        combined = abs(_uplift_relative_error(pair['uplift-reconstruction']))
        assert baseline > COMBINED_UPLIFT_RTOL
        assert baseline / max(combined, 1e-15) > MIN_UPLIFT_IMPROVEMENT_FACTOR, (
            f"human-only {baseline:.6%} vs reconstruction {combined:.6%} is only "
            f"{baseline / max(combined, 1e-15):.1f}x")

    def test_realized_present_day_progress_matches_the_p0_used(self, pair):
        residual = _present_day_anchor_residual(pair['uplift-reconstruction'])
        assert abs(residual) < COMBINED_PROGRESS_TOL, (
            f"realized present-day progress is {residual:+.6f} OOMs from the p0 the "
            f"anchors were built from, outside the pinned {COMBINED_PROGRESS_TOL:.1e}")

    def test_the_tolerance_is_not_vacuous_for_progress(self, pair):
        baseline = abs(_present_day_anchor_residual(pair['human-only']))
        combined = abs(_present_day_anchor_residual(pair['uplift-reconstruction']))
        assert baseline > COMBINED_PROGRESS_TOL
        assert baseline / max(combined, 1e-15) > MIN_IMPROVEMENT_FACTOR, (
            f"human-only {baseline:.6f} vs reconstruction {combined:.6f} OOMs is only "
            f"{baseline / max(combined, 1e-15):.1f}x")

    def test_the_human_only_bias_is_directional_and_the_remnant_is_not(self, pair):
        """Directional, and a genuinely different failure mode from 'small'.

        Human-only under-states present-day progress (the counterfactual has no
        accumulated uplift in it), so its residual is positive BY CONSTRUCTION --
        that is a claim about the mechanism, not about the size of the number, and
        it is asserted as such.

        What is left after the reconstruction is asserted on magnitude. This used to
        read `< 0`: at the old belief the extrapolated excess overshot slightly, so
        the residual crossed zero, and crossing was taken as evidence that the bias
        was gone rather than merely reduced. At the Q2 belief the remnant SHRANK by a
        further 25x and undershoots instead (+0.003205), which shows the sign was
        reading the approximation noise. Magnitude is the claim that survives.
        """
        assert _present_day_anchor_residual(pair['human-only']) > 0
        assert abs(_present_day_anchor_residual(
            pair['uplift-reconstruction'])) < COMBINED_PROGRESS_TOL

    def test_ac_moves_later_and_progress_at_aa_higher(self, pair):
        """Anchoring higher up the progress axis pushes AC out; the same
        direction test_reconstruction_anchoring.py pins for the horizon mode, which
        must also hold when AC is an OUTPUT of the uplift solve."""
        base, combined = pair['human-only'], pair['uplift-reconstruction']
        assert float(combined.results['aa_time']) > float(base.results['aa_time'])
        assert combined.params.progress_at_aa > base.params.progress_at_aa
        assert float(combined.params.r_software) < float(base.params.r_software)


# =============================================================================
# The collapse is not an artifact of the default numbers
# =============================================================================


class TestOffDefaultUpliftBeliefs:

    @pytest.mark.parametrize("overrides,label", [
        ({'serial_uplift_at_reference_date': 1.8,
          'excess_uplift_doubling_time': 9.0}, "stronger uplift, faster doubling"),
        ({'serial_uplift_at_reference_date': 1.1,
          'excess_uplift_doubling_time': 20.0}, "weaker uplift, slower doubling"),
    ])
    def test_the_collapse_survives_a_different_belief(self, time_series, overrides, label):
        combined = _run(time_series, ac_anchor_mode='uplift-solved', **overrides)
        baseline = _run_on_the_human_only_mapping(
            time_series, ac_anchor_mode='uplift-solved', **overrides)

        combined_err = abs(_uplift_relative_error(combined))
        baseline_err = abs(_uplift_relative_error(baseline))
        assert combined_err < OFF_DEFAULT_UPLIFT_RTOL, f"{label}: {combined_err:.6%}"
        assert baseline_err / max(combined_err, 1e-15) > MIN_IMPROVEMENT_FACTOR, (
            f"{label}: human-only {baseline_err:.6%} vs reconstruction {combined_err:.6%}")

        assert abs(_present_day_anchor_residual(combined)) < COMBINED_PROGRESS_TOL

    def test_a_reference_date_away_from_present_day(self, time_series):
        """t* is a genuinely independent parameter, so the composition has to
        hold when the uplift is elicited at a date other than present_day: the
        reconstruction's eval_date, p* and rate(t*) all have to follow it.

        The probe used to be 2026.578, which WAS such a date when present_day was
        2026.0959. present_day is the Q2 uplift vintage now (2026.578082), so that
        probe is present_day to within 3e-5 years and the test measured the base case
        twice under a different name.
        """
        t_star = 2027.0
        model = _run(time_series, ac_anchor_mode='uplift-solved',
                     uplift_reference_date=t_star)

        assert model._uplift_reconstruction_info['statistics_used']['eval_date'] == \
            pytest.approx(t_star, rel=1e-12)
        assert float(model.results['uplift_solve']['reference_date']) == \
            pytest.approx(t_star, rel=1e-12)

        error = _uplift_relative_error(model)
        assert abs(error) < OFFSET_REFERENCE_UPLIFT_RTOL, (
            f"realized uplift at t*={t_star} is {error:+.6%} from M*")
        assert abs(_present_day_anchor_residual(model)) < OFFSET_REFERENCE_PROGRESS_TOL


# =============================================================================
# Degenerate composition
# =============================================================================


class TestNonUpliftAnchorModesCompose:
    """The reconstruction is orthogonal to how AC's progress level is chosen, so
    every AC-anchor mode has to keep working under it and end up anchored on it.

    These modes no longer have a human-only baseline to be compared against. Their
    believed T_d is a guess rather than something they pin -- the derived schedule
    misses it by 27-345% -- so the T_d correction pass, which applies in every mode
    but 'uplift-solved', fires, and it rebuilds through
    reconstruct_uplifted_trajectory rather than the reconstruct_from_params door
    _run_on_the_human_only_mapping closes. Both runs therefore land on a
    reconstruction. What is asserted is that convergence, plus the one thing that
    still separates them: the r_software calibration, which happens before the pass.
    """

    @pytest.mark.parametrize("mode", ['time-horizon', 'revenue', 'eci'])
    def test_the_anchoring_converges_under_the_reconstruction(self, time_series, mode):
        base = _run_on_the_human_only_mapping(time_series, ac_anchor_mode=mode)
        combined = _run(time_series, ac_anchor_mode=mode)

        assert combined._uplift_reconstruction_info['available'] is True
        assert np.isfinite(float(combined.results['aa_time']))
        # every mode shares the r_software recalibration, which is upstream of the
        # correction pass and so still tells the two runs apart
        assert float(combined.params.r_software) < float(base.params.r_software)

        assert combined._td_correction_info['applied'] is True
        assert base._td_correction_info['applied'] is True
        # Measured: AC agrees to 0.004 years, progress_at_aa to 0.05 OOMs.
        assert abs(float(combined.results['aa_time'])
                   - float(base.results['aa_time'])) < 5e-3
        assert abs(float(combined.params.progress_at_aa)
                   - float(base.params.progress_at_aa)) < 5e-2

    def test_horizon_mode_anchor_residual_also_collapses(self, time_series):
        base = _run_on_the_human_only_mapping(
            time_series, ac_anchor_mode='time-horizon')
        combined = _run(time_series, ac_anchor_mode='time-horizon')

        baseline_residual = abs(_present_day_anchor_residual(base))
        combined_residual = abs(_present_day_anchor_residual(combined))
        assert combined_residual < HORIZON_PROGRESS_TOL, (
            f"horizon mode leaves {combined_residual:.6f} OOMs of anchor residual")
        # The baseline collapses too, via the correction pass (see the class
        # docstring), so the improvement FACTOR has nothing left to measure. What is
        # asserted instead is that both are far below the ~0.08 OOMs the human-only
        # mapping leaves when the pass does NOT fire -- see TestTheWrinkleCollapses,
        # where the same comparison in 'uplift-solved' still has a real baseline.
        assert baseline_residual < HORIZON_PROGRESS_TOL, (
            f"the correction pass left {baseline_residual:.6f} OOMs on the baseline run")

    def test_ac_time_is_where_the_trajectory_reaches_progress_at_aa(self, time_series):
        """Coherence of the milestone under the reconstruction: the AC time and
        progress_at_aa still describe the same point on the trajectory that was
        actually integrated."""
        model = _run(time_series, ac_anchor_mode='time-horizon')
        aa_time = float(model.results['aa_time'])
        progress_at_aa_time = float(np.interp(
            aa_time,
            np.asarray(model.results['times'], dtype=float),
            np.asarray(model.results['progress'], dtype=float),
        ))
        assert progress_at_aa_time == pytest.approx(model.params.progress_at_aa, abs=1e-3)
