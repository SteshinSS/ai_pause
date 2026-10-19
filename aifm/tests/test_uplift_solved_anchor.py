"""
Tests for the 'uplift-solved' AC-anchor mode.

Unlike the observable-inversion modes, this one solves the logistic automation
schedule's (k, x0) so that the model's OWN endogenous serial ref-present-day
coding-labor multiplier M hits a stated level with a stated doubling rate; the
AC progress level then falls out as the A(p) = 1 crossing.

Everything here calls production code — solve_logistic_from_uplift,
serial_uplift_for_logistic, AutomationModel, ProgressModel — and asserts on
values and relationships, never on shape.
"""

import re

import numpy as np
import pytest

from progress_model import config as cfg
from conftest import DEFAULT_TIME_RANGE
from utils import load_time_series_data
from progress_model import Parameters, ProgressModel
from progress_model.automation_model import (
    AutomationModel,
    UPLIFT_SOLVE_FD_STEP,
    serial_uplift_for_logistic,
    solve_lower_anchor_via_automation_model,
    solve_logistic_from_uplift,
    solve_logistic_with_asymptote_from_uplift,
)
from progress_model.ac_anchor import compute_gap_progress_units


# Half-width of the finite difference used to *verify* the doubling rate. Kept
# an order of magnitude away from the solver's own step so the checks aren't
# self-fulfilling. The optimal-CES frontier is interpolated on a finite grid, so
# M(p) is only piecewise smooth and the measured slope carries a step-dependent
# wobble of order 1e-4 relative — hence the two tolerances used below.
VERIFY_FD = 3e-2
INDEPENDENT_STEP_RTOL = 1e-3


@pytest.fixture(scope="module")
def time_series():
    """Module-scoped copy of the default input data (conftest's is function-scoped)."""
    return load_time_series_data()


@pytest.fixture(scope="module")
def anchor_stats(time_series):
    """Present-day human-only stats (H0, C0, p0, progress rate) from a real run.

    Taken from the production human-only trajectory rather than hard-coded so
    the solver is exercised on the inputs it actually sees in the pipeline.
    """
    model = ProgressModel(Parameters(), time_series)
    model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
    stats = model.human_only_results['anchor_stats']
    return {
        'H': float(stats['human_labor']),
        'C': float(stats['inference_H100e']),
        'progress': float(stats['progress']),
        'progress_rate': float(stats['progress_rate']),
        # The efficiency slope the frontier actually consumes is the converted,
        # per-progress-unit one; carry the post-run params so tests match.
        'params': model.params,
    }


def _measure(k, x0, L, stats, target_progress=None, fd=VERIFY_FD):
    """Re-measure (M(p*), achieved excess-doubling in progress units) from the
    solved schedule using production's uplift definition."""
    p_star = stats['progress'] if target_progress is None else target_progress
    probe = np.array([p_star - fd, p_star, p_star + fd])
    M = serial_uplift_for_logistic(
        probe, k, x0, L, stats['H'], stats['C'], stats['params']
    )
    excess = M - 1.0
    slope = (np.log(excess[2]) - np.log(excess[0])) / (2.0 * fd)
    return float(M[1]), float(np.log(2.0) / slope)


class TestSolverRoundTrip:
    """Solve, rebuild, re-measure: do we get the inputs back?"""

    @pytest.mark.parametrize("target_uplift,doubling_months", [
        (1.5, 12.0),
        (2.0, 12.0),
        (2.0, 6.0),
        (3.5, 24.0),
        (1.2, 18.0),
    ])
    def test_round_trip_uplift_and_doubling_time(self, anchor_stats, target_uplift, doubling_months):
        L = 1.1
        D_progress = doubling_months / 12.0 * anchor_stats['progress_rate']

        solved = solve_logistic_from_uplift(
            target_uplift, anchor_stats['progress'], D_progress, L,
            anchor_stats['H'], anchor_stats['C'], anchor_stats['params'],
        )
        assert solved is not None, f"solver failed for M*={target_uplift}, D={doubling_months}mo"
        k, x0, diagnostics = solved

        assert k > 0
        assert diagnostics['converged'] is True
        assert diagnostics['max_abs_residual'] < 1e-8

        # Measured at the solver's own step: this must be essentially exact,
        # since it is the condition the solver claims to have driven to zero.
        M_same, D_same = _measure(k, x0, L, anchor_stats, fd=UPLIFT_SOLVE_FD_STEP)
        assert M_same == pytest.approx(target_uplift, rel=1e-9)
        assert D_same == pytest.approx(D_progress, rel=1e-8)

        # Measured at a 30x larger, independent step: agreement here is what
        # says the fit tracks the real local behavior of M rather than an
        # artifact of the difference step.
        M_ind, D_ind = _measure(k, x0, L, anchor_stats)
        assert M_ind == pytest.approx(target_uplift, rel=1e-9)  # M is step-independent
        assert D_ind == pytest.approx(D_progress, rel=INDEPENDENT_STEP_RTOL)

    def test_reference_automation_fraction_is_on_the_solved_curve(self, anchor_stats):
        """A(p*) reported in the diagnostics must equal the logistic evaluated there."""
        L = 1.1
        D_progress = anchor_stats['progress_rate']
        k, x0, diagnostics = solve_logistic_from_uplift(
            2.0, anchor_stats['progress'], D_progress, L,
            anchor_stats['H'], anchor_stats['C'], anchor_stats['params'],
        )
        am = AutomationModel(anchor_stats['params'], logistic=(L, k, x0))
        assert am.get_automation_fraction(anchor_stats['progress']) == pytest.approx(
            diagnostics['automation_fraction_at_reference'], rel=1e-12
        )
        # And the crossing really is where A hits 1.0
        crossing = diagnostics['progress_at_automation_crossing']
        assert crossing == pytest.approx(x0 - np.log(L - 1.0) / k, rel=1e-12)
        assert am.get_automation_fraction(crossing) == pytest.approx(1.0, abs=1e-12)
        assert am.get_automation_fraction(crossing - 0.05) < 1.0

    def test_uplift_definition_agrees_with_the_lower_anchor_solver(self, anchor_stats):
        """Cross-check the two solvers against each other.

        solve_lower_anchor_via_automation_model targets
        swe_multiplier**parallel_penalty at the present day using the same
        implied-ratio definition. Feeding it the crossing we just solved for,
        it must recover the same lower anchor A(p*) — which is only true if both
        solvers mean the same thing by 'the multiplier'.
        """
        L = 1.1
        params = anchor_stats['params']
        k, x0, diagnostics = solve_logistic_from_uplift(
            2.0, anchor_stats['progress'], anchor_stats['progress_rate'], L,
            anchor_stats['H'], anchor_stats['C'], params,
        )
        crossing = diagnostics['progress_at_automation_crossing']

        saved_paa = params.progress_at_aa
        saved_aut = params.automation_fraction_at_coding_automation_anchor
        try:
            params.progress_at_aa = crossing
            params.automation_fraction_at_coding_automation_anchor = 1.0
            recovered = solve_lower_anchor_via_automation_model(
                2.0 ** (1.0 / params.parallel_penalty),  # serial M* -> parallel swe multiplier
                anchor_stats['progress'], anchor_stats['H'], anchor_stats['C'], params,
            )
        finally:
            params.progress_at_aa = saved_paa
            params.automation_fraction_at_coding_automation_anchor = saved_aut

        assert recovered == pytest.approx(
            diagnostics['automation_fraction_at_reference'], rel=1e-6
        )

    def test_synthesized_anchors_refit_to_the_same_logistic(self, anchor_stats):
        """The anchors stored for display must regenerate the solved schedule."""
        L = 1.1
        params = anchor_stats['params']
        k, x0, diagnostics = solve_logistic_from_uplift(
            2.0, anchor_stats['progress'], anchor_stats['progress_rate'], L,
            anchor_stats['H'], anchor_stats['C'], params,
        )
        crossing = diagnostics['progress_at_automation_crossing']

        saved_anchors = params.automation_anchors
        saved_L = params.automation_logistic_asymptote
        try:
            params.automation_logistic_asymptote = L
            params.automation_anchors = {
                anchor_stats['progress']: diagnostics['automation_fraction_at_reference'],
                crossing: 1.0,
            }
            refit = AutomationModel(params)
        finally:
            params.automation_anchors = saved_anchors
            params.automation_logistic_asymptote = saved_L

        assert refit.logistic_k == pytest.approx(k, rel=1e-9)
        assert refit.logistic_x0 == pytest.approx(x0, rel=1e-9)


class TestInvariants:
    """Monotone responses of the AC crossing to each input."""

    @staticmethod
    def _crossing(anchor_stats, target_uplift=2.0, doubling_months=12.0, L=1.1):
        D_progress = doubling_months / 12.0 * anchor_stats['progress_rate']
        solved = solve_logistic_from_uplift(
            target_uplift, anchor_stats['progress'], D_progress, L,
            anchor_stats['H'], anchor_stats['C'], anchor_stats['params'],
        )
        assert solved is not None, f"solver failed for M*={target_uplift}, D={doubling_months}, L={L}"
        return solved[2]['progress_at_automation_crossing']

    def test_slower_doubling_pushes_ac_later(self, anchor_stats):
        crossings = [self._crossing(anchor_stats, doubling_months=d)
                     for d in (4.0, 8.0, 12.0, 24.0, 48.0)]
        assert crossings == sorted(crossings), (
            f"AC crossing must increase with the excess-uplift doubling time, got {crossings}"
        )
        # And the effect is substantial, not a rounding wobble
        assert crossings[-1] - crossings[0] > 1.0

    def test_higher_present_uplift_pulls_ac_earlier(self, anchor_stats):
        crossings = [self._crossing(anchor_stats, target_uplift=m)
                     for m in (1.2, 1.5, 2.0, 3.0, 5.0)]
        assert crossings == sorted(crossings, reverse=True), (
            f"AC crossing must decrease as the reference uplift rises, got {crossings}"
        )
        assert crossings[0] - crossings[-1] > 1.0

    def test_higher_asymptote_pulls_the_crossing_earlier(self, anchor_stats):
        crossings = [self._crossing(anchor_stats, L=L) for L in (1.05, 1.1, 1.3, 2.0)]
        assert crossings == sorted(crossings, reverse=True), (
            f"AC crossing must decrease as the logistic asymptote rises, got {crossings}"
        )

    def test_infeasible_targets_return_none_rather_than_garbage(self, anchor_stats):
        """A doubling rate no logistic can reach at that uplift level must fail
        cleanly, not silently return a schedule that misses the targets."""
        # M* = 5 with a very fast doubling: the reachable slope range tops out
        # well below this (see the reachable-range warning the solver logs).
        solved = solve_logistic_from_uplift(
            5.0, anchor_stats['progress'], 0.02, 1.1,
            anchor_stats['H'], anchor_stats['C'], anchor_stats['params'],
        )
        assert solved is None


class TestPipelineIntegration:
    """The mode as wired into compute_progress_trajectory."""

    @pytest.fixture(scope="class")
    def uplift_model(self, time_series):
        params = Parameters()
        params.ac_anchor_mode = 'uplift-solved'
        params.serial_uplift_at_reference_date = 2.0
        params.excess_uplift_doubling_time = 12.0
        model = ProgressModel(params, time_series)
        model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
        return model

    def test_progress_at_aa_is_the_automation_crossing(self, uplift_model):
        """A(progress_at_aa - gap) == 1.0, i.e. progress_at_aa is an output."""
        params = uplift_model.params
        gap = compute_gap_progress_units(
            params, uplift_model.human_only_results['anchor_stats']['progress_rate']
        )
        crossing = params.progress_at_aa - gap
        am = params.automation_model
        assert am.get_automation_fraction(crossing) == pytest.approx(1.0, abs=1e-9)
        assert am.get_automation_fraction(crossing - 0.1) < 1.0
        assert crossing == pytest.approx(
            uplift_model._uplift_solve_diagnostics['progress_at_automation_crossing'], rel=1e-12
        )

    def test_gap_shifts_progress_at_aa_by_exactly_the_gap(self, default_time_series_data):
        """Turning the AC gap on must move progress_at_aa and nothing else about
        the solved schedule."""
        def run(include_gap):
            params = Parameters()
            params.ac_anchor_mode = 'uplift-solved'
            params.include_gap = include_gap
            params.gap_years = 0.7
            model = ProgressModel(params, default_time_series_data)
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
            return model

        no_gap, with_gap = run('no gap'), run('gap')
        d0, d1 = no_gap._uplift_solve_diagnostics, with_gap._uplift_solve_diagnostics
        # Same schedule...
        assert d1['k'] == pytest.approx(d0['k'], rel=1e-9)
        assert d1['x0'] == pytest.approx(d0['x0'], rel=1e-9)
        # ...but progress_at_aa displaced by 0.7 anchor-years of progress, converted at
        # the rate of the mapping actually anchored against. That is the uplift
        # RECONSTRUCTION, not the human-only run: the two coincided only while the
        # default uplift was small enough for the reconstruction's rate shift to hide
        # inside 1e-9. At the elicited belief it is a 7.5% shift (1.5057 -> 1.6184).
        rate = no_gap._anchor_stats_used['progress_rate']
        assert with_gap.params.progress_at_aa - no_gap.params.progress_at_aa == pytest.approx(
            0.7 * rate, rel=1e-9
        )

    def test_solved_schedule_hits_the_targets_at_the_anchoring_reference(self, uplift_model):
        """End-to-end, the schedule the pipeline installed still satisfies both
        conditions exactly at the progress level they were imposed at.

        That level comes off the mapping the run anchored against -- the uplift
        reconstruction -- not off the human-only run. They used to be close enough to
        pass at rel=1e-8; at the elicited belief the reconstruction sits +0.076 OOMs
        and 7.5% faster, so reading the human-only stats measures the schedule at the
        wrong place (M = 1.9378 rather than 2.0).
        """
        params = uplift_model.params
        stats = uplift_model._anchor_stats_used
        am = params.automation_model
        M, D = _measure(
            am.logistic_k, am.logistic_x0, am.logistic_L,
            {'H': float(stats['human_labor']), 'C': float(stats['inference_H100e']),
             'progress': float(stats['progress']), 'params': params},
        )
        assert M == pytest.approx(params.serial_uplift_at_reference_date, rel=1e-8)
        assert D == pytest.approx(
            params.excess_uplift_doubling_time / 12.0 * float(stats['progress_rate']),
            rel=INDEPENDENT_STEP_RTOL,
        )

    def test_realized_serial_uplift_at_reference_date_is_close_to_target(self, uplift_model):
        """The realized trajectory's uplift at t* is near — but not equal to — M*.

        The two conditions are imposed at the HUMAN-ONLY progress level at t*,
        because the automation schedule has to exist before the real trajectory
        can be integrated. Once automation feeds back into progress, the
        integrated trajectory reaches a higher progress level at t* than the
        human-only run did, and A (hence M) is read further along the schedule.
        That is the same rebase wrinkle the horizon/ECI/revenue observables have,
        and it is why this tolerance is percent-level rather than 1e-9.
        """
        params = uplift_model.params
        results = uplift_model.results
        times = np.asarray(results['times'], dtype=float)
        # The metric is the parallel-space multiplier; serial = metric**parallel_penalty
        parallel_mult = np.interp(
            params.present_day, times,
            np.asarray(results['ai_coding_labor_mult_ref_present_day'], dtype=float),
        )
        realized_serial = float(parallel_mult) ** params.parallel_penalty
        target = params.serial_uplift_at_reference_date

        assert realized_serial == pytest.approx(target, rel=0.10)
        # Directional: the feedback can only push realized progress (and so the
        # realized uplift) above the human-only level it was fitted at.
        assert realized_serial >= target

    def test_ac_time_is_finite_and_consistent_with_progress_at_aa(self, uplift_model):
        """The AC milestone time must be the moment the trajectory reaches
        progress_at_aa."""
        params = uplift_model.params
        results = uplift_model.results
        aa_time = results.get('aa_time')
        assert aa_time is not None and np.isfinite(aa_time)
        progress_at_aa_time = np.interp(
            aa_time,
            np.asarray(results['times'], dtype=float),
            np.asarray(results['progress'], dtype=float),
        )
        assert progress_at_aa_time == pytest.approx(params.progress_at_aa, rel=1e-4)

    def test_non_default_reference_date_moves_the_solve(self, default_time_series_data):
        """uplift_reference_date shifts p*, so the same M* implies a later AC."""
        def run(reference_date):
            params = Parameters()
            params.ac_anchor_mode = 'uplift-solved'
            params.uplift_reference_date = reference_date
            model = ProgressModel(params, default_time_series_data)
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
            return model

        base = run(None)
        later = run(Parameters().present_day + 1.0)

        # Asking for the same uplift a year later means capability arrives later
        assert later.params.progress_at_aa > base.params.progress_at_aa
        # p* actually moved to the later date's human-only progress level
        stats_progress = later.human_only_results['anchor_stats']['progress']
        assert later._uplift_solve_diagnostics['reference_progress'] > stats_progress

    def test_reference_date_none_uses_the_present_day_rate_exactly(self, uplift_model):
        """uplift_reference_date=None must reproduce the present-day anchor rate
        bit-for-bit, so the convenience default changes nothing."""
        d = uplift_model._uplift_solve_diagnostics
        assert d['reference_date'] == uplift_model.params.present_day
        assert d['reference_progress_rate'] == d['present_day_anchor_progress_rate']
        # ...on the mapping actually anchored against (the uplift reconstruction).
        assert d['reference_progress'] == pytest.approx(
            uplift_model._anchor_stats_used['progress'], rel=1e-12
        )

    @pytest.mark.parametrize("reference_date", [2027.0, 2027.5])
    def test_doubling_conversion_uses_the_reference_date_rate(
        self, time_series, reference_date
    ):
        """months -> progress units must convert at rate(t*), not the present-day
        anchor rate.

        Both probe dates are a good way clear of present_day (2026.578082), which is
        what makes the "rate(t*) != rate(present_day)" claim below bite. 2026.578 used
        to be such a date -- back when present_day was 2026.0959 -- and is now
        present_day to within 3e-5 years, where the two rates agree to 1e-5 and the
        assertion is vacuous.

        The rate is read off the mapping the run anchors against (the uplift
        reconstruction), not off the human-only run.
        """
        params = Parameters()
        params.ac_anchor_mode = 'uplift-solved'
        params.uplift_reference_date = reference_date
        model = ProgressModel(params, time_series)
        model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
        d = model._uplift_solve_diagnostics

        mapping = model._uplift_reconstruction or model.human_only_results
        expected_rate = float(np.interp(
            reference_date, mapping['times'], mapping['progress_rates'],
        ))
        assert d['reference_progress_rate'] == pytest.approx(expected_rate, rel=1e-12)
        # The distinction is real, not a rounding artifact
        assert d['reference_progress_rate'] != pytest.approx(
            d['present_day_anchor_progress_rate'], rel=1e-4
        )
        # ...and the doubling time was converted with it
        assert d['target_doubling_progress_units'] == pytest.approx(
            params.excess_uplift_doubling_time / 12.0 * expected_rate, rel=1e-12
        )

    def test_non_logistic_schedule_is_rejected_loudly(self, default_time_series_data):
        params = Parameters()
        params.ac_anchor_mode = 'uplift-solved'
        params.automation_interp_type = 'linear'
        model = ProgressModel(params, default_time_series_data)
        with pytest.raises(ValueError, match="requires automation_interp_type='logistic'"):
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

    def test_infeasible_targets_abort_the_run(self, default_time_series_data):
        params = Parameters()
        params.ac_anchor_mode = 'uplift-solved'
        params.serial_uplift_at_reference_date = 5.0
        params.excess_uplift_doubling_time = 0.6  # far faster than reachable at M* = 5
        model = ProgressModel(params, default_time_series_data)
        with pytest.raises(ValueError, match="could not find a logistic schedule"):
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)


class TestDefaultModeUnaffected:
    """The observable path must be unaffected by the new mode existing.

    'uplift-solved' IS the default now (the Q2 sheet's highest-weight anchor block),
    so "inert unless selected" is no longer the claim. What is still claimed, and
    tested here, is that selecting 'time-horizon' gets you exactly the observable
    path it always did.
    """

    def test_the_default_mode_is_the_uplift_solve(self):
        assert Parameters().ac_anchor_mode == 'uplift-solved'
        assert cfg.DEFAULT_AC_ANCHOR_MODE == 'uplift-solved'

    def test_time_horizon_mode_still_uses_the_observable_path(self, default_time_series_data):
        params = Parameters()
        params.ac_anchor_mode = 'time-horizon'
        model = ProgressModel(params, default_time_series_data)
        model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

        assert model._uplift_solve_diagnostics is None
        assert model._ac_anchor_observable is None  # time-horizon keeps its own attribute
        assert model._horizon_observable is not None

        # The default path builds its anchors the other way round: progress_at_aa
        # is the UPPER anchor's key (an input), carrying
        # automation_fraction_at_coding_automation_anchor as its value, and the
        # lower anchor was solved so the present-day multiplier matches
        # swe_multiplier_at_present_day. In 'uplift-solved' the upper anchor is
        # instead pinned at exactly 1.0 with progress_at_aa derived from it.
        # The lower anchor is keyed on the mapping actually anchored against (the
        # uplift reconstruction); the human-only run sits +0.076 OOMs below it.
        stats = model._anchor_stats_used
        anchors = params.automation_anchors
        assert sorted(anchors) == pytest.approx(
            sorted([float(stats['progress']), float(params.progress_at_aa)])
        )
        upper = anchors[max(anchors)]
        assert upper == pytest.approx(params.automation_fraction_at_coding_automation_anchor)

        # And the lower anchor really does reproduce the requested present-day
        # multiplier under the same uplift definition the new mode uses.
        am = params.automation_model
        realized_serial = float(serial_uplift_for_logistic(
            [float(stats['progress'])], am.logistic_k, am.logistic_x0, am.logistic_L,
            float(stats['human_labor']), float(stats['inference_H100e']), params,
        )[0])
        assert realized_serial == pytest.approx(
            params.swe_multiplier_at_present_day ** params.parallel_penalty, rel=1e-6
        )

    def test_uplift_solved_gives_a_different_answer_than_time_horizon(self, default_time_series_data):
        """Sanity that the mode is actually doing something."""
        def run(mode):
            params = Parameters()
            params.ac_anchor_mode = mode
            model = ProgressModel(params, default_time_series_data)
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
            return model.params.progress_at_aa

        assert run('uplift-solved') != pytest.approx(run('time-horizon'), rel=1e-3)


class TestDerivedUpliftDefault:
    """The FIELD serial_uplift_at_reference_date defaults to None = 'not stated'.

    The default BELIEF lives in DEFAULT_PARAMETERS and the resolvers apply it when
    neither container was stated, so both containers describe one number and entering
    the mode does not silently move the present-day uplift. Before this existed, a
    hard-coded 2.0 serial collided with the parallel default the other modes used and
    switching modes jumped the realized present-day uplift by a factor of ~1.54.

    The direction of derivation has since flipped: the belief is denominated in the
    SERIAL container and the parallel view is derived from it (2.0 serial <-> 4.0
    parallel at pp = 0.5), rather than the other way round. Both containers still
    agree, which is the property these tests pin.
    """

    def test_resolver_derives_from_the_present_day_belief(self):
        params = Parameters()
        assert params.serial_uplift_at_reference_date == pytest.approx(
            float(cfg.DEFAULT_PARAMETERS['serial_uplift_at_reference_date']), rel=1e-12)
        # The two containers are views of one number, so the round trip is the identity.
        expected = params.swe_multiplier_at_present_day ** params.parallel_penalty
        assert params.serial_uplift_at_reference_date == pytest.approx(expected, rel=1e-12)
        # sanity: it IS the configured belief, verbatim, not a re-derivation of it
        assert params.serial_uplift_at_reference_date == pytest.approx(
            float(cfg.DEFAULT_PARAMETERS['serial_uplift_at_reference_date']), rel=1e-12)

    def test_explicit_value_wins_over_the_derived_default(self):
        params = Parameters()
        params.serial_uplift_at_reference_date = 2.5
        assert params.serial_uplift_at_reference_date == pytest.approx(2.5, rel=1e-12)

    def test_a_value_that_is_not_an_uplift_is_rejected(self):
        """<= 1 is not an uplift. Raise rather than substitute the default: a run
        that answers a different question than the caller asked is worse than no run."""
        for bad in (0.5, 1.0):
            with pytest.raises(ValueError, match="must be a finite value > 1"):
                Parameters(serial_uplift_at_reference_date=bad)

    def test_solver_receives_the_derived_value(self, time_series):
        """End-to-end: with the param unset, the solve targets exactly
        swe_multiplier ** parallel_penalty -- computed independently here."""
        params = Parameters()
        params.ac_anchor_mode = 'uplift-solved'
        model = ProgressModel(params, time_series)
        model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

        diagnostics = model._uplift_solve_diagnostics
        expected = params.swe_multiplier_at_present_day ** params.parallel_penalty

        assert diagnostics['target_serial_uplift'] == pytest.approx(expected, rel=1e-12)
        # and the solve actually hit it
        assert diagnostics['achieved_serial_uplift'] == pytest.approx(expected, rel=1e-9)

    # A belief deliberately AWAY from the default, so a hard-coded 2.0 serial cannot
    # pass by coincidence. The default belief is itself 2.0 now, which is exactly the
    # number the old bug hard-coded -- at the default the two are indistinguishable and
    # a literal upper bound (this test used to assert `< 1.7`) has no bite at all.
    OFF_DEFAULT_SERIAL_BELIEF = 1.5

    @pytest.mark.parametrize("belief", [None, OFF_DEFAULT_SERIAL_BELIEF])
    def test_mode_switch_is_continuous_in_present_day_uplift(self, time_series, belief):
        """The realized present-day serial uplift must not jump when the mode
        changes, and must track the CONFIGURED belief in both modes. This is the
        regression the shared default exists to prevent: with a hard-coded 2.0 it read
        2.08 in uplift-solved vs 1.35 in time-horizon (a 1.54x jump).
        """
        def realized_present_day_serial_uplift(mode):
            params = Parameters()
            params.ac_anchor_mode = mode
            if belief is not None:
                params.serial_uplift_at_reference_date = belief
            model = ProgressModel(params, time_series)
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
            times = np.asarray(model.results['times'], dtype=float)
            parallel = np.interp(
                params.present_day, times,
                np.asarray(model.results['ai_coding_labor_mult_ref_present_day'], dtype=float),
            )
            return float(parallel) ** params.parallel_penalty

        baseline = realized_present_day_serial_uplift('time-horizon')
        solved = realized_present_day_serial_uplift('uplift-solved')

        # Both modes pin the same present-day belief, so they agree tightly.
        assert solved == pytest.approx(baseline, rel=0.01), (
            f"present-day uplift jumped on mode switch: {baseline:.6f} -> {solved:.6f}"
        )
        # ...and the number they agree on is the belief, not a constant baked into the
        # solve. At belief=1.5 the old hard-coded 2.0 would miss this by 33%.
        target = belief if belief is not None else Parameters().serial_uplift_at_reference_date
        assert solved == pytest.approx(target, rel=0.02), (
            f"realized uplift {solved:.6f} does not track the configured belief {target:.6f}"
        )


class TestOptionalSolvedAsymptote:
    """serial_uplift_at_ac: OPTIONAL third condition that solves the asymptote.

    The asymptote is the one input in this mode that cannot be elicited, and it
    alone swings the uplift present at AC by ~17x. Setting serial_uplift_at_ac
    replaces it with a statement about capability at AC and solves L instead.
    Off by default.
    """

    def test_off_by_default(self):
        assert Parameters().serial_uplift_at_ac is None

    def test_default_path_is_untouched(self, time_series):
        """With the option off, the asymptote is still a plain input."""
        params = Parameters()
        params.ac_anchor_mode = 'uplift-solved'
        assert params.serial_uplift_at_ac is None
        model = ProgressModel(params, time_series)
        model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
        d = model._uplift_solve_diagnostics
        assert d.get('asymptote_was_solved', False) is False
        assert d['asymptote_L'] == pytest.approx(
            params.automation_logistic_asymptote, rel=1e-12
        )

    def test_invalid_target_disables_the_option(self):
        """<= 1 is not a valid uplift; sanitizer turns the option back off
        rather than raising or silently solving for something meaningless."""
        assert Parameters(serial_uplift_at_ac=0.5).serial_uplift_at_ac is None
        assert Parameters(serial_uplift_at_ac=1.0).serial_uplift_at_ac is None

    @pytest.mark.parametrize("target_at_ac", [15.0, 20.0, 50.0, 200.0])
    def test_solved_asymptote_reproduces_the_requested_uplift_at_ac(
        self, anchor_stats, target_at_ac
    ):
        """Solve L, then independently re-measure M at the resulting crossing."""
        params = anchor_stats['params']
        D_progress = 12.0 / 12.0 * anchor_stats['progress_rate']
        solved = solve_logistic_with_asymptote_from_uplift(
            2.0, anchor_stats['progress'], D_progress, target_at_ac,
            anchor_stats['H'], anchor_stats['C'], params,
        )
        assert solved is not None, f"failed for uplift at AC = {target_at_ac}"
        L, k, x0, diagnostics = solved
        assert L > 1.0 and k > 0

        # All three conditions must hold simultaneously, re-measured from
        # production rather than read back out of the diagnostics.
        crossing = x0 - np.log(L - 1.0) / k
        assert crossing == pytest.approx(
            diagnostics['progress_at_automation_crossing'], rel=1e-12
        )
        M_at_ac = float(serial_uplift_for_logistic(
            [crossing], k, x0, L, anchor_stats['H'], anchor_stats['C'], params
        )[0])
        assert M_at_ac == pytest.approx(target_at_ac, rel=1e-6)

        # ...and the original two conditions are still satisfied at p*
        M_ref, D_measured = _measure(k, x0, L, anchor_stats, fd=UPLIFT_SOLVE_FD_STEP)
        assert M_ref == pytest.approx(2.0, rel=1e-8)
        assert D_measured == pytest.approx(D_progress, rel=1e-6)

    def test_uplift_at_ac_is_monotone_decreasing_in_the_asymptote(self, anchor_stats):
        """The property the bracketed outer solve relies on. If this breaks, the
        1-D bracket is no longer valid and the solver needs rethinking."""
        params = anchor_stats['params']
        D_progress = anchor_stats['progress_rate']
        values = []
        for L in (1.005, 1.05, 1.1, 1.5, 3.0, 10.0):
            solved = solve_logistic_from_uplift(
                2.0, anchor_stats['progress'], D_progress, L,
                anchor_stats['H'], anchor_stats['C'], params,
            )
            assert solved is not None
            k, x0, diagnostics = solved
            values.append(float(serial_uplift_for_logistic(
                [diagnostics['progress_at_automation_crossing']], k, x0, L,
                anchor_stats['H'], anchor_stats['C'], params
            )[0]))
        assert values == sorted(values, reverse=True), (
            f"uplift at AC must decrease monotonically in the asymptote, got {values}"
        )

    def test_target_below_the_floor_is_rejected(self, anchor_stats):
        """M at AC flattens for large L, so there is a hard floor. Below it the
        solver must return None rather than clamp to the nearest asymptote."""
        params = anchor_stats['params']
        D_progress = 12.0 / 12.0 * anchor_stats['progress_rate']
        assert solve_logistic_with_asymptote_from_uplift(
            2.0, anchor_stats['progress'], D_progress, 1.5,
            anchor_stats['H'], anchor_stats['C'], params,
        ) is None

    def test_pipeline_ignores_the_input_asymptote_when_opted_in(self, time_series):
        """Two runs differing only in automation_logistic_asymptote must land on
        the same solved schedule once serial_uplift_at_ac is set."""
        def run(asymptote_input):
            params = Parameters()
            params.ac_anchor_mode = 'uplift-solved'
            params.serial_uplift_at_reference_date = 2.0
            params.serial_uplift_at_ac = 50.0
            params.automation_logistic_asymptote = asymptote_input
            model = ProgressModel(params, time_series)
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
            return model

        low, high = run(1.05), run(8.0)
        dl, dh = low._uplift_solve_diagnostics, high._uplift_solve_diagnostics
        assert dl['asymptote_was_solved'] is True
        assert dh['asymptote_L'] == pytest.approx(dl['asymptote_L'], rel=1e-9)
        assert dh['k'] == pytest.approx(dl['k'], rel=1e-9)
        assert low.results['aa_time'] == pytest.approx(high.results['aa_time'], rel=1e-9)
        # and it actually hit the requested uplift at AC
        assert dl['achieved_serial_uplift_at_ac'] == pytest.approx(50.0, rel=1e-6)

    def test_pipeline_rejects_an_unreachable_target(self, time_series):
        params = Parameters()
        params.ac_anchor_mode = 'uplift-solved'
        params.serial_uplift_at_ac = 1.5   # far below the floor
        model = ProgressModel(params, time_series)
        with pytest.raises(ValueError, match="could not find an asymptote"):
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

    def test_unreachable_target_exception_carries_the_reachable_range(
        self, time_series, monkeypatch
    ):
        """The diagnostics must travel IN the exception, not only in a log record.

        Automated callers (Monte Carlo rejection/clamping) catch this exception
        without capturing logs, so the reachable range, the uplift level, the
        doubling time and the reference date all have to be in the message.

        The expected range is not recomputed here — it is read out of the
        production solver's own failure_info, invoked with exactly the arguments
        the pipeline passed it (captured by a pass-through spy).
        """
        import progress_model._impl as impl_mod

        real_solver = impl_mod.solve_logistic_with_asymptote_from_uplift
        captured = {}

        def spy(*a, **kw):
            captured['args'] = a
            captured['kwargs'] = kw
            return real_solver(*a, **kw)

        monkeypatch.setattr(
            impl_mod, 'solve_logistic_with_asymptote_from_uplift', spy
        )

        params = Parameters()
        params.ac_anchor_mode = 'uplift-solved'
        params.serial_uplift_at_ac = 1.5   # far below the floor
        model = ProgressModel(params, time_series)
        with pytest.raises(ValueError) as excinfo:
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
        message = str(excinfo.value)

        # Re-run the real solver on the very same inputs to get its own view of
        # the reachable range, with no reimplementation of the pipeline's setup.
        assert captured, "the pipeline never reached the three-condition solver"
        kwargs = {k: v for k, v in captured['kwargs'].items() if k != 'failure_info'}
        failure_info = {}
        assert real_solver(*captured['args'], failure_info=failure_info, **kwargs) is None
        assert failure_info['reason'] == 'target_outside_reachable_range'
        lo_expected, hi_expected = failure_info['reachable']

        # Parse the range back out of the message and compare to those values.
        m = re.search(
            r"reachable range at this level and doubling time is "
            r"\[([0-9.eE+-]+), ([0-9.eE+-]+)\]x",
            message,
        )
        assert m is not None, f"no reachable range in the exception text: {message}"
        lo_parsed, hi_parsed = float(m.group(1)), float(m.group(2))
        assert lo_parsed == pytest.approx(lo_expected, rel=1e-5)
        assert hi_parsed == pytest.approx(hi_expected, rel=1e-5)

        # The range must be a real, ordered interval that actually excludes the
        # target — otherwise the caller cannot act on it.
        assert np.isfinite([lo_parsed, hi_parsed]).all()
        assert lo_parsed < hi_parsed
        assert params.serial_uplift_at_ac < lo_parsed, (
            "1.5x should be rejected for being below the floor"
        )

        # The other numbers a caller needs to interpret the range.
        target_uplift = params.serial_uplift_at_reference_date
        assert f"{target_uplift}" in message           # M* actually used
        assert f"{params.uplift_reference_date or params.present_day}" in message
        assert f"{params.excess_uplift_doubling_time}-month" in message
        assert 'target_outside_reachable_range' in message

        # The advertised range is actionable, not decorative: a target inside it
        # solves through the same pipeline and lands on the requested uplift.
        inside = float(np.sqrt(lo_parsed * hi_parsed))
        ok_params = Parameters()
        ok_params.ac_anchor_mode = 'uplift-solved'
        ok_params.serial_uplift_at_ac = inside
        ok_model = ProgressModel(ok_params, time_series)
        ok_model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
        assert ok_model._uplift_solve_diagnostics['achieved_serial_uplift_at_ac'] == (
            pytest.approx(inside, rel=1e-6)
        )
