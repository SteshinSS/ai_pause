"""
Tests for the ANCHORING side of the one-shot uplift reconstruction: that
ProgressModel.compute_progress_trajectory derives every date-anchored quantity
against the reconstruction, unconditionally, in every ac_anchor_mode.

There is no mode parameter. Everything the model anchors at a date -- the
horizon trend, progress_at_aa, the per-anchor-year slope conversions, the
swe-multiplier anchor, the r_software calibration -- used to be derived in
HUMAN-ONLY coordinates and then consumed by the with-AI trajectory, whose
present-day progress and rates are higher because AI uplift has been
accumulating historically. That was a bias, not a choice, so the corrected
mapping is simply the behavior. Fixing it exactly would mean iterating full
integrations to a fixed point; the reconstruction approximates that in one
pass, with no extra ODE solve.

The human-only mapping survives as a LIMIT, not a branch: with no uplift to
M(s) == 1, and integrate_progress_human_only with a unit multiplier IS the
human-only quadrature. TestTheNoUpliftLimit pins that bit-for-bit.

The pure function is tested in test_uplift_reconstruction.py; the composition
with ac_anchor_mode='uplift-solved' in test_uplift_solved_with_reconstruction.py.

These tests call production code and assert on values.
"""

import numpy as np
import pytest

import progress_model._impl as _impl
from progress_model import Parameters, ProgressModel
from progress_model.integration import integrate_progress_human_only

from conftest import assert_scalar_close
from utils import DEFAULT_TIME_RANGE, load_time_series_data


# How much of the human-only mapping's anchor defect the reconstruction has to
# remove. Measured at the defaults: the present-day residual goes +0.058314 ->
# -0.001984 OOMs (29x) and the horizon rebase 6.0% -> 0.20% (30x).
MAX_SHARE_OF_BASELINE = 0.2


def _run(**overrides):
    data = load_time_series_data()
    model = ProgressModel(Parameters(**overrides), data)
    model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
    return model


def _run_anchored_on_the_human_only_mapping(**overrides):
    """The pre-reconstruction behavior, for measuring what the correction buys.

    Reached through production's OWN degenerate path rather than a
    reimplementation: when reconstruct_from_params returns None, both the
    r_software calibration and the anchored derivation read the human-only run,
    which is exactly what they did before the reconstruction existed. Patching
    it to None is therefore a faithful stand-in for the old behavior, and it
    exercises the fallback at the same time.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_impl, 'reconstruct_from_params', lambda *a, **k: None)
        return _run(**overrides)


def _progress_at_present_day(model):
    return float(np.interp(model.params.present_day,
                           model.results['times'], model.results['progress']))


def _anchor_residual(model):
    """realized present-day progress - the p0 the anchors were built from."""
    return _progress_at_present_day(model) - float(model._anchor_stats_used['progress'])


@pytest.fixture(scope="module")
def model():
    return _run()


@pytest.fixture(scope="module")
def baseline():
    return _run_anchored_on_the_human_only_mapping()


@pytest.fixture(scope="module")
def horizon_model():
    """A run whose AC level comes from the horizon observable rather than the uplift
    solve. 'uplift-solved' is the default mode now, so this has to be asked for."""
    return _run(ac_anchor_mode='time-horizon')


# =============================================================================
# The anchoring reads the reconstruction
# =============================================================================


class TestTheAnchorsComeFromTheReconstruction:

    def test_the_anchor_stats_are_the_reconstruction_s(self, model):
        recon = model._uplift_reconstruction
        assert recon is not None
        assert model._anchor_stats_used is recon['anchor_stats']
        assert model._anchor_stats_used is not model.human_only_results['anchor_stats']

    def test_the_statistics_used_are_the_documented_parameters(self, horizon_model):
        """The reconstruction is built from the uplift belief -- the same triple
        ac_anchor_mode='uplift-solved' imposes on the model's own endogenous
        uplift -- regardless of what determines AC. Here AC is the horizon
        trend, which imposes no uplift condition of its own, and the statistics
        are still those."""
        model = horizon_model
        params = model.params
        assert params.ac_anchor_mode == 'time-horizon'
        used = model._uplift_reconstruction_info['statistics_used']
        assert used['serial_multiplier'] == pytest.approx(
            params.serial_uplift_at_reference_date, rel=1e-12)
        # the parameter is in anchor-MONTHS
        assert used['doubling_time_years'] == pytest.approx(
            params.excess_uplift_doubling_time / 12.0, rel=1e-12)
        assert used['eval_date'] == pytest.approx(params.present_day, rel=1e-12)

    def test_the_reported_numbers_describe_the_mapping_that_was_used(self, model):
        info = model._uplift_reconstruction_info
        assert info['available'] is True
        stats = model._anchor_stats_used
        human_only = model.human_only_results['anchor_stats']
        assert info['reconstructed_present_day_progress'] == pytest.approx(stats['progress'])
        assert info['human_only_present_day_progress'] == pytest.approx(human_only['progress'])
        assert info['present_day_progress_shift'] == pytest.approx(
            stats['progress'] - human_only['progress'])
        # the shift is upward and material: historical AI uplift really did
        # raise present-day progress above the human-only counterfactual
        assert info['present_day_progress_shift'] > 0.02
        assert info['reference_sw_rate_ratio'] > 1.0

    def test_r_software_calibrates_the_reconstructed_reference_rate(self, model):
        """The calibration target is an observed (AI-assisted) rate, so it binds
        the reconstruction's reference-year rate, not the human-only one -- and
        exactly, because the reconstruction's sw rate is linear in r_software."""
        recon = model._uplift_reconstruction
        assert_scalar_close(
            float(recon['reference_sw_progress_rate']),
            float(model.params.software_progress_rate_at_reference_year),
            rtol=1e-3,
            msg="Reconstructed reference sw rate does not hit the calibration target",
        )
        # ...and the human-only rate is consequently NOT the calibrated one
        assert abs(model.human_only_results['reference_sw_progress_rate']
                   - model.params.software_progress_rate_at_reference_year) > 1e-3

    def test_it_integrates_exactly_once(self, monkeypatch):
        """The whole point of the one-shot design: no iteration. Anchoring
        against a with-AI trajectory costs one extra vectorized human-only
        quadrature, not another ODE solve."""
        calls = {'n': 0}
        real_integrate = _impl.integrate_progress

        def counting_integrate(*args, **kwargs):
            calls['n'] += 1
            return real_integrate(*args, **kwargs)

        monkeypatch.setattr(_impl, 'integrate_progress', counting_integrate)
        _run()
        assert calls['n'] == 1, f"the model ODE was integrated {calls['n']} times"


# =============================================================================
# The human-only limit
# =============================================================================


class TestTheCorrection:
    """Baseline throughout is the human-only mapping, i.e. what this code did
    before the reconstruction existed, reached through the documented fallback.
    Absolute measurements against the trajectory being approximated live in
    test_uplift_reconstruction.py."""

    def test_present_day_anchor_residual_collapses(self, model, baseline):
        """The headline: |realized present-day progress - the p0 the anchors were
        built from|. Measured 0.058314 OOMs against the human-only mapping and
        -0.001984 against the reconstruction."""
        off_residual = abs(_anchor_residual(baseline))
        on_residual = abs(_anchor_residual(model))
        assert off_residual > 0.02, (
            f"human-only anchor residual is only {off_residual:.5f} OOMs; nothing to fix"
        )
        assert on_residual < MAX_SHARE_OF_BASELINE * off_residual, (
            f"reconstruction leaves {on_residual:.5f} OOMs of anchor residual against "
            f"{off_residual:.5f} for the human-only mapping"
        )

    # What the reconstruction may leave behind at present day, in OOMs of progress.
    # The claim being made is BIAS REMOVAL, which is a statement about magnitude: the
    # human-only mapping is systematically short by ~0.08 OOMs and the reconstruction
    # has to land near zero. This used to be spelled `residual < 0`, because at the old
    # defaults the remnant happened to overshoot. At the aligned defaults it undershoots
    # instead (+0.0032) -- while SHRINKING by a further 25x -- so the sign was reading
    # the noise, not the bias. Pinned on |residual| at roughly 2.5x the measurement.
    MAX_RECONSTRUCTION_RESIDUAL = 8.0e-03

    def test_the_residual_is_bias_free(self, model, baseline):
        """Stronger than 'smaller'. The human-only mapping under-states present-day
        progress (its counterfactual has no accumulated uplift), so its residual is
        positive by construction and LARGE. The reconstruction's is small in absolute
        terms, i.e. what is left is approximation noise rather than leftover bias --
        it does not merely lean the same way by less."""
        assert _anchor_residual(baseline) > 0.02
        assert abs(_anchor_residual(model)) < self.MAX_RECONSTRUCTION_RESIDUAL

    def test_ac_time_is_later(self, model, baseline):
        """The human-only mapping understated present-day progress, so AC was
        being reached too early."""
        aa_on = model.results['aa_time']
        aa_off = baseline.results['aa_time']
        assert aa_on is not None and aa_off is not None
        assert aa_on > aa_off, f"AC time {aa_on} is not later than {aa_off}"
        assert (aa_on - aa_off) > 0.05, (
            f"AC time shift is only {aa_on - aa_off:.4f} years ({aa_off:.4f} -> {aa_on:.4f})"
        )

    def test_progress_at_aa_is_higher_and_r_software_lower(self, model, baseline):
        """Binding an observed rate to the with-AI trajectory rather than the
        human-only one needs less software-progress leverage."""
        assert model.params.progress_at_aa > baseline.params.progress_at_aa
        assert model.params.r_software < baseline.params.r_software

    def test_the_horizon_rebase_shrinks(self, model, baseline):
        """'The horizon is present_horizon minutes today' is imposed at the
        MAPPING's present-day progress, and then a post-integration rebase
        shifts the trend to the realized progress so the displayed metric still
        says present_horizon. The size of that shift is the mismatch, made
        visible: after the rebase the trend reads present_horizon at the
        realized p0 and something else at the p0 it was fit to, and the ratio
        between the two is how much work the rebase had to do."""
        assert model.params.present_horizon is not None

        def rebase_magnitude(m):
            realized = _progress_at_present_day(m)
            anchored = float(m._anchor_stats_used['progress'])
            return abs(float(m.horizon_trajectory(realized))
                       / float(m.horizon_trajectory(anchored)) - 1.0)

        on = rebase_magnitude(model)
        off = rebase_magnitude(baseline)
        assert off > 0.05, f"human-only horizon rebase is only {off:.4%}; nothing to fix"
        assert on < MAX_SHARE_OF_BASELINE * off, (
            f"horizon rebase magnitude: reconstruction {on:.4%}, human-only {off:.4%}"
        )

    @pytest.mark.parametrize("mode", ['time-horizon', 'revenue', 'eci', 'uplift-solved'])
    def test_every_ac_anchor_mode_gets_the_corrected_anchoring(self, mode):
        """The correction is to the SHARED anchoring, so it cannot be specific to how
        AC's progress level is chosen. Every mode has to end up anchored on a
        reconstruction; HOW it gets there is now mode-dependent, and that is what the
        branch below is about.

        In the three observable modes the baseline helper no longer produces a
        human-only-anchored run at all. The T_d correction pass, which applies in every
        mode but 'uplift-solved', measures the derived schedule's own uplift statistics
        and rebuilds the reconstruction from the measured ones -- reaching
        reconstruct_uplifted_trajectory DIRECTLY, not through the
        reconstruct_from_params door the helper closes. Those modes' T_d is a belief,
        not something they pin (the schedule misses it by 27-345%), so the pass fires in
        both runs and they converge on the same mapping. Measured: AC agrees to 0.004
        years, progress_at_aa to 0.05 OOMs, and both anchor residuals sit near zero
        instead of at the ~0.08 OOMs the human-only mapping leaves. That is the
        correction arriving by a second route, not the correction failing.

        'uplift-solved' keeps the ordering claim, and is where it still bites: its
        solve pins the schedule to the belief (mismatch 0.5%), the pass does not apply
        in that mode, and the baseline really is the pre-reconstruction behavior.
        Measured AC 2027.6435 -> 2027.7824, progress_at_aa 37.5581 -> 37.7524, anchor
        residual +0.0803 -> +0.0032.
        """
        off = _run_anchored_on_the_human_only_mapping(ac_anchor_mode=mode)
        on = _run(ac_anchor_mode=mode)
        assert np.isfinite(float(on.results['aa_time']))
        # every mode shares the r_software recalibration, which happens BEFORE the
        # correction pass and so still separates the two runs in all four modes
        assert float(on.params.r_software) < float(off.params.r_software)

        if not on._td_correction_info.get('applied'):
            assert float(on.results['aa_time']) > float(off.results['aa_time'])
            assert on.params.progress_at_aa > off.params.progress_at_aa
            assert abs(_anchor_residual(on)) < MAX_SHARE_OF_BASELINE * abs(_anchor_residual(off))
            return

        assert off._td_correction_info.get('applied'), (
            "the correction pass fired for the reconstruction-anchored run but not for "
            "the baseline; they cannot be expected to agree"
        )
        assert abs(float(on.results['aa_time'])
                   - float(off.results['aa_time'])) < 5.0e-03
        assert abs(float(on.params.progress_at_aa)
                   - float(off.params.progress_at_aa)) < 5.0e-02
        # Non-vacuous: both landed on a reconstruction. The human-only mapping leaves
        # ~0.08 OOMs of residual (see the uplift-solved branch), 8x this bound.
        assert abs(_anchor_residual(on)) < 1.0e-02
        assert abs(_anchor_residual(off)) < 1.0e-02

    def test_the_correction_is_decided_by_the_mode_and_recorded(self):
        """Whether the pass applies is the AC-anchor mode, not a tolerance, and the run
        records which statistics it anchored on so a rollout can be read back.

        'uplift-solved' pins the schedule to the belief: the statistics are measured
        and recorded, and the reconstruction the run anchored on is the one built from
        the BELIEF. 'time-horizon' treats T_d as an output of the horizon anchor: the
        pass applies, the reconstruction the run anchored on was built from the
        MEASURED statistics, and the parameter the run was given is not what it used.
        The public record is what batch_rollout persists and /api/plan-a returns.
        """
        solved = _run(ac_anchor_mode='uplift-solved')
        rec = solved.results['td_correction']
        used = solved._uplift_reconstruction['statistics_used']
        assert rec['ac_anchor_mode'] == 'uplift-solved'
        assert rec['applied'] is False
        assert rec['believed'] is not None and rec['measured'] is not None
        assert np.isfinite(rec['mismatch'])
        assert_scalar_close(used['doubling_time_years'],
                            rec['believed']['excess_doubling_time_months'] / 12.0)
        assert_scalar_close(used['doubling_time_years'],
                            float(solved.params.excess_uplift_doubling_time) / 12.0)

        horizon = _run(ac_anchor_mode='time-horizon')
        rec = horizon.results['td_correction']
        used = horizon._uplift_reconstruction['statistics_used']
        assert rec['ac_anchor_mode'] == 'time-horizon'
        assert rec['applied'] is True
        assert_scalar_close(used['doubling_time_years'],
                            rec['measured']['excess_doubling_time_months'] / 12.0)
        assert_scalar_close(used['serial_multiplier'], rec['measured']['serial_multiplier'])
        # the run's own parameter is the belief, and the belief is NOT what it anchored on
        assert_scalar_close(rec['believed']['excess_doubling_time_months'],
                            float(horizon.params.excess_uplift_doubling_time))
        assert rec['measured']['excess_doubling_time_months'] != \
            rec['believed']['excess_doubling_time_months']
