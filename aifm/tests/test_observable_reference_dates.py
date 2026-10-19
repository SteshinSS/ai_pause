"""
Per-observable reference dates (horizon_reference_date, revenue_reference_date,
eci_reference_date).

Each anchor observable's value and its companion per-year rate bind at the
observable's own elicitation vintage on the anchoring mapping, following the
uplift_reference_date convention: None => present_day, bit-identical to the
historical single-date behavior.

Most tests here run against fit_horizon_trend / fit_revenue_trend /
fit_eci_trend directly with a SYNTHETIC mapping whose progress and rates are
known in closed form, so the date arithmetic can be asserted exactly. One test
runs the full model to pin the None-default bit-identity end to end.

The load-bearing invariant is trend-sliding: moving an observable's reference
date ALONG ITS OWN FITTED TREND (adjusting the value and the per-year rate to
what the trend says they are at the new date) must leave the fitted observable
— and therefore progress_at_aa — unchanged. A failure here means the anchor
date leaked into the trend's shape instead of just its parameterization.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, '..')

from progress_model import (  # noqa: E402
    Parameters,
    ProgressModel,
    fit_horizon_trend,
    fit_revenue_trend,
    fit_eci_trend,
    resolve_observable_anchor,
)


# ---------------------------------------------------------------------------
# Synthetic mapping: progress quadratic in time so the rate genuinely varies,
# making any wrong-date rate lookup produce a measurably wrong answer.
# ---------------------------------------------------------------------------
T_START = 2018.0
T_END = 2040.0
TIMES = np.linspace(T_START, T_END, 4000)
PROG = 0.45 * (TIMES - T_START) + 0.015 * (TIMES - T_START) ** 2
RATES = 0.45 + 0.03 * (TIMES - T_START)  # exact d(PROG)/dt


def _progress_at(t: float) -> float:
    return float(np.interp(t, TIMES, PROG))


def _rate_at(t: float) -> float:
    return float(np.interp(t, TIMES, RATES))


# The model's own defaults now STATE all three observable vintages (the Q2 sheet dates:
# horizon 2026.269178, revenue 2026.374429, ECI 2026.524658). This module is about the
# mechanism, not the shipped belief: its baseline is the UNDATED path, and every test
# that says "unset" means unset. Leaving the shipped dates in would silently anchor the
# baseline at a vintage of its own and turn the None-vs-present_day identities into
# comparisons of two different dates. Tests that need a date pass one explicitly.
UNDATED = dict(horizon_reference_date=None,
               revenue_reference_date=None,
               eci_reference_date=None)


def _params(**overrides) -> Parameters:
    return Parameters(**{**UNDATED, **overrides})


class TestResolveObservableAnchor:
    def test_unset_date_returns_present_day_and_the_scalar_rate_verbatim(self):
        params = _params()
        scalar = 0.7654321
        t_obs, progress, rate = resolve_observable_anchor(
            params, 'horizon_reference_date', TIMES, PROG, scalar, RATES)
        assert t_obs == float(params.present_day)
        assert progress == _progress_at(params.present_day)
        # NOT re-interpolated: the bit-identity guarantee for the default path
        assert rate == scalar

    def test_set_date_interpolates_progress_and_rate_at_that_date(self):
        date = 2027.25
        params = _params(horizon_reference_date=date)
        t_obs, progress, rate = resolve_observable_anchor(
            params, 'horizon_reference_date', TIMES, PROG, 0.111, RATES)
        assert t_obs == date
        assert progress == pytest.approx(_progress_at(date), rel=1e-14)
        assert rate == pytest.approx(_rate_at(date), rel=1e-14)

    def test_set_date_without_rate_series_falls_back_to_the_scalar(self):
        params = _params(revenue_reference_date=2027.0)
        scalar = 0.4242
        _, _, rate = resolve_observable_anchor(
            params, 'revenue_reference_date', TIMES, PROG, scalar, None)
        assert rate == scalar

    def test_non_finite_dates_are_sanitized_to_none(self):
        params = _params(
            horizon_reference_date=float('nan'),
            revenue_reference_date=float('inf'),
            eci_reference_date=float('-inf'),
        )
        assert params.horizon_reference_date is None
        assert params.revenue_reference_date is None
        assert params.eci_reference_date is None


class TestDatedAnchorPoint:
    """The anchor point actually moves to the observable's date."""

    def test_horizon_anchor_binds_at_its_own_date(self):
        date = 2025.5
        base = dict(
            horizon_extrapolation_type='decaying doubling time',
            present_horizon=32.0,
            present_doubling_time=0.4,
            doubling_difficulty_growth_factor=0.92,
        )
        obs = fit_horizon_trend(
            _params(horizon_reference_date=date, **base),
            TIMES, PROG, _rate_at(_params().present_day),
            mapping_progress_rates=RATES,
        )
        assert obs is not None
        assert obs.fit_info['anchor_progress'] == pytest.approx(
            _progress_at(date), rel=1e-14)
        # The doubling time converts at the DATE's rate, not the present-day rate
        assert obs.fit_info['T_0'] == pytest.approx(0.4 * _rate_at(date), rel=1e-14)

    def test_revenue_and_eci_anchors_bind_at_their_own_dates(self):
        date = 2027.75
        rev = fit_revenue_trend(
            _params(revenue_reference_date=date),
            TIMES, PROG, _rate_at(_params().present_day),
            mapping_progress_rates=RATES,
        )
        eci = fit_eci_trend(
            _params(eci_reference_date=date),
            TIMES, PROG, _rate_at(_params().present_day),
            mapping_progress_rates=RATES,
        )
        assert rev.fit_info['anchor_progress'] == pytest.approx(
            _progress_at(date), rel=1e-14)
        assert eci.fit_info['anchor_progress'] == pytest.approx(
            _progress_at(date), rel=1e-14)
        # Per-year rates convert at the date's own progress rate
        p = _params()
        assert rev.fit_info['log_growth_per_progress'] == pytest.approx(
            np.log(p.present_revenue_growth_rate) / _rate_at(date), rel=1e-14)
        assert eci.fit_info['points_per_progress'] == pytest.approx(
            p.eci_growth_rate / _rate_at(date), rel=1e-14)

    def test_date_equal_to_present_day_matches_unset_exactly(self):
        """Setting the date to present_day must reproduce None bit-for-bit
        (the rate series interpolates to the same scalar on the same grid)."""
        pd_rate = _rate_at(_params().present_day)
        base = dict(
            horizon_extrapolation_type='exponential',
            present_horizon=32.0,
            present_doubling_time=0.4,
        )
        obs_none = fit_horizon_trend(
            _params(**base), TIMES, PROG, pd_rate, mapping_progress_rates=RATES)
        obs_dated = fit_horizon_trend(
            _params(horizon_reference_date=float(_params().present_day), **base),
            TIMES, PROG, pd_rate, mapping_progress_rates=RATES)
        probe = np.linspace(PROG[0], PROG[-1], 50)
        np.testing.assert_array_equal(obs_none.predict(probe), obs_dated.predict(probe))
        target = float(_params().ac_time_horizon_minutes)
        assert obs_none.invert(target) == obs_dated.invert(target)


class TestTrendSlidingInvariance:
    """Sliding the reference date along the fitted trend itself — restating the
    value and per-year rate as what the trend predicts at the new date — must
    leave the fitted observable unchanged."""

    def test_exponential_horizon(self):
        p0 = _params()
        t_base = float(p0.present_day)
        t_new = t_base + 0.75

        horizon_base = 32.0
        doubling_years_base = 0.4
        base = dict(horizon_extrapolation_type='exponential',
                    present_horizon=horizon_base,
                    present_doubling_time=doubling_years_base)

        obs_base = fit_horizon_trend(
            _params(**base), TIMES, PROG, _rate_at(t_base),
            mapping_progress_rates=RATES)

        # Restate the anchor at t_new per the base trend:
        # slope (per progress unit) is the trend's invariant
        t_prog_units = doubling_years_base * _rate_at(t_base)
        slope = np.log(2.0) / t_prog_units
        dp = _progress_at(t_new) - _progress_at(t_base)
        horizon_new = horizon_base * np.exp(slope * dp)
        doubling_years_new = t_prog_units / _rate_at(t_new)

        obs_slid = fit_horizon_trend(
            _params(horizon_reference_date=t_new,
                    horizon_extrapolation_type='exponential',
                    present_horizon=horizon_new,
                    present_doubling_time=doubling_years_new),
            TIMES, PROG, _rate_at(t_base),
            mapping_progress_rates=RATES)

        target = float(p0.ac_time_horizon_minutes)
        assert obs_slid.invert(target) == pytest.approx(
            obs_base.invert(target), rel=1e-10)
        probe = np.linspace(PROG[0], PROG[-1], 50)
        np.testing.assert_allclose(
            obs_slid.predict(probe), obs_base.predict(probe), rtol=1e-10)

    def test_revenue(self):
        p0 = _params()
        t_base = float(p0.present_day)
        t_new = t_base - 0.5  # backwards too

        obs_base = fit_revenue_trend(
            p0, TIMES, PROG, _rate_at(t_base), mapping_progress_rates=RATES)

        log_growth_per_progress = obs_base.fit_info['log_growth_per_progress']
        revenue_new = float(obs_base.predict(_progress_at(t_new)))
        growth_new = float(np.exp(log_growth_per_progress * _rate_at(t_new)))

        obs_slid = fit_revenue_trend(
            _params(revenue_reference_date=t_new,
                    present_revenue=revenue_new,
                    present_revenue_growth_rate=growth_new),
            TIMES, PROG, _rate_at(t_base), mapping_progress_rates=RATES)

        target = float(p0.revenue_required_for_ac)
        assert obs_slid.invert(target) == pytest.approx(
            obs_base.invert(target), rel=1e-10)

    def test_eci(self):
        p0 = _params()
        t_base = float(p0.present_day)
        t_new = t_base + 1.25

        obs_base = fit_eci_trend(
            p0, TIMES, PROG, _rate_at(t_base), mapping_progress_rates=RATES)

        points_per_progress = obs_base.fit_info['points_per_progress']
        eci_new = float(obs_base.predict(_progress_at(t_new)))
        eci_rate_new = float(points_per_progress * _rate_at(t_new))

        obs_slid = fit_eci_trend(
            _params(eci_reference_date=t_new,
                    eci_at_present_day=eci_new,
                    eci_growth_rate=eci_rate_new),
            TIMES, PROG, _rate_at(t_base), mapping_progress_rates=RATES)

        target = float(p0.eci_required_for_ac)
        assert obs_slid.invert(target) == pytest.approx(
            obs_base.invert(target), rel=1e-10)


class TestFullModelDefaultIdentity:
    """End to end: leaving the dates unset must not move the model at all, and
    setting them to present_day must be bit-identical to unset."""

    def test_dates_at_present_day_are_bit_identical_to_unset(self):
        from utils import load_time_series_data
        from conftest import DEFAULT_TIME_RANGE

        data = load_time_series_data()

        params_unset = Parameters()
        model_unset = ProgressModel(params_unset, data)
        model_unset.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

        pd = float(Parameters().present_day)
        params_dated = Parameters(
            horizon_reference_date=pd,
            revenue_reference_date=pd,
            eci_reference_date=pd,
        )
        model_dated = ProgressModel(params_dated, data)
        model_dated.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

        assert float(params_dated.progress_at_aa) == float(params_unset.progress_at_aa)
        np.testing.assert_array_equal(
            np.asarray(model_dated.results['progress']),
            np.asarray(model_unset.results['progress']))
        np.testing.assert_array_equal(
            np.asarray(model_dated.results['times']),
            np.asarray(model_unset.results['times']))


class TestPostIntegrationRebaseUsesEachObservablesOwnDate:
    """The rebase onto the realized trajectory must re-anchor each observable at ITS
    OWN vintage.

    The fit binds an observable at mapping(t_obs); after integration the anchor is
    moved onto the realized trajectory, and it has to land at realized(t_obs) — not
    realized(present_day). Getting this wrong slides a Q2 horizon anchor by the
    progress accrued between 2026.269 and 2026.578 (order 0.4-0.5 OOM), which tilts
    the drawn trend and corrupts compute_metr_mse.

    The observable identity these assert is exact: the shifted horizon form is
    H_0 * (1 - ...)^k with the bracket vanishing at the anchor, and the revenue form
    is present_revenue * exp(k * (p - anchor)), so predict(anchor) reproduces the
    stated level to the last bit.
    """

    @staticmethod
    def _run(**overrides):
        from utils import load_time_series_data
        from conftest import DEFAULT_TIME_RANGE

        params = Parameters(**overrides)
        model = ProgressModel(params, load_time_series_data())
        model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
        times = np.asarray(model.results['times'], dtype=float)
        progress = np.asarray(model.results['progress'], dtype=float)
        return model, params, times, progress

    def test_dated_horizon_is_rebased_at_its_own_date_not_present_day(self):
        t_horizon = 2026.269178
        present_horizon = 186.0
        model, params, times, progress = self._run(
            present_day=2026.578082,
            horizon_reference_date=t_horizon,
            present_horizon=present_horizon,
            horizon_extrapolation_type='decaying doubling time',
        )

        observable = model._horizon_observable
        assert observable is not None
        assert observable.fit_info.get('uses_shifted_form') is True

        realized_at_horizon_date = float(np.interp(t_horizon, times, progress))
        realized_at_present_day = float(np.interp(params.present_day, times, progress))

        # The anchor landed on the realized trajectory at the horizon's OWN date.
        assert observable.fit_info['anchor_progress'] == pytest.approx(
            realized_at_horizon_date, rel=1e-12)
        # And the stated level is reproduced exactly there.
        assert float(observable.predict(realized_at_horizon_date)) == pytest.approx(
            present_horizon, rel=1e-9)

        # The two dates are far enough apart to be a real error if confused, and the
        # trend does NOT pass through present_horizon at present_day's progress.
        assert realized_at_present_day - realized_at_horizon_date > 0.05
        assert float(observable.predict(realized_at_present_day)) > present_horizon * 1.05

    def test_dated_revenue_is_rebased_at_its_own_date_not_present_day(self):
        t_revenue = 2026.374429
        present_revenue = 4.7e10
        model, params, times, progress = self._run(
            present_day=2026.578082,
            ac_anchor_mode='revenue',
            revenue_reference_date=t_revenue,
            present_revenue=present_revenue,
            present_revenue_growth_rate=5.0,
        )

        observable = model._revenue_observable
        assert observable is not None

        realized_at_revenue_date = float(np.interp(t_revenue, times, progress))
        realized_at_present_day = float(np.interp(params.present_day, times, progress))

        assert observable.fit_info['anchor_progress'] == pytest.approx(
            realized_at_revenue_date, rel=1e-12)
        assert float(observable.predict(realized_at_revenue_date)) == pytest.approx(
            present_revenue, rel=1e-9)

        # Revenue is growing fast; anchoring at the wrong date would be a visible
        # multiple, not a rounding difference.
        assert realized_at_present_day - realized_at_revenue_date > 0.02
        assert float(observable.predict(realized_at_present_day)) > present_revenue * 1.05

    def test_two_observables_with_different_dates_each_get_their_own(self):
        """The whole point of the map: one run, two vintages, no cross-talk."""
        t_horizon, t_revenue = 2026.269178, 2026.374429
        model, params, times, progress = self._run(
            present_day=2026.578082,
            horizon_reference_date=t_horizon,
            revenue_reference_date=t_revenue,
            present_horizon=186.0,
            horizon_extrapolation_type='decaying doubling time',
        )

        assert model._horizon_observable.fit_info['anchor_progress'] == pytest.approx(
            float(np.interp(t_horizon, times, progress)), rel=1e-12)
        assert model._revenue_observable.fit_info['anchor_progress'] == pytest.approx(
            float(np.interp(t_revenue, times, progress)), rel=1e-12)
        # Distinct dates must give distinct anchors, or the test above would pass
        # vacuously under a present_day-for-everything regression.
        assert (model._horizon_observable.fit_info['anchor_progress']
                != model._revenue_observable.fit_info['anchor_progress'])

    def test_undated_observables_still_rebase_at_present_day_exactly(self):
        """The bit-identity guarantee: with no dates set, every anchor is the
        present-day interpolation, to the last bit — not merely close to it."""
        model, params, times, progress = self._run(**UNDATED)

        expected = np.interp(params.present_day, times, progress)
        for attr in ('_horizon_observable', '_eci_observable', '_revenue_observable'):
            observable = getattr(model, attr)
            if observable is None or not observable.fit_info.get('uses_shifted_form'):
                continue
            assert observable.fit_info['anchor_progress'] == float(expected), attr


class TestEciVelocitySeries:
    """`capability_eci_rates`: d(ECI)/dt in points per year.

    The ECI trend is exactly linear in progress, so its time derivative is analytic --
    points_per_progress * d(progress)/dt -- and that is how the series is built, rather
    than by differencing `capability_ecis`. These tests pin the two properties that
    justify the choice: it is exactly the plotted curve's slope, and it reduces to
    eci_growth_rate at the ECI reference date by construction.
    """

    @staticmethod
    def _run(**overrides):
        from utils import load_time_series_data
        from conftest import DEFAULT_TIME_RANGE

        params = Parameters(**overrides)
        model = ProgressModel(params, load_time_series_data())
        model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
        return model, params

    def test_the_series_is_exactly_points_per_progress_times_the_progress_rate(self):
        model, _ = self._run(ac_anchor_mode='eci', eci_at_present_day=162.0, eci_growth_rate=15.0)
        rates = np.asarray(model.results['capability_eci_rates'], dtype=float)
        progress_rates = np.asarray(model.results['progress_rates'], dtype=float)
        points_per_progress = model._eci_observable.fit_info['points_per_progress']

        assert rates.shape == progress_rates.shape
        np.testing.assert_allclose(rates, points_per_progress * progress_rates, rtol=0, atol=0)

    def test_integrating_the_velocity_reproduces_the_plotted_eci_curve(self):
        """This is the velocity OF THE DRAWN LINE, not of a parallel calculation:
        integrating it over the run must recover the curve's total rise.

        Checked by integration rather than by differencing capability_ecis, because on
        the 0.1-year output grid a centered difference is off by up to ~10% wherever the
        progress rate accelerates -- which is exactly why the series is computed
        analytically instead of by differencing in the first place.
        """
        model, _ = self._run(ac_anchor_mode='eci', eci_at_present_day=162.0, eci_growth_rate=15.0)
        times = np.asarray(model.results['times'], dtype=float)
        ecis = np.asarray(model.results['capability_ecis'], dtype=float)
        rates = np.asarray(model.results['capability_eci_rates'], dtype=float)

        integrated = float(np.trapezoid(rates, times))
        total_rise = float(ecis[-1] - ecis[0])
        assert integrated == pytest.approx(total_rise, rel=5e-3)
        # Non-vacuous: the curve rises by hundreds of points over the run, so agreement
        # to 0.5% is a real constraint rather than two small numbers matching.
        assert total_rise > 100.0

    def test_at_the_eci_reference_date_the_velocity_is_eci_growth_rate(self):
        """The construction identity, on the synthetic mapping where the progress rate at
        any date is known in closed form.

        velocity(t) = points_per_progress * rate(t), and points_per_progress is defined as
        eci_growth_rate / rate(t_obs), so velocity(t_obs) == eci_growth_rate exactly. The
        rate here comes from _rate_at, independently of anything the fit returned, so this
        is not a tautology over the fit's own numbers.
        """
        growth_rate = 15.0
        params = _params(eci_at_present_day=162.0, eci_growth_rate=growth_rate)
        t_obs = float(params.present_day)
        obs = fit_eci_trend(params, TIMES, PROG, _rate_at(t_obs), mapping_progress_rates=RATES)

        velocity_at_t_obs = obs.fit_info['points_per_progress'] * _rate_at(t_obs)
        assert velocity_at_t_obs == pytest.approx(growth_rate, rel=1e-12)

    def test_a_dated_eci_reference_binds_the_growth_rate_at_that_date(self):
        """With eci_reference_date set the identity must hold at THAT date and nowhere
        else -- the same vintage discipline the trends themselves follow."""
        growth_rate = 12.0
        t_eci = 2027.25
        params = _params(
            eci_reference_date=t_eci, eci_at_present_day=162.0, eci_growth_rate=growth_rate)
        obs = fit_eci_trend(
            params, TIMES, PROG, _rate_at(float(params.present_day)),
            mapping_progress_rates=RATES)

        points_per_progress = obs.fit_info['points_per_progress']
        assert points_per_progress * _rate_at(t_eci) == pytest.approx(growth_rate, rel=1e-12)
        # And NOT at present_day: the synthetic mapping's rate genuinely varies, so a
        # date mix-up would show up here rather than passing silently.
        assert points_per_progress * _rate_at(float(params.present_day)) != pytest.approx(
            growth_rate, rel=1e-6)

    def test_the_realized_velocity_near_the_reference_date_is_close_to_the_setting(self):
        """On the REALIZED trajectory the identity holds only up to the gap between the
        anchoring mapping's rate and the realized one -- the reconstruction residual.
        Pin that it is a small discrepancy rather than a different number."""
        growth_rate = 15.0
        model, params = self._run(
            ac_anchor_mode='eci', eci_at_present_day=162.0, eci_growth_rate=growth_rate)
        times = np.asarray(model.results['times'], dtype=float)
        rates = np.asarray(model.results['capability_eci_rates'], dtype=float)
        realized = float(np.interp(float(params.present_day), times, rates))
        assert realized == pytest.approx(growth_rate, rel=0.05)
