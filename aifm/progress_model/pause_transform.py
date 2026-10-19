#!/usr/bin/env python3
"""
Canonical "pause" (freeze + reduce) transform for input time series.

This is the Python counterpart to the frontend `applyPause()` in
`utils/pauseTransform.ts` (which CsvEditorSidebar.tsx imports). It must produce
numerically identical output to that function for the same (base series,
pause_time, reduction factors) so that any analysis computed here (e.g. the
pause-reduction experiment) matches what the website shows.

Semantics (mirrors applyPause exactly):
  * Inputs up to AND INCLUDING the pause instant are left at their original
    (un-reduced) values. The pause instant carries the FULL value, which keeps a
    "pause just after a milestone" from moving that milestone's crossing date.
  * The reduction is applied as a SHARP STEP: a point is inserted at
    pause + 1e-6 holding the reduced value, so inputs drop essentially
    instantaneously at the pause rather than ramping toward the next grid point.
    Every channel is then held flat at its pause-instant value divided by its
    reduction factor:
      - L_HUMAN, experiment_H100e, inference_H100e: linear-space division.
      - training_H100e: linear-space division, same as the others. (It was
        log10(FLOP/yr) before the H100e port, when a cut subtracted log10(R).)
  * Reduction factors of 1.0 reduce to a pure freeze (no step point inserted).

Interpolation at the (possibly fractional) pause instant matches the frontend:
  * all four channels: exponential interpolation in linear space, which equals
    log-space linear interpolation for positive values (with a linear fallback
    when an endpoint is non-positive). training_H100e is raw H100e, so it
    interpolates like the rest.

NOTE: This intentionally does NOT replace `ProgressModel.freeze_time_series_at_time`
(_impl.py), which is a freeze-only transform with a different grid convention
wired into the full-automation scenario. Consolidating the two is deferred to
avoid perturbing existing golden trajectories.
"""

from __future__ import annotations

import numpy as np

from .types import TimeSeriesData


def _exp_interp(a: float, b: float, t: float) -> float:
    """Exponential interpolation between a and b at fractional position t in [0, 1].

    Mirrors the frontend `expInterp`: a * (b / a)**t for positive endpoints,
    with a linear fallback when either endpoint is non-positive.
    """
    if a <= 0 or b <= 0:
        return a + (b - a) * t
    return a * (b / a) ** t


def freeze_and_reduce_time_series(
    data: TimeSeriesData,
    pause_time: float,
    *,
    training_reduction: float = 1.0,
    experiment_reduction: float = 1.0,
    inference_reduction: float = 1.0,
    labor_reduction: float = 1.0,
) -> TimeSeriesData:
    """Return a new TimeSeriesData frozen (and optionally reduced) at ``pause_time``.

    Reduction factors are divisive (e.g. 10.0 means "one tenth"); 1.0 == no
    reduction. Equivalent to the frontend ``applyPause``.

    Args:
        data: Base input series (training_H100e in raw H100e).
        pause_time: Decimal year at which to freeze.
        training_reduction: Factor R for training compute (subtract log10(R)).
        experiment_reduction: Factor R for experiment compute (divide by R).
        inference_reduction: Factor R for inference compute (divide by R).
        labor_reduction: Factor R for human labor (divide by R).

    Returns:
        A new TimeSeriesData. The input ``data`` is not mutated.
    """
    time = np.asarray(data.time, dtype=float)
    n = len(time)

    # loIdx = last index with time <= pause_time (mirrors the frontend scan).
    lo_idx = 0
    for i in range(n):
        if time[i] <= pause_time:
            lo_idx = i
        else:
            break

    hi_idx = min(lo_idx + 1, n - 1)
    t_lo = time[lo_idx]
    t_hi = time[hi_idx]
    frac = (pause_time - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0

    tc = np.asarray(data.training_H100e, dtype=float)
    ec = np.asarray(data.experiment_H100e, dtype=float)
    ic = np.asarray(data.inference_H100e, dtype=float)
    lh = np.asarray(data.L_HUMAN, dtype=float)

    # Full (un-reduced) values at the pause instant. training_H100e is raw H100e
    # (exponential), like the other channels, so it interpolates in log space.
    freeze_training_full = _exp_interp(tc[lo_idx], tc[hi_idx], frac)
    freeze_experiment_full = _exp_interp(ec[lo_idx], ec[hi_idx], frac)
    freeze_inference_full = _exp_interp(ic[lo_idx], ic[hi_idx], frac)
    freeze_labor_full = _exp_interp(lh[lo_idx], lh[hi_idx], frac)

    # Reduced values held flat strictly after the pause instant.
    freeze_training = freeze_training_full / training_reduction
    freeze_experiment = freeze_experiment_full / experiment_reduction
    freeze_inference = freeze_inference_full / inference_reduction
    freeze_labor = freeze_labor_full / labor_reduction

    # Sharp step: the pause instant carries the FULL value, then inputs DROP to
    # the reduced value immediately after (a point at pause + EPS). The reduction
    # does NOT ramp in over the interval to the next grid point. EPS matches the
    # convention used by ProgressModel.freeze_time_series_at_time.
    EPS = 1e-6
    do_step = any(r != 1.0 for r in (training_reduction, experiment_reduction,
                                     inference_reduction, labor_reduction))

    out_time: list[float] = []
    out_tc: list[float] = []
    out_ec: list[float] = []
    out_ic: list[float] = []
    out_lh: list[float] = []

    def _push(t, tc_v, ec_v, ic_v, lh_v):
        out_time.append(float(t))
        out_tc.append(float(tc_v))
        out_ec.append(float(ec_v))
        out_ic.append(float(ic_v))
        out_lh.append(float(lh_v))

    # Base points strictly before the pause (unchanged).
    for i in range(n):
        if time[i] < pause_time:
            _push(time[i], tc[i], ec[i], ic[i], lh[i])
    # The pause instant carries the FULL (un-reduced) value (preserves milestone date).
    _push(pause_time, freeze_training_full, freeze_experiment_full, freeze_inference_full, freeze_labor_full)
    # Sharp drop to the reduced value immediately after the pause.
    if do_step:
        _push(pause_time + EPS, freeze_training, freeze_experiment, freeze_inference, freeze_labor)
    # Base points strictly after the pause, held flat at the reduced value.
    for i in range(n):
        if time[i] > pause_time:
            _push(time[i], freeze_training, freeze_experiment, freeze_inference, freeze_labor)

    return TimeSeriesData(
        time=np.array(out_time, dtype=float),
        L_HUMAN=np.array(out_lh, dtype=float),
        inference_H100e=np.array(out_ic, dtype=float),
        experiment_H100e=np.array(out_ec, dtype=float),
        training_H100e=np.array(out_tc, dtype=float),
    )
