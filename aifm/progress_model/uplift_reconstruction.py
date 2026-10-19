#!/usr/bin/env python3
"""
One-shot reconstruction of the realized (with-AI) trajectory from the human-only
run plus TWO summary statistics.

WHY THIS EXISTS
---------------
Every date-anchored input of the model -- "the horizon is X minutes today", the
doubling times, the taste and coding-automation-efficiency slopes (per
anchor-YEAR), the swe-multiplier anchor, the r_software calibration -- is
derived against the HUMAN-ONLY counterfactual's date -> progress mapping and its
rates, and is then consumed by the REAL with-AI trajectory, whose present-day
progress and rates are higher because AI uplift has been accumulating
historically. That mismatch is a real bias (~0.16 OOM of present-day progress
under the old defaults).

Solving it exactly would mean iterating full integrations to a fixed point --
re-deriving the anchors against the trajectory they produce until they hold on
it, at ~9 extra ODE solves per run. This module is the cheap alternative:
reconstruct a close approximation of the realized trajectory in ONE pass, with
no ODE solves and no CES frontier evaluations, and anchor against that instead.

THE APPROXIMATION
-----------------
Information set: the human-only run, plus the serial coding-labor multiplier M
at one evaluation date and the instantaneous doubling time T_d of its excess
(M - 1) at that date. From those:

  1. extrapolate the excess exponentially in TIME,
         M(s) - 1 = (M(t_eval) - 1) * 2^((s - t_eval) / T_d),
     so the whole history of coding-labor uplift is an exogenous function of
     time (it never reads the trajectory back, which is what keeps this
     one-shot);
  2. multiply the human-only serial coding labor by M(s) and re-run the
     human-only quadrature. The CES experiment-capacity aggregator turns the
     labor uplift into a research-effort uplift, the research stock
     re-accumulates that uplifted effort (dRS/dt = RE), the software term
     r_software * log10(RS/RS_0) rebuilds ETSP = TC + SE, and cumulative
     effective compute is reintegrated from dE/dt = 10^ETSP.

Step 2 is literally `integrate_progress_human_only` with its optional
`serial_labor_multiplier_fn` hook, so the reconstruction inherits that
function's 300x-refined internal grid and its exact quadrature rather than
re-deriving the dynamics with a coarser scheme. With M == 1 it reproduces the
human-only trajectory bit for bit.

WHAT IT IS NOT
--------------
It is NOT a with-AI run. Automation fraction and aggregate research taste stay
at their human-only values; all of the AI feedback enters through the single
exogenous M(s) path. The reconstruction has no idea what the automation model
would say. That is the point -- it is the input to deriving the automation
model, not an output of it.

Measured accuracy at the default parameters (see tests/test_uplift_reconstruction.py):
the reconstructed d(progress)/dt tracks the truly-integrated one to well under
1% across the historical window that the anchoring actually reads.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from . import config as cfg
from .integration import integrate_progress_human_only
from .utils import _log_interp

logger = logging.getLogger(__name__)

LN10 = np.log(10.0)

# Ceiling on the extrapolated serial multiplier. The backward extrapolation is
# harmless (the excess decays), but forward of the evaluation date the excess
# doubles every T_d forever and a 35-to-185 year grid would overflow. The model
# already carries a ceiling for exactly this quantity, so reuse it rather than
# inventing a second one.
_FALLBACK_MAX_SERIAL_MULTIPLIER = 1e8


def uplift_statistics(params) -> Optional[Dict[str, float]]:
    """
    Source the two statistics the reconstruction consumes from `params`.

    They are exactly the two the 'uplift-solved' AC-anchor mode imposes on the
    model's own endogenous uplift, read at the same reference date:

        M            = params.serial_uplift_at_reference_date
                       (serial_uplift_at_reference_date, or the present-day
                       belief swe_multiplier_at_present_day ** parallel_penalty
                       when that is None)
        T_d          = excess_uplift_doubling_time, in anchor-MONTHS
        t_eval       = uplift_reference_date, or present_day when None

    Sharing the source is what makes the two modes compose: the reconstruction
    is built from statistics that are exogenous in TIME, so it can be built
    before the solve, and the solve then imposes the same level and doubling
    rate on the endogenous uplift at the same date -- against the
    reconstruction's own mapping. The two therefore agree at t* by
    construction, up to the reconstruction's approximation error, rather than
    disagreeing by the historical human-only/realized anchor gap.

    Deliberately isolated in one small function so the mapping from parameters
    to statistics has exactly one definition.

    Returns None -- meaning "no usable uplift statistics, fall back to the
    human-only mapping" -- rather than raising, so a degenerate parameter set
    degrades to the existing behavior instead of failing the run.

    Returns:
        {'serial_multiplier': M at the evaluation date (>= 1),
         'doubling_time_years': T_d of (M - 1), in years,
         'eval_date': the date both statistics describe}
    """
    # Imported here rather than at module scope: parameters.py does not import
    # this module, but keeping the edge local documents that this is the only
    # place the reconstruction reads the uplift belief.

    doubling_time_months = getattr(params, 'excess_uplift_doubling_time', None)
    eval_date = getattr(params, 'uplift_reference_date', None)
    if eval_date is None:
        eval_date = getattr(params, 'present_day', None)

    try:
        serial_multiplier = float(params.serial_uplift_at_reference_date)
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning(f"Uplift reconstruction could not resolve the serial uplift level: {e}")
        return None

    if doubling_time_months is None:
        logger.warning("Uplift reconstruction needs excess_uplift_doubling_time")
        return None
    if not (np.isfinite(serial_multiplier) and serial_multiplier > 1.0):
        logger.warning(f"Uplift reconstruction needs a serial uplift level > 1, "
                       f"got {serial_multiplier}")
        return None
    if not (np.isfinite(doubling_time_months) and doubling_time_months > 0):
        logger.warning(f"Invalid excess_uplift_doubling_time: {doubling_time_months}")
        return None
    if eval_date is None or not np.isfinite(eval_date):
        logger.warning(f"Invalid uplift reference date for uplift reconstruction: {eval_date}")
        return None

    return {
        # already SERIAL: the resolver applies ^parallel_penalty when it falls
        # back to the parallel-space present-day belief, and
        # serial_uplift_at_reference_date is serial by definition.
        'serial_multiplier': serial_multiplier,
        # the parameter is in anchor-MONTHS
        'doubling_time_years': float(doubling_time_months) / 12.0,
        'eval_date': float(eval_date),
    }


def serial_multiplier_path(
    times,
    serial_multiplier: float,
    doubling_time_years: float,
    eval_date: float,
    max_multiplier: Optional[float] = None,
) -> np.ndarray:
    """
    M(s) = 1 + (M_eval - 1) * 2^((s - eval_date) / T_d), capped at max_multiplier.

    Evaluated through the exponent so a far-future grid point cannot overflow
    before the cap is applied.
    """
    if max_multiplier is None or not (np.isfinite(max_multiplier) and max_multiplier > 1.0):
        max_multiplier = _FALLBACK_MAX_SERIAL_MULTIPLIER

    times = np.asarray(times, dtype=float)
    excess = float(serial_multiplier) - 1.0
    if excess <= 0.0:
        return np.ones_like(times)

    max_excess = float(max_multiplier) - 1.0
    log2_excess = np.log2(excess) + (times - float(eval_date)) / float(doubling_time_years)
    log2_excess = np.minimum(log2_excess, np.log2(max_excess))
    return 1.0 + np.exp2(log2_excess)


def reconstruct_uplifted_trajectory(
    human_only_results: Dict[str, Any],
    data,
    params,
    serial_multiplier: float,
    doubling_time_years: float,
    eval_date: float,
    anchor_date: Optional[float] = None,
    reference_year: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Reconstruct the with-AI trajectory from the human-only run + (M, T_d, t_eval).

    Args:
        human_only_results: the dict ProgressModel.compute_human_only_trajectory
            stores on self.human_only_results. Supplies the grid (its 'times'),
            the logE initial condition ('progress'[0]) and the initial research
            stock ('research_stock'[0]) so the reconstruction lives in exactly
            the same coordinates as the run it corrects.
        data: TimeSeriesData (exogenous inputs).
        params: model Parameters. r_software, the CES experiment-capacity
            parameters, parallel_penalty and coding_labor_normalization are read
            from here; nothing on params is mutated.
        serial_multiplier: M at eval_date (serial coding-labor multiplier).
        doubling_time_years: doubling time of (M - 1) at eval_date, in YEARS.
        eval_date: the date those two statistics describe.
        anchor_date: date to report anchor stats at (default params.present_day).
        reference_year: year the r_software calibration target refers to
            (default cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR).

    Returns:
        A dict shaped like human_only_results (same keys for the series), plus
        'serial_multipliers' and an 'anchor_stats' entry carrying the extra
        'reference_sw_progress_rate' key, so it can be substituted for
        human_only_results wherever compute_progress_trajectory derives a
        date-anchored quantity.
    """
    if anchor_date is None:
        anchor_date = float(params.present_day)
    if reference_year is None:
        reference_year = float(cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR)

    ho_times = np.asarray(human_only_results['times'], dtype=float)
    time_range = [float(ho_times[0]), float(ho_times[-1])]
    initial_logE = float(np.asarray(human_only_results['progress'], dtype=float)[0])
    RS_0 = float(np.asarray(human_only_results['research_stock'], dtype=float)[0])

    max_multiplier = getattr(params, 'max_serial_coding_labor_multiplier', None)

    def multiplier_fn(t):
        return serial_multiplier_path(
            t, serial_multiplier, doubling_time_years, eval_date, max_multiplier)

    # Same quadrature, same 300x internal grid, same initial conditions as the
    # human-only run -- only the serial coding labor is uplifted. With M == 1
    # this returns the human-only arrays unchanged.
    times, progress_values, research_stock_values, research_efforts, sw_progress_rates = (
        integrate_progress_human_only(
            time_range, initial_logE, RS_0, data, params,
            serial_labor_multiplier_fn=multiplier_fn,
        )
    )

    # Rate metrics, exactly as compute_human_only_trajectory derives them:
    #   etsp_rates     = d(ETSP)/dt = sw + TSP growth
    #   progress_rates = d(progress)/dt = 10^(ETSP - logE)/ln10 (the TRUE rate)
    tsp_growth = data.get_training_system_performance_growth_rate(times)
    etsp_rates = sw_progress_rates + tsp_growth
    tc_abs = np.interp(times, data.time, data.log10_training_H100e) + np.log10(cfg.FLOP_PER_H100E_YEAR)
    se = params.r_software * np.log10(np.maximum(research_stock_values, 1e-30) / RS_0)
    etsp = tc_abs + se
    progress_rates = np.power(10.0, etsp - progress_values) / LN10

    multipliers = serial_multiplier_path(
        times, serial_multiplier, doubling_time_years, eval_date, max_multiplier)

    # Anchor stats, in the shape compute_progress_trajectory's anchored
    # derivation consumes (the human-only run's 'anchor_stats' dict).
    anchor_progress = float(np.interp(anchor_date, times, progress_values))
    anchor_progress_rate = float(np.interp(anchor_date, times, progress_rates))
    anchor_etsp_rate = float(np.interp(anchor_date, times, etsp_rates))
    anchor_sw_rate = float(np.interp(anchor_date, times, sw_progress_rates))
    anchor_research_effort = float(_log_interp(anchor_date, times, np.asarray(research_efforts)))
    anchor_research_stock = float(_log_interp(anchor_date, times, np.asarray(research_stock_values)))
    reference_sw_rate = float(np.interp(reference_year, times, sw_progress_rates))

    exogenous = {}
    for key, series in (('human_labor', data.L_HUMAN),
                        ('inference_H100e', data.inference_H100e),
                        ('experiment_H100e', data.experiment_H100e)):
        if np.all(series > 0):
            exogenous[key] = float(_log_interp(anchor_date, data.time, series))
        else:
            exogenous[key] = float(np.interp(anchor_date, data.time, series))

    anchor_stats = {
        'progress': anchor_progress,
        'progress_rate': anchor_progress_rate,   # TRUE d(progress)/dt
        'etsp_rate': anchor_etsp_rate,
        'sw_progress_rate': anchor_sw_rate,
        'experiment_H100e': exogenous['experiment_H100e'],
        'human_labor': exogenous['human_labor'],
        'inference_H100e': exogenous['inference_H100e'],
        'research_effort': anchor_research_effort,
        'research_stock': anchor_research_stock,
        # not part of the human-only anchor_stats dict, which carries it one
        # level up in human_only_results; the r_software rescale needs it and
        # the top-level key below is the one that rescale reads
        'reference_sw_progress_rate': reference_sw_rate,
    }

    return {
        'times': times,
        'progress': progress_values,
        'research_stock': research_stock_values,
        'research_efforts': research_efforts,
        'sw_progress_rates': sw_progress_rates,
        'etsp_rates': etsp_rates,
        'progress_rates': progress_rates,
        'serial_multipliers': multipliers,
        'reference_sw_progress_rate': reference_sw_rate,
        'anchor_stats': anchor_stats,
        'statistics_used': {
            'serial_multiplier': float(serial_multiplier),
            'doubling_time_years': float(doubling_time_years),
            'eval_date': float(eval_date),
        },
    }


def reconstruct_from_params(
    human_only_results: Dict[str, Any],
    data,
    params,
    anchor_date: Optional[float] = None,
    reference_year: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    `uplift_statistics` + `reconstruct_uplifted_trajectory`, or None if the
    parameters do not yield usable statistics.
    """
    stats = uplift_statistics(params)
    if stats is None:
        return None
    return reconstruct_uplifted_trajectory(
        human_only_results, data, params,
        serial_multiplier=stats['serial_multiplier'],
        doubling_time_years=stats['doubling_time_years'],
        eval_date=stats['eval_date'],
        anchor_date=anchor_date,
        reference_year=reference_year,
    )


__all__ = [
    'uplift_statistics',
    'serial_multiplier_path',
    'reconstruct_uplifted_trajectory',
    'reconstruct_from_params',
]
