#!/usr/bin/env python3
"""
Progress Rate Functions

This module contains functions for computing progress rates in the AI progress model,
including research effort, software progress, automation fraction, and the main
progress_rate_at_time function that serves as the RHS of the ODE system.
"""

import numpy as np
from typing import List
import logging

from . import config as cfg
from .types import TimeSeriesData
from .parameters import Parameters
from .utils import should_reraise
from .ces_functions import _ces_function,  compute_coding_labor_deprecated
from .taste_distribution import (
    TasteDistribution,
    compute_ai_research_taste as _compute_ai_research_taste_core,
)

logger = logging.getLogger(__name__)


def compute_research_effort(
    experiment_H100e,
    serial_coding_labor,
    alpha_experiment_capacity,
    rho,
    experiment_compute_exponent,
    aggregate_research_taste=cfg.AGGREGATE_RESEARCH_TASTE_BASELINE
):
    """
    CES combination of compute and cognitive work to determine research stock growth rate.
    Handles both scalar and array inputs.

    Args:
        experiment_H100e: Experiment compute budget (scalar or array)
        serial_coding_labor: Output from cognitive work (scalar or array)
        alpha_experiment_capacity: Weight on experiment compute [0,1] (scalar or array)
        rho: Standard substitution parameter in (-inf, 1] (scalar or array)
        experiment_compute_exponent: Discounting factor (scalar or array)
        aggregate_research_taste: Multiplier for research effectiveness (scalar or array, default 1.0)

    Returns:
        Research stock growth rate RS'(t) (scalar if all inputs scalar, array otherwise)
    """
    # Track if input was scalar
    input_was_scalar = (np.ndim(experiment_H100e) == 0 and np.ndim(serial_coding_labor) == 0 and
                       np.ndim(alpha_experiment_capacity) == 0 and np.ndim(rho) == 0 and
                       np.ndim(experiment_compute_exponent) == 0)

    # Ensure all inputs are arrays
    experiment_H100e = np.atleast_1d(experiment_H100e)
    serial_coding_labor = np.atleast_1d(serial_coding_labor)
    alpha_experiment_capacity = np.atleast_1d(alpha_experiment_capacity)
    rho = np.atleast_1d(rho)
    experiment_compute_exponent = np.atleast_1d(experiment_compute_exponent)

    if aggregate_research_taste is None or (np.ndim(aggregate_research_taste) == 0 and aggregate_research_taste == cfg.AGGREGATE_RESEARCH_TASTE_BASELINE):
        aggregate_research_taste = np.ones_like(experiment_H100e) * cfg.AGGREGATE_RESEARCH_TASTE_BASELINE
    else:
        aggregate_research_taste = np.atleast_1d(aggregate_research_taste)

    # Broadcast to common shape
    shape = np.broadcast_shapes(
        experiment_H100e.shape, serial_coding_labor.shape,
        alpha_experiment_capacity.shape, rho.shape,
        experiment_compute_exponent.shape, aggregate_research_taste.shape
    )

    # Broadcast all arrays
    experiment_H100e = np.broadcast_to(experiment_H100e, shape)
    serial_coding_labor = np.broadcast_to(serial_coding_labor, shape)
    alpha_experiment_capacity = np.broadcast_to(alpha_experiment_capacity, shape)
    rho = np.broadcast_to(rho, shape)
    experiment_compute_exponent = np.broadcast_to(experiment_compute_exponent, shape)
    aggregate_research_taste = np.broadcast_to(aggregate_research_taste, shape)

    # Initialize result
    result = np.zeros(shape, dtype=np.float64)

    # Input validation
    valid = (np.isfinite(experiment_H100e) &
             np.isfinite(serial_coding_labor) &
             np.isfinite(alpha_experiment_capacity) &
             np.isfinite(rho) &
             np.isfinite(experiment_compute_exponent) &
             np.isfinite(aggregate_research_taste) &
             (experiment_H100e >= 0) &
             (serial_coding_labor >= 0) &
             (aggregate_research_taste >= 0))

    if not np.any(valid):
        return float(result[0]) if input_was_scalar else result

    # Clamp experiment_compute_exponent
    ece = np.clip(experiment_compute_exponent[valid],
                  cfg.experiment_compute_exponent_CLIP_MIN,
                  cfg.experiment_compute_exponent_CLIP_MAX)

    # Clamp alpha_experiment_capacity
    alpha = np.clip(alpha_experiment_capacity[valid], 0.0, 1.0)

    # Apply discounting factor to experiment compute
    discounted_experiment_compute = np.power(experiment_H100e[valid], ece)

    # Use the CES function (now handles both scalar and array)
    rate = _ces_function(
        discounted_experiment_compute,
        serial_coding_labor[valid],
        alpha,
        rho[valid]
    )

    # Cap extremely large rates
    rate = np.clip(rate, 0, cfg.MAX_RESEARCH_EFFORT)

    # Apply aggregate research taste multiplier
    final_rate = rate * aggregate_research_taste[valid]

    # Apply final cap
    final_rate = np.clip(final_rate, 0, cfg.MAX_RESEARCH_EFFORT)

    result[valid] = final_rate

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result


def compute_software_progress_rate(research_stock, research_effort, r_software):
    """
    Compute software progress rate using research stock and labor.
    Handles both scalar and array inputs.
    S(t) = RS'(t) / RS(t) * r_software

    Args:
        research_stock: Current research stock RS(t) (scalar or array)
        research_effort: Current research stock rate RS'(t) (scalar or array)
        r_software: Software progress share parameter s [0.1,10] (scalar or array)

    Returns:
        Software progress rate (scalar if all inputs scalar, array otherwise)
    """
    # Track if input was scalar
    input_was_scalar = (np.ndim(research_stock) == 0 and np.ndim(research_effort) == 0 and
                       np.ndim(r_software) == 0)

    # Ensure all inputs are arrays
    research_stock = np.atleast_1d(research_stock)
    research_effort = np.atleast_1d(research_effort)
    r_software = np.atleast_1d(r_software)

    # Broadcast to common shape
    shape = np.broadcast_shapes(research_stock.shape, research_effort.shape, r_software.shape)
    research_stock = np.broadcast_to(research_stock, shape)
    research_effort = np.broadcast_to(research_effort, shape)
    r_software = np.broadcast_to(r_software, shape)

    # Initialize result
    result = np.zeros(shape, dtype=np.float64)

    # Input validation
    valid = (np.isfinite(research_stock) &
             np.isfinite(research_effort) &
             np.isfinite(r_software) &
             (research_stock > 0) &
             (r_software >= 0.01))

    if not np.any(valid):
        return float(result[0]) if input_was_scalar else result

    # Compute software progress rate
    software_progress_rate = research_effort[valid] / research_stock[valid]

    # Validate intermediate results
    valid_spr = np.isfinite(software_progress_rate) & (software_progress_rate >= 0)

    # Apply software progress share multiplier
    final_rate = np.zeros(np.sum(valid), dtype=np.float64)
    final_rate[valid_spr] = software_progress_rate[valid_spr] * r_software[valid][valid_spr]

    # Convert to base-for-software-lom
    rate_at_base = final_rate / np.log(cfg.BASE_FOR_SOFTWARE_LOM)

    # Final validation
    valid_final = np.isfinite(rate_at_base) & (rate_at_base >= 0)

    # Store results
    indices = np.where(valid)[0]
    result[indices[valid_final]] = rate_at_base[valid_final]

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result


def compute_overall_progress_rate(software_progress_rate: float, training_system_performance_growth_rate: float) -> float:
    """
    Sum of software and training progress rates

    Args:
        software_progress_rate: Rate from software development (already adjusted by software share)
        training_system_performance_growth_rate: Training compute budget

    Returns:
        Overall progress rate (sum of software and hardware contributions)
    """
    return software_progress_rate + training_system_performance_growth_rate


def compute_automation_fraction(cumulative_progress: float, params: Parameters) -> float:
    """
    Interpolate automation fraction based on cumulative progress.

    - If params.automation_interp_type == "exponential": perform log-space interpolation between two anchors.
    - If params.automation_interp_type == "linear": perform linear interpolation between two anchors.

    Extrapolates beyond the anchors and clips the result to [0, 1].

    Args:
        cumulative_progress: Current cumulative progress.
        params: Model parameters containing automation_anchors and automation_interp_type.

    Returns:
        Automation fraction in [0, 1].
    """
    automation_model = params.automation_model
    return automation_model.get_automation_fraction(cumulative_progress)


def compute_ai_research_taste(cumulative_progress: float, params: Parameters) -> float:
    """
    Compute AI research taste based on cumulative progress.

    Thin wrapper around _compute_ai_research_taste_core that extracts params.
    """
    return _compute_ai_research_taste_core(
        cumulative_progress=cumulative_progress,
        taste_distribution=params.taste_distribution,
        slope=params.ai_research_taste_slope,
        anchor_progress=params.progress_at_aa,
        anchor_sd=params.ai_research_taste_at_coding_automation_anchor_sd,
        max_sd=cfg.AI_RESEARCH_TASTE_MAX_SD,
    )


def compute_aggregate_research_taste(ai_research_taste, taste_distribution: TasteDistribution):
    """
    Compute aggregate research taste using a taste distribution with a clip-and-keep floor.
    Handles both scalar and array inputs.

    The research taste of individuals follows the taste_distribution. A hard floor F
    (given by `ai_research_taste`) lifts every draw below F up to F while leaving the
    upper tail unchanged, i.e. we consider Y = max(T, F).

    Args:
        ai_research_taste: Floor value(s) F for research taste (scalar or array)
        taste_distribution: Pre-initialized TasteDistribution object

    Returns:
        Mean taste after clipping, E[max(T, F)] (scalar if input scalar, array otherwise)
    """
    # Track if input was scalar
    input_was_scalar = np.ndim(ai_research_taste) == 0

    # Ensure ai_research_taste is an array
    ai_research_taste = np.atleast_1d(ai_research_taste)

    # Input validation
    valid = np.isfinite(ai_research_taste)
    result = np.full_like(ai_research_taste, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK, dtype=np.float64)

    if not np.any(valid):
        return float(result[0]) if input_was_scalar else result

    # Clamp negative values to 0
    valid_values = np.where(ai_research_taste[valid] < 0, 0.0, ai_research_taste[valid])

    try:
        # Use the get_mean_with_floor method (will be updated to handle both scalar and array)
        result[valid] = taste_distribution.get_mean_with_floor(valid_values)
    except Exception as e:
        # Don't swallow timeout/interrupt exceptions - let them propagate
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.warning(f"Error computing aggregate research taste: {e}")
        # Fallback: return max of floor and baseline for each value
        result[valid] = np.maximum(valid_values, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result


def progress_rate_at_time(t: float, state: List[float], time_series_data: TimeSeriesData, params: Parameters, RS_0: float = None,
                          boost: float = 1.0, stolen_rs_growth_rate: float = 0.0) -> List[float]:
    """
    Compute instantaneous rates for both logE and research stock.
    This is the RHS of the coupled differential equation system.

    State: [logE, research_stock] where logE = log10(cumulative effective compute).
    d(logE)/dt = 10^(EC - logE) / ln(10), where EC = TC(t) + r_software * log10(RS/RS_0).
    d(RS)/dt = research_effort.

    Args:
        t: Current time
        state: [logE, research_stock]
        time_series_data: Input time series
        params: Model parameters
        RS_0: Initial research stock (for computing software efficiency from RS)
    Returns:
        [d(logE)/dt, dRS/dt] - rates for both state variables
    """
    # Input validation
    if not np.isfinite(t):
        logger.warning(f"Non-finite time input: {t}")
        return [0.0, 0.0]

    if len(state) != 2:
        logger.warning(f"Invalid state vector length: {len(state)}, expected 2")
        return [0.0, 0.0]

    logE, research_stock = state

    model_progress = logE

    if not np.isfinite(logE):
        logger.warning(f"Invalid logE: {logE}")
        logE = 0.0

    if not np.isfinite(research_stock) or research_stock <= 0:
        logger.warning(f"Invalid research stock: {research_stock}")

    # Validate time is within reasonable bounds
    time_min, time_max = time_series_data.time.min(), time_series_data.time.max()
    if t < time_min - cfg.TIME_EXTRAPOLATION_WINDOW or t > time_max + cfg.TIME_EXTRAPOLATION_WINDOW:  # Allow some extrapolation
        logger.warning(f"Time {t} far outside data range [{time_min}, {time_max}]")

    try:
        # Interpolate time series data to time t with validation
        # Use log-space interpolation for exponentially growing variables
        # This prevents scalloping on log plots and handles exponential growth better
        # NOTE: Logs are precomputed in TimeSeriesData.__post_init__ to avoid redundant computation
        if time_series_data.can_use_log_L_HUMAN:
            L_HUMAN = np.exp(np.interp(t, time_series_data.time, time_series_data.log_L_HUMAN))
        else:
            L_HUMAN = np.interp(t, time_series_data.time, time_series_data.L_HUMAN)

        if time_series_data.can_use_log_inference_H100e:
            inference_H100e = np.exp(np.interp(t, time_series_data.time, time_series_data.log_inference_H100e))
        else:
            inference_H100e = np.interp(t, time_series_data.time, time_series_data.inference_H100e)

        if time_series_data.can_use_log_experiment_H100e:
            experiment_H100e = np.exp(np.interp(t, time_series_data.time, time_series_data.log_experiment_H100e))
        else:
            experiment_H100e = np.interp(t, time_series_data.time, time_series_data.experiment_H100e)

        training_system_performance_growth_rate = time_series_data.get_training_system_performance_growth_rate(t)

        # Validate interpolated values
        if not all(np.isfinite([L_HUMAN, inference_H100e, experiment_H100e, training_system_performance_growth_rate])):
            logger.warning(f"Non-finite interpolated values at t={t}")
            return [0.0, 0.0]

        # Ensure non-negative values
        L_HUMAN = max(0.0, L_HUMAN)
        inference_H100e = max(0.0, inference_H100e)
        experiment_H100e = max(0.0, experiment_H100e)
        training_system_performance_growth_rate = max(0.0, training_system_performance_growth_rate)

        if params.human_only:
            automation_fraction = 0.0
            aggregate_research_taste = 1.0
            coding_labor = compute_coding_labor_deprecated(None, inference_H100e, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)
        else:
            # Compute automation fraction from cumulative progress
            automation_fraction = compute_automation_fraction(model_progress, params)
            if not (0 <= automation_fraction <= 1):
                logger.warning(f"Invalid automation fraction {automation_fraction} at logE {logE}")
                automation_fraction = np.clip(automation_fraction, 0.0, 1.0)

            # Compute AI research taste and aggregate research taste
            ai_research_taste = compute_ai_research_taste(model_progress, params)
            aggregate_research_taste = compute_aggregate_research_taste(ai_research_taste, params.taste_distribution)

            # Compute cognitive output with validation
            if getattr(params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
                # Map model quantities to H, C, E (E is effective compute)
                H = float(L_HUMAN)
                C = float(inference_H100e)
                lnE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * model_progress)
                try:
                    # Build or reuse AutomationModel for frontier (uses current anchors)
                    automation_model = params.automation_model
                    L_opt = automation_model.coding_labor_optimal_ces(H, C, lnE, params)
                    # Match units with compute_coding_labor_deprecated: apply parallel_penalty and normalization
                    coding_labor = float((L_opt ** params.parallel_penalty) * params.coding_labor_normalization)
                except Exception as e:
                    if should_reraise(e):
                        raise
                    logger.warning(f"Falling back to simple CES due to optimal_ces error: {e}")
                    coding_labor = compute_coding_labor_deprecated(automation_fraction, inference_H100e, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization)
            else:
                coding_labor = compute_coding_labor_deprecated(
                    automation_fraction, inference_H100e, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
                )

        if not np.isfinite(coding_labor) or coding_labor < 0:
            logger.warning(f"Invalid cognitive output: {coding_labor}")
            return [0.0, 0.0]

        # Compute research stock rate (dRS/dt) with validation, now named research effort
        research_effort = compute_research_effort(
            experiment_H100e, coding_labor, params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, aggregate_research_taste
        )

        # S-theft: augment research effort with boost on own R&D and stolen
        # proportional growth rate applied to RS_total.
        if boost != 1.0 or stolen_rs_growth_rate != 0.0:
            research_effort = boost * research_effort + stolen_rs_growth_rate * research_stock

        if not np.isfinite(research_effort) or research_effort < 0:
            logger.warning(f"Invalid research stock rate: {research_effort}")
            return [0.0, 0.0]

        # Compute software progress rate using research stock formulation
        software_progress_rate = compute_software_progress_rate(
                research_stock, research_effort,
                params.r_software
            )


        if not np.isfinite(software_progress_rate) or software_progress_rate < 0:
            logger.warning(f"Invalid software progress rate: {software_progress_rate}")
            return [0.0, 0.0]

        # Compute d(logE)/dt
        if RS_0 is not None and RS_0 > 0:
            # Cumulative formula: d(logE)/dt = 10^(EC - logE) / ln(10)
            # where EC = TC(t) + r_software * log10(RS / RS_0)
            # training_H100e is the raw instantaneous H100e in the frontier run; convert to
            # log10(FLOP/yr) by interpolating log10(H100e) (log-space) and adding log10(FLOP_PER_H100E_YEAR).
            tc_abs = np.interp(t, time_series_data.time, time_series_data.log10_training_H100e) + np.log10(cfg.FLOP_PER_H100E_YEAR) # log10(FLOP/yr)
            se = params.r_software * np.log10(max(research_stock, 1e-30) / RS_0) # units of 2025-FLOP per FLOP
            ec = tc_abs + se # units of 2025-flop per year
            d_logE_dt = 10.0 ** (ec - logE) / np.log(10.0)
        else:
            assert False
            # Quasi-steady fallback (no RS_0)
            d_logE_dt = compute_overall_progress_rate(
                software_progress_rate, training_system_performance_growth_rate
            ) 
           

        # Final validation
        if not np.isfinite(d_logE_dt) or d_logE_dt < 0:
            logger.warning(f"Invalid d(logE)/dt: {d_logE_dt}")
            return [0.0, 0.0]

        # Cap extremely large rates to prevent numerical issues
        if d_logE_dt > cfg.MAX_NORMALIZED_PROGRESS_RATE:
            logger.warning(f"Very large d(logE)/dt {d_logE_dt}, capping to {cfg.MAX_NORMALIZED_PROGRESS_RATE}")
            d_logE_dt = cfg.MAX_NORMALIZED_PROGRESS_RATE

        logger.debug(f"t={t:.2f}, logE={logE:.3f}, research_stock={research_stock:.3f}, "
                    f"automation={automation_fraction:.3f}, d(logE)/dt={d_logE_dt:.3f}, dRS/dt={research_effort:.3f}")


        return [d_logE_dt, research_effort]

    except Exception as e:
        # Don't swallow timeout/interrupt exceptions - let them propagate
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.error(f"Error computing rates at t={t}, state={state}: {e}")
        return [0.0, 0.0]  # Return zero rates on any error
