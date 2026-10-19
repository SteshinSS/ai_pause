#!/usr/bin/env python3
"""
ODE Integration Functions

This module contains functions for integrating the progress model differential
equations, including initial condition computation and the main ODE solver.
"""

import numpy as np
from scipy import integrate
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import logging

from . import config as cfg
from .types import TimeSeriesData, InitialConditions
from .parameters import Parameters
from .utils import _log_interp, should_reraise
from .ces_functions import compute_coding_labor_deprecated
from .progress_rate import (
    compute_research_effort,
    compute_software_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    progress_rate_at_time,
)

logger = logging.getLogger(__name__)

_ode_diagnostics_collector: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "_ode_diagnostics_collector",
    default=None,
)


@contextmanager
def collect_ode_diagnostics() -> Iterator[List[Dict[str, Any]]]:
    """
    Collect per-`solve_ivp` diagnostics for the duration of the context.

    Returns a mutable list that will be appended to by `integrate_progress`.
    """
    token = _ode_diagnostics_collector.set([])
    try:
        yield _ode_diagnostics_collector.get()
    finally:
        _ode_diagnostics_collector.reset(token)


def _find_exponential_crossing_time(
    times: np.ndarray,
    values: np.ndarray,
    target: float,
) -> Optional[float]:
    """
    Return the earliest time at which an exponentially growing series crosses a target.

    Assumes the series between adjacent samples grows exponentially. When the target lies
    between two samples (t0, v0) and (t1, v1), the crossing time t* is computed in log-space:

        f = (ln(target) - ln(v0)) / (ln(v1) - ln(v0))
        t* = t0 + f * (t1 - t0)

    Falls back to linear interpolation if non-positive values prevent log-space math.

    Args:
        times: 1-D array of time points (monotonic ascending recommended)
        values: 1-D array of series values at those times
        target: Threshold to cross (typically > 0 for exponential interpolation)

    Returns:
        The crossing time as float, or None if no crossing occurs or inputs invalid.
    """
    try:
        t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        y = float(target)

        if t.size == 0 or t.size != v.size or not np.isfinite(y):
            return None

        # Find the first index j where v[j] >= target
        crossing_indices = np.where(v >= y)[0]
        if crossing_indices.size == 0:
            return None

        j = int(crossing_indices[0])
        if j == 0:
            return float(t[0])

        v0 = v[j - 1]
        v1 = v[j]
        t0 = t[j - 1]
        t1 = t[j]

        if (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(t0) and np.isfinite(t1) and t1 != t0):
            # Prefer log-space interpolation if possible
            if v0 > 0 and v1 > 0 and y > 0 and v1 != v0:
                frac = (np.log(y) - np.log(v0)) / (np.log(v1) - np.log(v0))
                frac = float(min(max(frac, 0.0), 1.0))
                return float(t0 + frac * (t1 - t0))
            # Fallback to linear interpolation
            if v1 != v0:
                frac = (y - v0) / (v1 - v0)
                frac = float(min(max(frac, 0.0), 1.0))
                return float(t0 + frac * (t1 - t0))
            # If values are equal, step to the right edge
            return float(t1)

        return None
    except Exception as e:
        if should_reraise(e):
            raise
        return None


def calculate_initial_research_stock(time_series_data: TimeSeriesData, params: Parameters,
                                   initial_progress: float = 0.0) -> float:
    """
    Calculate initial research stock using the formula: RS(0) = (RS'(0))^2 / RS''(0)

    Args:
        time_series_data: Input time series data
        params: Model parameters
        initial_progress: Initial cumulative progress

    Returns:
        Calculated initial research stock value
    """
    try:
        start_time = time_series_data.time[0]
        dt = 1e-6  # Small time step for numerical differentiation

        # Get initial conditions at t=0


        L_HUMAN_0 = _log_interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
        experiment_compute_0 = _log_interp(start_time, time_series_data.time, time_series_data.experiment_H100e)
        initial_aggregate_research_taste = 1
        coding_labor_0 = L_HUMAN_0 ** params.parallel_penalty
        # Calculate RS'(0)
        rs_rate_0 = compute_research_effort(
            experiment_compute_0, coding_labor_0,
            params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, initial_aggregate_research_taste
        )

        # Calculate RS'(dt) for numerical differentiation
        # Use log-space interpolation for exponential trends
        L_HUMAN_dt = _log_interp(start_time + dt, time_series_data.time, time_series_data.L_HUMAN)
        experiment_compute_dt = _log_interp(start_time + dt, time_series_data.time, time_series_data.experiment_H100e)

        # Automation fraction changes very little over small dt, so use same value
        coding_labor_dt = L_HUMAN_dt ** params.parallel_penalty

        rs_rate_dt = compute_research_effort(
            experiment_compute_dt, coding_labor_dt,
            params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, initial_aggregate_research_taste
        )
        logger.info(f"rs_rate_dt: {rs_rate_dt}, rs_rate_0: {rs_rate_0}, dt: {dt}")

        # Calculate RS''(0) using forward difference
        rs_rate_second_derivative = (rs_rate_dt - rs_rate_0) / dt
        # logger.info(f"Calculated rs_rate_second_derivative: {rs_rate_second_derivative:.6f}")

        # Avoid division by zero or very small denominators
        if abs(rs_rate_second_derivative) < cfg.PARAM_CLIP_MIN:
            logger.warning(f"Very small research stock second derivative: {rs_rate_second_derivative}, using fallback")
            # Use a reasonable fallback value
            return max(cfg.PARAM_CLIP_MIN, rs_rate_0)

        # Calculate initial research stock: RS(0) = (RS'(0))^2 / RS''(0)
        initial_research_stock = (rs_rate_0 ** 2) / rs_rate_second_derivative

        # Ensure the result is positive and finite
        if not np.isfinite(initial_research_stock) or initial_research_stock <= 0:
            logger.warning(f"Invalid calculated initial research stock: {initial_research_stock}, using fallback")
            return max(cfg.PARAM_CLIP_MIN, rs_rate_0)

        # logger.info(f"Calculated initial research stock: RS(0) = {initial_research_stock:.6f} "
        #            f"(RS'(0) = {rs_rate_0:.6f}, RS''(0) = {rs_rate_second_derivative:.6f})")

        return initial_research_stock

    except Exception as e:
        if should_reraise(e):
            raise
        logger.error(f"Error calculating initial research stock: {e}")
        # Fallback to a reasonable default
        return 1.0


def compute_initial_conditions(time_series_data: TimeSeriesData, params: Parameters,
                             initial_progress: float = 0.0) -> InitialConditions:
    """
    Compute all initial conditions needed for model calculations, shared by every
    entry point that sets up a run.

    Args:
        time_series_data: Input time series
        params: Model parameters
        initial_progress: Initial cumulative progress

    Returns:
        InitialConditions object with all computed initial values

    TODO: maybe modify this to use the passed automation model in coding labor calculation
    """
    start_time = time_series_data.time[0]

    # TODO: may need to change to log-space interpolation
    L_HUMAN = np.interp(start_time, time_series_data.time, time_series_data.L_HUMAN)
    inference_H100e = np.interp(start_time, time_series_data.time, time_series_data.inference_H100e)
    experiment_H100e = np.interp(start_time, time_series_data.time, time_series_data.experiment_H100e)
    training_system_performance_growth_rate = time_series_data.get_training_system_performance_growth_rate(start_time)

    if params.human_only:
        initial_automation = 1.0
        initial_ai_research_taste = 0.0
        initial_aggregate_research_taste = 1.0
        coding_labor = compute_coding_labor_deprecated(None, inference_H100e, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)
        research_effort = compute_research_effort(experiment_H100e, coding_labor, params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, initial_aggregate_research_taste)
    else:
        initial_automation = compute_automation_fraction(initial_progress, params)
        initial_ai_research_taste = compute_ai_research_taste(initial_progress, params)
        initial_aggregate_research_taste = compute_aggregate_research_taste(initial_ai_research_taste, params.taste_distribution)
        coding_labor = compute_coding_labor_deprecated(
            initial_automation, inference_H100e, L_HUMAN,
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
        )

        research_effort = compute_research_effort(
            experiment_H100e, coding_labor,
            params.alpha_experiment_capacity, params.rho_experiment_capacity, params.experiment_compute_exponent, initial_aggregate_research_taste
        )

    # Validate and fallback for research stock rate
    if not np.isfinite(research_effort) or research_effort <= 0:
        logger.warning(f"Invalid initial research stock rate ({research_effort}), using fallback 1.0")
        research_effort = 1.0

    # Calculate initial research stock
    research_stock = calculate_initial_research_stock(time_series_data, params, initial_progress)

    return InitialConditions(
        start_time=start_time,
        initial_progress=initial_progress,
        initial_automation=initial_automation,
        L_HUMAN=L_HUMAN,
        inference_H100e=inference_H100e,
        experiment_H100e=experiment_H100e,
        training_system_performance_growth_rate=training_system_performance_growth_rate,
        coding_labor=coding_labor,
        research_effort=research_effort,
        research_stock=research_stock
    )


def setup_model(time_series_data: TimeSeriesData, params: Parameters,
               initial_progress: float = 0.0) -> Tuple[Parameters, InitialConditions]:
    """
    Set up model parameters and compute initial conditions, so callers do not each
    repeat the setup logic.

    Args:
        time_series_data: Input time series
        params: Model parameters
        initial_progress: Initial cumulative progress

    Returns:
        Tuple of (params, initial_conditions)
    """
    # Compute initial conditions
    initial_conditions = compute_initial_conditions(time_series_data, params, initial_progress)

    return params, initial_conditions


def integrate_progress_human_only(
    time_range: List[float],
    initial_logE: float,
    initial_research_stock: float,
    time_series_data: TimeSeriesData,
    params: Parameters,
    serial_labor_multiplier_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute human-only trajectory using vectorized operations with high-precision integration.

    In human-only mode, there is no feedback loop between logE and research effort:
    - automation_fraction = 0.0 (fixed)
    - aggregate_research_taste = 1.0 (fixed)
    - coding_labor depends only on L_HUMAN (no progress feedback)

    This allows us to compute RS independently, then integrate cumulative E:
    dE/dt = 10^EC where EC = TC(t) + r_software * log10(RS/RS_0).

    Args:
        time_range: [start_time, end_time]
        initial_logE: Initial log10(cumulative effective compute)
        initial_research_stock: Initial research stock value
        time_series_data: Input time series
        params: Model parameters
        serial_labor_multiplier_fn: Optional callable t -> M(t) multiplying the
            serial coding labor. None (the default) is the human-only run and
            leaves every array bit-identical. It exists so the uplift
            reconstruction (progress_model.uplift_reconstruction) can re-run this
            exact quadrature with an EXOGENOUS coding-labor uplift path, instead
            of duplicating the fine-grid numerics with a coarser scheme of its
            own. The result is still not a with-AI run: automation fraction and
            aggregate research taste stay at their human-only values, and M(t) is
            a supplied function of time, not of the trajectory.

    Returns:
        Tuple of (times, logE_values, research_stock_values, research_effort_values, sw_progress_rates)
    """
    from scipy.integrate import cumulative_trapezoid

    t_start, t_end = time_range
    data = time_series_data
    RS_0 = initial_research_stock

    # Calculate output grid size (same as ODE solver output)
    time_span = t_end - t_start
    num_output_points = max(2, int(np.ceil(time_span / cfg.DENSE_OUTPUT_STEP_SIZE)) + 1)

    # Use moderate internal grid for accuracy within 0.1% tolerance
    # Lower multiplier = faster computation, higher = more accuracy
    internal_multiplier = 300
    num_internal_points = (num_output_points - 1) * internal_multiplier + 1
    internal_times = np.linspace(t_start, t_end, num_internal_points)

    # ========== STEP 1: Interpolate all time series data on fine grid ==========
    if data.can_use_log_L_HUMAN:
        L_HUMAN_fine = np.exp(np.interp(internal_times, data.time, data.log_L_HUMAN))
    else:
        L_HUMAN_fine = np.interp(internal_times, data.time, data.L_HUMAN)

    if data.can_use_log_experiment_H100e:
        experiment_compute_fine = np.exp(np.interp(internal_times, data.time, data.log_experiment_H100e))
    else:
        experiment_compute_fine = np.interp(internal_times, data.time, data.experiment_H100e)

    # ========== STEP 2: Compute coding labor (vectorized) ==========
    serial_coding_labor_fine = np.power(L_HUMAN_fine, params.parallel_penalty) * params.coding_labor_normalization
    if serial_labor_multiplier_fn is not None:
        serial_coding_labor_fine = serial_coding_labor_fine * np.asarray(
            serial_labor_multiplier_fn(internal_times), dtype=float)

    # ========== STEP 3: Compute research effort (already vectorized) ==========
    aggregate_research_taste = 1.0

    research_effort_fine = compute_research_effort(
        experiment_compute_fine,
        serial_coding_labor_fine,
        params.alpha_experiment_capacity,
        params.rho_experiment_capacity,
        params.experiment_compute_exponent,
        aggregate_research_taste
    )

    research_effort_fine = np.where(np.isfinite(research_effort_fine), research_effort_fine, 0.0)
    research_effort_fine = np.maximum(research_effort_fine, 0.0)

    # ========== STEP 4: Integrate research stock using cumulative trapezoid ==========
    cumulative_rs = cumulative_trapezoid(research_effort_fine, internal_times, initial=0)
    research_stock_fine = initial_research_stock + cumulative_rs
    research_stock_fine = np.maximum(research_stock_fine, 1e-6)

    # ========== STEP 5: Compute software progress rate (vectorized) ==========
    sw_progress_rate_fine = compute_software_progress_rate(
        research_stock_fine,
        research_effort_fine,
        params.r_software
    )

    sw_progress_rate_fine = np.where(np.isfinite(sw_progress_rate_fine), sw_progress_rate_fine, 0.0)
    sw_progress_rate_fine = np.maximum(sw_progress_rate_fine, 0.0)

    # ========== STEP 6: Compute cumulative effective compute ==========
    # EC(t) = TC(t) + r_software * log10(RS(t) / RS_0)
    # training_H100e is raw H100e; convert to log10(FLOP/yr) via FLOP_PER_H100E_YEAR (log-space interp).
    tc_fine = np.interp(internal_times, data.time, data.log10_training_H100e) + np.log10(cfg.FLOP_PER_H100E_YEAR)
    se_fine = params.r_software * np.log10(np.maximum(research_stock_fine, 1e-30) / RS_0)
    ec_fine = tc_fine + se_fine

    # dE/dt = 10^EC (linear FLOP/year)
    dEdt_fine = 10.0 ** ec_fine

    # ========== STEP 7: Integrate cumulative E ==========
    E_init = 10.0 ** initial_logE
    E_cumulative = cumulative_trapezoid(dEdt_fine, internal_times, initial=0)
    E_fine = E_init + E_cumulative
    logE_fine = np.log10(np.maximum(E_fine, 1e-300))

    # ========== STEP 8: Downsample to output grid ==========
    output_indices = np.arange(0, num_internal_points, internal_multiplier)

    times = internal_times[output_indices]
    logE_values = logE_fine[output_indices]
    research_stock = research_stock_fine[output_indices]
    research_effort = research_effort_fine[output_indices]
    sw_progress_rate = sw_progress_rate_fine[output_indices]

    logger.debug(f"integrate_progress_human_only: computed {num_output_points} points via {num_internal_points}-point fine grid")

    return times, logE_values, research_stock, research_effort, sw_progress_rate


def integrate_progress(
    time_range: List[float],
    initial_progress: float,
    initial_research_stock: float,
    time_series_data: TimeSeriesData,
    params: Parameters,
    direction: str = 'forward',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the coupled differential equation system with robust fallback methods:
    d(logE)/dt = 10^(EC - logE) / ln(10)  where EC = TC(t) + r_software * log10(RS/RS_0)
    d(research_stock)/dt = research_effort(t, logE, research_stock)

    Args:
        time_range: [start_time, end_time]
        initial_progress: Initial log10(cumulative effective compute)
        initial_research_stock: Initial research stock value
        time_series_data: Input time series
        params: Model parameters
        direction: 'forward' or 'backward'

    Returns:
        Tuple of (times, logE_values, research_stock_values)
    """
    RS_0 = initial_research_stock

    def ode_func(t, y):
        try:
            rates = progress_rate_at_time(t, y, time_series_data, params, RS_0=RS_0)
            # Validate the rates
            if len(rates) != 2 or not all(np.isfinite(rate) and rate >= 0 for rate in rates):
                logger.warning(f"Invalid rates {rates} at time {t}, state {y}")
                return [0.0, 0.0]  # Stop integration if rates become invalid
            return rates
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Error computing rates at t={t}, state={y}: {e}")
            return [0.0, 0.0]  # Fail gracefully

    def ode_func_bounded(t, y):
        """ODE function with bounds checking to prevent extreme values"""
        try:
            # Prevent logE from becoming extremely large
            if y[0] > cfg.PROGRESS_ODE_CLAMP_MAX:
                logger.warning(f"logE {y[0]} too large at time {t}, clamping")
                y[0] = cfg.PROGRESS_ODE_CLAMP_MAX

            # Prevent research stock from going negative or becoming extremely large
            if y[1] <= 0:
                y[1] = max(1e-6, initial_research_stock)
            elif y[1] > cfg.RESEARCH_STOCK_ODE_CLAMP_MAX:
                logger.warning(f"Research stock {y[1]} too large at time {t}, clamping")
                y[1] = cfg.RESEARCH_STOCK_ODE_CLAMP_MAX

            rates = progress_rate_at_time(t, y, time_series_data, params, RS_0=RS_0)

            # Clamp rates to reasonable bounds
            for i in range(len(rates)):
                if not np.isfinite(rates[i]):
                    rates[i] = 0.0
                elif rates[i] < 0:
                    rates[i] = 0.0

            return rates
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Error in bounded ODE function at t={t}: {e}")
            return [0.0, 0.0]

    t_start, t_end = time_range
    if direction == 'backward':
        t_start, t_end = t_end, t_start

    # Validate initial conditions
    if not np.isfinite(initial_progress):
        logger.warning(f"Invalid initial progress {initial_progress}, using fallback")
        initial_progress = 0.0

    # Use research stock from initial conditions
    initial_state = [initial_progress, initial_research_stock]

    # Try multiple integration methods with increasing robustness
    # Try stiff-friendly methods first to avoid excessive time spent on RK methods
    methods_to_try = [
        ('Radau', {'rtol': 1e-3, 'atol': 1e-5}),      # Implicit method for stiff problems
        ('RK23', {'rtol': 1e-4, 'atol': 1e-6}),       # Lower order explicit method
        ('RK45', {'rtol': 1e-4, 'atol': 1e-6}),       # Relaxed precision
        ('RK45', {'rtol': 1e-6, 'atol': 1e-8}),       # Higher precision
        ('DOP853', {'rtol': 1e-3, 'atol': 1e-5}),      # High-order explicit method
        ('LSODA', {'rtol': 1e-3, 'atol': 1e-5})
    ]

    sol = None
    collector = _ode_diagnostics_collector.get()
    collecting = collector is not None
    for method, tolerances in methods_to_try:
        try:
            logger.debug(f"Trying integration with method {method}, tolerances {tolerances}")

            attempt_wall_start = time.perf_counter() if collecting else None
            attempt_cpu_start = time.process_time() if collecting else None

            # Use bounded ODE function for more robust integration
            sol = integrate.solve_ivp(
                ode_func_bounded,
                [t_start, t_end],
                initial_state,
                method=method,
                dense_output=True,
                **tolerances,
                max_step=cfg.ODE_MAX_STEP  # Limit step size for stability
            )

            if collecting:
                attempt_wall_elapsed = time.perf_counter() - attempt_wall_start
                attempt_cpu_elapsed = time.process_time() - attempt_cpu_start

                step_sizes = np.diff(sol.t) if hasattr(sol, 't') and sol.t is not None and len(sol.t) > 1 else None
                step_stats = None
                if step_sizes is not None and len(step_sizes) > 0:
                    min_step = float(np.min(step_sizes))
                    max_step = float(np.max(step_sizes))
                    step_stats = {
                        'num_steps': int(len(sol.t)),
                        'min_step': min_step,
                        'max_step': max_step,
                        'mean_step': float(np.mean(step_sizes)),
                        'step_variation_ratio': float(max_step / min_step) if min_step > 0 else None,
                    }

                collector.append(
                    {
                        'method': method,
                        'direction': direction,
                        'human_only': bool(getattr(params, 'human_only', False)),
                        't_span': [float(t_start), float(t_end)],
                        'tolerances': dict(tolerances),
                        'success': bool(getattr(sol, 'success', False)),
                        'message': getattr(sol, 'message', None),
                        'wall_time_seconds': float(attempt_wall_elapsed),
                        'cpu_time_seconds': float(attempt_cpu_elapsed),
                        'nfev': int(getattr(sol, 'nfev', 0) or 0),
                        'njev': int(getattr(sol, 'njev', 0) or 0),
                        'nlu': int(getattr(sol, 'nlu', 0) or 0),
                        'step_stats': step_stats,
                    }
                )

            if sol.success:
                logger.debug(f"Integration succeeded with method {method}")

                # Log ODE step size information
                if cfg.ODE_STEP_SIZE_LOGGING and hasattr(sol, 't') and len(sol.t) > 1:
                    step_sizes = np.diff(sol.t)
                    min_step = np.min(step_sizes)
                    max_step = np.max(step_sizes)
                    mean_step = np.mean(step_sizes)
                    total_steps = len(sol.t)

                    logger.info(f"ODE step size stats for {method}: "
                              f"min={min_step:.2e}, max={max_step:.2e}, "
                              f"mean={mean_step:.2e}, total_steps={total_steps}")

                    # Log if step sizes are very small (potential stiffness indicator)
                    if min_step < cfg.ODE_SMALL_STEP_THRESHOLD:
                        logger.warning(f"Very small step sizes detected with {method}: "
                                     f"min_step={min_step:.2e} - this may indicate stiff ODE")

                    # Log if step sizes vary significantly (potential instability)
                    step_variation = max_step / min_step if min_step > 0 else float('inf')
                    if step_variation > cfg.ODE_STEP_VARIATION_THRESHOLD:
                        logger.warning(f"Large step size variation with {method}: "
                                     f"max/min ratio={step_variation:.1f} - potential instability")

                break
            else:
                logger.warning(f"Integration with {method} failed: {sol.message}")

        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Integration method {method} raised exception: {e}")
            if collecting:
                attempt_wall_elapsed = time.perf_counter() - attempt_wall_start
                attempt_cpu_elapsed = time.process_time() - attempt_cpu_start
                collector.append(
                    {
                        'method': method,
                        'direction': direction,
                        'human_only': bool(getattr(params, 'human_only', False)),
                        't_span': [float(t_start), float(t_end)],
                        'tolerances': dict(tolerances),
                        'success': False,
                        'message': None,
                        'exception': str(e),
                        'wall_time_seconds': float(attempt_wall_elapsed),
                        'cpu_time_seconds': float(attempt_cpu_elapsed),
                        'nfev': 0,
                        'njev': 0,
                        'nlu': 0,
                        'step_stats': None,
                    }
                )
            continue

    # Create dense output over time range
    try:
        # Calculate number of points from step size
        time_span = max(time_range) - min(time_range)
        num_points = max(2, int(np.ceil(time_span / cfg.DENSE_OUTPUT_STEP_SIZE)) + 1)
        times = np.linspace(min(time_range), max(time_range), num_points)
        solution_values = sol.sol(times)
        logE_values = solution_values[0]
        research_stock_values = solution_values[1]

        # Validate results
        if not all(np.isfinite(logE_values)):
            logger.warning("Non-finite values in logE integration result")
            # Replace non-finite values with interpolation
            finite_mask = np.isfinite(logE_values)
            if np.any(finite_mask):
                logE_values = np.interp(times, times[finite_mask], logE_values[finite_mask])
            else:
                raise ValueError("All logE integration results are non-finite")

        if not all(np.isfinite(research_stock_values)):
            logger.warning("Non-finite values in research stock integration result")
            # Replace non-finite values with interpolation
            finite_mask = np.isfinite(research_stock_values)
            if np.any(finite_mask):
                research_stock_values = np.interp(times, times[finite_mask], research_stock_values[finite_mask])
            else:
                raise ValueError("All research stock integration results are non-finite")

        if np.any(research_stock_values <= 0):
            logger.warning("Non-positive research stock values detected, clamping to minimum")
            research_stock_values = np.maximum(research_stock_values, 1e-6)

        return times, logE_values, research_stock_values

    except Exception as e:
        if should_reraise(e):
            raise
        logger.error(f"Error creating dense output: {e}")
        raise RuntimeError(f"Integration succeeded but dense output failed: {e}")
