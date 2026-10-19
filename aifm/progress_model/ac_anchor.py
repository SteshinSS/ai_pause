"""
AC anchor methods: alternative ways of computing progress_at_aa.

Each method fits a trend of some observable vs progress (fit against the
human-only trajectory, since the real trajectory doesn't exist yet at fit
time), and the AC progress level is obtained by inverting that trend at a
target threshold. The fitted trend doubles as a metric-computer: the metrics
loop evaluates `predict` on the trajectory's progress values and stores the
result under `metric_key`.

Methods must never depend on the AutomationModel or the integrated
trajectory: progress_at_aa is an *input* to the automation anchors, so a
method that consumed model-computed quantities (e.g. the model's own uplift)
would be circular.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy import optimize

from .utils import should_reraise

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


# Global cache for benchmark_results_1_1.yaml to avoid repeated file I/O
_BENCHMARK_DATA_CACHE = None


def _load_benchmark_data():
    """
    Load benchmark_results_1_1.yaml with module-level caching.

    Returns:
        Dict containing benchmark data, or empty dict if file not found/error.
    """
    global _BENCHMARK_DATA_CACHE

    if _BENCHMARK_DATA_CACHE is None:
        if yaml is None:
            _BENCHMARK_DATA_CACHE = {}
        else:
            try:
                metr_yaml = (Path(__file__).resolve().parents[1]
                             / 'inputs' / 'metr' / 'benchmark_results_1_1.yaml')
                with open(metr_yaml, 'r') as f:
                    _BENCHMARK_DATA_CACHE = yaml.safe_load(f)
                logger.debug("Loaded benchmark_results_1_1.yaml into cache")
            except FileNotFoundError:
                logger.warning("benchmark_results_1_1.yaml file not found")
                _BENCHMARK_DATA_CACHE = {}
            except Exception as e:
                if should_reraise(e):
                    raise
                logger.warning(f"Error reading benchmark_results_1_1.yaml: {e}")
                _BENCHMARK_DATA_CACHE = {}

    return _BENCHMARK_DATA_CACHE


@dataclass
class FittedObservable:
    """A fitted trend of an observable vs progress.

    predict and invert must stay consistent: both are closures over the same
    fitted parameters, which is why they live on one object instead of being
    free functions over raw fit output.
    """
    metric_key: str
    # progress (scalar or array) -> observable, vectorized
    predict: Callable[[Any], Any]
    # target observable value -> progress level, or None if the inversion is
    # invalid/unreachable. Does NOT include the AC gap.
    invert: Callable[[float], Optional[float]]
    # new_anchor_progress -> re-anchored copy (for the post-integration anchor
    # fix); None when the fitted form has no anchor to move.
    rebase_anchor: Optional[Callable[[float], "FittedObservable"]] = None
    # fitted parameters, diagnostics (R^2, form, uses_shifted_form, ...)
    fit_info: Dict[str, Any] = field(default_factory=dict)


def compute_gap_progress_units(params, anchor_progress_rate: float) -> float:
    """Convert the AC gap (anchor-progress-years) to progress units.

    Returns 0.0 when gap-included mode is off.
    """
    include_gap_flag = False
    try:
        _inc = getattr(params, 'include_gap', 'no gap')
        if isinstance(_inc, str):
            include_gap_flag = _inc.strip().lower() == 'gap'
        else:
            include_gap_flag = bool(_inc)
    except Exception as e:
        if should_reraise(e):
            raise
        include_gap_flag = False

    if not include_gap_flag:
        return 0.0

    try:
        gap_anchor_years = float(params.gap_years)
        gap_progress_units = float(anchor_progress_rate) * gap_anchor_years
    except Exception as e:
        if should_reraise(e):
            raise
        gap_progress_units = float(params.gap_years)

    try:
        year_label = int(params.present_day) if getattr(params, 'present_day', None) is not None else 'anchor'
    except Exception as e:
        if should_reraise(e):
            raise
        year_label = 'anchor'
    logger.info(
        f"Gap-included mode: adding gap {params.gap_years} {year_label}-progress-years "
        f"(~{gap_progress_units:.6f} progress units)"
    )
    return gap_progress_units


def compute_progress_at_aa(
    observable: FittedObservable,
    target: float,
    gap_progress_units: float,
) -> Optional[float]:
    """Invert a fitted observable trend at the AC target and add the gap.

    Returns None when the inversion fails or produces a non-finite value.
    """
    progress = observable.invert(target)
    if progress is None:
        return None
    progress = progress + gap_progress_units
    if not np.isfinite(progress):
        logger.warning("Invalid progress_at_aa result")
        return None
    logger.info(
        f"Progress level at AC target ({observable.metric_key} = {target}): {progress:.4f}"
    )
    return progress


# ---------------------------------------------------------------------------
# Time-horizon method (METR p80 horizon length vs progress)
# ---------------------------------------------------------------------------


def _extract_metr_pairs(human_only_times: np.ndarray, human_only_progress: np.ndarray):
    """Extract (progress, p80_horizon) pairs from METR benchmark data,
    mapping model release dates to progress via the human-only trajectory.

    Returns (progress_values, horizon_values) arrays or None on failure.
    """
    benchmark_data = _load_benchmark_data()
    if not benchmark_data:
        logger.error("benchmark_results_1_1.yaml could not be loaded")
        return None

    progress_horizon_pairs = []

    for model_name, model_info in benchmark_data['results'].items():
        # Convert release date to decimal year
        release_date_obj = model_info['release_date']
        try:
            # Handle both string and date objects
            if isinstance(release_date_obj, str):
                release_date = datetime.strptime(release_date_obj, '%Y-%m-%d').date()
            else:
                release_date = release_date_obj

            decimal_year = release_date.year + (release_date.timetuple().tm_yday - 1) / 365.25
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not parse release date for {model_name}: {release_date_obj} ({e})")
            continue

        # Interpolate progress value at the release date using human-only trajectory results
        if decimal_year >= human_only_times.min() and decimal_year <= human_only_times.max():
            interpolated_progress = np.interp(decimal_year, human_only_times, human_only_progress)
        elif decimal_year < human_only_times.min():
            interpolated_progress = human_only_progress[0]
        else:
            interpolated_progress = human_only_progress[-1]

        # Extract p80_horizon_length from metrics
        if 'metrics' in model_info and 'p80_horizon_length' in model_info['metrics']:
            p80_data = model_info['metrics']['p80_horizon_length']
            p80_estimate = p80_data.get('estimate')

            if p80_estimate is not None and p80_estimate > 0:  # Must be positive for log transform
                progress_horizon_pairs.append((interpolated_progress, p80_estimate))

    if len(progress_horizon_pairs) < 2:
        logger.error("Not enough valid (progress, horizon) pairs for regression")
        return None

    progress_values = np.array([pair[0] for pair in progress_horizon_pairs])
    horizon_values = np.array([pair[1] for pair in progress_horizon_pairs])
    return progress_values, horizon_values


def _make_exponential_horizon_observable(slope: float, intercept: float) -> FittedObservable:
    def predict(progress):
        """Map progress to horizon length using fitted exponential model"""
        return np.exp(slope * progress + intercept)

    def invert(target: float) -> Optional[float]:
        # Solve: target = exp(slope * progress + intercept)
        # Therefore: progress = (log(target) - intercept) / slope
        try:
            progress = (np.log(target) - intercept) / slope
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Could not invert exponential horizon trend at {target}: {e}")
            return None
        return float(progress)

    return FittedObservable(
        metric_key='horizon_lengths',
        predict=predict,
        invert=invert,
        rebase_anchor=None,
        fit_info={'form': 'exponential', 'uses_shifted_form': False,
                  'slope': float(slope), 'intercept': float(intercept)},
    )


def _make_decaying_horizon_observable(
    H_0: float, A_0: float, T_0: float, anchor_progress: Optional[float]
) -> FittedObservable:
    """Decaying-doubling-time observable. anchor_progress None means the
    unshifted (auto-fit) form."""

    def predict(progress):
        """Map progress to horizon length using fitted decaying doubling time model"""
        try:
            # Handle scalar vs array inputs
            is_scalar = np.isscalar(progress)
            progress_arr = np.atleast_1d(progress)

            # Ensure parameters are within valid ranges
            # Allow A_0 < 0, disallow A_0 == 0 and A_0 >= 1
            if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                fallback = np.full_like(progress_arr, 1e12)
                return fallback[0] if is_scalar else fallback

            if anchor_progress is not None:
                # Shifted form: horizon(t) = H_0 * (1 - A_0 * (t - anchor_progress) / T_0)^exponent
                progress_shifted = progress_arr - anchor_progress
                base_term = 1 - A_0 * progress_shifted / T_0
            else:
                # Original form: horizon(t) = H_0 * (1 - A_0 * t / T_0)^exponent
                base_term = 1 - A_0 * progress_arr / T_0

            # Clamp to small positive to avoid domain errors
            base_term = np.maximum(base_term, 1e-12)

            # Calculate the exponent
            log_denominator = np.log(1 - A_0)
            if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                fallback = np.full_like(progress_arr, 1e12)
                return fallback[0] if is_scalar else fallback

            exponent = np.log(2) / log_denominator

            # Calculate the result
            result = H_0 * (base_term ** exponent)

            # Replace only invalid elements (not the entire array)
            result = np.where(
                np.isfinite(result) & (result > 0),
                result,
                1e12
            )

            return result[0] if is_scalar else result

        except (ValueError, ZeroDivisionError, OverflowError):
            progress_arr = np.atleast_1d(progress)
            fallback = np.full_like(progress_arr, 1e12)
            return fallback[0] if np.isscalar(progress) else fallback

    def invert(target: float) -> Optional[float]:
        try:
            if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                logger.warning("Invalid parameters for progress_at_aa calculation")
                return None
            if target <= 0:
                logger.warning("Invalid AC target for calculation")
                return None
            # Check if the ratio is valid
            ratio = target / H_0
            if ratio <= 0:
                logger.warning("Invalid ratio for progress_at_aa calculation")
                return None
            log_ratio = np.log(1 - A_0) / np.log(2)
            if not np.isfinite(log_ratio):
                logger.warning("Invalid log ratio for progress_at_aa calculation")
                return None
            ratio_term = ratio ** log_ratio
            if not np.isfinite(ratio_term):
                logger.warning("Invalid ratio_term for progress_at_aa calculation")
                return None
            if anchor_progress is not None:
                # Shifted form: target = H_0 * (1 - A_0 * (progress - anchor_progress) / T_0)^exponent
                # progress = anchor_progress + T_0 * (1 - (target / H_0)^(log(1-A_0)/log(2))) / A_0
                return float(anchor_progress + T_0 * (1 - ratio_term) / A_0)
            else:
                # Original form: target = H_0 * (1 - A_0 * progress / T_0)^exponent
                # progress = T_0 * (1 - (target / H_0)^(log(1-A_0)/log(2))) / A_0
                return float(T_0 * (1 - ratio_term) / A_0)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logger.warning(f"Could not invert decaying doubling time trend at {target}: {e}")
            return None

    rebase = None
    if anchor_progress is not None:
        def rebase(new_anchor_progress: float) -> FittedObservable:
            return _make_decaying_horizon_observable(H_0, A_0, T_0, new_anchor_progress)

    return FittedObservable(
        metric_key='horizon_lengths',
        predict=predict,
        invert=invert,
        rebase_anchor=rebase,
        fit_info={'form': 'decaying doubling time',
                  'uses_shifted_form': anchor_progress is not None,
                  'H_0': float(H_0), 'A_0': float(A_0), 'T_0': float(T_0),
                  'anchor_progress': None if anchor_progress is None else float(anchor_progress)},
    )


def resolve_observable_anchor(
    params,
    date_attr: str,
    mapping_times: np.ndarray,
    mapping_progress: np.ndarray,
    anchor_progress_rate: float,
    mapping_progress_rates: Optional[np.ndarray] = None,
) -> tuple:
    """Resolve WHERE an observable binds on the anchoring mapping.

    Returns (t_obs, progress_at_t_obs, rate_at_t_obs) for the observable whose
    reference date lives in params.<date_attr> (None => present_day). This is
    the uplift_reference_date convention generalized: each elicited observable
    carries its own vintage, and both its value and its companion per-year
    rate bind at that date.

    Bit-identity guarantee: when the date is None (the default), t_obs is
    present_day and the returned rate is the anchor_progress_rate scalar
    passed in — NOT re-interpolated — so the default pipeline is unchanged to
    the last bit. When a date IS set but no rate series is available, the
    present-day rate is used with a warning (a dated value with an undated
    rate conversion is better than refusing to fit).
    """
    date = getattr(params, date_attr, None)
    if date is None:
        t_obs = float(params.present_day)
        rate = float(anchor_progress_rate)
    else:
        t_obs = float(date)
        if mapping_progress_rates is not None:
            rate = float(np.interp(t_obs, np.asarray(mapping_times, dtype=float),
                                   np.asarray(mapping_progress_rates, dtype=float)))
            if not (np.isfinite(rate) and rate > 0):
                logger.warning(
                    f"{date_attr}: invalid progress rate {rate} at {t_obs}; "
                    f"falling back to the present-day anchor rate {anchor_progress_rate}")
                rate = float(anchor_progress_rate)
        else:
            logger.warning(
                f"{date_attr}={t_obs} set but no mapping progress rates supplied; "
                f"converting per-year quantities at the present-day anchor rate")
            rate = float(anchor_progress_rate)
    progress = float(np.interp(t_obs, mapping_times, mapping_progress))
    return t_obs, progress, rate


# Which reference-date parameter each anchored observable binds to, keyed by the
# observable's own metric_key. Anything absent (or mapped to None) has no vintage
# of its own and binds at present_day.
#
# resolve_observable_anchor above answers this question while FITTING, against the
# human-only anchoring mapping. This map exists for the second question, asked after
# integration: which date should an observable be RE-anchored at on the realized
# trajectory. Both must agree, or a trend would be fitted at one date and rebased at
# another.
REFERENCE_DATE_ATTR_BY_METRIC = {
    'horizon_lengths': 'horizon_reference_date',
    'predicted_revenue': 'revenue_reference_date',
    'capability_ecis': 'eci_reference_date',
}


def resolve_reference_date(params, metric_key: str) -> float:
    """The date an observable's stated level binds at (None => present_day)."""
    attr = REFERENCE_DATE_ATTR_BY_METRIC.get(metric_key)
    date = getattr(params, attr, None) if attr else None
    if date is None:
        return float(params.present_day)
    return float(date)


def fit_horizon_trend(
    params,
    human_only_times: np.ndarray,
    human_only_progress: np.ndarray,
    anchor_progress_rate: float,
    mapping_progress_rates: Optional[np.ndarray] = None,
) -> Optional[FittedObservable]:
    """
    Fit the horizon trend: log(p80_horizon_length) vs progress.

    The functional form depends on params.horizon_extrapolation_type:
    - "exponential": linear regression on log(horizon) vs progress
    - "decaying doubling time": decaying doubling time functional form

    Manual anchor parameters (present_horizon, present_doubling_time,
    doubling_difficulty_growth_factor) override or constrain the fit; METR
    benchmark data is only loaded when some of them are missing.

    Args:
        params: Model parameters
        human_only_times: Time array from human-only trajectory computation
        human_only_progress: Progress array from human-only trajectory computation
        anchor_progress_rate: Progress rate at anchor time (progress units per
            time unit), used to convert present_doubling_time from time units
            to progress units when horizon_reference_date is unset
        mapping_progress_rates: d(progress)/dt on the human_only_times grid.
            Only consulted when params.horizon_reference_date is set, to
            convert present_doubling_time at THAT date's rate.

    The anchor point (value and doubling time) binds at
    params.horizon_reference_date when set, else at present_day — see
    resolve_observable_anchor.

    Returns:
        FittedObservable, or None if fitting failed.
    """
    anchor_date, anchor_progress, anchor_rate = resolve_observable_anchor(
        params, 'horizon_reference_date', human_only_times, human_only_progress,
        anchor_progress_rate, mapping_progress_rates)
    # Determine if we need benchmark data for autofitting
    # Benchmark data is only needed if any manual parameters are missing
    needs_benchmark_data = False

    if params.horizon_extrapolation_type == "exponential":
        # For exponential: need data if present_horizon or present_doubling_time is None
        if params.present_horizon is None or params.present_doubling_time is None:
            needs_benchmark_data = True
    elif params.horizon_extrapolation_type == "decaying doubling time":
        # For decaying doubling time: need data if any of the three params is None
        if (params.present_horizon is None or
            params.present_doubling_time is None or
            params.doubling_difficulty_growth_factor is None):
            needs_benchmark_data = True

    # Initialize benchmark data variables (will be populated if needed)
    progress_values = np.array([])
    horizon_values = np.array([])

    if needs_benchmark_data:
        pairs = _extract_metr_pairs(human_only_times, human_only_progress)
        if pairs is None:
            return None
        progress_values, horizon_values = pairs

    # Fit functional form based on horizon_extrapolation_type
    log_horizon_values = np.log(horizon_values)

    # Define fitting functions
    def linear_func(x, a, b):
        return a * x + b

    def decaying_doubling_time_func(t, H_0, A_0, T_0):
        """Decaying doubling time function with numerical safeguards.
        Supports A_0 in (-inf, 1), excluding A_0 == 0. Handles both accelerating (A_0>0) and decelerating (A_0<0) cases.
        """
        try:
            # Handle scalar vs array inputs
            is_scalar = np.isscalar(t)
            t_arr = np.atleast_1d(t)

            # Ensure parameters are within valid ranges
            # A_0 must be < 1 and != 0; H_0 and T_0 must be > 0
            if T_0 <= 0 or H_0 <= 0 or A_0 >= 1 or A_0 == 0:
                fallback = np.full_like(t_arr, np.log(1e12))
                return fallback[0] if is_scalar else fallback

            # Calculate the base term (1 - A_0 * t / T_0)
            base_term = 1 - A_0 * t_arr / T_0
            # Clamp to small positive to avoid domain errors
            base_term = np.maximum(base_term, 1e-12)

            # Calculate the exponent
            log_denominator = np.log(1 - A_0)
            if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                fallback = np.full_like(t_arr, np.log(1e12))
                return fallback[0] if is_scalar else fallback
            exponent = np.log(2) / log_denominator

            # Calculate the result
            result = H_0 * (base_term ** exponent)

            # Replace only invalid elements (not the entire array)
            result = np.where(
                np.isfinite(result) & (result > 0),
                result,
                1e12
            )

            log_result = np.log(result)
            return log_result[0] if is_scalar else log_result

        except (ValueError, ZeroDivisionError, OverflowError):
            t_arr = np.atleast_1d(t)
            fallback = np.full_like(t_arr, np.log(1e12))
            return fallback[0] if np.isscalar(t) else fallback

    try:
        if params.horizon_extrapolation_type == "exponential":
            # Check if manual parameters are provided
            if params.present_horizon is not None:
                # Use manual fitting with the anchor point resolved above
                # If present_doubling_time is provided, use it to calculate slope
                if params.present_doubling_time is not None:
                    # Convert present_doubling_time from time units to progress units
                    # at the horizon's own reference date
                    doubling_time_in_progress_units = params.present_doubling_time * anchor_rate
                    # slope = log(2) / doubling_time (in progress units)
                    slope = np.log(2) / doubling_time_in_progress_units
                else:
                    # Optimize slope using data, but fix intercept using anchor point
                    def fit_slope_only(slope_val):
                        intercept_val = np.log(params.present_horizon) - slope_val * anchor_progress
                        predicted = linear_func(progress_values, slope_val, intercept_val)
                        return np.sum((log_horizon_values - predicted)**2)

                    result = optimize.minimize_scalar(fit_slope_only, bounds=(-10, 10), method='bounded')
                    slope = result.x

                # Calculate intercept from anchor point: log(present_horizon) = slope * anchor_progress + intercept
                intercept = np.log(params.present_horizon) - slope * anchor_progress

                logger.info(f"Manual exponential horizon trajectory: log(horizon) = {slope:.6f} * progress + {intercept:.6f}")
                logger.info(f"Using anchor point: time={anchor_date}, progress={anchor_progress:.4f}, horizon={params.present_horizon:.4f}")
                if params.present_doubling_time is not None:
                    logger.info(f"Anchor doubling time (time units): {params.present_doubling_time:.4f}, converted to progress units: {doubling_time_in_progress_units:.4f} (using progress rate: {anchor_rate:.4f})")
            else:
                # Use automatic curve fitting
                popt, pcov = optimize.curve_fit(linear_func, progress_values, log_horizon_values)
                slope, intercept = popt

                logger.info(f"Fitted exponential horizon trajectory: log(horizon) = {slope:.6f} * progress + {intercept:.6f}")
                logger.info(f"R-squared: {1 - np.sum((log_horizon_values - linear_func(progress_values, *popt))**2) / np.sum((log_horizon_values - np.mean(log_horizon_values))**2):.4f}")

            return _make_exponential_horizon_observable(slope, intercept)

        elif params.horizon_extrapolation_type == "decaying doubling time":
            # Determine approach based on whether present_doubling_time is specified
            # If present_doubling_time is specified, we MUST use the shifted function approach
            # to ensure the doubling time at the anchor point equals present_doubling_time
            if params.present_doubling_time is not None or params.present_horizon is not None:
                # Use shifted function approach, anchored at the point resolved above

                # Determine what parameters we have and what we need to optimize
                params_to_optimize = []
                fixed_params = {}

                # Handle H_0 (present_horizon)
                if params.present_horizon is not None:
                    fixed_params['H_0'] = params.present_horizon
                else:
                    params_to_optimize.append('H_0')

                # Handle T_0 (present_doubling_time)
                if params.present_doubling_time is not None:
                    # Convert present_doubling_time from time units to progress units
                    # at the horizon's own reference date
                    doubling_time_in_progress_units = params.present_doubling_time * anchor_rate
                    fixed_params['T_0'] = doubling_time_in_progress_units
                else:
                    params_to_optimize.append('T_0')

                # Handle A_0 (doubling_difficulty_growth_factor converted to decay_rate)
                if params.doubling_difficulty_growth_factor is not None:
                    fixed_params['A_0'] = 1.0 - params.doubling_difficulty_growth_factor
                else:
                    params_to_optimize.append('A_0')

                def shifted_func_factory(H_0_val, T_0_val, A_0_val):
                    # Use shifted function: horizon(t) = H_0 * (1 - A_0 * (t - anchor_progress) / T_0)^exponent
                    def shifted_func(t_vals):
                        try:
                            t_shifted = t_vals - anchor_progress
                            base_term = 1 - A_0_val * t_shifted / T_0_val
                            # Clamp to small positive to avoid domain errors
                            base_term = np.maximum(base_term, 1e-12)

                            # Guard against A_0 near 0 or >=1
                            log_denominator = np.log(1 - A_0_val)
                            if not np.isfinite(log_denominator) or abs(log_denominator) < 1e-12:
                                return np.full_like(t_vals, np.log(1e12))
                            exponent = np.log(2) / log_denominator
                            result = H_0_val * (base_term ** exponent)

                            # Replace only invalid elements
                            result = np.where(
                                np.isfinite(result) & (result > 0),
                                result,
                                1e12
                            )

                            return np.log(result)
                        except Exception as e:
                            if should_reraise(e):
                                raise
                            logger.warning(f"Error in shifted_func: {e}")
                            return np.full_like(t_vals, np.log(1e12))

                    return shifted_func

                if len(params_to_optimize) == 0:
                    # All parameters specified (Case 8)
                    H_0 = params.present_horizon
                    T_0 = doubling_time_in_progress_units
                    A_0 = 1.0 - params.doubling_difficulty_growth_factor
                    logger.info(f"Manual decaying doubling time: All parameters specified")
                elif len(params_to_optimize) == 1:
                    # Optimize one parameter
                    param_name = params_to_optimize[0]

                    def fit_single_param(param_val):
                        if param_name == 'A_0' and (param_val >= 1 or abs(param_val) < 1e-6):
                            return 1e6
                        if param_name in ['H_0', 'T_0'] and param_val <= 0:
                            return 1e6

                        # Reconstruct full parameter set
                        param_dict = fixed_params.copy()
                        param_dict[param_name] = param_val

                        shifted_func = shifted_func_factory(
                            param_dict['H_0'], param_dict['T_0'], param_dict['A_0'])
                        predicted = shifted_func(progress_values)
                        return np.sum((log_horizon_values - predicted)**2)

                    # Set bounds based on parameter type
                    if param_name == 'A_0':
                        bounds = (-0.999, 0.999)
                        initial_guess = 0.05
                    elif param_name == 'H_0':
                        bounds = (1e-6, 1e6)
                        initial_guess = 0.00001
                    elif param_name == 'T_0':
                        bounds = (1e-6, 100.0)
                        initial_guess = 1.35

                    result = optimize.minimize_scalar(fit_single_param, bounds=bounds, method='bounded')

                    # Extract optimized parameters
                    param_dict = fixed_params.copy()
                    param_dict[param_name] = result.x

                    H_0 = param_dict['H_0']
                    T_0 = param_dict['T_0']
                    A_0 = param_dict['A_0']

                    logger.info(f"Optimized {param_name}: {result.x:.6f}")
                else:
                    # Optimize multiple parameters
                    def fit_multiple_params(params_array):
                        # Reconstruct full parameter set
                        param_dict = fixed_params.copy()
                        for i, param_name in enumerate(params_to_optimize):
                            param_dict[param_name] = params_array[i]

                        H_0_val = param_dict['H_0']
                        T_0_val = param_dict['T_0']
                        A_0_val = param_dict['A_0']

                        if A_0_val >= 1 or abs(A_0_val) < 1e-6 or H_0_val <= 0 or T_0_val <= 0:
                            return 1e6

                        shifted_func = shifted_func_factory(H_0_val, T_0_val, A_0_val)
                        predicted = shifted_func(progress_values)
                        return np.sum((log_horizon_values - predicted)**2)

                    # Set up bounds and initial guesses
                    bounds = []
                    p0 = []
                    for param_name in params_to_optimize:
                        if param_name == 'A_0':
                            bounds.append((-0.999, 0.999))
                            p0.append(0.05)
                        elif param_name == 'H_0':
                            bounds.append((1e-6, 1e6))
                            p0.append(0.00001)
                        elif param_name == 'T_0':
                            bounds.append((1e-6, 100.0))
                            p0.append(1.35)

                    result = optimize.minimize(fit_multiple_params, p0, bounds=bounds, method='L-BFGS-B')

                    # Extract optimized parameters
                    param_dict = fixed_params.copy()
                    for i, param_name in enumerate(params_to_optimize):
                        param_dict[param_name] = result.x[i]

                    H_0 = param_dict['H_0']
                    T_0 = param_dict['T_0']
                    A_0 = param_dict['A_0']

                    logger.info(f"Optimized parameters: {', '.join([f'{name}={param_dict[name]:.6f}' for name in params_to_optimize])}")

                logger.info(f"Manual decaying doubling time horizon trajectory: H_0={H_0:.6f}, A_0={A_0:.6f}, T_0={T_0:.6f}")
                logger.info(f"Using anchor point: time={anchor_date}, progress={anchor_progress:.4f}")
                if params.present_horizon is not None:
                    logger.info(f"Anchor horizon: {params.present_horizon:.4f}")
                if params.present_doubling_time is not None:
                    logger.info(f"Anchor doubling time (time units): {params.present_doubling_time:.4f}, converted to progress units: {doubling_time_in_progress_units:.4f} (using progress rate: {anchor_rate:.4f})")

                # Shifted form is anchored at anchor_progress
                return _make_decaying_horizon_observable(H_0, A_0, T_0, anchor_progress)
            else:
                # Use automatic curve fitting (Cases 1-2: no anchor parameters specified)
                # These specific values are somewhat important for the optimization
                H_0_init = 0.00001
                A_0_init = 0.05
                T_0_init = 1.35

                popt, pcov = optimize.curve_fit(
                    decaying_doubling_time_func,
                    progress_values,
                    log_horizon_values,
                    p0=[H_0_init, A_0_init, T_0_init],
                    bounds=([1e-6, -0.999, 1e-6], [np.inf, 0.999, np.inf])  # Allow negative A_0 but exclude 0 implicitly via fit
                )
                H_0, A_0, T_0 = popt

                logger.info(f"Fitted decaying doubling time horizon trajectory: H_0={H_0:.6f}, A_0={A_0:.6f}, T_0={T_0:.6f}")
                logger.info(f"R-squared: {1 - np.sum((log_horizon_values - decaying_doubling_time_func(progress_values, *popt))**2) / np.sum((log_horizon_values - np.mean(log_horizon_values))**2):.4f}")

                # For automatic fitting, no anchor_progress shift is used
                return _make_decaying_horizon_observable(H_0, A_0, T_0, None)

        else:
            logger.error(f"Unknown horizon_extrapolation_type: {params.horizon_extrapolation_type}")
            return None

    except Exception as e:
        if should_reraise(e):
            raise
        logger.error(f"Error fitting horizon trajectory: {e}")
        return None


# ---------------------------------------------------------------------------
# Revenue-trend method
# ---------------------------------------------------------------------------


def fit_revenue_trend(
    params,
    human_only_times: np.ndarray,
    human_only_progress: np.ndarray,
    anchor_progress_rate: float,
    mapping_progress_rates: Optional[np.ndarray] = None,
) -> Optional[FittedObservable]:
    """
    Fit a revenue trend anchored at revenue_reference_date (None => present_day):

        revenue(progress) = present_revenue
                            * present_revenue_growth_rate ^ ((progress - anchor_progress) / rate(t_obs))

    present_revenue_growth_rate is the annual growth FACTOR at the reference
    date (e.g. 3.0 = revenue triples per year); the exponent converts progress
    units back to years via the progress rate AT THAT DATE (the present-day
    anchor rate when the date is unset), so along the anchoring trajectory
    revenue grows by exactly that factor per year there.

    AC is reached when revenue crosses params.revenue_required_for_ac.
    """
    present_revenue = getattr(params, 'present_revenue', None)
    growth_factor = getattr(params, 'present_revenue_growth_rate', None)

    if present_revenue is None or growth_factor is None:
        logger.error("Revenue trend requires present_revenue and present_revenue_growth_rate")
        return None
    if not (np.isfinite(present_revenue) and present_revenue > 0):
        logger.error(f"Invalid present_revenue: {present_revenue}")
        return None
    if not (np.isfinite(growth_factor) and growth_factor > 0 and growth_factor != 1.0):
        logger.error(f"Invalid present_revenue_growth_rate: {growth_factor}")
        return None
    if not (np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0):
        logger.error(f"Invalid anchor_progress_rate for revenue trend: {anchor_progress_rate}")
        return None

    anchor_date, anchor_progress, anchor_rate = resolve_observable_anchor(
        params, 'revenue_reference_date', human_only_times, human_only_progress,
        anchor_progress_rate, mapping_progress_rates)
    # ln(revenue) grows by this much per progress unit
    log_growth_per_progress = np.log(growth_factor) / anchor_rate

    def make_observable(anchor_progress_val: float) -> FittedObservable:
        def predict(progress):
            """Map progress to predicted revenue using the anchored exponential trend"""
            return present_revenue * np.exp(log_growth_per_progress * (np.asarray(progress) - anchor_progress_val))

        def invert(target: float) -> Optional[float]:
            if not (np.isfinite(target) and target > 0):
                logger.warning(f"Invalid revenue target: {target}")
                return None
            return float(anchor_progress_val + np.log(target / present_revenue) / log_growth_per_progress)

        def rebase(new_anchor_progress: float) -> FittedObservable:
            return make_observable(new_anchor_progress)

        return FittedObservable(
            metric_key='predicted_revenue',
            predict=predict,
            invert=invert,
            rebase_anchor=rebase,
            fit_info={'form': 'exponential revenue', 'uses_shifted_form': True,
                      'present_revenue': float(present_revenue),
                      'growth_factor': float(growth_factor),
                      'log_growth_per_progress': float(log_growth_per_progress),
                      'anchor_progress': float(anchor_progress_val)},
        )

    obs = make_observable(anchor_progress)
    logger.info(
        f"Revenue trend: present_revenue={present_revenue:.4g} at time {anchor_date} "
        f"(progress {anchor_progress:.4f}), "
        f"growth factor {growth_factor:.4g}/yr (~{log_growth_per_progress:.6f} ln-units/progress)"
    )
    return obs


# ---------------------------------------------------------------------------
# ECI (Epoch Capabilities Index) method
# ---------------------------------------------------------------------------


def fit_eci_trend(
    params,
    human_only_times: np.ndarray,
    human_only_progress: np.ndarray,
    anchor_progress_rate: float,
    mapping_progress_rates: Optional[np.ndarray] = None,
) -> Optional[FittedObservable]:
    """
    Build the ECI trend anchored at eci_reference_date (None => present_day):

        ECI(progress) = eci_at_present_day
                        + (eci_growth_rate / rate(t_obs)) * (progress - anchor_progress)

    eci_growth_rate is the instantaneous ECI rate at the reference date
    (points/year); dividing by the progress rate AT THAT DATE (OOMs of
    effective compute per year; the present-day anchor rate when the date is
    unset) converts it to points per progress unit. Linearity in progress
    means dECI/dt is the same multiple of d(logE)/dt at all times, so the
    reference-date (value, rate) pair pins down the whole mapping.

    AC is reached when ECI crosses params.eci_required_for_ac.
    """
    eci_present = getattr(params, 'eci_at_present_day', None)
    eci_rate = getattr(params, 'eci_growth_rate', None)

    if eci_present is None or eci_rate is None:
        logger.error("ECI trend requires eci_at_present_day and eci_growth_rate")
        return None
    if not np.isfinite(eci_present):
        logger.error(f"Invalid eci_at_present_day: {eci_present}")
        return None
    if not (np.isfinite(eci_rate) and eci_rate > 0):
        logger.error(f"Invalid eci_growth_rate: {eci_rate}")
        return None
    if not (np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0):
        logger.error(f"Invalid anchor_progress_rate for ECI trend: {anchor_progress_rate}")
        return None

    anchor_date, anchor_progress, anchor_rate = resolve_observable_anchor(
        params, 'eci_reference_date', human_only_times, human_only_progress,
        anchor_progress_rate, mapping_progress_rates)
    points_per_progress = float(eci_rate) / float(anchor_rate)

    def make_observable(anchor_progress_val: float) -> FittedObservable:
        def predict(progress):
            """Map progress to ECI score using the anchored linear trend"""
            return eci_present + points_per_progress * (np.asarray(progress) - anchor_progress_val)

        def invert(target: float) -> Optional[float]:
            if not np.isfinite(target):
                logger.warning(f"Invalid ECI target: {target}")
                return None
            return float(anchor_progress_val + (target - eci_present) / points_per_progress)

        def rebase(new_anchor_progress: float) -> FittedObservable:
            return make_observable(new_anchor_progress)

        return FittedObservable(
            metric_key='capability_ecis',
            predict=predict,
            invert=invert,
            rebase_anchor=rebase,
            fit_info={'form': 'linear ECI', 'uses_shifted_form': True,
                      'eci_at_present_day': float(eci_present),
                      'eci_growth_rate': float(eci_rate),
                      'points_per_progress': float(points_per_progress),
                      'anchor_progress': float(anchor_progress_val)},
        )

    obs = make_observable(anchor_progress)
    logger.info(
        f"ECI trend: eci_at_present_day={eci_present:.4g} at time {anchor_date} "
        f"(progress {anchor_progress:.4f}), "
        f"{eci_rate:.4g} pts/yr (~{points_per_progress:.6f} pts/progress-OOM)"
    )
    return obs
