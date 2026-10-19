#!/usr/bin/env python3
"""
Automation Model Module

Contains the AutomationModel class for computing automation fractions
and optimal CES frontier calculations.
"""

import numpy as np
import time
from typing import Optional, Dict, Any, Tuple, NamedTuple, TYPE_CHECKING
from scipy.optimize import brentq, minimize_scalar, least_squares
import logging

from . import config as cfg
from .ces_functions import compute_coding_labor_deprecated

if TYPE_CHECKING:
    from .parameters import Parameters

logger = logging.getLogger(__name__)


class AutomationModel:
    """Automation model"""
    def __init__(self, params, *, anchors=None, logistic=None):
        """
        Args:
            params: Parameters object (supplies schedule type, asymptote, CES params, ...).
            anchors: Optional {progress: automation_fraction} dict overriding
                params.automation_anchors. Lets solvers evaluate candidate
                schedules without deep-copying the whole params object.
            logistic: Optional (L, k, x0) triple that bypasses the anchor fit
                entirely and pins the logistic schedule directly. Forces
                schedule_type='logistic'. Mutually exclusive with `anchors`.
        """
        if logistic is not None and anchors is not None:
            raise ValueError("AutomationModel: pass at most one of `anchors` and `logistic`")

        self.initial_FTE_per_GPU = 1
        self.FTE_per_GPU_slope = 1.0
        self.progress_base_unit = cfg.BASE_FOR_SOFTWARE_LOM

        if logistic is not None:
            self._init_from_logistic(logistic)
            # Optimal CES frontier precompute cache (lazy init)
            self._frontier_pc: Optional[Dict[str, Any]] = None
            self._frontier_params_signature: Optional[Tuple] = None
            return

        self.schedule_type = getattr(params, 'automation_interp_type', cfg.DEFAULT_PARAMETERS['automation_interp_type'])
        anchor_source = params.automation_anchors if anchors is None else anchors
        anchors = list(anchor_source.items())
        anchors.sort(key=lambda x: x[0])
        self.anchor_points = anchors
        (prog_1, aut_1), (prog_2, aut_2) = self.anchor_points
        self.linear_aut_slope = (aut_2 - aut_1) / (prog_2 - prog_1)
        self.linear_aut_intercept = aut_1 - self.linear_aut_slope * prog_1
        self.exponential_aut_slope = (np.log(aut_2) - np.log(aut_1)) / (prog_2 - prog_1)

        # Logistic schedule parameters: f(x) = L / (1 + exp(-k(x - x0)))
        # Only compute if using logistic schedule to avoid errors with incompatible anchors
        if self.schedule_type == "logistic":
            self.logistic_L = getattr(params, 'automation_logistic_asymptote', cfg.DEFAULT_PARAMETERS['automation_logistic_asymptote'])
            # Compute k and x0 from anchor points
            # Need L > aut_1 and L > aut_2 for valid logistic fit
            if self.logistic_L > aut_1 and self.logistic_L > aut_2:
                ratio_1 = self.logistic_L / aut_1 - 1
                ratio_2 = self.logistic_L / aut_2 - 1
                if ratio_1 > 0 and ratio_2 > 0 and prog_2 != prog_1:
                    self.logistic_k = np.log(ratio_1 / ratio_2) / (prog_2 - prog_1)
                    if abs(self.logistic_k) > 1e-12:
                        self.logistic_x0 = prog_1 + np.log(ratio_1) / self.logistic_k
                    else:
                        assert False, "Logistic aut frac error, anchors are not as expected."
                        # k ~ 0 means roughly constant, use midpoint
                        self.logistic_x0 = (prog_1 + prog_2) / 2
                else:
                    # Fallback to linear-like behavior
                    self.logistic_k = 1.0
                    self.logistic_x0 = (prog_1 + prog_2) / 2
            else:
                assert False, f"Value for logistic_L not large enough."
                # L not large enough, fallback
                self.logistic_k = 1.0
                self.logistic_x0 = (prog_1 + prog_2) / 2
        else:
            # Set defaults for non-logistic modes (not used but avoids missing attributes)
            self.logistic_L = 1.0
            self.logistic_k = 1.0
            self.logistic_x0 = (prog_1 + prog_2) / 2

        # Optimal CES frontier precompute cache (lazy init)
        self._frontier_pc: Optional[Dict[str, Any]] = None
        self._frontier_params_signature: Optional[Tuple] = None

    def _init_from_logistic(self, logistic) -> None:
        """Pin the logistic schedule directly from (L, k, x0), skipping the anchor fit."""
        L, k, x0 = (float(v) for v in logistic)
        if not (np.isfinite(L) and np.isfinite(k) and np.isfinite(x0)):
            raise ValueError(f"AutomationModel: non-finite logistic params (L={L}, k={k}, x0={x0})")
        if not L > 0:
            raise ValueError(f"AutomationModel: logistic asymptote must be positive, got {L}")
        if not k > 0:
            raise ValueError(f"AutomationModel: logistic k must be positive, got {k}")

        self.schedule_type = "logistic"
        self.logistic_L = L
        self.logistic_k = k
        self.logistic_x0 = x0

        # Synthesize a canonical anchor pair (the L/4 and 3L/4 crossings). It is a
        # bijective function of (L, k, x0), so the frontier cache signature — which
        # keys on anchor_points — still identifies the schedule exactly.
        half_width = float(np.log(3.0) / k)
        self.anchor_points = [(x0 - half_width, 0.25 * L), (x0 + half_width, 0.75 * L)]
        (prog_1, aut_1), (prog_2, aut_2) = self.anchor_points
        self.linear_aut_slope = (aut_2 - aut_1) / (prog_2 - prog_1)
        self.linear_aut_intercept = aut_1 - self.linear_aut_slope * prog_1
        self.exponential_aut_slope = (np.log(aut_2) - np.log(aut_1)) / (prog_2 - prog_1)

    def get_automation_fraction(self, progress):
        """
        Compute the automation fraction.
        Handles both scalar and array inputs.

        Args:
            progress: Progress value(s) (scalar or array)

        Returns:
            Automation fraction (scalar if input is scalar, array if input is array)
        """
        # Track if input was scalar
        input_was_scalar = np.ndim(progress) == 0

        # Ensure progress is an array
        progress = np.atleast_1d(progress)

        if self.schedule_type == "linear":
            result = np.clip(
                self.linear_aut_intercept + self.linear_aut_slope * progress,
                0.0,
                1.0
            )
        elif self.schedule_type == "logistic":
            # Logistic function: f(x) = L / (1 + exp(-k(x - x0))), clipped to [0, 1]
            exponent = -self.logistic_k * (progress - self.logistic_x0)
            # Clamp exponent to avoid overflow
            exponent = np.clip(exponent, -cfg.SIGMOID_EXPONENT_CLAMP, cfg.SIGMOID_EXPONENT_CLAMP)
            raw_value = self.logistic_L / (1 + np.exp(exponent))
            result = np.clip(raw_value, 0.0, 1.0)
        elif self.schedule_type == "exponential":
            raise NotImplementedError("Exponential schedule type not implemented")
        else:
            raise ValueError("Invalid schedule type")

        # Return scalar if input was scalar, array otherwise
        return float(result[0]) if input_was_scalar else result

    def get_progress_from_index(self, index):
        """
        Get the progress from the index.
        Handles both scalar and array inputs.

        Args:
            index: Index value(s) (scalar or array)

        Returns:
            Progress value(s) (scalar if input is scalar, array if input is array)
        """
        # Track if input was scalar
        input_was_scalar = np.ndim(index) == 0

        # Ensure index is an array
        indices = np.atleast_1d(index)

        if self.schedule_type == "linear":
            (prog_1, aut_1), (prog_2, aut_2) = self.anchor_points
            if abs(self.linear_aut_slope) < 1e-12:
                # Fallback: interpolate progress directly between anchors
                result = (1.0 - indices) * prog_1 + indices * prog_2
            else:
                prog_slope = 1.0 / self.linear_aut_slope
                prog_for_zero = prog_1 - prog_slope * aut_1
                prog_for_one = prog_2 + prog_slope * (1.0 - aut_2)
                result = indices * prog_for_one + (1.0 - indices) * prog_for_zero

        elif self.schedule_type == "logistic":
            # Clamp indices to valid range for logistic inverse
            clamped = np.clip(indices, 1e-9, self.logistic_L - 1e-9)
            if abs(self.logistic_k) < 1e-12:
                # k ~ 0, use midpoint
                result = np.full_like(indices, self.logistic_x0)
            else:
                ratio = self.logistic_L / clamped - 1.0
                # Handle ratio <= 0 case (index >= L)
                result = np.where(
                    ratio > 0,
                    self.logistic_x0 - np.log(np.maximum(ratio, 1e-300)) / self.logistic_k,
                    self.logistic_x0 + cfg.SIGMOID_EXPONENT_CLAMP / abs(self.logistic_k)
                )

        elif self.schedule_type == "exponential":
            raise NotImplementedError("Exponential schedule type not implemented")
        else:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")

        # Return scalar if input was scalar, array otherwise
        return float(result[0]) if input_was_scalar else result

    def get_progress_from_index_vectorized(self, indices: np.ndarray) -> np.ndarray:
        """
        Vectorized version of get_progress_from_index.
        Always returns an array, even for scalar input.

        Args:
            indices: Index value(s) as array

        Returns:
            Progress values as array
        """
        # This is just a wrapper that ensures array output
        # The get_progress_from_index method already handles arrays
        indices = np.atleast_1d(indices)
        result = self.get_progress_from_index(indices)
        return np.atleast_1d(result)

    def get_FTE_per_GPU(self, index:float, progress:float) -> float:
        """Compute the FTE per GPU for a given task index at a given E.C. level"""
        index_progress = self.get_progress_from_index(index)
        if progress < index_progress:
            return 0.0
        progress_diff = progress - index_progress
        growth_factor = cfg.BASE_FOR_SOFTWARE_LOM ** (self.FTE_per_GPU_slope * progress_diff)
        return self.initial_FTE_per_GPU * growth_factor

    def get_crit_index(self, progress:float, aut_compute: float, L_HUMAN: float, rho: float) -> float:
        """
        Compute the critical index for a given progress
        TODO: Implement actual optimization to find the critical index
        """
        return self.get_automation_fraction(progress)

    def get_compute_allocation(self, index:float, progress:float, aut_compute: float, L_HUMAN: float, crit_index:float, rho: float) -> float:
        """Compute the optimal compute allocation for a given task index."""
        return 0


    def get_coding_labor(self, crit_index:float, progress:float, aut_compute: float, L_HUMAN: float, rho: float) -> float:
        """Compute the coding labor for a given critical index and progress"""


        return self.get_FTE_per_GPU(crit_index, progress)

    # ===================== Optimal CES fast path (embedded) =====================
    class _FrontierPrecomp(NamedTuple):
        grid_i: np.ndarray
        log_Eaut: np.ndarray
        log_B: np.ndarray
        log_F: np.ndarray
        log_Q: np.ndarray
        log_R: np.ndarray
        rho: float
        theta: float
        eta_init: float
        eps_i1: float

    @staticmethod
    def _interp(x_grid: np.ndarray, y_grid: np.ndarray, x):
        """Linear interpolation on grid (handles both scalar and array inputs)."""
        # Handle both scalar and array inputs
        x_input = np.atleast_1d(x)
        is_scalar = np.ndim(x) == 0

        # Find indices for each x value
        j = np.searchsorted(x_grid, x_input, side='right') - 1
        j = np.clip(j, 0, len(x_grid) - 2)

        # Get endpoints
        x0 = x_grid[j]
        x1 = x_grid[j + 1]

        # Compute interpolation weights
        t = np.where(x1 != x0, (x_input - x0) / (x1 - x0), 0.0)
        t = np.clip(t, 0.0, 1.0)

        # Linear interpolation
        result = (1.0 - t) * y_grid[j] + t * y_grid[j + 1]

        # Return scalar if input was scalar
        return float(result[0]) if is_scalar else result

    @staticmethod
    def _invert_monotone(x_grid: np.ndarray, y_grid: np.ndarray, y):
        """Inverse interpolation on monotone grid (handles both scalar and array inputs)."""
        # Handle both scalar and array inputs
        y_input = np.atleast_1d(y)
        is_scalar = np.ndim(y) == 0

        # Use searchsorted to find bracketing indices
        hi = np.searchsorted(y_grid, y_input, side='left')
        lo = hi - 1

        # Clamp to valid range
        lo = np.clip(lo, 0, len(x_grid) - 2)
        hi = np.clip(hi, 1, len(x_grid) - 1)

        # Handle edge cases
        below_min = y_input <= y_grid[0]
        above_max = y_input >= y_grid[-1]

        # Get bracket values
        y0 = y_grid[lo]
        y1 = y_grid[hi]

        # Handle non-finite brackets to avoid NaN (vectorized version of old logic)
        # If y0 is -inf and y1 is finite, any finite y should map to hi
        # If y0 is finite and y1 is +inf, any finite y should map to lo
        y0_is_ninf = np.isinf(y0) & (y0 < 0)
        y1_is_pinf = np.isinf(y1) & (y1 > 0)
        y_is_finite = np.isfinite(y_input)

        # Check for error conditions that old version would have raised AssertionError for
        # Case 1: Both endpoints non-finite (and not the expected -inf/+inf pattern with finite y)
        y0_nonfinite = ~np.isfinite(y0)
        y1_nonfinite = ~np.isfinite(y1)
        # The "both non-finite" error occurs when we can't handle it with the -inf/+inf special cases
        # Old logic: if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y)) and neither
        # of the two special cases (-inf with finite y1, or +inf with finite y0) applies
        unhandled_nonfinite = (
            ~(np.isfinite(y0) & np.isfinite(y1) & y_is_finite) &  # Not all finite
            ~below_min & ~above_max &  # Not at edges (which are handled separately)
            ~(y0_is_ninf & y_is_finite & np.isfinite(y1)) &  # Not the -inf case
            ~(y1_is_pinf & y_is_finite & np.isfinite(y0))    # Not the +inf case
        )
        if np.any(unhandled_nonfinite):
            raise AssertionError("Non-finite endpoints in _invert_monotone")

        # Case 2: Denominator is 0 (y1 == y0) for points that aren't at edges
        denom = y1 - y0
        zero_denom = (denom == 0) & ~below_min & ~above_max
        if np.any(zero_denom):
            raise AssertionError("Denominator is 0 in _invert_monotone")

        # Compute interpolation weights
        # Use np.where to handle the division safely (edges are handled separately anyway)
        with np.errstate(invalid='ignore', divide='ignore'):
            t = np.where((denom != 0) & np.isfinite(denom), (y_input - y0) / denom, 0.0)
        t = np.clip(t, 0.0, 1.0)

        # Linear interpolation
        result = (1.0 - t) * x_grid[lo] + t * x_grid[hi]

        # Apply edge case results
        result = np.where(below_min, x_grid[0], result)
        result = np.where(above_max, x_grid[-1], result)

        # Handle non-finite brackets: if y0=-inf and y is finite, use hi
        result = np.where(y0_is_ninf & y_is_finite & np.isfinite(y1), x_grid[hi], result)
        # Handle non-finite brackets: if y1=+inf and y is finite, use lo
        result = np.where(y1_is_pinf & y_is_finite & np.isfinite(y0), x_grid[lo], result)

        # Return scalar if input was scalar
        return float(result[0]) if is_scalar else result

    def _precompute_frontier(self, params) -> Optional[_FrontierPrecomp]:
        try:
            M = int(max(256, min(int(params.optimal_ces_grid_size), 16384)))
            tail_eps = None
            tail_param = getattr(params, 'optimal_ces_frontier_tail_eps', None)
            if tail_param is not None:
                try:
                    tail_candidate = float(tail_param)
                    if np.isfinite(tail_candidate) and tail_candidate > 0.0:
                        tail_eps = tail_candidate
                except (TypeError, ValueError):
                    tail_eps = None
            cap_value = getattr(params, 'optimal_ces_frontier_cap', None)
            if cap_value is not None:
                try:
                    cap_float = float(cap_value)
                except (TypeError, ValueError):
                    cap_float = None
                if cap_float is not None and np.isfinite(cap_float) and cap_float > 0.0:
                    rho = float(params.rho_coding_labor)
                    denom = 1.0 - rho
                    if abs(denom) > 1e-12:
                        exponent = rho / denom
                        try:
                            tail_from_cap = float(np.power(cap_float, exponent))
                        except (OverflowError, ValueError):
                            tail_from_cap = None
                        if tail_from_cap is not None and np.isfinite(tail_from_cap) and tail_from_cap > 0.0:
                            tail_eps = tail_from_cap
                    else:
                        logger.warning("Unable to derive frontier tail epsilon from cap: rho too close to 1.0")
            if tail_eps is None:
                tail_eps = 1e-6
            bounds_map = getattr(cfg, 'PARAMETER_BOUNDS', {})
            tail_bounds = bounds_map.get('optimal_ces_frontier_tail_eps', (1e-12, 0.1))
            tail_low, tail_high = (float(tail_bounds[0]), float(tail_bounds[1])) if isinstance(tail_bounds, (list, tuple)) and len(tail_bounds) == 2 else (1e-12, 0.1)
            tail_low = max(min(tail_low, 0.1), 1e-24)
            tail_high = min(max(tail_high, tail_low * 10.0), 0.5)
            tail_eps = float(np.clip(tail_eps, tail_low, tail_high))
            grid_max = float(max(1e-6, 1.0 - tail_eps))
            grid_i = np.linspace(0.0, grid_max, M)

            # Build E_aut(i) using existing inverse mapping: get_progress_from_index(i)
            # Then convert progress (OOMs of effective compute) to effective compute.
            log_base = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM))
            # progress threshold at each index (OOMs of effective compute)
            # Use vectorized version for performance (avoids 512+ Python function calls)
            idx_progress = self.get_progress_from_index(grid_i)
            idx_progress = np.maximum.accumulate(idx_progress)
            # Effective capability threshold in log-space to avoid overflow
            log_Eaut = idx_progress * log_base

            rho = float(params.rho_coding_labor)
            theta = float(params.coding_automation_efficiency_slope)
            # Use FTE-per-GPU as baseline eta; allow multiplicative tweak via optimal_ces_eta_init
            eta_init = float(self.initial_FTE_per_GPU) * float(params.optimal_ces_eta_init)
            if not (rho < 1 and abs(rho) > 1e-18 and theta > 0 and eta_init > 0):
                return None

            di = grid_i[1] - grid_i[0]
            alpha = rho / (1.0 - rho)
            beta = alpha * theta
            gamma = theta / (1.0 - rho)

            log_w = -beta * log_Eaut          # can be very large, but safe
            log_di = np.log(di)
            log_half = np.log(0.5)

            # Vectorized log_B computation (cumulative logaddexp)
            # Compute all increments at once: log_inc[k] = log_di + log_half + logaddexp(log_w[k], log_w[k+1])
            log_inc = log_di + log_half + np.logaddexp(log_w[:-1], log_w[1:])
            # Cumulative logaddexp: log_B[k+1] = logaddexp(log_B[k], log_inc[k])
            # This is equivalent to log(cumsum(exp(log_inc))) but numerically stable
            log_B = np.empty(M, dtype=np.float64)
            log_B[0] = -np.inf  # B[0] = 0 => log_B[0] = -inf by convention
            # Vectorized cumulative logaddexp - numerically stable for any range
            log_B[1:] = np.logaddexp.accumulate(log_inc)

            one_minus_i = 1.0 - grid_i
            # log_Q = (1-rho) * log(1-i)
            log_one_minus_i = np.log(one_minus_i + 1e-18)
            log_Q = (1.0 - rho) * log_one_minus_i
            # log_R = log_Q - theta * log_Eaut
            log_R = log_Q - theta * log_Eaut
            # log_F = log_B + gamma * log_Eaut - log(1 - i)
            log_F = log_B + (gamma * log_Eaut) - log_one_minus_i
            # Enforce monotone in log-space
            log_F = np.maximum.accumulate(log_F)

            return AutomationModel._FrontierPrecomp(grid_i, log_Eaut, log_B, log_F, log_Q, log_R, rho, theta, eta_init, float(1.0 - grid_i[-1]))
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Frontier precompute failed: {e}")
            return None

    def _ensure_frontier(self, params) -> Optional[_FrontierPrecomp]:
        cap_value = getattr(params, 'optimal_ces_frontier_cap', None)
        cap_signature = None
        if cap_value is not None:
            try:
                cap_signature = float(cap_value)
            except (TypeError, ValueError):
                cap_signature = None
        max_serial_value = getattr(params, 'max_serial_coding_labor_multiplier', None)
        max_serial_signature = None
        if max_serial_value is not None:
            try:
                max_serial_signature = float(max_serial_value)
            except (TypeError, ValueError):
                max_serial_signature = None
        tail_signature = getattr(params, 'optimal_ces_frontier_tail_eps', 1e-6)
        try:
            tail_signature = float(tail_signature)
        except (TypeError, ValueError):
            tail_signature = 1e-6
        signature = (
            float(params.rho_coding_labor),
            float(params.coding_automation_efficiency_slope),
            float(params.optimal_ces_eta_init),
            int(params.optimal_ces_grid_size),
            tail_signature,
            cap_signature,
            max_serial_signature,
            tuple(self.anchor_points),
        )
        if self._frontier_pc is None or self._frontier_params_signature != signature:
            pc = self._precompute_frontier(params)
            if pc is None:
                self._frontier_pc = None
                self._frontier_params_signature = None
            else:
                self._frontier_pc = pc
                self._frontier_params_signature = signature
        return self._frontier_pc

    def coding_labor_optimal_ces(self, H, C, logE, params, return_details: bool = False):
        """Optimal CES coding labor computation (handles both scalar and array inputs).

        Args:
            H: Human labor (scalar or array)
            C: Compute (scalar or array)
            logE: Log of effective compute (scalar or array)
            params: Parameters object
            return_details: If True, return tuple (L, details_dict). Only works for scalar inputs.

        Returns:
            Coding labor value(s). Returns scalar if all inputs are scalar, otherwise array.
        """
        pc = self._ensure_frontier(params)
        if pc is None:
            return None if return_details else None

        try:
            # Determine if inputs are scalar
            H_input = np.atleast_1d(H)
            C_input = np.atleast_1d(C)
            logE_input = np.atleast_1d(logE)
            is_scalar = np.ndim(H) == 0 and np.ndim(C) == 0 and np.ndim(logE) == 0

            # For return_details mode, only support scalar inputs
            if return_details and not is_scalar:
                raise ValueError("return_details=True only supported for scalar inputs")

            # Broadcast to same shape
            N = max(len(H_input), len(C_input), len(logE_input))
            H_arr = np.broadcast_to(H_input, N) if len(H_input) == 1 else H_input
            C_arr = np.broadcast_to(C_input, N) if len(C_input) == 1 else C_input
            logE_arr = np.broadcast_to(logE_input, N) if len(logE_input) == 1 else logE_input

            # Find capability-limited indices for all logE values
            j = np.searchsorted(pc.log_Eaut, logE_arr, side='right') - 1
            j = np.clip(j, 0, len(pc.grid_i) - 2)

            # Interpolate i_E for all points
            i0, i1 = pc.grid_i[j], pc.grid_i[j + 1]
            le0, le1 = pc.log_Eaut[j], pc.log_Eaut[j + 1]
            tE = np.where(le1 != le0, np.clip((logE_arr - le0) / (le1 - le0), 0.0, 1.0), 0.0)
            i_E = (1.0 - tE) * i0 + tE * i1

            # Compute kappa and logS for all points
            kappa = C_arr / np.maximum(H_arr, 1e-18)
            logS = np.log(np.maximum(kappa * pc.eta_init, 1e-300)) + (pc.theta * logE_arr)

            # Interpolate log_F at i_E for all points
            log_F_iE = AutomationModel._interp(pc.grid_i, pc.log_F, i_E)

            # Determine cases with masks
            H_zero = H_arr <= 0.0
            interior_case = ~H_zero & (logS <= log_F_iE)
            boundary_case = ~H_zero & ~interior_case

            # Initialize results
            log_L = np.zeros(N)

            # Handle interior case
            if np.any(interior_case):
                i_c_int = AutomationModel._invert_monotone(pc.grid_i, pc.log_F, logS[interior_case])
                log_human_int = AutomationModel._interp(pc.grid_i, pc.log_Q, i_c_int)
                log_R_i_int = AutomationModel._interp(pc.grid_i, pc.log_R, i_c_int)
                log_comp_int = logS[interior_case] + log_R_i_int
                log_L_norm_rho_int = np.logaddexp(log_human_int, log_comp_int)
                log_L[interior_case] = np.log(H_arr[interior_case]) + (1.0 / pc.rho) * log_L_norm_rho_int

            # Handle boundary case
            if np.any(boundary_case):
                i_c_bnd = i_E[boundary_case]
                log_human_bnd = AutomationModel._interp(pc.grid_i, pc.log_Q, i_c_bnd)
                log_B_i_bnd = AutomationModel._interp(pc.grid_i, pc.log_B, i_c_bnd)
                log_comp_bnd = (pc.rho * logS[boundary_case]) + ((1.0 - pc.rho) * log_B_i_bnd)
                log_L_norm_rho_bnd = np.logaddexp(log_human_bnd, log_comp_bnd)
                log_L[boundary_case] = np.log(H_arr[boundary_case]) + (1.0 / pc.rho) * log_L_norm_rho_bnd

            # Convert to linear space
            L_arr = np.exp(log_L)

            # Handle H_zero case (must be done after exp to avoid -inf issues)
            L_arr[H_zero] = 0.0

            # Return details if requested (scalar only)
            if return_details:
                i_c = i_E[0] if not (interior_case[0] or boundary_case[0]) else (
                    i_c_int[0] if interior_case[0] else i_c_bnd[0]
                )
                case = "H_zero" if H_zero[0] else ("interior" if interior_case[0] else "boundary")
                log_human = log_human_int[0] if interior_case[0] else (log_human_bnd[0] if boundary_case[0] else 0.0)
                log_comp = log_comp_int[0] if interior_case[0] else (log_comp_bnd[0] if boundary_case[0] else 0.0)
                human = float(np.exp(log_human)) if not H_zero[0] else 0.0
                comp = float(np.exp(log_comp)) if not H_zero[0] else 0.0
                return float(L_arr[0]), {"i_cut": float(i_c), "i_E": float(i_E[0]), "case": case, "human_term": human, "comp_term": comp}

            # Return scalar if input was scalar
            return float(L_arr[0]) if is_scalar else L_arr

        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"coding_labor_optimal_ces failed at runtime: {e}")
            return None if not return_details else (None, {"error": str(e)})

    def compute_ai_only_coding_labor(self, C, logE, params, automation_fraction=None):
        """Compute AI-only coding labor (limit as H -> 0).

        In this branch (main), AI can only perform tasks up to the automation fraction.
        When automation_fraction < 1, a pure AI-only workforce cannot complete all tasks,
        so AI-only coding labor is 0.

        When automation_fraction = 1 (full automation), the formula is:
            L_ai_only = C * eta_init * E^theta * B(1)^((1-rho)/rho)

        In log-space:
            log_L_ai_only = log(C) + log(eta_init) + theta*logE + ((1-rho)/rho) * log_B[grid_max]

        Args:
            C: Inference compute (scalar or array)
            logE: Log of effective compute at the target progress level (scalar or array)
            params: Parameters object
            automation_fraction: Automation fraction(s) (scalar or array). If None, returns 0.

        Returns:
            AI-only coding labor value(s). Returns 0 when automation_fraction < 1.
            Returns scalar if all inputs are scalar, array otherwise.
        """
        # Handle scalar vs array inputs
        C_input = np.atleast_1d(C)
        logE_input = np.atleast_1d(logE)
        is_scalar = np.ndim(C) == 0 and np.ndim(logE) == 0

        if automation_fraction is None:
            # No automation fraction provided, return zeros
            result = np.zeros(max(len(C_input), len(logE_input)))
            return float(result[0]) if is_scalar else result

        aut_frac_input = np.atleast_1d(automation_fraction)
        is_scalar = is_scalar and np.ndim(automation_fraction) == 0

        # Broadcast to same shape
        N = max(len(C_input), len(logE_input), len(aut_frac_input))
        C_arr = np.broadcast_to(C_input, N) if len(C_input) == 1 else C_input
        logE_arr = np.broadcast_to(logE_input, N) if len(logE_input) == 1 else logE_input
        aut_frac_arr = np.broadcast_to(aut_frac_input, N) if len(aut_frac_input) == 1 else aut_frac_input

        # Initialize result array with zeros
        result = np.zeros(N)

        # Only compute for points where automation_fraction >= 1
        # Use a small tolerance for floating point comparison
        full_automation_mask = aut_frac_arr >= (1.0 - 1e-9)

        if not np.any(full_automation_mask):
            # No points have full automation, return zeros
            return float(result[0]) if is_scalar else result

        # Ensure frontier is computed
        pc = self._ensure_frontier(params)
        if pc is None:
            return float(result[0]) if is_scalar else result

        try:
            # Get log_B at grid_max (i = 1 - tail_eps, meaning AI does all tasks)
            log_B_at_max = pc.log_B[-1]

            # Compute exponent: (1 - rho) / rho
            if abs(pc.rho) < 1e-18:
                logger.warning("rho too close to zero in compute_ai_only_coding_labor")
                return float(result[0]) if is_scalar else result

            exponent = (1.0 - pc.rho) / pc.rho

            # Compute in log-space for numerical stability (only for full automation points)
            C_full = C_arr[full_automation_mask]
            logE_full = logE_arr[full_automation_mask]

            log_C = np.log(np.maximum(C_full, 1e-300))
            log_eta = np.log(max(pc.eta_init, 1e-300))

            log_L_ai_only = log_C + log_eta + pc.theta * logE_full + exponent * log_B_at_max

            # Cap exponent at 700 to avoid overflow (np.exp(710) → inf)
            log_L_ai_only = np.minimum(log_L_ai_only, 700.0)

            L_ai_only = np.exp(log_L_ai_only)

            # Replace non-finite values with 0
            L_ai_only = np.where(np.isfinite(L_ai_only), L_ai_only, 0.0)

            result[full_automation_mask] = L_ai_only

            return float(result[0]) if is_scalar else result

        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"compute_ai_only_coding_labor failed: {e}")
            return float(result[0]) if is_scalar else result


def aut_frac_from_swe_multiplier(swe_multiplier: float, L_HUMAN: float, inference_H100e: float, params: "Parameters") -> float:
    """
    Compute automation fraction from swe multiplier.

    Solve for A in:
      (swe_multiplier)**params.parallel_penalty * compute_coding_labor_deprecated(A, inference_H100e, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True) = compute_coding_labor_deprecated(A, inference_H100e, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization)
    where p = params.rho_coding_labor.
    Returns A in (0, 1). If there are multiple solutions, return the lower one.

    """
    # Input validation
    if not all(np.isfinite([swe_multiplier, L_HUMAN, inference_H100e])):
        logger.warning("Non-finite inputs to aut_frac_from_swe_multiplier")
        return 0.0

    if swe_multiplier <= 0 or L_HUMAN <= 0 or inference_H100e < 0:
        logger.warning("Invalid inputs to aut_frac_from_swe_multiplier")
        return 0.0

    # Target value we want to achieve
    target_output = swe_multiplier**params.parallel_penalty * compute_coding_labor_deprecated(0, inference_H100e, L_HUMAN, params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization, human_only=True)

    # Define the objective function to minimize
    def objective(A_candidate):
        """Return the difference between target and actual cognitive output"""
        try:
            actual_output = compute_coding_labor_deprecated(
                A_candidate, inference_H100e, L_HUMAN,
                params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
            )
            return actual_output - target_output
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Error in objective function: {e}")
            return float('inf')

    # Use bounds slightly inside (0, 1) to avoid numerical issues
    bounds = (cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)

    try:
        # Check if the function changes sign over the interval
        f_low = objective(bounds[0])
        f_high = objective(bounds[1])

        if f_low * f_high <= 0:
            # Sign change exists - we can find a root
            result = brentq(objective, bounds[0], bounds[1], xtol=1e-12, maxiter=100)
            result = np.clip(result, bounds[0], bounds[1])
            return float(result)
        else:
            # No sign change - target may not be achievable
            # Find the automation fraction that minimizes the absolute error
            result = minimize_scalar(
                lambda A: abs(objective(A)),
                bounds=bounds,
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 100}
            )

            if result.success:
                return float(np.clip(result.x, bounds[0], bounds[1]))
            else:
                raise RuntimeError("Optimization failed")

    except Exception as e:
        # Don't swallow timeout/interrupt exceptions - let them propagate
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.warning(f"Root finding/optimization failed in aut_frac_from_swe_multiplier: {e}")

        # Fallback: use grid search to find the best approximation
        try:
            A_candidates = np.linspace(bounds[0], bounds[1], 1000)
            # Vectorize grid search: compute_coding_labor_deprecated supports array inputs
            try:
                actual_outputs = compute_coding_labor_deprecated(
                    A_candidates, inference_H100e, L_HUMAN,
                    params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
                )
                errors = np.abs(actual_outputs - target_output)
            except Exception as vec_e:
                # Don't swallow timeout/interrupt exceptions - let them propagate
                if isinstance(vec_e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(vec_e).__name__:
                    raise
                # Fallback to element-wise if vectorization fails
                errors = np.array([abs(objective(A)) for A in A_candidates])

            best_idx = np.argmin(errors)
            result = A_candidates[best_idx]

            logger.info(f"Used grid search fallback, error: {errors[best_idx]}")
            return float(result)

        except Exception as e2:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e2, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e2).__name__:
                raise
            logger.warning(f"Grid search fallback also failed: {e2}")
            # Return a reasonable default
            return 0.5


def solve_lower_anchor_via_automation_model(
    swe_multiplier: float,
    anchor_progress: float,
    L_HUMAN: float,
    inference_H100e: float,
    params: "Parameters",
) -> float:
    """
    Solve for the lower anchor automation fraction at the anchor progress such that,
    when initializing AutomationModel with anchors
        { anchor_progress: A_lower, params.progress_at_aa: params.automation_fraction_at_coding_automation_anchor },
    the implied coding-labor multiplier at anchor_progress matches swe_multiplier.

    The multiplier is defined on coding labor after applying parallel_penalty and normalization, i.e.
        coding_labor_with_AI = swe_multiplier**parallel_penalty * coding_labor_human_only.

    If the optimal-CES frontier is unavailable for given parameters, falls back to the simple CES formulation.
    Returns a clipped value in (cfg.AUTOMATION_FRACTION_CLIP_MIN, 1 - cfg.AUTOMATION_FRACTION_CLIP_MIN).
    """
    # Profiling: track overall function time
    _prof_start = time.perf_counter()
    _prof_setup_time = 0.0
    _prof_implied_ratio_time = 0.0
    _prof_implied_ratio_calls = 0
    _prof_anchor_build_time = 0.0
    _prof_automationmodel_init_time = 0.0
    _prof_coding_labor_optimal_ces_time = 0.0
    _prof_solver_time = 0.0
    _prof_solver_method = "none"

    try:
        _prof_setup_start = time.perf_counter()
        # Validate inputs
        if not all(np.isfinite([swe_multiplier, anchor_progress, L_HUMAN, inference_H100e])):
            logger.warning("Non-finite inputs to solve_lower_anchor_via_automation_model")
            return 0.01
        if swe_multiplier <= 0 or L_HUMAN <= 0 or inference_H100e < 0:
            logger.warning("Invalid inputs to solve_lower_anchor_via_automation_model")
            return 0.01

        # Ensure an upper anchor exists
        progress_at_aa = getattr(params, 'progress_at_aa', None)
        aut_at_sc = getattr(params, 'automation_fraction_at_coding_automation_anchor', None)
        if progress_at_aa is None or not np.isfinite(progress_at_aa) or aut_at_sc is None:
            logger.warning("Missing progress_at_aa or automation_fraction_at_coding_automation_anchor; falling back to direct solver")
            return aut_frac_from_swe_multiplier(swe_multiplier, L_HUMAN, inference_H100e, params)

        # Target coding-labor ratio in parallel_penalty space
        target_ratio = float(np.power(swe_multiplier, params.parallel_penalty))

        # Baseline human-only coding labor (consistent with definition of multiplier)
        baseline = compute_coding_labor_deprecated(
            0, inference_H100e, L_HUMAN,
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization,
            human_only=True
        )
        if baseline <= 0 or not np.isfinite(baseline):
            logger.warning("Invalid baseline coding labor in anchor solver; using fallback")
            return aut_frac_from_swe_multiplier(swe_multiplier, L_HUMAN, inference_H100e, params)

        logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * anchor_progress)
        _prof_setup_time = time.perf_counter() - _prof_setup_start

        def implied_ratio_for_anchor(a_lower: float) -> float:
            nonlocal _prof_implied_ratio_calls, _prof_implied_ratio_time
            nonlocal _prof_anchor_build_time, _prof_automationmodel_init_time, _prof_coding_labor_optimal_ces_time
            _prof_call_start = time.perf_counter()
            _prof_implied_ratio_calls += 1

            # Build the candidate anchor pair (passed explicitly so we don't have
            # to deep-copy the whole params object once per solver iteration)
            _prof_ab_start = time.perf_counter()
            # Clip candidate to safe open interval
            a_clipped = float(np.clip(a_lower, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN))
            candidate_anchors = {
                float(anchor_progress): a_clipped,
                float(progress_at_aa): float(np.clip(aut_at_sc, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)),
            }
            _prof_anchor_build_time += time.perf_counter() - _prof_ab_start
            try:
                _prof_am_start = time.perf_counter()
                am = AutomationModel(params, anchors=candidate_anchors)
                _prof_automationmodel_init_time += time.perf_counter() - _prof_am_start

                H = float(L_HUMAN)
                C = float(inference_H100e)

                _prof_ces_start = time.perf_counter()
                L_opt = am.coding_labor_optimal_ces(H, C, logE, params)
                _prof_coding_labor_optimal_ces_time += time.perf_counter() - _prof_ces_start

                if L_opt is None or not np.isfinite(L_opt):
                    # Fallback to simple CES using the schedule's automation at the anchor (which equals a_lower)
                    A = am.get_automation_fraction(anchor_progress)
                    L_ai = compute_coding_labor_deprecated(
                        A, inference_H100e, L_HUMAN,
                        params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
                    )
                else:
                    # Match units with compute_coding_labor_deprecated
                    L_ai = float((L_opt ** params.parallel_penalty) * params.coding_labor_normalization)
                if not np.isfinite(L_ai) or L_ai <= 0:
                    _prof_implied_ratio_time += time.perf_counter() - _prof_call_start
                    return 0.0
                _prof_implied_ratio_time += time.perf_counter() - _prof_call_start
                return float(L_ai / baseline)
            except Exception as e:
                # Don't swallow timeout/interrupt exceptions - let them propagate
                if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                    raise
                logger.warning(f"Error computing implied ratio for anchor {a_lower}: {e}")
                _prof_implied_ratio_time += time.perf_counter() - _prof_call_start
                return 0.0

        def objective(a_lower: float) -> float:
            return implied_ratio_for_anchor(a_lower) - target_ratio

        # Bounds slightly inside (0, 1) to avoid numerical issues
        lo = float(cfg.AUTOMATION_FRACTION_CLIP_MIN)
        hi = float(1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)

        try:
            _prof_solver_start = time.perf_counter()
            f_lo = objective(lo)
            f_hi = objective(hi)
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0:
                _prof_solver_method = "brentq"
                root = brentq(objective, lo, hi, xtol=1e-8, maxiter=100)
                _prof_solver_time = time.perf_counter() - _prof_solver_start
                result = float(np.clip(root, lo, hi))
                _log_profiling_results(_prof_start, _prof_setup_time, _prof_implied_ratio_time,
                                       _prof_implied_ratio_calls, _prof_anchor_build_time,
                                       _prof_automationmodel_init_time, _prof_coding_labor_optimal_ces_time,
                                       _prof_solver_time, _prof_solver_method)
                return result

            # If no sign change, minimize absolute error
            _prof_solver_method = "minimize_scalar"
            res = minimize_scalar(lambda x: abs(objective(x)), bounds=(lo, hi), method='bounded', options={'xatol': 1e-8, 'maxiter': 200})
            _prof_solver_time = time.perf_counter() - _prof_solver_start
            if getattr(res, 'success', False):
                result = float(np.clip(res.x, lo, hi))
                _log_profiling_results(_prof_start, _prof_setup_time, _prof_implied_ratio_time,
                                       _prof_implied_ratio_calls, _prof_anchor_build_time,
                                       _prof_automationmodel_init_time, _prof_coding_labor_optimal_ces_time,
                                       _prof_solver_time, _prof_solver_method)
                return result
            else:
                raise RuntimeError("Anchor solver bounded minimization failed")
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"Anchor solver root-finding failed; using grid search fallback: {e}")
            try:
                _prof_solver_start = time.perf_counter()
                _prof_solver_method = "grid_search"
                grid = np.linspace(lo, hi, 512)
                # Note: objective builds an AutomationModel per candidate, cannot be vectorized
                errs = np.array([abs(objective(a)) for a in grid])
                idx = int(np.argmin(errs))
                _prof_solver_time = time.perf_counter() - _prof_solver_start
                result = float(grid[idx])
                _log_profiling_results(_prof_start, _prof_setup_time, _prof_implied_ratio_time,
                                       _prof_implied_ratio_calls, _prof_anchor_build_time,
                                       _prof_automationmodel_init_time, _prof_coding_labor_optimal_ces_time,
                                       _prof_solver_time, _prof_solver_method)
                return result
            except Exception as e2:
                # Don't swallow timeout/interrupt exceptions - let them propagate
                if isinstance(e2, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e2).__name__:
                    raise
                logger.warning(f"Grid search fallback failed in anchor solver: {e2}")
                return 0.01
    except Exception as e:
        # Don't swallow timeout/interrupt exceptions - let them propagate
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.warning(f"solve_lower_anchor_via_automation_model failed unexpectedly: {e}")
        return 0.01


# ===================== uplift-solved logistic schedule =====================
#
# The 'uplift-solved' AC-anchor mode inverts the usual direction of the fit:
# instead of taking progress_at_aa from an exogenous observable and solving for
# the lower automation anchor, the user states the model's own ENDOGENOUS serial
# ref-present-day coding-labor multiplier M at a reference progress level, plus
# the doubling rate of its excess (M - 1) there, and we solve the logistic
# schedule's (k, x0) to match both. progress_at_aa then becomes an output.

# Half-width of the centred finite difference used for d ln(M - 1)/dp, in
# progress units (OOMs of cumulative effective compute).
UPLIFT_SOLVE_FD_STEP = 1e-3
# Newton is declared converged when both (dimensionless) residuals are below this.
UPLIFT_SOLVE_TOL = 1e-9
# Residual level we are still willing to return (with a warning) after the
# least-squares fallback has run.
UPLIFT_SOLVE_ACCEPT_TOL = 1e-6
UPLIFT_SOLVE_MAX_ITER = 60
# Bounds on the logistic steepness k (progress units^-1); also clamp the seed.
UPLIFT_SOLVE_K_MIN = 1e-4
UPLIFT_SOLVE_K_MAX = 1e4


def serial_uplift_for_logistic(
    progress_values,
    k: float,
    x0: float,
    asymptote_L: float,
    present_day_human_labor: float,
    present_day_inference_compute: float,
    params: "Parameters",
    baseline: Optional[float] = None,
) -> np.ndarray:
    """
    The model's endogenous serial ref-present-day coding-labor multiplier M(p)
    under the logistic automation schedule A(p) = L / (1 + exp(-k(p - x0))).

    "Ref-present-day" means human labor and inference compute are held at their
    present-day levels while only the automation schedule and the effective
    compute level move with p. This is exactly the ratio that
    solve_lower_anchor_via_automation_model's implied_ratio_for_anchor targets
    at the present day, i.e. the quantity that equals
    swe_multiplier_at_present_day ** parallel_penalty there. Keeping the two
    definitions identical is what makes the two AC-anchor modes comparable.

        L_opt   = automation_model.coding_labor_optimal_ces(H0, C0, ln(10) * p)
        L_ai    = L_opt ** parallel_penalty * coding_labor_normalization
        M(p)    = L_ai / human-only baseline coding labor

    Falls back to the simple CES formulation (as implied_ratio_for_anchor does)
    wherever the optimal-CES frontier is unavailable or non-finite.

    Args:
        progress_values: scalar or array of progress levels (OOMs).
        k, x0, asymptote_L: logistic schedule parameters.
        present_day_human_labor / present_day_inference_compute: H0 / C0.
        params: Parameters object.
        baseline: precomputed human-only baseline coding labor; computed here
            when omitted.

    Returns:
        Array of multipliers, same length as progress_values.
    """
    H = float(present_day_human_labor)
    C = float(present_day_inference_compute)
    if baseline is None:
        baseline = float(compute_coding_labor_deprecated(
            0, C, H,
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization,
            human_only=True
        ))
    if not (np.isfinite(baseline) and baseline > 0):
        raise ValueError(f"Invalid human-only baseline coding labor: {baseline}")

    am = AutomationModel(params, logistic=(asymptote_L, k, x0))
    return serial_uplift_for_schedule(
        progress_values, am, present_day_human_labor, present_day_inference_compute,
        params, baseline=baseline)


def serial_uplift_for_schedule(
    progress_values,
    automation_model,
    present_day_human_labor: float,
    present_day_inference_compute: float,
    params: "Parameters",
    baseline: Optional[float] = None,
) -> np.ndarray:
    """M(p) for an ALREADY-DERIVED automation schedule.

    Same quantity as serial_uplift_for_logistic -- that function is now this one with
    the schedule built from a (L, k, x0) triple. Split out so the T_d correction pass
    can measure what the schedule it just derived actually produces, whatever route
    produced it (the uplift solve's logistic, or an anchor-fitted AutomationModel).

    Pure evaluation: CES calls on the given progress values, no integration.
    """
    H = float(present_day_human_labor)
    C = float(present_day_inference_compute)
    if baseline is None:
        baseline = float(compute_coding_labor_deprecated(
            0, C, H,
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization,
            human_only=True
        ))
    if not (np.isfinite(baseline) and baseline > 0):
        raise ValueError(f"Invalid human-only baseline coding labor: {baseline}")

    progress_arr = np.atleast_1d(np.asarray(progress_values, dtype=float))
    am = automation_model
    logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM)) * progress_arr

    L_opt = am.coding_labor_optimal_ces(H, C, logE, params)
    if L_opt is None:
        usable = np.zeros(progress_arr.shape, dtype=bool)
        L_ai = np.empty(progress_arr.shape, dtype=float)
    else:
        L_opt = np.atleast_1d(np.asarray(L_opt, dtype=float))
        usable = np.isfinite(L_opt)
        L_ai = np.empty(progress_arr.shape, dtype=float)
        L_ai[usable] = (
            np.power(L_opt[usable], params.parallel_penalty) * params.coding_labor_normalization
        )

    if not np.all(usable):
        # Simple-CES fallback at the automation fraction the schedule implies
        A_fallback = np.atleast_1d(am.get_automation_fraction(progress_arr[~usable]))
        L_ai[~usable] = np.atleast_1d(compute_coding_labor_deprecated(
            A_fallback, C, H,
            params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization
        ))

    return L_ai / baseline


def _seed_logistic_from_uplift(
    target_serial_uplift: float,
    reference_progress: float,
    doubling_progress_units: float,
    asymptote_L: float,
    H: float,
    C: float,
    baseline: float,
    params: "Parameters",
) -> Tuple[float, float]:
    """
    Cheap (k, x0) seed for the 2-D solve, ignoring the frontier-shape dependence
    of M on the whole schedule below p*: pretend M depends on p only through
    A(p), invert M* to an automation fraction A* with the simple CES, then

        A'(p*) = (dM/dp)|target / (dM/dA)      with (dM/dp) = (M* - 1) ln2 / D
        k      = A'(p*) / (A* (1 - A*/L))      (logistic slope identity)
        x0     = p* + ln(L/A* - 1) / k
    """
    parallel_penalty = float(params.parallel_penalty)
    swe_multiplier_guess = float(target_serial_uplift) ** (1.0 / parallel_penalty)
    A_star = float(aut_frac_from_swe_multiplier(swe_multiplier_guess, H, C, params))
    A_star = float(np.clip(
        A_star,
        cfg.AUTOMATION_FRACTION_CLIP_MIN,
        min(1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN, asymptote_L * (1.0 - 1e-9)),
    ))

    def simple_multiplier(A):
        return float(compute_coding_labor_deprecated(
            A, C, H,
            params.rho_coding_labor, parallel_penalty, params.coding_labor_normalization
        )) / baseline

    h = min(1e-5, 0.5 * A_star, 0.5 * (1.0 - A_star))
    dM_dA = (simple_multiplier(A_star + h) - simple_multiplier(A_star - h)) / (2.0 * h)
    dM_dp = (float(target_serial_uplift) - 1.0) * np.log(2.0) / float(doubling_progress_units)

    k_seed = np.nan
    if np.isfinite(dM_dA) and dM_dA > 0:
        A_prime = dM_dp / dM_dA
        denom = A_star * (1.0 - A_star / asymptote_L)
        if denom > 0:
            k_seed = A_prime / denom
    if not (np.isfinite(k_seed) and k_seed > 0):
        # Routine with strong complementarity (very negative rho_coding_labor):
        # the simple CES multiplier plateaus in A, so its slope carries no signal.
        # Newton usually recovers from k=1; the bracketed seed catches it if not.
        logger.info(
            f"uplift-solved seed: could not derive k from the chain rule "
            f"(A*={A_star:.6g}, dM/dA={dM_dA:.6g}); using k=1.0"
        )
        k_seed = 1.0
    k_seed = float(np.clip(k_seed, UPLIFT_SOLVE_K_MIN, UPLIFT_SOLVE_K_MAX))

    x0_seed = float(reference_progress) + np.log(asymptote_L / A_star - 1.0) / k_seed
    return k_seed, x0_seed


def _bracketed_seed_logistic_from_uplift(
    target_serial_uplift: float,
    reference_progress: float,
    target_slope: float,
    asymptote_L: float,
    H: float,
    C: float,
    baseline: float,
    params: "Parameters",
    fd_step: float,
) -> Optional[Tuple[float, float]]:
    """
    Robust (but pricier) seed that exploits the problem's monotone structure.

    Along the manifold M(p*) = M*, x0 is pinned by k, and the achieved
    d ln(M - 1)/dp increases monotonically in k. So the 2-D problem collapses to
    a bracketed 1-D root find: brentq on ln k, with an inner brentq on the
    reference automation fraction a = A(p*) enforcing M(p*) = M*.

    Needed because the cheap chain-rule seed degenerates whenever the *simple*
    CES multiplier plateaus in A (strong complementarity, i.e. very negative
    rho_coding_labor) — dM/dA there is numerically zero even though the true
    optimal-CES multiplier keeps climbing, which throws k to its clamp.
    """
    a_lo = cfg.AUTOMATION_FRACTION_CLIP_MIN
    a_hi = asymptote_L * (1.0 - 1e-12)
    probe = np.array([reference_progress - fd_step, reference_progress,
                      reference_progress + fd_step], dtype=float)

    def x0_from_a(a: float, k: float) -> float:
        return reference_progress + np.log(asymptote_L / a - 1.0) / k

    def x0_for_k(k: float) -> Optional[float]:
        """Solve A(p*) = a such that M(p*) = M*; M is increasing in a."""
        def f(a: float) -> float:
            M = serial_uplift_for_logistic(
                [reference_progress], k, x0_from_a(a, k), asymptote_L, H, C, params, baseline=baseline
            )
            return float(M[0]) - target_serial_uplift

        try:
            f_lo, f_hi = f(a_lo), f(a_hi)
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            return None
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            return None
        try:
            a_root = brentq(f, a_lo, a_hi, xtol=1e-14, rtol=1e-14, maxiter=100)
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            return None
        return float(x0_from_a(float(a_root), k))

    cache: Dict[float, Optional[Tuple[float, float]]] = {}

    def slope_for_log_k(log_k: float) -> Optional[Tuple[float, float]]:
        if log_k in cache:
            return cache[log_k]
        k = float(np.exp(log_k))
        result = None
        x0 = x0_for_k(k)
        if x0 is not None:
            excess = serial_uplift_for_logistic(
                probe, k, x0, asymptote_L, H, C, params, baseline=baseline
            ) - 1.0
            if np.all(np.isfinite(excess)) and np.all(excess > 0):
                slope = float(np.log(excess[2]) - np.log(excess[0])) / (2.0 * fd_step)
                if np.isfinite(slope):
                    result = (slope, x0)
        cache[log_k] = result
        return result

    def g(log_k: float) -> float:
        got = slope_for_log_k(log_k)
        if got is None:
            raise ValueError(f"infeasible k={np.exp(log_k):.6g}")
        return got[0] - target_slope

    # Find a bracket on a geometric ladder over k, skipping infeasible rungs.
    ladder = np.log(np.geomspace(UPLIFT_SOLVE_K_MIN, UPLIFT_SOLVE_K_MAX, 25))
    feasible = []
    for log_k in ladder:
        got = slope_for_log_k(float(log_k))
        if got is not None:
            feasible.append((float(log_k), got[0]))
    if len(feasible) < 2:
        logger.warning(
            f"uplift-solved bracketed seed: only {len(feasible)} feasible k values on the ladder "
            f"(M*={target_serial_uplift}, target slope={target_slope:.6g})"
        )
        return None

    bracket = None
    for (lk_a, s_a), (lk_b, s_b) in zip(feasible[:-1], feasible[1:]):
        if (s_a - target_slope) * (s_b - target_slope) <= 0:
            bracket = (lk_a, lk_b)
            break
    if bracket is None:
        reachable = (min(s for _, s in feasible), max(s for _, s in feasible))
        logger.warning(
            f"uplift-solved bracketed seed: target slope {target_slope:.6g} "
            f"(D={np.log(2.0)/target_slope:.6g} progress units) is outside the reachable range "
            f"[{reachable[0]:.6g}, {reachable[1]:.6g}] for M*={target_serial_uplift} — "
            f"no logistic schedule matches both conditions"
        )
        return None

    try:
        log_k_root = brentq(g, bracket[0], bracket[1], xtol=1e-13, rtol=1e-14, maxiter=100)
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.warning(f"uplift-solved bracketed seed: outer brentq failed: {e}")
        return None

    got = slope_for_log_k(float(log_k_root))
    if got is None:
        return None
    return float(np.exp(log_k_root)), float(got[1])


def solve_logistic_from_uplift(
    target_serial_uplift: float,
    reference_progress: float,
    doubling_progress_units: float,
    asymptote_L: float,
    present_day_human_labor: float,
    present_day_inference_compute: float,
    params: "Parameters",
    fd_step: float = UPLIFT_SOLVE_FD_STEP,
) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """
    Solve the logistic automation schedule's (k, x0) so that the model's own
    endogenous serial ref-present-day uplift M satisfies, at p* = reference_progress:

        M(p*)                  = target_serial_uplift              (M*)
        d ln(M - 1)/dp |_{p*}  = ln 2 / doubling_progress_units    (excess doubles every D)

    This is a genuine 2-D solve rather than a closed form: M(p*) depends on the
    entire schedule below p* through the optimal-CES frontier (the frontier
    integrates over task indices whose capability thresholds are the schedule's
    inverse), not only on A(p*). Every residual evaluation therefore builds a
    candidate AutomationModel — but only one per evaluation, since M at
    p* - delta, p*, p* + delta comes from a single vectorized frontier call.

    Method: damped Newton on u = (ln k, x0) — the log parametrization enforces
    k > 0 — with a finite-difference Jacobian (2 extra model builds per step),
    falling back to scipy.optimize.least_squares within bounds if Newton stalls.

    Residuals are reported raw and used normalized (both dimensionless, so one
    tolerance is meaningful for both):

        r1 = ln M(p*) - ln M*
        r2 = (centred FD of ln(M - 1) at p*) / (ln 2 / D) - 1

    Returns:
        (k, x0, diagnostics) on success, or None if no acceptable solve was
        found (a warning with diagnostics is logged in that case).
    """
    target_serial_uplift = float(target_serial_uplift)
    reference_progress = float(reference_progress)
    doubling_progress_units = float(doubling_progress_units)
    asymptote_L = float(asymptote_L)
    H = float(present_day_human_labor)
    C = float(present_day_inference_compute)
    fd_step = float(fd_step)

    if not np.isfinite([target_serial_uplift, reference_progress, doubling_progress_units,
                        asymptote_L, H, C, fd_step]).all():
        logger.warning("solve_logistic_from_uplift: non-finite inputs")
        return None
    if not target_serial_uplift > 1.0:
        logger.warning(f"solve_logistic_from_uplift: target serial uplift must exceed 1, got {target_serial_uplift}")
        return None
    if not doubling_progress_units > 0:
        logger.warning(f"solve_logistic_from_uplift: doubling_progress_units must be positive, got {doubling_progress_units}")
        return None
    if not asymptote_L > 1.0:
        # p_AC = x0 - ln(L - 1)/k only exists when the logistic actually crosses 1.
        logger.warning(f"solve_logistic_from_uplift: asymptote must exceed 1, got {asymptote_L}")
        return None
    if not (H > 0 and C >= 0 and fd_step > 0):
        logger.warning("solve_logistic_from_uplift: invalid H0/C0/fd_step")
        return None

    baseline = float(compute_coding_labor_deprecated(
        0, C, H,
        params.rho_coding_labor, params.parallel_penalty, params.coding_labor_normalization,
        human_only=True
    ))
    if not (np.isfinite(baseline) and baseline > 0):
        logger.warning(f"solve_logistic_from_uplift: invalid human-only baseline {baseline}")
        return None

    target_slope = np.log(2.0) / doubling_progress_units
    probe_progress = np.array([
        reference_progress - fd_step, reference_progress, reference_progress + fd_step
    ], dtype=float)

    n_builds = 0

    def measure(u: np.ndarray) -> Optional[Tuple[float, float]]:
        """(M at p*, centred FD slope of ln(M - 1) at p*) for u = (ln k, x0)."""
        nonlocal n_builds
        log_k = float(u[0])
        # Range-check in log space so a wild Newton step can't overflow np.exp
        if not (np.isfinite(log_k) and np.isfinite(u[1])):
            return None
        if not (np.log(UPLIFT_SOLVE_K_MIN) - 7.0 <= log_k <= np.log(UPLIFT_SOLVE_K_MAX) + 7.0):
            return None
        k = float(np.exp(log_k))
        try:
            n_builds += 1
            M = serial_uplift_for_logistic(
                probe_progress, k, float(u[1]), asymptote_L, H, C, params, baseline=baseline
            )
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.debug(f"solve_logistic_from_uplift: model build failed at {u}: {e}")
            return None
        excess = np.asarray(M, dtype=float) - 1.0
        if not np.all(np.isfinite(excess)) or np.any(excess <= 0.0):
            return None
        slope = float(np.log(excess[2]) - np.log(excess[0])) / (2.0 * fd_step)
        return float(M[1]), slope

    def residuals(u: np.ndarray) -> Optional[np.ndarray]:
        got = measure(u)
        if got is None:
            return None
        M_ref, slope = got
        r1 = float(np.log(M_ref) - np.log(target_serial_uplift))
        r2 = float(slope / target_slope - 1.0)
        if not (np.isfinite(r1) and np.isfinite(r2)):
            return None
        return np.array([r1, r2], dtype=float)

    best_u = None
    best_norm = np.inf
    newton_iters = 0
    jac_step = np.array([1e-5, 1e-5], dtype=float)

    def run_newton(u_start: np.ndarray) -> None:
        """Damped Newton on (ln k, x0); updates best_u / best_norm / newton_iters."""
        nonlocal best_u, best_norm, newton_iters
        u = np.asarray(u_start, dtype=float).copy()
        r = residuals(u)
        if r is not None and float(np.max(np.abs(r))) < best_norm:
            best_u, best_norm = u.copy(), float(np.max(np.abs(r)))

        for _ in range(UPLIFT_SOLVE_MAX_ITER):
            if r is None or float(np.max(np.abs(r))) <= UPLIFT_SOLVE_TOL:
                break
            newton_iters += 1

            J = np.empty((2, 2), dtype=float)
            jac_ok = True
            for j in range(2):
                u_pert = u.copy()
                u_pert[j] += jac_step[j]
                r_pert = residuals(u_pert)
                if r_pert is None:
                    u_pert[j] = u[j] - jac_step[j]
                    r_pert = residuals(u_pert)
                    if r_pert is None:
                        jac_ok = False
                        break
                    J[:, j] = (r - r_pert) / jac_step[j]
                else:
                    J[:, j] = (r_pert - r) / jac_step[j]
            if not jac_ok or not np.all(np.isfinite(J)):
                break

            try:
                delta = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
            if not np.all(np.isfinite(delta)):
                break

            # Backtracking line search on max|r|
            step = 1.0
            improved = False
            while step >= 1.0 / 1024.0:
                u_try = u + step * delta
                r_try = residuals(u_try)
                if r_try is not None and float(np.max(np.abs(r_try))) < float(np.max(np.abs(r))):
                    u, r = u_try, r_try
                    improved = True
                    break
                step *= 0.5
            if float(np.max(np.abs(r))) < best_norm:
                best_u, best_norm = u.copy(), float(np.max(np.abs(r)))
            if not improved:
                break

    # ---- cheap chain-rule seed + Newton (the fast path: ~12 model builds) ----
    k_seed, x0_seed = _seed_logistic_from_uplift(
        target_serial_uplift, reference_progress, doubling_progress_units,
        asymptote_L, H, C, baseline, params
    )
    run_newton(np.array([np.log(k_seed), x0_seed], dtype=float))
    method = "newton"

    # ---- bracketed 1-D pre-solve, then Newton again, if the cheap seed missed ----
    if not (best_norm <= UPLIFT_SOLVE_TOL):
        logger.info(
            f"uplift-solved: Newton from the chain-rule seed stalled at max|r|={best_norm:.3g}; "
            f"retrying from a bracketed 1-D seed"
        )
        bracketed = _bracketed_seed_logistic_from_uplift(
            target_serial_uplift, reference_progress, target_slope,
            asymptote_L, H, C, baseline, params, fd_step
        )
        if bracketed is not None:
            k_seed, x0_seed = bracketed
            run_newton(np.array([np.log(k_seed), x0_seed], dtype=float))
            method = "bracketed+newton"

    # ---- least-squares fallback if both Newton runs stalled short of tolerance ----
    if not (best_norm <= UPLIFT_SOLVE_TOL):
        logger.info(
            f"uplift-solved: stalled at max|r|={best_norm:.3g} after {newton_iters} "
            f"Newton iterations; trying least_squares"
        )
        start = best_u if best_u is not None else np.array([np.log(k_seed), x0_seed])
        lo = np.array([np.log(UPLIFT_SOLVE_K_MIN), reference_progress - 500.0])
        hi = np.array([np.log(UPLIFT_SOLVE_K_MAX), reference_progress + 500.0])
        start = np.clip(start, lo + 1e-9, hi - 1e-9)

        def ls_residuals(u_vec):
            out = residuals(np.asarray(u_vec, dtype=float))
            # Large-but-finite penalty keeps least_squares inside the feasible region
            return np.array([1e3, 1e3]) if out is None else out

        try:
            res = least_squares(ls_residuals, start, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14)
            r_ls = residuals(np.asarray(res.x, dtype=float))
            if r_ls is not None and float(np.max(np.abs(r_ls))) < best_norm:
                best_u = np.asarray(res.x, dtype=float)
                best_norm = float(np.max(np.abs(r_ls)))
                method = "least_squares"
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning(f"uplift-solved: least_squares fallback failed: {e}")

    if best_u is None or not np.isfinite(best_norm):
        logger.warning(
            f"solve_logistic_from_uplift failed: no feasible (k, x0) found "
            f"(M*={target_serial_uplift}, p*={reference_progress}, D={doubling_progress_units}, "
            f"L={asymptote_L}, seed k={k_seed:.6g} x0={x0_seed:.6g}, builds={n_builds})"
        )
        return None

    k_solved = float(np.exp(best_u[0]))
    x0_solved = float(best_u[1])
    measured = measure(best_u)
    M_ref, slope = measured if measured is not None else (float('nan'), float('nan'))
    p_ac = x0_solved - np.log(asymptote_L - 1.0) / k_solved
    achieved_doubling = float(np.log(2.0) / slope) if np.isfinite(slope) and slope > 0 else float('nan')

    diagnostics = {
        'method': method,
        'converged': bool(best_norm <= UPLIFT_SOLVE_TOL),
        'newton_iterations': newton_iters,
        'model_builds': n_builds,
        'max_abs_residual': best_norm,
        'residual_log_uplift': float(np.log(M_ref) - np.log(target_serial_uplift)) if np.isfinite(M_ref) else float('nan'),
        'residual_slope_raw': float(slope - target_slope) if np.isfinite(slope) else float('nan'),
        'residual_slope_normalized': float(slope / target_slope - 1.0) if np.isfinite(slope) else float('nan'),
        'k': k_solved,
        'x0': x0_solved,
        'asymptote_L': asymptote_L,
        'reference_progress': reference_progress,
        'target_serial_uplift': target_serial_uplift,
        'achieved_serial_uplift': M_ref,
        'target_doubling_progress_units': doubling_progress_units,
        'achieved_doubling_progress_units': achieved_doubling,
        'automation_fraction_at_reference': float(
            AutomationModel(params, logistic=(asymptote_L, k_solved, x0_solved))
            .get_automation_fraction(reference_progress)
        ),
        'progress_at_automation_crossing': float(p_ac),
        'seed_k': float(k_seed),
        'seed_x0': float(x0_seed),
        'fd_step': fd_step,
    }

    if best_norm > UPLIFT_SOLVE_ACCEPT_TOL:
        logger.warning(f"solve_logistic_from_uplift did not converge: {diagnostics}")
        return None
    if not diagnostics['converged']:
        logger.warning(
            f"solve_logistic_from_uplift converged only loosely (max|r|={best_norm:.3g}): {diagnostics}"
        )
    else:
        logger.info(
            f"uplift-solved logistic: k={k_solved:.6f}, x0={x0_solved:.6f}, "
            f"A(p*)={diagnostics['automation_fraction_at_reference']:.6f}, "
            f"M(p*)={M_ref:.6f} (target {target_serial_uplift:.6f}), "
            f"D_achieved={achieved_doubling:.6f} progress units (target {doubling_progress_units:.6f}), "
            f"p_AC={p_ac:.6f}, {newton_iters} Newton iterations, {n_builds} model builds"
        )
    return k_solved, x0_solved, diagnostics


def solve_logistic_with_asymptote_from_uplift(
    target_serial_uplift: float,
    reference_progress: float,
    doubling_progress_units: float,
    target_serial_uplift_at_ac: float,
    present_day_human_labor: float,
    present_day_inference_compute: float,
    params: "Parameters",
    fd_step: float = UPLIFT_SOLVE_FD_STEP,
    failure_info: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float, float, Dict[str, Any]]]:
    """
    OPTIONAL three-condition variant: also solve for the logistic asymptote L.

    The two-condition solve takes L as a given and fits (k, x0). But L is the one
    input in this mode that isn't elicitable -- it is a shape parameter about a
    regime nobody observes, and it single-handedly decides how much uplift has
    accumulated by the time A(p) reaches 1 (a ~17x swing across a plausible
    range). This variant replaces that guess with a third stated observable:

        M(p*)                  = target_serial_uplift
        d ln(M - 1)/dp |_{p*}  = ln2 / doubling_progress_units
        M(p_AC)                = target_serial_uplift_at_ac        <-- new

    with p_AC = x0 - ln(L - 1)/k as before. AC is still defined as the A(p) = 1
    crossing; the new condition does NOT redefine the milestone, it just pins the
    curvature by saying how capable the system is when it gets there. (Contrast
    the removed 'fake-uplift' mode, where an uplift threshold *was* the
    definition of AC.)

    Structure: rather than a fresh 3-D Newton, this wraps the existing, proven
    two-condition solve in a bracketed 1-D outer solve on ln(L - 1). That is
    justified by monotonicity -- along the (k, x0) manifold that satisfies the
    first two conditions, M(p_AC) is strictly decreasing in L (verified across
    L in [1.0001, 100]). Bracketed and monotone beats unbracketed and
    higher-dimensional.

    Note M(p_AC) flattens out for large L, so there is a hard floor on the
    achievable uplift at AC; targets below it are reported as out of range
    rather than silently clamped.

    Returns (L, k, x0, diagnostics) or None.

    On failure the return stays None (callers rely on that), but if a mutable
    ``failure_info`` dict is supplied it is populated in place with why the solve
    failed and the numbers needed to explain it -- notably 'reachable' , the
    (min, max) uplift-at-AC span the ladder actually reached. Callers that turn
    the None into an exception can then put those numbers IN the exception rather
    than leaving them behind in a log record, which matters for automated callers
    (e.g. Monte Carlo rejection/clamping) that catch without capturing logs.
    """
    def _fail(reason: str, **extra: Any) -> None:
        """Record why we failed, then return None for the caller to propagate."""
        if failure_info is not None:
            failure_info.clear()
            failure_info.update({
                "reason": reason,
                "target_serial_uplift_at_ac": target_serial_uplift_at_ac,
                "target_serial_uplift": target_serial_uplift,
                "doubling_progress_units": doubling_progress_units,
                **extra,
            })
        return None

    target_serial_uplift_at_ac = float(target_serial_uplift_at_ac)
    if not (np.isfinite(target_serial_uplift_at_ac) and target_serial_uplift_at_ac > 1.0):
        logger.warning(
            f"solve_logistic_with_asymptote_from_uplift: uplift at AC must exceed 1, "
            f"got {target_serial_uplift_at_ac}"
        )
        return _fail("target_not_above_one")

    bounds = getattr(cfg, "PARAMETER_BOUNDS", {}).get("automation_logistic_asymptote", (1.0001, 100.0))
    L_min, L_max = float(bounds[0]), float(bounds[1])
    if not (L_min > 1.0 and L_max > L_min):
        L_min, L_max = 1.0001, 100.0

    cache: Dict[float, Optional[Tuple[float, float, Dict[str, Any], float]]] = {}
    inner_calls = 0

    def inner(log_excess_L: float):
        """Solve (k, x0) at this L; return (k, x0, diag, M_at_AC)."""
        nonlocal inner_calls
        if log_excess_L in cache:
            return cache[log_excess_L]
        L = 1.0 + float(np.exp(log_excess_L))
        result = None
        inner_calls += 1
        solved = solve_logistic_from_uplift(
            target_serial_uplift, reference_progress, doubling_progress_units, L,
            present_day_human_labor, present_day_inference_compute, params, fd_step=fd_step,
        )
        if solved is not None:
            k, x0, diag = solved
            p_ac = diag["progress_at_automation_crossing"]
            try:
                M_ac = float(serial_uplift_for_logistic(
                    [p_ac], k, x0, L, present_day_human_labor,
                    present_day_inference_compute, params,
                )[0])
            except Exception as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                    raise
                M_ac = float("nan")
            if np.isfinite(M_ac) and M_ac > 1.0:
                result = (k, x0, diag, M_ac)
        cache[log_excess_L] = result
        return result

    # Ladder over L to bracket the target, skipping any L where the inner solve fails.
    ladder = np.log(np.geomspace(L_min - 1.0, L_max - 1.0, 18))
    feasible = []
    for le in ladder:
        got = inner(float(le))
        if got is not None:
            feasible.append((float(le), got[3]))
    if len(feasible) < 2:
        logger.warning(
            f"uplift-at-AC solve: only {len(feasible)} feasible asymptotes on the ladder; "
            f"cannot bracket target {target_serial_uplift_at_ac}"
        )
        return _fail(
            "too_few_feasible_asymptotes",
            feasible_count=len(feasible),
            ladder_points=int(len(ladder)),
            reachable=(
                (min(v for _, v in feasible), max(v for _, v in feasible))
                if feasible else None
            ),
        )

    reachable = (min(v for _, v in feasible), max(v for _, v in feasible))
    bracket = None
    for (a, va), (b, vb) in zip(feasible[:-1], feasible[1:]):
        if (va - target_serial_uplift_at_ac) * (vb - target_serial_uplift_at_ac) <= 0:
            bracket = (a, b)
            break
    if bracket is None:
        logger.warning(
            f"uplift-at-AC solve: target {target_serial_uplift_at_ac:.6g}x is outside the "
            f"reachable range [{reachable[0]:.6g}, {reachable[1]:.6g}] for M*="
            f"{target_serial_uplift} and D={doubling_progress_units:.6g} progress units. "
            f"M at AC is monotone decreasing in the asymptote and flattens for large L, so "
            f"low targets in particular may be unreachable at any asymptote."
        )
        return _fail(
            "target_outside_reachable_range",
            reachable=reachable,
            feasible_count=len(feasible),
        )

    def g(le: float) -> float:
        got = inner(le)
        if got is None:
            raise ValueError(f"inner solve failed at L={1.0 + np.exp(le):.6g}")
        return got[3] - target_serial_uplift_at_ac

    try:
        le_root = brentq(g, bracket[0], bracket[1], xtol=1e-12, rtol=1e-14, maxiter=100)
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
            raise
        logger.warning(f"uplift-at-AC solve: outer brentq failed: {e}")
        return _fail("outer_brentq_failed", reachable=reachable, error=str(e))

    got = inner(float(le_root))
    if got is None:
        return _fail(
            "inner_solve_failed_at_root",
            reachable=reachable,
            asymptote_L=1.0 + float(np.exp(le_root)),
        )
    k, x0, diag, M_ac = got
    L = 1.0 + float(np.exp(le_root))

    diagnostics = dict(diag)
    diagnostics.update({
        "asymptote_was_solved": True,
        "asymptote_L": L,
        "target_serial_uplift_at_ac": target_serial_uplift_at_ac,
        "achieved_serial_uplift_at_ac": M_ac,
        "asymptote_reachable_uplift_at_ac": reachable,
        "asymptote_inner_solves": inner_calls,
    })
    logger.info(
        f"uplift-at-AC solve: L={L:.6f} (k={k:.6f}, x0={x0:.6f}) gives {M_ac:.6f}x at AC "
        f"(target {target_serial_uplift_at_ac:.6f}x); {inner_calls} inner two-condition solves"
    )
    return L, k, x0, diagnostics


def _log_profiling_results(
    prof_start: float,
    setup_time: float,
    implied_ratio_time: float,
    implied_ratio_calls: int,
    anchor_build_time: float,
    automationmodel_init_time: float,
    coding_labor_optimal_ces_time: float,
    solver_time: float,
    solver_method: str,
) -> None:
    """Log profiling results for solve_lower_anchor_via_automation_model."""
    total_time = time.perf_counter() - prof_start
    avg_call_time = implied_ratio_time / implied_ratio_calls if implied_ratio_calls > 0 else 0.0
    avg_anchor_build = anchor_build_time / implied_ratio_calls if implied_ratio_calls > 0 else 0.0
    avg_am_init = automationmodel_init_time / implied_ratio_calls if implied_ratio_calls > 0 else 0.0
    avg_ces = coding_labor_optimal_ces_time / implied_ratio_calls if implied_ratio_calls > 0 else 0.0

    logger.info(
        f"PROFILING solve_lower_anchor_via_automation_model:\n"
        f"  Total time: {total_time*1000:.2f}ms\n"
        f"  Setup time: {setup_time*1000:.2f}ms ({100*setup_time/total_time:.1f}%)\n"
        f"  Solver method: {solver_method}\n"
        f"  Solver time (incl. objective calls): {solver_time*1000:.2f}ms ({100*solver_time/total_time:.1f}%)\n"
        f"  implied_ratio_for_anchor:\n"
        f"    Total calls: {implied_ratio_calls}\n"
        f"    Total time: {implied_ratio_time*1000:.2f}ms ({100*implied_ratio_time/total_time:.1f}%)\n"
        f"    Avg per call: {avg_call_time*1000:.3f}ms\n"
        f"    Breakdown per call:\n"
        f"      anchor dict build: {avg_anchor_build*1000:.3f}ms ({100*anchor_build_time/implied_ratio_time:.1f}% of implied_ratio)\n"
        f"      AutomationModel.__init__: {avg_am_init*1000:.3f}ms ({100*automationmodel_init_time/implied_ratio_time:.1f}% of implied_ratio)\n"
        f"      coding_labor_optimal_ces: {avg_ces*1000:.3f}ms ({100*coding_labor_optimal_ces_time/implied_ratio_time:.1f}% of implied_ratio)"
    )
