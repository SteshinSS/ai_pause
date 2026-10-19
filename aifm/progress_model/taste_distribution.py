#!/usr/bin/env python3
"""
Taste Distribution Module

Manages human research taste distribution and provides methods for working
with taste values in terms of quantiles and standard deviations.

The distribution is modeled by applying a bounded transformation to a normal distribution.
"""

import numpy as np
from scipy import optimize
from collections import OrderedDict
import logging

from . import config as cfg
from .utils import _coerce_float_scalar, _gauss_hermite_expectation

logger = logging.getLogger(__name__)

# Global cache for TasteDistribution instances to avoid expensive reinitialization
# Using OrderedDict to support LRU eviction (most recently used at end)
_taste_distribution_cache: OrderedDict = OrderedDict()
_TASTE_CACHE_MAX_SIZE = 1000  # Max ~5 MB (each instance ~5 KB)


class TasteDistribution:
    """
    Manages human research taste distribution and provides methods for working
    with taste values in terms of quantiles and standard deviations.

    The distribution is modeled by applying a bounded transformation to a normal distribution.
    The transformation function is: T = (A^ρ + (1 - A^ρ) * exp(v*x)^ρ)^(1/ρ)
    where v = 1/(1 - A^ρ), x ~ Normal(μ, σ²), A is the taste_limit parameter,
    L is taste_limit_smoothing, and ρ = 2*ln(1/L - 1) / ln(A).

    This transformation ensures that taste values are bounded above by A, with L controlling the smoothness.

    Parameters are derived from empirical anchors:
    - top_percentile: quantile threshold for "top" researchers (e.g., 0.999 = 99.9th percentile)
    - median_to_top_gap: ratio of threshold taste to median taste
    - baseline_mean: company-wide mean taste
    - taste_limit: upper bound A for the transformation
    - taste_limit_smoothing: smoothing parameter L (0 < L < 1)

    Example usage:
        # Create distribution with default parameters
        taste_dist = TasteDistribution()

        # Get taste at 90th percentile
        top_taste = taste_dist.get_taste_at_quantile(0.9)

        # Get taste at 2 standard deviations above mean in normal space
        ai_taste = taste_dist.get_taste_at_sd(2.0)

        # Compute aggregate taste with AI floor growing at 0.5 SD per progress unit
        progress = 10.0
        ai_research_taste = taste_dist.get_taste_at_sd(0.5 * progress)
        aggregate_taste = taste_dist.get_mean_with_floor(ai_research_taste)
    """

    def __init__(self,
                 top_percentile: float = cfg.TOP_PERCENTILE,
                 median_to_top_gap: float = cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER,
                 baseline_mean: float = cfg.AGGREGATE_RESEARCH_TASTE_BASELINE,
                 taste_limit_m2b: float = cfg.DEFAULT_PARAMETERS['taste_limit'],
                 taste_limit_smoothing: float = cfg.DEFAULT_PARAMETERS['taste_limit_smoothing']):
        """
        Initialize the taste distribution with empirical anchors.

        Args:
            top_percentile: Quantile threshold for "top" researchers (e.g., 0.999 = 99.9th percentile)
            median_to_top_gap: Ratio of threshold taste to median taste
            baseline_mean: Company-wide mean taste
            taste_limit_m2b: Used to compute upper bound A for the transformation function
            taste_limit_smoothing: Smoothing parameter u (u > 0)
        """
        from scipy.stats import norm
        from scipy.optimize import minimize
        import math
        logger.info("Initializing Taste Distribution")

        # Sanitize scalar inputs before using them numerically
        top_percentile = _coerce_float_scalar(top_percentile, "top_percentile")
        median_to_top_gap = _coerce_float_scalar(median_to_top_gap, "median_to_top_gap")
        baseline_mean = _coerce_float_scalar(baseline_mean, "baseline_mean")
        taste_limit_m2b = _coerce_float_scalar(taste_limit_m2b, "taste_limit_m2b")
        taste_limit_smoothing = _coerce_float_scalar(taste_limit_smoothing, "taste_limit_smoothing")

        # Store parameters
        self.top_percentile = top_percentile
        self.median_to_top_gap = median_to_top_gap
        self.baseline_mean = baseline_mean
        self.taste_limit = median_to_top_gap**(1+taste_limit_m2b)
        self.taste_limit_smoothing = taste_limit_smoothing
        logger.info(f"taste limit (absolute): {self.taste_limit}")
        # Validate parameters
        if not (0 < top_percentile < 1):
            raise ValueError(f"top_percentile must be between 0 and 1, got {top_percentile}")
        if median_to_top_gap <= 1:
            raise ValueError(f"median_to_top_gap must be > 1, got {median_to_top_gap}")
        # When median_to_top_gap > 1, we need top_percentile > 0.5 for the constraint system to be satisfiable
        if median_to_top_gap > 1 and top_percentile <= 0.5:
            raise ValueError(f"top_percentile must be > 0.5 when median_to_top_gap > 1 (got top_percentile={top_percentile}, median_to_top_gap={median_to_top_gap})")
        if baseline_mean <= 0:
            raise ValueError(f"baseline_mean must be > 0, got {baseline_mean}")
        if taste_limit_m2b < 0:
            raise ValueError(f"taste_limit_m2b must be >= 0, got {taste_limit_m2b}")
        if not (0 < taste_limit_smoothing < 1):
            raise ValueError(f"taste_limit_smoothing must be between 0 and 1, got {taste_limit_smoothing}")
        # baseline_mean must be less than taste_limit since the transform outputs values in (0, taste_limit)
        if baseline_mean >= self.taste_limit:
            raise ValueError(f"baseline_mean must be < median_to_top_gap**(1+taste_limit_m2b) (got baseline_mean={baseline_mean}, taste_limit_m2b={taste_limit_m2b})")

        # Define the transformation function T = (A^ρ + (1 - A^ρ) * exp(v*x)^ρ)^(1/ρ)
        # where v = 1/(1 - A^ρ) and ρ = 2*ln(1/L - 1) / ln(A)
        A = self.taste_limit
        L = self.taste_limit_smoothing

        cobb_douglas_limit = abs(L - 0.5) < 1e-6

        if cobb_douglas_limit:
            self._rho = 0.0
            self._x_at_zero = float('-inf')
            log_A = np.log(A)

            def transform(x):
                x_arr = np.asarray(x, dtype=np.float64)
                scaled = np.clip(x_arr / log_A, -700.0, 700.0)
                exp_term = np.exp(-scaled)
                log_values = log_A * (1.0 - exp_term)
                values = np.exp(log_values)
                values = np.minimum(A, values)
                if np.ndim(x_arr) == 0:
                    return float(values)
                return values

            def log_transform(x):
                x_arr = np.asarray(x, dtype=np.float64)
                scaled = np.clip(x_arr / log_A, -700.0, 700.0)
                exp_term = np.exp(-scaled)
                log_values = log_A * (1.0 - exp_term)
                if np.ndim(x_arr) == 0:
                    return float(log_values)
                return log_values

            def inverse_transform(T):
                taste_arr = np.asarray(T, dtype=np.float64)
                result = np.empty_like(taste_arr, dtype=np.float64)
                result.fill(np.nan)

                non_finite = ~np.isfinite(taste_arr)
                result[non_finite] = np.nan

                le_zero = taste_arr <= 0
                ge_limit = taste_arr >= A
                mid = ~(non_finite | le_zero | ge_limit)

                if np.any(le_zero):
                    result[le_zero] = -np.inf
                if np.any(ge_limit):
                    result[ge_limit] = np.inf

                if np.any(mid):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = np.log(taste_arr[mid]) / log_A
                    inner = 1.0 - ratio
                    inner = np.clip(inner, 1e-300, np.exp(700.0))
                    result[mid] = -log_A * np.log(inner)

                if np.ndim(taste_arr) == 0:
                    return float(result)
                return result

            self._transform = transform
            self._log_transform = log_transform
            self._inverse_transform = inverse_transform

        else:
            one_over_L_minus_1 = 1.0 / L - 1.0
            A_pow_rho = one_over_L_minus_1 * one_over_L_minus_1
            c1 = 1.0 - A_pow_rho
            if c1 == 0:
                raise ValueError("taste_limit_smoothing produced undefined scale; choose L != 0.5.")

            log_A = np.log(A)
            rho = 2.0 * np.log(one_over_L_minus_1) / log_A
            self._rho = rho

            # v = 1/c1, so exp(v*x)^rho = exp(rho*x/c1)
            # The inner term becomes zero when exp(rho*x/c1) = -A_pow_rho/c1 = A_pow_rho/(A_pow_rho - 1)
            # This only happens when c1 < 0 (i.e., A_pow_rho > 1)
            if c1 < 0:
                self._x_at_zero = c1 * np.log(A_pow_rho / (-c1)) / rho
            else:
                self._x_at_zero = float('-inf')

            def transform(x):
                x_arr = np.asarray(x, dtype=np.float64)
                # exp(v*x)^rho = exp(rho*v*x) = exp(rho*x/c1)
                exp_arg = rho * x_arr / c1
                exp_arg = np.clip(exp_arg, -700.0, 700.0)
                inner = A_pow_rho + c1 * np.exp(exp_arg)
                if c1 < 0:
                    inner = np.clip(inner, 0.0, None)
                else:
                    inner = np.clip(inner, 1e-300, None)
                values = np.power(inner, 1.0 / rho)
                if np.ndim(x_arr) == 0:
                    return float(values)
                return values

            def log_transform(x):
                x_arr = np.asarray(x, dtype=np.float64)
                exp_arg = rho * x_arr / c1
                exp_arg = np.clip(exp_arg, -700.0, 700.0)
                inner = A_pow_rho + c1 * np.exp(exp_arg)
                if c1 < 0:
                    inner = np.clip(inner, 0.0, None)
                else:
                    inner = np.clip(inner, 1e-300, None)
                with np.errstate(divide='ignore'):
                    log_inner = np.log(inner)
                log_values = log_inner / rho
                if np.ndim(x_arr) == 0:
                    return float(log_values)
                return log_values

            def inverse_transform(T):
                # Given T = (A^rho + c1 * exp(rho*x/c1))^(1/rho), solve for x:
                # T^rho = A^rho + c1 * exp(rho*x/c1)
                # (T^rho - A^rho) / c1 = exp(rho*x/c1)
                # ln((T^rho - A^rho) / c1) = rho*x/c1
                # x = c1 * ln((T^rho - A^rho) / c1) / rho
                taste_arr = np.asarray(T, dtype=np.float64)
                result = np.empty_like(taste_arr, dtype=np.float64)
                result.fill(np.nan)

                non_finite = ~np.isfinite(taste_arr)
                result[non_finite] = np.nan

                le_zero = taste_arr <= 0
                ge_limit = taste_arr >= A
                mid = ~(non_finite | le_zero | ge_limit)

                if np.any(le_zero):
                    result[le_zero] = self._x_at_zero
                if np.any(ge_limit):
                    result[ge_limit] = np.inf

                if np.any(mid):
                    T_pow = np.power(taste_arr[mid], rho)
                    ratio = (T_pow - A_pow_rho) / c1
                    ratio = np.clip(ratio, 1e-300, None)
                    result[mid] = c1 * np.log(ratio) / rho

                if np.ndim(taste_arr) == 0:
                    return float(result)
                return result

            self._transform = transform
            self._log_transform = log_transform
            self._inverse_transform = inverse_transform

        # Solve for μ and σ numerically so that the transformed distribution matches
        # both the ratio constraint and the mean constraint. This preserves the
        # original behavior prior to the analytic shortcut.

        z_p = norm.ppf(top_percentile)

        target_log_ratio = math.log(median_to_top_gap)

        def constraint_residuals(params):
            mu_trial, log_sigma_trial = params
            sigma_trial = np.exp(log_sigma_trial)

            residual_ratio = 1e6
            residual_mean = 1e6

            try:
                log_T_top = self._log_transform(mu_trial + sigma_trial * z_p)
                log_T_median = self._log_transform(mu_trial)
                log_ratio = log_T_top - log_T_median
                residual_ratio = log_ratio - target_log_ratio
            except Exception as e:
                # Don't swallow timeout/interrupt exceptions - let them propagate
                if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                    raise
                residual_ratio = 1e6

            try:
                # Use Gauss-Hermite quadrature for fast expectation computation
                mean_val = _gauss_hermite_expectation(self._transform, mu_trial, sigma_trial)
                residual_mean = mean_val - baseline_mean
            except Exception as e:
                # Don't swallow timeout/interrupt exceptions - let them propagate
                if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                    raise
                residual_mean = 1e6

            return np.array([residual_ratio, residual_mean])

        sigma_init = math.log(median_to_top_gap) / z_p
        mu_init = math.log(baseline_mean) - 0.5 * sigma_init ** 2
        log_sigma_init = math.log(max(sigma_init, 1e-12))

        result = optimize.least_squares(
            constraint_residuals,
            x0=np.array([mu_init, log_sigma_init]),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )

        if not result.success:
            logger.warning(f"TasteDistribution calibration did not fully converge: {result.message}")

        self.mu = float(result.x[0])
        self.sigma = float(np.exp(result.x[1]))

        if self.sigma <= 0:
            raise ValueError(
                f"Calibration produced non-positive sigma={self.sigma:.4e}. "
                "Check that the provided anchors are compatible."
            )

        # Precompute the actual mean for get_mean() using Gauss-Hermite quadrature
        try:
            self._computed_mean = _gauss_hermite_expectation(self._transform, self.mu, self.sigma)
        except Exception as e:
            # Don't swallow timeout/interrupt exceptions - let them propagate
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__:
                raise
            logger.warning("Failed to compute mean, using baseline_mean")
            self._computed_mean = baseline_mean

        # Precompute grid for get_mean_with_floor interpolation
        # This significantly speeds up repeated calls by avoiding numerical integration
        self._precompute_floor_grid()

        logger.debug(f"TasteDistribution initialized: μ={self.mu:.4f}, σ={self.sigma:.4f}, A={self.taste_limit:.4e}")

    def get_taste_at_quantile(self, quantile: float) -> float:
        """
        Get the taste value at a given quantile of the distribution.

        Args:
            quantile: Quantile (0 to 1)

        Returns:
            Taste value at the specified quantile
        """
        from scipy.stats import norm

        if not (0 <= quantile <= 1):
            raise ValueError(f"quantile must be between 0 and 1, got {quantile}")

        if quantile == 0:
            return 0.0
        if quantile == 1:
            return self.taste_limit

        # Apply transformation to the quantile of the underlying normal distribution
        x_quantile = self.mu + self.sigma * norm.ppf(quantile)
        return self._transform(x_quantile)

    def get_quantile_of_taste(self, taste):
        """
        Get the quantile of a given taste value in the distribution.
        Handles both scalar and array inputs.

        Args:
            taste: Taste value(s) (scalar or array)

        Returns:
            Quantile (0 to 1) of the taste value (scalar if input scalar, array otherwise)
        """
        from scipy.stats import norm

        # Track if input was scalar
        input_was_scalar = np.ndim(taste) == 0

        # Ensure taste is an array
        taste = np.atleast_1d(taste)
        result = np.zeros_like(taste, dtype=np.float64)

        # Handle edge cases
        below_zero = taste <= 0
        above_limit = taste >= self.taste_limit
        valid = ~(below_zero | above_limit)

        result[below_zero] = 0.0
        result[above_limit] = 1.0

        if np.any(valid):
            # Invert the transformation to get x, then find the quantile of x
            x = self._inverse_transform(taste[valid])
            result[valid] = norm.cdf((x - self.mu) / self.sigma)

        # Return scalar if input was scalar, array otherwise
        return float(result[0]) if input_was_scalar else result

    def get_taste_at_sd(self, num_sds):
        """
        Get taste value at a given number of standard deviations in the underlying normal distribution.
        Handles both scalar and array inputs.

        This method returns transform(μ + num_sds * σ).

        Args:
            num_sds: Number of standard deviations (can be negative) (scalar or array)

        Returns:
            Taste value at the specified standard deviation (scalar if input scalar, array otherwise)
        """
        # Track if input was scalar
        input_was_scalar = np.ndim(num_sds) == 0

        # Ensure num_sds is an array
        num_sds = np.atleast_1d(num_sds)

        # Compute x values
        x = self.mu + num_sds * self.sigma

        # Apply transformation
        result = self._transform(x)

        # Return scalar if input was scalar, array otherwise
        return float(result[0]) if input_was_scalar else result

    def get_sd_of_taste(self, taste):
        """
        Get how many standard deviations a taste value is in the underlying normal distribution.
        Handles both scalar and array inputs.

        Args:
            taste: Taste value(s) (scalar or array)

        Returns:
            Number of standard deviations (scalar if input scalar, array otherwise)
        """
        # Track if input was scalar
        input_was_scalar = np.ndim(taste) == 0

        # Ensure taste is an array
        taste = np.atleast_1d(taste)
        result = np.zeros_like(taste, dtype=np.float64)

        # Handle edge cases
        below_zero = taste <= 0
        above_limit = taste >= self.taste_limit
        valid = ~(below_zero | above_limit)

        result[below_zero] = float('-inf')
        result[above_limit] = float('inf')

        if np.any(valid):
            # Invert the transformation to get x, then find how many SDs it is
            x = self._inverse_transform(taste[valid])
            result[valid] = (x - self.mu) / self.sigma

        # Return scalar if input was scalar, array otherwise
        return float(result[0]) if input_was_scalar else result

    def _precompute_floor_grid(self):
        """
        Precompute a grid of (floor_taste, mean_with_floor) values for fast interpolation.

        The grid covers the range where the floor meaningfully affects the mean.
        Below the low percentile, the floor has negligible effect (return _computed_mean).
        Above the high percentile, mean ≈ floor (return floor directly).
        """
        import time
        from scipy.stats import norm
        from scipy.integrate import quad

        _grid_start = time.perf_counter()

        # Determine grid bounds based on distribution percentiles
        low_quantile = cfg.TASTE_FLOOR_GRID_LOW_PERCENTILE
        high_quantile = cfg.TASTE_FLOOR_GRID_HIGH_PERCENTILE

        self._floor_grid_low = self.get_taste_at_quantile(low_quantile)
        self._floor_grid_high = self.get_taste_at_quantile(high_quantile)
        _bounds_time = time.perf_counter() - _grid_start

        # Create logarithmically spaced grid points for better resolution at lower values
        # Handle negative values by using a hybrid approach
        n_points = cfg.TASTE_FLOOR_GRID_POINTS

        if self._floor_grid_low <= 0 and self._floor_grid_high > 0:
            # Grid spans negative to positive: use linear spacing
            grid_floors = np.linspace(self._floor_grid_low, self._floor_grid_high, n_points)
        elif self._floor_grid_high <= 0:
            # Entire grid is negative: use linear spacing
            grid_floors = np.linspace(self._floor_grid_low, self._floor_grid_high, n_points)
        else:
            # Entire grid is positive: use log spacing for better resolution
            grid_floors = np.logspace(
                np.log10(max(self._floor_grid_low, 1e-10)),
                np.log10(self._floor_grid_high),
                n_points
            )

        # Compute cumulative tail integral ∫_z^∞ T(μ + σx) φ(x) dx using a single pass
        # Strategy: Evaluate integrand on a fine grid, then compute cumulative sum
        _integration_start = time.perf_counter()

        # Create a fine grid in z-space for numerical integration
        # We need to cover the range where the PDF has significant mass
        z_fine = np.linspace(-10, 10, 2000)  # Fine grid for integration

        # Evaluate the integrand at each point
        def integrand_vectorized(z_arr):
            return self._transform(self.mu + self.sigma * z_arr) * norm.pdf(z_arr)

        integrand_values = integrand_vectorized(z_fine)

        # Compute cumulative integral from right to left (∫_z^∞ = reverse cumsum)
        # Using trapezoidal rule
        dz = z_fine[1] - z_fine[0]
        # Cumulative sum from right: tail_integral[i] = ∫_{z_fine[i]}^∞ integrand(z) dz
        tail_integral_fine = np.cumsum(integrand_values[::-1])[::-1] * dz

        # Now for each floor value in our grid, compute mean_with_floor
        grid_means = np.zeros(n_points)

        for i, floor in enumerate(grid_floors):
            if floor <= 0:
                grid_means[i] = self._computed_mean
                continue

            if floor >= self.taste_limit:
                grid_means[i] = floor
                continue

            try:
                x_floor = self._inverse_transform(floor)

                if not np.isfinite(x_floor):
                    grid_means[i] = self._computed_mean
                    continue

                if self._x_at_zero > float('-inf') and x_floor <= self._x_at_zero:
                    grid_means[i] = self._computed_mean
                    continue

                z_floor = (x_floor - self.mu) / self.sigma

                if not np.isfinite(z_floor) or z_floor <= -10.0:
                    grid_means[i] = self._computed_mean
                    continue

                # Interpolate the tail integral at z_floor
                upper = np.interp(z_floor, z_fine, tail_integral_fine)
                lower = floor * norm.cdf(z_floor)

                clipped_mean = lower + upper

                if not np.isfinite(clipped_mean):
                    grid_means[i] = max(floor, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)
                else:
                    grid_means[i] = max(clipped_mean, floor)

            except (ValueError, OverflowError) as e:
                logger.warning(f"Error computing grid point at floor={floor}: {e}")
                grid_means[i] = max(floor, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)

        # Store grid for interpolation
        self._floor_grid_floors = grid_floors
        self._floor_grid_means = grid_means

        _integration_time = time.perf_counter() - _integration_start
        _grid_time = time.perf_counter() - _grid_start

        logger.info(
            f"Precomputed floor grid: {n_points} points from "
            f"{self._floor_grid_low:.4g} to {self._floor_grid_high:.4g} "
            f"in {_grid_time:.3f}s (bounds: {_bounds_time:.3f}s, integration: {_integration_time:.3f}s)"
        )

    def get_mean_with_floor(self, floor_taste):
        """
        Compute the mean of the distribution with a floor applied using clip-and-keep logic.
        Handles both scalar and array inputs.

        This computes E[max(T, F)] where T is the taste distribution and F is the floor value.
        Uses precomputed grid interpolation for fast evaluation during trajectory computation.

        Args:
            floor_taste: Floor value(s) (any draw below this is lifted to this value) (scalar or array)

        Returns:
            Mean taste after applying the floor (scalar if input scalar, array otherwise)
        """
        # Track if input was scalar
        input_was_scalar = np.ndim(floor_taste) == 0

        # Ensure floor_taste is an array
        floor_taste = np.atleast_1d(floor_taste)
        result = np.zeros_like(floor_taste, dtype=np.float64)

        # Input validation
        valid = np.isfinite(floor_taste)
        invalid_mask = ~valid
        if np.any(invalid_mask):
            logger.warning(f"Non-finite floor_taste: {floor_taste[invalid_mask]}")
        if not np.any(valid):
            fallback = np.full_like(floor_taste, cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK)
            return float(fallback[0]) if input_was_scalar else fallback

        # Handle different regions
        below_grid = valid & (floor_taste < self._floor_grid_low)
        above_grid = valid & (floor_taste > self._floor_grid_high)
        above_limit = valid & (floor_taste >= self.taste_limit)
        in_grid = valid & ~(below_grid | above_grid | above_limit)

        # Fast path: floor below grid range (negligible effect on mean)
        result[below_grid] = self._computed_mean

        # Fast path: floor above grid range or at/above taste_limit (mean ≈ floor)
        result[above_grid | above_limit] = floor_taste[above_grid | above_limit]

        # Interpolate using precomputed grid for values in range
        if np.any(in_grid):
            result[in_grid] = np.interp(
                floor_taste[in_grid],
                self._floor_grid_floors,
                self._floor_grid_means
            )
            # Ensure result is at least the floor (numerical precision safeguard)
            result[in_grid] = np.maximum(result[in_grid], floor_taste[in_grid])

        # Handle invalid inputs (warning already emitted above)
        result[invalid_mask] = cfg.AGGREGATE_RESEARCH_TASTE_FALLBACK

        # Return scalar if input was scalar, array otherwise
        return float(result[0]) if input_was_scalar else result

    def get_median(self) -> float:
        """Get the median of the distribution."""
        return self.get_taste_at_quantile(0.5)

    def get_mean(self) -> float:
        """Get the unconditional mean of the distribution."""
        return self._computed_mean

    def __repr__(self) -> str:
        return (f"TasteDistribution(top_percentile={self.top_percentile:.3f}, "
                f"median_to_top_gap={self.median_to_top_gap:.2f}, "
                f"baseline_mean={self.baseline_mean:.2f}, "
                f"taste_limit={self.taste_limit:.2e}, "
                f"taste_limit_smoothing={self.taste_limit_smoothing:.2f})")


def compute_ai_research_taste(
    cumulative_progress: float,
    taste_distribution: TasteDistribution,
    slope: float,
    anchor_progress: float,
    anchor_sd: float,
    max_sd: float = 100.0
) -> float:
    """
    Compute AI research taste at a given progress level using the SD-per-progress schedule.

    Formula: taste = taste_distribution.get_taste_at_sd(slope * progress + offset)
    where offset ensures the curve passes through (anchor_progress, anchor_sd).

    Args:
        cumulative_progress: Current cumulative progress (scalar or array)
        taste_distribution: TasteDistribution instance
        slope: SD per progress unit
        anchor_progress: Progress value at anchor point
        anchor_sd: SD value at anchor point
        max_sd: Maximum SD to clamp to (prevents numerical issues)

    Returns:
        AI research taste value (scalar if input is scalar, array if input is array)
    """
    # Handle both scalar and array inputs
    cumulative_progress = np.asarray(cumulative_progress)
    is_scalar = cumulative_progress.ndim == 0

    # Replace non-finite values with 0.0
    cumulative_progress = np.where(np.isfinite(cumulative_progress), cumulative_progress, 0.0)

    if anchor_progress is None:
        raise ValueError("anchor_progress is None - cannot compute AI research taste")

    # offset ensures curve passes through (anchor_progress, anchor_sd)
    offset = anchor_sd - slope * anchor_progress
    current_sd = slope * cumulative_progress + offset
    current_sd = np.minimum(current_sd, max_sd)

    result = taste_distribution.get_taste_at_sd(current_sd)

    # Return scalar if input was scalar
    return float(result) if is_scalar else result


def get_or_create_taste_distribution(
    top_percentile: float = cfg.TOP_PERCENTILE,
    median_to_top_gap: float = cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER,
    baseline_mean: float = cfg.AGGREGATE_RESEARCH_TASTE_BASELINE,
    taste_limit: float = cfg.DEFAULT_PARAMETERS['taste_limit'],
    taste_limit_smoothing: float = cfg.DEFAULT_PARAMETERS['taste_limit_smoothing']
) -> TasteDistribution:
    """
    Get or create a cached TasteDistribution instance with LRU eviction.

    TasteDistribution initialization is expensive (~1s), so we cache instances
    based on their parameters to avoid redundant computation. The cache uses
    LRU (Least Recently Used) eviction to prevent unbounded memory growth.

    Memory usage: ~5 KB per cached instance, max 1000 entries = ~5 MB.

    Args:
        top_percentile: Quantile threshold for "top" researchers
        median_to_top_gap: Ratio of threshold taste to median taste
        baseline_mean: Company-wide mean taste
        taste_limit: Upper bound for the transformation function
        taste_limit_smoothing: Smoothing parameter

    Returns:
        Cached or newly created TasteDistribution instance
    """
    top_percentile = _coerce_float_scalar(top_percentile, "top_percentile")
    median_to_top_gap = _coerce_float_scalar(median_to_top_gap, "median_to_top_gap")
    baseline_mean = _coerce_float_scalar(baseline_mean, "baseline_mean")
    taste_limit = _coerce_float_scalar(taste_limit, "taste_limit")
    taste_limit_smoothing = _coerce_float_scalar(taste_limit_smoothing, "taste_limit_smoothing")

    # Create cache key from parameters (rounded to avoid floating point issues)
    cache_key = (
        round(top_percentile, 6),
        round(median_to_top_gap, 6),
        round(baseline_mean, 6),
        round(taste_limit, 6),
        round(taste_limit_smoothing, 6)
    )

    if cache_key in _taste_distribution_cache:
        # Move to end (mark as recently used)
        _taste_distribution_cache.move_to_end(cache_key)
        logger.debug(f"Using cached TasteDistribution for parameters: {cache_key}")
    else:
        # Create new instance
        logger.debug(f"Creating new TasteDistribution for parameters: {cache_key}")
        _taste_distribution_cache[cache_key] = TasteDistribution(
            top_percentile=top_percentile,
            median_to_top_gap=median_to_top_gap,
            baseline_mean=baseline_mean,
            taste_limit_m2b=taste_limit,
            taste_limit_smoothing=taste_limit_smoothing
        )

        # Enforce size limit (LRU eviction)
        if len(_taste_distribution_cache) > _TASTE_CACHE_MAX_SIZE:
            # Remove oldest (first) item
            evicted_key = next(iter(_taste_distribution_cache))
            _taste_distribution_cache.pop(evicted_key)
            logger.debug(f"Evicted LRU TasteDistribution from cache: {evicted_key}")

    return _taste_distribution_cache[cache_key]
