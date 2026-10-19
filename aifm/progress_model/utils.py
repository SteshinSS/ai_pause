#!/usr/bin/env python3
"""
Utility functions for progress modeling.

Contains scalar coercion, Gauss-Hermite quadrature utilities, and
other shared helper functions.
"""

import numpy as np
from typing import Any, Callable
from numpy.polynomial.hermite import hermgauss


def should_reraise(e: BaseException) -> bool:
    """
    Check if an exception should be re-raised rather than caught.

    Timeouts, keyboard interrupts, and system exits should propagate
    to allow proper cleanup and termination of batch processing.

    Args:
        e: The exception to check

    Returns:
        True if the exception should be re-raised
    """
    return isinstance(e, (KeyboardInterrupt, SystemExit)) or 'TimeoutError' in type(e).__name__


def _coerce_float_scalar(value: Any, name: str) -> float:
    """
    Convert a value to a finite float scalar.

    Accepts Python scalars, NumPy scalars, or size-1 arrays/lists. Raises
    ValueError for non-numeric inputs, NaNs/Infs, or multi-value inputs.
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar, got {value!r}") from exc

    if arr.size == 0:
        raise ValueError(f"{name} must be a real scalar, got empty input")

    if arr.ndim == 0:
        result = float(arr)
    else:
        if arr.size != 1:
            raise ValueError(f"{name} must be a real scalar, got array with shape {arr.shape}")
        result = float(arr.reshape(-1)[0])

    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")

    return result


# Precompute Gauss-Hermite quadrature nodes and weights for fast normal expectations
# 32 points provides ~14 digits of accuracy for smooth functions
_GH_DEGREE = 32
_gh_nodes, _gh_weights = hermgauss(_GH_DEGREE)
# Transform nodes for standard normal: x = sqrt(2) * u maps Hermite to N(0,1)
_gh_nodes_scaled = np.sqrt(2.0) * _gh_nodes
# Adjust weights: factor of 1/sqrt(pi) for N(0,1) expectation
_gh_weights_scaled = _gh_weights / np.sqrt(np.pi)


def _gauss_hermite_expectation(func: Callable, mu: float, sigma: float) -> float:
    """
    Compute E[func(mu + sigma * X)] where X ~ N(0,1) using Gauss-Hermite quadrature.

    This is much faster than scipy.integrate.quad for smooth functions, computing
    the expectation as a weighted sum of ~32 function evaluations.

    Args:
        func: Function to compute expectation of (should handle array input)
        mu: Mean of the normal distribution (or shift parameter)
        sigma: Standard deviation (or scale parameter)

    Returns:
        Approximate expectation value
    """
    # Evaluate func at quadrature points: mu + sigma * sqrt(2) * nodes
    x_points = mu + sigma * _gh_nodes_scaled
    f_values = func(x_points)
    return float(np.dot(_gh_weights_scaled, f_values))


def _log_interp(x, xp: np.ndarray, fp: np.ndarray):
    """
    Perform log-space interpolation for exponential trends.
    Handles both scalar and array inputs for x.

    Args:
        x: Point(s) to interpolate at (scalar or array)
        xp: Known x-coordinates (must be sorted)
        fp: Known y-coordinates (must be positive for log-space)

    Returns:
        Interpolated value (scalar if x is scalar, array if x is array)
    """
    # Track if input was scalar
    input_was_scalar = np.ndim(x) == 0

    # Ensure x is an array
    x = np.atleast_1d(x)

    # Ensure all values are positive for log-space interpolation
    if np.any(fp <= 0):
        # Fall back to linear interpolation if any values are non-positive
        result = np.interp(x, xp, fp)
        return float(result[0]) if input_was_scalar else result

    # Perform log-space interpolation in one vectorized operation
    log_fp = np.log(fp)
    log_interpolated = np.interp(x, xp, log_fp)
    result = np.exp(log_interpolated)

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result
