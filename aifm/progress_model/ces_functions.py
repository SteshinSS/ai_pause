#!/usr/bin/env python3
"""
CES (Constant Elasticity of Substitution) production functions.

This module contains the core CES mathematical functions used throughout the
progress model for combining inputs like human labor, AI labor, and compute.
"""

import numpy as np
from scipy import optimize
import logging
from . import config as cfg
from .utils import should_reraise

logger = logging.getLogger(__name__)


def _ces_function(X1, X2, w1, rho):
    """
    Computes the CES function with the standard substitution parameter rho.
    Handles both scalar and array inputs.
    Y = (w1*X1^rho + (1-w1)*X2^rho)^(1/rho)

    Args:
        X1: First input (scalar or array)
        X2: Second input (scalar or array)
        w1: Weight of the first input [0,1] (scalar or array)
        rho: Standard substitution parameter in (-inf, 1] (scalar or array)

    Returns:
        Combined output (scalar if all inputs are scalar, array otherwise)
    """
    # Track if input was scalar
    input_was_scalar = (np.ndim(X1) == 0 and np.ndim(X2) == 0 and
                       np.ndim(w1) == 0 and np.ndim(rho) == 0)

    # Normalize to arrays
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    w1 = np.atleast_1d(w1)
    rho = np.atleast_1d(rho)

    # Clip w1 to valid range
    w1 = np.clip(w1, 0.0, 1.0)
    w2 = 1 - w1

    # Broadcast to common shape
    shape = np.broadcast_shapes(X1.shape, X2.shape, w1.shape, rho.shape)
    X1 = np.broadcast_to(X1, shape)
    X2 = np.broadcast_to(X2, shape)
    w1 = np.broadcast_to(w1, shape)
    w2 = np.broadcast_to(w2, shape)
    rho = np.broadcast_to(rho, shape)

    # Initialize result array
    result = np.zeros(shape, dtype=np.float64)

    # Create masks for different cases
    cobb_douglas_mask = np.abs(rho) < cfg.RHO_COBB_DOUGLAS_THRESHOLD
    perfect_sub_mask = (rho == 1.0) & ~cobb_douglas_mask
    leontief_mask = (rho < cfg.RHO_LEONTIEF_THRESHOLD) & ~cobb_douglas_mask
    standard_mask = ~(cobb_douglas_mask | perfect_sub_mask | leontief_mask)

    # Case 1: Cobb-Douglas (rho -> 0)
    if np.any(cobb_douglas_mask):
        mask = cobb_douglas_mask
        valid = mask & (X1 > 0) & (X2 > 0)
        if np.any(valid):
            try:
                log_result = w1[valid] * np.log(X1[valid]) + w2[valid] * np.log(X2[valid])
                result[valid] = np.exp(log_result)
                # Handle overflow/non-finite results
                overflow = valid & ~np.isfinite(result)
                if np.any(overflow):
                    result[overflow] = w1[overflow] * X1[overflow] + w2[overflow] * X2[overflow]
            except (ValueError, RuntimeWarning):
                # Fallback to linear
                result[valid] = w1[valid] * X1[valid] + w2[valid] * X2[valid]
        # Zero result where either input is zero
        result[mask & ((X1 == 0) | (X2 == 0))] = 0.0

    # Case 2: Perfect substitutes (rho = 1)
    if np.any(perfect_sub_mask):
        mask = perfect_sub_mask
        result[mask] = w1[mask] * X1[mask] + w2[mask] * X2[mask]

    # Case 3: Leontief (rho -> -inf)
    if np.any(leontief_mask):
        mask = leontief_mask
        result[mask] = np.minimum(X1[mask], X2[mask])

    # Case 4: Standard CES
    if np.any(standard_mask):
        mask = standard_mask

        # Handle edge cases for weights
        w1_zero = mask & (w1 == 0)
        w2_zero = mask & (w2 == 0)
        both_inputs_zero = mask & (X1 == 0) & (X2 == 0)
        X1_zero_only = mask & (X1 == 0) & (X2 != 0) & (w1 != 0) & (w2 != 0)
        X2_zero_only = mask & (X2 == 0) & (X1 != 0) & (w1 != 0) & (w2 != 0)

        if np.any(w1_zero):
            result[w1_zero] = X2[w1_zero]
        if np.any(w2_zero):
            result[w2_zero] = X1[w2_zero]
        if np.any(both_inputs_zero):
            result[both_inputs_zero] = 0.0
        if np.any(X1_zero_only):
            result[X1_zero_only] = np.power(w2[X1_zero_only], 1 / rho[X1_zero_only]) * X2[X1_zero_only]
        if np.any(X2_zero_only):
            result[X2_zero_only] = np.power(w1[X2_zero_only], 1 / rho[X2_zero_only]) * X1[X2_zero_only]

        # Standard CES for remaining cases
        standard_ces_mask = (mask & (X1 > 0) & (X2 > 0) & (w1 > 0) & (w2 > 0) &
                            ~w1_zero & ~w2_zero & ~both_inputs_zero & ~X1_zero_only & ~X2_zero_only)

        if np.any(standard_ces_mask):
            try:
                term1 = w1[standard_ces_mask] * np.power(X1[standard_ces_mask], rho[standard_ces_mask])
                term2 = w2[standard_ces_mask] * np.power(X2[standard_ces_mask], rho[standard_ces_mask])
                total = term1 + term2

                # Check for non-positive totals
                non_positive = total <= 0
                if np.any(non_positive):
                    # Use min as fallback
                    idx = np.where(standard_ces_mask)[0][non_positive]
                    result[idx] = np.minimum(X1[idx], X2[idx])
                    # Update mask to exclude these
                    valid_total = ~non_positive
                else:
                    valid_total = np.ones(np.sum(standard_ces_mask), dtype=bool)

                if np.any(valid_total):
                    ces_result = np.power(total[valid_total], 1 / rho[standard_ces_mask][valid_total])

                    # Check for non-finite results
                    finite = np.isfinite(ces_result)
                    idx_valid = np.where(standard_ces_mask)[0][valid_total][finite]
                    result[idx_valid] = ces_result[finite]

                    # Fallback to linear for non-finite
                    idx_invalid = np.where(standard_ces_mask)[0][valid_total][~finite]
                    if len(idx_invalid) > 0:
                        result[idx_invalid] = w1[idx_invalid] * X1[idx_invalid] + w2[idx_invalid] * X2[idx_invalid]

            except (ValueError, RuntimeWarning, FloatingPointError):
                # Fallback to linear for all standard CES cases on error
                result[standard_ces_mask] = (w1[standard_ces_mask] * X1[standard_ces_mask] +
                                            w2[standard_ces_mask] * X2[standard_ces_mask])

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result


def compute_coding_labor_deprecated(
    automation_fraction,
    inference_H100e,
    L_HUMAN,
    rho,
    parallel_penalty,
    cognitive_normalization=1.0,
    human_only=False
):
    """
    CES combination of AI and human labor using an alternative formulation.
    Handles both scalar and array inputs.
    Y = ( (A^(1-rho) * inference_H100e^rho) + ((1-A)^(1-rho) * L_HUMAN^rho) )^(1/rho)

    Args:
        automation_fraction: Fraction of work automated (A) [0,1] (scalar or array)
        inference_H100e: AI labor supply (scalar or array)
        L_HUMAN: Human labor supply (scalar or array)
        rho: Standard substitution parameter in (-inf, 1] (scalar or array)
        parallel_penalty: Power transformation parameter (scalar or array)
        cognitive_normalization: Normalization constant (scalar or array, default 1.0)
        human_only: Boolean mask for human-only mode (scalar or array, default False)

    Returns:
        Cognitive output (scalar if all inputs are scalar, array otherwise)
    """
    # Track if input was scalar
    input_was_scalar = (np.ndim(automation_fraction) == 0 and np.ndim(inference_H100e) == 0 and
                       np.ndim(L_HUMAN) == 0 and np.ndim(rho) == 0 and
                       np.ndim(parallel_penalty) == 0)

    # Ensure all inputs are arrays
    automation_fraction = np.atleast_1d(automation_fraction)
    inference_H100e = np.atleast_1d(inference_H100e)
    L_HUMAN = np.atleast_1d(L_HUMAN)
    rho = np.atleast_1d(rho)
    parallel_penalty = np.atleast_1d(parallel_penalty)

    if cognitive_normalization is None or (np.ndim(cognitive_normalization) == 0 and cognitive_normalization == 1.0):
        cognitive_normalization = np.ones_like(automation_fraction)
    else:
        cognitive_normalization = np.atleast_1d(cognitive_normalization)

    if human_only is None or (np.ndim(human_only) == 0 and not human_only):
        human_only = np.zeros_like(automation_fraction, dtype=bool)
    else:
        human_only = np.atleast_1d(human_only).astype(bool)

    # Broadcast to common shape
    shape = np.broadcast_shapes(
        automation_fraction.shape, inference_H100e.shape, L_HUMAN.shape,
        rho.shape, parallel_penalty.shape, cognitive_normalization.shape, human_only.shape
    )
    automation_fraction = np.broadcast_to(automation_fraction, shape)
    inference_H100e = np.broadcast_to(inference_H100e, shape)
    L_HUMAN = np.broadcast_to(L_HUMAN, shape)
    rho = np.broadcast_to(rho, shape)
    parallel_penalty = np.broadcast_to(parallel_penalty, shape)
    cognitive_normalization = np.broadcast_to(cognitive_normalization, shape)
    human_only = np.broadcast_to(human_only, shape)

    # Initialize result
    result = np.zeros(shape, dtype=np.float64)

    # Handle human_only mode
    if np.any(human_only):
        result[human_only] = (np.power(L_HUMAN[human_only], parallel_penalty[human_only]) *
                             cognitive_normalization[human_only])

    # Process non-human-only cases
    non_human_only = ~human_only
    if not np.any(non_human_only):
        return float(result[0]) if input_was_scalar else result

    # Input validation for non-human-only cases
    valid = (non_human_only &
             np.isfinite(automation_fraction) &
             np.isfinite(inference_H100e) &
             np.isfinite(L_HUMAN) &
             np.isfinite(rho) &
             np.isfinite(parallel_penalty) &
             np.isfinite(cognitive_normalization) &
             (inference_H100e >= 0) &
             (L_HUMAN >= 0))

    if not np.any(valid):
        return float(result[0]) if input_was_scalar else result

    # Clamp automation fraction to valid range
    a = np.clip(automation_fraction[valid],
                cfg.AUTOMATION_FRACTION_CLIP_MIN,
                1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)

    # Extract other values for valid cases
    inf_comp = inference_H100e[valid]
    l_human = L_HUMAN[valid]
    rho_val = rho[valid]
    pp = parallel_penalty[valid]
    cn = cognitive_normalization[valid]

    # Create masks for different rho cases
    cobb_douglas_mask = np.abs(rho_val) < cfg.RHO_COBB_DOUGLAS_THRESHOLD
    perfect_sub_mask = (rho_val == 1.0) & ~cobb_douglas_mask
    leontief_mask = (rho_val < cfg.RHO_LEONTIEF_THRESHOLD) & ~cobb_douglas_mask
    standard_mask = ~(cobb_douglas_mask | perfect_sub_mask | leontief_mask)

    # Initialize ces_result
    ces_result = np.zeros(np.sum(valid), dtype=np.float64)

    # Case 1: Cobb-Douglas
    if np.any(cobb_douglas_mask):
        mask = cobb_douglas_mask
        valid_cd = mask & (inf_comp > 0) & (l_human > 0)
        if np.any(valid_cd):
            try:
                term1 = a[valid_cd] * (np.log(inf_comp[valid_cd]) - np.log(a[valid_cd]))
                term2 = (1 - a[valid_cd]) * (np.log(l_human[valid_cd]) - np.log(1 - a[valid_cd]))
                log_result = term1 + term2
                ces_result[valid_cd] = np.exp(log_result)
                # Handle overflow
                overflow = valid_cd & ~np.isfinite(ces_result)
                if np.any(overflow):
                    ces_result[overflow] = a[overflow] * inf_comp[overflow] + (1 - a[overflow]) * l_human[overflow]
            except (ValueError, RuntimeWarning):
                ces_result[valid_cd] = a[valid_cd] * inf_comp[valid_cd] + (1 - a[valid_cd]) * l_human[valid_cd]
        # Zero result where either input is zero
        ces_result[mask & ((inf_comp == 0) | (l_human == 0))] = 0.0

    # Case 2: Perfect substitutes
    if np.any(perfect_sub_mask):
        mask = perfect_sub_mask
        ces_result[mask] = inf_comp[mask] + l_human[mask]

    # Case 3: Leontief
    if np.any(leontief_mask):
        mask = leontief_mask
        ces_result[mask] = np.minimum(inf_comp[mask] / a[mask], l_human[mask] / (1 - a[mask]))

    # Case 4: Standard CES
    if np.any(standard_mask):
        mask = standard_mask
        try:
            term1 = np.where(inf_comp[mask] > 0,
                           np.power(a[mask], 1 - rho_val[mask]) * np.power(inf_comp[mask], rho_val[mask]),
                           0)
            term2 = np.where(l_human[mask] > 0,
                           np.power(1 - a[mask], 1 - rho_val[mask]) * np.power(l_human[mask], rho_val[mask]),
                           0)

            total = term1 + term2

            # Handle non-positive totals
            non_positive = total <= 0
            if np.any(non_positive):
                idx = np.where(mask)[0][non_positive]
                ces_result[idx] = np.minimum(inf_comp[idx] / a[idx], l_human[idx] / (1 - a[idx]))
                valid_total = ~non_positive
            else:
                valid_total = np.ones(np.sum(mask), dtype=bool)

            if np.any(valid_total):
                ces_result_temp = np.power(total[valid_total], 1 / rho_val[mask][valid_total])

                # Check for non-finite
                finite = np.isfinite(ces_result_temp)
                idx_valid = np.where(mask)[0][valid_total][finite]
                ces_result[idx_valid] = ces_result_temp[finite]

                # Fallback for non-finite
                idx_invalid = np.where(mask)[0][valid_total][~finite]
                if len(idx_invalid) > 0:
                    ces_result[idx_invalid] = (a[idx_invalid] * inf_comp[idx_invalid] +
                                               (1 - a[idx_invalid]) * l_human[idx_invalid])

        except (ValueError, RuntimeWarning, FloatingPointError):
            ces_result[mask] = a[mask] * inf_comp[mask] + (1 - a[mask]) * l_human[mask]

    # Apply parallel_penalty transformation
    try:
        result_with_lambda = np.power(ces_result, pp)
        # Handle non-finite results
        non_finite = ~np.isfinite(result_with_lambda)
        if np.any(non_finite):
            result_with_lambda[non_finite] = ces_result[non_finite]
    except (ValueError, RuntimeWarning, FloatingPointError):
        result_with_lambda = ces_result

    # Apply normalization and store in result
    result[valid] = result_with_lambda * cn

    # Return scalar if input was scalar, array otherwise
    return float(result[0]) if input_was_scalar else result


def compute_rho_from_asymptotes(inf_labor_asymptote: float, inf_compute_asymptote: float) -> float:
    """
    Compute the substitution parameter rho from the asymptotes of the experiment capacity CES.
    Solve for rho in the equation:
    inf_labor_asymptote**rho + inf_compute_asymptote**rho = 1
    """
    # Validate inputs
    if not np.isfinite(inf_labor_asymptote) or not np.isfinite(inf_compute_asymptote):
        logger.warning("Non-finite asymptotes provided to compute_rho_from_asymptotes; falling back to 0.0")
        return 0.0

    # Ensure strictly positive bases to avoid undefined powers
    a = float(inf_labor_asymptote)
    b = float(inf_compute_asymptote)

    if a <= 0 or b <= 0:
        logger.warning(f"Non-positive asymptote(s) a={a}, b={b}; falling back to 0.0")
        return 0.0

    # Small epsilon to avoid numerical issues when values are extremely close to 0
    a = max(a, cfg.NORMALIZATION_MIN)
    b = max(b, cfg.NORMALIZATION_MIN)

    def equation(rho: float) -> float:
        return np.power(a, rho) + np.power(b, rho) - 1.0

    # Bracket within the valid CES range for rho
    lower = cfg.RHO_CLIP_MIN
    upper = 1.0

    f_lower = equation(lower)
    f_upper = equation(upper)

    # If already satisfies at a bound, return that bound (rare but possible)
    if abs(f_lower) < 1e-12:
        return float(lower)
    if abs(f_upper) < 1e-12:
        return float(upper)

    # Check for a valid bracket
    if f_lower * f_upper > 0:
        # This indicates inputs inconsistent with rho in (-inf, 1]; fallback to Cobb-Douglas
        logger.warning(
            f"Asymptotes a={a}, b={b} do not bracket a root in [${lower}, ${upper}]; falling back to rho=0.0"
        )
        return 0.0

    try:
        rho = optimize.brentq(equation, lower, upper, maxiter=100, xtol=1e-12)
    except Exception as e:
        if should_reraise(e):
            raise
        logger.warning(f"Root finding failed in compute_rho_from_asymptotes: {e}; falling back to 0.0")
        return 0.0

    # Clip to valid range and return
    return float(np.clip(rho, cfg.RHO_CLIP_MIN, 1.0))


def compute_experiment_compute_exponent_from_anchor(
    inf_compute_asymptote: float,
    inf_labor_asymptote: float,
    compute_anchor: tuple[float, float],
    rho: float
) -> float:
    """
    Compute the experiment compute exponent from the asymptotes and anchor.
    """
    k = (inf_compute_asymptote / inf_labor_asymptote) ** rho
    N = compute_anchor[0]
    M = compute_anchor[1]

    res = (1 / (rho * np.log(N))) * np.log((1 + k) * (M ** rho) - k)
    return res


def compute_alpha_experiment_capacity_from_asymptotes(
    inf_labor_asymptote: float,
    inf_compute_asymptote: float,
    experiment_compute_exponent: float,
    current_exp_compute: float,
    current_serial_coding_labor: float,
    rho: float
) -> float:
    """
    Compute the alpha parameter for the experiment capacity CES from the asymptotes and anchors.
    """
    zeta = experiment_compute_exponent
    C = current_exp_compute
    L = current_serial_coding_labor
    res = 1 / (1 + ((C ** zeta / L) * (inf_compute_asymptote / inf_labor_asymptote)) ** rho)
    return res


def compute_exp_capacity_params_from_anchors(
    inf_labor_asymptote: float,
    inf_compute_asymptote: float,
    compute_anchor: tuple[float, float],
    labor_anchor: tuple[float, float],
    current_exp_compute: float,
    current_coding_labor: float,
    parallel_penalty: float = 0.0
) -> tuple[float, float, float]:
    """
    Compute the parameters for the experiment capacity CES from the asymptotes and anchors.

    Returns:
        Tuple of (rho, alpha_experiment_capacity, experiment_compute_exponent)
    """
    rho = compute_rho_from_asymptotes(inf_labor_asymptote, inf_compute_asymptote)
    experiment_compute_exponent = compute_experiment_compute_exponent_from_anchor(
        inf_compute_asymptote, inf_labor_asymptote, compute_anchor, rho
    )
    serial_coding_labor = current_coding_labor ** parallel_penalty
    alpha_experiment_capacity = compute_alpha_experiment_capacity_from_asymptotes(
        inf_labor_asymptote, inf_compute_asymptote, experiment_compute_exponent,
        current_exp_compute, serial_coding_labor, rho
    )
    return rho, alpha_experiment_capacity, experiment_compute_exponent
