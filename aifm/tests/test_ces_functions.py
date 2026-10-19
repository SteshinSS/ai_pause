"""
Tests for CES production functions and coding labor calculations.

These tests verify that the core mathematical functions produce consistent outputs
to catch any behavioral changes from performance optimizations.
"""

import pytest
import numpy as np
import json

from progress_model import _ces_function, compute_coding_labor_deprecated
from conftest import assert_scalar_close
from utils import GOLDEN_DATA_DIR


class TestCESFunction:
    """Tests for the _ces_function CES production function."""

    def test_cobb_douglas_limit(self):
        """When rho→0, CES becomes Cobb-Douglas (geometric mean for w=0.5)."""
        result = _ces_function(2.0, 3.0, 0.5, 1e-10)
        expected = np.sqrt(2.0 * 3.0)  # geometric mean for w=0.5
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_perfect_substitutes(self):
        """When rho=1, CES becomes linear weighted sum."""
        result = _ces_function(2.0, 3.0, 0.5, 1.0)
        expected = 0.5 * 2.0 + 0.5 * 3.0
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_leontief_limit(self):
        """When rho→-∞, CES becomes min function (Leontief)."""
        result = _ces_function(2.0, 3.0, 0.5, -100)
        assert abs(result - 2.0) < 1e-4, f"Expected ~2.0, got {result}"

    def test_weight_zero_returns_x2(self):
        """When w1=0, result should equal X2."""
        result = _ces_function(5.0, 3.0, 0.0, -2.0)
        assert abs(result - 3.0) < 1e-10, f"Expected 3.0, got {result}"

    def test_weight_one_returns_x1(self):
        """When w1=1, result should equal X1."""
        result = _ces_function(5.0, 3.0, 1.0, -2.0)
        assert abs(result - 5.0) < 1e-10, f"Expected 5.0, got {result}"

    def test_zero_inputs_return_zero(self):
        """When both inputs are zero, result should be zero."""
        result = _ces_function(0.0, 0.0, 0.5, -2.0)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_one_zero_input_with_negative_rho(self):
        """With one zero input and rho<0 (complements), result scaled appropriately."""
        result = _ces_function(0.0, 3.0, 0.5, -2.0)
        # When X1=0 and rho<0, we should get (w2)^(1/rho) * X2
        expected = (0.5 ** (1/-2.0)) * 3.0
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_symmetry(self):
        """CES should be symmetric when weights are swapped."""
        result1 = _ces_function(2.0, 3.0, 0.3, -2.0)
        result2 = _ces_function(3.0, 2.0, 0.7, -2.0)
        assert abs(result1 - result2) < 1e-10, f"Symmetry violated: {result1} != {result2}"

    def test_golden_data_matching(self, regen_golden):
        """Verify CES function outputs match golden data."""
        golden_path = GOLDEN_DATA_DIR / 'ces_function_golden.json'

        if regen_golden:
            pytest.skip("Regenerating golden data, skipping comparison")

        with open(golden_path) as f:
            golden_data = json.load(f)

        for case in golden_data['test_cases']:
            result = _ces_function(case['x1'], case['x2'], case['w1'], case['rho'])
            expected = case['result']

            # Allow for some numerical tolerance
            if np.isfinite(expected) and np.isfinite(result):
                assert_scalar_close(
                    result, expected, rtol=1e-10,
                    msg=f"CES mismatch for x1={case['x1']}, x2={case['x2']}, w1={case['w1']}, rho={case['rho']}"
                )
            else:
                # Both should be the same type of non-finite
                assert np.isfinite(result) == np.isfinite(expected), \
                    f"Finite mismatch: expected {expected}, got {result}"


class TestComputeCodingLabor:
    """Tests for the compute_coding_labor_deprecated function."""

    def test_human_only_mode(self):
        """In human_only mode, result should only depend on L_HUMAN and parallel_penalty."""
        result = compute_coding_labor_deprecated(
            automation_fraction=0.5,
            inference_H100e=1000.0,
            L_HUMAN=100.0,
            rho=-2.0,
            parallel_penalty=0.5,
            cognitive_normalization=1.0,
            human_only=True
        )
        expected = (100.0 ** 0.5) * 1.0  # L_HUMAN^parallel_penalty * normalization
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_zero_labor_returns_zero(self):
        """With zero human labor, result should still work (AI only)."""
        result = compute_coding_labor_deprecated(
            automation_fraction=0.9,
            inference_H100e=1000.0,
            L_HUMAN=0.0,
            rho=-2.0,
            parallel_penalty=0.5,
            cognitive_normalization=1.0,
            human_only=False
        )
        assert np.isfinite(result), f"Expected finite result, got {result}"

    def test_zero_inference_compute(self):
        """With zero inference compute, result should still work (human only)."""
        result = compute_coding_labor_deprecated(
            automation_fraction=0.5,
            inference_H100e=0.0,
            L_HUMAN=100.0,
            rho=-2.0,
            parallel_penalty=0.5,
            cognitive_normalization=1.0,
            human_only=False
        )
        assert np.isfinite(result), f"Expected finite result, got {result}"
        assert result > 0, f"Expected positive result, got {result}"

    def test_normalization_scaling(self):
        """Normalization should scale the result linearly."""
        result1 = compute_coding_labor_deprecated(0.5, 1000.0, 100.0, -2.0, 0.5, 1.0, False)
        result2 = compute_coding_labor_deprecated(0.5, 1000.0, 100.0, -2.0, 0.5, 2.0, False)
        assert abs(result2 / result1 - 2.0) < 1e-10, "Normalization should scale linearly"

    def test_parallel_penalty_effect(self):
        """Parallel penalty should affect the output."""
        result_low = compute_coding_labor_deprecated(0.5, 1000.0, 100.0, -2.0, 0.3, 1.0, False)
        result_high = compute_coding_labor_deprecated(0.5, 1000.0, 100.0, -2.0, 0.7, 1.0, False)
        # Verify that changing parallel_penalty changes the result
        assert result_low != result_high, \
            f"Different parallel penalties should give different outputs: {result_low} vs {result_high}"
        # Both should be finite and positive
        assert np.isfinite(result_low) and result_low > 0
        assert np.isfinite(result_high) and result_high > 0

    def test_golden_data_matching(self, regen_golden):
        """Verify coding labor outputs match golden data."""
        golden_path = GOLDEN_DATA_DIR / 'coding_labor_golden.json'

        if regen_golden:
            pytest.skip("Regenerating golden data, skipping comparison")

        with open(golden_path) as f:
            golden_data = json.load(f)

        for case in golden_data['test_cases']:
            result = compute_coding_labor_deprecated(
                automation_fraction=case['automation_fraction'],
                inference_H100e=case['inference_compute'],
                L_HUMAN=case['L_HUMAN'],
                rho=case['rho'],
                parallel_penalty=case['parallel_penalty'],
                cognitive_normalization=case['cognitive_normalization'],
                human_only=case['human_only']
            )
            expected = case['result']

            if np.isfinite(expected) and np.isfinite(result):
                assert_scalar_close(
                    result, expected, rtol=1e-10,
                    msg=f"Coding labor mismatch for {case}"
                )


class TestCESEdgeCases:
    """Test edge cases and numerical stability."""

    def test_extreme_rho_values(self):
        """Test CES behavior with extreme rho values."""
        # Very negative rho (near Leontief)
        result_neg = _ces_function(2.0, 3.0, 0.5, -50.0)
        assert np.isfinite(result_neg), f"Non-finite result for rho=-50: {result_neg}"
        assert abs(result_neg - 2.0) < 0.1, f"Expected ~min(2,3)=2, got {result_neg}"

        # Small positive rho (near linear)
        result_pos = _ces_function(2.0, 3.0, 0.5, 0.99)
        assert np.isfinite(result_pos), f"Non-finite result for rho=0.99: {result_pos}"

    def test_extreme_input_values(self):
        """Test with very large and very small inputs."""
        # Very large inputs
        result_large = _ces_function(1e10, 1e10, 0.5, -2.0)
        assert np.isfinite(result_large), f"Non-finite result for large inputs: {result_large}"

        # Very small inputs
        result_small = _ces_function(1e-10, 1e-10, 0.5, -2.0)
        assert np.isfinite(result_small), f"Non-finite result for small inputs: {result_small}"

        # Mixed extreme inputs
        result_mixed = _ces_function(1e-10, 1e10, 0.5, -2.0)
        assert np.isfinite(result_mixed), f"Non-finite result for mixed inputs: {result_mixed}"
