"""
Tests for TasteDistribution class.

These tests verify that the taste distribution calculations are consistent,
which is important for research taste computations in the model.

The TasteDistribution class applies a bounded transformation to a normal
distribution and numerically optimizes mu and sigma to satisfy two constraints:

1. RATIO CONSTRAINT: The ratio of taste at the top percentile (e.g., 99.9th)
   to median taste must equal median_to_top_gap (default: 3.7x). This encodes
   the empirical observation about how much better "top" researchers are
   compared to median researchers.

2. MEAN CONSTRAINT: The expected value (mean) of the transformed distribution
   must equal baseline_mean (default: 1.0). This ensures the company-wide
   average taste matches the specified baseline.

Additional design properties:
- BOUNDED ABOVE: Taste values are strictly bounded by taste_limit
- SMOOTHNESS: taste_limit_smoothing controls how values approach the upper bound
- INVERTIBILITY: The transformation has an inverse for quantile/SD conversions
"""

import pytest
import numpy as np

from conftest import (
    assert_scalar_close,
    assert_ratio_constraint,
    assert_mean_constraint,
    assert_monotonic,
    STRICT_RTOL,
    NORMAL_RTOL,
    OPTIMIZATION_RTOL,
    INVERSE_FUNCTION_RTOL,
    LOOSE_RTOL,
    VERY_LOOSE_RTOL,
    EDGE_CASE_RTOL,
    EXTREME_PARAMS_RTOL,
)
from progress_model import config as cfg


class TestTasteDistributionOptimizationConstraints:
    """
    Test that the distribution satisfies the two optimization constraints.

    The TasteDistribution numerically optimizes mu and sigma to satisfy:
    1. Ratio constraint: taste(top_percentile) / taste(median) = median_to_top_gap
    2. Mean constraint: E[taste] = baseline_mean
    """

    def test_ratio_constraint_satisfied(self, default_params):
        """
        Verify the ratio constraint: taste at top_percentile / median = median_to_top_gap.

        This constraint encodes the empirical observation that "top" researchers
        (e.g., 99.9th percentile) have taste ~3.7x higher than median researchers.
        """
        taste_dist = default_params.taste_distribution
        # The optimization should achieve this constraint with high precision
        assert_ratio_constraint(taste_dist, taste_dist.median_to_top_gap, rtol=OPTIMIZATION_RTOL)

    def test_mean_constraint_satisfied(self, default_params):
        """
        Verify the mean constraint: E[taste] = baseline_mean.

        This constraint ensures the company-wide average taste matches
        the specified baseline (default: 1.0).
        """
        taste_dist = default_params.taste_distribution
        # The optimization should achieve this constraint with high precision
        assert_mean_constraint(taste_dist, rtol=OPTIMIZATION_RTOL)

    def test_taste_bounded_above(self, default_params):
        """
        Verify taste values are strictly bounded above by taste_limit.

        The transformation function ensures T < A (taste_limit) for all finite inputs.
        """
        taste_dist = default_params.taste_distribution

        # Even at very high quantiles/SDs, taste should remain below limit
        extreme_quantiles = [0.99, 0.999, 0.9999, 0.99999]
        for q in extreme_quantiles:
            taste = taste_dist.get_taste_at_quantile(q)
            assert taste < taste_dist.taste_limit, \
                f"Taste at q={q} ({taste}) should be < taste_limit ({taste_dist.taste_limit})"

        extreme_sds = [3.0, 5.0, 10.0, 20.0]
        for sd in extreme_sds:
            taste = taste_dist.get_taste_at_sd(sd)
            assert taste < taste_dist.taste_limit, \
                f"Taste at sd={sd} ({taste}) should be < taste_limit ({taste_dist.taste_limit})"

    def test_default_parameters_match_config(self, default_params):
        """Verify that the default distribution is built from config, not from
        literals buried in TasteDistribution.

        median_to_top_gap is read from DEFAULT_PARAMETERS rather than from
        cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER because those two are no longer the same
        number: the Q2 alignment put the elicited belief (3.9664793948382653, Daniel's
        sqrt(0.8 * 11) + 1) in DEFAULT_PARAMETERS and left the module constant at 3.7.
        DEFAULT_PARAMETERS is what Parameters() actually reads, so it is the config
        this test is about. The constant survives as TasteDistribution's own signature
        default and is what api/index.py reports as modelConstants.medianToTopGap --
        see tests/GOLDEN_STATUS.md, which flags the divergence for review.
        """
        taste_dist = default_params.taste_distribution

        assert taste_dist.top_percentile == cfg.TOP_PERCENTILE, \
            f"top_percentile mismatch: {taste_dist.top_percentile} != {cfg.TOP_PERCENTILE}"
        expected_gap = cfg.DEFAULT_PARAMETERS['median_to_top_taste_multiplier']
        assert taste_dist.median_to_top_gap == expected_gap, \
            f"median_to_top_gap mismatch: {taste_dist.median_to_top_gap} != {expected_gap}"
        assert taste_dist.baseline_mean == cfg.AGGREGATE_RESEARCH_TASTE_BASELINE, \
            f"baseline_mean mismatch: {taste_dist.baseline_mean} != {cfg.AGGREGATE_RESEARCH_TASTE_BASELINE}"


class TestTasteDistributionProperties:
    """Test TasteDistribution class properties and methods."""

    def test_median_positive(self, default_params):
        """The median of the taste distribution should be positive and finite."""
        median = default_params.taste_distribution.get_median()
        assert median > 0, f"Expected positive median, got {median}"
        assert np.isfinite(median), f"Expected finite median, got {median}"

    def test_taste_at_median_quantile(self, default_params):
        """Taste at quantile 0.5 should equal median."""
        taste_dist = default_params.taste_distribution
        taste_at_50 = taste_dist.get_taste_at_quantile(0.5)
        median = taste_dist.get_median()
        assert abs(taste_at_50 - median) < STRICT_RTOL, \
            f"Taste at q=0.5 ({taste_at_50}) should equal median ({median})"

    def test_quantile_taste_inverse(self, default_params):
        """get_taste_at_quantile and get_quantile_of_taste should be inverses."""
        taste_dist = default_params.taste_distribution

        test_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        for q in test_quantiles:
            taste = taste_dist.get_taste_at_quantile(q)
            recovered_q = taste_dist.get_quantile_of_taste(taste)
            assert abs(recovered_q - q) < OPTIMIZATION_RTOL, \
                f"Inverse failed for q={q}: taste={taste}, recovered={recovered_q}"

    def test_sd_taste_inverse(self, default_params):
        """get_taste_at_sd and get_sd_of_taste should be inverses."""
        taste_dist = default_params.taste_distribution

        test_sds = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        for sd in test_sds:
            taste = taste_dist.get_taste_at_sd(sd)
            recovered_sd = taste_dist.get_sd_of_taste(taste)
            assert abs(recovered_sd - sd) < OPTIMIZATION_RTOL, \
                f"Inverse failed for sd={sd}: taste={taste}, recovered={recovered_sd}"

    def test_taste_monotonic_with_quantile(self, default_params):
        """Taste should increase monotonically with quantile."""
        taste_dist = default_params.taste_distribution

        quantiles = np.linspace(0.01, 0.99, 20)
        tastes = [taste_dist.get_taste_at_quantile(q) for q in quantiles]
        assert_monotonic(tastes, labels=quantiles, msg="Taste not monotonic with quantile")

    def test_taste_monotonic_with_sd(self, default_params):
        """Taste should increase monotonically with SD."""
        taste_dist = default_params.taste_distribution

        sds = np.linspace(-3.0, 5.0, 20)
        tastes = [taste_dist.get_taste_at_sd(sd) for sd in sds]
        assert_monotonic(tastes, labels=sds, msg="Taste not monotonic with SD")

    def test_mean_with_floor_increases(self, default_params):
        """Mean with floor should increase as floor increases."""
        taste_dist = default_params.taste_distribution

        floors = [0.1, 0.5, 1.0, 2.0, 5.0]
        means = [taste_dist.get_mean_with_floor(f) for f in floors]
        assert_monotonic(means, labels=floors, strict=False, msg="Mean not monotonic with floor")

    def test_mean_with_high_floor_approaches_floor(self, default_params):
        """When floor is very high, mean should approach the floor."""
        taste_dist = default_params.taste_distribution

        high_floor = 100.0
        mean_with_floor = taste_dist.get_mean_with_floor(high_floor)

        # With very high floor, almost all mass is at the floor, so mean ≈ floor
        assert abs(mean_with_floor - high_floor) / high_floor < VERY_LOOSE_RTOL, \
            f"Mean with high floor ({mean_with_floor}) should be close to floor ({high_floor})"

    def test_mean_positive(self, default_params):
        """Mean should always be positive."""
        mean = default_params.taste_distribution.get_mean()
        assert mean > 0, f"Mean should be positive, got {mean}"


class TestTasteDistributionGolden:
    """Test TasteDistribution against golden data."""

    def test_taste_at_quantile_matches(self, default_params, golden_taste_distribution):
        """Verify taste_at_quantile matches golden data."""
        taste_dist = default_params.taste_distribution
        for q_str, expected in golden_taste_distribution['taste_at_quantile'].items():
            q = float(q_str)
            actual = taste_dist.get_taste_at_quantile(q)
            assert_scalar_close(
                actual, expected, rtol=STRICT_RTOL,
                msg=f"taste_at_quantile mismatch for q={q}"
            )

    def test_taste_at_sd_matches(self, default_params, golden_taste_distribution):
        """Verify taste_at_sd matches golden data."""
        taste_dist = default_params.taste_distribution
        for sd_str, expected in golden_taste_distribution['taste_at_sd'].items():
            sd = float(sd_str)
            actual = taste_dist.get_taste_at_sd(sd)
            assert_scalar_close(
                actual, expected, rtol=STRICT_RTOL,
                msg=f"taste_at_sd mismatch for sd={sd}"
            )

    def test_mean_with_floor_matches(self, default_params, golden_taste_distribution):
        """Verify mean_with_floor matches golden data."""
        taste_dist = default_params.taste_distribution
        for floor_str, expected in golden_taste_distribution['mean_with_floor'].items():
            floor = float(floor_str)
            actual = taste_dist.get_mean_with_floor(floor)
            assert_scalar_close(
                actual, expected, rtol=NORMAL_RTOL,
                msg=f"mean_with_floor mismatch for floor={floor}"
            )

    def test_distribution_properties_match(self, default_params, golden_taste_distribution):
        """Verify distribution properties match golden data."""
        taste_dist = default_params.taste_distribution
        props = golden_taste_distribution['properties']

        assert_scalar_close(taste_dist.mu, props['mu'], rtol=STRICT_RTOL, msg="mu mismatch")
        assert_scalar_close(taste_dist.sigma, props['sigma'], rtol=STRICT_RTOL, msg="sigma mismatch")
        assert_scalar_close(taste_dist.get_median(), props['median'], rtol=STRICT_RTOL, msg="median mismatch")
        assert_scalar_close(taste_dist.get_mean(), props['mean'], rtol=NORMAL_RTOL, msg="mean mismatch")


class TestTasteDistributionEdgeCases:
    """Test edge cases and numerical stability."""

    def test_extreme_quantiles(self, default_params):
        """Test behavior at extreme quantiles."""
        taste_dist = default_params.taste_distribution

        # Very low quantile
        taste_low = taste_dist.get_taste_at_quantile(0.001)
        assert np.isfinite(taste_low), f"Non-finite taste at q=0.001: {taste_low}"
        assert taste_low > 0, f"Taste at q=0.001 should be positive: {taste_low}"

        # Very high quantile
        taste_high = taste_dist.get_taste_at_quantile(0.999)
        assert np.isfinite(taste_high), f"Non-finite taste at q=0.999: {taste_high}"
        assert taste_high > taste_low, "High quantile taste should be greater than low"

    def test_extreme_sd_values(self, default_params):
        """Test behavior at extreme SD values."""
        taste_dist = default_params.taste_distribution

        # Very negative SD
        taste_neg = taste_dist.get_taste_at_sd(-5.0)
        assert np.isfinite(taste_neg), f"Non-finite taste at sd=-5: {taste_neg}"
        assert taste_neg > 0, f"Taste at sd=-5 should be positive: {taste_neg}"

        # Very positive SD
        taste_pos = taste_dist.get_taste_at_sd(10.0)
        assert np.isfinite(taste_pos), f"Non-finite taste at sd=10: {taste_pos}"
        assert taste_pos > taste_neg, "High SD taste should be greater than low"

    def test_zero_floor(self, default_params):
        """Test mean_with_floor with floor=0."""
        taste_dist = default_params.taste_distribution
        mean_no_floor = taste_dist.get_mean()
        mean_zero_floor = taste_dist.get_mean_with_floor(0.0)

        # With floor=0, mean should equal baseline mean
        assert abs(mean_zero_floor - mean_no_floor) < OPTIMIZATION_RTOL, \
            f"Mean with floor=0 ({mean_zero_floor}) should equal baseline mean ({mean_no_floor})"

    def test_small_floor(self, default_params):
        """Test mean_with_floor with very small floor."""
        taste_dist = default_params.taste_distribution
        mean_small_floor = taste_dist.get_mean_with_floor(1e-10)
        mean_no_floor = taste_dist.get_mean()

        # With very small floor, should be close to baseline mean
        assert abs(mean_small_floor - mean_no_floor) / mean_no_floor < VERY_LOOSE_RTOL, \
            f"Mean with small floor ({mean_small_floor}) should be close to baseline ({mean_no_floor})"


# =============================================================================
# Tests with Non-Default Parameter Values
# =============================================================================
# These tests verify the TasteDistribution works correctly across the FULL
# parameter ranges allowed by progress_model.config.PARAMETER_BOUNDS.
#
# Parameter bounds from progress_model/config.py:
# - median_to_top_taste_multiplier: (1.01, 100.0)
# - top_percentile: (0.5, 0.99999)
# - taste_limit: (0, 100)
# - taste_limit_smoothing: (0.001, 0.999)

class TestTasteDistributionNonDefaultParameters:
    """
    Test TasteDistribution with parameter values across the full allowed range.

    Tests should cover values near bounds and throughout the range as defined
    in progress_model.config.PARAMETER_BOUNDS. If tests fail within allowed bounds,
    that indicates a code bug or bounds that need adjustment.
    """

    # Values spanning the full allowed range from progress_model.config.PARAMETER_BOUNDS
    # median_to_top_taste_multiplier: (1.01, 100.0)
    MEDIAN_TO_TOP_VALUES = [1.05, 1.5, 2.0, 3.7, 5.0, 10.0, 20.0, 50.0, 90.0]
    # taste_limit: (0, 100)
    TASTE_LIMIT_VALUES = [
        0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0, 80.0, 99.0,
    ]
    # taste_limit_smoothing: (0.001, 0.999)
    TASTE_SMOOTHING_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    TASTE_SMOOTHING_VALUES_FOR_INVERSE = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

    @pytest.mark.parametrize("median_to_top", MEDIAN_TO_TOP_VALUES)
    def test_ratio_constraint_with_varying_median_to_top(self, make_taste_params, median_to_top):
        """Verify ratio constraint holds for different median_to_top values."""
        params = make_taste_params(median_to_top=median_to_top)
        assert_ratio_constraint(
            params.taste_distribution, median_to_top, rtol=LOOSE_RTOL,
            msg=f"median_to_top={median_to_top}"
        )

    @pytest.mark.parametrize("median_to_top", MEDIAN_TO_TOP_VALUES)
    def test_mean_constraint_with_varying_median_to_top(self, make_taste_params, median_to_top):
        """Verify mean constraint holds for different median_to_top values."""
        params = make_taste_params(median_to_top=median_to_top)
        assert_mean_constraint(
            params.taste_distribution, rtol=LOOSE_RTOL,
            msg=f"median_to_top={median_to_top}"
        )

    @pytest.mark.parametrize("taste_limit", TASTE_LIMIT_VALUES)
    def test_bounded_above_with_varying_taste_limit(self, make_taste_params, taste_limit):
        """Verify taste values stay bounded for different taste_limit values."""
        params = make_taste_params(taste_limit=taste_limit)
        taste_dist = params.taste_distribution

        extreme_quantiles = [0.99, 0.999, 0.9999]
        for q in extreme_quantiles:
            taste = taste_dist.get_taste_at_quantile(q)
            assert taste < taste_dist.taste_limit, \
                f"Taste at q={q} ({taste:.4f}) should be < taste_limit ({taste_dist.taste_limit:.4f}) for taste_limit={taste_limit}"

        extreme_sds = [3.0, 5.0, 10.0]
        for sd in extreme_sds:
            taste = taste_dist.get_taste_at_sd(sd)
            assert taste < taste_dist.taste_limit, \
                f"Taste at sd={sd} ({taste:.4f}) should be < taste_limit ({taste_dist.taste_limit:.4f}) for taste_limit={taste_limit}"

    @pytest.mark.parametrize("taste_smoothing", TASTE_SMOOTHING_VALUES)
    def test_monotonicity_with_varying_smoothing(self, make_taste_params, taste_smoothing):
        """Verify taste increases monotonically with quantile for different smoothing values."""
        params = make_taste_params(taste_smoothing=taste_smoothing)
        taste_dist = params.taste_distribution

        quantiles = np.linspace(0.01, 0.99, 20)
        tastes = [taste_dist.get_taste_at_quantile(q) for q in quantiles]
        assert_monotonic(tastes, labels=quantiles, msg=f"taste_smoothing={taste_smoothing}")

    @pytest.mark.parametrize("taste_smoothing", TASTE_SMOOTHING_VALUES_FOR_INVERSE)
    def test_inverse_functions_with_varying_smoothing(self, make_taste_params, taste_smoothing):
        """Verify inverse functions work correctly for different smoothing values."""
        params = make_taste_params(taste_smoothing=taste_smoothing)
        taste_dist = params.taste_distribution

        # Test quantile inverse
        test_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in test_quantiles:
            taste = taste_dist.get_taste_at_quantile(q)
            recovered_q = taste_dist.get_quantile_of_taste(taste)
            assert abs(recovered_q - q) < INVERSE_FUNCTION_RTOL, \
                f"Quantile inverse failed at taste_smoothing={taste_smoothing}: q={q}, recovered={recovered_q}"

        # Test SD inverse
        test_sds = [-2.0, 0.0, 2.0]
        for sd in test_sds:
            taste = taste_dist.get_taste_at_sd(sd)
            recovered_sd = taste_dist.get_sd_of_taste(taste)
            assert abs(recovered_sd - sd) < INVERSE_FUNCTION_RTOL, \
                f"SD inverse failed at taste_smoothing={taste_smoothing}: sd={sd}, recovered={recovered_sd}"

    @pytest.mark.parametrize("median_to_top,taste_limit,taste_smoothing", [
        (2.0, 4.0, 0.3),   # Low values
        (3.7, 8.0, 0.5),   # Default-ish values
        (5.0, 16.0, 0.7),  # Medium-high values
        (8.0, 32.0, 0.8),  # Higher values
        (10.0, 50.0, 0.9), # Near upper bounds
    ])
    def test_combined_parameter_variations(self, make_taste_params, median_to_top, taste_limit, taste_smoothing):
        """Test distribution properties with combinations of non-default parameters."""
        params = make_taste_params(
            median_to_top=median_to_top,
            taste_limit=taste_limit,
            taste_smoothing=taste_smoothing
        )
        taste_dist = params.taste_distribution

        # Verify basic properties
        median = taste_dist.get_median()
        assert median > 0, f"Median should be positive: {median}"
        assert np.isfinite(median), f"Median should be finite: {median}"

        mean = taste_dist.get_mean()
        assert mean > 0, f"Mean should be positive: {mean}"
        assert np.isfinite(mean), f"Mean should be finite: {mean}"

        # Verify ratio constraint
        assert_ratio_constraint(
            taste_dist, median_to_top, rtol=VERY_LOOSE_RTOL,
            msg=f"median_to_top={median_to_top}, taste_limit={taste_limit}, smoothing={taste_smoothing}"
        )


class TestTasteDistributionEdgeCasesExtended:
    """
    Extended edge case tests covering parameter bounds from progress_model/config.py.

    These tests use values at or near the bounds defined in PARAMETER_BOUNDS:
    - median_to_top_taste_multiplier: (1.01, 100.0)
    - taste_limit: (0, 100)
    - taste_limit_smoothing: (0.001, 0.999)
    """

    def test_near_minimum_median_to_top(self, make_taste_params):
        """Test with median_to_top near minimum bound (1.01)."""
        params = make_taste_params(median_to_top=1.02)
        assert_ratio_constraint(
            params.taste_distribution, 1.02, rtol=VERY_LOOSE_RTOL,
            msg="near minimum bound"
        )

    def test_near_maximum_median_to_top(self, make_taste_params):
        """Test with median_to_top near maximum bound (100.0)."""
        params = make_taste_params(median_to_top=90.0)
        assert_ratio_constraint(
            params.taste_distribution, 90.0, rtol=EDGE_CASE_RTOL,
            msg="near maximum bound"
        )

    def test_high_median_to_top_with_high_taste_limit(self, make_taste_params):
        """Test with high median_to_top (20.0) and high taste_limit (50.0).

        Both values are within allowed bounds. If this fails, either the code
        needs to handle the combination better, or the bounds need adjustment.
        """
        params = make_taste_params(median_to_top=20.0, taste_limit=50.0)
        assert_ratio_constraint(
            params.taste_distribution, 20.0, rtol=EDGE_CASE_RTOL,
            msg="high value combination"
        )

    def test_near_minimum_smoothing(self, make_taste_params):
        """Test with taste_limit_smoothing near minimum (0.001)."""
        params = make_taste_params(taste_smoothing=0.005)
        taste_dist = params.taste_distribution

        quantiles = np.linspace(0.1, 0.9, 10)
        tastes = [taste_dist.get_taste_at_quantile(q) for q in quantiles]
        assert_monotonic(tastes, labels=quantiles, strict=False, msg="low smoothing")

    def test_near_maximum_smoothing(self, make_taste_params):
        """Test with taste_limit_smoothing near maximum (0.999)."""
        params = make_taste_params(taste_smoothing=0.995)
        taste_dist = params.taste_distribution

        quantiles = np.linspace(0.1, 0.9, 10)
        tastes = [taste_dist.get_taste_at_quantile(q) for q in quantiles]
        assert_monotonic(tastes, labels=quantiles, strict=False, msg="high smoothing")

    def test_cobb_douglas_limit_smoothing(self, make_taste_params):
        """Test with taste_limit_smoothing = 0.5 (Cobb-Douglas limit case)."""
        params = make_taste_params(taste_smoothing=0.5)
        taste_dist = params.taste_distribution

        # Verify basic properties still hold in Cobb-Douglas limit
        median = taste_dist.get_median()
        assert median > 0 and np.isfinite(median), f"Invalid median: {median}"

        mean = taste_dist.get_mean()
        assert mean > 0 and np.isfinite(mean), f"Invalid mean: {mean}"

        # Verify bounded above
        high_taste = taste_dist.get_taste_at_quantile(0.999)
        assert high_taste < taste_dist.taste_limit, \
            f"Taste at q=0.999 ({high_taste}) should be < taste_limit ({taste_dist.taste_limit})"

    def test_near_zero_taste_limit(self, make_taste_params):
        """Test with taste_limit near zero (lower bound is 0)."""
        # taste_limit=0.5 with small median_to_top to keep actual limit reasonable
        params = make_taste_params(median_to_top=1.5, taste_limit=0.5)
        taste_dist = params.taste_distribution

        # All taste values should be below the limit
        for q in [0.5, 0.9, 0.99, 0.999]:
            taste = taste_dist.get_taste_at_quantile(q)
            assert taste < taste_dist.taste_limit, \
                f"Taste at q={q} ({taste}) exceeds limit ({taste_dist.taste_limit})"

    def test_near_maximum_taste_limit(self, make_taste_params):
        """Test with taste_limit near maximum (100)."""
        params = make_taste_params(taste_limit=99.0)
        taste_dist = params.taste_distribution

        # Distribution should still work correctly
        median = taste_dist.get_median()
        mean = taste_dist.get_mean()
        assert median > 0 and np.isfinite(median), f"Invalid median: {median}"
        assert mean > 0 and np.isfinite(mean), f"Invalid mean: {mean}"

    def test_mean_with_floor_various_params(self, make_taste_params):
        """Test mean_with_floor across different parameter combinations."""
        param_combos = [
            {'median_to_top': 2.0, 'taste_limit': 5.0},
            {'median_to_top': 5.0, 'taste_limit': 15.0},
            {'median_to_top': 8.0, 'taste_limit': 30.0},
            {'median_to_top': 50.0, 'taste_limit': 2.0},  # High m2t, low limit
        ]

        for kwargs in param_combos:
            params = make_taste_params(**kwargs)
            taste_dist = params.taste_distribution

            # Mean with floor should be monotonic with floor
            floors = [0.1, 0.5, 1.0, 2.0]
            means = [taste_dist.get_mean_with_floor(f) for f in floors]
            assert_monotonic(means, labels=floors, strict=False, msg=f"params={kwargs}")

    @pytest.mark.parametrize("median_to_top,taste_limit", [
        (1.02, 99.0),   # Min m2t, max limit
        (90.0, 0.5),    # Near max m2t, min limit
        (50.0, 50.0),   # Mid-high both
        (100.0, 1.0),   # Max m2t, low limit
    ])
    def test_extreme_parameter_combinations(self, make_taste_params, median_to_top, taste_limit):
        """Test extreme but valid parameter combinations at bounds."""
        params = make_taste_params(median_to_top=median_to_top, taste_limit=taste_limit)
        taste_dist = params.taste_distribution

        # Basic properties should hold
        median = taste_dist.get_median()
        mean = taste_dist.get_mean()

        assert median > 0 and np.isfinite(median), \
            f"Invalid median for m2t={median_to_top}, limit={taste_limit}: {median}"
        assert mean > 0 and np.isfinite(mean), \
            f"Invalid mean for m2t={median_to_top}, limit={taste_limit}: {mean}"

        # Ratio constraint should still hold
        assert_ratio_constraint(
            taste_dist, median_to_top, rtol=EXTREME_PARAMS_RTOL,
            msg=f"m2t={median_to_top}, limit={taste_limit}"
        )


class TestTasteDistributionNumericalStability:
    """
    Test numerical stability across the full allowed parameter range.

    Parameter bounds from progress_model/config.py:
    - median_to_top_taste_multiplier: (1.01, 100.0)
    - taste_limit: (0, 100)
    - taste_limit_smoothing: (0.001, 0.999)
    """

    # Cover full range including near-bounds values
    @pytest.mark.parametrize("median_to_top,taste_smoothing", [
        *[(m, s) for m in [1.05, 2.0, 5.0, 10.0, 20.0, 50.0, 90.0] for s in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]],
    ])
    def test_no_nans_or_infs(self, make_taste_params, median_to_top, taste_smoothing):
        """Verify no NaN or Inf values are produced across parameter combinations."""
        # Use default taste_limit (8.0) - the actual limit is median_to_top**(1+taste_limit)
        params = make_taste_params(median_to_top=median_to_top, taste_smoothing=taste_smoothing)
        taste_dist = params.taste_distribution

        # Test various quantiles
        quantiles = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
        for q in quantiles:
            taste = taste_dist.get_taste_at_quantile(q)
            assert np.isfinite(taste), \
                f"Non-finite taste at q={q} for median_to_top={median_to_top}, smoothing={taste_smoothing}: {taste}"

        # Test various SDs
        sds = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0]
        for sd in sds:
            taste = taste_dist.get_taste_at_sd(sd)
            assert np.isfinite(taste), \
                f"Non-finite taste at sd={sd} for median_to_top={median_to_top}, smoothing={taste_smoothing}: {taste}"

    def test_extreme_sd_values_stability(self, default_params):
        """Test numerical stability at very extreme SD values."""
        taste_dist = default_params.taste_distribution

        # Very negative SDs should produce small positive values
        for sd in [-10.0, -20.0, -50.0]:
            taste = taste_dist.get_taste_at_sd(sd)
            assert taste > 0, f"Taste should be positive at sd={sd}: {taste}"
            assert np.isfinite(taste), f"Taste should be finite at sd={sd}: {taste}"

        # Very positive SDs should produce values approaching but not exceeding limit
        for sd in [10.0, 20.0, 50.0]:
            taste = taste_dist.get_taste_at_sd(sd)
            assert taste < taste_dist.taste_limit, \
                f"Taste should be < limit at sd={sd}: taste={taste}, limit={taste_dist.taste_limit}"
            assert np.isfinite(taste), f"Taste should be finite at sd={sd}: {taste}"

    def test_consistency_across_equivalent_quantile_sd(self, default_params):
        """Verify taste_at_quantile and taste_at_sd produce consistent results."""
        from scipy.stats import norm

        taste_dist = default_params.taste_distribution

        # The relationship: quantile q corresponds to sd = norm.ppf(q)
        test_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        for q in test_quantiles:
            taste_from_q = taste_dist.get_taste_at_quantile(q)
            sd = norm.ppf(q)
            taste_from_sd = taste_dist.get_taste_at_sd(sd)

            assert abs(taste_from_q - taste_from_sd) / max(taste_from_q, STRICT_RTOL) < OPTIMIZATION_RTOL, \
                f"Inconsistent results for q={q}: taste_at_quantile={taste_from_q}, taste_at_sd={taste_from_sd}"
