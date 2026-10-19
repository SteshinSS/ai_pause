"""Tests for the cdf_points distribution kind (elicited CDF anchor points).

The spec encodes elicitations of the form "P(> value is infeasible/suboptimal) = p",
i.e. CDF anchors P(X <= value) = p, interpolated linearly in value or log(value).
Mass below the first anchor / above the last collapses onto the endpoint values.
"""

import numpy as np
import pytest

from scripts.pipelines.batch_rollout import _sample_from_dist, _sample_from_dist_with_quantile

# The site-size elicitation these were built for (covert_bottom_up.yaml):
# P(>10 MW infeasible)=10%, P(>100 MW)=35%, P(>1 GW)=80%.
SITE_SPEC = {
    "dist": "cdf_points",
    "scale": "log",
    "points": [[10, 0.10], [100, 0.35], [1000, 0.80]],
    "min": 10,
    "max": 1000,
    "clip_to_bounds": True,
}


class TestQuantileValues:
    def test_anchor_quantiles_return_anchor_values(self):
        assert _sample_from_dist_with_quantile(SITE_SPEC, 0.10) == pytest.approx(10.0)
        assert _sample_from_dist_with_quantile(SITE_SPEC, 0.35) == pytest.approx(100.0)
        assert _sample_from_dist_with_quantile(SITE_SPEC, 0.80) == pytest.approx(1000.0)

    def test_log_interpolation_between_anchors(self):
        # Halfway in probability between (10, 0.10) and (100, 0.35) -> 10^1.5 in log space.
        assert _sample_from_dist_with_quantile(SITE_SPEC, 0.225) == pytest.approx(10 ** 1.5)
        # Median: (0.5 - 0.35) / (0.80 - 0.35) = 1/3 of the way from 100 to 1000 in log space.
        assert _sample_from_dist_with_quantile(SITE_SPEC, 0.5) == pytest.approx(10 ** (2 + 1 / 3))

    def test_tails_collapse_onto_endpoints(self):
        # 10% of mass at exactly 10 MW, 20% at exactly 1000 MW.
        assert _sample_from_dist_with_quantile(SITE_SPEC, 0.0) == pytest.approx(10.0)
        assert _sample_from_dist_with_quantile(SITE_SPEC, 0.05) == pytest.approx(10.0)
        assert _sample_from_dist_with_quantile(SITE_SPEC, 0.90) == pytest.approx(1000.0)
        assert _sample_from_dist_with_quantile(SITE_SPEC, 1.0) == pytest.approx(1000.0)

    def test_linear_scale(self):
        spec = {"dist": "cdf_points", "scale": "linear",
                "points": [[0.0, 0.25], [4.0, 0.75]], "min": 0.0, "max": 4.0,
                "clip_to_bounds": True}
        assert _sample_from_dist_with_quantile(spec, 0.5) == pytest.approx(2.0)
        assert _sample_from_dist_with_quantile(spec, 0.375) == pytest.approx(1.0)

    def test_scale_defaults_to_linear(self):
        spec = {"dist": "cdf_points", "points": [[10, 0.10], [1000, 0.80]],
                "min": 10, "max": 1000, "clip_to_bounds": True}
        # Linear midpoint of [10, 1000] at the probability midpoint 0.45, not 10^2.
        assert _sample_from_dist_with_quantile(spec, 0.45) == pytest.approx(505.0)


class TestRandomSampling:
    def test_endpoint_point_masses_and_monotonicity(self):
        rng = np.random.default_rng(0)
        draws = np.array([_sample_from_dist(SITE_SPEC, rng) for _ in range(20000)])
        assert draws.min() >= 10.0 and draws.max() <= 1000.0
        # Point masses match the elicited tail probabilities.
        assert np.mean(draws == 10.0) == pytest.approx(0.10, abs=0.01)
        assert np.mean(draws == 1000.0) == pytest.approx(0.20, abs=0.01)
        # Empirical CDF reproduces the interior anchor.
        assert np.mean(draws <= 100.0) == pytest.approx(0.35, abs=0.01)

    def test_quantile_monotonicity(self):
        qs = np.linspace(0.0, 1.0, 201)
        xs = [_sample_from_dist_with_quantile(SITE_SPEC, q) for q in qs]
        assert all(b >= a for a, b in zip(xs, xs[1:]))


class TestValidation:
    def test_rejects_single_point(self):
        with pytest.raises(ValueError, match="at least two"):
            _sample_from_dist_with_quantile(
                {"dist": "cdf_points", "points": [[10, 0.5]], "min": 10, "max": 10}, 0.5)

    def test_rejects_non_increasing_values(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _sample_from_dist_with_quantile(
                {"dist": "cdf_points", "points": [[100, 0.1], [10, 0.8]],
                 "min": 10, "max": 100}, 0.5)

    def test_rejects_non_increasing_probs(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _sample_from_dist_with_quantile(
                {"dist": "cdf_points", "points": [[10, 0.8], [100, 0.1]],
                 "min": 10, "max": 100}, 0.5)

    def test_rejects_probs_outside_unit_interval(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _sample_from_dist_with_quantile(
                {"dist": "cdf_points", "points": [[10, 0.0], [100, 1.2]],
                 "min": 10, "max": 100}, 0.5)

    def test_rejects_log_scale_with_nonpositive_values(self):
        with pytest.raises(ValueError, match="positive"):
            _sample_from_dist_with_quantile(
                {"dist": "cdf_points", "scale": "log", "points": [[0.0, 0.1], [100, 0.8]],
                 "min": 0, "max": 100}, 0.5)

    def test_rejects_unknown_scale(self):
        with pytest.raises(ValueError, match="'linear' or 'log'"):
            _sample_from_dist_with_quantile(
                {"dist": "cdf_points", "scale": "log10", "points": [[10, 0.1], [100, 0.8]],
                 "min": 10, "max": 100}, 0.5)
