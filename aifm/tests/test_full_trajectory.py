"""
Snapshot tests for full model trajectory outputs.

These tests verify that the complete model trajectory matches golden baseline data.
This is the most important test file for catching any behavioral changes.
"""

import numpy as np
import pytest

from conftest import (
    assert_arrays_close,
    assert_scalar_close,
    compute_trajectory_for_params,
    STRICT_RTOL,
    TRAJECTORY_RTOL,
    MILESTONE_RTOL,
)


class TestFullTrajectory:
    """Test full trajectory computation against golden data."""

    def test_trajectory_times_match(self, computed_model, golden_trajectory):
        """Verify trajectory time points match golden data."""
        assert_arrays_close(
            computed_model.results['times'],
            golden_trajectory['times'],
            rtol=STRICT_RTOL,
            msg="Trajectory times don't match"
        )

    def test_progress_values_match(self, computed_model, golden_trajectory):
        """Verify progress values match golden data."""
        assert_arrays_close(
            computed_model.results['progress'],
            golden_trajectory['progress'],
            rtol=TRAJECTORY_RTOL,
            msg="Progress values don't match"
        )

    def test_research_stock_values_match(self, computed_model, golden_trajectory):
        """Verify research stock values match golden data."""
        # Use trajectory tolerance since research_stock is ODE-derived and
        # accumulates numerical differences through integration
        assert_arrays_close(
            computed_model.results['research_stock'],
            golden_trajectory['research_stock'],
            rtol=TRAJECTORY_RTOL,
            msg="Research stock values don't match"
        )

    def test_automation_fractions_match(self, computed_model, golden_trajectory):
        """Verify automation fractions match golden data."""
        assert_arrays_close(
            computed_model.results['automation_fraction'],
            golden_trajectory['automation_fractions'],
            rtol=TRAJECTORY_RTOL,
            msg="Automation fractions don't match"
        )

    def test_ai_research_tastes_match(self, computed_model, golden_trajectory):
        """Verify AI research taste values match golden data."""
        assert_arrays_close(
            computed_model.results['ai_research_taste'],
            golden_trajectory['ai_research_tastes'],
            rtol=TRAJECTORY_RTOL,
            msg="AI research tastes don't match"
        )

    def test_aggregate_research_tastes_match(self, computed_model, golden_trajectory):
        """Verify aggregate research taste values match golden data."""
        assert_arrays_close(
            computed_model.results['aggregate_research_taste'],
            golden_trajectory['aggregate_research_tastes'],
            rtol=TRAJECTORY_RTOL,
            msg="Aggregate research tastes don't match"
        )

    def test_coding_labors_match(self, computed_model, golden_trajectory):
        """Verify coding labor values match golden data."""
        assert_arrays_close(
            computed_model.results['coding_labors'],
            golden_trajectory['coding_labors'],
            rtol=TRAJECTORY_RTOL,
            msg="Coding labors don't match"
        )

    def test_software_progress_rates_match(self, computed_model, golden_trajectory):
        """Verify software progress rates match golden data."""
        assert_arrays_close(
            computed_model.results['software_progress_rates'],
            golden_trajectory['software_progress_rates'],
            rtol=TRAJECTORY_RTOL,
            msg="Software progress rates don't match"
        )

    def test_effective_compute_match(self, computed_model, golden_trajectory):
        """Verify effective compute values match golden data."""
        assert_arrays_close(
            computed_model.results['effective_compute'],
            golden_trajectory['effective_compute'],
            rtol=TRAJECTORY_RTOL,
            msg="Effective compute values don't match"
        )

    def test_horizon_lengths_match(self, computed_model, golden_trajectory):
        """Verify horizon length values match golden data."""
        assert_arrays_close(
            computed_model.results['horizon_lengths'],
            golden_trajectory['horizon_lengths'],
            rtol=TRAJECTORY_RTOL,
            msg="Horizon lengths don't match"
        )


class TestMilestones:
    """Test milestone computations against golden data."""

    def test_aa_time_matches(self, computed_model, golden_milestones):
        """Verify AC (full automation) time matches golden data."""
        expected = golden_milestones['aa_time']
        actual = computed_model.results.get('aa_time')

        if expected is not None and actual is not None:
            assert_scalar_close(
                actual, expected, rtol=MILESTONE_RTOL,
                msg="AC time doesn't match"
            )
        else:
            assert actual == expected, f"AC time mismatch: expected {expected}, got {actual}"

    def test_progress_at_aa_matches(self, computed_model, golden_milestones):
        """Verify progress at AA matches golden data."""
        expected = golden_milestones['progress_at_aa']
        actual = computed_model.params.progress_at_aa

        if expected is not None and actual is not None:
            assert_scalar_close(
                actual, expected, rtol=TRAJECTORY_RTOL,
                msg="Progress at AA doesn't match"
            )

    def test_final_progress_matches(self, computed_model, golden_milestones):
        """Verify final progress matches golden data."""
        expected = golden_milestones['final_progress']
        actual = float(computed_model.results['progress'][-1])

        assert_scalar_close(
            actual, expected, rtol=TRAJECTORY_RTOL,
            msg="Final progress doesn't match"
        )

    def test_sc_sw_multiplier_matches(self, computed_model, golden_milestones):
        """Verify SC software multiplier matches golden data."""
        expected = golden_milestones['sc_sw_multiplier']
        actual = getattr(computed_model, 'sc_sw_multiplier', None)

        if expected is not None and actual is not None:
            assert_scalar_close(
                actual, expected, rtol=MILESTONE_RTOL,
                msg="SC software multiplier doesn't match"
            )


class TestTrajectoryProperties:
    """Test general properties that should always hold."""

    def test_progress_is_monotonic(self, model_with_trajectory):
        """Progress should be monotonically increasing."""
        progress = np.array(model_with_trajectory.results['progress'])
        diffs = np.diff(progress)
        assert np.all(diffs >= -1e-10), "Progress should be monotonically increasing"

    def test_automation_fraction_bounded(self, model_with_trajectory):
        """Automation fraction should be between 0 and 1."""
        af = np.array(model_with_trajectory.results['automation_fraction'])
        assert np.all(af >= 0), "Automation fraction should be >= 0"
        assert np.all(af <= 1), "Automation fraction should be <= 1"

    def test_research_stock_positive(self, model_with_trajectory):
        """Research stock should always be positive."""
        rs = np.array(model_with_trajectory.results['research_stock'])
        assert np.all(rs > 0), "Research stock should always be positive"

    def test_coding_labor_positive(self, model_with_trajectory):
        """Coding labor should always be positive."""
        cl = np.array(model_with_trajectory.results['coding_labors'])
        assert np.all(cl > 0), "Coding labor should always be positive"

    def test_all_outputs_finite(self, model_with_trajectory):
        """All output arrays should contain only finite values."""
        results = model_with_trajectory.results
        arrays_to_check = [
            'times', 'progress', 'research_stock', 'progress_rates',
            'automation_fraction', 'ai_research_taste', 'aggregate_research_taste',
            'coding_labors', 'software_progress_rates', 'effective_compute'
        ]

        for key in arrays_to_check:
            if key in results and len(results[key]) > 0:
                arr = np.array(results[key])
                assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"


# =============================================================================
# Tests with Non-Default Parameter Values
# =============================================================================
# These tests verify the model runs correctly across the full range of
# parameter values from progress_model.config.PARAMETER_BOUNDS and sampling_config.yaml


class TestNonDefaultTrajectoryGolden:
    """Test non-default parameter trajectories against golden baseline data.

    These tests compare full trajectory outputs against saved golden data
    for each representative parameter scenario. This catches any unintended
    behavioral changes when parameters are modified.

    To regenerate all golden data after intentional changes:
        pytest tests/test_full_trajectory.py --regen-golden
    (This regenerates goldens and exits; rerun without the flag to execute tests.)
    """

    # Fields to check against golden data (model key -> golden key if different)
    GOLDEN_FIELDS = [
        'progress',
        'research_stock',
        'automation_fraction',
        'ai_research_taste',
        'coding_labors',
        'effective_compute',
    ]

    def test_trajectory_matches_golden(
        self, param_set_scenario, computed_scenario_model, scenario_golden_data
    ):
        """Verify all trajectory outputs match golden data for each scenario."""
        scenario_name = param_set_scenario['name']

        for field in self.GOLDEN_FIELDS:
            assert_arrays_close(
                computed_scenario_model.results[field],
                scenario_golden_data[field],
                rtol=TRAJECTORY_RTOL,
                msg=f"{field} mismatch for scenario {scenario_name}"
            )

    def test_milestone_times_match_golden(
        self, param_set_scenario, computed_scenario_model, scenario_golden_data
    ):
        """Verify milestone times match golden data for each scenario."""
        scenario_name = param_set_scenario['name']

        # Compare AA time if both exist
        expected_aa = scenario_golden_data.get('aa_time')
        actual_aa = computed_scenario_model.results.get('aa_time')
        if expected_aa is not None and actual_aa is not None:
            assert_scalar_close(
                actual_aa, expected_aa, rtol=MILESTONE_RTOL,
                msg=f"AA time mismatch for scenario {scenario_name}"
            )
        elif expected_aa is None and actual_aa is None:
            pass  # Both None is fine
        else:
            # One is None and the other isn't - this is a difference
            assert expected_aa == actual_aa, \
                f"AA time mismatch for {scenario_name}: expected {expected_aa}, got {actual_aa}"


class TestTrajectoryWithNonDefaultParams:
    """Test full trajectory computation with non-default parameter sets.

    These tests verify that the model:
    1. Runs without errors for various parameter combinations
    2. Produces valid output (finite values, monotonic progress, etc.)
    3. Respects physical constraints (automation fraction bounded, etc.)

    Uses cached computed_scenario_model fixture to avoid redundant computation.
    """

    # Arrays to check for finiteness
    ARRAYS_TO_CHECK = [
        'times', 'progress', 'research_stock', 'progress_rates',
        'automation_fraction', 'ai_research_taste', 'aggregate_research_taste',
        'coding_labors', 'software_progress_rates', 'effective_compute',
    ]

    def test_trajectory_properties_for_each_scenario(
        self, param_set_scenario, computed_scenario_model
    ):
        """Verify all trajectory properties for each representative parameter set.

        Checks:
        - Computation produced results with required fields
        - Progress is monotonically increasing
        - Automation fraction is bounded in [0, 1]
        - All output arrays contain only finite values
        - Research stock is always positive
        """
        scenario_name = param_set_scenario['name']
        model = computed_scenario_model

        # Verify computation produced results
        assert model.results is not None, f"No results for scenario {scenario_name}"
        assert 'times' in model.results, f"No times in results for {scenario_name}"
        assert 'progress' in model.results, f"No progress in results for {scenario_name}"
        assert len(model.results['times']) > 0, f"Empty times for {scenario_name}"

        # Progress should be monotonically increasing
        progress = np.array(model.results['progress'])
        diffs = np.diff(progress)
        assert np.all(diffs >= -1e-10), f"Progress not monotonic for scenario {scenario_name}"

        # Automation fraction should be in [0, 1]
        af = np.array(model.results['automation_fraction'])
        assert np.all(af >= 0), f"Automation fraction < 0 for {scenario_name}"
        assert np.all(af <= 1), f"Automation fraction > 1 for {scenario_name}"

        # All output arrays should contain only finite values
        for key in self.ARRAYS_TO_CHECK:
            if key in model.results and len(model.results[key]) > 0:
                arr = np.array(model.results[key])
                assert np.all(np.isfinite(arr)), \
                    f"{key} contains non-finite values for scenario {scenario_name}"

        # Research stock should always be positive
        rs = np.array(model.results['research_stock'])
        assert np.all(rs > 0), f"Research stock not positive for {scenario_name}"


class TestTrajectoryParameterSensitivity:
    """Test trajectory sensitivity to individual parameter variations.

    These tests verify the model behaves sensibly when individual parameters
    are varied across their allowed ranges.
    """

    @pytest.mark.parametrize("rho", [-5, -2, -1])
    def test_rho_coding_labor_variations(
        self, rho, default_time_series_data
    ):
        """Test model with different rho_coding_labor values from sampling config."""
        model = compute_trajectory_for_params(
            {'rho_coding_labor': rho},
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        # Verify basic properties
        progress = np.array(model.results['progress'])
        assert np.all(np.diff(progress) >= -1e-10), f"Progress not monotonic for rho={rho}"
        assert np.all(np.isfinite(progress)), f"Non-finite progress for rho={rho}"

    @pytest.mark.parametrize("doubling_time", [0.29, 0.5, 0.717])
    def test_present_doubling_time_variations(
        self, doubling_time, default_time_series_data
    ):
        """Test model with different present_doubling_time values."""
        model = compute_trajectory_for_params(
            {
                'present_doubling_time': doubling_time,
                'horizon_extrapolation_type': 'decaying doubling time',
            },
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.diff(progress) >= -1e-10), f"Progress not monotonic for doubling_time={doubling_time}"

    @pytest.mark.parametrize("growth_factor", [0.82, 0.92, 1.02])
    def test_doubling_difficulty_growth_factor_variations(
        self, growth_factor, default_time_series_data
    ):
        """Test model with different doubling_difficulty_growth_factor values."""
        model = compute_trajectory_for_params(
            {
                'doubling_difficulty_growth_factor': growth_factor,
                'horizon_extrapolation_type': 'decaying doubling time',
            },
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), f"Non-finite progress for growth_factor={growth_factor}"

    @pytest.mark.parametrize("taste_slope", [0.7, 2.0, 6.9])
    def test_ai_research_taste_slope_variations(
        self, taste_slope, default_time_series_data
    ):
        """Test model with different ai_research_taste_slope values."""
        model = compute_trajectory_for_params(
            {'ai_research_taste_slope': taste_slope},
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        taste = np.array(model.results['ai_research_taste'])
        assert np.all(taste > 0), f"Taste not positive for taste_slope={taste_slope}"
        assert np.all(np.isfinite(taste)), f"Non-finite taste for taste_slope={taste_slope}"

    @pytest.mark.parametrize("inf_compute", [25, 1000, 40000])
    def test_inf_compute_asymptote_variations(
        self, inf_compute, default_time_series_data
    ):
        """Test model with different inf_compute_asymptote values."""
        model = compute_trajectory_for_params(
            {'inf_compute_asymptote': inf_compute},
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), f"Non-finite progress for inf_compute={inf_compute}"


class TestTrajectoryComparisons:
    """Test that parameter variations produce expected relative effects on trajectories.

    These tests compare trajectories to verify that changing parameters
    affects the results in expected ways.
    """

    def test_faster_doubling_time_leads_to_more_progress(
        self, default_time_series_data
    ):
        """Faster doubling time should lead to more progress by end of simulation.

        ac_anchor_mode is stated because this is a property OF THE HORIZON ANCHOR:
        present_doubling_time only reaches the trajectory through the horizon trend
        that fixes AC's progress level. The default mode is 'uplift-solved' now, where
        AC comes out of the uplift solve and the horizon trend is display-only -- both
        runs there return the same final progress to the last bit (66.12452731679527),
        so the comparison has nothing to compare.
        """
        model_fast = compute_trajectory_for_params(
            {
                'present_doubling_time': 0.3,
                'horizon_extrapolation_type': 'decaying doubling time',
                'ac_anchor_mode': 'time-horizon',
            },
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        model_slow = compute_trajectory_for_params(
            {
                'present_doubling_time': 0.7,
                'horizon_extrapolation_type': 'decaying doubling time',
                'ac_anchor_mode': 'time-horizon',
            },
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        final_progress_fast = model_fast.results['progress'][-1]
        final_progress_slow = model_slow.results['progress'][-1]

        assert final_progress_fast > final_progress_slow, \
            f"Faster doubling time ({final_progress_fast}) should yield more progress than slower ({final_progress_slow})"

    def test_lower_growth_factor_leads_to_more_progress(
        self, default_time_series_data
    ):
        """Lower doubling_difficulty_growth_factor should lead to more progress.

        ac_anchor_mode stated for the same reason as the test above: the growth factor
        is a horizon-trend parameter, and only the horizon anchor turns it into
        trajectory.
        """
        model_low = compute_trajectory_for_params(
            {
                'doubling_difficulty_growth_factor': 0.82,
                'horizon_extrapolation_type': 'decaying doubling time',
                'ac_anchor_mode': 'time-horizon',
            },
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        model_high = compute_trajectory_for_params(
            {
                'doubling_difficulty_growth_factor': 1.0,
                'horizon_extrapolation_type': 'decaying doubling time',
                'ac_anchor_mode': 'time-horizon',
            },
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        final_progress_low = model_low.results['progress'][-1]
        final_progress_high = model_high.results['progress'][-1]

        assert final_progress_low > final_progress_high, \
            f"Lower growth factor ({final_progress_low}) should yield more progress than higher ({final_progress_high})"

    def test_higher_taste_slope_leads_to_more_progress(
        self, default_time_series_data
    ):
        """Higher ai_research_taste_slope should generally lead to more progress."""
        model_high = compute_trajectory_for_params(
            {'ai_research_taste_slope': 5.0},
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        model_low = compute_trajectory_for_params(
            {'ai_research_taste_slope': 0.8},
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        final_progress_high = model_high.results['progress'][-1]
        final_progress_low = model_low.results['progress'][-1]

        assert final_progress_high > final_progress_low, \
            f"Higher taste slope ({final_progress_high}) should yield more progress than lower ({final_progress_low})"

    def test_different_rho_produces_different_trajectories(
        self, default_time_series_data
    ):
        """Different rho_coding_labor values should produce meaningfully different trajectories."""
        model_rho_minus5 = compute_trajectory_for_params(
            {'rho_coding_labor': -5},
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        model_rho_minus1 = compute_trajectory_for_params(
            {'rho_coding_labor': -1},
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        progress_minus5 = np.array(model_rho_minus5.results['progress'])
        progress_minus1 = np.array(model_rho_minus1.results['progress'])

        # Trajectories should differ meaningfully (not just numerical noise)
        # Use the last half of the simulation where differences should be more pronounced
        mid_idx = len(progress_minus5) // 2
        relative_diff = np.abs(progress_minus5[mid_idx:] - progress_minus1[mid_idx:]) / (
            np.maximum(np.abs(progress_minus5[mid_idx:]), 1e-10)
        )
        max_relative_diff = np.max(relative_diff)

        assert max_relative_diff > 0.01, \
            f"Rho variations should produce meaningfully different trajectories (max rel diff: {max_relative_diff})"


class TestEdgeCaseParameters:
    """Test model behavior at edge cases and boundary parameter values.

    These tests ensure the model handles extreme but valid parameter values gracefully.
    """

    def test_very_fast_doubling_time(self, default_time_series_data):
        """Test with very fast doubling time near lower bound."""
        model = compute_trajectory_for_params(
            {
                'present_doubling_time': 0.1,
                'horizon_extrapolation_type': 'decaying doubling time',
            },
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), "Non-finite progress with very fast doubling"

    def test_very_high_rho(self, default_time_series_data):
        """Test with very high (near 0) rho value."""
        model = compute_trajectory_for_params(
            {'rho_coding_labor': -0.001},
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), "Non-finite progress with high rho"

    def test_very_low_rho(self, default_time_series_data):
        """Test with very low rho value (high complementarity)."""
        model = compute_trajectory_for_params(
            {'rho_coding_labor': -8},
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), "Non-finite progress with low rho"

    def test_high_inf_asymptotes(self, default_time_series_data):
        """Test with high experiment capacity asymptotes."""
        model = compute_trajectory_for_params(
            {
                'inf_compute_asymptote': 100000,
                'inf_labor_asymptote': 1000,
            },
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), "Non-finite progress with high asymptotes"

    def test_low_inf_asymptotes(self, default_time_series_data):
        """Test with low experiment capacity asymptotes."""
        model = compute_trajectory_for_params(
            {
                'inf_compute_asymptote': 5,
                'inf_labor_asymptote': 2,
            },
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), "Non-finite progress with low asymptotes"

    def test_extreme_taste_parameters(self, default_time_series_data):
        """Test with extreme but valid taste distribution parameters."""
        model = compute_trajectory_for_params(
            {
                'ai_research_taste_slope': 8.0,
                'ai_research_taste_at_coding_automation_anchor_sd': 3.0,
                'median_to_top_taste_multiplier': 20.0,
                'taste_limit': 30.0,
            },
            default_time_series_data,
            time_range=[2015.0, 2040.0],
        )

        taste = np.array(model.results['ai_research_taste'])
        assert np.all(np.isfinite(taste)), "Non-finite taste with extreme parameters"
        assert np.all(taste > 0), "Taste not positive with extreme parameters"


class TestGapModeTrajectories:
    """Test trajectories with gap mode enabled."""

    def test_gap_mode_completes(self, default_time_series_data):
        """Test that gap mode trajectories complete successfully."""
        model = compute_trajectory_for_params(
            {
                'include_gap': 'gap',
                'gap_years': 3.0,
                'ac_time_horizon_minutes': 1.2e7,
            },
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), "Non-finite progress in gap mode"
        assert np.all(np.diff(progress) >= -1e-10), "Progress not monotonic in gap mode"

    @pytest.mark.parametrize("gap_years", [0.5, 2.0, 5.0, 10.0])
    def test_different_gap_lengths(self, gap_years, default_time_series_data):
        """Test gap mode with different gap lengths."""
        model = compute_trajectory_for_params(
            {
                'include_gap': 'gap',
                'gap_years': gap_years,
            },
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        progress = np.array(model.results['progress'])
        assert np.all(np.isfinite(progress)), f"Non-finite progress for gap_years={gap_years}"

    def test_gap_mode_produces_different_trajectory_than_no_gap(
        self, default_time_series_data
    ):
        """Verify gap mode produces a different trajectory than no-gap mode."""
        model_gap = compute_trajectory_for_params(
            {
                'include_gap': 'gap',
                'gap_years': 5.0,
            },
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        model_no_gap = compute_trajectory_for_params(
            {
                'include_gap': 'no gap',
            },
            default_time_series_data,
            time_range=[2015.0, 2050.0],
        )

        progress_gap = np.array(model_gap.results['progress'])
        progress_no_gap = np.array(model_no_gap.results['progress'])

        # Trajectories should differ
        assert not np.allclose(progress_gap, progress_no_gap, rtol=0.01), \
            "Gap mode should produce different trajectory than no-gap mode"


class TestHumanOnlyAnchorStats:
    """Test human-only trajectory anchor stats against golden data.

    These tests specifically verify the values computed by the human-only solver
    that feed into the full trajectory computation. This helps isolate issues
    in the vectorized human-only integration from issues in the main ODE solver.

    The anchor stats are the critical outputs:
    - progress: cumulative progress at present_day
    - progress_rate: overall progress rate at present_day
    - sw_progress_rate: software progress rate at present_day
    - research_effort: research effort at present_day
    - research_stock: accumulated research stock at present_day
    - human_labor: human coding labor at present_day
    - inference_compute: inference compute at present_day
    - experiment_compute: experiment compute at present_day
    """

    # Fields that should be present in anchor_stats
    ANCHOR_STAT_FIELDS = [
        'progress',
        'progress_rate',
        'sw_progress_rate',
        'research_effort',
        'research_stock',
        'human_labor',
        'inference_compute',
        'experiment_compute',
    ]

    def test_anchor_stats_match_golden(
        self, param_set_scenario, computed_scenario_model, scenario_golden_data
    ):
        """Verify human-only anchor stats match golden data for each scenario."""
        scenario_name = param_set_scenario['name']
        model = computed_scenario_model

        # Skip if golden data doesn't have human-only anchor stats (old format)
        if 'human_only_anchor_stats' not in scenario_golden_data:
            pytest.skip(f"Golden data for '{scenario_name}' doesn't include human_only_anchor_stats. "
                       "Regenerate with --regen-golden.")

        golden_anchor = scenario_golden_data['human_only_anchor_stats']
        actual_anchor = model.human_only_results.get('anchor_stats', {})

        for field in self.ANCHOR_STAT_FIELDS:
            if field not in golden_anchor:
                continue  # Skip if field not in golden data

            expected = golden_anchor[field]
            actual = actual_anchor.get(field)

            if expected is None and actual is None:
                continue

            assert actual is not None, \
                f"Missing anchor stat '{field}' for scenario {scenario_name}"

            assert_scalar_close(
                actual, expected, rtol=TRAJECTORY_RTOL,
                msg=f"anchor_stats['{field}'] mismatch for scenario {scenario_name}"
            )

    def test_reference_sw_progress_rate_matches_golden(
        self, param_set_scenario, computed_scenario_model, scenario_golden_data
    ):
        """Verify reference year software progress rate matches golden data."""
        scenario_name = param_set_scenario['name']
        model = computed_scenario_model

        # Skip if golden data doesn't have this field (old format)
        if 'human_only_reference_sw_progress_rate' not in scenario_golden_data:
            pytest.skip(f"Golden data for '{scenario_name}' doesn't include reference_sw_progress_rate. "
                       "Regenerate with --regen-golden.")

        expected = scenario_golden_data['human_only_reference_sw_progress_rate']
        actual = model.human_only_results.get('reference_sw_progress_rate')

        if expected is not None and actual is not None:
            assert_scalar_close(
                actual, expected, rtol=TRAJECTORY_RTOL,
                msg=f"reference_sw_progress_rate mismatch for scenario {scenario_name}"
            )

    def test_anchor_stats_are_finite(
        self, param_set_scenario, computed_scenario_model
    ):
        """Verify all anchor stats are finite for each scenario."""
        scenario_name = param_set_scenario['name']
        model = computed_scenario_model

        anchor_stats = model.human_only_results.get('anchor_stats', {})

        for field in self.ANCHOR_STAT_FIELDS:
            if field in anchor_stats:
                value = anchor_stats[field]
                assert np.isfinite(value), \
                    f"anchor_stats['{field}'] is not finite for scenario {scenario_name}: {value}"

    def test_anchor_stats_have_expected_signs(
        self, param_set_scenario, computed_scenario_model
    ):
        """Verify anchor stats have physically sensible signs."""
        scenario_name = param_set_scenario['name']
        model = computed_scenario_model

        anchor_stats = model.human_only_results.get('anchor_stats', {})

        # All these should be positive
        positive_fields = [
            'progress_rate', 'sw_progress_rate', 'research_effort',
            'research_stock', 'human_labor', 'inference_compute', 'experiment_compute'
        ]

        for field in positive_fields:
            if field in anchor_stats:
                value = anchor_stats[field]
                assert value > 0, \
                    f"anchor_stats['{field}'] should be positive for scenario {scenario_name}: {value}"
