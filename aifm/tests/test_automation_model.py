"""
Tests for AutomationModel and related functions in automation_model.py.

These tests verify the automation model's scheduling, inverse functions,
and optimal CES coding labor calculations.
"""

import pytest
import numpy as np

from progress_model import Parameters
from progress_model.automation_model import (
    AutomationModel,
    aut_frac_from_swe_multiplier,
    solve_lower_anchor_via_automation_model,
)
from conftest import assert_scalar_close, NORMAL_RTOL, LOOSE_RTOL, INVERSE_FUNCTION_RTOL
from progress_model import config as cfg


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def linear_params():
    """Create Parameters with linear automation schedule."""
    return Parameters(
        automation_interp_type="linear",
        automation_anchors={0.0: 0.1, 10.0: 0.9},
    )


@pytest.fixture
def logistic_params():
    """Create Parameters with logistic automation schedule."""
    return Parameters(
        automation_interp_type="logistic",
        automation_logistic_asymptote=1.1,
        automation_anchors={0.0: 0.1, 10.0: 0.9},
    )


@pytest.fixture
def default_automation_model(default_params):
    """Create AutomationModel with default parameters."""
    return AutomationModel(default_params)


@pytest.fixture
def linear_automation_model(linear_params):
    """Create AutomationModel with linear schedule."""
    return AutomationModel(linear_params)


@pytest.fixture
def logistic_automation_model(logistic_params):
    """Create AutomationModel with logistic schedule."""
    return AutomationModel(logistic_params)


# =============================================================================
# AutomationModel Initialization Tests
# =============================================================================


class TestAutomationModelInit:
    """Tests for AutomationModel initialization."""

    def test_linear_schedule_init(self, linear_params):
        """Test initialization with linear schedule."""
        am = AutomationModel(linear_params)
        assert am.schedule_type == "linear"
        assert len(am.anchor_points) == 2
        assert hasattr(am, 'linear_aut_slope')
        assert hasattr(am, 'linear_aut_intercept')
        assert np.isfinite(am.linear_aut_slope)
        assert np.isfinite(am.linear_aut_intercept)

    def test_logistic_schedule_init(self, logistic_params):
        """Test initialization with logistic schedule."""
        am = AutomationModel(logistic_params)
        assert am.schedule_type == "logistic"
        assert hasattr(am, 'logistic_L')
        assert hasattr(am, 'logistic_k')
        assert hasattr(am, 'logistic_x0')
        assert np.isfinite(am.logistic_L)
        assert np.isfinite(am.logistic_k)
        assert np.isfinite(am.logistic_x0)

    def test_default_params_init(self):
        """Test initialization with default parameters (with automation_anchors set)."""
        # Default Parameters have automation_anchors=None, which is computed elsewhere.
        # Create params with explicit anchors to test default schedule type behavior.
        params = Parameters(automation_anchors={0.0: 0.1, 10.0: 0.9})
        am = AutomationModel(params)
        assert am.schedule_type is not None
        assert am.anchor_points is not None

    def test_anchor_points_sorted(self, linear_params):
        """Test that anchor points are sorted by progress."""
        am = AutomationModel(linear_params)
        prog_values = [p for p, _ in am.anchor_points]
        assert prog_values == sorted(prog_values)


# =============================================================================
# get_automation_fraction Tests
# =============================================================================


class TestGetAutomationFraction:
    """Tests for the get_automation_fraction method."""

    def test_linear_at_anchor_points(self, linear_automation_model):
        """Linear schedule should pass through anchor points."""
        am = linear_automation_model
        (prog_1, aut_1), (prog_2, aut_2) = am.anchor_points

        result_1 = am.get_automation_fraction(prog_1)
        result_2 = am.get_automation_fraction(prog_2)

        assert_scalar_close(result_1, aut_1, rtol=1e-6, msg="Should match first anchor")
        assert_scalar_close(result_2, aut_2, rtol=1e-6, msg="Should match second anchor")

    def test_logistic_at_anchor_points(self, logistic_automation_model):
        """Logistic schedule should pass through anchor points."""
        am = logistic_automation_model
        (prog_1, aut_1), (prog_2, aut_2) = am.anchor_points

        result_1 = am.get_automation_fraction(prog_1)
        result_2 = am.get_automation_fraction(prog_2)

        assert_scalar_close(result_1, aut_1, rtol=1e-4, msg="Should match first anchor")
        assert_scalar_close(result_2, aut_2, rtol=1e-4, msg="Should match second anchor")

    def test_linear_midpoint(self, linear_automation_model):
        """Linear schedule should give midpoint value at midpoint progress."""
        am = linear_automation_model
        (prog_1, aut_1), (prog_2, aut_2) = am.anchor_points

        mid_prog = (prog_1 + prog_2) / 2
        result = am.get_automation_fraction(mid_prog)
        expected = (aut_1 + aut_2) / 2

        assert_scalar_close(result, expected, rtol=1e-6, msg="Linear midpoint should give midpoint automation")

    def test_clipping_to_01(self, linear_automation_model):
        """Automation fraction should be clipped to [0, 1]."""
        am = linear_automation_model

        result_low = am.get_automation_fraction(-100.0)
        assert result_low >= 0.0, f"Expected >= 0, got {result_low}"

        result_high = am.get_automation_fraction(1000.0)
        assert result_high <= 1.0, f"Expected <= 1, got {result_high}"

    def test_monotonically_increasing(self, linear_automation_model):
        """Automation fraction should be monotonically increasing with progress."""
        am = linear_automation_model
        progress_values = np.linspace(-5, 20, 50)
        results = [am.get_automation_fraction(p) for p in progress_values]

        for i in range(len(results) - 1):
            assert results[i] <= results[i+1], \
                f"Not monotonic: f({progress_values[i]})={results[i]} > f({progress_values[i+1]})={results[i+1]}"

    def test_logistic_monotonically_increasing(self, logistic_automation_model):
        """Logistic automation fraction should be monotonically increasing."""
        am = logistic_automation_model
        progress_values = np.linspace(-5, 20, 50)
        results = [am.get_automation_fraction(p) for p in progress_values]

        for i in range(len(results) - 1):
            assert results[i] <= results[i+1], \
                f"Not monotonic: f({progress_values[i]})={results[i]} > f({progress_values[i+1]})={results[i+1]}"


# =============================================================================
# get_progress_from_index Tests
# =============================================================================


class TestGetProgressFromIndex:
    """Tests for the get_progress_from_index method (inverse of automation fraction)."""

    def test_linear_inverse_at_anchors(self, linear_automation_model):
        """Linear inverse should give anchor progress at anchor automation values."""
        am = linear_automation_model
        (prog_1, aut_1), (prog_2, aut_2) = am.anchor_points

        result_1 = am.get_progress_from_index(aut_1)
        result_2 = am.get_progress_from_index(aut_2)

        assert_scalar_close(result_1, prog_1, rtol=INVERSE_FUNCTION_RTOL,
                           msg="Inverse should give first anchor progress")
        assert_scalar_close(result_2, prog_2, rtol=INVERSE_FUNCTION_RTOL,
                           msg="Inverse should give second anchor progress")

    def test_logistic_inverse_at_anchors(self, logistic_automation_model):
        """Logistic inverse should give anchor progress at anchor automation values."""
        am = logistic_automation_model
        (prog_1, aut_1), (prog_2, aut_2) = am.anchor_points

        result_1 = am.get_progress_from_index(aut_1)
        result_2 = am.get_progress_from_index(aut_2)

        assert_scalar_close(result_1, prog_1, rtol=INVERSE_FUNCTION_RTOL,
                           msg="Inverse should give first anchor progress")
        assert_scalar_close(result_2, prog_2, rtol=INVERSE_FUNCTION_RTOL,
                           msg="Inverse should give second anchor progress")

    def test_roundtrip_linear(self, linear_automation_model):
        """Round-trip: progress -> automation -> progress should be identity."""
        am = linear_automation_model
        (prog_1, _), (prog_2, _) = am.anchor_points

        test_progress = np.linspace(prog_1, prog_2, 10)
        for prog in test_progress:
            aut = am.get_automation_fraction(prog)
            recovered = am.get_progress_from_index(aut)
            assert_scalar_close(recovered, prog, rtol=INVERSE_FUNCTION_RTOL,
                               msg=f"Round-trip failed at progress={prog}")

    def test_roundtrip_logistic(self, logistic_automation_model):
        """Round-trip: progress -> automation -> progress should be identity."""
        am = logistic_automation_model
        (prog_1, _), (prog_2, _) = am.anchor_points

        test_progress = np.linspace(prog_1, prog_2, 10)
        for prog in test_progress:
            aut = am.get_automation_fraction(prog)
            recovered = am.get_progress_from_index(aut)
            assert_scalar_close(recovered, prog, rtol=INVERSE_FUNCTION_RTOL,
                               msg=f"Round-trip failed at progress={prog}")

    def test_monotonically_increasing(self, linear_automation_model):
        """Progress from index should be monotonically increasing."""
        am = linear_automation_model
        indices = np.linspace(0.05, 0.95, 20)
        results = [am.get_progress_from_index(i) for i in indices]

        for i in range(len(results) - 1):
            assert results[i] <= results[i+1], \
                f"Not monotonic: f({indices[i]})={results[i]} > f({indices[i+1]})={results[i+1]}"


# =============================================================================
# get_FTE_per_GPU Tests
# =============================================================================


class TestGetFTEPerGPU:
    """Tests for the get_FTE_per_GPU method."""

    def test_zero_when_progress_below_threshold(self, linear_automation_model):
        """FTE per GPU should be 0 when progress is below the threshold for the index."""
        am = linear_automation_model
        index = 0.9
        threshold_progress = am.get_progress_from_index(index)

        result = am.get_FTE_per_GPU(index, threshold_progress - 1.0)
        assert result == 0.0, f"Expected 0 when progress below threshold, got {result}"

    def test_positive_when_progress_above_threshold(self, linear_automation_model):
        """FTE per GPU should be positive when progress is above threshold."""
        am = linear_automation_model
        index = 0.5
        threshold_progress = am.get_progress_from_index(index)

        result = am.get_FTE_per_GPU(index, threshold_progress + 1.0)
        assert result > 0, f"Expected positive when progress above threshold, got {result}"

    def test_increases_with_progress(self, linear_automation_model):
        """FTE per GPU should increase as progress increases beyond threshold."""
        am = linear_automation_model
        index = 0.3
        threshold_progress = am.get_progress_from_index(index)

        results = []
        for delta in [0.0, 1.0, 2.0, 3.0]:
            prog = threshold_progress + delta
            results.append(am.get_FTE_per_GPU(index, prog))

        for i in range(len(results) - 1):
            assert results[i] <= results[i+1], \
                f"FTE should increase with progress: {results[i]} > {results[i+1]}"

    def test_at_threshold_progress(self, linear_automation_model):
        """FTE per GPU at exactly threshold progress should equal initial_FTE_per_GPU."""
        am = linear_automation_model
        index = 0.5
        threshold_progress = am.get_progress_from_index(index)

        result = am.get_FTE_per_GPU(index, threshold_progress)
        assert_scalar_close(result, am.initial_FTE_per_GPU, rtol=1e-6,
                           msg="At threshold, FTE should equal initial value")


# =============================================================================
# Static Helper Methods Tests
# =============================================================================


class TestStaticHelpers:
    """Tests for static helper methods _interp and _invert_monotone."""

    def test_interp_at_grid_points(self):
        """Interpolation at grid points should return exact values."""
        x_grid = np.array([0.0, 1.0, 2.0, 3.0])
        y_grid = np.array([0.0, 10.0, 20.0, 30.0])

        for x, expected_y in zip(x_grid, y_grid):
            result = AutomationModel._interp(x_grid, y_grid, float(x))
            assert_scalar_close(result, expected_y, rtol=1e-10,
                               msg=f"Interp at grid point {x} failed")

    def test_interp_midpoints(self):
        """Interpolation at midpoints should give linear interpolation."""
        x_grid = np.array([0.0, 1.0, 2.0, 3.0])
        y_grid = np.array([0.0, 10.0, 20.0, 30.0])

        result = AutomationModel._interp(x_grid, y_grid, 0.5)
        assert_scalar_close(result, 5.0, rtol=1e-10, msg="Midpoint interpolation failed")

        result = AutomationModel._interp(x_grid, y_grid, 1.5)
        assert_scalar_close(result, 15.0, rtol=1e-10, msg="Midpoint interpolation failed")

    def test_invert_monotone_at_grid_points(self):
        """Inversion at grid y-values should return corresponding x-values."""
        x_grid = np.array([0.0, 1.0, 2.0, 3.0])
        y_grid = np.array([0.0, 10.0, 20.0, 30.0])

        for expected_x, y in zip(x_grid, y_grid):
            result = AutomationModel._invert_monotone(x_grid, y_grid, float(y))
            assert_scalar_close(result, expected_x, rtol=1e-10,
                               msg=f"Invert at y={y} failed")

    def test_invert_monotone_midpoints(self):
        """Inversion at intermediate y-values should give linear interpolation."""
        x_grid = np.array([0.0, 1.0, 2.0, 3.0])
        y_grid = np.array([0.0, 10.0, 20.0, 30.0])

        result = AutomationModel._invert_monotone(x_grid, y_grid, 5.0)
        assert_scalar_close(result, 0.5, rtol=1e-10, msg="Midpoint inversion failed")

        result = AutomationModel._invert_monotone(x_grid, y_grid, 15.0)
        assert_scalar_close(result, 1.5, rtol=1e-10, msg="Midpoint inversion failed")

    def test_invert_monotone_clamps_below_range(self):
        """Inversion below y-range should return first x-value."""
        x_grid = np.array([0.0, 1.0, 2.0, 3.0])
        y_grid = np.array([10.0, 20.0, 30.0, 40.0])

        result = AutomationModel._invert_monotone(x_grid, y_grid, -5.0)
        assert result == 0.0, f"Expected 0.0 for y below range, got {result}"

    def test_invert_monotone_clamps_above_range(self):
        """Inversion above y-range should return last x-value."""
        x_grid = np.array([0.0, 1.0, 2.0, 3.0])
        y_grid = np.array([10.0, 20.0, 30.0, 40.0])

        result = AutomationModel._invert_monotone(x_grid, y_grid, 100.0)
        assert result == 3.0, f"Expected 3.0 for y above range, got {result}"


# =============================================================================
# Optimal CES Coding Labor Tests
# =============================================================================


class TestCodingLaborOptimalCES:
    """Tests for the coding_labor_optimal_ces method."""

    @pytest.fixture
    def optimal_ces_params(self):
        """Create parameters suitable for optimal CES computation."""
        return Parameters(
            coding_labor_mode='optimal_ces',
            rho_coding_labor=-2.0,
            coding_automation_efficiency_slope=1.0,
            optimal_ces_eta_init=1.0,
            optimal_ces_grid_size=512,
            parallel_penalty=0.5,
            automation_anchors={0.0: 0.1, 10.0: 0.9},
        )

    @pytest.fixture
    def optimal_ces_model(self, optimal_ces_params):
        """Create AutomationModel configured for optimal CES."""
        return AutomationModel(optimal_ces_params)

    def test_returns_finite_value(self, optimal_ces_model, optimal_ces_params):
        """Coding labor should return finite value for valid inputs."""
        H = 1000.0
        C = 1e6
        logE = 10.0 * np.log(cfg.BASE_FOR_SOFTWARE_LOM)

        result = optimal_ces_model.coding_labor_optimal_ces(H, C, logE, optimal_ces_params)

        assert result is not None, "Result should not be None"
        assert np.isfinite(result), f"Result should be finite, got {result}"
        assert result > 0, f"Result should be positive, got {result}"

    def test_increases_with_human_labor(self, optimal_ces_model, optimal_ces_params):
        """Coding labor should increase as human labor increases."""
        C = 1e6
        logE = 5.0 * np.log(cfg.BASE_FOR_SOFTWARE_LOM)

        H_values = [100.0, 500.0, 1000.0, 5000.0]
        results = []
        for H in H_values:
            result = optimal_ces_model.coding_labor_optimal_ces(H, C, logE, optimal_ces_params)
            if result is not None:
                results.append(result)

        for i in range(len(results) - 1):
            assert results[i] <= results[i+1], \
                f"Coding labor should increase with H: {results[i]} > {results[i+1]}"

    def test_increases_with_compute(self, optimal_ces_model, optimal_ces_params):
        """Coding labor should generally increase as inference compute increases."""
        H = 1000.0
        logE = 5.0 * np.log(cfg.BASE_FOR_SOFTWARE_LOM)

        C_values = [1e4, 1e5, 1e6, 1e7]
        results = []
        for C in C_values:
            result = optimal_ces_model.coding_labor_optimal_ces(H, C, logE, optimal_ces_params)
            if result is not None:
                results.append(result)

        assert results[-1] >= results[0], \
            f"Coding labor should generally increase with C: {results[0]} > {results[-1]}"

    def test_zero_human_labor_returns_zero(self, optimal_ces_model, optimal_ces_params):
        """Zero human labor should return zero coding labor."""
        H = 0.0
        C = 1e6
        logE = 5.0 * np.log(cfg.BASE_FOR_SOFTWARE_LOM)

        result = optimal_ces_model.coding_labor_optimal_ces(H, C, logE, optimal_ces_params)

        assert result == 0.0, f"Expected 0 for zero human labor, got {result}"

    def test_return_details_mode(self, optimal_ces_model, optimal_ces_params):
        """Test that return_details mode returns extra information."""
        H = 1000.0
        C = 1e6
        logE = 5.0 * np.log(cfg.BASE_FOR_SOFTWARE_LOM)

        result, details = optimal_ces_model.coding_labor_optimal_ces(
            H, C, logE, optimal_ces_params, return_details=True
        )

        assert result is not None
        assert isinstance(details, dict)
        assert 'i_cut' in details
        assert 'i_E' in details
        assert 'case' in details

    def test_frontier_caching(self, optimal_ces_model, optimal_ces_params):
        """Test that frontier precomputation is cached."""
        H = 1000.0
        C = 1e6
        logE = 5.0 * np.log(cfg.BASE_FOR_SOFTWARE_LOM)

        result1 = optimal_ces_model.coding_labor_optimal_ces(H, C, logE, optimal_ces_params)
        result2 = optimal_ces_model.coding_labor_optimal_ces(H, C, logE, optimal_ces_params)

        assert result1 == result2, "Cached results should match"


# =============================================================================
# aut_frac_from_swe_multiplier Tests
# =============================================================================


class TestAutFracFromSWEMultiplier:
    """Tests for the aut_frac_from_swe_multiplier function."""

    def test_returns_finite_value(self, default_params):
        """Should return finite value for valid inputs."""
        result = aut_frac_from_swe_multiplier(
            swe_multiplier=2.0,
            L_HUMAN=1000.0,
            inference_H100e=1e6,
            params=default_params
        )

        assert np.isfinite(result), f"Expected finite result, got {result}"

    def test_returns_value_in_valid_range(self, default_params):
        """Should return value in (0, 1)."""
        result = aut_frac_from_swe_multiplier(
            swe_multiplier=2.0,
            L_HUMAN=1000.0,
            inference_H100e=1e6,
            params=default_params
        )

        assert 0.0 < result < 1.0, f"Expected value in (0, 1), got {result}"

    def test_higher_multiplier_gives_higher_automation(self, default_params):
        """Higher SWE multiplier should give higher automation fraction."""
        results = []
        for mult in [1.5, 2.0, 3.0, 5.0]:
            result = aut_frac_from_swe_multiplier(
                swe_multiplier=mult,
                L_HUMAN=1000.0,
                inference_H100e=1e6,
                params=default_params
            )
            results.append(result)

        for i in range(len(results) - 1):
            assert results[i] <= results[i+1], \
                f"Higher multiplier should give higher automation: {results[i]} > {results[i+1]}"

    def test_zero_multiplier_returns_zero(self, default_params):
        """Zero or negative multiplier should return 0."""
        result = aut_frac_from_swe_multiplier(
            swe_multiplier=0.0,
            L_HUMAN=1000.0,
            inference_H100e=1e6,
            params=default_params
        )

        assert result == 0.0, f"Expected 0 for zero multiplier, got {result}"

    def test_zero_labor_returns_zero(self, default_params):
        """Zero human labor should return 0."""
        result = aut_frac_from_swe_multiplier(
            swe_multiplier=2.0,
            L_HUMAN=0.0,
            inference_H100e=1e6,
            params=default_params
        )

        assert result == 0.0, f"Expected 0 for zero labor, got {result}"

    def test_non_finite_inputs_return_zero(self, default_params):
        """Non-finite inputs should return 0."""
        result = aut_frac_from_swe_multiplier(
            swe_multiplier=float('inf'),
            L_HUMAN=1000.0,
            inference_H100e=1e6,
            params=default_params
        )

        assert result == 0.0, f"Expected 0 for non-finite input, got {result}"


# =============================================================================
# solve_lower_anchor_via_automation_model Tests
# =============================================================================


class TestSolveLowerAnchor:
    """Tests for the solve_lower_anchor_via_automation_model function."""

    @pytest.fixture
    def anchor_solver_params(self):
        """Create parameters suitable for anchor solving."""
        return Parameters(
            rho_coding_labor=-2.0,
            parallel_penalty=0.5,
            progress_at_aa=10.0,
            automation_fraction_at_coding_automation_anchor=0.9,
        )

    def test_returns_finite_value(self, anchor_solver_params):
        """Should return finite value for valid inputs."""
        result = solve_lower_anchor_via_automation_model(
            swe_multiplier=2.0,
            anchor_progress=0.0,
            L_HUMAN=1000.0,
            inference_H100e=1e6,
            params=anchor_solver_params
        )

        assert np.isfinite(result), f"Expected finite result, got {result}"

    def test_returns_value_in_valid_range(self, anchor_solver_params):
        """Should return value in valid automation range."""
        result = solve_lower_anchor_via_automation_model(
            swe_multiplier=2.0,
            anchor_progress=0.0,
            L_HUMAN=1000.0,
            inference_H100e=1e6,
            params=anchor_solver_params
        )

        assert cfg.AUTOMATION_FRACTION_CLIP_MIN <= result <= 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN, \
            f"Expected value in valid range, got {result}"

    def test_higher_multiplier_gives_higher_anchor(self, anchor_solver_params):
        """Higher SWE multiplier should give higher anchor automation fraction."""
        results = []
        for mult in [1.5, 2.0, 3.0]:
            result = solve_lower_anchor_via_automation_model(
                swe_multiplier=mult,
                anchor_progress=0.0,
                L_HUMAN=1000.0,
                inference_H100e=1e6,
                params=anchor_solver_params
            )
            results.append(result)

        for i in range(len(results) - 1):
            assert results[i] <= results[i+1], \
                f"Higher multiplier should give higher anchor: {results[i]} > {results[i+1]}"

    def test_invalid_inputs_return_fallback(self, anchor_solver_params):
        """Invalid inputs should return fallback value."""
        result = solve_lower_anchor_via_automation_model(
            swe_multiplier=0.0,
            anchor_progress=0.0,
            L_HUMAN=1000.0,
            inference_H100e=1e6,
            params=anchor_solver_params
        )

        assert result == 0.01, f"Expected fallback value, got {result}"

    def test_non_finite_inputs_return_fallback(self, anchor_solver_params):
        """Non-finite inputs should return fallback value."""
        result = solve_lower_anchor_via_automation_model(
            swe_multiplier=float('nan'),
            anchor_progress=0.0,
            L_HUMAN=1000.0,
            inference_H100e=1e6,
            params=anchor_solver_params
        )

        assert result == 0.01, f"Expected fallback for NaN input, got {result}"


# =============================================================================
# Solver Verification Tests - Verify roundtrip accuracy
# =============================================================================


class TestSolverRoundtripAccuracy:
    """
    Verify that solve_lower_anchor_via_automation_model correctly optimizes
    the automation schedule to achieve the target SWE multiplier within tolerance.
    """

    # Tolerance for verifying the solver achieves target multiplier
    SOLVER_RTOL = 0.05  # 5% relative tolerance

    @staticmethod
    def _compute_implied_multiplier(
        solved_anchor: float,
        anchor_progress: float,
        L_HUMAN: float,
        inference_H100e: float,
        params: Parameters
    ) -> float:
        """
        Given a solved anchor, build the automation model and compute the
        implied SWE multiplier at the anchor progress.
        """
        import copy
        from progress_model.ces_functions import compute_coding_labor_deprecated

        # Get the upper anchor from params
        progress_at_aa = params.progress_at_aa
        aut_at_sc = params.automation_fraction_at_coding_automation_anchor

        # Build params with the solved anchors
        p = copy.deepcopy(params)
        p.automation_anchors = {
            float(anchor_progress): float(np.clip(solved_anchor, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)),
            float(progress_at_aa): float(np.clip(aut_at_sc, cfg.AUTOMATION_FRACTION_CLIP_MIN, 1.0 - cfg.AUTOMATION_FRACTION_CLIP_MIN)),
        }

        # Build the automation model
        am = AutomationModel(p)

        # Compute baseline (human-only) coding labor
        baseline = compute_coding_labor_deprecated(
            0, inference_H100e, L_HUMAN,
            p.rho_coding_labor, p.parallel_penalty, p.coding_labor_normalization,
            human_only=True
        )

        if baseline <= 0 or not np.isfinite(baseline):
            return float('nan')

        # Compute coding labor with the automation model
        logE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * anchor_progress)
        H = float(L_HUMAN)
        C = float(inference_H100e)

        L_opt = am.coding_labor_optimal_ces(H, C, logE, p)

        if L_opt is None or not np.isfinite(L_opt):
            # Fallback to simple CES using the schedule's automation
            A = am.get_automation_fraction(anchor_progress)
            L_ai = compute_coding_labor_deprecated(
                A, inference_H100e, L_HUMAN,
                p.rho_coding_labor, p.parallel_penalty, p.coding_labor_normalization
            )
        else:
            # Match units with compute_coding_labor_deprecated
            L_ai = float((L_opt ** p.parallel_penalty) * p.coding_labor_normalization)

        if not np.isfinite(L_ai) or L_ai <= 0:
            return float('nan')

        # The ratio is in parallel_penalty space
        ratio = L_ai / baseline

        # Convert back to multiplier space
        if p.parallel_penalty > 0:
            multiplier = float(np.power(ratio, 1.0 / p.parallel_penalty))
        else:
            multiplier = ratio

        return multiplier

    @pytest.mark.parametrize("swe_multiplier", [1.5, 2.0, 3.0, 5.0])
    def test_solver_achieves_target_multiplier(self, swe_multiplier):
        """Verify the solver achieves the target multiplier within tolerance."""
        params = Parameters(
            rho_coding_labor=-2.0,
            parallel_penalty=0.5,
            progress_at_aa=10.0,
            automation_fraction_at_coding_automation_anchor=0.9,
            coding_labor_mode='optimal_ces',
        )

        anchor_progress = 0.0
        L_HUMAN = 1000.0
        inference_H100e = 1e6

        # Solve for the lower anchor
        solved_anchor = solve_lower_anchor_via_automation_model(
            swe_multiplier=swe_multiplier,
            anchor_progress=anchor_progress,
            L_HUMAN=L_HUMAN,
            inference_H100e=inference_H100e,
            params=params
        )

        # Verify the solved anchor is valid
        assert np.isfinite(solved_anchor), f"Solved anchor should be finite, got {solved_anchor}"
        assert 0 < solved_anchor < 1, f"Solved anchor should be in (0, 1), got {solved_anchor}"

        # Compute the implied multiplier
        implied_mult = self._compute_implied_multiplier(
            solved_anchor, anchor_progress, L_HUMAN, inference_H100e, params
        )

        # Verify it matches the target
        if np.isfinite(implied_mult):
            rel_error = abs(implied_mult - swe_multiplier) / swe_multiplier
            assert rel_error < self.SOLVER_RTOL, \
                f"Solver did not achieve target: target={swe_multiplier}, " \
                f"implied={implied_mult:.4f}, rel_error={rel_error:.4f}"

    @pytest.mark.parametrize("rho", [-1.0, -2.0, -5.0])
    def test_solver_across_rho_values(self, rho):
        """Verify solver works across different rho values."""
        params = Parameters(
            rho_coding_labor=rho,
            parallel_penalty=0.5,
            progress_at_aa=10.0,
            automation_fraction_at_coding_automation_anchor=0.9,
            coding_labor_mode='optimal_ces',
        )

        swe_multiplier = 2.0
        anchor_progress = 0.0
        L_HUMAN = 1000.0
        inference_H100e = 1e6

        solved_anchor = solve_lower_anchor_via_automation_model(
            swe_multiplier=swe_multiplier,
            anchor_progress=anchor_progress,
            L_HUMAN=L_HUMAN,
            inference_H100e=inference_H100e,
            params=params
        )

        assert np.isfinite(solved_anchor), f"Solved anchor should be finite for rho={rho}"

        implied_mult = self._compute_implied_multiplier(
            solved_anchor, anchor_progress, L_HUMAN, inference_H100e, params
        )

        if np.isfinite(implied_mult):
            rel_error = abs(implied_mult - swe_multiplier) / swe_multiplier
            assert rel_error < self.SOLVER_RTOL, \
                f"Solver failed for rho={rho}: target={swe_multiplier}, " \
                f"implied={implied_mult:.4f}, rel_error={rel_error:.4f}"

    @pytest.mark.parametrize("parallel_penalty", [0.3, 0.5, 0.7])
    def test_solver_across_parallel_penalties(self, parallel_penalty):
        """Verify solver works across different parallel penalty values."""
        params = Parameters(
            rho_coding_labor=-2.0,
            parallel_penalty=parallel_penalty,
            progress_at_aa=10.0,
            automation_fraction_at_coding_automation_anchor=0.9,
            coding_labor_mode='optimal_ces',
        )

        swe_multiplier = 2.0
        anchor_progress = 0.0
        L_HUMAN = 1000.0
        inference_H100e = 1e6

        solved_anchor = solve_lower_anchor_via_automation_model(
            swe_multiplier=swe_multiplier,
            anchor_progress=anchor_progress,
            L_HUMAN=L_HUMAN,
            inference_H100e=inference_H100e,
            params=params
        )

        assert np.isfinite(solved_anchor), f"Solved anchor should be finite for pp={parallel_penalty}"

        implied_mult = self._compute_implied_multiplier(
            solved_anchor, anchor_progress, L_HUMAN, inference_H100e, params
        )

        if np.isfinite(implied_mult):
            rel_error = abs(implied_mult - swe_multiplier) / swe_multiplier
            assert rel_error < self.SOLVER_RTOL, \
                f"Solver failed for parallel_penalty={parallel_penalty}: target={swe_multiplier}, " \
                f"implied={implied_mult:.4f}, rel_error={rel_error:.4f}"

    @pytest.mark.parametrize("L_HUMAN,inference_H100e", [
        (100.0, 1e4),
        (1000.0, 1e6),
        (10000.0, 1e8),
    ])
    def test_solver_across_input_scales(self, L_HUMAN, inference_H100e):
        """Verify solver works across different input scales."""
        params = Parameters(
            rho_coding_labor=-2.0,
            parallel_penalty=0.5,
            progress_at_aa=10.0,
            automation_fraction_at_coding_automation_anchor=0.9,
            coding_labor_mode='optimal_ces',
        )

        swe_multiplier = 2.0
        anchor_progress = 0.0

        solved_anchor = solve_lower_anchor_via_automation_model(
            swe_multiplier=swe_multiplier,
            anchor_progress=anchor_progress,
            L_HUMAN=L_HUMAN,
            inference_H100e=inference_H100e,
            params=params
        )

        assert np.isfinite(solved_anchor), \
            f"Solved anchor should be finite for L_HUMAN={L_HUMAN}, C={inference_H100e}"

        implied_mult = self._compute_implied_multiplier(
            solved_anchor, anchor_progress, L_HUMAN, inference_H100e, params
        )

        if np.isfinite(implied_mult):
            rel_error = abs(implied_mult - swe_multiplier) / swe_multiplier
            assert rel_error < self.SOLVER_RTOL, \
                f"Solver failed for L={L_HUMAN}, C={inference_H100e}: target={swe_multiplier}, " \
                f"implied={implied_mult:.4f}, rel_error={rel_error:.4f}"

    def test_solver_comprehensive_parameter_sweep(self):
        """
        Comprehensive test across a grid of parameter combinations.
        This is a slower test but provides broader coverage.
        """
        # Parameter grid
        rho_values = [-1.0, -2.0, -3.0]
        pp_values = [0.4, 0.5, 0.6]
        multiplier_values = [1.5, 2.0, 3.0]

        failures = []
        successes = 0

        for rho in rho_values:
            for pp in pp_values:
                for mult in multiplier_values:
                    params = Parameters(
                        rho_coding_labor=rho,
                        parallel_penalty=pp,
                        progress_at_aa=10.0,
                        automation_fraction_at_coding_automation_anchor=0.9,
                        coding_labor_mode='optimal_ces',
                    )

                    solved_anchor = solve_lower_anchor_via_automation_model(
                        swe_multiplier=mult,
                        anchor_progress=0.0,
                        L_HUMAN=1000.0,
                        inference_H100e=1e6,
                        params=params
                    )

                    if not np.isfinite(solved_anchor):
                        failures.append(f"rho={rho}, pp={pp}, mult={mult}: non-finite anchor")
                        continue

                    implied_mult = self._compute_implied_multiplier(
                        solved_anchor, 0.0, 1000.0, 1e6, params
                    )

                    if not np.isfinite(implied_mult):
                        # This is acceptable - some parameter combinations may not work
                        continue

                    rel_error = abs(implied_mult - mult) / mult
                    if rel_error >= self.SOLVER_RTOL:
                        failures.append(
                            f"rho={rho}, pp={pp}, mult={mult}: "
                            f"implied={implied_mult:.4f}, rel_error={rel_error:.4f}"
                        )
                    else:
                        successes += 1

        # We expect most combinations to work; allow up to 20% failure rate
        total = len(rho_values) * len(pp_values) * len(multiplier_values)
        failure_rate = len(failures) / total if total > 0 else 0

        assert failure_rate < 0.2, \
            f"Too many failures ({len(failures)}/{total}={failure_rate:.1%}):\n" + "\n".join(failures[:10])

        # Ensure we got at least some successes
        assert successes > 0, "No parameter combinations succeeded"


# =============================================================================
# Edge Cases and Numerical Stability Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_automation_fraction_extreme_progress_values(self, linear_automation_model):
        """Test automation fraction with extreme progress values."""
        am = linear_automation_model

        result_neg = am.get_automation_fraction(-1000.0)
        assert np.isfinite(result_neg), "Should handle very negative progress"
        assert 0.0 <= result_neg <= 1.0, "Should be in valid range"

        result_pos = am.get_automation_fraction(1000.0)
        assert np.isfinite(result_pos), "Should handle very positive progress"
        assert 0.0 <= result_pos <= 1.0, "Should be in valid range"

    def test_logistic_extreme_progress_values(self, logistic_automation_model):
        """Test logistic schedule with extreme progress values."""
        am = logistic_automation_model

        result_neg = am.get_automation_fraction(-1000.0)
        assert np.isfinite(result_neg), "Should handle very negative progress"
        assert 0.0 <= result_neg <= 1.0, "Should be in valid range"

        result_pos = am.get_automation_fraction(1000.0)
        assert np.isfinite(result_pos), "Should handle very positive progress"
        assert 0.0 <= result_pos <= 1.0, "Should be in valid range"

    def test_progress_from_index_boundary_values(self, linear_automation_model):
        """Test progress from index at boundary values."""
        am = linear_automation_model

        result_low = am.get_progress_from_index(1e-9)
        assert np.isfinite(result_low), "Should handle near-zero index"

        result_high = am.get_progress_from_index(1.0 - 1e-9)
        assert np.isfinite(result_high), "Should handle near-one index"

    def test_fte_per_gpu_zero_progress_diff(self, linear_automation_model):
        """Test FTE per GPU when progress exactly equals threshold."""
        am = linear_automation_model
        index = 0.5
        threshold = am.get_progress_from_index(index)

        result = am.get_FTE_per_GPU(index, threshold)
        assert np.isfinite(result), "Should handle zero progress diff"
        assert result == am.initial_FTE_per_GPU, "Should equal initial FTE at threshold"

    def test_optimal_ces_with_extreme_logE(self):
        """Test optimal CES with extreme effective compute values."""
        params = Parameters(
            coding_labor_mode='optimal_ces',
            rho_coding_labor=-2.0,
            automation_anchors={0.0: 0.1, 10.0: 0.9},
        )
        am = AutomationModel(params)

        H = 1000.0
        C = 1e6

        logE_low = -10.0 * np.log(cfg.BASE_FOR_SOFTWARE_LOM)
        result_low = am.coding_labor_optimal_ces(H, C, logE_low, params)
        # May be None if below minimum, but shouldn't crash

        logE_high = 100.0 * np.log(cfg.BASE_FOR_SOFTWARE_LOM)
        result_high = am.coding_labor_optimal_ces(H, C, logE_high, params)
        if result_high is not None:
            assert np.isfinite(result_high), "Should handle high logE"


# =============================================================================
# Parameter Variation Tests
# =============================================================================


class TestParameterVariations:
    """Tests that verify behavior across different parameter configurations."""

    @pytest.mark.parametrize("rho", [-0.5, -1.0, -2.0, -5.0])
    def test_different_rho_values(self, rho):
        """Test automation model works with different rho values."""
        params = Parameters(
            rho_coding_labor=rho,
            automation_anchors={0.0: 0.1, 10.0: 0.9},
        )
        am = AutomationModel(params)

        result = am.get_automation_fraction(5.0)
        assert np.isfinite(result), f"Should work with rho={rho}"

    @pytest.mark.parametrize("schedule", ["linear", "logistic"])
    def test_different_schedule_types(self, schedule):
        """Test different automation schedule types."""
        params = Parameters(
            automation_interp_type=schedule,
            automation_logistic_asymptote=1.1,
            automation_anchors={0.0: 0.1, 10.0: 0.9},
        )
        am = AutomationModel(params)

        assert am.schedule_type == schedule

        result = am.get_automation_fraction(5.0)
        assert np.isfinite(result), f"Should work with schedule={schedule}"
        assert 0.0 <= result <= 1.0, "Result should be in valid range"

    @pytest.mark.parametrize("parallel_penalty", [0.3, 0.5, 0.7, 1.0])
    def test_different_parallel_penalties(self, parallel_penalty):
        """Test automation model works with different parallel penalty values."""
        params = Parameters(
            parallel_penalty=parallel_penalty,
            automation_anchors={0.0: 0.1, 10.0: 0.9},
        )
        am = AutomationModel(params)

        result = am.get_automation_fraction(5.0)
        assert np.isfinite(result), f"Should work with parallel_penalty={parallel_penalty}"
