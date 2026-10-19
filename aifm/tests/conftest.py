"""
Shared pytest fixtures for model behavior tests.
"""

import pytest
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from progress_model import TimeSeriesData, Parameters, ProgressModel
from utils import (
    load_time_series_data,
    save_golden_data,
    load_golden_data,
    convert_for_json,
    GOLDEN_DATA_DIR,
    DEFAULT_TIME_RANGE,
    serialize_parameters,
    parameters_from_dict,
)

# Standard tolerance levels for assertions
# NOTE: The ODE solver uses rtol=1e-3 to 1e-4, so trajectory-derived values
# cannot be tested to higher precision than that.
STRICT_RTOL = 1e-10   # For exact comparisons (e.g., deterministic time arrays)
NORMAL_RTOL = 1e-8    # For typical numerical comparisons (non-ODE derived)
OPTIMIZATION_RTOL = 1e-6  # For well-converged optimization constraints
INVERSE_FUNCTION_RTOL = 1e-5  # For inverse function tests with non-default params

# ODE/trajectory tolerances - these must match or be looser than the solver's tolerances
# The solver uses rtol=1e-3 to 1e-4, so we use 1e-4 for trajectory values
TRAJECTORY_RTOL = 1e-3  # 0.1% tolerance - allows fast vectorized human-only integration
MILESTONE_RTOL = 1e-3   # For milestone times derived from trajectory integration

LOOSE_RTOL = 1e-3     # For optimization constraints that may not converge exactly
VERY_LOOSE_RTOL = 1e-2  # For edge cases near parameter bounds
EDGE_CASE_RTOL = 0.05  # For edge cases near parameter bounds where accuracy may degrade
EXTREME_PARAMS_RTOL = 0.1  # For extreme parameter combinations at bounds


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help="Regenerate golden data files and exit without running comparisons"
    )


def pytest_configure(config):
    """Handle golden data regeneration before tests run."""
    # Avoid duplicate regeneration in xdist workers
    if getattr(config, "workerinput", None) is not None:
        return

    if config.getoption("--regen-golden"):
        from generate_golden_data import main as generate_all_golden_data

        generate_all_golden_data()
        pytest.exit(
            "Regenerated golden data; rerun pytest without --regen-golden to execute tests.",
            returncode=0,
        )


@pytest.fixture
def regen_golden(request):
    """Return True if --regen-golden flag was passed."""
    return request.config.getoption("--regen-golden")


@pytest.fixture
def default_time_series_data():
    """Load default inputs/input_data.csv into TimeSeriesData."""
    return load_time_series_data()


@pytest.fixture
def default_params():
    """Create Parameters with defaults from progress_model/config.py."""
    return Parameters()


@pytest.fixture
def default_model(default_params, default_time_series_data):
    """Create ProgressModel with default parameters."""
    return ProgressModel(default_params, default_time_series_data)


@pytest.fixture
def golden_trajectory():
    """Load golden trajectory data."""
    data = load_golden_data('default_trajectory.json')
    if data is None:
        pytest.fail(
            "Golden data file missing: default_trajectory.json. "
            "Run `pytest --regen-golden` or `python tests/generate_golden_data.py` to regenerate."
        )
    return data


@pytest.fixture
def golden_trajectory_params(golden_trajectory):
    """Get Parameters object from golden trajectory data.

    This ensures tests use the same parameters that generated the golden data,
    isolating golden tests from changes to default parameter values.
    """
    params_dict = golden_trajectory.get('_generation_parameters')
    if params_dict is None:
        # Fallback for old golden data without stored parameters
        return Parameters()
    return parameters_from_dict(params_dict)


@pytest.fixture
def golden_milestones():
    """Load golden milestone data."""
    data = load_golden_data('default_milestones.json')
    if data is None:
        pytest.fail(
            "Golden data file missing: default_milestones.json. "
            "Run `pytest --regen-golden` or `python tests/generate_golden_data.py` to regenerate."
        )
    return data


def assert_arrays_close(actual, expected, rtol=1e-8, atol=0, msg=""):
    """Assert two arrays are close within tolerance."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        if msg:
            raise AssertionError(f"{msg}\n{e}") from None
        raise


def assert_scalar_close(actual, expected, rtol=1e-8, atol=0, msg=""):
    """Assert two scalars are close within tolerance."""
    if expected is None and actual is None:
        return
    if expected is None or actual is None:
        raise AssertionError(f"{msg}: expected {expected}, got {actual}")

    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{msg}: expected {expected}, got {actual}, "
            f"diff={abs(actual - expected)}, rel_diff={abs(actual - expected) / max(abs(expected), 1e-10)}"
        )


@pytest.fixture
def golden_taste_distribution(regen_golden):
    """Load golden taste distribution data."""
    if regen_golden:
        pytest.skip("Regenerating golden data")

    data = load_golden_data('taste_distribution_golden.json')
    if data is None:
        pytest.fail(
            "Golden data file missing: taste_distribution_golden.json. "
            "Run `pytest --regen-golden` or `python tests/generate_golden_data.py` to regenerate."
        )
    return data


@pytest.fixture
def make_taste_params():
    """Factory fixture for creating Parameters with custom taste distribution settings."""
    def _make(median_to_top=None, taste_limit=None, taste_smoothing=None):
        kwargs = {}
        if median_to_top is not None:
            kwargs['median_to_top_taste_multiplier'] = median_to_top
        if taste_limit is not None:
            kwargs['taste_limit'] = taste_limit
        if taste_smoothing is not None:
            kwargs['taste_limit_smoothing'] = taste_smoothing
        return Parameters(**kwargs)
    return _make


@pytest.fixture
def computed_model(golden_trajectory_params, default_time_series_data, regen_golden):
    """Return model with pre-computed trajectory using params from golden data.

    Uses parameters stored in golden data to ensure tests compare against
    the same model behavior, regardless of changes to default parameters.
    Uses DEFAULT_TIME_RANGE for consistency with scenario tests.
    """
    model = ProgressModel(golden_trajectory_params, default_time_series_data)
    model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
    return model


@pytest.fixture
def model_with_trajectory(default_model, default_time_series_data):
    """Return model with pre-computed trajectory (for property tests, doesn't skip on regen).

    Uses DEFAULT_TIME_RANGE for consistency with scenario tests.
    """
    default_model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
    return default_model


def assert_ratio_constraint(taste_dist, expected_ratio, rtol=LOOSE_RTOL, msg=""):
    """Assert the taste distribution satisfies the ratio constraint."""
    median_taste = taste_dist.get_median()
    top_taste = taste_dist.get_taste_at_quantile(taste_dist.top_percentile)
    actual_ratio = top_taste / median_taste

    if abs(actual_ratio - expected_ratio) / expected_ratio >= rtol:
        full_msg = f"Ratio constraint violated: actual={actual_ratio:.6f}, expected={expected_ratio:.6f}"
        if msg:
            full_msg = f"{msg}: {full_msg}"
        raise AssertionError(full_msg)


def assert_mean_constraint(taste_dist, rtol=LOOSE_RTOL, msg=""):
    """Assert the taste distribution satisfies the mean constraint."""
    actual_mean = taste_dist.get_mean()
    expected_mean = taste_dist.baseline_mean

    if abs(actual_mean - expected_mean) / expected_mean >= rtol:
        full_msg = f"Mean constraint violated: actual={actual_mean:.6f}, expected={expected_mean:.6f}"
        if msg:
            full_msg = f"{msg}: {full_msg}"
        raise AssertionError(full_msg)


def assert_monotonic(values, labels=None, strict=True, msg=""):
    """Assert values are monotonically increasing.

    Args:
        values: Sequence of values to check
        labels: Optional labels for error messages (e.g., quantile values)
        strict: If True, require strictly increasing (default). If False, allow equal values.
        msg: Optional message prefix for assertion errors
    """
    for i in range(len(values) - 1):
        if strict:
            ok = values[i] < values[i + 1]
        else:
            ok = values[i] <= values[i + 1]

        if not ok:
            if labels is not None:
                detail = f"values[{labels[i]}]={values[i]} {'>' if strict else '>='} values[{labels[i+1]}]={values[i+1]}"
            else:
                detail = f"values[{i}]={values[i]} {'>' if strict else '>='} values[{i+1}]={values[i+1]}"
            full_msg = f"Monotonicity violated: {detail}"
            if msg:
                full_msg = f"{msg}: {full_msg}"
            raise AssertionError(full_msg)


# =============================================================================
# Parameter Set Generation for Non-Default Tests
# =============================================================================
# These fixtures and utilities create parameter sets spanning the allowed ranges
# from progress_model.config.PARAMETER_BOUNDS and config/sampling_config.yaml

from progress_model import config as cfg

# Representative parameter sets based on sampling_config.yaml distributions
# Each dict represents a coherent scenario that the model should handle
REPRESENTATIVE_PARAMETER_SETS = [
    # Scenario 1: Fast doubling time, high growth factor (optimistic progress)
    {
        'name': 'fast_progress',
        'rho_coding_labor': -2,
        'present_doubling_time': 0.3,
        'doubling_difficulty_growth_factor': 0.85,
        'ai_research_taste_slope': 3.0,
        'ai_research_taste_at_coding_automation_anchor_sd': 0.5,
        'ac_time_horizon_minutes': 1.2e7,  # ~100 work years
        'horizon_extrapolation_type': 'decaying doubling time',
    },
    # Scenario 2: Slow doubling time, low growth factor (pessimistic progress)
    {
        'name': 'slow_progress',
        'rho_coding_labor': -1,
        'present_doubling_time': 0.7,
        'doubling_difficulty_growth_factor': 1.0,
        'ai_research_taste_slope': 1.0,
        'ai_research_taste_at_coding_automation_anchor_sd': -1.0,
        'ac_time_horizon_minutes': 1e9,  # ~8000 work years
        'horizon_extrapolation_type': 'decaying doubling time',
    },
    # Scenario 3: High complementarity (Leontief-ish) in coding labor
    {
        'name': 'high_complementarity',
        'rho_coding_labor': -5,
        'present_doubling_time': 0.5,
        'doubling_difficulty_growth_factor': 0.9,
        'ai_research_taste_slope': 2.0,
        'coding_automation_efficiency_slope': 2.0,
        'serial_uplift_at_reference_date': 1.4142135623730951,
    },
    # Scenario 4: Experiment capacity variations
    {
        'name': 'high_exp_capacity',
        'inf_compute_asymptote': 5000,
        'inf_labor_asymptote': 50,
        'inv_compute_anchor_exp_cap': 3.0,
        'parallel_penalty': 0.3,
    },
    # Scenario 5: Research taste variations
    {
        'name': 'high_taste_growth',
        'ai_research_taste_slope': 5.0,
        'ai_research_taste_at_coding_automation_anchor_sd': 2.0,
        'median_to_top_taste_multiplier': 5.0,
        'taste_limit': 10.0,
        'taste_limit_smoothing': 0.6,
    },
    # Scenario 6: Gap mode enabled
    {
        'name': 'with_gap',
        'include_gap': 'gap',
        'gap_years': 3.0,
        # NOTE: 'pre_gap_ac_time_horizon' removed — no longer a Parameters field after the
        # continuous-training port (it raised TypeError, blocking golden regen for this and
        # all later scenarios). Gap is now specified via include_gap + gap_years.
        'ac_time_horizon_minutes': 1.2e7,
    },
    # Scenario 7: Present day variation
    {
        'name': 'later_present_day',
        'present_day': 2026.5,
        'present_horizon': 50.0,
        'present_doubling_time': 0.4,
        'doubling_difficulty_growth_factor': 0.95,
    },
    # Scenario 8: High coding automation efficiency
    {
        'name': 'high_automation_efficiency',
        'coding_automation_efficiency_slope': 5.0,
        'max_serial_coding_labor_multiplier': 1e8,
        'serial_uplift_at_reference_date': 1.7320508075688772,
    },
    # Scenario 9: Extreme but valid parameter combinations
    {
        'name': 'boundary_values',
        'rho_coding_labor': -8,  # Near but not at boundary
        'present_doubling_time': 0.1,  # Fast doubling
        'doubling_difficulty_growth_factor': 0.7,  # Low growth factor
        'software_progress_rate_at_reference_year': 2.0,
    },
    # Scenario 10: Conservative scenario
    {
        'name': 'conservative',
        'rho_coding_labor': -1,  # Lower complementarity
        'present_doubling_time': 0.6,
        'doubling_difficulty_growth_factor': 1.01,  # Near maximum
        'ai_research_taste_slope': 0.8,
        'inf_compute_asymptote': 10,  # Low asymptote
        'inf_labor_asymptote': 5,
    },
]


@pytest.fixture
def make_params():
    """Factory fixture for creating Parameters with custom settings."""
    def _make(**kwargs):
        return Parameters(**kwargs)
    return _make


@pytest.fixture
def make_model(default_time_series_data):
    """Factory fixture for creating ProgressModel with custom parameters."""
    def _make(**param_kwargs):
        params = Parameters(**param_kwargs)
        return ProgressModel(params, default_time_series_data)
    return _make


@pytest.fixture
def representative_param_sets():
    """Return the list of representative parameter sets for testing."""
    return REPRESENTATIVE_PARAMETER_SETS


@pytest.fixture(params=REPRESENTATIVE_PARAMETER_SETS, ids=lambda p: p['name'])
def param_set_scenario(request):
    """Parametrized fixture that yields each representative parameter set."""
    return request.param


def compute_trajectory_for_params(param_dict, time_series_data, time_range=None, initial_progress=0.0):
    """Helper to compute a trajectory for a given parameter dictionary.

    Args:
        param_dict: Dictionary of parameter overrides (may include 'name' key which is ignored)
        time_series_data: TimeSeriesData object
        time_range: Optional [start, end] time range; defaults to full data range
        initial_progress: Initial progress value

    Returns:
        ProgressModel with computed trajectory
    """
    # Filter out non-parameter keys
    param_kwargs = {k: v for k, v in param_dict.items() if k != 'name'}
    params = Parameters(**param_kwargs)
    model = ProgressModel(params, time_series_data)

    if time_range is None:
        time_range = [float(time_series_data.time[0]), float(time_series_data.time[-1])]

    model.compute_progress_trajectory(time_range, initial_progress)
    return model


# =============================================================================
# Golden Data for Non-Default Parameter Sets
# =============================================================================

def get_scenario_golden_filename(scenario_name: str) -> str:
    """Get the golden data filename for a scenario."""
    return f"trajectory_{scenario_name}.json"


def load_scenario_golden_data(scenario_name: str):
    """Load golden data for a specific scenario.

    Returns None if the file doesn't exist.
    """
    return load_golden_data(get_scenario_golden_filename(scenario_name))


def save_scenario_golden_data(scenario_name: str, model, pre_compute_params: dict):
    """Save golden data for a specific scenario from a computed model.

    Args:
        scenario_name: Name of the scenario
        model: Computed ProgressModel with trajectory results
        pre_compute_params: Parameters serialized BEFORE model computation
            (captures the exact inputs including defaults at generation time)
    """
    data = {
        'scenario_name': scenario_name,
        # Store the complete parameter inputs (pre-normalization)
        '_generation_parameters': pre_compute_params,
        'times': model.results['times'],
        'progress': model.results['progress'],
        'research_stock': model.results['research_stock'],
        'automation_fraction': model.results['automation_fraction'],
        'ai_research_taste': model.results['ai_research_taste'],
        'aggregate_research_taste': model.results['aggregate_research_taste'],
        'coding_labors': model.results['coding_labors'],
        'software_progress_rates': model.results['software_progress_rates'],
        'effective_compute': model.results['effective_compute'],
        'horizon_lengths': model.results.get('horizon_lengths', []),
        # Include milestone times if available
        'aa_time': model.results.get('aa_time'),
        'sc_time': model.results.get('sc_time'),
        'sar_time': model.results.get('sar_time'),
    }

    # Include human-only trajectory anchor stats if available
    # These are the critical values computed from the human-only solver
    # that feed into the full trajectory (especially in non-SOS mode)
    if hasattr(model, 'human_only_results') and model.human_only_results is not None:
        human_only = model.human_only_results
        data['human_only_anchor_stats'] = human_only.get('anchor_stats', {})
        data['human_only_reference_sw_progress_rate'] = human_only.get('reference_sw_progress_rate')

    save_golden_data(data, get_scenario_golden_filename(scenario_name))
    return data


@pytest.fixture
def scenario_golden_data(param_set_scenario):
    """Load golden data for a parameter set scenario.

    Golden data is generated by running `pytest --regen-golden` or
    `python tests/generate_golden_data.py`.
    """
    scenario_name = param_set_scenario['name']

    data = load_scenario_golden_data(scenario_name)
    if data is None:
        pytest.fail(
            f"Golden data not found for scenario '{scenario_name}'. "
            "Run `pytest --regen-golden` or `python tests/generate_golden_data.py` to regenerate."
        )

    return data


# =============================================================================
# Cached Model Computation for Efficiency
# =============================================================================
# These fixtures cache computed models to avoid redundant trajectory calculations


# Module-scoped cache to store computed models by scenario name
_scenario_model_cache = {}


@pytest.fixture
def computed_scenario_model(param_set_scenario, scenario_golden_data, default_time_series_data):
    """Get or compute model for a scenario using inputs from golden data.

    Uses the full parameter set stored in golden data to ensure tests use
    the exact same inputs that generated the golden data. This isolates
    tests from changes to both scenario definitions and default parameters.

    Falls back to current scenario definition if golden data doesn't have
    stored parameters (for backwards compatibility).

    This fixture should be used by tests that check multiple properties of
    the same trajectory to avoid computing it multiple times.
    """
    scenario_name = param_set_scenario['name']

    if scenario_name not in _scenario_model_cache:
        # Use stored parameters from golden data if available
        stored_params = scenario_golden_data.get('_generation_parameters')
        if stored_params is not None:
            # Use the full parameter set from golden data
            params = parameters_from_dict(stored_params)
            model = ProgressModel(params, default_time_series_data)
            model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)
            _scenario_model_cache[scenario_name] = model
        else:
            # Fallback for old golden data without stored parameters
            _scenario_model_cache[scenario_name] = compute_trajectory_for_params(
                param_set_scenario,
                default_time_series_data,
                time_range=DEFAULT_TIME_RANGE,
            )

    return _scenario_model_cache[scenario_name]


def clear_scenario_model_cache():
    """Clear the scenario model cache (useful for testing the tests)."""
    _scenario_model_cache.clear()
