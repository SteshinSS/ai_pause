#!/usr/bin/env python3
"""
Generate golden data files for regression testing.

Run this script to create baseline outputs that tests will compare against.
Only run this when you intentionally want to update the expected behavior.

Usage:
    python tests/generate_golden_data.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from progress_model import Parameters, ProgressModel, _ces_function, compute_coding_labor_deprecated
from utils import (
    load_time_series_data,
    save_golden_data,
    DEFAULT_TIME_RANGE,
    get_input_parameters,
    parameters_from_dict,
)
from conftest import (
    REPRESENTATIVE_PARAMETER_SETS,
    compute_trajectory_for_params,
    save_scenario_golden_data,
)


def generate_trajectory_golden_data():
    """Generate golden data for full trajectory test."""
    print("Generating trajectory golden data...")

    time_series_data = load_time_series_data()

    # Capture input parameters (pre-transform defaults at generation time)
    input_params = get_input_parameters()

    # Build the model from the SAME dict that gets recorded as _generation_parameters.
    # The golden tests reconstruct via parameters_from_dict(_generation_parameters), so
    # generating from a bare Parameters() would pin arrays that the tests can never
    # reproduce wherever a dataclass field default diverges from DEFAULT_PARAMETERS.
    params = parameters_from_dict(input_params)
    model = ProgressModel(params, time_series_data)

    # Run trajectory computation with DEFAULT_TIME_RANGE for consistency with scenario tests
    model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

    # Extract key results to save
    results = model.results
    trajectory_data = {
        # Store input parameters used to generate this golden data
        '_generation_parameters': input_params,

        # Time and progress arrays
        'times': results.get('times', []),
        'progress': results.get('progress', []),
        'research_stock': results.get('research_stock', []),

        # Derived metrics (note: model uses singular key names)
        'progress_rates': results.get('progress_rates', []),   # d(progress)/dt (true), OOMs/yr
        'etsp_rates': results.get('etsp_rates', []),           # d(ETSP)/dt = sw + TSP, OOMs/yr
        'research_efforts': results.get('research_efforts', []),
        'automation_fractions': results.get('automation_fraction', []),  # singular in model
        'ai_research_tastes': results.get('ai_research_taste', []),  # singular in model
        'ai_research_taste_sds': results.get('ai_research_taste_sd', []),  # singular in model
        'aggregate_research_tastes': results.get('aggregate_research_taste', []),  # singular in model
        'coding_labors': results.get('coding_labors', []),
        'serial_coding_labors': results.get('serial_coding_labors', []),
        'software_progress_rates': results.get('software_progress_rates', []),
        'software_efficiency': results.get('software_efficiency', []),
        'effective_compute': results.get('effective_compute', []),
        'training_compute': results.get('training_compute', []),
        'horizon_lengths': results.get('horizon_lengths', []),
        'experiment_capacity': results.get('experiment_capacity', []),

        # Human-only metrics
        'human_only_progress_rates': results.get('human_only_progress_rates', []),
        'human_only_research_efforts': results.get('human_only_research_efforts', []),

        # Multipliers
        'ai_coding_labor_multipliers': results.get('ai_coding_labor_multipliers', []),
        'ai_sw_progress_mult_ref_present_day': results.get('ai_sw_progress_mult_ref_present_day', []),
    }

    save_golden_data(trajectory_data, 'default_trajectory.json')


def generate_milestones_golden_data():
    """Generate golden data for milestone tests."""
    print("Generating milestones golden data...")

    time_series_data = load_time_series_data()

    # Capture input parameters (pre-transform defaults at generation time)
    input_params = get_input_parameters()

    # Same construction path as the tests -- see generate_trajectory_golden_data().
    params = parameters_from_dict(input_params)
    model = ProgressModel(params, time_series_data)

    # Run trajectory computation with DEFAULT_TIME_RANGE for consistency with scenario tests
    model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

    results = model.results
    progress_array = results.get('progress', [])
    milestones_data = {
        # Store input parameters used to generate this golden data
        '_generation_parameters': input_params,

        # Key milestone times
        'aa_time': results.get('aa_time'),
        'ai2027_sc_time': results.get('ai2027_sc_time'),

        # Progress values
        'progress_at_aa': params.progress_at_aa,
        'final_progress': float(progress_array[-1]) if len(progress_array) > 0 else None,

        # Anchor stats
        'anchor_progress_rate': results.get('anchor_progress_rate'),

        # SC multiplier
        'sc_sw_multiplier': getattr(model, 'sc_sw_multiplier', None),

        # Initial conditions used
        'initial_progress': results.get('initial_progress'),

        # Horizon trajectory parameters (if available)
        'horizon_uses_shifted_form': getattr(model, '_horizon_uses_shifted_form', False),
    }

    save_golden_data(milestones_data, 'default_milestones.json')


def generate_ces_function_golden_data():
    """Generate golden data for CES function tests."""
    print("Generating CES function golden data...")

    test_cases = []

    # Test various rho values
    rho_values = [1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -5.0, -10.0, -50.0, -100.0]
    x1_values = [1.0, 2.0, 10.0]
    x2_values = [1.0, 3.0, 5.0]
    w1_values = [0.3, 0.5, 0.7]

    for rho in rho_values:
        for x1 in x1_values:
            for x2 in x2_values:
                for w1 in w1_values:
                    result = _ces_function(x1, x2, w1, rho)
                    test_cases.append({
                        'x1': x1,
                        'x2': x2,
                        'w1': w1,
                        'rho': rho,
                        'result': result
                    })

    # Edge cases
    edge_cases = [
        {'x1': 0.0, 'x2': 1.0, 'w1': 0.5, 'rho': -2.0},
        {'x1': 1.0, 'x2': 0.0, 'w1': 0.5, 'rho': -2.0},
        {'x1': 0.0, 'x2': 0.0, 'w1': 0.5, 'rho': -2.0},
        {'x1': 1e-10, 'x2': 1e10, 'w1': 0.5, 'rho': -2.0},
        {'x1': 100.0, 'x2': 100.0, 'w1': 0.0, 'rho': -2.0},
        {'x1': 100.0, 'x2': 100.0, 'w1': 1.0, 'rho': -2.0},
    ]
    for case in edge_cases:
        result = _ces_function(case['x1'], case['x2'], case['w1'], case['rho'])
        case['result'] = result
        test_cases.append(case)

    save_golden_data({'test_cases': test_cases}, 'ces_function_golden.json')


def generate_coding_labor_golden_data():
    """Generate golden data for compute_coding_labor_deprecated tests."""
    print("Generating coding labor golden data...")

    test_cases = []

    # Various parameter combinations
    automation_fractions = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    inference_computes = [100.0, 1000.0, 10000.0]
    l_humans = [100.0, 500.0, 1000.0]
    rhos = [-0.5, -1.0, -2.0, -5.0]
    parallel_penalties = [0.3, 0.5, 0.7, 1.0]

    for af in automation_fractions:
        for ic in inference_computes:
            for lh in l_humans:
                for rho in rhos:
                    for pp in parallel_penalties:
                        result = compute_coding_labor_deprecated(af, ic, lh, rho, pp, 1.0, False)
                        test_cases.append({
                            'automation_fraction': af,
                            'inference_compute': ic,
                            'L_HUMAN': lh,
                            'rho': rho,
                            'parallel_penalty': pp,
                            'cognitive_normalization': 1.0,
                            'human_only': False,
                            'result': result
                        })

    # Human-only cases
    for lh in l_humans:
        for pp in parallel_penalties:
            result = compute_coding_labor_deprecated(0.5, 1000.0, lh, -2.0, pp, 1.0, True)
            test_cases.append({
                'automation_fraction': 0.5,
                'inference_compute': 1000.0,
                'L_HUMAN': lh,
                'rho': -2.0,
                'parallel_penalty': pp,
                'cognitive_normalization': 1.0,
                'human_only': True,
                'result': result
            })

    save_golden_data({'test_cases': test_cases}, 'coding_labor_golden.json')


def generate_taste_distribution_golden_data():
    """Generate golden data for TasteDistribution tests."""
    print("Generating taste distribution golden data...")

    time_series_data = load_time_series_data()

    # Capture input parameters (pre-transform defaults at generation time)
    input_params = get_input_parameters()

    params = Parameters()

    # Get the taste distribution from params
    taste_dist = params.taste_distribution

    # Test get_taste_at_quantile
    quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    taste_at_quantile = {str(q): taste_dist.get_taste_at_quantile(q) for q in quantiles}

    # Test get_taste_at_sd
    sds = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
    taste_at_sd = {str(sd): taste_dist.get_taste_at_sd(sd) for sd in sds}

    # Test get_mean_with_floor
    floor_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    mean_with_floor = {str(f): taste_dist.get_mean_with_floor(f) for f in floor_values}

    # Distribution properties
    properties = {
        'mu': taste_dist.mu,
        'sigma': taste_dist.sigma,
        'taste_limit': taste_dist.taste_limit,
        'median': taste_dist.get_median(),
        'mean': taste_dist.get_mean(),
        'top_percentile': taste_dist.top_percentile,
        'median_to_top_gap': taste_dist.median_to_top_gap,
    }

    golden_data = {
        # Store input parameters used to generate this golden data
        '_generation_parameters': input_params,
        'taste_at_quantile': taste_at_quantile,
        'taste_at_sd': taste_at_sd,
        'mean_with_floor': mean_with_floor,
        'properties': properties,
    }

    save_golden_data(golden_data, 'taste_distribution_golden.json')


def generate_scenario_golden_data():
    """Generate golden data for all representative parameter set scenarios."""
    print("Generating scenario golden data...")

    time_series_data = load_time_series_data()

    for scenario in REPRESENTATIVE_PARAMETER_SETS:
        scenario_name = scenario['name']
        print(f"  - {scenario_name}")

        # Get scenario overrides (everything except 'name')
        scenario_kwargs = {k: v for k, v in scenario.items() if k != 'name'}

        # Capture input parameters: defaults merged with scenario overrides
        # This is the pre-transform input that would be passed to Parameters()
        input_params = get_input_parameters(scenario_kwargs)

        # Run model computation from the recorded dict, not from the bare overrides,
        # so the golden arrays match what parameters_from_dict(_generation_parameters)
        # reproduces in the tests.
        params = parameters_from_dict(input_params)
        model = ProgressModel(params, time_series_data)
        model.compute_progress_trajectory(DEFAULT_TIME_RANGE, initial_progress=0.0)

        save_scenario_golden_data(scenario_name, model, input_params)

    print(f"  Generated {len(REPRESENTATIVE_PARAMETER_SETS)} scenario files")


def main():
    """Generate all golden data files."""
    print("=" * 60)
    print("Generating golden data for regression tests")
    print("=" * 60)

    generate_ces_function_golden_data()
    generate_coding_labor_golden_data()
    generate_taste_distribution_golden_data()
    generate_trajectory_golden_data()
    generate_milestones_golden_data()
    generate_scenario_golden_data()

    print("=" * 60)
    print("Golden data generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
