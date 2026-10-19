#!/usr/bin/env python3
"""
Progress Model Package

This package provides the core modeling infrastructure for AI progress trajectories
using nested CES production functions with feedback loops.

For backwards compatibility, all public symbols from the original progress_model.py
are re-exported from this package. The implementation is progressively being
modularized into separate submodules.

Submodules:
- utils: Utility functions (scalar coercion, Gauss-Hermite quadrature)
- types: Basic data types (TimeSeriesData, InitialConditions)
- _impl: Main implementation (TasteDistribution, Parameters, ProgressModel, etc.)

Future modularization will extract:
- taste_distribution: TasteDistribution class
- automation_model: AutomationModel class
- ces_functions: CES production functions
- research: Research effort and progress rate functions
- integration: ODE integration functions
- core: ProgressModel class
"""

# Re-export utilities
from .utils import (
    should_reraise,
    _coerce_float_scalar,
    _gauss_hermite_expectation,
)

# Re-export types
from .types import (
    TimeSeriesData,
    InitialConditions,
)

# Re-export CES functions from dedicated module
from .ces_functions import (
    _ces_function,
    compute_coding_labor_deprecated,
    compute_rho_from_asymptotes,
    compute_alpha_experiment_capacity_from_asymptotes,
    compute_experiment_compute_exponent_from_anchor,
    compute_exp_capacity_params_from_anchors,
)

# Re-export TasteDistribution from dedicated module
from .taste_distribution import (
    TasteDistribution,
    get_or_create_taste_distribution,
    _taste_distribution_cache,
    _TASTE_CACHE_MAX_SIZE,
)

# Re-export AutomationModel from dedicated module
from .automation_model import AutomationModel

# Re-export Parameters from dedicated module
from .parameters import Parameters

# Re-export progress rate functions from dedicated module
from .progress_rate import (
    compute_research_effort,
    compute_software_progress_rate,
    compute_overall_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    progress_rate_at_time,
)

# Re-export integration functions from dedicated module
from .integration import (
    _find_exponential_crossing_time,
    calculate_initial_research_stock,
    compute_initial_conditions,
    setup_model,
    integrate_progress,
)

# Re-export remaining symbols from _impl for backwards compatibility
from ._impl import (
    # Classes
    ProgressModel,

    # Caching functions
    _load_benchmark_data,

    # I/O functions
    load_time_series_data,
)

# AC anchor methods (progress_at_aa computation strategies)
from .ac_anchor import (
    FittedObservable,
    fit_horizon_trend,
    fit_revenue_trend,
    fit_eci_trend,
    compute_progress_at_aa,
    compute_gap_progress_units,
    resolve_observable_anchor,
    resolve_reference_date,
    REFERENCE_DATE_ATTR_BY_METRIC,
)

# Define public API
__all__ = [
    # Types
    'TimeSeriesData',
    'InitialConditions',

    # Main classes
    'TasteDistribution',
    'AutomationModel',
    'Parameters',
    'ProgressModel',

    # Factory functions
    'get_or_create_taste_distribution',

    # CES functions
    '_ces_function',
    'compute_coding_labor_deprecated',
    'compute_rho_from_asymptotes',
    'compute_alpha_experiment_capacity_from_asymptotes',
    'compute_experiment_compute_exponent_from_anchor',
    'compute_exp_capacity_params_from_anchors',

    # Research functions
    'compute_research_effort',
    'compute_software_progress_rate',
    'compute_overall_progress_rate',
    'compute_automation_fraction',
    'compute_ai_research_taste',
    'compute_aggregate_research_taste',

    # Integration functions
    'progress_rate_at_time',
    '_find_exponential_crossing_time',
    'calculate_initial_research_stock',
    'compute_initial_conditions',
    'setup_model',
    'integrate_progress',

    # I/O functions
    'load_time_series_data',

    # AC anchor methods
    'FittedObservable',
    'fit_horizon_trend',
    'fit_revenue_trend',
    'fit_eci_trend',
    'compute_progress_at_aa',
    'compute_gap_progress_units',
    'resolve_observable_anchor',
    'resolve_reference_date',
    'REFERENCE_DATE_ATTR_BY_METRIC',

    # Utilities
    'should_reraise',
    '_coerce_float_scalar',
    '_gauss_hermite_expectation',
]
