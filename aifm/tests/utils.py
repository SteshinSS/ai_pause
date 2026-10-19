"""
Shared test utilities.

This module contains common functions used across test files and generate_golden_data.py
to avoid code duplication.
"""

import csv
import json
from dataclasses import fields, MISSING
from pathlib import Path

import numpy as np


# Standard time range used in trajectory tests
DEFAULT_TIME_RANGE = [2015.0, 2050.0]

# Golden data directory
GOLDEN_DATA_DIR = Path(__file__).parent / 'golden_data'


def load_time_series_data():
    """Load default inputs/input_data.csv into TimeSeriesData.

    Returns:
        TimeSeriesData object with data from inputs/input_data.csv
    """
    # Import here to avoid circular imports
    from progress_model import TimeSeriesData

    csv_path = Path(__file__).parent.parent / 'inputs' / 'input_data.csv'

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return TimeSeriesData(
        time=np.array([float(r['time']) for r in rows]),
        L_HUMAN=np.array([float(r['L_HUMAN']) for r in rows]),
        inference_H100e=np.array([float(r['inference_H100e']) for r in rows]),
        experiment_H100e=np.array([float(r['experiment_H100e']) for r in rows]),
        training_H100e=np.array([float(r['training_H100e']) for r in rows])
    )


def convert_for_json(obj):
    """Convert numpy arrays and types to JSON-serializable format.

    Args:
        obj: Object to convert (can be nested dicts, lists, numpy arrays, etc.)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj


def save_golden_data(data: dict, filename: str):
    """Save data to golden data directory.

    Args:
        data: Dictionary of data to save
        filename: Name of the file to save to (relative to golden_data dir)
    """
    GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DATA_DIR / filename

    with open(golden_path, 'w') as f:
        json.dump(convert_for_json(data), f, indent=2)

    print(f"Saved golden data to {golden_path}")


def load_golden_data(filename: str):
    """Load data from golden data directory.

    Args:
        filename: Name of the file to load (relative to golden_data dir)

    Returns:
        Loaded JSON data, or None if file doesn't exist
    """
    golden_path = GOLDEN_DATA_DIR / filename
    if not golden_path.exists():
        return None

    with open(golden_path) as f:
        return json.load(f)


def get_input_parameters(scenario_overrides: dict = None) -> dict:
    """Get input parameters that would be passed to Parameters().

    This returns pre-transform INPUT values, not post-init values.
    For golden data generation, this captures the exact defaults at generation time,
    making tests robust to future changes in progress_model.config.DEFAULT_PARAMETERS.

    Args:
        scenario_overrides: Optional dict of parameter overrides for a scenario

    Returns:
        Dictionary of parameter names to input values
    """
    from progress_model import config as cfg
    from progress_model import Parameters

    # Get all valid init field names from Parameters dataclass
    valid_fields = {f.name for f in fields(Parameters) if f.init}

    # Start with defaults, filtered to valid fields only
    result = {k: v for k, v in cfg.DEFAULT_PARAMETERS.items() if k in valid_fields}

    # Apply scenario overrides
    if scenario_overrides:
        for k, v in scenario_overrides.items():
            if k in valid_fields:
                result[k] = v

    return result


def serialize_parameters(params) -> dict:
    """Serialize a Parameters object to a JSON-serializable dict.

    NOTE: This reads POST-init values. For golden data generation,
    prefer get_input_parameters() which captures PRE-transform values.

    Args:
        params: Parameters dataclass instance

    Returns:
        Dictionary of parameter names to values
    """
    # Fields that are non-serializable or computed
    exclude_fields = {
        'taste_distribution',  # Complex object
        'automation_model',  # Complex object
        'automation_anchors',  # Dict that may need special handling
    }

    result = {}
    for f in fields(params):
        if not f.init:
            continue
        if f.name in exclude_fields:
            continue

        value = getattr(params, f.name)
        if value is None:
            continue

        # Convert numpy types to Python types
        if isinstance(value, (np.floating, np.integer)):
            value = float(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()

        result[f.name] = value

    return result


def parameters_from_dict(params_dict: dict):
    """Create a Parameters object from a dictionary.

    Args:
        params_dict: Dictionary of parameter names to values

    Returns:
        Parameters instance
    """
    # Import here to avoid circular imports
    from progress_model import Parameters

    # Filter to only valid Parameter field names
    valid_fields = {f.name for f in fields(Parameters) if f.init}
    filtered_dict = {k: v for k, v in params_dict.items() if k in valid_fields}

    return Parameters(**filtered_dict)
