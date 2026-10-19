#!/usr/bin/env python3
"""
Module for generating TimeSeriesData instances with uncertainty.

This module provides functionality to generate input time series data with parametric
uncertainty for Monte Carlo simulations. It takes a base CSV file and allows varying
certain time series according to sampled parameters.

Current methodology:
1. Human labor (L_HUMAN) - copied from base CSV
2. Inference compute - copied from base CSV
3. Experiment compute - copied from base CSV
4. Training compute growth rate - constant up to slowdown_year, then switches to
   post_slowdown_training_system_performance_growth_rate

This is designed to be extensible as more sophisticated uncertainty models are developed.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from progress_model import TimeSeriesData, load_time_series_data


@dataclass
class TimeSeriesGenerationParams:
    """Parameters controlling time series generation with uncertainty.

    Attributes:
        base_csv_path: Path to the base CSV file to use as template
        constant_training_system_performance_growth_rate: Growth rate before slowdown
        slowdown_year: Year at which training compute growth rate changes
        post_slowdown_training_system_performance_growth_rate: Growth rate after slowdown
    """
    base_csv_path: str
    constant_training_system_performance_growth_rate: float
    slowdown_year: float
    post_slowdown_training_system_performance_growth_rate: float


def _log_interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Interpolate in log space for exponentially growing quantities."""
    if np.all(fp > 0):
        return np.exp(np.interp(x, xp, np.log(fp)))
    else:
        return np.interp(x, xp, fp)


def generate_time_series_data(
    params: TimeSeriesGenerationParams,
    base_data: Optional[TimeSeriesData] = None
) -> TimeSeriesData:
    """Generate a TimeSeriesData instance with parametric uncertainty.

    This function creates a new TimeSeriesData instance by:
    1. Loading (or using provided) base time series data
    2. Inserting a new data point at slowdown_year (if not already present)
    3. Keeping L_HUMAN, inference_compute, and experiment_compute unchanged
       (interpolated at the new point)
    4. Applying a step function for training_system_performance_growth_rate at slowdown_year

    Args:
        params: Parameters controlling the generation
        base_data: Optional pre-loaded base time series. If None, will load from
                   params.base_csv_path

    Returns:
        A new TimeSeriesData instance with modified training compute growth rate
    """
    # Load base data if not provided
    if base_data is None:
        base_data = load_time_series_data(params.base_csv_path)

    # TEMPORARILY DISABLED: Return base data unchanged (keeping training compute from CSV)
    return base_data

    # Copy the base arrays
    time = base_data.time.copy()
    L_HUMAN = base_data.L_HUMAN.copy()
    inference_compute = base_data.inference_compute.copy()
    experiment_compute = base_data.experiment_compute.copy()

    # Insert data points to create a proper step function at slowdown_year
    # We need a point just before slowdown_year (with pre-slowdown rate) and
    # a point at slowdown_year (with post-slowdown rate) to prevent interpolation ramps
    slowdown_year = params.slowdown_year
    epsilon = 1e-6  # Small offset to create step function

    # Insert point just before slowdown_year (for pre-slowdown rate)
    pre_slowdown_time = slowdown_year - epsilon
    if pre_slowdown_time not in time and pre_slowdown_time > time.min():
        insert_idx = np.searchsorted(time, pre_slowdown_time)

        # Interpolate other time series (log-space for exponential growth)
        L_HUMAN_val = _log_interp(pre_slowdown_time, time, L_HUMAN)
        inference_val = _log_interp(pre_slowdown_time, time, inference_compute)
        experiment_val = _log_interp(pre_slowdown_time, time, experiment_compute)

        time = np.insert(time, insert_idx, pre_slowdown_time)
        L_HUMAN = np.insert(L_HUMAN, insert_idx, L_HUMAN_val)
        inference_compute = np.insert(inference_compute, insert_idx, inference_val)
        experiment_compute = np.insert(experiment_compute, insert_idx, experiment_val)

    # Insert point at slowdown_year (for post-slowdown rate) if not already present
    if slowdown_year not in time:
        insert_idx = np.searchsorted(time, slowdown_year)

        # Interpolate other time series (log-space for exponential growth)
        L_HUMAN_val = _log_interp(slowdown_year, time, L_HUMAN)
        inference_val = _log_interp(slowdown_year, time, inference_compute)
        experiment_val = _log_interp(slowdown_year, time, experiment_compute)

        time = np.insert(time, insert_idx, slowdown_year)
        L_HUMAN = np.insert(L_HUMAN, insert_idx, L_HUMAN_val)
        inference_compute = np.insert(inference_compute, insert_idx, inference_val)
        experiment_compute = np.insert(experiment_compute, insert_idx, experiment_val)

    # Generate training compute growth rate with proper step function at slowdown_year
    # Use < for times before slowdown, >= for times at or after slowdown
    training_system_performance_growth_rate = np.where(
        time < slowdown_year,
        params.constant_training_system_performance_growth_rate,
        params.post_slowdown_training_system_performance_growth_rate
    )

    return TimeSeriesData(
        time=time,
        L_HUMAN=L_HUMAN,
        inference_compute=inference_compute,
        experiment_compute=experiment_compute,
        training_system_performance_growth_rate=training_system_performance_growth_rate
    )


def generate_time_series_from_dict(
    param_dict: dict,
    base_csv_path: str,
    base_data: Optional[TimeSeriesData] = None
) -> TimeSeriesData:
    """Convenience function to generate time series from a parameter dictionary.

    This is useful for integration with Monte Carlo sampling where parameters
    are stored as dictionaries.

    Args:
        param_dict: Dictionary containing time series generation parameters:
            - constant_training_system_performance_growth_rate (required)
            - slowdown_year (required)
            - post_slowdown_training_system_performance_growth_rate (required)
        base_csv_path: Path to base CSV file
        base_data: Optional pre-loaded base time series

    Returns:
        A new TimeSeriesData instance

    Raises:
        KeyError: If required parameters are missing from param_dict
    """
    params = TimeSeriesGenerationParams(
        base_csv_path=base_csv_path,
        constant_training_system_performance_growth_rate=param_dict['constant_training_system_performance_growth_rate'],
        slowdown_year=param_dict['slowdown_year'],
        post_slowdown_training_system_performance_growth_rate=param_dict['post_slowdown_training_system_performance_growth_rate']
    )

    return generate_time_series_data(params, base_data)
