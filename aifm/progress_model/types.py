#!/usr/bin/env python3
"""
Basic data types and dataclasses for progress modeling.

Contains TimeSeriesData and InitialConditions.
Parameters is in a separate module due to its complexity and dependencies.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TimeSeriesData:
    """Input time series data"""
    time: np.ndarray  # Decimal years
    L_HUMAN: np.ndarray  # Human labor supply
    inference_H100e: np.ndarray  # Inference compute capacity (H100-equivalents)
    experiment_H100e: np.ndarray  # Experiment compute capacity (H100-equivalents)
    training_H100e: np.ndarray  # Training system performance: instantaneous H100e in the frontier run (raw H100e)
    # Optional: total AI-relevant compute IN THE WORLD (raw H100e), used by the covert
    # pipeline's bottom_up_estimate sizing to scale a diverted FRACTION of world compute into
    # an absolute H100e count at the pause. None when the input CSV has no `world_H100e` column
    # (the core progress model never reads it).
    world_H100e: Optional[np.ndarray] = None
    # Optional: TOTAL compute operated by the leading AI company (raw H100e), i.e. its
    # R&D compute (inference + experiment + training) plus commercial serving compute.
    # None when the input CSV has no `total_ai_company_H100e` column (the core progress
    # model never reads it; it only feeds display/diagnostics).
    total_ai_company_H100e: Optional[np.ndarray] = None

    def __post_init__(self):
        """Precompute log-space arrays for efficient interpolation.

        Log-space interpolation is used for exponentially growing variables to
        prevent scalloping on log plots and handle exponential growth better.
        Precomputing the logs avoids redundant computation in the ODE solver inner loop.
        """
        # Precompute log arrays for variables that are always positive
        # Use np.where to handle any zeros safely
        self.log_L_HUMAN = np.where(
            self.L_HUMAN > 0,
            np.log(self.L_HUMAN),
            -np.inf
        )
        self.log_inference_H100e = np.where(
            self.inference_H100e > 0,
            np.log(self.inference_H100e),
            -np.inf
        )
        self.log_experiment_H100e = np.where(
            self.experiment_H100e > 0,
            np.log(self.experiment_H100e),
            -np.inf
        )

        # Flags to indicate if log interpolation is safe (all values positive)
        self.can_use_log_L_HUMAN = np.all(self.L_HUMAN > 0)
        self.can_use_log_inference_H100e = np.all(self.inference_H100e > 0)
        self.can_use_log_experiment_H100e = np.all(self.experiment_H100e > 0)

        # Precompute segment rates for training compute growth rate lookup.
        # training_H100e is a raw H100e capacity, so the growth rate in OOMs/year is the
        # slope of log10(training_H100e) (= the slope of log10(FLOP/yr), since FLOP_PER_H100E_YEAR
        # is a constant factor).
        # segment_rates[i] = rate for interval [time[i], time[i+1])
        # log10 of training H100e; the model converts to OOMs of FLOP/yr at point of use by
        # adding log10(FLOP_PER_H100E_YEAR). Precomputed here for the ODE inner loop.
        self.log10_training_H100e = np.log10(np.maximum(self.training_H100e, 1e-300))
        n = len(self.time)
        if n > 1:
            dt = np.diff(self.time)
            self._segment_rates = np.diff(self.log10_training_H100e) / dt
        else:
            self._segment_rates = np.array([0.0])

    def get_training_system_performance_growth_rate(self, t):
        """
        Get training compute growth rate at time t (scalar or array).
        - For t inside an interval: returns segment rate
        - For t exactly at a grid point: returns average of adjacent segment rates
        - For t at first/last time point: returns the single adjacent segment rate
        """
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        n_seg = len(self._segment_rates)
        n_time = len(self.time)

        # Find segment index for each t
        # For t in [time[i], time[i+1]), segment index is i
        indices = np.searchsorted(self.time, t, side='right') - 1
        indices = np.clip(indices, 0, n_seg - 1)

        # Start with segment rates
        rates = self._segment_rates[indices].copy()

        # Check which t values are exactly on interior grid points (vectorized)
        # t is in segment 'indices', so it could be close to time[indices] (left) or time[indices+1] (right)
        eps = 1e-10

        # Check left endpoint of segment (time[indices])
        close_to_left = np.abs(t - self.time[indices]) < eps
        left_is_interior = (indices >= 1) & (indices < n_time - 1)
        on_left_interior = close_to_left & left_is_interior

        if np.any(on_left_interior):
            left_idx = indices[on_left_interior]
            rates[on_left_interior] = (self._segment_rates[left_idx - 1] + self._segment_rates[left_idx]) / 2

        # Check right endpoint of segment (time[indices+1])
        right_grid_idx = np.minimum(indices + 1, n_time - 1)
        close_to_right = np.abs(t - self.time[right_grid_idx]) < eps
        right_is_interior = (right_grid_idx >= 1) & (right_grid_idx < n_time - 1)
        on_right_interior = close_to_right & right_is_interior

        if np.any(on_right_interior):
            right_idx = right_grid_idx[on_right_interior]
            rates[on_right_interior] = (self._segment_rates[right_idx - 1] + self._segment_rates[right_idx]) / 2

        return float(rates[0]) if scalar_input else rates


@dataclass
class InitialConditions:
    """Container for initial model conditions"""
    start_time: float
    initial_progress: float
    initial_automation: float
    L_HUMAN: float
    inference_H100e: float
    experiment_H100e: float
    training_system_performance_growth_rate: float
    coding_labor: float
    research_effort: float
    research_stock: float
