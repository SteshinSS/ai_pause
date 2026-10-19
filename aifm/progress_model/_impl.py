#!/usr/bin/env python3
"""
Progress Modeling Script

Models AI progress over time using nested CES production functions with
feedback loops between automation fraction and cumulative progress.

This is the implementation module. For external use, import from the
progress_model package instead of this module directly.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union, NamedTuple
from scipy import optimize, integrate, interpolate
from collections import OrderedDict
import logging
import time
from . import config as cfg
try:
    import yaml
except ImportError:
    yaml = None
import copy
from datetime import datetime

# Import utilities from the new modular structure
from .utils import _coerce_float_scalar, _gauss_hermite_expectation, should_reraise
from .types import TimeSeriesData, InitialConditions
from .ces_functions import (
    _ces_function,
    compute_coding_labor_deprecated,
    compute_rho_from_asymptotes,
    compute_experiment_compute_exponent_from_anchor,
    compute_alpha_experiment_capacity_from_asymptotes,
    compute_exp_capacity_params_from_anchors,
)
from .taste_distribution import (
    TasteDistribution,
    get_or_create_taste_distribution,
    _taste_distribution_cache,
    _TASTE_CACHE_MAX_SIZE,
    compute_ai_research_taste as _compute_ai_research_taste_core,
)
from .automation_model import (
    AutomationModel,
    serial_uplift_for_logistic,
    serial_uplift_for_schedule,
    solve_logistic_with_asymptote_from_uplift,
    aut_frac_from_swe_multiplier,
    solve_lower_anchor_via_automation_model,
    solve_logistic_from_uplift,
)
from .parameters import (
    Parameters,
    resolve_swe_multiplier_at_present_day,
)
from .progress_rate import (
    compute_research_effort,
    compute_software_progress_rate,
    compute_overall_progress_rate,
    compute_automation_fraction,
    compute_ai_research_taste,
    compute_aggregate_research_taste,
    progress_rate_at_time,
)
from .utils import _log_interp
from .integration import (
    _find_exponential_crossing_time,
    calculate_initial_research_stock,
    compute_initial_conditions,
    setup_model,
    integrate_progress,
    integrate_progress_human_only,
)
from .metrics_computation import compute_metrics_loop
from .uplift_reconstruction import reconstruct_from_params, reconstruct_uplifted_trajectory, uplift_statistics
from .ac_anchor import (
    FittedObservable,
    fit_horizon_trend,
    fit_revenue_trend,
    fit_eci_trend,
    compute_gap_progress_units,
    compute_progress_at_aa,
    resolve_reference_date,
    _load_benchmark_data,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TasteDistribution and get_or_create_taste_distribution moved to taste_distribution.py
# _load_benchmark_data and horizon/AC-anchor fitting moved to ac_anchor.py


# _log_interp moved to utils.py
# _find_exponential_crossing_time, calculate_initial_research_stock, compute_initial_conditions,
# setup_model, integrate_progress moved to integration.py
# aut_frac_from_swe_multiplier, solve_lower_anchor_via_automation_model moved to automation_model.py


class ProgressModel:
    """Main class for AI progress modeling"""
    
    def __init__(self, params: Parameters, time_series_data: TimeSeriesData):
        """
        Time series data is from the capabilities input spreadsheet.
        """
        self.params = params
        self.data = time_series_data
        self.human_only_results = {}
        self.results = {}
        self.horizon_trajectory = None
        self.sc_sw_multiplier = None

        # Fitted observables (see ac_anchor.py)
        self._horizon_observable = None      # always fit; drives horizon_lengths metric
        self._ac_anchor_observable = None    # alias of the active AC-anchor mode's observable, if non-horizon
        self._eci_observable = None          # always built; drives capability_ecis metric
        self._revenue_observable = None      # always built; drives predicted_revenue metric
        self._uplift_solve_diagnostics = None  # populated in 'uplift-solved' AC-anchor mode
        self._uplift_solve_public = None       # subset surfaced through the API
        self._horizon_uses_shifted_form = False
        # Pre-conversion slopes, captured once so the per-anchor-year -> per-progress-unit
        # conversions stay idempotent across the T_d correction pass's re-derivation
        # (and across a second compute_progress_trajectory on the same model).
        self._original_taste_slope = None
        self._original_automation_efficiency_slope = None
        # T_d correction pass outcome; see _apply_td_correction_pass.
        self._td_correction_info = {'applied': False}

        # The anchor_stats dict every date-anchored quantity was derived from:
        # the reconstruction's, or the human-only run's in the degenerate
        # no-uplift case where the two coincide.
        self._anchor_stats_used = None
        # The (times, progress) date -> progress mapping those quantities (and
        # the horizon fit) were derived from -- same source as
        # _anchor_stats_used. compute_metr_mse reads it so the MSE evaluates
        # the fitted curve in the same progress coordinates it was fit on.
        self._anchor_mapping = None
        # Diagnostics from the one-shot uplift reconstruction the anchoring is
        # derived against, and the reconstructed trajectory itself. 'available'
        # is False only in the degenerate no-uplift case, where the
        # reconstruction would be the human-only run anyway.
        self._uplift_reconstruction_info = {'available': False}
        self._uplift_reconstruction = None

        # Use taste distribution from params (initialized in Parameters.__post_init__)
        self.taste_distribution = self.params.taste_distribution
    def freeze_time_series_at_time(self, freeze_time: float):
        """
        Freeze time series data at a specified time.

        Human labor, experiment compute, and inference compute remain unchanged up to freeze_time,
        then stay constant at their freeze_time values thereafter.

        Training compute is unchanged up to freeze_time, then stays constant (zero growth rate)
        after freeze_time.

        Args:
            freeze_time: Time at which to freeze the time series (decimal year)
        """
        # Get the freeze_time values using log interpolation
        freeze_human_labor = _log_interp(freeze_time, self.data.time, self.data.L_HUMAN)
        freeze_experiment_compute = _log_interp(freeze_time, self.data.time, self.data.experiment_H100e)
        freeze_inference_compute = _log_interp(freeze_time, self.data.time, self.data.inference_H100e)
        freeze_training_compute = _log_interp(freeze_time, self.data.time, self.data.training_H100e)

        insertion_idx = np.searchsorted(self.data.time, freeze_time)

        time_with_freeze = np.insert(self.data.time, insertion_idx, freeze_time)
        human_labor_with_freeze = np.insert(self.data.L_HUMAN, insertion_idx, freeze_human_labor)
        experiment_compute_with_freeze = np.insert(self.data.experiment_H100e, insertion_idx, freeze_experiment_compute)
        inference_compute_with_freeze = np.insert(self.data.inference_H100e, insertion_idx, freeze_inference_compute)
        training_compute_with_freeze = np.insert(self.data.training_H100e, insertion_idx, freeze_training_compute)

        # Find where to insert the freeze_time point
        # Insert freeze_time + epsilon to ensure proper ordering and transition
        epsilon = 1e-6  # Small epsilon to ensure freeze_time + epsilon comes after freeze_time
        freeze_time_after = freeze_time + epsilon

        # Find the insertion point (first index where time > freeze_time)
        new_insertion_idx = insertion_idx + 1

        # Create new arrays with the freeze point inserted
        new_time = np.insert(time_with_freeze, new_insertion_idx, freeze_time_after)
        new_L_HUMAN = np.insert(human_labor_with_freeze, new_insertion_idx, freeze_human_labor)
        new_experiment_compute = np.insert(experiment_compute_with_freeze, new_insertion_idx, freeze_experiment_compute)
        new_inference_compute = np.insert(inference_compute_with_freeze, new_insertion_idx, freeze_inference_compute)
        # Training compute stays constant after freeze_time (same value = zero growth rate)
        new_training_compute = np.insert(training_compute_with_freeze, new_insertion_idx, freeze_training_compute)

        # Find indices where time >= freeze_time_after (including the newly inserted point)
        freeze_mask = new_time >= freeze_time_after

        # Set frozen values for human labor, experiment compute, and inference compute
        new_L_HUMAN[freeze_mask] = freeze_human_labor
        new_experiment_compute[freeze_mask] = freeze_experiment_compute
        new_inference_compute[freeze_mask] = freeze_inference_compute
        # Training compute stays constant at freeze value (zero growth rate)
        new_training_compute[freeze_mask] = freeze_training_compute

        # Create new TimeSeriesData object
        self.data = TimeSeriesData(
            time=new_time,
            L_HUMAN=new_L_HUMAN,
            inference_H100e=new_inference_compute,
            experiment_H100e=new_experiment_compute,
            training_H100e=new_training_compute,
        )
        
        logger.info(f"Frozen time series at time {freeze_time}:")
        logger.info(f"  Human labor: {freeze_human_labor:.6f}")
        logger.info(f"  Experiment compute: {freeze_experiment_compute:.6f}")
        logger.info(f"  Inference compute: {freeze_inference_compute:.6f}")
        logger.info(f"  Training compute: {freeze_training_compute:.6f} (frozen)")
        logger.info(f"  Inserted freeze point at time {freeze_time_after:.6f}")

    def find_sos_milestone_time(
        self,
        times: np.ndarray,
        progress_values: np.ndarray,
        research_stock_values: np.ndarray,
        milestone: str
    ) -> Optional[float]:
        """
        Find the time when the specified SOS milestone is reached.

        Args:
            times: Array of time points from integration
            progress_values: Array of progress values from integration
            research_stock_values: Array of research stock values from integration
            milestone: The milestone name (e.g., 'AC', 'AI2027-SC', 'AIR-5x')

        Returns:
            Time when milestone is reached, or None if not reached
        """
        if milestone == 'AC':
            # AC uses progress directly
            if progress_values[-1] > self.params.progress_at_aa:
                return float(np.interp(self.params.progress_at_aa, progress_values, times))
            return None

        # For other milestones, we need to compute metrics
        # Get present-day reference values
        present_day_human_labor = _log_interp(self.params.present_day, self.data.time, self.data.L_HUMAN)
        present_day_inference_compute = _log_interp(self.params.present_day, self.data.time, self.data.inference_H100e)
        present_day_experiment_compute = _log_interp(self.params.present_day, self.data.time, self.data.experiment_H100e)

        # Baseline for the AIR uplift metric: the human-only run's present-day
        # research effort -- the same value compute_metrics_loop divides by, so
        # the milestone crossing found here agrees with the stored metric array.
        present_day_human_only_research_effort = float(
            self.human_only_results['anchor_stats']['research_effort'])

        # Determine which metric and target we need
        if milestone == 'AI2027-SC':
            target = (30 ** (1 / self.params.parallel_penalty)) * 30 * cfg.PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC
            metric_name = 'ai_coding_labor_mult_ref_present_day'
        elif milestone.startswith('AIR-'):
            # Extract multiplier from milestone name (e.g., 'AIR-5x' -> 5)
            mult_str = milestone.replace('AIR-', '').replace('x', '')
            target = float(mult_str)
            metric_name = 'ai_sw_progress_mult_ref_present_day'
        else:
            logger.warning(f"Unknown SOS milestone: {milestone}, falling back to AC")
            if progress_values[-1] > self.params.progress_at_aa:
                return float(np.interp(self.params.progress_at_aa, progress_values, times))
            return None

        # Compute the relevant metric for each time point
        metric_values = []
        for i, (t, progress, rs) in enumerate(zip(times, progress_values, research_stock_values)):
            L_HUMAN = _log_interp(t, self.data.time, self.data.L_HUMAN)
            inference_H100e = _log_interp(t, self.data.time, self.data.inference_H100e)
            experiment_H100e = _log_interp(t, self.data.time, self.data.experiment_H100e)

            # Compute automation fraction and taste
            ai_taste = compute_ai_research_taste(progress, self.params)
            aggregate_taste = compute_aggregate_research_taste(ai_taste, self.params.taste_distribution)

            # Compute coding labor
            if getattr(self.params, 'coding_labor_mode', 'simple_ces') == 'optimal_ces':
                H = float(L_HUMAN)
                C = float(inference_H100e)
                lnE = float(np.log(cfg.BASE_FOR_SOFTWARE_LOM) * progress)
                automation_model = self.params.automation_model
                coding_labor = automation_model.coding_labor_optimal_ces(H, C, lnE, self.params)
                coding_labor_with_present_resources = automation_model.coding_labor_optimal_ces(
                    present_day_human_labor, present_day_inference_compute, lnE, self.params
                )
                serial_coding_labor = float((coding_labor ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
                serial_coding_labor_with_present_resources = float((coding_labor_with_present_resources ** self.params.parallel_penalty) * self.params.coding_labor_normalization)
            else:
                automation_fraction = compute_automation_fraction(progress, self.params)
                serial_coding_labor = compute_coding_labor_deprecated(
                    automation_fraction, inference_H100e, L_HUMAN,
                    self.params.rho_coding_labor, self.params.parallel_penalty, self.params.coding_labor_normalization
                )
                serial_coding_labor_with_present_resources = compute_coding_labor_deprecated(
                    automation_fraction, present_day_inference_compute, present_day_human_labor,
                    self.params.rho_coding_labor, self.params.parallel_penalty, self.params.coding_labor_normalization
                )
                coding_labor_with_present_resources = serial_coding_labor_with_present_resources

            if metric_name == 'ai_coding_labor_mult_ref_present_day':
                # ai_coding_labor_mult_ref_present_day = coding_labor_with_present_resources / present_day_human_labor
                metric_val = coding_labor_with_present_resources / present_day_human_labor if present_day_human_labor > 0 else 1.0
            else:  # ai_sw_progress_mult_ref_present_day
                # Research-effort ratio vs the human-only present day (the same
                # definition compute_metrics_loop uses)
                research_effort_present_resources = compute_research_effort(
                    present_day_experiment_compute, serial_coding_labor_with_present_resources,
                    self.params.alpha_experiment_capacity, self.params.rho_experiment_capacity,
                    self.params.experiment_compute_exponent, aggregate_taste
                )
                metric_val = research_effort_present_resources / present_day_human_only_research_effort if present_day_human_only_research_effort > 0 else 0.0

            metric_values.append(metric_val if np.isfinite(metric_val) else 0.0)

        # Find crossing time using exponential interpolation
        crossing_time = _find_exponential_crossing_time(
            np.asarray(times, dtype=float),
            np.asarray(metric_values, dtype=float),
            float(target)
        )

        return crossing_time

    def estimate_horizon_trajectory(self, human_only_times: np.ndarray, human_only_progress: np.ndarray, anchor_progress_rate: float,
                                    mapping_progress_rates: Optional[np.ndarray] = None):
        """
        Fit the horizon trend (log p80_horizon_length vs progress) and store it.

        The fitting itself lives in ac_anchor.fit_horizon_trend; this method
        just stores the result on the model. Note: unlike the historical
        version, this no longer computes progress_at_aa — that is done by
        _compute_ac_anchor, which dispatches on params.ac_anchor_mode.

        Args:
            human_only_times: Time array from human-only trajectory computation
            human_only_progress: Progress array from human-only trajectory computation
            anchor_progress_rate: Progress rate at anchor time (progress units per time unit)
            mapping_progress_rates: d(progress)/dt on the same grid; consulted
                only when params.horizon_reference_date is set

        Returns:
            Function that maps progress to horizon length, or None on failure
        """
        observable = fit_horizon_trend(
            self.params, human_only_times, human_only_progress, anchor_progress_rate,
            mapping_progress_rates=mapping_progress_rates,
        )
        if observable is None:
            return None
        self._horizon_observable = observable
        self.horizon_trajectory = observable.predict
        self._horizon_uses_shifted_form = bool(observable.fit_info.get('uses_shifted_form', False))
        return observable.predict

    def _compute_ac_anchor(self, human_only_times: np.ndarray, human_only_progress: np.ndarray, anchor_progress_rate: float,
                           mapping_progress_rates: Optional[np.ndarray] = None):
        """
        Set params.progress_at_aa according to params.ac_anchor_mode.

        - "time-horizon" (default): invert the fitted horizon trend at
          ac_time_horizon_minutes.
        - "revenue": invert the revenue trend (anchored at present_day) at
          revenue_required_for_ac.
        - "eci": invert the ECI trend (linear in progress, anchored at
          present_day) at eci_required_for_ac.
        - "uplift-solved": no observable inversion at all — progress_at_aa is an
          OUTPUT of the joint (k, x0) solve against the model's own endogenous
          uplift, which happens later in compute_progress_trajectory (it needs
          the automation-efficiency slope already converted to progress units).
          This method only builds the metric observables and leaves
          _ac_anchor_observable as None.

        The ECI and revenue trends are built regardless of mode (their metrics
        are always emitted); the selected mode's observable is also kept on
        self._ac_anchor_observable.

        If the relevant observable could not be fit, progress_at_aa is left
        unchanged (matching historical behavior when the horizon fit failed).
        """
        mode = getattr(self.params, 'ac_anchor_mode', 'time-horizon')

        # The ECI and revenue trends are built in every mode — they define the
        # capability_ecis / predicted_revenue metrics; each one only drives
        # progress_at_aa when its mode is selected.
        self._eci_observable = fit_eci_trend(
            self.params, human_only_times, human_only_progress, anchor_progress_rate,
            mapping_progress_rates=mapping_progress_rates,
        )
        self._revenue_observable = fit_revenue_trend(
            self.params, human_only_times, human_only_progress, anchor_progress_rate,
            mapping_progress_rates=mapping_progress_rates,
        )
        if mode == 'uplift-solved':
            # progress_at_aa is solved for later, jointly with the automation
            # schedule; there is no exogenous observable to invert here.
            self._ac_anchor_observable = None
            return

        if mode == 'revenue':
            observable = self._revenue_observable
            self._ac_anchor_observable = observable
            target = getattr(self.params, 'revenue_required_for_ac', None)
        elif mode == 'eci':
            observable = self._eci_observable
            self._ac_anchor_observable = observable
            target = getattr(self.params, 'eci_required_for_ac', None)
        else:
            if mode != 'time-horizon':
                logger.warning(f"Unknown ac_anchor_mode: {mode}, falling back to time-horizon")
            observable = self._horizon_observable
            target = self.params.ac_time_horizon_minutes

        if observable is None:
            logger.warning(f"AC anchor observable unavailable (mode={mode}); progress_at_aa unchanged")
            return

        if target is None or not target > 0:
            return

        gap_progress_units = compute_gap_progress_units(self.params, anchor_progress_rate)
        self.params.progress_at_aa = compute_progress_at_aa(observable, target, gap_progress_units)

    def _solve_uplift_anchored_automation_model(
        self,
        mapping_times: np.ndarray,
        mapping_progress: np.ndarray,
        anchor_progress_rate: float,
        present_day_progress: float,
        present_day_human_labor: float,
        present_day_inference_compute: float,
        mapping_progress_rates: Optional[np.ndarray] = None,
    ):
        """
        'uplift-solved' AC-anchor mode: solve the logistic automation schedule
        directly from two statements about the model's OWN endogenous serial
        ref-present-day coding uplift M, rather than reading progress_at_aa off
        an exogenous trend and then back-solving the lower anchor.

            M(p*)                 = serial_uplift_at_reference_date
            d ln(M - 1)/dp |_{p*} = ln2 / D_progress

        with p* the progress level at uplift_reference_date on the anchoring
        MAPPING (the one-shot reconstruction of the realized trajectory) and
        D_progress the excess-uplift doubling time converted from anchor-months.

        uplift_reference_date is a genuinely independent parameter, NOT an alias
        for present_day: the uplift elicitation is anchored at a later date than
        the other observables (~2026.578 vs present_day 2026.0959), so t* must
        stay separately settable. `None` means "use present_day" and is only a
        convenience default.

        Unit conversion for the doubling time uses the progress rate AT THE
        REFERENCE DATE, rate(t*), looked up from the same mapping's
        progress_rates — not the present-day anchor rate. The two
        coincide exactly when t* == present_day (anchor_stats['progress_rate']
        is that same np.interp lookup), so the default is unchanged, but the
        rates diverge materially for a t* a year out. This is deliberately
        stricter than the other observables, which hardwire the anchor rate.

        Sets params.progress_at_aa (the A(p) = 1 crossing plus the AC gap),
        params.automation_anchors (synthesized for display/back-compat) and
        params.automation_model.

        Must run AFTER coding_automation_efficiency_slope has been converted to
        progress units, because the optimal-CES frontier consumes it.
        """
        if self.params.automation_interp_type != 'logistic':
            raise ValueError(
                f"ac_anchor_mode='uplift-solved' requires automation_interp_type='logistic', "
                f"got '{self.params.automation_interp_type}'. The mode solves for the logistic "
                f"schedule's (k, x0) directly, so no other schedule shape is meaningful."
            )

        # p* and rate(t*) are read off the SAME mapping every other date-anchored
        # quantity is derived from: the one-shot reconstruction of the realized
        # trajectory, or the human-only run in the degenerate no-uplift case
        # where the two coincide. The rate array is optional only because a
        # caller may not have one on the mapping's own grid; the human-only
        # rates are then exact, because the mapping is then the human-only run.
        if mapping_progress_rates is None:
            rate_times = np.asarray(self.human_only_results['times'], dtype=float)
            rate_values = np.asarray(self.human_only_results['progress_rates'], dtype=float)
        else:
            rate_times = np.asarray(mapping_times, dtype=float)
            rate_values = np.asarray(mapping_progress_rates, dtype=float)

        reference_date = getattr(self.params, 'uplift_reference_date', None)
        if reference_date is None:
            reference_date = float(self.params.present_day)
        else:
            reference_date = float(reference_date)
        reference_progress = float(np.interp(reference_date, mapping_times, mapping_progress))
        # rate(t*), the TRUE d(progress)/dt at the reference date. At
        # t* == present_day this is bit-identical to anchor_progress_rate.
        reference_progress_rate = float(np.interp(reference_date, rate_times, rate_values))
        if not (np.isfinite(reference_progress_rate) and reference_progress_rate > 0):
            logger.warning(
                f"uplift-solved: invalid progress rate {reference_progress_rate} at reference date "
                f"{reference_date}; falling back to the present-day anchor rate {anchor_progress_rate}"
            )
            reference_progress_rate = float(anchor_progress_rate)

        target_uplift = float(self.params.serial_uplift_at_reference_date)
        doubling_months = float(self.params.excess_uplift_doubling_time)
        # reference-months -> progress units, at the reference date's own rate
        doubling_progress_units = doubling_months / 12.0 * reference_progress_rate
        asymptote_L = float(self.params.automation_logistic_asymptote)

        logger.info(
            f"uplift-solved AC anchor: targeting serial uplift {target_uplift}"
            f" at """
            f"{reference_date} (progress {reference_progress:.6f}), excess doubling every "
            f"{doubling_months:.4g} months = {doubling_progress_units:.6f} progress units "
            f"(rate at reference date {reference_progress_rate:.6f}; present-day anchor rate "
            f"{anchor_progress_rate:.6f}), asymptote L={asymptote_L}"
        )

        # OPTIONAL third condition: if serial_uplift_at_ac is set, the asymptote is
        # solved for too rather than taken from automation_logistic_asymptote. Off by
        # default; the two-condition path below is unchanged when it is None.
        uplift_at_ac = getattr(self.params, 'serial_uplift_at_ac', None)
        if uplift_at_ac is not None:
            # The numbers explaining a failure travel back in this dict so they can go
            # IN the exception below rather than only into a log record -- automated
            # callers (Monte Carlo rejection/clamping) catch without capturing logs.
            solve_failure: Dict[str, Any] = {}
            solved3 = solve_logistic_with_asymptote_from_uplift(
                target_uplift,
                reference_progress,
                doubling_progress_units,
                float(uplift_at_ac),
                present_day_human_labor,
                present_day_inference_compute,
                self.params,
                failure_info=solve_failure,
            )
            if solved3 is None:
                reachable = solve_failure.get('reachable')
                if reachable is not None:
                    range_note = (
                        f"reachable range at this level and doubling time is "
                        f"[{reachable[0]:.6g}, {reachable[1]:.6g}]x (uplift at AC is monotone "
                        f"decreasing in the asymptote and flattens for large L, so low targets "
                        f"in particular are unreachable at any asymptote)"
                    )
                else:
                    range_note = (
                        "no reachable range could be established -- the inner two-condition "
                        "solve failed at essentially every asymptote on the ladder"
                    )
                raise ValueError(
                    f"ac_anchor_mode='uplift-solved' with serial_uplift_at_ac="
                    f"{uplift_at_ac} could not find an asymptote reproducing that uplift at "
                    f"AC, given serial uplift {target_uplift} at {reference_date} and a "
                    f"{doubling_months}-month excess doubling time "
                    f"({doubling_progress_units:.6g} progress units). {range_note}. "
                    f"[failure reason: {solve_failure.get('reason', 'unknown')}]"
                )
            asymptote_L, logistic_k, logistic_x0, diagnostics = solved3
            logger.info(
                f"uplift-solved AC anchor: asymptote SOLVED to L={asymptote_L:.6f} from "
                f"uplift at AC = {uplift_at_ac} (automation_logistic_asymptote input ignored)"
            )
            solved = (logistic_k, logistic_x0, diagnostics)
        else:
            solved = solve_logistic_from_uplift(
                target_uplift,
                reference_progress,
                doubling_progress_units,
                asymptote_L,
                present_day_human_labor,
                present_day_inference_compute,
                self.params,
            )
        if solved is None:
            raise ValueError(
                f"ac_anchor_mode='uplift-solved' could not find a logistic schedule with serial "
                f"uplift {target_uplift} at {reference_date} and an excess-uplift doubling time of "
                f"{doubling_months} months ({doubling_progress_units:.6f} progress units at the "
                f"reference date's rate of {reference_progress_rate:.6f}) "
                f"under asymptote L={asymptote_L}. See the preceding warning for the reachable range."
            )

        logistic_k, logistic_x0, diagnostics = solved

        # Always record the serial uplift implied at the A(p)=1 crossing, whether or
        # not the asymptote was solved from it. The UI needs a feasible seed when the
        # user switches the solved-asymptote option ON: the reachable floor moves with
        # M* and D (2.3x to 369x across plausible inputs), so there is no safe
        # constant -- but the value the current schedule already implies is on the
        # feasible manifold by construction.
        _crossing = logistic_x0 - float(np.log(asymptote_L - 1.0)) / logistic_k
        try:
            _implied_at_ac = float(serial_uplift_for_logistic(
                [_crossing], logistic_k, logistic_x0, asymptote_L,
                present_day_human_labor, present_day_inference_compute, self.params,
            )[0])
        except Exception as e:
            if should_reraise(e):
                raise
            _implied_at_ac = float('nan')

        diagnostics.update({
            'serial_uplift_at_ac_implied': _implied_at_ac,
            'reference_date': reference_date,
            'reference_progress_rate': reference_progress_rate,
            'present_day_anchor_progress_rate': float(anchor_progress_rate),
            'excess_uplift_doubling_time_months': doubling_months,
        })
        self._uplift_solve_diagnostics = diagnostics
        self._uplift_solve_public = {
            'asymptote_L': float(asymptote_L),
            'asymptote_was_solved': bool(diagnostics.get('asymptote_was_solved', False)),
            'serial_uplift_at_ac_implied': _implied_at_ac,
            'reference_date': diagnostics.get('reference_date'),
            'reference_progress_rate': diagnostics.get('reference_progress_rate'),
        }

        # progress_at_aa is now an OUTPUT: the A(p) = 1 crossing, plus the AC gap.
        # The gap keeps its own convention (anchor-progress-years at the
        # present-day rate) shared with every other AC-anchor mode.
        progress_at_crossing = logistic_x0 - float(np.log(asymptote_L - 1.0)) / logistic_k
        gap_progress_units = compute_gap_progress_units(self.params, anchor_progress_rate)
        self.params.progress_at_aa = progress_at_crossing + gap_progress_units

        # Synthesize the anchor pair for display / back-compat. Both points lie
        # exactly on the solved logistic, so re-fitting AutomationModel from these
        # anchors recovers the same (k, x0).
        automation_model = AutomationModel(self.params, logistic=(asymptote_L, logistic_k, logistic_x0))
        self.params.automation_anchors = {
            float(present_day_progress): float(automation_model.get_automation_fraction(present_day_progress)),
            float(progress_at_crossing): 1.0,
        }
        self.params.automation_model = automation_model

        logger.info(
            f"uplift-solved AC anchor: k={logistic_k:.6f}, x0={logistic_x0:.6f}, "
            f"A(p)=1 crossing at progress {progress_at_crossing:.6f}, "
            f"gap {gap_progress_units:.6f} -> progress_at_aa={self.params.progress_at_aa:.6f}; "
            f"anchors {self.params.automation_anchors}"
        )
        return diagnostics

    def extra_observables(self):
        """Non-horizon fitted observables to emit as metrics (deduped:
        _ac_anchor_observable aliases the ECI or revenue trend in those modes)."""
        observables = []
        for obs in (self._ac_anchor_observable, self._eci_observable, self._revenue_observable):
            if obs is not None and not any(obs is o for o in observables):
                observables.append(obs)
        return observables or None

    def _derive_automation_schedule(self, mapping_times, mapping_progress,
                                    mapping_progress_rates, anchor_progress_rate,
                                    present_day_progress, present_day_human_labor,
                                    present_day_inference_compute):
        """Derive the automation schedule against a mapping. Two routes, as before:
        the uplift solve, or the lower-anchor solve plus an anchor-fitted model.

        Extracted so the T_d correction pass re-derives through the SAME code as the
        primary path rather than a second copy of it.
        """
        if getattr(self.params, 'ac_anchor_mode', 'time-horizon') == 'uplift-solved':
            self._solve_uplift_anchored_automation_model(
                mapping_times, mapping_progress, anchor_progress_rate,
                float(present_day_progress), float(present_day_human_labor),
                float(present_day_inference_compute),
                mapping_progress_rates=mapping_progress_rates,
            )
            return

        # The anchor solve targets the PARALLEL multiplier at present_day, which is a
        # derived view of serial_uplift_at_reference_date -- go through the resolver.
        _swe_target = resolve_swe_multiplier_at_present_day(self.params)
        anchor_aut_frac = solve_lower_anchor_via_automation_model(
            _swe_target,
            float(present_day_progress),
            float(present_day_human_labor),
            float(present_day_inference_compute),
            self.params,
        )
        logger.info(f"calculated anchor automation fraction (via AM solver): {anchor_aut_frac} "
                    f"from swe_multiplier_at_present_day: {_swe_target} and present_day: {float(self.params.present_day)}")
        self.params.automation_anchors = {
            present_day_progress: anchor_aut_frac,
            self.params.progress_at_aa: self.params.automation_fraction_at_coding_automation_anchor
        }
        logger.info(f"Automation anchors: {self.params.automation_anchors}")
        self.params.automation_model = AutomationModel(self.params)

    def _convert_anchor_year_slopes(self, anchor_progress_rate):
        """Per-anchor-year -> per-progress-unit for the taste and automation-efficiency
        slopes. Idempotent by form: always converts from the captured originals, so the
        correction pass can re-derive without compounding the division."""
        if not (anchor_progress_rate is not None and np.isfinite(anchor_progress_rate)
                and anchor_progress_rate > 0):
            logger.warning(f"Invalid anchor_progress_rate for slope conversion: {anchor_progress_rate}")
            return
        if self.params.taste_schedule_type == "SDs per progress-year":
            self.params.ai_research_taste_slope = (
                float(self._original_taste_slope) / anchor_progress_rate)
        self.params.coding_automation_efficiency_slope = (
            float(self._original_automation_efficiency_slope) / anchor_progress_rate)

    def _apply_td_correction_pass(self, mapping, present_day_progress,
                                  present_day_human_labor, present_day_inference_compute):
        """Rebuild the anchoring mapping from the schedule's MEASURED uplift statistics.

        The reconstruction above was built from the BELIEVED (M, T_d), before any
        schedule existed. Whether the schedule then derived against it reproduces that
        belief is a property of the AC-anchor MODE, not of the parameter draw:

          'uplift-solved'   the solve pins the schedule to the belief. Measured
                            mismatch 0.01-0.5% across randomized draws -- inside the
                            reconstruction's own ~0.4% rate error, so there is nothing
                            to correct, and correcting anyway made the present-day
                            anchor residual WORSE (4.02% -> 4.18%).
          every other mode  T_d is an OUTPUT of the horizon / revenue / ECI anchor and
                            the belief is only the starting guess. The derived schedule
                            misses it on every draw, never marginally: 9-127% in
                            time-horizon, 13-67% in revenue.

        So the decision is the mode. Outside 'uplift-solved' the pass ALWAYS rebuilds the
        reconstruction from the measured statistics and re-derives the horizon trend, the
        AC anchor, the slope conversions and the automation schedule against it, once.
        That replaces the stated -- or, in a Monte Carlo, the SAMPLED --
        excess_uplift_doubling_time with the schedule's own. The replacement is written
        to results['td_correction'] (believed, measured, applied), so every rollout
        record and API response says which T_d the run actually anchored on. In
        'uplift-solved' the statistics are measured and recorded the same way and the
        mapping is left exactly as built.

        One caveat on the recorded mismatch in 'uplift-solved': the solve pins T_d at
        uplift_reference_date, while _measure_schedule_uplift_statistics reads it as a
        secant around present_day. When the two dates coincide (the Q2 presets) the
        mismatch is the ~0.5% above; when t* is moved away it reads the logistic's local
        doubling time at a different date and can be several percent (6.8% at
        t* = 2027.0). That is a comparison across dates, not a failed solve, and the
        pass still does not apply.

        Until Sep 2026 the decision was a 5% relative-mismatch threshold. Given the
        numbers above it never discriminated within a mode, so it was a mode switch
        spelled as a tolerance -- one that hid the fact that the sampled T_d is
        discarded in two of the three sampled modes. (Its one real effect was the
        offset-t* case just described, where it fired inside 'uplift-solved' too.)

        Costs one reconstruction plus one anchor re-derivation, both pre-integration:
        no extra ODE solve (~25 ms against a ~350 ms run).

        Returns the mapping to anchor against -- the rebuilt one, or the original.
        """
        mode = getattr(self.params, 'ac_anchor_mode', 'time-horizon')
        info = {'applied': False, 'ac_anchor_mode': mode,
                'believed': None, 'measured': None, 'mismatch': None, 'reason': None}
        self._td_correction_info = info

        believed = uplift_statistics(self.params)
        measured = self._measure_schedule_uplift_statistics(
            mapping['times'], mapping['progress'],
            np.asarray(mapping['progress_rates'], dtype=float),
            present_day_human_labor, present_day_inference_compute)
        info['believed'] = believed
        info['measured'] = measured

        if believed is not None and measured is not None:
            def _rel(a, b):
                return abs(a - b) / abs(b) if b else float('inf')
            info['mismatch'] = max(
                _rel(measured['serial_multiplier'], believed['serial_multiplier']),
                _rel(measured['doubling_time_years'], believed['doubling_time_years']))

        if mode == 'uplift-solved':
            info['reason'] = "ac_anchor_mode='uplift-solved' pins the schedule to the belief"
            if info['mismatch'] is not None:
                logger.info(f"T_d correction pass: not applied in uplift-solved mode; the "
                            f"schedule reproduces the belief to {info['mismatch']:.4%}")
            return mapping
        if believed is None or measured is None:
            info['reason'] = ("the belief or the derived schedule's uplift statistics could "
                              "not be measured; anchoring against the mapping as built")
            logger.warning(f"T_d correction pass: {info['reason']}")
            return mapping

        corrected = reconstruct_uplifted_trajectory(
            self.human_only_results, self.data, self.params,
            serial_multiplier=measured['serial_multiplier'],
            doubling_time_years=measured['doubling_time_years'],
            eval_date=measured['eval_date'],
        )
        if corrected is None:
            info['reason'] = 'rebuilding the reconstruction from the measured statistics failed'
            logger.warning(f"T_d correction pass: {info['reason']}; keeping the original mapping")
            return mapping

        logger.info(
            f"T_d correction pass APPLIED (ac_anchor_mode='{mode}'): the stated "
            f"excess_uplift_doubling_time is not what this mode's schedule produces, so the "
            f"anchoring uses the schedule's own. believed (M={believed['serial_multiplier']:.6f}, "
            f"T_d={believed['doubling_time_years']*12:.3f} mo) -> measured "
            f"(M={measured['serial_multiplier']:.6f}, T_d={measured['doubling_time_years']*12:.3f} mo), "
            f"mismatch {info['mismatch']:.4%}")
        info['applied'] = True

        rates = np.asarray(corrected['progress_rates'], dtype=float)
        rate = corrected['anchor_stats']['progress_rate']
        self.estimate_horizon_trajectory(corrected['times'], corrected['progress'], rate,
                                         mapping_progress_rates=rates)
        self._compute_ac_anchor(corrected['times'], corrected['progress'], rate,
                                mapping_progress_rates=rates)
        self._convert_anchor_year_slopes(rate)
        self._derive_automation_schedule(
            corrected['times'], corrected['progress'], rates, rate,
            float(np.interp(float(self.params.present_day), corrected['times'], corrected['progress'])),
            present_day_human_labor, present_day_inference_compute)
        self._uplift_reconstruction = corrected
        return corrected

    def _td_correction_public(self):
        """results['td_correction']: which uplift statistics this run anchored on.

        JSON-ready so it lands verbatim in every rollout record and API summary. T_d is
        reported in months to match the excess_uplift_doubling_time parameter it
        stands in for.
        """
        info = self._td_correction_info or {}

        def _stats(s):
            if not s:
                return None
            return {'serial_multiplier': float(s['serial_multiplier']),
                    'excess_doubling_time_months': 12.0 * float(s['doubling_time_years']),
                    'eval_date': float(s['eval_date'])}

        mismatch = info.get('mismatch')
        return {
            'ac_anchor_mode': info.get('ac_anchor_mode'),
            'applied': bool(info.get('applied', False)),
            'believed': _stats(info.get('believed')),
            'measured': _stats(info.get('measured')),
            'mismatch': None if mismatch is None else float(mismatch),
            'reason': info.get('reason'),
        }

    def _measure_schedule_uplift_statistics(self, mapping_times, mapping_progress,
                                            mapping_progress_rates,
                                            present_day_human_labor,
                                            present_day_inference_compute):
        """What the DERIVED schedule actually produces, as uplift statistics.

        The anchoring reconstruction is built from the believed (M, T_d) before any
        schedule exists. This reads the same two statistics back off the schedule that
        was then derived, so the two can be compared:

            M    = serial uplift at the reference date
            T_d  = doubling time of the excess (M - 1) around present_day

        Pure quadrature on the mapping's progress axis -- CES evaluations at a handful
        of progress levels, no integration. Returns None if anything is unusable, which
        makes the correction pass skip rather than fail the run.
        """
        try:
            automation_model = getattr(self.params, 'automation_model', None)
            if automation_model is None:
                return None

            present_day = float(self.params.present_day)
            t_star = getattr(self.params, 'uplift_reference_date', None)
            t_star = present_day if t_star is None else float(t_star)

            # Half-window for the doubling-time secant, in years. Wide enough that the
            # excess moves well clear of numerical noise, narrow enough to stay local.
            half = 0.25
            probe_times = np.array([t_star, present_day - half, present_day + half], dtype=float)
            probe_progress = np.interp(probe_times, mapping_times, mapping_progress)

            multipliers = np.asarray(serial_uplift_for_schedule(
                probe_progress, automation_model,
                present_day_human_labor, present_day_inference_compute, self.params,
            ), dtype=float)
            if not np.all(np.isfinite(multipliers)) or np.any(multipliers <= 1.0):
                return None

            excess = multipliers - 1.0
            # d ln(excess)/dt across the window -> doubling time of the excess
            growth_per_year = float(np.log(excess[2] / excess[1]) / (2.0 * half))
            if not np.isfinite(growth_per_year) or growth_per_year <= 0:
                return None

            return {
                'serial_multiplier': float(multipliers[0]),
                'doubling_time_years': float(np.log(2.0) / growth_per_year),
                'eval_date': t_star,
            }
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Could not measure the derived schedule's uplift statistics: {e}")
            return None

    def _update_horizon_trajectory_anchor(self, new_anchor_progress: float):
        """
        Update the horizon trajectory function with a new anchor progress value.
        This fixes the Case 2 issue where the fitted anchor progress differs from
        the actual integrated progress at present_day.

        Note: progress_at_aa is deliberately NOT re-derived here — it already
        shaped the automation anchors used inside the integration, so updating
        it post hoc would require re-integrating to a fixed point.
        """
        observable = getattr(self, '_horizon_observable', None)
        if observable is None or observable.rebase_anchor is None:
            logger.warning("Cannot update horizon trajectory: no stored parameters")
            return

        observable = observable.rebase_anchor(new_anchor_progress)
        self._horizon_observable = observable
        self.horizon_trajectory = observable.predict
        logger.info(f"Updated horizon trajectory anchor progress to {new_anchor_progress:.6f}")

    def compute_metr_mse(self) -> Optional[float]:
        """
        Compute Mean Squared Error between model predictions and METR benchmark data.

        Returns MSE in log-space: MSE = mean((log(predicted_horizon) - log(actual_horizon))^2)
        Returns None if METR data cannot be loaded or horizon trajectory is not available.
        """
        # Check if horizon trajectory is available
        if not hasattr(self, 'horizon_trajectory') or self.horizon_trajectory is None:
            logger.warning("Cannot compute METR MSE: horizon trajectory not available")
            return None

        # Check if results are available
        if not hasattr(self, 'results') or not self.results:
            logger.warning("Cannot compute METR MSE: model results not available")
            return None

        # Load METR benchmark data from cache
        benchmark_data = _load_benchmark_data()
        if not benchmark_data:
            logger.warning("Cannot compute METR MSE: benchmark_results_1_1.yaml could not be loaded")
            return None

        # Get the date -> progress mapping the horizon trajectory was fitted
        # against (the uplift reconstruction, or the human-only run in the
        # degenerate no-uplift case), so the MSE evaluates the fitted curve in
        # the same progress coordinates it was fit on. Fall back to the
        # human-only trajectory only if compute_progress_trajectory has not
        # stored a mapping yet.
        if getattr(self, '_anchor_mapping', None) is not None:
            mapping_times, mapping_progress = self._anchor_mapping
        else:
            if not hasattr(self, 'human_only_results') or not self.human_only_results:
                logger.warning("Cannot compute METR MSE: no anchoring mapping or human-only results available")
                return None
            logger.warning("compute_metr_mse: anchoring mapping not available; falling back to human-only trajectory")
            mapping_times = self.human_only_results['times']
            mapping_progress = self.human_only_results['progress']

        # Extract (progress, horizon) pairs from METR data
        squared_errors = []

        for model_name, model_info in benchmark_data['results'].items():
            # Convert release date to decimal year
            release_date_obj = model_info['release_date']
            try:
                # Handle both string and date objects
                if isinstance(release_date_obj, str):
                    release_date = datetime.strptime(release_date_obj, '%Y-%m-%d').date()
                else:
                    release_date = release_date_obj

                decimal_year = release_date.year + (release_date.timetuple().tm_yday - 1) / 365.25
            except (ValueError, AttributeError) as e:
                logger.debug(f"Could not parse release date for {model_name}: {release_date_obj} ({e})")
                continue

            # Interpolate progress value at the release date using the anchoring mapping
            if decimal_year >= mapping_times.min() and decimal_year <= mapping_times.max():
                interpolated_progress = np.interp(decimal_year, mapping_times, mapping_progress)
            elif decimal_year < mapping_times.min():
                interpolated_progress = mapping_progress[0]
            else:
                interpolated_progress = mapping_progress[-1]

            # Extract p80_horizon_length from metrics
            if 'metrics' in model_info and 'p80_horizon_length' in model_info['metrics']:
                p80_data = model_info['metrics']['p80_horizon_length']
                actual_horizon = p80_data.get('estimate')

                if actual_horizon is not None and actual_horizon > 0:
                    # Get predicted horizon from model
                    try:
                        predicted_horizon = self.horizon_trajectory(interpolated_progress)

                        # Ensure predicted_horizon is valid
                        if np.isfinite(predicted_horizon) and predicted_horizon > 0:
                            # Compute squared error in log space
                            log_error = np.log(predicted_horizon) - np.log(actual_horizon)
                            squared_errors.append(log_error ** 2)
                        else:
                            logger.debug(f"Invalid predicted horizon for {model_name}: {predicted_horizon}")
                    except Exception as e:
                        if should_reraise(e):
                            raise
                        logger.debug(f"Error predicting horizon for {model_name}: {e}")
                        continue

        if len(squared_errors) == 0:
            logger.warning("Cannot compute METR MSE: no valid prediction-observation pairs")
            return None

        # Compute MSE
        mse = np.mean(squared_errors)
        logger.info(f"METR MSE (log-space): {mse:.6f} over {len(squared_errors)} data points")

        return mse

    def compute_human_only_trajectory(self, time_range: List[float], initial_progress: Optional[float] = None, use_ode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute human-only trajectory over specified time range.

        Args:
            time_range: [start_time, end_time]
            initial_progress: Initial logE value (default: TC(t0) - 1)
            use_ode: If True, use ODE solver for higher accuracy (needed for r_software calibration).
                     If False (default), use fast vectorized integration.
        """
        # logE init: legacy callers pass 0.0 ("baseline start"); treat like None.
        if initial_progress is None or initial_progress == 0.0:
            initial_progress = np.log10(max(float(self.data.training_H100e[0]), 1e-300) * cfg.FLOP_PER_H100E_YEAR) - 1.0

        # Create human-only params for initial conditions computation
        human_only_params = copy.deepcopy(self.params)
        human_only_params.human_only = True

        # Compute initial research stock (still need this for the starting point)
        initial_conditions = compute_initial_conditions(self.data, human_only_params, initial_progress)
        initial_research_stock_val = initial_conditions.research_stock
        logger.info(f"HUMAN-ONLY::: initial_research_stock_val: {initial_research_stock_val}")

        if use_ode:
            # Use ODE solver for higher accuracy (needed for r_software calibration)
            times, progress_values, research_stock_values = integrate_progress(
                time_range, initial_progress, initial_research_stock_val, self.data, human_only_params
            )
            # Compute derived metrics via loop
            research_efforts = []
            sw_progress_rates = []
            RS_0 = initial_research_stock_val
            for i, (t, p, rs) in enumerate(zip(times, progress_values, research_stock_values)):
                state = [p, rs]
                progress_rate, research_effort = progress_rate_at_time(t, state, self.data, human_only_params, RS_0=RS_0)
                research_efforts.append(research_effort)
                sw_progress_rate = compute_software_progress_rate(rs, research_effort, human_only_params.r_software)
                sw_progress_rates.append(sw_progress_rate)
            research_efforts = np.array(research_efforts)
            sw_progress_rates = np.array(sw_progress_rates)
        else:
            # Use fast vectorized integration (no ODE solver needed for human-only mode)
            times, progress_values, research_stock_values, research_efforts, sw_progress_rates = integrate_progress_human_only(
                time_range, initial_progress, initial_research_stock_val, self.data, human_only_params
            )

        # Compute rate metrics (vectorized). Two distinct quantities, both OOMs/yr:
        #   etsp_rates     = d(ETSP)/dt = sw + TSP growth (the legacy "progress_rates").
        #   progress_rates = d(progress)/dt = 10^(ETSP - logE)/ln10 (true derivative of logE).
        # etsp and progress_values are RAW; their difference is rebase-invariant.
        training_system_performance_growth_rate = self.data.get_training_system_performance_growth_rate(times)
        etsp_rates = sw_progress_rates + training_system_performance_growth_rate
        tc_abs = np.interp(times, self.data.time, self.data.log10_training_H100e) + np.log10(cfg.FLOP_PER_H100E_YEAR)
        se = human_only_params.r_software * np.log10(np.maximum(research_stock_values, 1e-30) / initial_research_stock_val)
        etsp = tc_abs + se
        progress_rates = np.power(10.0, etsp - progress_values) / np.log(10.0)

        # Anchor stats at params.present_day
        present_day = human_only_params.present_day
        present_day_progress = np.interp(present_day, times, progress_values)
        # anchor progress rate = TRUE d(progress)/dt: this is the OOM/yr exchange rate used to
        # convert the taste-slope (SD/progress-year -> SD/OOM), automation-efficiency slope, and
        # doubling times. Using the true rate (not the ETSP rate) is the #2 fix.
        present_day_progress_rate = np.interp(present_day, times, progress_rates)
        present_day_etsp_rate = np.interp(present_day, times, etsp_rates)
        reference_sw_progress_rate = np.interp(cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR, times, sw_progress_rates)
        present_day_sw_progress_rate = np.interp(present_day, times, sw_progress_rates)
        present_day_research_effort = _log_interp(present_day, times, np.array(research_efforts))
        present_day_research_stock = _log_interp(present_day, times, np.array(research_stock_values))
        # Interpolate human and AI labor at anchor time using log-space when positive
        if np.all(self.data.L_HUMAN > 0):
            present_day_human_labor = _log_interp(present_day, self.data.time, self.data.L_HUMAN)
        else:
            present_day_human_labor = np.interp(present_day, self.data.time, self.data.L_HUMAN)
        if np.all(self.data.inference_H100e > 0):
            present_day_inference_compute = _log_interp(present_day, self.data.time, self.data.inference_H100e)
        else:
            present_day_inference_compute = np.interp(present_day, self.data.time, self.data.inference_H100e)
        if np.all(self.data.experiment_H100e > 0):
            present_day_experiment_compute = _log_interp(present_day, self.data.time, self.data.experiment_H100e)
        else:
            present_day_experiment_compute = np.interp(present_day, self.data.time, self.data.experiment_H100e)
        self.human_only_results = {
            'times': times,
            'progress': progress_values,
            'research_stock': research_stock_values,
            'progress_rates': progress_rates,   # d(progress)/dt (true), OOMs/yr
            'etsp_rates': etsp_rates,            # d(ETSP)/dt = sw + TSP, OOMs/yr
            'research_efforts': research_efforts,
            'sw_progress_rates': sw_progress_rates,
            'reference_sw_progress_rate': reference_sw_progress_rate,
            'anchor_stats': {
                'progress': present_day_progress,
                'progress_rate': present_day_progress_rate,   # TRUE d(progress)/dt (drives slope conversions, #2 fix)
                'etsp_rate': present_day_etsp_rate,            # d(ETSP)/dt = sw + TSP
                'sw_progress_rate': present_day_sw_progress_rate,
                'experiment_H100e': present_day_experiment_compute,
                'human_labor': present_day_human_labor,
                'inference_H100e': present_day_inference_compute,
                'research_effort': present_day_research_effort,
                'research_stock': present_day_research_stock
            },
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'inference_H100e': self.data.inference_H100e,
                'experiment_H100e': self.data.experiment_H100e,
                'training_H100e': self.data.training_H100e,
                'training_system_performance_growth_rate': self.data.get_training_system_performance_growth_rate(self.data.time),
                # Optional display-only series (None when absent from the input CSV)
                'world_H100e': self.data.world_H100e,
                'total_ai_company_H100e': self.data.total_ai_company_H100e,
            }
        }

        return times, progress_values, research_stock_values
        
    def compute_progress_trajectory(self, time_range: List[float], initial_progress: Optional[float] = None, rebase_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute progress over specified time range with comprehensive metrics

        Args:
            time_range: [start_time, end_time]
            initial_progress: Initial progress (defaults to 0.0)
            rebase_output: If True (default), shift software_efficiency,
                training_compute and effective_compute to a reference-year
                baseline.  Set False to keep raw ODE coordinates.

        Returns:
            Tuple of (times, cumulative_progress_values, research_stock_values)
        """
        # In the cumulative-effective-compute formulation, initial_progress is the
        # logE init (log10 of cumulative effective compute). Legacy callers pass 0.0
        # to mean "baseline start"; logE == 0 (cumulative E == 1 FLOP) is never a
        # valid init, so treat 0.0 like None and auto-compute from training compute.
        if initial_progress is None or initial_progress == 0.0:
            # Compute logE_init from training compute at t0
            # Approximate steady-state: logE ≈ TC(t0) - 1 (ODE converges quickly)
            initial_progress = np.log10(max(float(self.data.training_H100e[0]), 1e-300) * cfg.FLOP_PER_H100E_YEAR) - 1.0
        _fn_start_time = time.perf_counter()
        _fn_start_cpu_time = time.process_time()
        _timing_wall: Dict[str, float] = {}
        _timing_cpu: Dict[str, float] = {}
        _timing_meta: Dict[str, Any] = {}
        # first, compute exp capacity params from asymptotes and anchors
        if not self.params.direct_input_exp_cap_ces_params:
            # Use present_day for reference instead of fixed REFERENCE_YEAR (2024.8)
            # Note: Uses human labor only, not accounting for automation
            present_day = self.params.present_day
            ref_exp_compute = _log_interp(present_day, self.data.time, self.data.experiment_H100e)
            ref_coding_labor = _log_interp(present_day, self.data.time, self.data.L_HUMAN)
            logger.info(f"ref_exp_compute: {ref_exp_compute}, ref_coding_labor: {ref_coding_labor}")
            rho, alpha_experiment_capacity, experiment_compute_exponent = compute_exp_capacity_params_from_anchors(
                self.params.inf_labor_asymptote,
                self.params.inf_compute_asymptote,
                (cfg.REFERENCE_COMPUTE_CHANGE, self.params.compute_anchor_exp_cap),
                (cfg.REFERENCE_LABOR_CHANGE, self.params.labor_anchor_exp_cap),
                ref_exp_compute, ref_coding_labor,
                self.params.parallel_penalty
            )
            self.params.rho_experiment_capacity = rho
            self.params.alpha_experiment_capacity = alpha_experiment_capacity
            self.params.experiment_compute_exponent = experiment_compute_exponent
            logger.info(f"computed exp capacity params: rho: {rho}, alpha: {alpha_experiment_capacity}, experiment_compute_exponent: {experiment_compute_exponent}")
        else:
            logger.info(f"using direct input exp capacity params: rho: {self.params.rho_experiment_capacity}, alpha: {self.params.alpha_experiment_capacity}, experiment_compute_exponent: {self.params.experiment_compute_exponent}")
        
        # hackily handle doubling_difficulty_growth_factor = 1 case (equivalent to decay_rate = 0)
        if self.params.doubling_difficulty_growth_factor == 1.0:
            logger.info(f"doubling_difficulty_growth_factor is 1.0 (decay_rate = 0), setting to exponential")
            self.params.horizon_extrapolation_type = "exponential"

        # next compute human-only trajectory
        # Need to do this for various reasons, e.g. to auto-fit the METR trajectory you need to the (effective compute, horizon) pairs (we assume no automation)
        # Also if we want to specify current doubling time or gap size in years rather than progress units, need to know how much EC was increased in present day
        # First, run a SHORT human-only trajectory just to get anchor stats for r_software calibration
        # Only need values at reference_year (2024) and present_day (~2025.6), so end 1 year after present_day
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        short_end_time = min(self.params.present_day + 1.0, time_range[1])
        short_time_range = [time_range[0], short_end_time]
        self.compute_human_only_trajectory(short_time_range, initial_progress)
        _timing_wall['human_only_trajectory_first'] = time.perf_counter() - _t0
        _timing_cpu['human_only_trajectory_first'] = time.process_time() - _c0

        # Then, scale r_software so that sw_progress_rate at anchor time is 1
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        logger.info(f"reference year: {cfg.SOFTWARE_PROGRESS_SCALE_REFERENCE_YEAR}")
        logger.info(f"reference sw_progress_rate: {self.human_only_results['reference_sw_progress_rate']}")
        logger.info(f"desired software_progress_rate_at_reference_year: {self.params.software_progress_rate_at_reference_year}")
        # software_progress_rate_at_reference_year is an OBSERVED rate, and the
        # world that produced it already had AI uplift in it, so bind the
        # RECONSTRUCTED (with-AI) reference-year rate rather than the human-only
        # counterfactual's. One rescale is exact here, not an iteration: the
        # reconstruction's research effort and stock are functions of the
        # exogenous uplift path alone, so its sw rate is exactly linear in
        # r_software.
        _calibration_ref_rate = self.human_only_results['reference_sw_progress_rate']
        _short_reconstruction = reconstruct_from_params(
            self.human_only_results, self.data, self.params)
        if _short_reconstruction is not None:
            _calibration_ref_rate = _short_reconstruction['reference_sw_progress_rate']
            logger.info(
                f"uplift reconstruction: calibrating r_software against the reconstructed "
                f"reference sw rate {_calibration_ref_rate:.6f} instead of the human-only "
                f"{self.human_only_results['reference_sw_progress_rate']:.6f}")
        else:
            # No uplift to reconstruct (see uplift_statistics): M(s) == 1, so the
            # reconstruction WOULD return the human-only arrays. Skipping it is
            # an efficiency guard, not a different rule.
            logger.warning(
                "uplift reconstruction: no usable uplift statistics; calibrating "
                "r_software against the human-only reference rate")
        self.params.r_software = self.params.software_progress_rate_at_reference_year * self.params.r_software/_calibration_ref_rate
        logger.info(f"new r_software: {self.params.r_software}")
        _timing_wall['human_only_rescale_r_software'] = time.perf_counter() - _t0
        _timing_cpu['human_only_rescale_r_software'] = time.process_time() - _c0

        # Finally, recompute the human-only trajectory with the new r_software
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        human_only_times, human_only_progress, _ = self.compute_human_only_trajectory(time_range, initial_progress)
        _timing_wall['human_only_trajectory_second'] = time.perf_counter() - _t0
        _timing_cpu['human_only_trajectory_second'] = time.process_time() - _c0
        logger.info(f"new reference sw_progress_rate: {self.human_only_results['reference_sw_progress_rate']}")
        _timing_wall['human_only_total'] = (
            _timing_wall['human_only_trajectory_first']
            + _timing_wall['human_only_rescale_r_software']
            + _timing_wall['human_only_trajectory_second']
        )
        _timing_cpu['human_only_total'] = (
            _timing_cpu['human_only_trajectory_first']
            + _timing_cpu['human_only_rescale_r_software']
            + _timing_cpu['human_only_trajectory_second']
        )
        logger.info(f"Timing: human-only trajectory computed in {_timing_wall['human_only_total']:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        
        # Derive every date-anchored quantity from ONE date -> progress mapping:
        # a one-shot reconstruction of the REALIZED (with-AI) trajectory, built
        # once from the human-only run that was just recomputed with the
        # recalibrated r_software. Nothing here iterates and nothing here
        # integrates the model ODE. See progress_model/uplift_reconstruction.py
        # for why anchoring against the human-only counterfactual is a bias.
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        _reconstruction = reconstruct_from_params(
            self.human_only_results, self.data, self.params)
        _timing_wall['uplift_reconstruction'] = time.perf_counter() - _t0
        _timing_cpu['uplift_reconstruction'] = time.process_time() - _c0
        if _reconstruction is None:
            # No uplift to reconstruct. M(s) == 1 makes the reconstruction the
            # human-only run bit-for-bit (integrate_progress_human_only with a
            # unit multiplier IS the human-only quadrature), so reading the
            # human-only arrays directly is the same answer for less work --
            # an efficiency guard, not a second rule.
            logger.warning(
                "uplift reconstruction: the parameters yield no usable uplift statistics "
                "(M == 1); the reconstruction is the human-only run, so anchoring reads it "
                "directly")
            _mapping_anchor_stats = self.human_only_results['anchor_stats']
            _mapping_times = human_only_times
            _mapping_progress = human_only_progress
            # d(progress)/dt on the same grid. Only ac_anchor_mode='uplift-solved'
            # reads it (rate(t*) for the excess-uplift doubling-time conversion),
            # but it travels with the mapping so the two can never disagree.
            _mapping_progress_rates = np.asarray(
                self.human_only_results['progress_rates'], dtype=float)
            self._uplift_reconstruction_info = {'available': False}
        else:
            _mapping_anchor_stats = _reconstruction['anchor_stats']
            _mapping_times = _reconstruction['times']
            _mapping_progress = _reconstruction['progress']
            _mapping_progress_rates = np.asarray(
                _reconstruction['progress_rates'], dtype=float)
            self._uplift_reconstruction = _reconstruction
            _ho_stats = self.human_only_results['anchor_stats']
            _stats_used = _reconstruction['statistics_used']
            self._uplift_reconstruction_info = {
                'available': True,
                    'statistics_used': dict(_stats_used),
                'human_only_present_day_progress': float(_ho_stats['progress']),
                'reconstructed_present_day_progress': float(_mapping_anchor_stats['progress']),
                'present_day_progress_shift': float(
                    _mapping_anchor_stats['progress'] - _ho_stats['progress']),
                'human_only_present_day_rate': float(_ho_stats['progress_rate']),
                'reconstructed_present_day_rate': float(_mapping_anchor_stats['progress_rate']),
                'human_only_reference_sw_rate': float(
                    self.human_only_results['reference_sw_progress_rate']),
                'reconstructed_reference_sw_rate': float(
                    _reconstruction['reference_sw_progress_rate']),
                'reference_sw_rate_ratio': float(
                    _reconstruction['reference_sw_progress_rate']
                    / max(self.human_only_results['reference_sw_progress_rate'], 1e-30)),
                'r_software': float(self.params.r_software),
            }
            logger.info(
                f"Uplift reconstruction: M={_stats_used['serial_multiplier']:.4f} at "
                f"{_stats_used['eval_date']:.4f}, (M-1) doubling time "
                f"{_stats_used['doubling_time_years']:.4f} yr; present-day progress "
                f"{_ho_stats['progress']:.6f} (human-only) -> "
                f"{_mapping_anchor_stats['progress']:.6f} (reconstructed, "
                f"{self._uplift_reconstruction_info['present_day_progress_shift']:+.6f} OOMs); "
                f"present-day rate {_ho_stats['progress_rate']:.6f} -> "
                f"{_mapping_anchor_stats['progress_rate']:.6f}; reference-year sw rate ratio "
                f"{self._uplift_reconstruction_info['reference_sw_rate_ratio']:.6f}")
        self._anchor_stats_used = _mapping_anchor_stats
        self._anchor_mapping = (_mapping_times, _mapping_progress)

        # Store the various facts about the anchor date, read off that mapping
        present_day = self.params.present_day
        present_day_progress = _mapping_anchor_stats['progress']
        present_day_progress_rate = _mapping_anchor_stats['progress_rate']
        present_day_research_stock = _mapping_anchor_stats['research_stock']
        # Baseline for the AI software R&D uplift metric: the HUMAN-ONLY run's
        # present-day research effort (no AI assistance), not the mapping
        # anchor's -- the uplift is measured against a company with no AIs.
        present_day_human_only_research_effort = float(
            self.human_only_results['anchor_stats']['research_effort'])
        present_day_human_labor = _mapping_anchor_stats['human_labor']
        present_day_inference_compute = _mapping_anchor_stats['inference_H100e']
        present_day_experiment_compute = _mapping_anchor_stats['experiment_H100e']
        logger.info(f"present_day_human_labor: {present_day_human_labor}, present_day_inference_compute: {present_day_inference_compute}")

        # estimate horizon trajectory from METR data using the anchoring mapping
        _t_horizon_est_start = time.perf_counter()
        _c_horizon_est_start = time.process_time()
        try:
            anchor_progress_rate = _mapping_anchor_stats['progress_rate']
            self.estimate_horizon_trajectory(_mapping_times, _mapping_progress, anchor_progress_rate,
                                             mapping_progress_rates=_mapping_progress_rates)
            self._compute_ac_anchor(_mapping_times, _mapping_progress, anchor_progress_rate,
                                    mapping_progress_rates=_mapping_progress_rates)
            _dt_horizon_est = time.perf_counter() - _t_horizon_est_start
            _dt_horizon_est_cpu = time.process_time() - _c_horizon_est_start
            _timing_wall['horizon_trajectory_estimation'] = _dt_horizon_est
            _timing_cpu['horizon_trajectory_estimation'] = _dt_horizon_est_cpu
            logger.info(f"Timing: horizon trajectory estimation completed in {_dt_horizon_est:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed to estimate horizon trajectory: {e}")
            _dt_horizon_est = time.perf_counter() - _t_horizon_est_start
            _dt_horizon_est_cpu = time.process_time() - _c_horizon_est_start
            _timing_wall['horizon_trajectory_estimation'] = _dt_horizon_est
            _timing_cpu['horizon_trajectory_estimation'] = _dt_horizon_est_cpu
            _timing_meta['horizon_trajectory_estimation_failed'] = True
            logger.info(f"Timing: horizon trajectory estimation failed after {_dt_horizon_est:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
            self.horizon_trajectory = None

        # Convert AI research taste slope if using "SDs per progress-year" mode
        # This must be done after computing anchor_progress_rate but before the main trajectory
        # Store original slope for display purposes
        # IDEMPOTENT BY FORM: capture the pre-conversion slope once, then always
        # convert FROM that stored original. The T_d correction pass re-derives the
        # anchored quantities against a rebuilt mapping, so this runs more than once
        # per model; converting from the live (already-converted) field instead would
        # divide by the anchor rate twice and silently scale the taste schedule by
        # 1/rate**2. Same pattern below for the automation-efficiency slope.
        if getattr(self, '_original_taste_slope', None) is None:
            self._original_taste_slope = self.params.ai_research_taste_slope
        if self.params.taste_schedule_type == "SDs per progress-year":
            try:
                # anchor_progress_rate is the TRUE d(progress)/dt = d(logE)/dt (OOMs of
                # cumulative progress per year), NOT the ETSP rate. The taste schedule's x-axis
                # is `progress` (= logE), so this is the correct OOM/yr exchange rate (#2 fix).
                anchor_progress_rate = _mapping_anchor_stats['progress_rate']
                if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0:
                    # Convert from SDs/progress-year to SDs/progress-unit by multiplying by (1 year / anchor_progress_rate progress-units)
                    # which simplifies to dividing by anchor_progress_rate
                    original_slope = float(self._original_taste_slope)
                    converted_slope = original_slope / anchor_progress_rate
                    logger.info(f"Converting taste slope from {original_slope:.6f} SDs/progress-year to {converted_slope:.6f} SDs/progress-unit (anchor rate: {anchor_progress_rate:.6f})")
                    self.params.ai_research_taste_slope = converted_slope
                else:
                    logger.warning(f"Invalid anchor_progress_rate for taste slope conversion: {anchor_progress_rate}")
            except Exception as e:
                if should_reraise(e):
                    raise
                logger.warning(f"Failed to convert taste slope for progress-year mode: {e}")

        # Convert coding automation efficiency slope to OOMs/progress-unit
        # This conversion is always applied (OOMs/progress-year → OOMs/OOM)
        if getattr(self, '_original_automation_efficiency_slope', None) is None:
            self._original_automation_efficiency_slope = self.params.coding_automation_efficiency_slope
        try:
            anchor_progress_rate = _mapping_anchor_stats['progress_rate']
            if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0:
                # Convert from OOMs/progress-year to OOMs/progress-unit by dividing by anchor_progress_rate
                original_eff_slope = float(self._original_automation_efficiency_slope)
                converted_eff_slope = original_eff_slope / anchor_progress_rate
                logger.info(f"Converting automation efficiency slope from {original_eff_slope:.6f} OOMs/progress-year to {converted_eff_slope:.6f} OOMs/progress-unit (anchor rate: {anchor_progress_rate:.6f})")
                self.params.coding_automation_efficiency_slope = converted_eff_slope
            else:
                logger.warning(f"Invalid anchor_progress_rate for automation efficiency slope conversion: {anchor_progress_rate}")
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed to convert automation efficiency slope: {e}")

        # Determine the automation schedule. Two routes:
        #  - 'uplift-solved': solve the logistic (k, x0) jointly against the
        #    model's OWN endogenous uplift level and doubling rate; progress_at_aa
        #    falls out as the A(p) = 1 crossing.
        #  - everything else: progress_at_aa is already fixed (from the AC-anchor
        #    observable), so only the lower anchor needs solving.
        _t_anchor_solver_start = time.perf_counter()
        _c_anchor_solver_start = time.process_time()
        self._derive_automation_schedule(
            _mapping_times, _mapping_progress, _mapping_progress_rates,
            anchor_progress_rate, float(present_day_progress),
            float(present_day_human_labor), float(present_day_inference_compute))
        _dt_anchor_solver = time.perf_counter() - _t_anchor_solver_start
        _timing_wall['automation_anchor_solver'] = _dt_anchor_solver
        _timing_cpu['automation_anchor_solver'] = time.process_time() - _c_anchor_solver_start
        _timing_wall['automation_model_init'] = 0.0
        _timing_cpu['automation_model_init'] = 0.0
        logger.info(f"Timing: automation schedule derived in {_dt_anchor_solver:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")

        # T_d correction pass. Outside 'uplift-solved' the belief the reconstruction was
        # built from is only a guess, so re-anchor against what the derived schedule
        # actually produces; in 'uplift-solved' the solve already pinned it. Recorded in
        # results['td_correction']. See _apply_td_correction_pass.
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        _corrected_mapping = self._apply_td_correction_pass(
            {'times': _mapping_times, 'progress': _mapping_progress,
             'progress_rates': _mapping_progress_rates,
             'anchor_stats': _mapping_anchor_stats},
            float(present_day_progress),
            float(present_day_human_labor), float(present_day_inference_compute))
        if self._td_correction_info.get('applied'):
            _mapping_times = _corrected_mapping['times']
            _mapping_progress = _corrected_mapping['progress']
            _mapping_progress_rates = np.asarray(_corrected_mapping['progress_rates'], dtype=float)
            _mapping_anchor_stats = _corrected_mapping['anchor_stats']
            self._anchor_stats_used = _mapping_anchor_stats
            self._anchor_mapping = (_mapping_times, _mapping_progress)
            anchor_progress_rate = _mapping_anchor_stats['progress_rate']
            present_day_progress = np.interp(float(self.params.present_day),
                                             _mapping_times, _mapping_progress)
        _timing_wall['td_correction_pass'] = time.perf_counter() - _t0
        _timing_cpu['td_correction_pass'] = time.process_time() - _c0

        # Use utility function to compute initial conditions with the correct parameters
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        initial_conditions = compute_initial_conditions(self.data, self.params, initial_progress)
        _timing_wall['initial_conditions'] = time.perf_counter() - _t0
        _timing_cpu['initial_conditions'] = time.process_time() - _c0
        initial_research_effort_val = initial_conditions.research_effort
        initial_research_stock_val = initial_conditions.research_stock
        logger.info(f"ACTUAL::: initial_research_stock_val: {initial_research_stock_val}, initial_research_effort_val: {initial_research_effort_val}")

        # Below gives you at each time what is the effectige compute value and what is the research stock value. It runs the whole model.
        # With just time -> effective compute and research stock, you can compute all the other metrics.
        _t_integrate_start = time.perf_counter()
        _c_integrate_start = time.process_time()
        times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, initial_research_stock_val, self.data, self.params)
        _dt_integrate = time.perf_counter() - _t_integrate_start
        _dt_integrate_cpu = time.process_time() - _c_integrate_start
        _timing_wall['integrate_progress'] = _dt_integrate
        _timing_cpu['integrate_progress'] = _dt_integrate_cpu
        logger.info(f"Timing: integrate_progress completed in {_dt_integrate:.3f}s (elapsed {time.perf_counter() - _fn_start_time:.3f}s)")
        
        # SIE MODE
        takeoff_start_human_only_stats = None
        if self.params.sos_mode:
            # Find the time when the selected SOS milestone is reached
            sos_milestone = getattr(self.params, 'sos_start_milestone', 'AC')
            full_automation_time = self.find_sos_milestone_time(
                times, progress_values, research_stock_values, sos_milestone
            )

            if full_automation_time is not None:
                takeoff_start_human_only_stats = {
                    'time': full_automation_time,
                    'human_labor': _log_interp(full_automation_time, self.data.time, self.data.L_HUMAN),
                    'inference_H100e': _log_interp(full_automation_time, self.data.time, self.data.inference_H100e),
                    'experiment_H100e': _log_interp(full_automation_time, self.data.time, self.data.experiment_H100e),
                    'research_stock': _log_interp(full_automation_time, np.array(times), np.array(self.human_only_results['research_stock'])),
                    'research_effort': _log_interp(full_automation_time, np.array(times), np.array(self.human_only_results['research_efforts'])),
                }
                logger.info(f"SIE mode freeze time ({sos_milestone}): {full_automation_time}")
                self.freeze_time_series_at_time(full_automation_time)
                times, progress_values, research_stock_values = integrate_progress(time_range, initial_progress, initial_research_stock_val, self.data, self.params)
                takeoff_start_research_stock = _log_interp(full_automation_time, times, research_stock_values)
                self.human_only_results['takeoff_start_stats'] = takeoff_start_human_only_stats
            else:
                logger.info(f"SIE mode enabled but milestone {sos_milestone} not reached")
        # How much self-consistency defect the anchoring left behind: the
        # realized present-day progress against the p0 the anchors were built
        # from. Recorded rather than asserted -- the reconstruction is an
        # approximation, so the horizon/observable rebase below stays a real
        # (if much smaller) correction. This is the headline number for "did
        # the reconstruction buy anything".
        if self._uplift_reconstruction_info.get('available'):
            _realized_p0 = float(np.interp(self.params.present_day, times, progress_values))
            _ho_p0 = float(self._uplift_reconstruction_info['human_only_present_day_progress'])
            self._uplift_reconstruction_info['realized_present_day_progress'] = _realized_p0
            self._uplift_reconstruction_info['present_day_progress_residual'] = (
                _realized_p0 - float(present_day_progress))
            self._uplift_reconstruction_info['human_only_present_day_residual'] = (
                _realized_p0 - _ho_p0)
            logger.info(
                f"Uplift reconstruction: realized present-day progress {_realized_p0:.6f} vs "
                f"anchored {float(present_day_progress):.6f} "
                f"(residual {self._uplift_reconstruction_info['present_day_progress_residual']:+.6f} "
                f"OOMs; the human-only mapping would have left "
                f"{self._uplift_reconstruction_info['human_only_present_day_residual']:+.6f})")

        # Post-integration rebase: re-anchor each displayed observable onto the REALIZED
        # trajectory, correcting the Case 2 anchor blowup where the fitted anchor progress
        # differs from what the integration actually produced.
        #
        # Each observable is re-anchored at ITS OWN elicitation vintage, not at present_day.
        # A horizon elicited at 2026.269 has to land back on realized(2026.269): rebasing it
        # at present_day would slide the anchor by whatever progress accrued in between
        # (~0.4-0.5 OOM for the Q2 blocks), tilting the drawn trend and corrupting
        # compute_metr_mse. The fitting side resolves the same date via
        # resolve_observable_anchor, so the two stay in agreement.
        #
        # progress_at_aa is deliberately NOT re-derived here -- it already shaped the
        # automation anchors used inside the integration (see
        # _update_horizon_trajectory_anchor).
        actual_present_day_progress = np.interp(self.params.present_day, times, progress_values)

        def _realized_anchor_progress(metric_key: str) -> float:
            """Realized progress at this observable's own reference date.

            Undated observables get `actual_present_day_progress` verbatim rather than a
            fresh interpolation at the same x, so the default path is bit-identical to the
            single-date behaviour this replaced.
            """
            t_obs = resolve_reference_date(self.params, metric_key)
            if t_obs == float(self.params.present_day):
                return actual_present_day_progress
            return np.interp(t_obs, times, progress_values)

        if (self.horizon_trajectory is not None and
            hasattr(self, '_horizon_uses_shifted_form') and self._horizon_uses_shifted_form and
            self.params.present_horizon is not None):

            _horizon_anchor = _realized_anchor_progress('horizon_lengths')
            logger.info(
                f"Rebasing horizon anchor at {resolve_reference_date(self.params, 'horizon_lengths'):.6f}: "
                f"fitted progress {present_day_progress:.6f} -> realized {_horizon_anchor:.6f}")
            self._update_horizon_trajectory_anchor(_horizon_anchor)

        # Same correction for the other anchored observables (revenue/ECI trends), each at
        # its own vintage. _ac_anchor_observable is an alias of one of these (or None), so
        # it is re-pointed at the rebased copy to keep the identity dedup working.
        for _attr in ('_eci_observable', '_revenue_observable'):
            _obs = getattr(self, _attr)
            if _obs is not None and _obs.rebase_anchor is not None:
                _rebased = _obs.rebase_anchor(_realized_anchor_progress(_obs.metric_key))
                if self._ac_anchor_observable is _obs:
                    self._ac_anchor_observable = _rebased
                setattr(self, _attr, _rebased)

        # Calculate all metrics in a single pass to avoid redundancy
        logger.info("Computing metrics")
        metrics, timing_wall_cpu, timing_meta_metrics = compute_metrics_loop(
            times=times,
            progress_values=progress_values,
            research_stock_values=research_stock_values,
            data=self.data,
            params=self.params,
            initial_progress=initial_progress,
            present_day_human_labor=present_day_human_labor,
            present_day_inference_compute=present_day_inference_compute,
            present_day_experiment_compute=present_day_experiment_compute,
            present_day_research_stock=present_day_research_stock,
            present_day_human_only_research_effort=present_day_human_only_research_effort,
            taste_distribution=self.taste_distribution,
            horizon_trajectory=self.horizon_trajectory,
            takeoff_start_human_only_stats=takeoff_start_human_only_stats,
            RS_0=initial_research_stock_val,
            extra_observables=self.extra_observables(),
        )

        # Extract metrics from the returned dictionary
        progress_rates = metrics['progress_rates']   # d(progress)/dt (true), OOMs/yr
        etsp_rates = metrics['etsp_rates']           # d(ETSP)/dt = sw + TSP, OOMs/yr
        research_efforts = metrics['research_efforts']
        automation_fractions = metrics['automation_fractions']
        ai_research_tastes = metrics['ai_research_tastes']
        ai_research_taste_sds = metrics['ai_research_taste_sds']
        ai_research_taste_quantiles = metrics['ai_research_taste_quantiles']
        aggregate_research_tastes = metrics['aggregate_research_tastes']
        coding_labors = metrics['coding_labors']
        coding_labors_with_present_resources = metrics['coding_labors_with_present_resources']
        serial_coding_labors = metrics['serial_coding_labors']
        software_progress_rates = metrics['software_progress_rates']
        software_progress_rates_present_resources = metrics['software_progress_rates_present_resources']
        software_efficiency = metrics['software_efficiency']
        human_only_research_efforts = metrics['human_only_research_efforts']
        human_only_software_progress_rates = metrics['human_only_software_progress_rates']
        human_only_progress_rates = metrics['human_only_progress_rates']
        ai_labor_contributions = metrics['ai_labor_contributions']
        human_labor_contributions = metrics['human_labor_contributions']
        ai_coding_labor_multipliers = metrics['ai_coding_labor_multipliers']
        ai_coding_labor_mult_ref_present_day = metrics['ai_coding_labor_mult_ref_present_day']
        excess_serial_coding_labor_uplift_ref_present_day = metrics['excess_serial_coding_labor_uplift_ref_present_day']
        serial_coding_labor_multipliers = metrics['serial_coding_labor_multipliers']
        ai_sw_progress_mult_ref_present_day = metrics['ai_sw_progress_mult_ref_present_day']
        takeoff_progress_multipliers = metrics['takeoff_progress_multipliers']
        discounted_exp_compute = metrics['discounted_exp_compute']
        horizon_lengths = metrics['horizon_lengths']
        effective_compute = metrics['effective_compute']
        training_compute = metrics['training_compute']
        training_system_performance = metrics['training_system_performance']
        # capability_ecis comes from the ECI extra-observable (absent if its fit failed)
        capability_ecis = np.asarray(metrics.get('capability_ecis', np.zeros(len(times))), dtype=float)
        # d(ECI)/dt in points/yr, analytic from the same linear trend (see metrics_computation)
        capability_eci_rates = np.asarray(
            metrics.get('capability_eci_rates', np.zeros(len(times))), dtype=float)
        experiment_capacity = metrics['experiment_capacity']
        exp_cap_mult_with_infinite_labor = metrics['exp_cap_mult_with_infinite_labor']
        exp_cap_mult_with_infinite_compute = metrics['exp_cap_mult_with_infinite_compute']
        ai_only_coding_labors = metrics['ai_only_coding_labors']
        mrts_exp_compute_human_labor = metrics['mrts_exp_compute_human_labor']
        training_system_performance_growth_rate = metrics['training_system_performance_growth_rate']
        optimal_training_run_length = metrics['optimal_training_run_length']

        # Update timing information
        _timing_wall['metrics_loop'] = timing_wall_cpu['wall']
        _timing_cpu['metrics_loop'] = timing_wall_cpu['cpu']
        _timing_meta.update(timing_meta_metrics)
        # Calculate time when superhuman coder level is reached
        aa_time = None
        if self.params.progress_at_aa is not None:
            # Find the time when progress reaches progress_at_aa
            sc_progress_target = self.params.progress_at_aa
            
            # Check if SC is reached within the trajectory
            if progress_values[-1] >= sc_progress_target:
                # Find the exact time by interpolation
                if progress_values[0] >= sc_progress_target:
                    # SC level already reached at start
                    aa_time = times[0]
                else:
                    # Interpolate to find when progress crosses sc_progress_target
                    try:
                        aa_time = np.interp(sc_progress_target, progress_values, times)
                    except Exception as e:
                        if should_reraise(e):
                            raise
                        logger.warning(f"Error interpolating SC time: {e}")
                        aa_time = None

                logger.info(f"AC level ({sc_progress_target:.3f}) reached at time {aa_time:.3f}")
            else:
                logger.info(f"AC level ({sc_progress_target:.3f}) not reached within trajectory (final progress: {progress_values[-1]:.3f})")
        
        # Calculate software progress multiplier at SC
        if aa_time is not None:
            self.sc_sw_multiplier = _log_interp(aa_time, times, np.asarray(ai_sw_progress_mult_ref_present_day, dtype=float))
        else:
            self.sc_sw_multiplier = None
        
        # Compute the time when ai_coding_labor_mult_ref_present_day first reaches the required threshold
        # using exponential (log-space) interpolation between adjacent samples.
        ai2027_sc_time = None
        try:
            ai2027_sc_required_mult = (30 ** (1 / self.params.parallel_penalty)) * 30 * cfg.PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC
            ai2027_sc_time = _find_exponential_crossing_time(
                np.asarray(times, dtype=float),
                np.asarray(ai_coding_labor_mult_ref_present_day, dtype=float),
                float(ai2027_sc_required_mult),
            )
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Error computing ai2027_sc_time: {e}")

        # Calculate progress rate at anchor time
        present_day = self.params.present_day
        anchor_progress_rate = None
        if present_day is not None:
            # Check if anchor time is within our trajectory
            if times[0] <= present_day <= times[-1]:
                anchor_progress_rate = np.interp(present_day, times, progress_rates)
                logger.info(f"Progress rate at anchor time ({present_day:.3f}): {anchor_progress_rate:.6f}")
            else:
                logger.warning(f"Anchor time {present_day:.3f} is outside trajectory range [{times[0]:.3f}, {times[-1]:.3f}]")

        # Compute instantaneous doubling time at the anchor (years)
        instantaneous_anchor_doubling_time_years = None
        try:
            if (self.horizon_trajectory is not None and
                anchor_progress_rate is not None and np.isfinite(anchor_progress_rate) and anchor_progress_rate > 0):
                anchor_progress_value = self._anchor_stats_used['progress']
                progress = float(anchor_progress_value)
                # Numerical derivative of ln(horizon) with respect to progress at anchor
                eps = 1e-6 * max(1.0, abs(progress))
                if eps == 0:
                    eps = 1e-6
                H_p = self.horizon_trajectory(progress)
                H_p_eps = self.horizon_trajectory(progress + eps)
                if (np.isfinite(H_p) and np.isfinite(H_p_eps) and H_p > 0 and H_p_eps > 0):
                    dlnH_dprogress = (np.log(H_p_eps) - np.log(H_p)) / eps
                    if np.isfinite(dlnH_dprogress) and dlnH_dprogress > 0:
                        instantaneous_anchor_doubling_time_years = float(np.log(2) / (dlnH_dprogress * anchor_progress_rate))
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing instantaneous doubling time at anchor: {e}")

        # Compute AI research taste slope in SD per anchor-progress-year (SD/year at anchor)
        ai_taste_slope_per_anchor_progress_year = None
        ai_taste_slope_per_effective_oom = None
        try:
            if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate):
                if self.params.taste_schedule_type == "SDs per progress-year":
                    # For progress-year mode, use original input value for progress-year display
                    ai_taste_slope_per_anchor_progress_year = float(self._original_taste_slope)
                    # And compute effective OOM value using converted slope
                    ai_taste_slope_per_effective_oom = float(self.params.ai_research_taste_slope)
                else:
                    # For effective OOM mode, compute progress-year display from effective OOM input
                    ai_taste_slope_per_effective_oom = float(self.params.ai_research_taste_slope)
                    ai_taste_slope_per_anchor_progress_year = float(self.params.ai_research_taste_slope) * float(anchor_progress_rate)
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing taste slope conversions: {e}")

        # Compute automation efficiency slope in OOMs per anchor-progress-year (OOMs/year at anchor)
        automation_efficiency_slope_per_anchor_progress_year = None
        automation_efficiency_slope_per_effective_oom = None
        try:
            if anchor_progress_rate is not None and np.isfinite(anchor_progress_rate):
                # Always show original input (which is in OOMs/progress-year)
                automation_efficiency_slope_per_anchor_progress_year = float(self._original_automation_efficiency_slope)
                # And the converted value (in OOMs/OOM)
                automation_efficiency_slope_per_effective_oom = float(self.params.coding_automation_efficiency_slope)
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing automation efficiency slope conversions: {e}")

        # Compute top taste percentile metrics for display
        top_taste_percentile = self.params.top_percentile
        top_taste_value = None
        top_taste_num_sds = None
        f_multiplier_per_sd = None
        slope_times_log_f = None
        m_over_beta = None
        sie_uplift_doubling_ratio = None

        try:
            # Get taste value at top percentile (e.g., 99.9th percentile if top_percentile = 0.999)
            top_taste_value = self.taste_distribution.get_taste_at_quantile(top_taste_percentile)
            
            # Get how many standard deviations this represents
            top_taste_num_sds = self.taste_distribution.get_sd_of_taste(top_taste_value)
            
            # Compute f: multiplier per standard deviation
            # f = median_to_top_taste_multiplier^(1/num_sds)
            if top_taste_num_sds is not None and np.isfinite(top_taste_num_sds) and top_taste_num_sds > 0:
                f_multiplier_per_sd = self.params.median_to_top_taste_multiplier ** (1.0 / top_taste_num_sds)
                
                # Compute s * log10(f), where s is ai_research_taste_slope (SDs per effective OOM)
                if (ai_taste_slope_per_effective_oom is not None and
                    np.isfinite(ai_taste_slope_per_effective_oom) and
                    f_multiplier_per_sd > 0):
                    slope_times_log_f = ai_taste_slope_per_effective_oom * np.log10(f_multiplier_per_sd)

            # Compute m_over_beta (m/β) and sie_uplift_doubling_ratio
            # m = slope_times_log_f, β = beta_software = 1/r_software
            beta_software_local = 1.0 / self.params.r_software if self.params.r_software != 0 else None
            if (slope_times_log_f is not None and beta_software_local is not None and
                    np.isfinite(slope_times_log_f) and np.isfinite(beta_software_local) and
                    beta_software_local > 0):
                m_over_beta = slope_times_log_f / beta_software_local
                # sie_uplift_doubling_ratio = 2^((β/m) - 1)
                if slope_times_log_f > 0:
                    sie_uplift_doubling_ratio = 2 ** ((beta_software_local / slope_times_log_f) - 1)
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing top taste metrics: {e}")

        # REBASE SOFTWARE EFFICIENCY to 0 at the reference year (SE is defined relative to the
        # reference-year algorithms). This is the ONLY rebasing of the compute quantities:
        # training_compute (stock), training_system_performance (rate), and effective_compute are
        # all left in their true absolute units (log10 FLOP / log10 FLOP/yr) now that the input is
        # the actual H100e in the frontier run. Capability (ECI) needs no rebasing: it is
        # anchored at present_day by construction (see ac_anchor.fit_eci_trend) and depends
        # only on progress differences, so it is invariant to this shift.
        if rebase_output:
            # SE is defined relative to the reference-year algorithms: subtract its value at the
            # reference year, so SE(ref) == 0 and therefore ETSP == TSP at the reference date --
            # i.e. the instantaneous effective-compute RATE equals raw hardware FLOP/yr there.
            # effective_compute (logE) was integrated with SE measured relative to the t0
            # research-stock baseline, so shift it by the SAME amount to keep it consistent with
            # the rebased SE (effective compute in *reference-year* terms).
            #
            # This is a statement about RATES only. The cumulative series do NOT coincide at the
            # reference year: logE and log(training_compute) are integrals of 10**(TSP+SE) and
            # 10**TSP, so logE - log(stock) != SE pointwise, and a single scalar shift cannot make
            # them meet: they sit a fraction of an OOM apart at the reference year, which is why
            # the two chart lines cross some months later rather than there. The size of the gap
            # depends on the parameters and the input series, so it is not quoted here.
            #
            # It is NOT the old "+ TRAINING_COMPUTE_REFERENCE_OOMS". Display-only either way:
            # shifting the reference year leaves progress, rates and every milestone time
            # bit-identical and moves only effective_compute and software_efficiency, together.
            se_at_ref = np.interp(cfg.TRAINING_COMPUTE_REFERENCE_YEAR, times, software_efficiency)
            software_efficiency = software_efficiency - se_at_ref
            effective_compute = effective_compute - se_at_ref

        # Compute doubling times for ai_sw_progress_mult_ref_present_day
        # Doubling time = log(2) / (d log(value) / dt)
        ai_sw_uplift_doubling_times = None
        ai_sw_uplift_doubling_time_of_doubling_times = None
        try:
            sw_uplift = np.asarray(ai_sw_progress_mult_ref_present_day, dtype=float)
            times_arr = np.asarray(times, dtype=float)
            if len(sw_uplift) > 1 and np.all(sw_uplift > 0):
                # Compute log of the uplift values
                log_sw_uplift = np.log(sw_uplift)
                # Compute derivative of log values using central differences
                # d(log(y))/dt at each point
                d_log_dt = np.gradient(log_sw_uplift, times_arr)
                # Doubling time = log(2) / (d log(y) / dt)
                # Handle division by zero and negative growth rates
                with np.errstate(divide='ignore', invalid='ignore'):
                    doubling_times = np.log(2) / d_log_dt
                    # Replace inf/nan with a large number or None-like indicator
                    doubling_times = np.where(np.isfinite(doubling_times) & (doubling_times > 0),
                                              doubling_times, np.nan)
                ai_sw_uplift_doubling_times = doubling_times.tolist()

                # Compute doubling time of the doubling times
                # This measures how fast the doubling times are shrinking
                # Filter to finite positive doubling times
                valid_mask = np.isfinite(doubling_times) & (doubling_times > 0)
                if np.sum(valid_mask) > 2:
                    valid_times = times_arr[valid_mask]
                    valid_dt = doubling_times[valid_mask]
                    log_valid_dt = np.log(valid_dt)
                    d_log_dt_dt = np.gradient(log_valid_dt, valid_times)
                    # Compute the median "halving time" of the doubling times
                    # (negative because doubling times should be decreasing)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        halving_times = -np.log(2) / d_log_dt_dt
                    finite_halving = halving_times[np.isfinite(halving_times) & (halving_times > 0)]
                    if len(finite_halving) > 0:
                        ai_sw_uplift_doubling_time_of_doubling_times = float(np.median(finite_halving))
        except Exception as e:
            if should_reraise(e):
                raise
            logger.warning(f"Failed computing ai_sw_uplift doubling times: {e}")

        _t_results_assembly_start = time.perf_counter()
        _c_results_assembly_start = time.process_time()
        # Store comprehensive results
        self.results = {
            'times': times,
            'progress': progress_values,
            'research_stock': research_stock_values,
            'automation_fraction': automation_fractions,
            'ai_research_taste': ai_research_tastes,
            'ai_research_taste_sd': ai_research_taste_sds,
            'ai_research_taste_quantile': ai_research_taste_quantiles,
            'aggregate_research_taste': aggregate_research_tastes,
            'progress_rates': progress_rates,   # d(progress)/dt (true), OOMs/yr
            'etsp_rates': etsp_rates,            # d(ETSP)/dt = sw + TSP, OOMs/yr
            'research_efforts': research_efforts,
            'coding_labors': coding_labors,
            'serial_coding_labors': serial_coding_labors,
            'coding_labors_with_present_resources': coding_labors_with_present_resources,
            'software_progress_rates': software_progress_rates,
            'software_efficiency': software_efficiency,
            'human_only_progress_rates': human_only_progress_rates,
            'ai_labor_contributions': ai_labor_contributions,
            'human_labor_contributions': human_labor_contributions,
            'ai_coding_labor_multipliers': ai_coding_labor_multipliers,
            'ai_coding_labor_mult_ref_present_day': ai_coding_labor_mult_ref_present_day,
            'excess_serial_coding_labor_uplift_ref_present_day': excess_serial_coding_labor_uplift_ref_present_day,
            'serial_coding_labor_multipliers': serial_coding_labor_multipliers,
            'ai_sw_progress_mult_ref_present_day': ai_sw_progress_mult_ref_present_day,
            'takeoff_progress_multipliers': takeoff_progress_multipliers,
            'discounted_exp_compute': discounted_exp_compute,
            'horizon_lengths': horizon_lengths,
            'effective_compute': effective_compute,
            'capability_ecis': capability_ecis,
            'capability_eci_rates': capability_eci_rates,   # d(ECI)/dt, points/yr
            'training_compute': training_compute,
            'training_system_performance': training_system_performance,
            'experiment_capacity': experiment_capacity,
            'aa_time': aa_time,  # Time when superhuman coder level is reached
            'uplift_solve': getattr(self, '_uplift_solve_public', None),
            'td_correction': self._td_correction_public(),  # which (M, T_d) the anchoring actually used
            'sc_progress_level': self.params.progress_at_aa,  # Progress level for SC
            'sc_sw_multiplier': self.sc_sw_multiplier,  # Software progress multiplier at SC
            'ai2027_sc_time': ai2027_sc_time,  # Time when @AI2027 SC condition is met
            'present_day': present_day,  # Anchor time for manual horizon fitting
            'anchor_progress_rate': anchor_progress_rate,  # Progress rate at anchor time
            'instantaneous_anchor_doubling_time_years': instantaneous_anchor_doubling_time_years,  # Instantaneous doubling time of horizon at anchor (years)
            'ai_research_taste_slope_per_anchor_progress_year': ai_taste_slope_per_anchor_progress_year,  # SD per anchor-progress-year
            'ai_research_taste_slope_per_effective_oom': ai_taste_slope_per_effective_oom,  # SD per effective OOM
            'automation_efficiency_slope_per_anchor_progress_year': automation_efficiency_slope_per_anchor_progress_year,  # OOMs per anchor-progress-year
            'automation_efficiency_slope_per_effective_oom': automation_efficiency_slope_per_effective_oom,  # OOMs per effective OOM
            'input_time_series': {
                'time': self.data.time,
                'L_HUMAN': self.data.L_HUMAN,
                'inference_H100e': self.data.inference_H100e,
                'experiment_H100e': self.data.experiment_H100e,
                'training_H100e': self.data.training_H100e,
                'training_system_performance_growth_rate': self.data.get_training_system_performance_growth_rate(self.data.time),
                # Optional display-only series (None when absent from the input CSV)
                'world_H100e': self.data.world_H100e,
                'total_ai_company_H100e': self.data.total_ai_company_H100e,
            },
            'exp_capacity_params': {
                'rho': self.params.rho_experiment_capacity,
                'alpha': self.params.alpha_experiment_capacity,
                'experiment_compute_exponent': self.params.experiment_compute_exponent,
            },
            'r_software': self.params.r_software,  # Calibrated r_software value
            'beta_software': 1.0 / self.params.r_software if self.params.r_software != 0 else None,  # Beta (inverse of r_software)
            'top_taste_percentile': top_taste_percentile,  # Top percentile (e.g., 0.01 for 99th percentile)
            'top_taste_num_sds': top_taste_num_sds,  # Number of SDs the top percentile represents
            'f_multiplier_per_sd': f_multiplier_per_sd,  # Multiplier per standard deviation
            'slope_times_log_f': slope_times_log_f,  # s * log10(f), where s is SDs per effective OOM
            'm_over_beta': m_over_beta,  # m/β ratio (>1 means superexponential growth from taste alone)
            'sie_uplift_doubling_ratio': sie_uplift_doubling_ratio,  # 2^((β/m) - 1), approx ratio of successive doubling times
            'ai_sw_uplift_doubling_times': ai_sw_uplift_doubling_times,  # Doubling times for ai_sw_progress_mult_ref_present_day
            'ai_sw_uplift_doubling_time_of_doubling_times': ai_sw_uplift_doubling_time_of_doubling_times,  # Median halving time of doubling times
            'exp_cap_mult_with_infinite_labor': exp_cap_mult_with_infinite_labor,
            'exp_cap_mult_with_infinite_compute': exp_cap_mult_with_infinite_compute,
            'ai_only_coding_labors': ai_only_coding_labors,
            'mrts_exp_compute_human_labor': mrts_exp_compute_human_labor,
            'training_system_performance_growth_rate': training_system_performance_growth_rate,
            'optimal_training_run_length': optimal_training_run_length,
        }
        # Pass through metric series from the extra observables (e.g.
        # 'predicted_revenue'; 'capability_ecis' is already assembled above)
        for _obs in (self.extra_observables() or []):
            if _obs.metric_key in metrics and _obs.metric_key not in self.results:
                self.results[_obs.metric_key] = metrics[_obs.metric_key]
        _timing_wall['results_assembly'] = time.perf_counter() - _t_results_assembly_start
        _timing_cpu['results_assembly'] = time.process_time() - _c_results_assembly_start

        _t0 = time.perf_counter()
        _c0 = time.process_time()
        self.results['milestones'] = self.compute_milestones()
        _timing_wall['milestones'] = time.perf_counter() - _t0
        _timing_cpu['milestones'] = time.process_time() - _c0

        # Compute METR MSE
        _t0 = time.perf_counter()
        _c0 = time.process_time()
        metr_mse = self.compute_metr_mse()
        self.results['metr_mse'] = metr_mse
        _timing_wall['metr_mse'] = time.perf_counter() - _t0
        _timing_cpu['metr_mse'] = time.process_time() - _c0

        self.results['timing'] = {
            'wall_time_seconds_total': float(time.perf_counter() - _fn_start_time),
            'cpu_time_seconds_total': float(time.process_time() - _fn_start_cpu_time),
            'wall_time_breakdown_seconds': _timing_wall,
            'cpu_time_breakdown_seconds': _timing_cpu,
            'meta': _timing_meta,
        }

        # logger.info(f"Computed trajectory from {time_range[0]} to {time_range[1]}")
        # logger.info(f"Progress: {progress_values[0]:.3f} -> {progress_values[-1]:.3f}")
        # logger.info(f"Research Stock: {research_stock_values[0]:.3f} -> {research_stock_values[-1]:.3f}")
        # logger.info(f"Automation: {automation_fractions[0]:.3f} -> {automation_fractions[-1]:.3f}")
        return times, progress_values, research_stock_values
    
    def compute_milestones(self):
        """
        Compute milestones for the model.
        """
        # First, determine if SAR experiment selection would be achieved before AC
        # This affects how we calculate TED-AI and ASI targets
        ac_time = self.results.get('aa_time')
        sar_exp_selection_time_unconstrained = None
        sd_at_ac = None

        if ac_time is not None and 'ai_research_taste_sd' in self.results:
            # Find when SAR experiment selection skill would be achieved (ignoring AC requirement)
            sar_exp_selection_time_unconstrained = _find_exponential_crossing_time(
                np.asarray(self.results['times'], dtype=float),
                np.asarray(self.results['ai_research_taste_sd'], dtype=float),
                float(cfg.TOP_RESEARCHER_SD)
            )

            # Get the SD level at AC time
            sd_at_ac = _log_interp(
                ac_time,
                self.results['times'],
                np.asarray(self.results['ai_research_taste_sd'], dtype=float)
            )

        # Determine baseline SD for TED-AI calculation
        # If SAR exp selection would happen before AC, use SD at AC; otherwise use TOP_RESEARCHER_SD
        use_sar_baseline = (
            sar_exp_selection_time_unconstrained is not None and
            ac_time is not None and
            sar_exp_selection_time_unconstrained < ac_time
        )

        if use_sar_baseline:
            ted_ai_baseline_sd = sd_at_ac
        else:
            ted_ai_baseline_sd = cfg.TOP_RESEARCHER_SD

        milestones = {
            'AC': {
                'metric': 'progress',
                'target': self.results['sc_progress_level'],
                'interpolation_type': 'linear',            },
            'AI2027-SC': {
                'metric': 'ai_coding_labor_mult_ref_present_day',
                'target': (30 ** (1 / self.params.parallel_penalty)) * 30 * cfg.PARALLEL_LABOR_MULT_BETWEEN_AVERAGE_AND_TOP_FOR_AI2027_SC,
                'interpolation_type': 'exponential',
            },
            'AIR-5x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 5,
                'interpolation_type': 'exponential',
                'progress_multiplier': 5,
            },
            'AIR-25x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 25,
                'interpolation_type': 'exponential',
                'progress_multiplier': 25
            },
            'AIR-250x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 250,
                'interpolation_type': 'exponential',
                'progress_multiplier': 250
            },
            'AIR-2000x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 2000,
                'interpolation_type': 'exponential',
                'progress_multiplier': 2000
            },
            'AIR-10000x': {
                'metric': 'ai_sw_progress_mult_ref_present_day',
                'target': 10000,
                'interpolation_type': 'exponential',
                'progress_multiplier': 10000
            },
            'SAR-level-experiment-selection-skill': {
                'metric': 'ai_research_taste_sd',
                'target': cfg.TOP_RESEARCHER_SD,  # SD of top researcher (99.9th percentile)
                'interpolation_type': 'exponential',
                'requires_ac': True  # SAR also requires achieving AC coding requirement
            },
            'SIAR-level-experiment-selection-skill': {
                'metric': 'ai_research_taste_sd',
                'target': cfg.TOP_RESEARCHER_SD * 3.0,  # 3x the SD of top researcher
                'interpolation_type': 'exponential',
                'requires_ac': True  # SIAR also requires achieving AC coding requirement
            },
            'STRAT-AI': {
                'metric': 'ai_research_taste_sd',
                'target': cfg.TOP_RESEARCHER_SD * (1.0 + self.params.strat_ai_m2b),  # Linearly spaced in SD
                'interpolation_type': 'exponential'
            },
            'TED-AI': {
                'metric': 'ai_research_taste_sd',
                'target': ted_ai_baseline_sd + cfg.TOP_RESEARCHER_SD * self.params.ted_ai_m2b,  # M2B jumps measured in TOP_RESEARCHER_SD units
                'interpolation_type': 'exponential'
            },
            'ASI': {
                'metric': 'ai_research_taste_sd',
                'target': ted_ai_baseline_sd + cfg.TOP_RESEARCHER_SD * (self.params.ted_ai_m2b + 2.0),  # 2 M2Bs above TED-AI
                'interpolation_type': 'exponential'
            },
            'Maximum possible experiment selection skill': {
                'metric': 'ai_research_taste_sd',
                'target': cfg.AI_RESEARCH_TASTE_MAX_SD,
                'interpolation_type': 'exponential'
            }
        }

        def _ec_at_metric_crossing(metric, ec, target, log_space):
            """Effective compute at the point where ``metric`` first reaches ``target``,
            interpolating EC against the MILESTONE METRIC (not against time).

            During fast takeoff EC moves many OOMs per time-grid cell, so reading EC by
            ``np.interp(crossing_time, times, EC)`` linearly chords a near-vertical
            exponential and is dominated by grid-quantisation noise. EC, the metric, and
            progress are all smooth monotone functions of one another (EC = progress + const),
            so interpolating EC vs the metric in the space where the metric is linear in
            progress is well-conditioned (exact when EC is linear in the (log-)metric).
            Mirrors the bracketing in _find_exponential_crossing_time. Returns None if the
            metric never reaches the target."""
            idx = np.where(metric >= target)[0]
            if idx.size == 0:
                return None
            j = int(idx[0])
            if j == 0:
                return float(ec[0])
            v0, v1, e0, e1 = metric[j - 1], metric[j], ec[j - 1], ec[j]
            if log_space and v0 > 0 and v1 > 0 and target > 0 and v1 != v0:
                frac = (np.log(target) - np.log(v0)) / (np.log(v1) - np.log(v0))
            elif v1 != v0:
                frac = (target - v0) / (v1 - v0)
            else:
                frac = 1.0
            frac = float(min(max(frac, 0.0), 1.0))
            return float(e0 + frac * (e1 - e0))

        for milestone_name, milestone in milestones.items():
            # Special case: Use pre-computed aa_time for AC milestone to ensure consistency
            if milestone_name == 'AC' and self.results.get('aa_time') is not None:
                time = self.results['aa_time']
            elif milestone['interpolation_type'] == 'exponential':
                time = _find_exponential_crossing_time(
                    np.asarray(self.results['times'], dtype=float),
                    np.asarray(self.results[milestone['metric']], dtype=float),
                    float(milestone['target'])
                )
            else:
                time = np.interp(milestone['target'], self.results[milestone['metric']], self.results['times'])
                if np.interp(time, self.results['times'], self.results[milestone['metric']]) < milestone['target'] - 1e-4:
                    time = None

            # Check if milestone also requires AC coding requirement. Track whether AC is the
            # BINDING (later) constraint: if so the milestone is realised at ac_time (where the
            # metric is already past target), so EC must be read at ac_time, not at the metric
            # crossing. When the skill crossing binds (the usual case — and always for SIAR, whose
            # 3x-top-researcher taste is reached long after AC), the metric-space EC read applies
            # like any other milestone.
            _ac_bound = False
            if time is not None and milestone.get('requires_ac', False):
                # Get AC time (from aa_time)
                ac_time = self.results.get('aa_time')
                if ac_time is not None:
                    if ac_time > time:        # AC, not the skill, is the later (binding) requirement
                        _ac_bound = True
                    # SAR can only be achieved at the later of the two requirements
                    time = max(time, ac_time)
                else:
                    # If AC is never achieved, SAR cannot be achieved either
                    time = None

            if time is not None:
                milestone['time'] = time
                if 'progress_multiplier' not in milestone:
                    milestone['progress_multiplier'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['ai_sw_progress_mult_ref_present_day'], dtype=float))
                if 'effective_compute_ooms' not in milestone:
                    # Read EC at the metric crossing (not via crossing-time interp) to avoid the
                    # takeoff grid-quantisation artifact; fall back to time-interp if the metric
                    # never reaches target or AC is the binding constraint (achievement at ac_time,
                    # pre-takeoff and stable under time-interp).
                    # Interp space follows how the METRIC relates to progress (EC = progress + const):
                    # SD/progress metrics are ~linear in progress -> LINEAR (exact); multiplier/uplift
                    # metrics are exp in progress -> LOG (exact). This is NOT interpolation_type, which
                    # governs the crossing-TIME read (SD grows exponentially in TIME during takeoff);
                    # reading EC via the crossing time is equivalent to LOG-metric interp, which is the
                    # bug for SD milestones across a near-vertical takeoff cell.
                    _ec_arr = np.asarray(self.results['effective_compute'], dtype=float)
                    _metric_arr = np.asarray(self.results[milestone['metric']], dtype=float)
                    _mult_metrics = ('ai_sw_progress_mult_ref_present_day', 'ai_coding_labor_mult_ref_present_day')
                    _ec_cross = _ec_at_metric_crossing(
                        _metric_arr, _ec_arr, float(milestone['target']),
                        log_space=(milestone['metric'] in _mult_metrics))
                    if _ec_cross is None or _ac_bound:
                        milestone['effective_compute_ooms'] = float(np.interp(milestone['time'], self.results['times'], _ec_arr))
                    else:
                        milestone['effective_compute_ooms'] = _ec_cross
                milestone['research_effort'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['research_efforts'], dtype=float))
                milestone['research_stock'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['research_stock'], dtype=float))
                if self.results['takeoff_progress_multipliers']:
                    milestone['takeoff_progress_multiplier'] = _log_interp(milestone['time'], self.results['times'], np.asarray(self.results['takeoff_progress_multipliers'], dtype=float))
        milestones_list = list(zip(milestones.keys(), milestones.values()))
        # Filter out milestones without time and sort by time
        milestones_list = [(name, milestone) for name, milestone in milestones_list if 'time' in milestone]
        milestones_list.sort(key=lambda x: x[1]['time'])
        for i in range(len(milestones_list) - 1):
            this_RS = milestones_list[i][1]['research_stock']
            next_RS = milestones_list[i+1][1]['research_stock']
            if 'takeoff_start_stats' in self.human_only_results:
                milestones[milestones_list[i][0]]['human_only_years_to_next_milestone'] = (next_RS - this_RS) / self.human_only_results['takeoff_start_stats']['research_effort']

        return milestones
    
    def get_progress_at_time(self, time: float) -> float:
        """
        Get progress value at a specific time by interpolating computed results.
        
        Args:
            time: Time in decimal years
            
        Returns:
            Interpolated progress value
            
        Raises:
            ValueError: If no results are available or time is outside computed range
        """
        if 'times' not in self.results or 'progress' not in self.results:
            raise ValueError("No results available. Run compute_progress_trajectory first.")
        
        times = self.results['times']
        progress_values = self.results['progress']
        
        # Check if time is within the computed range
        if time < times[0] or time > times[-1]:
            raise ValueError(f"Time {time} is outside computed range [{times[0]:.3f}, {times[-1]:.3f}]")
        
        # Use numpy interpolation
        interpolated_progress = np.interp(time, times, progress_values)
        
        logger.debug(f"Interpolated progress at time {time}: {interpolated_progress:.6f}")
        
        return float(interpolated_progress)
    
    def compute_aggregate_taste_with_sd_schedule(self, progress: float, sd_per_progress_unit: float = 0.5) -> float:
        """
        Compute aggregate research taste with AI research taste specified in standard deviations.
        
        This is a convenience method that implements the pattern:
        ai_research_taste = taste_dist.get_taste_at_sd(sd_per_progress_unit * progress)
        aggregate_taste = taste_dist.get_mean_with_floor(ai_research_taste)
        
        Args:
            progress: Current progress level
            sd_per_progress_unit: How many standard deviations AI research taste grows per progress unit
            
        Returns:
            Aggregate research taste with the AI floor applied
            
        Example:
            # AI research taste grows at 0.5 SD per progress unit
            model = ProgressModel(params, data) 
            progress = 10.0
            aggregate_taste = model.compute_aggregate_taste_with_sd_schedule(progress, 0.5)
        """
        ai_research_taste = self.taste_distribution.get_taste_at_sd(sd_per_progress_unit * progress)
        return self.taste_distribution.get_mean_with_floor(ai_research_taste)


def load_time_series_data(filename: str) -> TimeSeriesData:
    """Load time series data from CSV file"""
    import csv

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    time = np.array([float(row['time']) for row in data])
    L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
    inference_H100e = np.array([float(row['inference_H100e']) for row in data])
    experiment_H100e = np.array([float(row['experiment_H100e']) for row in data])
    training_H100e = np.array([float(row['training_H100e']) for row in data])
    # Optional world AI-relevant compute column (raw H100e); only the covert pipeline's
    # bottom_up_estimate sizing uses it. Absent => None (core model ignores it).
    world_H100e = (np.array([float(row['world_H100e']) for row in data])
                   if data and 'world_H100e' in data[0] else None)
    # Optional total compute of the leading AI company (raw H100e); display/diagnostics only.
    total_ai_company_H100e = (np.array([float(row['total_ai_company_H100e']) for row in data])
                              if data and 'total_ai_company_H100e' in data[0] else None)

    return TimeSeriesData(time, L_HUMAN, inference_H100e, experiment_H100e, training_H100e,
                          world_H100e=world_H100e,
                          total_ai_company_H100e=total_ai_company_H100e)
