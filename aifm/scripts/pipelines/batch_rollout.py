#!/usr/bin/env python3
"""
Batch rollout script for the AI progress model (optimized).

Performance optimizations:
- Fork context on Linux for ~3-5x faster worker spawning (avoids re-importing modules)
- Batched I/O operations reduce disk write overhead by ~50%
- Larger chunksize reduces pool scheduling overhead
- Increased maxtasksperchild reduces process recreation overhead
- Auto-tuned defaults for optimal throughput on multi-core systems

Key options:
- --mp-context: Choose multiprocessing context (default: auto=fork on Linux)
- --io-buffer-size: Buffer results before flushing (default: 50)
- --pool-chunksize: Task batching size (default: auto)
- --maxtasksperchild: Worker process lifetime (default: 200)

IMPORTANT: Fork context is faster but less safe with threads. If you experience
crashes or hangs, fall back to spawn with --mp-context=spawn

Usage (from scripts/ directory):
  python pipelines/batch_rollout.py \\
    --config config/sampling_config.yaml \\
    --num-samples 5000 \\
    --num-workers 32

If --config is omitted, a default independent distribution is derived from
progress_model.config.PARAMETER_BOUNDS (uniform within bounds) and fixed defaults otherwise.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
from numpy.random import SeedSequence
import yaml
import warnings
import contextlib
import functools
import io
import importlib
import logging
import signal
import multiprocessing as mp
import scipy.special
import time as time_module
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    _tqdm = importlib.import_module("tqdm").tqdm  # type: ignore[attr-defined]
except Exception:
    _tqdm = None


# Ensure repository root is on sys.path so we can import project modules
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from progress_model import config as cfg
from progress_model import (
    ProgressModel, Parameters, load_time_series_data
)
from progress_model.time_series import generate_time_series_from_dict

# Track which parameters we have already warned about for CI-first precedence
_ci_first_warned_params: Set[str] = set()


def _run_post_processing_script(name: str, cmd: list) -> tuple:
    """Run a post-processing script and return (name, success, error_msg).

    Used for parallel execution of independent post-processing scripts.
    """
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            return (name, False, result.stderr or result.stdout or "Unknown error")
        return (name, True, None)
    except Exception as e:
        return (name, False, str(e))


# Non-daemon pool classes for multiprocessing with 'spawn' context
# Must be at module level to be picklable
import multiprocessing.pool as _mp_pool


class _NoDaemonProcess(mp.Process):
    """A Process subclass that is never daemonic, allowing it to spawn subprocesses."""
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass


class _NoDaemonPool(_mp_pool.Pool):
    """A Pool that uses non-daemon processes, allowing subprocess spawning for timeouts."""
    def Process(self, *args, **kwds):
        proc = super().Process(*args, **kwds)
        proc.__class__ = _NoDaemonProcess
        return proc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPTIMIZED batch rollout generator for AI progress model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config describing parameter distributions")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of parameter samples to draw and roll out")
    parser.add_argument("--input-data", type=str, default=str(REPO_ROOT / "inputs" / "input_data.csv"), help="Path to input time series CSV")
    parser.add_argument("--time-range", type=float, nargs=2, default=None, help="Start and end time (decimal years) for rollout")
    parser.add_argument("--initial-progress", type=float, default=0.0, help="Initial progress value")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs"), help="Base output directory")
    parser.add_argument("--per-sample-timeout", type=float, default=None, help="Max seconds to spend per rollout (None disables)")
    parser.add_argument("--post-process", action="store_true", help="Automatically run post-processing scripts (plots, sensitivity analysis, etc.)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: CPU count - 2, min 1)")
    parser.add_argument("--skip-summary", action="store_true", help="Skip automatic generation of rollout summary, batch plots, and milestone tables")
    parser.add_argument("--use-new-pipeline", action="store_true", help="Use new plots/pipelines/post_rollout.py instead of legacy post-processing")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution (run rollouts sequentially)")
    # OPTIMIZATION PARAMETERS
    parser.add_argument("--mp-context", type=str, default="auto", choices=["auto", "fork", "spawn", "forkserver"],
                        help="Multiprocessing context (default: auto=fork on Linux, spawn elsewhere). Fork is 3-5x faster but less safe.")
    parser.add_argument("--io-buffer-size", type=int, default=50,
                        help="Number of results to buffer before flushing to disk (default: 50). Higher = less I/O overhead.")
    parser.add_argument("--pool-chunksize", type=int, default=None,
                        help="Chunksize for pool.imap_unordered (default: auto = num_samples//(num_workers*4)). Higher = less scheduling overhead.")
    parser.add_argument("--maxtasksperchild", type=int, default=200,
                        help="Max tasks per worker process before recreation (default: 200). Higher = less process recreation overhead.")
    return parser.parse_args()


def _flag_provided(flag: str) -> bool:
    """Return True if a given CLI flag (e.g., "--seed") was provided by the user."""
    return flag in sys.argv


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _deep_merge_configs(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge child config into parent config.

    Child values override parent values. For nested dicts, merge recursively.
    For lists and other types, child completely replaces parent.
    Special case: if child value is None, keep parent value (don't replace with None).
    Special case: if child value is an empty dict {}, it replaces parent (to allow clearing).
    """
    result = parent.copy()
    for key, child_value in child.items():
        # Skip None values - keep parent value instead
        if child_value is None:
            continue

        # Empty dict {} means explicit replacement (e.g., to clear correlation_matrix)
        if isinstance(child_value, dict) and len(child_value) == 0:
            result[key] = child_value
        elif key in result and isinstance(result[key], dict) and isinstance(child_value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge_configs(result[key], child_value)
        else:
            # Child value completely replaces parent value
            result[key] = child_value
    return result


def _load_config(path: Optional[str], _visited: Optional[Set[Path]] = None) -> Dict[str, Any]:
    """Load config with support for inheritance via 'parent' field.

    If the config contains a 'parent' field, it will load that parent config first
    and deep merge the child config on top of it. Supports relative paths in 'parent'.

    Args:
        path: Path to config file
        _visited: Internal set to detect circular inheritance (do not use)

    Returns:
        Merged config dict

    Example child config:
        parent: "sampling_config.yaml"
        parameters:
          present_day:
            dist: fixed
            value: 2026.0  # Override parent value
    """
    if path is None:
        return {}

    p = Path(path).resolve()

    # Detect circular inheritance
    if _visited is None:
        _visited = set()
    if p in _visited:
        raise ValueError(f"Circular config inheritance detected: {p} has already been loaded")
    _visited.add(p)

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    text = p.read_text()
    try:
        if p.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(text)
        else:
            config = json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse config {p}: {e}")

    # Handle inheritance
    if config and "parent" in config:
        parent_path = config.pop("parent")  # Remove 'parent' field from final config

        # Resolve parent path relative to current config's directory
        if not Path(parent_path).is_absolute():
            parent_path = (p.parent / parent_path).resolve()
        else:
            parent_path = Path(parent_path).resolve()

        # Recursively load parent
        parent_config = _load_config(str(parent_path), _visited)

        # Deep merge child config on top of parent
        config = _deep_merge_configs(parent_config, config)

    return config


def _snapshot_model_config(output_dir: Path) -> None:
    # 1) write raw file snapshot for reproducibility
    raw_path = output_dir / "model_config_snapshot.py"
    raw_path.write_text((REPO_ROOT / "progress_model" / "config.py").read_text())

    # 2) write key structures as JSON for convenient reuse
    snapshot: Dict[str, Any] = {}
    keys_to_capture = [
        # constants and dicts
        "RHO_COBB_DOUGLAS_THRESHOLD", "RHO_LEONTIEF_THRESHOLD", "SIGMOID_EXPONENT_CLAMP",
        "AUTOMATION_FRACTION_CLIP_MIN", "RHO_CLIP_MIN", "PARAM_CLIP_MIN",
        "RESEARCH_STOCK_START_MIN",
        "NORMALIZATION_MIN", "experiment_compute_exponent_CLIP_MIN", "experiment_compute_exponent_CLIP_MAX",
        "AGGREGATE_RESEARCH_TASTE_BASELINE", "AGGREGATE_RESEARCH_TASTE_FALLBACK",
        "TOP_PERCENTILE", "MEDIAN_TO_TOP_TASTE_GAP",
        "TASTE_SCHEDULE_TYPES", "DEFAULT_TASTE_SCHEDULE_TYPE",
        "HORIZON_EXTRAPOLATION_TYPES", "DEFAULT_HORIZON_EXTRAPOLATION_TYPE",
        "DEFAULT_present_day", "DEFAULT_present_horizon", "DEFAULT_present_doubling_time",
        "DEFAULT_DOUBLING_DIFFICULTY_GROWTH_FACTOR", "AI_RESEARCH_TASTE_MIN", "AI_RESEARCH_TASTE_MAX",
        "AI_RESEARCH_TASTE_MAX_SD", "BASE_FOR_SOFTWARE_LOM", "MAX_RESEARCH_EFFORT", "MAX_NORMALIZED_PROGRESS_RATE",
        "TIME_EXTRAPOLATION_WINDOW", "PROGRESS_ODE_CLAMP_MAX", "RESEARCH_STOCK_ODE_CLAMP_MAX",
        "ODE_MAX_STEP", "EULER_FALLBACK_MIN_STEPS", "EULER_FALLBACK_STEPS_PER_YEAR",
        "DENSE_OUTPUT_STEP_SIZE", "ODE_STEP_SIZE_LOGGING", "ODE_SMALL_STEP_THRESHOLD",
        "ODE_STEP_VARIATION_THRESHOLD", "RELATIVE_ERROR_CLIP", "PARAMETER_BOUNDS",
        "PARAM_VALIDATION_THRESHOLDS", "DEFAULT_PARAMETERS",
    ]
    for k in keys_to_capture:
        if hasattr(cfg, k):
            snapshot[k] = getattr(cfg, k)
    (output_dir / "model_config_snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str))


def _build_default_distribution_config() -> Dict[str, Any]:
    # Use uniform within PARAMETER_BOUNDS where available, else fixed at DEFAULT_PARAMETERS
    param_cfg: Dict[str, Any] = {}
    for name, default_val in cfg.DEFAULT_PARAMETERS.items():
        param_cfg[name] = {"dist": "fixed", "value": default_val}

    # Also include categorical parameters if not already present explicitly
    if "horizon_extrapolation_type" not in param_cfg:
        param_cfg["horizon_extrapolation_type"] = {"dist": "choice", "values": cfg.HORIZON_EXTRAPOLATION_TYPES, "p": None}
    if "taste_schedule_type" not in param_cfg:
        param_cfg["taste_schedule_type"] = {"dist": "choice", "values": cfg.TASTE_SCHEDULE_TYPES, "p": None}

    return {
        "seed": 42,
        "parameters": param_cfg,
        # Additional run settings can be overridden by CLI
        "time_range": None,
        "initial_progress": 0.0,
    }


@contextlib.contextmanager
def _suppress_noise():
    # Silence Python warnings, logging, and stdout/stderr temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        previous_disable_threshold = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            logging.disable(previous_disable_threshold)


@functools.lru_cache(maxsize=512)
def _fit_beta_shapes_ci80(u10: float, u90: float) -> Tuple[float, float]:
    """Fit Beta(alpha, beta) shape params on the unit interval so its 10th and 90th
    percentiles equal ``u10`` and ``u90`` (both in (0,1), u10 < u90). Lets a beta dist
    be specified by an 80% CI instead of raw shapes. Memoized (one solve per CI)."""
    from scipy.optimize import least_squares
    def resid(logab):
        a, b = np.exp(logab)
        return [scipy.special.betaincinv(a, b, 0.10) - u10,
                scipy.special.betaincinv(a, b, 0.90) - u90]
    sol = least_squares(resid, x0=[0.0, 0.0], method="lm")  # x0 = log(alpha)=log(beta)=0 -> (1,1)
    a, b = (float(v) for v in np.exp(sol.x))
    resid_norm = float(np.hypot(*resid(sol.x)))
    if not (np.isfinite(a) and np.isfinite(b)) or resid_norm > 1e-4:
        raise ValueError(f"could not fit a Beta to 80% CI percentiles ({u10}, {u90}); "
                         f"got alpha={a:.4g}, beta={b:.4g}, residual={resid_norm:.2g}")
    return a, b


def _sample_from_dist(dist_spec: Dict[str, Any], rng: np.random.Generator, param_name: Optional[str] = None) -> Any:
    """Sample from a distribution using random sampling.
    
    This is a wrapper around _sample_from_dist_with_quantile that generates a random quantile.
    
    Args:
        dist_spec: Distribution specification
        rng: Random number generator
        param_name: Parameter name for error messages and warnings
        
    Returns:
        Sampled value
    """
    # Generate a random quantile and use the quantile-based sampling function
    quantile = rng.uniform(0.0, 1.0)
    return _sample_from_dist_with_quantile(dist_spec, quantile, param_name)


def _sample_from_dist_with_quantile(dist_spec: Dict[str, Any], quantile: float, param_name: Optional[str] = None) -> Any:
    """Sample from a distribution using a specific quantile (for correlated sampling).
    
    Args:
        dist_spec: Distribution specification
        quantile: Quantile value in [0, 1]
        param_name: Parameter name for error messages
        
    Returns:
        Sampled value
    """
    kind = dist_spec.get("dist", "fixed")

    if kind == "fixed":
        return dist_spec.get("value")
    elif kind == "choice":
        pass
    elif (dist_spec.get("min") is None or dist_spec.get("max") is None) and dist_spec.get("clip_to_bounds", True):
        raise ValueError(f"min and max are required for {kind} distribution for parameter {param_name}")

    if kind == "uniform":
        a = float(dist_spec["min"])
        b = float(dist_spec["max"])
        return a + quantile * (b - a)

    if kind == "normal":
        # Support parameterization by mean/sd (or sigma) OR by 80% CI (q10,q90)
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            # Warn once per parameter if both CI and mean/sd are provided
            if (
                ("mean" in dist_spec or "sd" in dist_spec or "sigma" in dist_spec)
                and param_name is not None and param_name not in _ci_first_warned_params
            ):
                import warnings as _warnings
                _warnings.warn(
                    f"Parameter '{param_name}': 80% CI (ci80_low/ci80_high) and mean/sd provided; "
                    f"using 80% CI and ignoring mean/sd.",
                )
                _ci_first_warned_params.add(param_name)
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            mu = 0.5 * (q10 + q90)
            sigma = (q90 - q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mean"])
            sigma = float(dist_spec["sd"]) if "sd" in dist_spec else float(dist_spec.get("sigma", 1.0))

        # Use inverse normal CDF
        x = mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1)
        if dist_spec.get("clip_to_bounds") and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "lognormal":
        # Support parameterization by mu/sigma in log-space OR by 80% CI in original space
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            # Warn once per parameter if both CI and mu/sigma are provided
            if (
                ("mu" in dist_spec or "sigma" in dist_spec)
                and param_name is not None and param_name not in _ci_first_warned_params
            ):
                import warnings as _warnings
                _warnings.warn(
                    f"Parameter '{param_name}': 80% CI (ci80_low/ci80_high) and mu/sigma provided; "
                    f"using 80% CI and ignoring mu/sigma.",
                )
                _ci_first_warned_params.add(param_name)
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            ln_q10 = float(np.log(q10))
            ln_q90 = float(np.log(q90))
            mu = 0.5 * (ln_q10 + ln_q90)
            sigma = (ln_q90 - ln_q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mu"])
            sigma = float(dist_spec["sigma"])

        # Use inverse lognormal CDF
        x = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
        if dist_spec.get("clip_to_bounds") and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "shifted_lognormal":
        # Shifted lognormal: x = shift + LogNormal(mu, sigma)
        has_pair = "ci80_low" in dist_spec and "ci80_high" in dist_spec
        has_array = "ci80" in dist_spec
        if has_array or has_pair:
            # Warn once per parameter if both CI and mu/sigma are provided
            if (
                ("mu" in dist_spec or "sigma" in dist_spec)
                and param_name is not None and param_name not in _ci_first_warned_params
            ):
                import warnings as _warnings
                _warnings.warn(
                    f"Parameter '{param_name}': 80% CI (ci80_low/ci80_high) and mu/sigma provided; "
                    f"using 80% CI and ignoring mu/sigma.",
                )
                _ci_first_warned_params.add(param_name)
            if has_array:
                ci = dist_spec["ci80"]
                q10 = float(ci[0])
                q90 = float(ci[1])
            else:
                q10 = float(dist_spec["ci80_low"])
                q90 = float(dist_spec["ci80_high"])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            ln_q10 = float(np.log(q10))
            ln_q90 = float(np.log(q90))
            mu = 0.5 * (ln_q10 + ln_q90)
            sigma = (ln_q90 - ln_q10) / (2.0 * z)
        else:
            mu = float(dist_spec["mu"])
            sigma = float(dist_spec["sigma"])

        # Use inverse shifted lognormal CDF
        x_core = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
        shift = dist_spec.get("shift")
        if shift is None:
            raise ValueError(f"shift for shifted_lognormal is not set for parameter {param_name}")
        shift = float(shift)
        x = float(shift + x_core)
        if dist_spec.get("clip_to_bounds") and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "beta":
        lo = float(dist_spec.get("min", 0.0))
        hi = float(dist_spec.get("max", 1.0))
        if "alpha" in dist_spec and "beta" in dist_spec:
            a = float(dist_spec["alpha"])
            b = float(dist_spec["beta"])
        elif "ci80" in dist_spec:
            # Fit shapes to an 80% CI (given in data units; mapped onto the [min,max] support).
            c0, c1 = (float(x) for x in dist_spec["ci80"])
            u10, u90 = (c0 - lo) / (hi - lo), (c1 - lo) / (hi - lo)
            if not (0.0 < u10 < u90 < 1.0):
                raise ValueError(f"beta ci80 {dist_spec['ci80']} must lie strictly inside "
                                 f"(min,max)=({lo},{hi}) for parameter {param_name}")
            a, b = _fit_beta_shapes_ci80(round(u10, 9), round(u90, 9))
        else:
            raise ValueError(f"beta distribution for parameter {param_name} needs either "
                             f"alpha+beta or ci80 (plus optional min/max support)")
        # Use inverse beta CDF
        x01 = scipy.special.betaincinv(a, b, quantile)
        return lo + (hi - lo) * x01

    if kind == "cdf_points":
        # Distribution given directly as elicited CDF anchor points:
        #   points: [[value, P(X <= value)], ...]
        # matching elicitations of the form "P(> value is infeasible/suboptimal) = p"
        # (which is exactly P(X <= value) = p). The CDF is interpolated between anchors
        # (linearly in value, or in log(value) with scale: log); the p1 of mass below the
        # first anchor and the 1-pn above the last collapse onto the endpoint values
        # (point masses), since the elicitation says nothing about those tails.
        pts = dist_spec.get("points")
        if not isinstance(pts, (list, tuple)) or len(pts) < 2:
            raise ValueError(f"cdf_points for parameter {param_name} needs a 'points' list "
                             f"of at least two [value, cum_prob] rows")
        try:
            vals = [float(v) for v, _ in pts]
            probs = [float(p) for _, p in pts]
        except (TypeError, ValueError) as e:
            raise ValueError(f"cdf_points rows for parameter {param_name} must be "
                             f"[value, cum_prob] pairs, got {pts!r}") from e
        if any(b <= a for a, b in zip(vals, vals[1:])):
            raise ValueError(f"cdf_points values for parameter {param_name} must be "
                             f"strictly increasing, got {vals}")
        if any(b <= a for a, b in zip(probs, probs[1:])) or not all(0.0 < p <= 1.0 for p in probs):
            raise ValueError(f"cdf_points cum_probs for parameter {param_name} must be "
                             f"strictly increasing within (0, 1], got {probs}")
        scale = dist_spec.get("scale", "linear")
        if scale not in ("linear", "log"):
            raise ValueError(f"cdf_points scale for parameter {param_name} must be "
                             f"'linear' or 'log', got {scale!r}")
        if scale == "log" and vals[0] <= 0.0:
            raise ValueError(f"cdf_points scale=log for parameter {param_name} requires "
                             f"positive values, got {vals}")
        if quantile <= probs[0]:
            x = vals[0]      # exact endpoint (a point mass, not an interpolation artifact)
        elif quantile >= probs[-1]:
            x = vals[-1]
        elif scale == "log":
            x = float(np.exp(np.interp(quantile, probs, np.log(vals))))
        else:
            x = float(np.interp(quantile, probs, vals))
        if dist_spec.get("clip_to_bounds") and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "choice":
        values = dist_spec["values"]
        p = dist_spec.get("p")
        if p is None:
            # Uniform choice
            idx = int(quantile * len(values))
            idx = min(idx, len(values) - 1)
            return values[idx]
        else:
            # Weighted choice - find cumulative sum
            cumsum = np.cumsum(p)
            idx = np.searchsorted(cumsum, quantile)
            idx = min(idx, len(values) - 1)
            return values[idx]

    raise ValueError(f"Unknown distribution kind: {kind}")


def _clip_to_param_bounds(param_name: str, value: Any) -> Any:
    bounds = cfg.PARAMETER_BOUNDS.get(param_name)
    if value is None:
        assert False, f"value is None for parameter {param_name}"
    if bounds is None:
        return value
    lo, hi = bounds
    return float(np.clip(float(value), float(lo), float(hi)))


def _get_median_parameter_dict(param_dists: Dict[str, Any]) -> Dict[str, Any]:
    """Get the median value for each parameter distribution.

    Uses quantile=0.5 for all distributions to get the median value.

    Args:
        param_dists: Parameter distribution specifications

    Returns:
        Dict of parameter names to their median values
    """
    median_params: Dict[str, Any] = {}

    for name, spec in param_dists.items():
        if name == "automation_anchors":
            continue

        val = _sample_from_dist_with_quantile(spec, 0.5, name)

        # Clip to bounds unless explicitly disabled
        clip_requested = spec.get("clip_to_bounds", True)
        if clip_requested:
            val = _clip_to_param_bounds(name, val)

        median_params[name] = val

    return median_params


def run_median_trajectory_and_save(
    param_dists: Dict[str, Any],
    ts_param_dists: Dict[str, Any],
    input_data_path: str,
    time_range: List[float],
    initial_progress: float,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Run a single trajectory with median parameter values and save milestone times to CSV.

    Args:
        param_dists: Model parameter distribution specifications
        ts_param_dists: Time series parameter distribution specifications
        input_data_path: Path to input time series CSV
        time_range: [start_time, end_time] for rollout
        initial_progress: Initial progress value
        output_dir: Directory to save output CSV

    Returns:
        Dict with milestone times, or None if trajectory failed
    """
    import csv

    print("\nRunning median parameter trajectory...")

    # Get median values for all parameters
    median_model_params = _get_median_parameter_dict(param_dists)
    median_ts_params = _get_median_parameter_dict(ts_param_dists)

    # Ensure categorical defaults exist
    median_model_params.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
    median_model_params.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)

    try:
        # Generate time series with median parameters
        if median_ts_params:
            from progress_model.time_series import generate_time_series_from_dict as _gen_ts
            base_data = load_time_series_data(input_data_path)
            rollout_data = _gen_ts(median_ts_params, input_data_path, base_data)
        else:
            rollout_data = load_time_series_data(input_data_path)

        # Run the trajectory
        params_obj = Parameters(**median_model_params)
        model = ProgressModel(params_obj, rollout_data)
        with _suppress_noise():
            model.compute_progress_trajectory(time_range, initial_progress)

        # Extract milestones
        results = model.results
        milestones = results.get("milestones", {})

        # Prepare CSV data
        csv_rows = []
        csv_rows.append(["Milestone", "Time (Year)", "Progress Multiplier", "Effective Compute OOMs"])

        # Define milestone order for consistent output
        milestone_order = [
            "AC", "SAR", "SIAR", "TED-AI", "ASI",
            "AIR-5x", "AIR-25x", "AIR-250x", "AIR-2000x", "AIR-10000x",
        ]

        # Add milestones in order, then any additional ones
        added_milestones = set()
        for milestone_name in milestone_order:
            if milestone_name in milestones:
                m = milestones[milestone_name]
                time_val = m.get("time")
                progress_mult = m.get("progress_multiplier")
                eff_compute = m.get("effective_compute_ooms")

                time_str = f"{time_val:.3f}" if time_val is not None else "Not achieved"
                mult_str = f"{progress_mult:.2f}" if progress_mult is not None else "N/A"
                compute_str = f"{eff_compute:.2f}" if eff_compute is not None else "N/A"

                csv_rows.append([milestone_name, time_str, mult_str, compute_str])
                added_milestones.add(milestone_name)

        # Add any remaining milestones not in the predefined order
        for milestone_name, m in sorted(milestones.items()):
            if milestone_name in added_milestones:
                continue
            time_val = m.get("time")
            progress_mult = m.get("progress_multiplier")
            eff_compute = m.get("effective_compute_ooms")

            time_str = f"{time_val:.3f}" if time_val is not None else "Not achieved"
            mult_str = f"{progress_mult:.2f}" if progress_mult is not None else "N/A"
            compute_str = f"{eff_compute:.2f}" if eff_compute is not None else "N/A"

            csv_rows.append([milestone_name, time_str, mult_str, compute_str])

        # Write CSV
        csv_path = output_dir / "median_trajectory_milestones.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

        print(f"  Median trajectory milestones saved to: {csv_path}")

        # Write the full rollout to JSONL (same format as rollouts.jsonl)
        rollout_path = output_dir / "median_trajectory_rollout.jsonl"
        rollout_record = {
            "sample_id": "median",
            "parameters": _to_jsonable(median_model_params),
            "time_series_parameters": _to_jsonable(median_ts_params),
            "results": _to_jsonable(results),
        }
        with rollout_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(rollout_record) + "\n")
        print(f"  Median trajectory rollout saved to: {rollout_path}")

        # Also write the median parameters to a JSON file for reference
        params_path = output_dir / "median_parameters.json"
        params_out = {
            "model_parameters": _to_jsonable(median_model_params),
            "time_series_parameters": _to_jsonable(median_ts_params),
        }
        params_path.write_text(json.dumps(params_out, indent=2))
        print(f"  Median parameters saved to: {params_path}")

        # Print summary to console
        print("\n  Median Trajectory Milestone Times:")
        print("  " + "-" * 50)
        for row in csv_rows[1:]:  # Skip header
            print(f"    {row[0]:20s}: {row[1]}")

        return {"milestones": milestones, "parameters": median_model_params, "ts_parameters": median_ts_params}

    except Exception as e:
        import traceback
        print(f"  WARNING: Median trajectory computation failed: {e}")
        print(traceback.format_exc())
        return None


# =======================
# Correlation calibration
# =======================

def _gaussian_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


def _quantile_transform_vectorized(dist_spec: Dict[str, Any], u: np.ndarray, param_name: Optional[str] = None) -> np.ndarray:
    # Thin wrapper to reuse the scalar quantile sampler in vectorized fashion
    # Using np.vectorize keeps code simple; performance is adequate for calibration sizes (~1e4-5e4)
    vec = np.vectorize(lambda q: _sample_from_dist_with_quantile(dist_spec, float(q), param_name), otypes=[float])
    return vec(u)


def _estimate_pearson_for_rho(
    spec_i: Dict[str, Any],
    spec_j: Dict[str, Any],
    rho_z: float,
    base_normals: Tuple[np.ndarray, np.ndarray],
    param_i_name: str,
    param_j_name: str,
) -> float:
    # Build correlated normals from shared base normals (A,B) for reproducibility across rho values
    A, B = base_normals
    # Guard against numerical issues near |rho|=1
    rho = float(np.clip(rho_z, -0.9999, 0.9999))
    Z1 = A
    Z2 = rho * A + np.sqrt(max(0.0, 1.0 - rho * rho)) * B
    U1 = _gaussian_cdf(Z1)
    U2 = _gaussian_cdf(Z2)
    X1 = _quantile_transform_vectorized(spec_i, U1, param_i_name)
    X2 = _quantile_transform_vectorized(spec_j, U2, param_j_name)
    c = np.corrcoef(X1, X2)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)


def _solve_latent_rho_for_pair(
    spec_i: Dict[str, Any],
    spec_j: Dict[str, Any],
    target_corr: float,
    base_normals: Tuple[np.ndarray, np.ndarray],
    param_i_name: str,
    param_j_name: str,
    tol: float = 1e-2,
    max_iter: int = 40,
) -> float:
    # Bisection on rho_z in [-0.999, 0.999] to achieve target Pearson correlation after transforms
    lo, hi = -0.999, 0.999
    f_lo = _estimate_pearson_for_rho(spec_i, spec_j, lo, base_normals, param_i_name, param_j_name)
    f_hi = _estimate_pearson_for_rho(spec_i, spec_j, hi, base_normals, param_i_name, param_j_name)

    # If target is outside achievable range, clamp to nearest endpoint
    if target_corr <= min(f_lo, f_hi):
        return lo if f_lo <= f_hi else hi
    if target_corr >= max(f_lo, f_hi):
        return lo if f_lo >= f_hi else hi

    # Ensure f is increasing with rho for bisection; if not, swap ends
    increasing = f_lo < f_hi
    left, right = (lo, hi) if increasing else (hi, lo)
    f_left = f_lo if increasing else f_hi
    f_right = f_hi if increasing else f_lo

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = _estimate_pearson_for_rho(spec_i, spec_j, mid, base_normals, param_i_name, param_j_name)
        if abs(f_mid - target_corr) <= tol:
            return float(mid)
        if f_mid < target_corr:
            left, f_left = mid, f_mid
        else:
            right, f_right = mid, f_mid

    return float(0.5 * (left + right))


def _nearest_correlation_matrix(A: np.ndarray, tol: float = 1e-8, max_iters: int = 100) -> np.ndarray:
    # Higham's nearest correlation matrix algorithm (simplified)
    np.linalg.cholesky(A)
    return A
    Y = A.copy()
    np.fill_diagonal(Y, 1.0)
    delta_S = np.zeros_like(Y)
    for _ in range(max_iters):
        R = Y - delta_S
        # PSD projection
        eigvals, eigvecs = np.linalg.eigh((R + R.T) * 0.5)
        eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
        X = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        delta_S = X - R
        Y = X
        # Unit diagonal and symmetry
        np.fill_diagonal(Y, 1.0)
        Y = (Y + Y.T) * 0.5
        # Convergence check (diagonal already unit; check min eigenvalue)
        if np.min(eigvals_clipped) >= -tol:
            break
    # Ensure strict PD by jitter if needed
    try:
        np.linalg.cholesky(Y)
    except np.linalg.LinAlgError:
        # Add a small jitter
        jitter = 1e-8
        for _ in range(5):
            try:
                Y_j = Y + np.eye(Y.shape[0]) * jitter
                np.fill_diagonal(Y_j, 1.0)
                np.linalg.cholesky(Y_j)
                Y = Y_j
                break
            except np.linalg.LinAlgError:
                jitter *= 10
    return Y


def _calibrate_gaussian_copula_latent_corr(
    param_dists: Dict[str, Any],
    correlated_params: List[str],
    target_corr: np.ndarray,
    rng: np.random.Generator,
    interpretation: str = "spearman",
    calibration_samples: int = 20000,
    tol: float = 1e-2,
) -> np.ndarray:
    n = len(correlated_params)
    if n <= 1:
        return np.eye(n)

    # If interpreting as Spearman rank correlation, convert to latent Gaussian Pearson via closed form
    if interpretation.lower() in {"spearman", "rank", "spearman_rho"}:
        # For a Gaussian copula: rho_S = (6/pi) * arcsin(rho/2)  =>  rho = 2 * sin(pi * rho_S / 6)
        rho_latent = 2.0 * np.sin(np.pi * target_corr / 6.0)
        rho_latent = np.clip(rho_latent, -0.9999, 0.9999)
        np.fill_diagonal(rho_latent, 1.0)
        return _nearest_correlation_matrix(rho_latent)

    # Otherwise, interpret target_corr as Pearson in original space; use NORTA pairwise calibration
    # Pre-generate base normals shared across all pairs and iterations for reproducibility and monotonic mapping
    A = rng.standard_normal(calibration_samples)
    B = rng.standard_normal(calibration_samples)
    base_normals = (A, B)

    rho_latent = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            name_i = correlated_params[i]
            name_j = correlated_params[j]
            spec_i = param_dists[name_i]
            spec_j = param_dists[name_j]
            target = float(target_corr[i, j])
            # Solve latent rho
            rho_ij = _solve_latent_rho_for_pair(spec_i, spec_j, target, base_normals, name_i, name_j, tol=tol)
            rho_latent[i, j] = rho_ij
            rho_latent[j, i] = rho_ij

    # Project to nearest valid correlation matrix
    rho_latent = _nearest_correlation_matrix(rho_latent)
    return rho_latent


def _sample_parameter_dict(param_dists: Dict[str, Any], rng: np.random.Generator, correlation_matrix: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sample parameters, optionally applying correlations between specified parameters.
    
    Args:
        param_dists: Parameter distribution specifications
        rng: Random number generator
        correlation_matrix: Optional correlation specification with format:
            {
                "parameters": ["param1", "param2", ...],  # List of parameter names
                "correlation_matrix": [[1.0, 0.5, ...], [0.5, 1.0, ...], ...]  # Correlation matrix
            }
    """
    sampled: Dict[str, Any] = {}
    
    # Handle correlated parameters if specified
    if correlation_matrix is not None:
        correlated_params = correlation_matrix.get("parameters", [])
        corr_matrix = correlation_matrix.get("correlation_matrix")
        if correlated_params and corr_matrix is not None:
            # Validate correlation matrix
            n_params = len(correlated_params)
            if len(corr_matrix) != n_params:
                raise ValueError(f"Correlation matrix must be {n_params}x{n_params} for {n_params} parameters")
            for i, row in enumerate(corr_matrix):
                if len(row) != n_params:
                    raise ValueError(f"Correlation matrix row {i} must have {n_params} elements")
            
            # Convert to numpy array and validate
            # Prefer a pre-calibrated latent correlation matrix if provided
            latent_corr = correlation_matrix.get("latent_correlation_matrix")
            corr_array = np.array(latent_corr if latent_corr is not None else corr_matrix, dtype=float)
            assert latent_corr is not None
            if not np.allclose(corr_array, corr_array.T):
                raise ValueError("Correlation matrix must be symmetric")
            if not np.allclose(np.diag(corr_array), 1.0):
                raise ValueError("Correlation matrix diagonal must be 1.0")
            
            # Check if correlation matrix is positive semi-definite
            try:
                np.linalg.cholesky(corr_array)
            except np.linalg.LinAlgError:
                raise ValueError("Correlation matrix must be positive semi-definite")
            
            # Sample correlated parameters via Gaussian copula
            # Generate multivariate normal samples.
            #
            # method="cholesky", not the default "svd": on a (near-)degenerate
            # correlation matrix, LAPACK's SVD returns an arbitrary-but-valid
            # basis, and which one it returns depends on the CPU's SIMD kernel.
            # Same seed on two machines then gives materially different draws
            # (measured max relative difference 3.5e+01, every element). The
            # Cholesky factor is unique for a PD matrix, so the seed->draw map
            # stops depending on the machine; residual disagreement drops to
            # last-bit rounding (1.0e-11). Any factor A with A@A.T == corr
            # yields a valid draw, so the joint being sampled is unchanged.
            # Cholesky's one downside -- it raises on singular input where SVD
            # would cope -- cannot bite here: the check just above already
            # required np.linalg.cholesky(corr_array) to succeed.
            #
            # This was live, not hypothetical. The 10-parameter Q2 configs
            # (q2_brendan, q2_eli) calibrate to a latent matrix with EXACTLY
            # repeated eigenvalues (smallest eigengap ~4e-16, vs ~7e-02 for the
            # 9-parameter configs), which is exactly the degeneracy that makes
            # the SVD basis arbitrary. Same latent matrix, same seed, two
            # machines: svd disagreed by up to 4.5 and 6.4 in latent-normal
            # units on ~97% of elements, while cholesky agreed to a single ULP
            # (and bit-exactly on the 9-parameter configs).
            mv_samples = rng.multivariate_normal(
                np.zeros(n_params), corr_array, method="cholesky"
            )
            
            # Convert to uniform marginals using normal CDF
            uniform_samples = np.array([0.5 * (1 + scipy.special.erf(x / np.sqrt(2))) for x in mv_samples])
            
            # Sample each correlated parameter using its distribution with the uniform quantile
            for i, param_name in enumerate(correlated_params):
                if param_name not in param_dists:
                    raise ValueError(f"Correlated parameter '{param_name}' not found in parameter distributions")
                
                spec = param_dists[param_name]
                val = _sample_from_dist_with_quantile(spec, uniform_samples[i], param_name)
                
                # Clip to bounds unless explicitly disabled
                clip_requested = spec.get("clip_to_bounds", True)
                if clip_requested:
                    val = _clip_to_param_bounds(param_name, val)
                
                sampled[param_name] = val
    
    # Sample remaining independent parameters
    for name, spec in param_dists.items():
        if name == "automation_anchors":
            continue
        if name in sampled:  # Skip if already sampled as part of correlated group
            continue
            
        val = _sample_from_dist(spec, rng, name)

        # Clip to bounds unless explicitly disabled, when bounds exist
        clip_requested = spec.get("clip_to_bounds", True)
        if clip_requested:
            val = _clip_to_param_bounds(name, val)

        sampled[name] = val

    return sampled


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _write_ndjson(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str))
            f.write("\n")


def _to_jsonable(obj: Any) -> Any:
    # Convert nested structures and numpy types to JSON-friendly forms
    try:
        import numpy as _np
    except Exception:
        _np = None

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        # NaNs/inf handled below
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            val = obj.item()
            if isinstance(val, float) and not np.isfinite(val):
                return None
            return val
    # Fallback to string
    return str(obj)


def _rollout_worker(conn, sampled_params: Dict[str, Any], sampled_ts_params: Dict[str, Any], input_data_path: str, time_range: List[float], initial_progress: float) -> None:
    """Subprocess worker: runs one rollout and sends back results or error."""
    try:
        # Lazy imports in subprocess to avoid parent state issues
        from progress_model import ProgressModel as _PM, Parameters as _Params, load_time_series_data as _load
        from progress_model.time_series import generate_time_series_from_dict as _gen_ts

        # Load base time series data
        base_data = _load(input_data_path)

        # Check if time series parameters are present; if so, generate new time series
        if sampled_ts_params:
            # Generate time series with uncertainty
            data = _gen_ts(sampled_ts_params, input_data_path, base_data)
        else:
            # Use base data as-is
            data = base_data

        params_obj = _Params(**sampled_params)
        model = _PM(params_obj, data)
        with _suppress_noise():
            model.compute_progress_trajectory(time_range, initial_progress)
        # Include computed r_software in results
        conn.send({"ok": True, "results": _to_jsonable(model.results), "r_software": float(model.params.r_software)})
    except Exception as e:  # pragma: no cover
        import traceback
        conn.send({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
    finally:
        try:
            conn.close()
        except Exception as e2:
            print(f"Error closing connection: {e2}")
            pass


def _run_rollout_subprocess(sampled_params: Dict[str, Any], sampled_ts_params: Dict[str, Any], input_data_path: str, time_range: List[float], initial_progress: float, timeout_s: Optional[float]) -> Dict[str, Any]:
    """Run one rollout in a child process; enforce timeout by termination.

    Returns dict with 'results' and 'r_software' on success. Raises TimeoutError on timeout. Raises Exception on failure.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_rollout_worker, args=(child_conn, sampled_params, sampled_ts_params, input_data_path, time_range, initial_progress))
    proc.daemon = False
    proc.start()
    child_conn.close()  # close in parent
    try:
        if timeout_s is None or timeout_s <= 0:
            # Wait indefinitely
            result = parent_conn.recv()
        else:
            if parent_conn.poll(timeout_s):
                result = parent_conn.recv()
            else:
                # Timeout: kill child
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                raise TimeoutError(f"Rollout timed out after {timeout_s} seconds")
        # Ensure the child is fully reaped (avoid zombies)
        try:
            proc.join(timeout=1)
        except Exception as e:
            print(f"Error joining process: {e}")
            pass
        if not isinstance(result, dict) or not result.get("ok"):
            err = None if not isinstance(result, dict) else result.get("error")
            traceback = None if not isinstance(result, dict) else result.get("traceback")
            error_msg = err or "Unknown rollout failure"
            if traceback:
                error_msg += f"\nSubprocess traceback:\n{traceback}"
            raise RuntimeError(error_msg)
        # Return both results and r_software
        return {"results": result["results"], "r_software": result.get("r_software")}
    finally:
        try:
            parent_conn.close()
        except Exception as e:
            print(f"Error closing parent connection: {e}")
            pass
        try:
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception as e:
                    print(f"Error terminating process: {e}")
                    pass
            proc.join(timeout=1)
        except Exception as e:
            print(f"Error joining process: {e}")
            pass

def _run_with_timeout(seconds: Optional[float], func, *args, **kwargs):
    """Run func(*args, **kwargs) with a wall-clock timeout in seconds using SIGALRM.

    If seconds is None or <= 0, runs without a timeout. Raises TimeoutError on expiry.
    """
    if seconds is None or seconds <= 0:
        return func(*args, **kwargs)

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"Rollout timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        # setitimer supports sub-second precision
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        return func(*args, **kwargs)
    finally:
        # Always cancel timer and restore previous handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _subprocess_rollout_worker(conn, args_tuple):
    """Subprocess worker that runs rollout and sends result back via pipe."""
    try:
        result = _run_single_rollout(args_tuple)
        conn.send(result)
    except Exception as e:
        import traceback
        sample_id = args_tuple[0]
        conn.send({
            'sample_id': sample_id,
            'parameters': None,
            'time_series_parameters': None,
            'results': None,
            'error': f"Subprocess error: {str(e)}",
            'traceback': traceback.format_exc(),
            'success': False,
            'timeout': False,
        })
    finally:
        conn.close()


class _TimeoutError(Exception):
    """Raised when a rollout times out."""
    pass


def _run_single_rollout_with_signal_timeout(args_tuple):
    """Wrapper that enforces timeout using signal.alarm (main process only).

    This is simpler and works reliably in the main process without subprocess overhead.
    Only works on Unix systems and only in the main thread.
    """
    per_sample_timeout = args_tuple[7]
    timeout_seconds = int(per_sample_timeout) if per_sample_timeout and per_sample_timeout > 0 else 120

    def timeout_handler(signum, frame):
        raise _TimeoutError(f"Rollout timed out after {timeout_seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = _run_single_rollout(args_tuple)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_single_rollout_optimized(args_tuple):
    """OPTIMIZED: Use signal-based timeout in worker processes to avoid subprocess overhead.

    When using fork context, we're already in a separate process, so we can use
    signal.setitimer directly without spawning yet another subprocess. This eliminates
    the massive overhead of nested process creation (~200ms per rollout overhead eliminated).

    This function should be used with fork context pools. For spawn context, use
    _run_single_rollout_with_timeout which provides subprocess isolation.
    """
    per_sample_timeout = args_tuple[7]
    timeout_seconds = float(per_sample_timeout) if per_sample_timeout and per_sample_timeout > 0 else 120.0

    def timeout_handler(signum, frame):
        raise _TimeoutError(f"Rollout timed out after {timeout_seconds} seconds")

    # Use setitimer for sub-second precision
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        result = _run_single_rollout(args_tuple)
        return result
    finally:
        # Always cancel timer and restore previous handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_single_rollout_with_timeout(args_tuple):
    """Wrapper that enforces a hard timeout on worker execution using subprocess isolation.

    This approach spawns each rollout in its own subprocess to ensure reliable timeout
    enforcement without signal handler conflicts in multiprocessing pools.
    """
    # Extract timeout from args
    per_sample_timeout = args_tuple[7]
    timeout_seconds = per_sample_timeout if per_sample_timeout and per_sample_timeout > 0 else 120.0

    # Run in isolated subprocess with timeout
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_subprocess_rollout_worker, args=(child_conn, args_tuple))
    proc.daemon = False
    proc.start()
    child_conn.close()

    try:
        if parent_conn.poll(timeout_seconds):
            result = parent_conn.recv()
            proc.join(timeout=1)
            return result
        else:
            # Timeout: kill subprocess
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)

            sample_id = args_tuple[0]
            return {
                'sample_id': sample_id,
                'parameters': None,
                'time_series_parameters': None,
                'results': None,
                'error': f"Rollout timed out after {timeout_seconds} seconds",
                'success': False,
                'timeout': True,
            }
    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


def _run_single_rollout(args_tuple):
    """Worker function for parallel rollout execution.

    Args:
        args_tuple: Tuple containing (sample_id, seed_seq, all_param_dists, correlation_matrix,
                    input_data_path, time_range, initial_progress, per_sample_timeout,
                    model_param_names, ts_param_names)
                    where seed_seq is a SeedSequence for this specific sample.

    Returns:
        Dict with sample_id, parameters, time_series_parameters, results, and profiling info
    """
    (sample_id, seed_seq, all_param_dists, correlation_matrix, input_data_path,
     time_range, initial_progress, per_sample_timeout, model_param_names, ts_param_names) = args_tuple

    # Create RNG from pre-spawned SeedSequence for statistically independent streams
    rng = np.random.default_rng(seed_seq)

    profiling = {}
    sampled_params = None
    sampled_ts_params = None

    # Suppress all output from worker process
    with _suppress_noise():
        try:
            # Sample all parameters together (enables correlations across parameter groups)
            t0 = time_module.perf_counter()
            all_sampled = _sample_parameter_dict(all_param_dists, rng, correlation_matrix)

            # Split sampled parameters back into model and time series groups
            sampled_params = {k: v for k, v in all_sampled.items() if k in model_param_names}
            sampled_ts_params = {k: v for k, v in all_sampled.items() if k in ts_param_names}

            # Make sure categorical defaults exist if user omitted
            sampled_params.setdefault("horizon_extrapolation_type", cfg.DEFAULT_HORIZON_EXTRAPOLATION_TYPE)
            sampled_params.setdefault("taste_schedule_type", cfg.DEFAULT_TASTE_SCHEDULE_TYPE)
            profiling['sampling'] = time_module.perf_counter() - t0

            # Generate time series with uncertainty if time series parameters are present
            t0 = time_module.perf_counter()
            if sampled_ts_params:
                from progress_model.time_series import generate_time_series_from_dict as _gen_ts
                data = load_time_series_data(input_data_path)
                rollout_data = _gen_ts(sampled_ts_params, input_data_path, data)
            else:
                # Use base data as-is
                rollout_data = load_time_series_data(input_data_path)
            profiling['time_series_gen'] = time_module.perf_counter() - t0

            # Run the rollout (always in-process when called from pool worker)
            # Timeout is enforced by signal handler in _run_single_rollout_with_timeout wrapper
            t0 = time_module.perf_counter()

            t1 = time_module.perf_counter()
            params_obj = Parameters(**sampled_params)
            profiling['params_init'] = time_module.perf_counter() - t1

            t1 = time_module.perf_counter()
            model = ProgressModel(params_obj, rollout_data)
            profiling['model_init'] = time_module.perf_counter() - t1

            t1 = time_module.perf_counter()
            times, progress_values, research_stock_values = model.compute_progress_trajectory(
                time_range, initial_progress
            )
            profiling['trajectory_compute'] = time_module.perf_counter() - t1

            # Add computed r_software to sampled_params for future analysis
            sampled_params["r_software"] = model.params.r_software

            t1 = time_module.perf_counter()
            rollout_results = _to_jsonable(model.results)
            profiling['results_serialization'] = time_module.perf_counter() - t1

            profiling['model_exec_total'] = time_module.perf_counter() - t0

            return {
                'sample_id': sample_id,
                'parameters': _to_jsonable(sampled_params),
                'time_series_parameters': _to_jsonable(sampled_ts_params),
                'results': rollout_results,
                'profiling': profiling,
                'success': True,
            }

        except (TimeoutError, _TimeoutError) as e:
            return {
                'sample_id': sample_id,
                'parameters': _to_jsonable(sampled_params),
                'time_series_parameters': _to_jsonable(sampled_ts_params),
                'results': None,
                'error': str(e),
                'success': False,
                'timeout': True,
            }
        except Exception as e:
            import traceback
            return {
                'sample_id': sample_id,
                'parameters': _to_jsonable(sampled_params),
                'time_series_parameters': _to_jsonable(sampled_ts_params),
                'results': None,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False,
                'timeout': False,
            }


def main() -> None:
    args = parse_args()

    # Load distribution config or build default
    user_cfg = _load_config(args.config)
    if not user_cfg:
        user_cfg = _build_default_distribution_config()

    # Resolve effective run settings (CLI overrides config)
    # Seed
    if _flag_provided("--seed"):
        seed = int(args.seed)
    else:
        seed = int(user_cfg.get("seed", 42))

    # Number of samples
    if _flag_provided("--num-samples"):
        num_samples = int(args.num_samples)
    else:
        num_samples = int(user_cfg.get("num_samples", user_cfg.get("num_rollouts", args.num_samples)))

    # Input data path
    if _flag_provided("--input-data"):
        input_data_path = str(args.input_data)
    else:
        input_data_path = str(user_cfg.get("input_data", args.input_data))

    # Initial progress
    if _flag_provided("--initial-progress"):
        initial_progress = float(args.initial_progress)
    else:
        initial_progress = float(user_cfg.get("initial_progress", 0.0))

    # Per-sample timeout (seconds)
    if _flag_provided("--per-sample-timeout"):
        per_sample_timeout = float(args.per_sample_timeout) if args.per_sample_timeout is not None else None
    else:
        _cfg_timeout = user_cfg.get("per_sample_timeout")
        per_sample_timeout = float(_cfg_timeout) if _cfg_timeout is not None else None

    # Prepare output directories
    # Create master SeedSequence for spawning independent streams
    master_ss = SeedSequence(seed)
    base_out = Path(args.output_dir)
    _ensure_dir(base_out)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / timestamp
    _ensure_dir(run_dir)

    # Load time series data once
    data = load_time_series_data(input_data_path)

    # Determine time range
    if _flag_provided("--time-range") and args.time_range is not None:
        time_range = [float(args.time_range[0]), float(args.time_range[1])]
    else:
        cfg_time_range = user_cfg.get("time_range")
        if cfg_time_range is not None:
            time_range = [float(cfg_time_range[0]), float(cfg_time_range[1])]
        else:
            # Default to full available range in input_data.csv
            time_range = [float(np.min(data.time)), float(np.max(data.time))]

    # Persist input config for reproducibility
    # Update user_cfg snapshot to reflect effective settings
    user_cfg["seed"] = seed
    user_cfg["initial_progress"] = initial_progress
    user_cfg["time_range"] = time_range
    user_cfg["num_samples"] = num_samples
    user_cfg["input_data"] = str(Path(input_data_path).resolve())
    user_cfg["per_sample_timeout"] = per_sample_timeout
    input_cfg_path = run_dir / "input_distributions.yaml"
    input_cfg_path.write_text(yaml.safe_dump(user_cfg))

    # Snapshot model_config
    _snapshot_model_config(run_dir)

    # Prepare output files for samples and rollouts
    samples_out_path = run_dir / "samples.jsonl"
    rollouts_out_path = run_dir / "rollouts.jsonl"

    # Avoid holding everything in memory: stream as NDJSON
    param_dists: Dict[str, Any] = user_cfg.get("parameters", {})
    ts_param_dists: Dict[str, Any] = user_cfg.get("time_series_parameters", {})
    correlation_matrix: Optional[Dict[str, Any]] = user_cfg.get("correlation_matrix")

    # print(param_dists["saturation_horizon_minutes"])

    # Write minimal metadata
    _write_json(run_dir / "metadata.json", {
        "created_at": datetime.now().isoformat(),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "input_data_path": str(Path(input_data_path).resolve()),
        "time_range": time_range,
        "per_sample_timeout": per_sample_timeout,
    })

    # Export correlation matrix for results pipeline (if present)
    if correlation_matrix is not None:
        corr_export = {
            "parameters": correlation_matrix.get("parameters", []),
            "correlation_matrix": correlation_matrix.get("correlation_matrix", []),
            "correlation_type": correlation_matrix.get("correlation_type", "spearman"),
        }
        _write_json(run_dir / "correlation_matrix.json", corr_export)

    # Precompute latent correlation (Gaussian space) once, if correlations are configured
    if correlation_matrix is not None:
        try:
            correlated_params = correlation_matrix.get("parameters", [])
            target_corr = correlation_matrix.get("correlation_matrix")
            if correlated_params and target_corr is not None:
                # Allow specifying interpretation of provided correlations: 'pearson' (default) or 'spearman'
                interpretation = str(correlation_matrix.get("correlation_type", correlation_matrix.get("interpretation", "spearman")))
                calib_samples = int(correlation_matrix.get("calibration_samples", 20000))
                # Spawn a dedicated sequence for calibration to keep it independent from sample streams
                calib_ss = master_ss.spawn(1)[0]
                calib_rng = np.random.default_rng(calib_ss)
                latent = _calibrate_gaussian_copula_latent_corr(
                    param_dists,
                    correlated_params,
                    np.array(target_corr, dtype=float),
                    calib_rng,
                    interpretation=interpretation,
                    calibration_samples=calib_samples,
                )
                correlation_matrix["latent_correlation_matrix"] = latent.tolist()
        except Exception as e:
            # If calibration fails, fall back to user-provided matrix; sampling will validate PSD
            print(f"WARNING: correlation calibration failed; using provided correlation_matrix. Error: {e}")

    # Merge parameter distributions to enable cross-group correlations
    # All parameters are sampled together, then split back into their respective groups
    all_param_dists = {}
    all_param_dists.update(param_dists)
    all_param_dists.update(ts_param_dists)
    
    # Track which parameters belong to which group for later splitting
    model_param_names = set(param_dists.keys())
    ts_param_names = set(ts_param_dists.keys())

    # Determine number of workers
    if args.no_parallel:
        num_workers = 1
        print("Parallel execution disabled (--no-parallel)")
    elif _flag_provided("--num-workers") and args.num_workers is not None:
        num_workers = int(args.num_workers)
        print(f"Using {num_workers} parallel worker(s)")
    else:
        # OPTIMIZED: Default to CPU count (not CPU count - 2) since fork context is lightweight
        num_workers = max(1, os.cpu_count())
        print(f"Using {num_workers} parallel worker(s) [OPTIMIZED default]")

    # Determine multiprocessing context
    if args.mp_context == "auto":
        # Use fork on Linux/Unix for better performance, spawn elsewhere
        import platform
        if platform.system() == "Linux":
            mp_context_name = "fork"
            print("Using 'fork' multiprocessing context (Linux auto-default for speed)")
        else:
            mp_context_name = "spawn"
            print(f"Using 'spawn' multiprocessing context (non-Linux auto-default)")
    else:
        mp_context_name = args.mp_context
        print(f"Using '{mp_context_name}' multiprocessing context (user-specified)")

    # Calculate optimal chunksize if not specified
    if args.pool_chunksize is None:
        # Auto-tune: aim for ~4 chunks per worker to balance overhead vs granularity
        pool_chunksize = max(1, int(num_samples) // (num_workers * 4))
        print(f"Using auto-calculated pool chunksize: {pool_chunksize}")
    else:
        pool_chunksize = args.pool_chunksize
        print(f"Using user-specified pool chunksize: {pool_chunksize}")

    # I/O buffer settings
    io_buffer_size = args.io_buffer_size
    print(f"I/O buffer size: {io_buffer_size} (flush every {io_buffer_size} results)")

    # Worker lifetime setting
    maxtasksperchild = args.maxtasksperchild
    print(f"Worker process lifetime: {maxtasksperchild} tasks per child")

    # Set default timeout if not specified to prevent infinite hangs
    if per_sample_timeout is None or per_sample_timeout <= 0:
        per_sample_timeout = 120.0  # Default 2-minute timeout per rollout
        print(f"Using default per-sample timeout: {per_sample_timeout}s")
    else:
        print(f"Using per-sample timeout: {per_sample_timeout}s")

    # Profiling: track timing for different operations
    profiling_times = defaultdict(list)

    # Run median trajectory FIRST (before Monte Carlo rollouts)
    run_median_trajectory_and_save(
        param_dists=param_dists,
        ts_param_dists=ts_param_dists,
        input_data_path=input_data_path,
        time_range=time_range,
        initial_progress=initial_progress,
        output_dir=run_dir,
    )

    # Prepare arguments for all rollouts
    # Spawn independent SeedSequences for each sample to guarantee statistical independence
    sample_seeds = master_ss.spawn(int(num_samples))
    rollout_args = [
        (i, sample_seeds[i], all_param_dists, correlation_matrix, str(Path(input_data_path).resolve()),
         time_range, initial_progress, per_sample_timeout, model_param_names, ts_param_names)
        for i in range(int(num_samples))
    ]

    # Sampling and rollouts with parallel execution
    with samples_out_path.open("w", encoding="utf-8") as f_samples, rollouts_out_path.open("w", encoding="utf-8") as f_rollouts:
        if num_workers == 1:
            # Sequential execution (no parallelism)
            # Use signal-based timeout when --no-parallel to avoid subprocess overhead
            use_signal_timeout = args.no_parallel
            progress_iter = _tqdm(rollout_args, desc="Rollouts", unit="run") if _tqdm is not None else rollout_args
            for rollout_arg in progress_iter:
                if use_signal_timeout:
                    result = _run_single_rollout_with_signal_timeout(rollout_arg)
                else:
                    result = _run_single_rollout_with_timeout(rollout_arg)

                # Extract profiling data
                if 'profiling' in result:
                    for key, value in result['profiling'].items():
                        profiling_times[key].append(value)

                # Write results
                t0 = time_module.perf_counter()
                sample_record = {
                    "sample_id": result['sample_id'],
                    "parameters": result['parameters'],
                    "time_series_parameters": result['time_series_parameters'],
                }
                if not result['success']:
                    sample_record['error'] = result.get('error')
                    if 'traceback' in result:
                        sample_record['traceback'] = result['traceback']

                f_samples.write(json.dumps(sample_record) + "\n")
                f_samples.flush()

                rollout_record = {
                    "sample_id": result['sample_id'],
                    "parameters": result['parameters'],
                    "time_series_parameters": result['time_series_parameters'],
                    "results": result['results'],
                }
                if not result['success']:
                    rollout_record['error'] = result.get('error')
                    if 'traceback' in result:
                        rollout_record['traceback'] = result['traceback']

                f_rollouts.write(json.dumps(rollout_record) + "\n")
                f_rollouts.flush()
                profiling_times['io_write'].append(time_module.perf_counter() - t0)

                # Print errors
                if not result['success']:
                    print(f"Rollout {result['sample_id']} failed: {result.get('error')}")
                    if result.get('traceback') and not result.get('timeout'):
                        print(result['traceback'])

        else:
            # Parallel execution using module-level non-daemon pool classes
            completed_count = 0
            failed_count = 0
            if _tqdm is not None:
                progress_bar = _tqdm(total=num_samples, desc="Rollouts", unit="run")

            # OPTIMIZED: Use configurable context and larger chunksize/maxtasksperchild
            ctx = mp.get_context(mp_context_name)
            pool = _NoDaemonPool(processes=num_workers, maxtasksperchild=maxtasksperchild, context=ctx)

            # OPTIMIZED: Choose worker function based on context
            # Fork context: use signal-based timeout (no subprocess overhead)
            # Spawn context: use subprocess-based timeout (safer but slower)
            if mp_context_name == "fork":
                worker_func = _run_single_rollout_optimized
                print(f"Using signal-based timeout (no subprocess overhead)")
            else:
                worker_func = _run_single_rollout_with_timeout
                print(f"Using subprocess-based timeout (spawn context safety)")

            # OPTIMIZED: Buffer for batched I/O
            sample_buffer = []
            rollout_buffer = []

            # Result collection timeout (per result, in seconds)
            result_timeout = (per_sample_timeout * 2) if per_sample_timeout else 60.0

            def process_result(result):
                """Process a single result and write to buffers."""
                nonlocal completed_count, failed_count

                # Extract profiling data
                if 'profiling' in result:
                    for key, value in result['profiling'].items():
                        profiling_times[key].append(value)

                # OPTIMIZED: Buffer results instead of immediate write
                t0 = time_module.perf_counter()
                sample_record = {
                    "sample_id": result['sample_id'],
                    "parameters": result['parameters'],
                    "time_series_parameters": result['time_series_parameters'],
                }
                if not result.get('success', False):
                    sample_record['error'] = result.get('error')
                    if 'traceback' in result:
                        sample_record['traceback'] = result['traceback']

                sample_buffer.append(json.dumps(sample_record) + "\n")

                rollout_record = {
                    "sample_id": result['sample_id'],
                    "parameters": result['parameters'],
                    "time_series_parameters": result['time_series_parameters'],
                    "results": result['results'],
                }
                if not result.get('success', False):
                    rollout_record['error'] = result.get('error')
                    if 'traceback' in result:
                        rollout_record['traceback'] = result['traceback']

                rollout_buffer.append(json.dumps(rollout_record) + "\n")

                # Flush buffers when they reach the configured size
                if len(sample_buffer) >= io_buffer_size:
                    f_samples.writelines(sample_buffer)
                    f_samples.flush()
                    sample_buffer.clear()

                    f_rollouts.writelines(rollout_buffer)
                    f_rollouts.flush()
                    rollout_buffer.clear()

                profiling_times['io_write'].append(time_module.perf_counter() - t0)

                # Update progress
                completed_count += 1
                if _tqdm is not None:
                    progress_bar.update(1)
                else:
                    pct = int(100 * completed_count / num_samples)
                    print(f"\rRollouts: {completed_count}/{num_samples} ({pct}%)", end="", flush=True)

                # Print errors
                if not result.get('success', False):
                    failed_count += 1
                    if _tqdm is not None:
                        progress_bar.write(f"Rollout {result['sample_id']} failed: {result.get('error')}")
                    else:
                        print(f"\nRollout {result['sample_id']} failed: {result.get('error')}")
                    if result.get('traceback') and not result.get('timeout'):
                        if _tqdm is not None:
                            progress_bar.write(result['traceback'])
                        else:
                            print(result['traceback'])

            try:
                # Use apply_async and collect results as they complete (not in order)
                async_results = {}
                for task_args in rollout_args:
                    sample_id = task_args[0]
                    async_results[sample_id] = pool.apply_async(worker_func, (task_args,))

                # Poll for completed results with global timeout
                pending = set(async_results.keys())
                poll_interval = 0.05  # 50ms
                # Global timeout: expected time + generous buffer (at least 60s)
                expected_time = num_samples * (per_sample_timeout or 30) / max(num_workers, 1)
                global_timeout = max(expected_time * 2 + 30, 60.0)
                start_time = time_module.perf_counter()

                while pending:
                    elapsed = time_module.perf_counter() - start_time
                    if elapsed > global_timeout:
                        # Global timeout - mark all pending as failed
                        for sample_id in list(pending):
                            result = {
                                'sample_id': sample_id,
                                'parameters': None,
                                'time_series_parameters': None,
                                'results': None,
                                'error': f"Global timeout after {elapsed:.1f}s",
                                'success': False,
                                'timeout': True,
                            }
                            process_result(result)
                        break

                    # Check each pending result
                    newly_done = []
                    for sample_id in list(pending):
                        async_result = async_results[sample_id]
                        if async_result.ready():
                            newly_done.append(sample_id)
                            try:
                                result = async_result.get(timeout=1)
                                process_result(result)
                            except Exception as e:
                                result = {
                                    'sample_id': sample_id,
                                    'parameters': None,
                                    'time_series_parameters': None,
                                    'results': None,
                                    'error': f"Worker error: {str(e)}",
                                    'success': False,
                                    'timeout': False,
                                }
                                process_result(result)

                    for sample_id in newly_done:
                        pending.discard(sample_id)

                    if pending and not newly_done:
                        time_module.sleep(poll_interval)

            except KeyboardInterrupt:
                print("\nInterrupted by user, cleaning up...")
            finally:
                # OPTIMIZED: Flush any remaining buffered data
                if sample_buffer:
                    f_samples.writelines(sample_buffer)
                    f_samples.flush()
                    sample_buffer.clear()
                if rollout_buffer:
                    f_rollouts.writelines(rollout_buffer)
                    f_rollouts.flush()
                    rollout_buffer.clear()

                # Terminate pool immediately to avoid hangs
                try:
                    pool.terminate()
                except Exception:
                    pass
                # Brief wait for cleanup
                try:
                    pool.join()
                except Exception:
                    pass

            if _tqdm is not None:
                progress_bar.close()
            else:
                print()  # Newline after progress

            if failed_count > 0:
                print(f"WARNING: {failed_count}/{num_samples} rollouts failed")

    # Print profiling summary
    print("\n" + "="*80)
    print("PROFILING SUMMARY (OPTIMIZED VERSION)")
    print("="*80)
    print(f"Optimizations applied:")
    print(f"  - Multiprocessing context: {mp_context_name}")
    print(f"  - Pool chunksize: {pool_chunksize}")
    print(f"  - I/O buffer size: {io_buffer_size}")
    print(f"  - Worker lifetime: {maxtasksperchild} tasks")
    print("="*80)
    for key in sorted(profiling_times.keys()):
        times = profiling_times[key]
        if times:
            mean_time = np.mean(times)
            median_time = np.median(times)
            p95_time = np.percentile(times, 95)
            total_time = np.sum(times)
            print(f"{key:25s}: mean={mean_time*1000:7.2f}ms  median={median_time*1000:7.2f}ms  "
                  f"p95={p95_time*1000:7.2f}ms  total={total_time:7.2f}s")

    # Calculate breakdown percentages for in-process execution
    if profiling_times['model_exec_total']:
        print("\n" + "-"*80)
        print("MODEL EXECUTION BREAKDOWN (in-process only)")
        print("-"*80)
        total_exec = np.sum(profiling_times['model_exec_total'])
        breakdown_keys = ['model_init', 'trajectory_compute', 'results_serialization']
        for key in breakdown_keys:
            if profiling_times[key]:
                portion = np.sum(profiling_times[key])
                pct = 100.0 * portion / total_exec if total_exec > 0 else 0
                print(f"{key:25s}: {portion:7.2f}s ({pct:5.1f}%)")

    # Calculate overall iteration breakdown
    if profiling_times['iteration_total']:
        print("\n" + "-"*80)
        print("ITERATION BREAKDOWN")
        print("-"*80)
        total_iter = np.sum(profiling_times['iteration_total'])
        iter_keys = ['sampling', 'time_series_gen', 'params_init', 'model_exec_total', 'io_write']
        for key in iter_keys:
            if profiling_times[key]:
                portion = np.sum(profiling_times[key])
                pct = 100.0 * portion / total_iter if total_iter > 0 else 0
                print(f"{key:25s}: {portion:7.2f}s ({pct:5.1f}%)")

    print("="*80 + "\n")

    print(f"Run complete. Artifacts in: {run_dir}")

    # Use new pipeline if requested
    if args.use_new_pipeline and not args.skip_summary:
        print("\nUsing new post-rollout pipeline...")
        python_exe = sys.executable
        try:
            pipeline_args = [
                python_exe, "-m", "pipelines.post_rollout",
                "--run-dir", str(run_dir)
            ]
            if not args.post_process:
                # If --post-process not set, run fast-only mode
                pipeline_args.append("--fast-only")
            subprocess.run(
                pipeline_args,
                cwd=str(SCRIPTS_DIR),
                check=True
            )
            print("\nNew pipeline complete!")
        except Exception as e:
            print(f"WARNING: New pipeline failed: {e}")
            print("Falling back to legacy post-processing...")
            args.use_new_pipeline = False  # Fall through to legacy

    # Generate rollout summary, batch plots, and milestone summary table (unless --skip-summary is set)
    if not args.skip_summary and not args.use_new_pipeline:
        python_exe = sys.executable

        # Create cache for faster post-processing (100-1000x smaller than JSONL)
        try:
            print("\nCreating milestone cache for faster post-processing...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "utilities" / "create_milestones_cache.py"),
                 str(run_dir / "rollouts.jsonl"), "--force"],
                cwd=str(REPO_ROOT),
                check=True
            )
            print("Cache created successfully!")
        except Exception as e:
            print(f"WARNING: Cache creation failed: {e}")

        # Run post-processing scripts in parallel for faster execution
        # Phase 1: Fast scripts that use cache (histograms, summaries, tables)
        parallel_scripts = {
            "Rollout summary": [python_exe, str(SCRIPTS_DIR / "analysis" / "rollout_summary.py"), str(run_dir)],
            "Batch histograms": [python_exe, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir), "--batch-all", "--histograms-only"],
            "Milestone summary table": [python_exe, str(SCRIPTS_DIR / "utilities" / "milestone_summary_table.py"), str(run_dir / "rollouts.jsonl")],
            "Short timelines analysis": [python_exe, str(SCRIPTS_DIR / "analysis" / "short_timelines_analysis.py"), "--run-dir", str(run_dir)],
            "Fast takeoff analysis": [python_exe, str(SCRIPTS_DIR / "analysis" / "fast_takeoff_analysis.py"), "--run-dir", str(run_dir)],
        }

        print(f"\nPhase 1: Running {len(parallel_scripts)} fast post-processing scripts in parallel...")

        # Use up to 4 workers (balances parallelism vs resource usage)
        max_workers = min(4, len(parallel_scripts))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_post_processing_script, name, cmd): name
                for name, cmd in parallel_scripts.items()
            }

            completed = 0
            for future in as_completed(futures):
                name, success, error_msg = future.result()
                completed += 1
                if success:
                    print(f"  [{completed}/{len(parallel_scripts)}] {name}: complete!")
                else:
                    print(f"  [{completed}/{len(parallel_scripts)}] WARNING: {name} failed: {error_msg}")

        # Phase 2: Slow scripts that require reading full JSONL (trajectory plots)
        print("\nPhase 2: Generating trajectory plots (requires full JSONL parsing)...")
        try:
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir), "--batch-horizon"],
                cwd=str(REPO_ROOT),
                check=True
            )
            print("Trajectory plots complete!")
        except Exception as e:
            print(f"WARNING: Trajectory plots failed: {e}")

    # Run additional post-processing scripts if requested
    if args.post_process:
        print("\nRunning post-processing scripts...")
        python_exe = sys.executable

        # 1. AC time histogram
        try:
            print("  - Generating AC time histogram...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir)],
                cwd=str(REPO_ROOT),
                check=True
            )
        except Exception as e:
            print(f"    WARNING: plot_rollouts failed: {e}")

        # 2. Median paradox analysis
        try:
            print("  - Generating median paradox analysis...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "explain_median_difference_paradox.py"), "--rollouts", str(run_dir / "rollouts.jsonl")],
                cwd=str(REPO_ROOT),
                check=True
            )
            print("  - Median paradox analysis complete!")
        except Exception as e:
            print(f"    WARNING: median paradox analysis failed: {e}")

        # 3. Horizon trajectories plots with MSE filtering (run in parallel)
        mse_thresholds = [10.0, 1.5, 1.0, 0.7, 0.5]
        print(f"  - Generating {len(mse_thresholds)} horizon trajectories plots in parallel...")

        horizon_scripts = {
            f"Horizon (MSE<={t})": [
                python_exe, str(SCRIPTS_DIR / "plot_rollouts.py"), "--run-dir", str(run_dir),
                "--mode", "horizon_trajectories", "--shade-by-mse", "--mse-threshold", str(t)
            ]
            for t in mse_thresholds
        }

        with ProcessPoolExecutor(max_workers=min(4, len(mse_thresholds))) as executor:
            futures = {
                executor.submit(_run_post_processing_script, name, cmd): name
                for name, cmd in horizon_scripts.items()
            }
            for future in as_completed(futures):
                name, success, error_msg = future.result()
                if success:
                    print(f"    {name}: complete!")
                else:
                    print(f"    WARNING: {name} failed: {error_msg}")

        # 4. Sensitivity analysis - run both censored and complete-only versions
        # First, compute median simulation end time for censored analysis (using cache for speed)
        try:
            print("  - Computing median simulation end time...")
            sys.path.insert(0, str(SCRIPTS_DIR))
            from plotting_utils.rollouts_reader import RolloutsReader
            reader = RolloutsReader(run_dir / "rollouts.jsonl")
            sim_ends = []
            for rollout in reader.iter_normalized_rollouts():
                sim_ends.append(rollout.simulation_end)
            median_sim_end = float(np.median(sim_ends)) if sim_ends else 2050.0
            print(f"    Median simulation end time: {median_sim_end}")
        except Exception as e:
            print(f"    WARNING: Could not compute median sim end, using 2050.0: {e}")
            median_sim_end = 2050.0

        # 3a. Censored sensitivity analysis (treats non-achievement as achieved at sim end)
        try:
            print(f"  - Running censored sensitivity analysis (dummy year = {median_sim_end})...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "sensitivity_analysis.py"), "--run-dir", str(run_dir), "--plot",
                 "--upper-dummy-year", str(median_sim_end), "--target-statistic", "aa_time"],
                cwd=str(REPO_ROOT),
                check=True
            )
            # Rename output to include "_censored" suffix
            sensitivity_json = run_dir / "sensitivity_summary.json"
            if sensitivity_json.exists():
                sensitivity_json.rename(run_dir / "sensitivity_summary_censored.json")
        except Exception as e:
            print(f"    WARNING: censored sensitivity_analysis failed: {e}")

        # 3b. Complete-runs-only sensitivity analysis (effectively excludes non-achievers)
        try:
            print("  - Running complete-runs-only sensitivity analysis (dummy year = 1e10)...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "sensitivity_analysis.py"), "--run-dir", str(run_dir), "--plot",
                 "--upper-dummy-year", "10000000000", "--target-statistic", "aa_time"],
                cwd=str(REPO_ROOT),
                check=True
            )
            # Rename output to include "_complete_only" suffix
            sensitivity_json = run_dir / "sensitivity_summary.json"
            if sensitivity_json.exists():
                sensitivity_json.rename(run_dir / "sensitivity_summary_complete_only.json")
        except Exception as e:
            print(f"    WARNING: complete-runs-only sensitivity_analysis failed: {e}")

        # 5. SC by quarter table
        try:
            print("  - Generating SC-by-quarter table...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "sc_by_quarter.py"), "--run-dir", str(run_dir)],
                cwd=str(REPO_ROOT),
                check=True
            )
        except Exception as e:
            print(f"    WARNING: sc_by_quarter failed: {e}")

        # 6. Slack summary table for all MSE thresholds
        try:
            print("  - Generating slack summary table...")
            sys.path.insert(0, str(SCRIPTS_DIR))
            from slack_summary import compute_slack_summary, format_year
            import csv

            mse_thresholds_for_summary = [10.0, 1.5, 1.0, 0.7, 0.5]
            milestone = "AC"
            rollouts_file = run_dir / "rollouts.jsonl"

            # Collect results
            summary_results = []
            for mse_threshold in mse_thresholds_for_summary:
                try:
                    result = compute_slack_summary(rollouts_file, mse_threshold, milestone)
                    summary_results.append(result)
                except Exception as e:
                    print(f"    WARNING: Could not compute summary for MSE <= {mse_threshold}: {e}")

            if summary_results:
                # Write to CSV
                csv_path = run_dir / "slack_summary.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # Header
                    writer.writerow([
                        "MSE Threshold",
                        "N Trajectories",
                        f"{milestone} Median",
                        f"{milestone} P10",
                        f"{milestone} P90",
                        "10y Horizon Median",
                        "10y Horizon P10",
                        "10y Horizon P90",
                        "100y Horizon Median",
                        "100y Horizon P10",
                        "100y Horizon P90"
                    ])
                    # Rows
                    for result in summary_results:
                        writer.writerow([
                            f"<= {result['mse_threshold']}",
                            result['num_trajectories'],
                            format_year(result['milestone_median']) if result['milestone_median'] is not None else "N/A",
                            format_year(result['milestone_p10']) if result['milestone_p10'] is not None else "N/A",
                            format_year(result['milestone_p90']) if result['milestone_p90'] is not None else "N/A",
                            format_year(result['horizon_10y_median']) if result['horizon_10y_median'] is not None else "N/A",
                            format_year(result['horizon_10y_p10']) if result['horizon_10y_p10'] is not None else "N/A",
                            format_year(result['horizon_10y_p90']) if result['horizon_10y_p90'] is not None else "N/A",
                            format_year(result['horizon_100y_median']) if result['horizon_100y_median'] is not None else "N/A",
                            format_year(result['horizon_100y_p10']) if result['horizon_100y_p10'] is not None else "N/A",
                            format_year(result['horizon_100y_p90']) if result['horizon_100y_p90'] is not None else "N/A",
                        ])
                print(f"    Saved slack summary table to {csv_path}")

                # Also print to console
                print("\n    === Slack Summary Table ===")
                for result in summary_results:
                    print(f"\n    MSE <= {result['mse_threshold']} ({result['num_trajectories']} trajectories):")
                    if result['milestone_median'] is not None:
                        print(f"      {milestone}: {format_year(result['milestone_median'])} [{format_year(result['milestone_p10'])}, {format_year(result['milestone_p90'])}]")
                    else:
                        print(f"      {milestone}: N/A")
                    if result['horizon_10y_median'] is not None:
                        print(f"      10y horizons: {format_year(result['horizon_10y_median'])} [{format_year(result['horizon_10y_p10'])}, {format_year(result['horizon_10y_p90'])}]")
                    else:
                        print(f"      10y horizons: N/A")
                    if result['horizon_100y_median'] is not None:
                        print(f"      100y horizons: {format_year(result['horizon_100y_median'])} [{format_year(result['horizon_100y_p10'])}, {format_year(result['horizon_100y_p90'])}]")
                    else:
                        print(f"      100y horizons: N/A")
        except Exception as e:
            print(f"    WARNING: slack_summary generation failed: {e}")

        # 7. Split by growth factor and generate plots
        try:
            print("  - Generating growth factor split analysis...")
            subprocess.run(
                [python_exe, str(SCRIPTS_DIR / "split_by_growth_factor.py"),
                 "--run-dir", str(run_dir), "--mse-threshold", "1.5"],
                cwd=str(REPO_ROOT),
                check=True
            )
        except Exception as e:
            print(f"    WARNING: split_by_growth_factor failed: {e}")

        print("Post-processing complete!")


if __name__ == "__main__":
    main()

