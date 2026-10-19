"""
Cross-language golden test for the pause (freeze + reduce) transform.

progress_model.pause_transform.freeze_and_reduce_time_series is the Python
counterpart of the frontend applyPause (utils/pauseTransform.ts). This test pins
the two together: the golden fixture is produced by running the REAL applyPause
(see tests/gen_pause_transform_golden.ts), and here we feed the same base series
through the real Python function and assert the outputs match element-for-element.

Neither side reimplements the transform — the JS side is the source of truth for
the fixture, the Python side calls production code, and we compare values.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import assert_arrays_close, STRICT_RTOL
from progress_model.types import TimeSeriesData
from progress_model.pause_transform import freeze_and_reduce_time_series

GOLDEN_PATH = Path(__file__).parent / "golden_data" / "pause_transform_golden.json"

CHANNELS = ["time", "L_HUMAN", "inference_H100e", "experiment_H100e", "training_H100e"]


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _base_to_tsd(base: dict) -> TimeSeriesData:
    return TimeSeriesData(
        time=np.array(base["time"], dtype=float),
        L_HUMAN=np.array(base["L_HUMAN"], dtype=float),
        inference_H100e=np.array(base["inference_H100e"], dtype=float),
        experiment_H100e=np.array(base["experiment_H100e"], dtype=float),
        training_H100e=np.array(base["training_H100e"], dtype=float),
    )


def _case_ids(golden_data):
    return [c["name"] for c in golden_data["cases"]]


def test_golden_has_cases(golden):
    assert len(golden["cases"]) >= 1, "golden fixture has no cases"


@pytest.mark.parametrize("case_idx", range(6))
def test_pause_transform_matches_frontend(golden, case_idx):
    """Each golden case: Python freeze_and_reduce == frontend applyPause output."""
    cases = golden["cases"]
    if case_idx >= len(cases):
        pytest.skip("fewer golden cases than parametrized")
    case = cases[case_idx]
    base = _base_to_tsd(golden["base"])
    r = case["reductions"]

    result = freeze_and_reduce_time_series(
        base,
        case["pause_year"],
        training_reduction=r["training"],
        experiment_reduction=r["experiment"],
        inference_reduction=r["inference"],
        labor_reduction=r["labor"],
    )

    expected = case["output"]
    for channel in CHANNELS:
        actual_arr = getattr(result, channel)
        assert len(actual_arr) == len(expected[channel]), (
            f"[{case['name']}] channel {channel}: length "
            f"{len(actual_arr)} != golden {len(expected[channel])}"
        )
        assert_arrays_close(
            actual_arr,
            expected[channel],
            rtol=STRICT_RTOL,
            msg=f"[{case['name']}] channel {channel} mismatch vs frontend applyPause",
        )


def test_reduction_one_is_pure_freeze(golden):
    """A reduction factor of 1.0 must equal holding the pause-instant value flat
    (no scaling) — guards the divide-by-1 / subtract-log10(1) paths."""
    base = _base_to_tsd(golden["base"])
    result = freeze_and_reduce_time_series(base, 2027.4)  # all reductions default 1.0
    t = np.asarray(result.time)
    after = t > 2027.4
    # Every channel is constant strictly after the pause instant.
    for field in ["L_HUMAN", "inference_H100e", "experiment_H100e", "training_H100e"]:
        vals = np.asarray(getattr(result, field))[after]
        assert np.allclose(vals, vals[0], rtol=STRICT_RTOL), f"{field} not flat after pause"
