"""The uplift belief has exactly one container.

`serial_uplift_at_reference_date` (serial space, at `uplift_reference_date`) is
the only input. `swe_multiplier_at_present_day` is a derived view of it, and is
not a constructor argument.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import fields as dataclass_fields  # noqa: E402

from progress_model import Parameters, config as cfg  # noqa: E402
from progress_model.parameters import (  # noqa: E402
    resolve_swe_multiplier_at_present_day,
)

# Serial value -> the parallel value it implies, for the four frozen April presets.
SERIAL_TO_PARALLEL = {
    1.8: 1.3416407864998738,
    2.0: 1.4142135623730951,
}


def _params(**over):
    """Parameters with the dates pinned so the growth factor is 1 unless a test says otherwise."""
    base = dict(uplift_reference_date=None, present_day=2026.0959)
    base.update(over)
    return Parameters(**base)


class TestSingleContainer:
    def test_stated_serial_is_returned_verbatim(self):
        p = _params(serial_uplift_at_reference_date=1.7)
        assert p.serial_uplift_at_reference_date == 1.7

    def test_unstated_serial_falls_back_to_the_model_default_belief(self):
        expected = cfg.DEFAULT_PARAMETERS['serial_uplift_at_reference_date']
        p = _params()
        assert p.serial_uplift_at_reference_date == expected

    def test_parallel_view_is_derived_from_the_serial_value(self):
        # W = (1 + (S - 1) / g) ** (1 / pp); with g == 1 this is S ** (1 / pp).
        p = _params(serial_uplift_at_reference_date=1.3416407864998738)
        pp = p.parallel_penalty
        assert resolve_swe_multiplier_at_present_day(p) == pytest.approx(
            1.3416407864998738 ** (1.0 / pp), rel=0, abs=1e-12)

    @pytest.mark.parametrize("parallel,serial", sorted(SERIAL_TO_PARALLEL.items()))
    def test_preset_serial_values_imply_their_parallel_counterparts(self, parallel, serial):
        """The serial value each April preset states implies the parallel value it displays."""
        p = _params(serial_uplift_at_reference_date=serial)
        # One ulp, not bit-exact: (2.0 ** 0.5) ** 2 == 2.0000000000000004. The
        # model consumes the serial value, so only this derived view rounds.
        assert resolve_swe_multiplier_at_present_day(p) == pytest.approx(parallel, rel=1e-15)

    def test_derived_value_is_materialized_onto_the_instance(self):
        p = _params(serial_uplift_at_reference_date=1.4142135623730951)
        assert p.swe_multiplier_at_present_day == pytest.approx(2.0, rel=1e-15)


class TestParallelContainerCannotReturn:
    def test_parallel_multiplier_is_not_a_constructor_argument(self):
        """init=False keeps it out of api/index.py's _KNOWN_PARAM_KEYS, so a caller
        stating it cannot override the serial source of truth."""
        swe = [f for f in dataclass_fields(Parameters)
               if f.name == 'swe_multiplier_at_present_day']
        assert len(swe) == 1
        assert swe[0].init is False

    def test_stating_it_is_rejected_outright(self):
        with pytest.raises(TypeError):
            Parameters(swe_multiplier_at_present_day=1.8)

    def test_no_parallel_space_constant_is_consulted(self):
        assert not hasattr(cfg, 'LEGACY_SWE_MULTIPLIER_AT_PRESENT_DAY')


class TestGrowthFactor:
    def test_coincident_dates_give_no_date_conversion(self):
        """t* == present_day => g == 1, so the serial reading needs no shifting."""
        same = _params(serial_uplift_at_reference_date=1.5,
                       uplift_reference_date=2026.0959, present_day=2026.0959)
        null = _params(serial_uplift_at_reference_date=1.5)
        assert (resolve_swe_multiplier_at_present_day(same)
                == resolve_swe_multiplier_at_present_day(null))

    def test_a_reference_date_away_from_present_day_shifts_the_parallel_view(self):
        near = _params(serial_uplift_at_reference_date=1.5,
                       uplift_reference_date=2026.0959, present_day=2026.0959)
        far = _params(serial_uplift_at_reference_date=1.5,
                      uplift_reference_date=2027.0959, present_day=2026.0959,
                      excess_uplift_doubling_time=6.0)
        # A year of excess growth at a 6-month doubling time is a factor of 4, so
        # the same serial level read a year earlier implies a SMALLER present-day value.
        assert resolve_swe_multiplier_at_present_day(far) < resolve_swe_multiplier_at_present_day(near)
        excess_near = near.swe_multiplier_at_present_day ** near.parallel_penalty - 1.0
        excess_far = far.swe_multiplier_at_present_day ** far.parallel_penalty - 1.0
        assert excess_near / excess_far == pytest.approx(4.0, rel=1e-9)

    def test_finite_and_positive(self):
        p = _params(serial_uplift_at_reference_date=1.9)
        w = resolve_swe_multiplier_at_present_day(p)
        assert np.isfinite(w) and w > 1.0
