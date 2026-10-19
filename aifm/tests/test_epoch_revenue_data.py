"""
Observed Anthropic revenue datapoints (epoch_anthropic_revenue.csv).

These are Epoch AI's reported annualized-revenue figures, overlaid on the
Predicted AI Revenue chart so the extrapolated trend can be eyeballed against
what has actually been reported. They are DISPLAY ONLY.

Two things are worth pinning:

1. UNITS. `present_revenue` is in DOLLARS ($4.7e10 = $47B), while the Q2
   elicitation sheet quotes trillions ($0.047T). We have already had one silent
   units bug across that boundary. The magnitude assertions below fail loudly if
   anyone re-imports this file in trillions, billions, or millions.

2. DISPLAY-ONLY-NESS. The revenue trend is anchored purely on the user's
   `present_revenue` / `present_revenue_growth_rate` parameters. If somebody
   later least-squares-fits the trend to these points, the anchoring identity
   `predict(anchor_progress) == present_revenue` stops holding exactly, and no
   model code should ever have read the datapoints in the first place.

The generated TypeScript module (utils/revenueData.ts) is also checked against
the CSV, since drift there would silently plot stale numbers.
"""

import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from progress_model import Parameters, fit_revenue_trend  # noqa: E402

CSV_PATH = PROJECT_ROOT / "inputs" / "epoch" / "epoch_anthropic_revenue.csv"
TS_PATH = PROJECT_ROOT / "utils" / "revenueData.ts"

# A whole-company annualized revenue for a frontier AI lab lives between $10M and
# $10T. Anything outside that is a units error, not a business outcome:
#   trillions  ($0.047T)  -> 4.7e-2   (below the floor)
#   billions   ($47B)     -> 4.7e1    (below the floor)
#   millions   ($47000M)  -> 4.7e4    (below the floor)
#   dollars    ($47B)     -> 4.7e10   (inside)
MIN_PLAUSIBLE_USD = 1e7
MAX_PLAUSIBLE_USD = 1e13


def _rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def rows():
    return _rows()


class TestCuratedCsv:
    def test_columns_are_the_documented_schema(self, rows):
        assert list(rows[0].keys()) == [
            "company",
            "date",
            "annualized_revenue_usd",
            "revenue_type",
            "confidence",
            "source_type",
            "source_url",
            "report_date",
        ]

    def test_every_row_is_anthropic_whole_company(self, rows):
        # A product-line run rate (e.g. "Claude Code $1B") is not comparable to the
        # model's whole-company `present_revenue`, so it must not be in here.
        assert {row["company"] for row in rows} == {"Anthropic"}

    def test_values_are_in_dollars_not_trillions_billions_or_millions(self, rows):
        for row in rows:
            value = float(row["annualized_revenue_usd"])
            assert MIN_PLAUSIBLE_USD <= value <= MAX_PLAUSIBLE_USD, (
                f"{row['date']}: {value:g} is outside [{MIN_PLAUSIBLE_USD:g}, "
                f"{MAX_PLAUSIBLE_USD:g}] USD -- almost certainly a units error"
            )

    def test_values_are_whole_dollars(self, rows):
        for row in rows:
            assert float(row["annualized_revenue_usd"]).is_integer(), row["date"]

    def test_dates_are_iso_sorted_and_not_in_the_future_of_the_report(self, rows):
        dates = [date.fromisoformat(row["date"]) for row in rows]
        assert dates == sorted(dates)
        for row, observed in zip(rows, dates):
            reported = date.fromisoformat(row["report_date"])
            # A run rate is always reported on or after the date it describes.
            assert reported >= observed, f"{row['date']} reported {row['report_date']}"

    def test_every_row_carries_a_source_url(self, rows):
        for row in rows:
            assert row["source_url"].startswith("https://"), row["date"]


class TestGeneratedTypeScriptMatchesCsv:
    """utils/revenueData.ts is generated from the CSV; drift plots stale numbers."""

    @staticmethod
    def _ts_pairs():
        text = TS_PATH.read_text(encoding="utf-8")
        dates = re.findall(r'^\s*date: "([^"]+)",$', text, flags=re.MULTILINE)
        values = re.findall(r"^\s*annualizedRevenueUsd: (\d+),$", text, flags=re.MULTILINE)
        return list(zip(dates, [float(v) for v in values]))

    def test_same_datapoints_in_the_same_order(self, rows):
        csv_pairs = [(row["date"], float(row["annualized_revenue_usd"])) for row in rows]
        assert self._ts_pairs() == csv_pairs

    def test_attribution_survives_regeneration(self):
        text = TS_PATH.read_text(encoding="utf-8")
        # CC-BY obliges us to credit Epoch wherever the data is shown; the chart
        # tooltip reads this constant.
        assert "Epoch AI, Data on AI Companies (CC-BY 4.0)" in text
        assert "https://epoch.ai/data/ai-companies" in text


class TestDatapointsAreDisplayOnly:
    def test_no_model_or_api_code_reads_the_datapoints(self):
        # The overlay must never become an input. Guard the boundary structurally:
        # only the frontend bridge, the frontend loader and this test may name it.
        allowed = {
            Path("scripts/bridge/epoch_revenue_to_ts.py"),
            Path("tests/test_epoch_revenue_data.py"),
        }
        offenders = []
        for directory in ("progress_model", "api", "scripts/pipelines", "scripts/analysis"):
            for path in (PROJECT_ROOT / directory).rglob("*.py"):
                relative = path.relative_to(PROJECT_ROOT)
                if relative in allowed:
                    continue
                if "epoch_anthropic_revenue" in path.read_text(encoding="utf-8"):
                    offenders.append(str(relative))
        assert offenders == [], (
            f"{offenders} reference the observed-revenue datapoints; they are "
            "display-only and must not reach the model or the API"
        )

    def test_the_revenue_trend_is_anchored_on_the_parameter_not_on_the_data(self, rows):
        # If the trend were ever fit to the observed points, it would no longer pass
        # exactly through present_revenue at the anchor. Pin the identity, and pin
        # that the fit is insensitive to the data by construction: the same
        # present_revenue yields the same curve regardless of what the CSV says.
        times = np.linspace(2018.0, 2040.0, 4000)
        progress = 0.45 * (times - 2018.0) + 0.015 * (times - 2018.0) ** 2
        rates = 0.45 + 0.03 * (times - 2018.0)

        present_revenue = 4.7e10
        params = Parameters(
            ac_anchor_mode="revenue",
            present_revenue=present_revenue,
            present_revenue_growth_rate=5.0,
            # Undated: this test is about WHAT the trend is fitted to, not WHEN it is
            # anchored. The shipped default states a revenue vintage (2026.374429), and
            # leaving it in would anchor the fit there while `anchor_rate` below is read
            # at present_day, so "one year on" would be one year at the wrong rate.
            revenue_reference_date=None,
        )
        anchor_rate = float(np.interp(params.present_day, times, rates))
        observable = fit_revenue_trend(params, times, progress, anchor_rate, rates)

        assert observable is not None
        anchor_progress = observable.fit_info["anchor_progress"]
        assert float(observable.predict(anchor_progress)) == pytest.approx(
            present_revenue, rel=1e-12
        )

        # One year later (in anchor-rate progress units) the trend must have grown by
        # exactly the parameter's growth factor -- a data-driven fit would not.
        one_year = anchor_progress + anchor_rate
        assert float(observable.predict(one_year)) == pytest.approx(
            present_revenue * 5.0, rel=1e-12
        )

    def test_the_default_present_revenue_is_the_same_order_as_the_latest_datapoint(self, rows):
        # The real guard against the trillions/dollars mix-up: the model's default
        # and Epoch's latest figure describe the same quantity, so they must agree
        # to within an order of magnitude. A units slip is a factor of 1e3 or more.
        latest = float(rows[-1]["annualized_revenue_usd"])
        default_revenue = float(Parameters().present_revenue)
        assert 0.1 <= default_revenue / latest <= 10.0, (
            f"Parameters().present_revenue={default_revenue:g} vs latest Epoch "
            f"datapoint {latest:g} -- check units (dollars, not trillions)"
        )

    def test_frontend_defaults_agree_with_the_python_default(self):
        # constants/parameters.ts is the frontend's source of truth for the slider;
        # if it drifts into trillions the chart points would sit 1e12 off the curve.
        text = (PROJECT_ROOT / "constants" / "parameters.ts").read_text(encoding="utf-8")
        values = {
            float(match)
            for match in re.findall(r"^\s*present_revenue:\s*([0-9.eE+-]+),", text, flags=re.MULTILINE)
        }
        assert values, "no present_revenue defaults found in constants/parameters.ts"
        for value in values:
            assert MIN_PLAUSIBLE_USD <= value <= MAX_PLAUSIBLE_USD, (
                f"constants/parameters.ts present_revenue={value:g} is not in dollars"
            )


def test_datapoints_bracket_the_trend_the_chart_draws():
    """The overlay is only useful if it lands on the plotted curve, not off-axis.

    The chart's y-axis is driven by the modelled series; a point three orders of
    magnitude away would be clipped and silently vanish. Assert the observed points
    sit within the band the default revenue trend sweeps over the same dates.
    """
    rows = _rows()
    times = np.linspace(2018.0, 2040.0, 4000)
    progress = 0.45 * (times - 2018.0) + 0.015 * (times - 2018.0) ** 2
    rates = 0.45 + 0.03 * (times - 2018.0)

    params = Parameters(
        ac_anchor_mode="revenue",
        present_revenue=4.7e10,
        present_revenue_growth_rate=5.0,
        present_day=2026.578082,
    )
    anchor_rate = float(np.interp(params.present_day, times, rates))
    observable = fit_revenue_trend(params, times, progress, anchor_rate, rates)
    assert observable is not None

    for row in rows:
        observed_date = date.fromisoformat(row["date"])
        decimal_year = observed_date.year + (observed_date.timetuple().tm_yday - 1) / 365.25
        trend_value = float(observable.predict(float(np.interp(decimal_year, times, progress))))
        observed_value = float(row["annualized_revenue_usd"])
        # Within 1.5 OOMs of the trend on a log axis spanning ~10 OOMs: close enough
        # that the marker lands on the visible curve rather than the axis edge.
        assert abs(np.log10(observed_value / trend_value)) < 1.5, (
            f"{row['date']}: observed {observed_value:g} vs trend {trend_value:g}"
        )
