"""
The observed ECI frontier (epoch_capabilities_index.csv -> utils/eciData.ts).

Epoch's Capabilities Index scores, reduced to the FRONTIER -- the running maximum of
score over release date, one point per date, all labs competing -- and overlaid on the
Epoch Capabilities Index chart. DISPLAY ONLY.

What is worth pinning here:

1. THE FRONTIER RULE. The reduction runs at generation time in
   scripts/bridge/epoch_eci_to_ts.py, so the rule would otherwise be unobserved by
   any test. These tests recompute it from the CSV independently of the generated
   module and assert the properties that make it a frontier: strictly increasing,
   one point per date, and nothing dropped that should have been kept.

2. DISPLAY-ONLY-NESS. The ECI trend is anchored purely on `eci_at_present_day` and
   `eci_growth_rate`. No model or API code may read the datapoints, and the
   anchoring identity must hold exactly.

3. DRIFT. utils/eciData.ts is generated; if it falls behind the CSV the chart plots
   stale models.
"""

import csv
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from progress_model import Parameters, fit_eci_trend  # noqa: E402

CSV_PATH = PROJECT_ROOT / "inputs" / "epoch" / "epoch_capabilities_index.csv"
TS_PATH = PROJECT_ROOT / "utils" / "eciData.ts"

# ECI is a dimensionless index; the covered era runs roughly 100 (LLaMA-65B, 2023) to
# a bit over 160. Anything outside this window is a parse or units problem, not a
# capability result.
MIN_PLAUSIBLE_ECI = 50.0
MAX_PLAUSIBLE_ECI = 400.0


def _csv_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plottable(rows):
    usable = []
    for row in rows:
        score = (row.get("ECI Score") or "").strip()
        released = (row.get("Release date") or "").strip()
        if not score or not released:
            continue
        try:
            float(score)
        except ValueError:
            continue
        usable.append(row)
    return usable


def _frontier_from_csv(rows):
    """Recomputed here on purpose: an independent statement of the frontier rule."""
    by_date = defaultdict(list)
    for row in _plottable(rows):
        by_date[row["Release date"].strip()].append(row)

    frontier = []
    running_max = float("-inf")
    for release_date in sorted(by_date):
        best = max(
            by_date[release_date],
            key=lambda row: (float(row["ECI Score"]), row["Model version"]),
        )
        score = float(best["ECI Score"])
        if score > running_max:
            running_max = score
            frontier.append(best)
    return frontier


@pytest.fixture(scope="module")
def rows():
    return _csv_rows()


@pytest.fixture(scope="module")
def frontier(rows):
    return _frontier_from_csv(rows)


class TestSourceCsv:
    def test_schema_is_the_epoch_export_app_py_expects(self, rows):
        # app.py's process_eci_data() reads 'ECI Score' and 'Release date' straight off
        # this file, so a refresh must not reshape it.
        assert list(rows[0].keys()) == [
            "Model version",
            "ECI Score",
            "Release date",
            "Organization",
            "Country",
            "Model accessibility",
            "Training compute (FLOP)",
            "Confidence",
            "Model name",
            "Description",
            "Display name",
        ]

    def test_scores_are_index_points_in_the_plausible_band(self, rows):
        for row in _plottable(rows):
            score = float(row["ECI Score"])
            assert MIN_PLAUSIBLE_ECI <= score <= MAX_PLAUSIBLE_ECI, (
                f"{row['Model version']}: ECI {score:g} outside "
                f"[{MIN_PLAUSIBLE_ECI:g}, {MAX_PLAUSIBLE_ECI:g}]"
            )

    def test_most_rows_are_unusable_for_plotting_and_that_is_expected(self, rows):
        # Epoch ships every benchmarked model version, scored or not. Documenting the
        # ratio makes an unexpected collapse in coverage visible instead of silent.
        usable = _plottable(rows)
        assert len(rows) > 700
        assert 400 < len(usable) < len(rows)


class TestFrontierRule:
    def test_scores_strictly_increase(self, frontier):
        scores = [float(row["ECI Score"]) for row in frontier]
        assert all(b > a for a, b in zip(scores, scores[1:])), scores

    def test_release_dates_strictly_increase(self, frontier):
        dates = [date.fromisoformat(row["Release date"].strip()) for row in frontier]
        assert all(b > a for a, b in zip(dates, dates[1:])), dates

    def test_at_most_one_point_per_release_date(self, frontier):
        dates = [row["Release date"].strip() for row in frontier]
        assert len(dates) == len(set(dates))

    def test_every_frontier_model_beats_all_strictly_earlier_models(self, rows, frontier):
        usable = _plottable(rows)
        for row in frontier:
            released = date.fromisoformat(row["Release date"].strip())
            score = float(row["ECI Score"])
            earlier_best = max(
                (
                    float(other["ECI Score"])
                    for other in usable
                    if date.fromisoformat(other["Release date"].strip()) < released
                ),
                default=float("-inf"),
            )
            assert score > earlier_best, (
                f"{row['Model version']} ({score:g}) does not beat the pre-"
                f"{released} best of {earlier_best:g}"
            )

    def test_no_model_that_should_be_on_the_frontier_is_missing(self, rows, frontier):
        # The complement of the previous test: walk every usable row and confirm the
        # only reason it is absent is that something released no later already matched
        # or beat it.
        usable = _plottable(rows)
        kept = {(row["Release date"].strip(), row["Model version"]) for row in frontier}
        for row in usable:
            identity = (row["Release date"].strip(), row["Model version"])
            if identity in kept:
                continue
            released = date.fromisoformat(row["Release date"].strip())
            score = float(row["ECI Score"])
            best_no_later = max(
                float(other["ECI Score"])
                for other in usable
                if date.fromisoformat(other["Release date"].strip()) <= released
                and (other["Release date"].strip(), other["Model version"]) != identity
            )
            assert best_no_later >= score, (
                f"{row['Model version']} ({score:g} on {released}) was dropped but "
                f"nothing released by then reached it (best {best_no_later:g})"
            )

    def test_the_frontier_is_a_large_reduction(self, rows, frontier):
        usable = _plottable(rows)
        assert len(frontier) < len(usable) / 10, (
            f"{len(frontier)} frontier points out of {len(usable)} usable rows -- "
            "the reduction should be drastic; suspect a broken running maximum"
        )

    def test_multiple_organizations_hold_the_frontier(self, frontier):
        # The overlay is explicitly multi-lab; if this collapses to one org the mark
        # colouring is pointless and the filter has probably gone wrong.
        organizations = {row["Organization"].strip() for row in frontier}
        assert len(organizations) >= 3, organizations

    def test_the_covered_era_starts_and_ends_where_epoch_says(self, frontier):
        first, last = frontier[0], frontier[-1]
        assert date.fromisoformat(first["Release date"].strip()).year == 2023
        assert float(first["ECI Score"]) == pytest.approx(109.24, abs=0.01)
        # The newest frontier entry defines the snapshot's vintage.
        assert date.fromisoformat(last["Release date"].strip()) >= date(2026, 7, 1)
        assert float(last["ECI Score"]) > 160.0


class TestGeneratedTypeScriptMatchesCsv:
    @staticmethod
    def _ts_rows():
        text = TS_PATH.read_text(encoding="utf-8")
        dates = re.findall(r'^\s*releaseDate: "([^"]+)",$', text, flags=re.MULTILINE)
        scores = re.findall(r"^\s*eciScore: ([0-9.eE+-]+),$", text, flags=re.MULTILINE)
        versions = re.findall(r'^\s*modelVersion: "([^"]*)",$', text, flags=re.MULTILINE)
        return list(zip(dates, [float(s) for s in scores], versions))

    def test_same_frontier_in_the_same_order(self, frontier):
        expected = [
            (row["Release date"].strip(), float(row["ECI Score"]), row["Model version"].strip())
            for row in frontier
        ]
        assert self._ts_rows() == expected

    def test_every_point_carries_an_organization_and_a_display_name(self, frontier):
        text = TS_PATH.read_text(encoding="utf-8")
        organizations = re.findall(r'^\s*organization: "([^"]*)",$', text, flags=re.MULTILINE)
        display_names = re.findall(r'^\s*displayName: "([^"]*)",$', text, flags=re.MULTILINE)
        assert len(organizations) == len(frontier)
        assert len(display_names) == len(frontier)
        # Epoch leaves 'Display name' blank on older entries; the bridge must have
        # fallen back rather than emitting an empty tooltip header.
        assert all(name.strip() for name in display_names)
        assert all(org.strip() for org in organizations)

    def test_declared_vintage_matches_the_newest_frontier_release(self, frontier):
        text = TS_PATH.read_text(encoding="utf-8")
        declared = re.search(r'EPOCH_ECI_LATEST_RELEASE_DATE = "([^"]+)"', text)
        assert declared is not None
        assert declared.group(1) == frontier[-1]["Release date"].strip()

    def test_attribution_survives_regeneration(self):
        text = TS_PATH.read_text(encoding="utf-8")
        # CC-BY obliges us to credit Epoch wherever the data is shown; the chart
        # tooltip reads this constant.
        assert "Epoch AI, AI Benchmarking Hub (CC-BY 4.0)" in text
        assert "https://epoch.ai/benchmarks" in text


class TestFrontierIsDisplayOnly:
    def test_no_model_or_api_code_reads_the_frontier_module(self):
        # app.py may read epoch_capabilities_index.csv (it plots the same data on the
        # legacy Flask dashboard), but nothing in the model or the API may consume the
        # frontier the chart overlays.
        allowed = {
            Path("scripts/bridge/epoch_eci_to_ts.py"),
            Path("tests/test_epoch_eci_data.py"),
        }
        offenders = []
        for directory in ("progress_model", "api", "scripts/pipelines", "scripts/analysis"):
            for path in (PROJECT_ROOT / directory).rglob("*.py"):
                relative = path.relative_to(PROJECT_ROOT)
                if relative in allowed:
                    continue
                text = path.read_text(encoding="utf-8")
                if "epoch_capabilities_index" in text or "eciData" in text:
                    offenders.append(str(relative))
        assert offenders == [], (
            f"{offenders} reference the ECI datapoints; they are display-only and "
            "must not reach the model or the API"
        )

    def test_the_eci_trend_is_anchored_on_the_parameter_not_on_the_frontier(self):
        # A trend fit to the observed frontier would not pass exactly through
        # eci_at_present_day, nor grow at exactly eci_growth_rate.
        times = np.linspace(2018.0, 2040.0, 4000)
        progress = 0.45 * (times - 2018.0) + 0.015 * (times - 2018.0) ** 2
        rates = 0.45 + 0.03 * (times - 2018.0)

        eci_at_present_day = 162.0
        growth_rate = 12.0
        params = Parameters(
            ac_anchor_mode="eci",
            eci_at_present_day=eci_at_present_day,
            eci_growth_rate=growth_rate,
            # Undated: this test is about WHAT the trend is fitted to, not WHEN it is
            # anchored. The shipped default states an ECI vintage (2026.524658), and
            # leaving it in would anchor the fit there while `anchor_rate` below is read
            # at present_day, so "one year on" would be one year at the wrong rate.
            eci_reference_date=None,
        )
        anchor_rate = float(np.interp(params.present_day, times, rates))
        observable = fit_eci_trend(params, times, progress, anchor_rate, rates)

        assert observable is not None
        anchor_progress = observable.fit_info["anchor_progress"]
        assert float(observable.predict(anchor_progress)) == pytest.approx(
            eci_at_present_day, rel=1e-12
        )
        # One year on (in anchor-rate progress units) the linear trend must have added
        # exactly eci_growth_rate points.
        one_year = anchor_progress + anchor_rate
        assert float(observable.predict(one_year)) == pytest.approx(
            eci_at_present_day + growth_rate, rel=1e-12
        )


def test_the_default_present_day_eci_agrees_with_the_latest_frontier_point():
    """The model's present-day anchor and Epoch's newest frontier score describe the
    same quantity at nearly the same date, so a wide gap means one of them is stale."""
    frontier = _frontier_from_csv(_csv_rows())
    latest = float(frontier[-1]["ECI Score"])
    default_eci = float(Parameters().eci_at_present_day)
    assert abs(default_eci - latest) < 10.0, (
        f"Parameters().eci_at_present_day={default_eci:g} vs newest frontier score "
        f"{latest:g} ({frontier[-1]['Release date']}) -- one of them needs refreshing"
    )


def test_frontier_points_land_on_the_trend_the_chart_draws():
    """The overlay is only useful if it lands on the plotted curve.

    ECI is on a linear axis whose domain is driven by the modelled series, so a point
    tens of index points away would be clipped. Assert every frontier point sits close
    to the default ECI trend evaluated at the same date.
    """
    frontier = _frontier_from_csv(_csv_rows())
    times = np.linspace(2018.0, 2040.0, 4000)
    progress = 0.45 * (times - 2018.0) + 0.015 * (times - 2018.0) ** 2
    rates = 0.45 + 0.03 * (times - 2018.0)

    params = Parameters(
        ac_anchor_mode="eci",
        eci_at_present_day=162.0,
        present_day=2026.578082,
    )
    anchor_rate = float(np.interp(params.present_day, times, rates))
    observable = fit_eci_trend(params, times, progress, anchor_rate, rates)
    assert observable is not None

    for row in frontier:
        released = date.fromisoformat(row["Release date"].strip())
        decimal_year = released.year + (released.timetuple().tm_yday - 1) / 365.25
        trend_value = float(observable.predict(float(np.interp(decimal_year, times, progress))))
        observed = float(row["ECI Score"])
        assert abs(observed - trend_value) < 30.0, (
            f"{row['Model version']} on {released}: observed ECI {observed:g} vs "
            f"trend {trend_value:g}"
        )
