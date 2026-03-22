"""Tests for toklog/gain.py — cumulative gain tracking."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from toklog.gain import load_gain, save_gain, update_gain


@pytest.fixture()
def tmp_gain_file(tmp_path):
    """Redirect _GAIN_FILE to a temp path for isolation."""
    gain_file = tmp_path / "gain.json"
    with patch("toklog.gain._GAIN_FILE", gain_file):
        yield gain_file


def _make_report(spend: float = 10.0, waste: float = 2.0, calls: int = 100) -> dict:
    return {
        "total_spend_usd": spend,
        "estimated_waste_usd": waste,
        "total_calls": calls,
        "detectors": [
            {"name": "cache_miss_opportunity", "triggered": True, "estimated_waste_usd": waste},
        ],
    }


def test_load_gain_returns_empty_dict_when_missing(tmp_gain_file):
    assert load_gain() == {}


def test_update_gain_creates_snapshot(tmp_gain_file):
    report = _make_report(spend=10.0, waste=2.0, calls=100)
    result = update_gain(report)

    assert result["total_spend_observed_usd"] == 10.0
    assert result["total_waste_detected_usd"] == 2.0
    assert result["total_calls_observed"] == 100
    assert len(result["snapshots"]) == 1
    assert "first_seen" in result


def test_update_gain_idempotent_same_day(tmp_gain_file):
    """Calling update_gain twice on the same day replaces the snapshot, not duplicates it."""
    update_gain(_make_report(spend=10.0, waste=2.0, calls=100))
    update_gain(_make_report(spend=12.0, waste=3.0, calls=120))

    result = load_gain()
    # Only one snapshot for today
    assert len(result["snapshots"]) == 1
    # Totals reflect the latest call
    assert result["total_spend_observed_usd"] == 12.0
    assert result["total_waste_detected_usd"] == 3.0
    assert result["total_calls_observed"] == 120


def test_update_gain_accumulates_across_days(tmp_gain_file):
    """Snapshots from different days accumulate in totals."""
    # Write a snapshot for day 1 directly (simulating a prior report run)
    gain_day1 = {
        "first_seen": "2025-03-01T00:00:00+00:00",
        "last_updated": "2025-03-01T00:00:00+00:00",
        "snapshots": [{"date": "2025-03-01", "spend": 10.0, "waste": 2.0, "calls": 100}],
        "total_calls_observed": 100,
        "total_spend_observed_usd": 10.0,
        "total_waste_detected_usd": 2.0,
        "by_detector": {},
    }
    save_gain(gain_day1)

    # Now call update_gain for today (a different date from "2025-03-01")
    update_gain(_make_report(spend=5.0, waste=1.0, calls=50))

    result = load_gain()
    # Should have 2 snapshots: the manually written one + today's
    assert len(result["snapshots"]) == 2
    assert result["total_spend_observed_usd"] == pytest.approx(15.0, abs=0.001)
    assert result["total_waste_detected_usd"] == pytest.approx(3.0, abs=0.001)


def test_update_gain_persists_to_file(tmp_gain_file):
    report = _make_report()
    update_gain(report)

    assert tmp_gain_file.exists()
    raw = json.loads(tmp_gain_file.read_text())
    assert "snapshots" in raw
    assert raw["total_calls_observed"] == 100


def test_update_gain_by_detector(tmp_gain_file):
    """by_detector reflects the latest report's triggered detectors."""
    report = {
        "total_spend_usd": 10.0,
        "estimated_waste_usd": 5.0,
        "total_calls": 100,
        "detectors": [
            {"name": "cache_miss_opportunity", "triggered": True, "estimated_waste_usd": 3.0},
            {"name": "tool_schema_bloat", "triggered": True, "estimated_waste_usd": 2.0},
            {"name": "high_spend_process", "triggered": False, "estimated_waste_usd": 0.0},
        ],
    }
    result = update_gain(report)
    assert result["by_detector"]["cache_miss_opportunity"] == 3.0
    assert result["by_detector"]["tool_schema_bloat"] == 2.0
    assert "high_spend_process" not in result["by_detector"]
