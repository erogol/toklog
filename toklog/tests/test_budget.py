"""Tests for toklog.proxy.budget — daily budget enforcement."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from toklog.proxy.budget import (
    BudgetTracker,
    configure,
    check,
    record,
    status,
    load_status_from_file,
)


@pytest.fixture
def tmp_state(tmp_path):
    """Return a temporary state file path."""
    return tmp_path / "budget_state.json"


# ------------------------------------------------------------------
# BudgetTracker — no limit
# ------------------------------------------------------------------

class TestValidation:
    def test_negative_limit_raises(self, tmp_state):
        with pytest.raises(ValueError, match="positive"):
            BudgetTracker(limit_usd=-5.0, state_path=tmp_state)

    def test_zero_limit_raises(self, tmp_state):
        with pytest.raises(ValueError, match="positive"):
            BudgetTracker(limit_usd=0.0, state_path=tmp_state)

    def test_string_limit_raises(self, tmp_state):
        with pytest.raises(ValueError, match="positive number or None"):
            BudgetTracker(limit_usd="ten", state_path=tmp_state)

    def test_none_limit_is_valid(self, tmp_state):
        t = BudgetTracker(limit_usd=None, state_path=tmp_state)
        assert t.status()["limit_usd"] is None


# ------------------------------------------------------------------
# BudgetTracker — no limit
# ------------------------------------------------------------------

class TestNoLimit:
    def test_check_always_allows(self, tmp_state):
        t = BudgetTracker(limit_usd=None, state_path=tmp_state)
        allowed, info = t.check()
        assert allowed is True
        assert info == {}

    def test_record_still_tracks_spend(self, tmp_state):
        t = BudgetTracker(limit_usd=None, state_path=tmp_state)
        t.record(1.50)
        t.record(2.25)
        assert t.status()["daily_spend"] == 3.75

    def test_status_shows_not_enforcing(self, tmp_state):
        t = BudgetTracker(limit_usd=None, state_path=tmp_state)
        s = t.status()
        assert s["enforcing"] is False
        assert s["limit_usd"] is None
        assert s["remaining"] is None


# ------------------------------------------------------------------
# BudgetTracker — with limit
# ------------------------------------------------------------------

class TestWithLimit:
    def test_under_budget_allows(self, tmp_state):
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        t.record(5.0)
        allowed, info = t.check()
        assert allowed is True
        assert info == {}

    def test_at_budget_denies(self, tmp_state):
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        t.record(10.0)
        allowed, info = t.check()
        assert allowed is False
        assert info["type"] == "budget_exceeded"
        assert info["daily_spend"] == 10.0
        assert info["budget_limit"] == 10.0
        assert info["rejected_count"] == 1

    def test_over_budget_denies(self, tmp_state):
        t = BudgetTracker(limit_usd=5.0, state_path=tmp_state)
        t.record(7.50)
        allowed, info = t.check()
        assert allowed is False
        assert info["daily_spend"] == 7.50

    def test_rejected_count_increments(self, tmp_state):
        t = BudgetTracker(limit_usd=1.0, state_path=tmp_state)
        t.record(2.0)
        t.check()
        t.check()
        _, info = t.check()
        assert info["rejected_count"] == 3

    def test_message_contains_amounts(self, tmp_state):
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        t.record(12.34)
        _, info = t.check()
        assert "$10.00" in info["message"]
        assert "$12.34" in info["message"]

    def test_status_shows_enforcing(self, tmp_state):
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        t.record(3.0)
        s = t.status()
        assert s["enforcing"] is True
        assert s["limit_usd"] == 10.0
        assert s["daily_spend"] == 3.0
        assert s["remaining"] == 7.0

    def test_remaining_floors_at_zero(self, tmp_state):
        t = BudgetTracker(limit_usd=5.0, state_path=tmp_state)
        t.record(8.0)
        assert t.status()["remaining"] == 0.0


# ------------------------------------------------------------------
# Record accumulation
# ------------------------------------------------------------------

class TestRecord:
    def test_accumulates(self, tmp_state):
        t = BudgetTracker(limit_usd=100.0, state_path=tmp_state)
        for _ in range(10):
            t.record(0.5)
        assert t.status()["daily_spend"] == 5.0

    def test_small_amounts_precision(self, tmp_state):
        t = BudgetTracker(limit_usd=100.0, state_path=tmp_state)
        for _ in range(100):
            t.record(0.001)
        # Floating point: should be close to 0.1
        assert abs(t.status()["daily_spend"] - 0.1) < 0.001


# ------------------------------------------------------------------
# Date rollover
# ------------------------------------------------------------------

class TestDateRollover:
    def test_resets_on_new_day(self, tmp_state):
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        t.record(12.0)
        allowed, _ = t.check()  # rejected — over budget
        assert allowed is False
        assert t.status()["daily_spend"] == 12.0
        assert t.status()["rejected_count"] == 1

        # Simulate date change
        with patch("toklog.proxy.budget._today", return_value="2099-01-01"):
            allowed, info = t.check()
            assert allowed is True
            assert t.status()["daily_spend"] == 0.0
            assert t.status()["rejected_count"] == 0
            assert t.status()["date"] == "2099-01-01"

    def test_record_after_rollover_starts_fresh(self, tmp_state):
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        t.record(5.0)

        with patch("toklog.proxy.budget._today", return_value="2099-01-01"):
            t.record(1.0)
            assert t.status()["daily_spend"] == 1.0


# ------------------------------------------------------------------
# State file
# ------------------------------------------------------------------

class TestStateFile:
    def test_record_writes_state_file(self, tmp_state):
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        assert not tmp_state.exists()
        t.record(3.0)
        assert tmp_state.exists()
        data = json.loads(tmp_state.read_text())
        assert data["daily_spend"] == 3.0
        assert data["limit_usd"] == 10.0
        assert "updated_at" in data

    def test_rejection_writes_state_file(self, tmp_state):
        t = BudgetTracker(limit_usd=1.0, state_path=tmp_state)
        t.record(2.0)
        t.check()
        data = json.loads(tmp_state.read_text())
        assert data["rejected_count"] == 1

    def test_load_status_from_file(self, tmp_state):
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        t.record(4.50)
        loaded = load_status_from_file(tmp_state)
        assert loaded is not None
        assert loaded["daily_spend"] == 4.50
        assert loaded["limit_usd"] == 10.0

    def test_load_status_missing_file(self, tmp_path):
        result = load_status_from_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_status_corrupt_file(self, tmp_state):
        tmp_state.write_text("not json{{{")
        result = load_status_from_file(tmp_state)
        assert result is None

    def test_atomic_write_no_partial(self, tmp_state):
        """State file should not contain partial data even after multiple writes."""
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        for i in range(20):
            t.record(0.1)
        # Should always be valid JSON
        data = json.loads(tmp_state.read_text())
        assert data["daily_spend"] > 0

    def test_state_file_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "budget_state.json"
        t = BudgetTracker(limit_usd=5.0, state_path=deep_path)
        t.record(1.0)
        assert deep_path.exists()

    def test_load_status_stale_date_resets(self, tmp_state):
        """State file from yesterday should show zeroed counters."""
        t = BudgetTracker(limit_usd=10.0, state_path=tmp_state)
        t.record(7.0)
        t.check()  # still under budget, no rejection

        # Manually set the date to yesterday in the file
        data = json.loads(tmp_state.read_text())
        data["date"] = "2020-01-01"
        tmp_state.write_text(json.dumps(data))

        loaded = load_status_from_file(tmp_state)
        assert loaded["daily_spend"] == 0.0
        assert loaded["rejected_count"] == 0
        assert loaded["remaining"] == 10.0


# ------------------------------------------------------------------
# Module-level API (configure / check / record / status)
# ------------------------------------------------------------------

class TestModuleAPI:
    def test_configure_replaces_tracker(self, tmp_state):
        configure(limit_usd=5.0, state_path=tmp_state)
        s = status()
        assert s["limit_usd"] == 5.0
        assert s["enforcing"] is True

        configure(limit_usd=None, state_path=tmp_state)
        s = status()
        assert s["limit_usd"] is None
        assert s["enforcing"] is False

    def test_configure_writes_initial_state(self, tmp_state):
        configure(limit_usd=20.0, state_path=tmp_state)
        assert tmp_state.exists()
        data = json.loads(tmp_state.read_text())
        assert data["limit_usd"] == 20.0
        assert data["daily_spend"] == 0.0

    def test_module_check_and_record(self, tmp_state):
        configure(limit_usd=2.0, state_path=tmp_state)
        allowed, _ = check()
        assert allowed is True
        record(1.5)
        allowed, _ = check()
        assert allowed is True
        record(1.0)
        allowed, info = check()
        assert allowed is False
        assert info["type"] == "budget_exceeded"

    def test_module_status(self, tmp_state):
        configure(limit_usd=10.0, state_path=tmp_state)
        record(3.0)
        s = status()
        assert s["daily_spend"] == 3.0
        assert s["remaining"] == 7.0
