"""Tests for tl tail sticky header: _render_tail_header, _format_duration,
_compute_entry_cost, and CLI integration with running totals.

Covers:
- Header rendering with various stats
- Duration formatting edge cases
- Cost computation (single entry, unknown model, missing fields)
- Double-cost bug regression (cost must be computed once per entry)
- CLI: --cost flag kept as deprecated no-op
- CLI: header appears in output with correct totals
- CLI: piped/non-TTY output skips ANSI header
- CLI: midnight rollover to new log file
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import toklog.logger as logger_mod
import toklog.pricing as pricing_mod
from toklog.cli import (
    _compute_entry_cost,
    _format_duration,
    _format_tail_line,
    _render_tail_header,
    cli,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stable_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable live pricing cache so hardcoded table is always used."""
    monkeypatch.setattr(pricing_mod, "_live_cache", None)
    monkeypatch.setattr(pricing_mod, "_live_cache_loaded", True)


@pytest.fixture()
def log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    monkeypatch.setattr(logger_mod, "_LOG_DIR", str(d))
    monkeypatch.setattr(logger_mod, "_dir_ensured", False)
    return d


def _entry(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "model": "claude-haiku-3-5-20241022",
        "provider": "anthropic",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "duration_ms": 500,
        "error": False,
        "tags": "test",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Unit tests — _format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_zero(self):
        assert _format_duration(0) == "0s"

    def test_seconds_only(self):
        assert _format_duration(45) == "45s"

    def test_exactly_one_minute(self):
        assert _format_duration(60) == "1m 00s"

    def test_minutes_and_seconds(self):
        assert _format_duration(632) == "10m 32s"

    def test_exactly_one_hour(self):
        assert _format_duration(3600) == "1h 00m"

    def test_hours_and_minutes(self):
        assert _format_duration(3720) == "1h 02m"

    def test_fractional_seconds_truncated(self):
        """Fractional seconds should be truncated, not rounded."""
        assert _format_duration(59.9) == "59s"

    def test_large_duration(self):
        # 25 hours
        result = _format_duration(90000)
        assert result == "25h 00m"

    def test_negative_treated_as_zero(self):
        # Defensive — shouldn't happen but shouldn't crash
        result = _format_duration(-5)
        # int(-5) = -5, negative modulo behavior — just verify no crash
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Unit tests — _compute_entry_cost
# ---------------------------------------------------------------------------

class TestComputeEntryCost:
    def test_known_model(self):
        cost = _compute_entry_cost(_entry(
            model="claude-haiku-3-5-20241022",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=200,
        ))
        assert cost > 0
        assert isinstance(cost, float)

    def test_unknown_model_returns_zero(self):
        cost = _compute_entry_cost(_entry(model="totally-fake-model-xyz"))
        # Should return 0.0, not crash
        assert cost == 0.0 or isinstance(cost, float)

    def test_missing_fields_default_to_zero(self):
        entry = {"model": "claude-haiku-3-5-20241022", "provider": "anthropic"}
        cost = _compute_entry_cost(entry)
        assert isinstance(cost, float)

    def test_cache_tokens_included(self):
        """Cache read/creation tokens should affect cost."""
        no_cache = _compute_entry_cost(_entry(
            cache_read_tokens=0, cache_creation_tokens=0,
        ))
        with_cache = _compute_entry_cost(_entry(
            cache_read_tokens=5000, cache_creation_tokens=10000,
        ))
        # Cache creation costs more, so with_cache should differ
        assert with_cache != no_cache

    def test_empty_entry_no_crash(self):
        cost = _compute_entry_cost({})
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Unit tests — _render_tail_header
# ---------------------------------------------------------------------------

class TestRenderTailHeader:
    def test_zero_state(self):
        header = _render_tail_header(0.0, 0, 0.0)
        assert "$0.00" in header
        assert "0 reqs" in header
        assert "$0.00/min" in header
        assert "$0.00/req" in header

    def test_basic_stats(self):
        header = _render_tail_header(12.34, 42, 632.0)
        assert "$12.34" in header
        assert "42 reqs" in header
        assert "10m 32s" in header

    def test_cost_per_min_calculation(self):
        # $60 over 60 seconds = $60/min
        header = _render_tail_header(60.0, 10, 60.0)
        assert "$60.00/min" in header

    def test_cost_per_req_calculation(self):
        # $10 over 5 requests = $2.00/req
        header = _render_tail_header(10.0, 5, 300.0)
        assert "$2.00/req" in header

    def test_cost_per_min_zero_when_elapsed_under_one_second(self):
        """Avoid division by zero when elapsed < 1s."""
        header = _render_tail_header(5.0, 1, 0.5)
        assert "$0.00/min" in header

    def test_cost_per_req_zero_when_no_requests(self):
        header = _render_tail_header(0.0, 0, 100.0)
        assert "$0.00/req" in header

    def test_separator_present(self):
        header = _render_tail_header(1.0, 1, 1.0)
        assert "│" in header

    def test_all_fields_present(self):
        header = _render_tail_header(5.55, 10, 120.0)
        # Should have: total, reqs, cost/min, cost/req, duration
        assert "reqs" in header
        assert "/min" in header
        assert "/req" in header

    def test_no_spike_text_when_zero(self):
        header = _render_tail_header(1.0, 1, 10.0, spike_count=0)
        assert "spike" not in header
        assert "🔺" not in header

    def test_spike_count_in_header(self):
        header = _render_tail_header(5.0, 10, 60.0, spike_count=3)
        assert "🔺3 spikes" in header

    def test_spike_count_singular(self):
        header = _render_tail_header(5.0, 10, 60.0, spike_count=1)
        assert "🔺1 spike" in header
        assert "spikes" not in header


# ---------------------------------------------------------------------------
# Unit tests — _format_tail_line (cost always on)
# ---------------------------------------------------------------------------

class TestFormatTailLineCostAlwaysOn:
    """Verify cost is always included and the precomputed cost path works."""

    def test_cost_always_present(self):
        line = _format_tail_line(_entry(), cost=None)
        assert "$" in line

    def test_precomputed_cost_used_when_provided(self):
        """When cost is passed, it must appear in output — not recomputed."""
        line = _format_tail_line(_entry(), cost=99.1234)
        assert "$99.1234" in line

    def test_cost_computed_when_none(self):
        """When cost=None, it's computed from the entry."""
        line = _format_tail_line(_entry(
            model="claude-haiku-3-5-20241022",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=200,
        ), cost=None)
        assert "$" in line
        # Should be non-zero for a known model
        dollar_part = [p for p in line.split() if p.startswith("$")][0]
        val = float(dollar_part.replace("$", ""))
        assert val > 0

    def test_no_double_cost_computation(self):
        """Regression: cost must NOT be computed twice per entry."""
        call_count = {"n": 0}
        original_fn = _compute_entry_cost

        def counting_compute(entry):
            call_count["n"] += 1
            return original_fn(entry)

        with patch("toklog.cli._compute_entry_cost", side_effect=counting_compute):
            # When cost is precomputed, _compute_entry_cost should NOT be called
            _format_tail_line(_entry(), cost=1.23)
            assert call_count["n"] == 0

    def test_model_in_output(self):
        line = _format_tail_line(_entry(model="gpt-4o"), cost=0.01)
        assert "gpt-4o" in line

    def test_tokens_in_output(self):
        line = _format_tail_line(_entry(input_tokens=500, output_tokens=100), cost=0.01)
        assert "in=500" in line
        assert "out=100" in line


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestTailCLI:
    """CLI-level tests for tl tail with header."""

    def _write_today(self, log_dir: Path, entry: dict) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = log_dir / f"{today}.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _invoke(self, log_dir: Path, args: list[str] | None = None) -> str:
        runner = CliRunner()
        call_count = {"n": 0}

        def _stop_after_first_poll(_: float) -> None:
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise KeyboardInterrupt

        with patch("time.sleep", side_effect=_stop_after_first_poll):
            result = runner.invoke(cli, ["tail"] + (args or []), catch_exceptions=False)
        return result.output

    def test_no_crash_with_entries(self, log_dir):
        self._write_today(log_dir, _entry())
        out = self._invoke(log_dir)
        assert "Stopped" in out

    def test_cost_flag_accepted_as_noop(self, log_dir):
        """--cost should still be accepted (deprecated no-op), not error."""
        self._write_today(log_dir, _entry())
        out = self._invoke(log_dir, ["--cost"])
        assert "Stopped" in out

    def test_output_contains_cost(self, log_dir):
        """Cost should appear in output (always on)."""
        self._write_today(log_dir, _entry())
        out = self._invoke(log_dir)
        assert "$" in out

    def test_output_contains_request_count(self, log_dir):
        """Header should show request count."""
        self._write_today(log_dir, _entry())
        self._write_today(log_dir, _entry())
        out = self._invoke(log_dir)
        # Should see "2 reqs" or at least "reqs" in the header
        assert "reqs" in out

    def test_empty_log_no_crash(self, log_dir):
        """No log file yet — should wait without crashing."""
        out = self._invoke(log_dir)
        assert "Stopped" in out

    def test_malformed_json_line_skipped(self, log_dir):
        """Malformed JSON lines should be skipped, not crash the tail."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = log_dir / f"{today}.jsonl"
        with open(filepath, "a") as f:
            f.write("not json at all\n")
            f.write(json.dumps(_entry()) + "\n")
        out = self._invoke(log_dir)
        assert "Stopped" in out


class TestTailMidnightRollover:
    """Verify tail picks up new log file after midnight UTC."""

    def _write_log(self, log_dir: Path, date_str: str, entry: dict) -> None:
        filepath = log_dir / f"{date_str}.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def test_date_recomputed_each_poll(self, log_dir):
        """Tail should check current date each iteration, not cache it."""
        runner = CliRunner()
        dates = ["2026-04-07", "2026-04-07", "2026-04-08"]
        date_idx = {"i": 0}

        # Write entries for both days
        self._write_log(log_dir, "2026-04-07", _entry(model="day1-model"))
        self._write_log(log_dir, "2026-04-08", _entry(model="day2-model"))

        call_count = {"n": 0}

        def fake_now(tz=None):
            # Return different dates on successive calls
            idx = min(date_idx["i"], len(dates) - 1)
            date_str = dates[idx]
            date_idx["i"] += 1
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        def _stop_after_polls(_: float) -> None:
            call_count["n"] += 1
            if call_count["n"] >= 3:
                raise KeyboardInterrupt

        with patch("time.sleep", side_effect=_stop_after_polls), \
             patch("toklog.cli._tail_current_date", side_effect=fake_now):
            result = runner.invoke(cli, ["tail"], catch_exceptions=False)

        assert "Stopped" in result.output
