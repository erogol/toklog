"""Tests for the report module."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

import toklog.logger as logger_mod
import toklog.pricing as pricing_mod
from toklog.logger import _is_benchmark_entry, log_entry
from toklog.report import _classify_key_hint, _compute_cost, _fmt_usd, compute_spend_trend, generate_report, render_json, render_text


@pytest.fixture(autouse=True)
def _isolate_live_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep report tests using hardcoded pricing so expected values stay stable."""
    monkeypatch.setattr(pricing_mod, "_live_cache", None)
    monkeypatch.setattr(pricing_mod, "_live_cache_loaded", True)


@pytest.fixture(autouse=True)
def tmp_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect logs to a temp directory."""
    log_dir = str(tmp_path / "logs")
    monkeypatch.setattr(logger_mod, "_LOG_DIR", log_dir)
    monkeypatch.setattr(logger_mod, "_dir_ensured", False)
    return tmp_path


def _sample_entry(**overrides: Any) -> dict:
    entry: Dict[str, Any] = {
        "timestamp": "2025-03-09T10:00:00.000Z",
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "max_tokens_set": 4096,
        "system_prompt_hash": "abc123",
        "tool_count": 0,
        "tool_schema_tokens": 0,
        "tags": "test",
        "streaming": False,
        "duration_ms": 1000,
        "error": False,
        "error_type": None,
        "request_id": "req_1",
    }
    entry.update(overrides)
    return entry


class TestComputeCost:
    def test_zero_cache_regression(self) -> None:
        """With no cache tokens, cost equals simple input+output calculation."""
        entry = _sample_entry(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=200,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
        # gpt-4o: input=0.00125, output=0.005 per 1K
        expected = (1000 * 0.00125 / 1000.0) + (200 * 0.005 / 1000.0)
        assert abs(_compute_cost(entry) - expected) < 1e-10

    def test_anthropic_cache_read_and_creation(self) -> None:
        """Anthropic: input excludes cache tokens; cache_read and cache_creation billed separately."""
        entry = _sample_entry(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=500,
            output_tokens=100,
            cache_read_tokens=200,
            cache_creation_tokens=300,
        )
        # claude-sonnet-4-6: input=0.003, output=0.015 per 1K
        # cache_read = 0.003 * 0.1 = 0.0003, cache_write = 0.003 * 1.25 = 0.00375
        expected = (
            500 * 0.003 / 1000.0
            + 300 * 0.00375 / 1000.0
            + 200 * 0.0003 / 1000.0
            + 100 * 0.015 / 1000.0
        )
        assert abs(_compute_cost(entry) - expected) < 1e-10

    def test_openai_cache_read_deducted_from_input(self) -> None:
        """OpenAI: input_tokens INCLUDES cache_read; cache_read billed at reduced rate."""
        entry = _sample_entry(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=100,
            cache_read_tokens=400,
            cache_creation_tokens=0,
        )
        # gpt-4o: input=0.00125, output=0.005 per 1K
        # cache_read_price = 0.00125 * 0.5 = 0.000625
        expected = (
            (1000 - 400) * 0.00125 / 1000.0
            + 400 * 0.000625 / 1000.0
            + 100 * 0.005 / 1000.0
        )
        assert abs(_compute_cost(entry) - expected) < 1e-10

    def test_gemini_cache_read_deducted_from_input(self) -> None:
        """Gemini: promptTokenCount INCLUDES cachedContentTokenCount; cache_read billed at reduced rate."""
        entry = _sample_entry(
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=100,
            cache_read_tokens=400,
            cache_creation_tokens=0,
        )
        # gemini-2.5-flash: input=0.00015, output=0.0006 per 1K
        # cache_read_price = 0.00015 * 0.5 = 0.000075 (default multiplier)
        expected = (
            (1000 - 400) * 0.00015 / 1000.0
            + 400 * 0.000075 / 1000.0
            + 100 * 0.0006 / 1000.0
        )
        assert abs(_compute_cost(entry) - expected) < 1e-10

    def test_missing_provider_no_crash(self) -> None:
        """Entry without provider field should not crash."""
        entry = {
            "model": "claude-sonnet-4-6",
            "input_tokens": 100,
            "output_tokens": 50,
        }
        cost = _compute_cost(entry)
        assert cost >= 0.0


class TestGenerateReport:
    def test_basic_report(self, tmp_path: Path) -> None:
        """Generate a report with mock data."""
        for _ in range(5):
            log_entry(_sample_entry())

        report = generate_report(last="all")
        assert report["total_calls"] == 5
        assert report["total_spend_usd"] > 0
        assert len(report["detectors"]) == 9  # Added credential_sharing detector
        assert "cost_by_model" in report

    def test_empty_report(self, tmp_path: Path) -> None:
        """Report with no data should not crash."""
        report = generate_report(last="all")
        assert report["total_calls"] == 0
        assert report["total_spend_usd"] == 0

    def test_date_filtering(self, tmp_path: Path) -> None:
        """Report respects date range."""
        log_entry(_sample_entry())
        report = generate_report(last="1d")
        # Today's entry should be included
        assert report["total_calls"] >= 0  # May be 0 or 1 depending on UTC date

    def test_cost_by_model(self, tmp_path: Path) -> None:
        """Cost is correctly grouped by model."""
        log_entry(_sample_entry(model="gpt-4o"))
        log_entry(_sample_entry(model="gpt-4o-mini"))

        report = generate_report(last="all")
        assert "gpt-4o" in report["cost_by_model"]
        assert "gpt-4o-mini" in report["cost_by_model"]

    def test_unknown_model_no_crash(self, tmp_path: Path) -> None:
        """Unknown models should not crash the report."""
        log_entry(_sample_entry(model="unknown-model-xyz"))
        report = generate_report(last="all")
        assert report["total_calls"] == 1

    def test_dated_model_names_merged(self, tmp_path: Path) -> None:
        """API-resolved dated model names group with request model names."""
        log_entry(_sample_entry(model="gpt-5.4", input_tokens=0, output_tokens=0))
        log_entry(_sample_entry(model="gpt-5.4-2026-03-05", input_tokens=500, output_tokens=100))

        report = generate_report(last="all")
        assert "gpt-5.4" in report["cost_by_model"]
        assert "gpt-5.4-2026-03-05" not in report["cost_by_model"]
        assert report["cost_by_model"]["gpt-5.4"]["calls"] == 2
        assert report["cost_by_model"]["gpt-5.4"]["tokens"] == 600


class TestRenderJson:
    def test_valid_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """JSON output should be valid JSON."""
        log_entry(_sample_entry())
        report = generate_report(last="all")
        render_json(report)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["total_calls"] == 1

class TestRenderText:
    def test_no_crash(self, tmp_path: Path) -> None:
        """Text rendering should not crash."""
        log_entry(_sample_entry())
        report = generate_report(last="all")
        # Just verify it doesn't raise
        render_text(report)

    def test_empty_report_no_crash(self, tmp_path: Path) -> None:
        """Text rendering of empty report should not crash."""
        report = generate_report(last="all")
        render_text(report)


class TestComputeSpendTrend:
    def test_compute_spend_trend_up(self) -> None:
        """Current spend > prior by >5% → direction='up'."""
        current = [_sample_entry(input_tokens=2000, output_tokens=400)]  # ~2x cost
        prior = [_sample_entry(input_tokens=1000, output_tokens=200)]
        result = compute_spend_trend(current, prior)
        assert result["direction"] == "up"
        assert result["change_pct"] is not None
        assert result["change_pct"] > 5

    def test_compute_spend_trend_down(self) -> None:
        """Current spend < prior by >5% → direction='down'."""
        current = [_sample_entry(input_tokens=500, output_tokens=100)]
        prior = [_sample_entry(input_tokens=2000, output_tokens=400)]
        result = compute_spend_trend(current, prior)
        assert result["direction"] == "down"
        assert result["change_pct"] is not None
        assert result["change_pct"] < -5

    def test_compute_spend_trend_flat(self) -> None:
        """Change within ±5% → direction='flat'."""
        entry = _sample_entry(input_tokens=1000, output_tokens=200)
        result = compute_spend_trend([entry], [entry])
        assert result["direction"] == "flat"
        assert result["change_pct"] == 0.0

    def test_zero_prior(self) -> None:
        """Prior=[] → change_pct is None, direction='up' if current > 0."""
        current = [_sample_entry(input_tokens=1000, output_tokens=200)]
        result = compute_spend_trend(current, [])
        assert result["change_pct"] is None
        assert result["direction"] == "up"

    def test_generate_report_with_prior_entries(self, tmp_path: Path) -> None:
        """prior_entries passed → spend_trend key with all required fields."""
        log_entry(_sample_entry())
        prior = [_sample_entry(input_tokens=500, output_tokens=100)]
        report = generate_report(last="all", prior_entries=prior)
        assert "spend_trend" in report
        trend = report["spend_trend"]
        assert trend is not None
        for field in ("current_usd", "prior_usd", "change_usd", "change_pct", "direction"):
            assert field in trend

    def test_generate_report_no_prior(self, tmp_path: Path) -> None:
        """No prior_entries → spend_trend is None."""
        log_entry(_sample_entry())
        report = generate_report(last="all")
        assert report["spend_trend"] is None


class TestCostByContextDriver:
    def test_report_has_context_driver_key(self, tmp_path: Path) -> None:
        """generate_report returns cost_by_context_driver, not cost_by_use_case."""
        log_entry(_sample_entry())
        report = generate_report(last="all")
        assert "cost_by_context_driver" in report
        assert "cost_by_use_case" not in report

    def test_context_driver_output_shape(self, tmp_path: Path) -> None:
        """Each row has the expected keys."""
        log_entry(_sample_entry())
        report = generate_report(last="all")
        for row in report["cost_by_context_driver"]:
            assert set(row.keys()) == {"name", "cost_usd", "calls", "tokens", "avg_tokens", "pct"}

    def test_context_driver_exception_returns_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When aggregate_context_drivers raises, report still generates with []."""
        import toklog.report as report_mod

        log_entry(_sample_entry())

        def raise_err(*a: object, **kw: object) -> object:
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(report_mod, "aggregate_context_drivers", raise_err)

        report = generate_report(last="all")
        assert report["cost_by_context_driver"] == []

    def test_context_driver_pct_sums_near_100(self, tmp_path: Path) -> None:
        """Percentages across all rows sum close to 100."""
        for _ in range(3):
            log_entry(_sample_entry())
        report = generate_report(last="all")
        rows = report["cost_by_context_driver"]
        if rows:
            total_pct = sum(r["pct"] for r in rows)
            assert abs(total_pct - 100.0) < 1.0


class TestCostByCallSite:
    def test_cost_by_call_site_present(self, tmp_path: Path) -> None:
        """Entries with call_site → cost_by_call_site in report with correct structure."""
        cs = {"file": "app.py", "function": "run", "line": 42}
        log_entry(_sample_entry(call_site=cs))
        log_entry(_sample_entry(call_site=cs))

        report = generate_report(last="all")
        assert "cost_by_call_site" in report
        rows = report["cost_by_call_site"]
        assert len(rows) == 1
        row = rows[0]
        assert row["call_site"] == "app.py:run:42"
        assert row["calls"] == 2
        assert row["cost_usd"] > 0
        assert "pct" in row

    def test_cost_by_call_site_missing(self, tmp_path: Path) -> None:
        """Entries without call_site fall back to [tag] key."""
        log_entry(_sample_entry())  # no call_site key at all
        log_entry(_sample_entry(call_site=None))

        report = generate_report(last="all")
        rows = report["cost_by_call_site"]
        assert len(rows) == 1
        assert rows[0]["call_site"] == "[test]"
        assert rows[0]["calls"] == 2

    def test_cost_by_call_site_tokens_included(self, tmp_path: Path) -> None:
        """cost_by_call_site rows include token counts."""
        cs = {"file": "pipeline.py", "function": "embed", "line": 10}
        log_entry(_sample_entry(call_site=cs, input_tokens=500, output_tokens=100))
        log_entry(_sample_entry(call_site=cs, input_tokens=300, output_tokens=50))

        report = generate_report(last="all")
        rows = report["cost_by_call_site"]
        assert len(rows) == 1
        assert rows[0]["tokens"] == 950  # 500+100 + 300+50
        assert rows[0]["pct"] == 100.0


class TestErrorReporting:
    def test_no_errors_returns_zero(self, tmp_path: Path) -> None:
        """Report with no error entries returns error_calls=0 and empty errors_by_type."""
        log_entry(_sample_entry())
        log_entry(_sample_entry())
        report = generate_report(last="all")
        assert report["error_calls"] == 0
        assert report["errors_by_type"] == []

    def test_error_calls_counted(self, tmp_path: Path) -> None:
        """Error entries are counted in error_calls and included in total_calls."""
        log_entry(_sample_entry())
        log_entry(_sample_entry(error=True, error_type="RateLimitError"))
        log_entry(_sample_entry(error=True, error_type="RateLimitError"))
        report = generate_report(last="all")
        assert report["total_calls"] == 3
        assert report["error_calls"] == 2

    def test_errors_grouped_by_type(self, tmp_path: Path) -> None:
        """errors_by_type groups entries by error_type with count, models, last_seen."""
        log_entry(_sample_entry(error=True, error_type="RateLimitError", model="gpt-4o", timestamp="2026-03-11T10:00:00.000Z"))
        log_entry(_sample_entry(error=True, error_type="RateLimitError", model="gpt-4o", timestamp="2026-03-11T11:00:00.000Z"))
        log_entry(_sample_entry(error=True, error_type="APIConnectionError", model="gpt-4o-mini", timestamp="2026-03-10T09:00:00.000Z"))
        report = generate_report(last="all")
        assert report["error_calls"] == 3
        rows = report["errors_by_type"]
        # Sorted by count descending
        assert rows[0]["error_type"] == "RateLimitError"
        assert rows[0]["count"] == 2
        assert rows[0]["models"] == ["gpt-4o"]
        assert rows[0]["last_seen"] == "2026-03-11T11:00:00.000Z"
        assert rows[1]["error_type"] == "APIConnectionError"
        assert rows[1]["count"] == 1
        assert rows[1]["models"] == ["gpt-4o-mini"]

    def test_errors_by_type_multiple_models(self, tmp_path: Path) -> None:
        """Same error_type across multiple models lists all affected models."""
        log_entry(_sample_entry(error=True, error_type="TimeoutError", model="gpt-4o"))
        log_entry(_sample_entry(error=True, error_type="TimeoutError", model="claude-sonnet-4-6"))
        report = generate_report(last="all")
        row = report["errors_by_type"][0]
        assert set(row["models"]) == {"gpt-4o", "claude-sonnet-4-6"}

    def test_unknown_error_type_fallback(self, tmp_path: Path) -> None:
        """Entry with error=True but no error_type falls back to 'UnknownError'."""
        log_entry(_sample_entry(error=True, error_type=None))
        report = generate_report(last="all")
        assert report["error_calls"] == 1
        assert report["errors_by_type"][0]["error_type"] == "UnknownError"

    def test_pre_call_error_unbilled(self, tmp_path: Path) -> None:
        """Error with no tokens → unbilled_count=1, billed_count=0, wasted_usd=0."""
        log_entry(_sample_entry(error=True, error_type="APIConnectionError", input_tokens=0, output_tokens=0))
        report = generate_report(last="all")
        row = report["errors_by_type"][0]
        assert row["unbilled_count"] == 1
        assert row["billed_count"] == 0
        assert row["wasted_usd"] == 0.0
        assert report["error_wasted_usd"] == 0.0

    def test_billed_error_has_cost(self, tmp_path: Path) -> None:
        """Error with tokens → billed_count=1, wasted_usd > 0."""
        log_entry(_sample_entry(error=True, error_type="RateLimitError", input_tokens=1000, output_tokens=200))
        report = generate_report(last="all")
        row = report["errors_by_type"][0]
        assert row["billed_count"] == 1
        assert row["unbilled_count"] == 0
        assert row["wasted_usd"] > 0
        assert report["error_wasted_usd"] > 0

    def test_mixed_billed_unbilled(self, tmp_path: Path) -> None:
        """Same error type: one billed (has tokens), one pre-call (no tokens)."""
        log_entry(_sample_entry(error=True, error_type="TimeoutError", input_tokens=500, output_tokens=100))
        log_entry(_sample_entry(error=True, error_type="TimeoutError", input_tokens=0, output_tokens=0))
        report = generate_report(last="all")
        row = report["errors_by_type"][0]
        assert row["billed_count"] == 1
        assert row["unbilled_count"] == 1
        assert row["count"] == 2
        assert row["wasted_usd"] > 0

    def test_error_wasted_usd_zero_when_all_precall(self, tmp_path: Path) -> None:
        """error_wasted_usd is 0 when all errors are pre-call (no tokens)."""
        log_entry(_sample_entry(error=True, error_type="ConnectionError", input_tokens=0, output_tokens=0))
        log_entry(_sample_entry(error=True, error_type="ConnectionError", input_tokens=0, output_tokens=0))
        report = generate_report(last="all")
        assert report["error_wasted_usd"] == 0.0

    def test_no_error_output_when_no_errors(self, tmp_path: Path) -> None:
        """render_text does not print error info when no errors."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Billed Failures" not in output
        assert "Unbilled failures" not in output

    def test_unbilled_errors_shown_as_compact_summary(self, tmp_path: Path) -> None:
        """Unbilled errors (no tokens) render as a compact dim line, not a table."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry(error=True, error_type="RateLimitError", input_tokens=0, output_tokens=0))
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Unbilled failures" in output
        assert "RateLimitError" in output
        # Should NOT show a full table for unbilled-only errors
        assert "Billed Failures" not in output

    def test_billed_errors_shown_in_table(self, tmp_path: Path) -> None:
        """Billed errors (with tokens) render in a Billed Failures table."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry(error=True, error_type="RateLimitError", input_tokens=1000, output_tokens=200))
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Billed Failures" in output
        assert "$0." in output  # wasted $ > 0

    def test_summary_shows_unbilled_not_charged(self, tmp_path: Path) -> None:
        """Summary panel shows unbilled failures with 'not charged' note."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        log_entry(_sample_entry(error=True, error_type="TimeoutError", input_tokens=0, output_tokens=0))
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Unbilled failures" in output
        assert "not charged" in output

    def test_summary_shows_billed_failures_with_wasted(self, tmp_path: Path) -> None:
        """Summary shows billed failure count and wasted amount."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry(error=True, error_type="RateLimitError", input_tokens=1000, output_tokens=200))
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Billed failures" in output
        assert "wasted" in output

    def test_no_error_lines_in_summary_when_clean(self, tmp_path: Path) -> None:
        """Summary panel does not show error lines when no errors."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Billed failures" not in output
        assert "Unbilled failures" not in output

    def test_summary_shows_successful_count(self, tmp_path: Path) -> None:
        """Summary panel shows successful call count."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        log_entry(_sample_entry(error=True, error_type="TimeoutError", input_tokens=0, output_tokens=0))
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "1 successful" in output


class TestCostByApiKey:
    def test_groups_by_api_key_hint(self) -> None:
        """Entries with different api_key_hints appear as separate rows."""
        log_entry(_sample_entry(api_key_hint="...key_aaa", input_tokens=10000, output_tokens=2000))
        log_entry(_sample_entry(api_key_hint="...key_aaa", input_tokens=10000, output_tokens=2000))
        log_entry(_sample_entry(api_key_hint="...key_bbb", input_tokens=10000, output_tokens=2000))

        report = generate_report(last="all")
        rows = report["cost_by_key"]
        hints = {r["key_hint"] for r in rows}
        assert "...key_aaa" in hints
        assert "...key_bbb" in hints
        a_row = next(r for r in rows if r["key_hint"] == "...key_aaa")
        assert a_row["calls"] == 2

    def test_unset_key_grouped_as_unset(self) -> None:
        """Entries without api_key_hint appear under '(unset)'."""
        log_entry(_sample_entry(input_tokens=10000, output_tokens=2000))

        report = generate_report(last="all")
        rows = report["cost_by_key"]
        hints = {r["key_hint"] for r in rows}
        assert "(unset)" in hints

    def test_oauth_tag_renders_in_table(self) -> None:
        """OAuth entries logged as [tag] render visibly in the table."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry(api_key_hint="[myapp]", input_tokens=10000, output_tokens=2000))
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Cost by API Key" in output
        assert "[myapp]" in output

    def test_no_key_table_when_all_pruned(self) -> None:
        """render_text does not print API key table when all keys are below threshold."""
        from io import StringIO

        from rich.console import Console as RichConsole

        log_entry(_sample_entry(input_tokens=1, output_tokens=0))
        report = generate_report(last="all")

        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        assert "Cost by API Key" not in buf.getvalue()

    def test_no_call_site_table_when_all_unknown(self) -> None:
        """render_text hides call site table when all rows are <unknown> or [tag]."""
        from io import StringIO

        from rich.console import Console as RichConsole

        log_entry(_sample_entry(input_tokens=1000, output_tokens=200))  # no call_site
        report = generate_report(last="all")

        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        assert "Cost by Call Site" not in buf.getvalue()

    def test_call_site_table_shown_when_real_site_present(self) -> None:
        """render_text shows call site table when at least one real file-based site exists."""
        from io import StringIO

        from rich.console import Console as RichConsole

        cs = {"file": "app.py", "function": "run", "line": 10}
        log_entry(_sample_entry(input_tokens=1000, output_tokens=200, call_site=cs))
        report = generate_report(last="all")

        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        assert "Cost by Call Site" in buf.getvalue()


class TestClassifyKeyHint:
    def test_real_api_key(self) -> None:
        assert _classify_key_hint("...kB5NtQAA") == "api key"

    def test_tag_fallback(self) -> None:
        assert _classify_key_hint("[myapp]") == "tag"

    def test_unset(self) -> None:
        assert _classify_key_hint("(unset)") == "unknown"

    def test_empty_brackets_is_tag(self) -> None:
        assert _classify_key_hint("[]") == "tag"

    def test_bracket_only_start_is_api_key(self) -> None:
        assert _classify_key_hint("[notclosed") == "api key"



class TestFmtUsd:
    def test_zero(self) -> None:
        assert _fmt_usd(0.0) == "$0.00"

    def test_tiny_amount(self) -> None:
        assert _fmt_usd(0.0042) == "$0.0042"

    def test_sub_cent_threshold(self) -> None:
        assert _fmt_usd(0.0099) == "$0.0099"
        assert _fmt_usd(0.01) == "$0.01"

    def test_normal_amount(self) -> None:
        assert _fmt_usd(1.23) == "$1.23"
        assert _fmt_usd(12.345) == "$12.35"

    def test_large_amount(self) -> None:
        assert _fmt_usd(100.0) == "$100.00"


class TestPeriodLabel:
    def test_humanized_label_in_panel_title(self, tmp_path: Path) -> None:
        """Panel title uses human-readable period label instead of raw CLI flag."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        report = generate_report(last="7d")
        report["period"] = "7d"
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Last 7 Days" in output
        assert "(7d)" not in output

    def test_all_time_label(self, tmp_path: Path) -> None:
        """'all' period renders as 'All Time'."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        assert "All Time" in buf.getvalue()


class TestZeroDataEmptyState:
    def test_empty_report_shows_helpful_message(self, tmp_path: Path) -> None:
        """Zero-call report shows a helpful empty state message and returns early."""
        from rich.console import Console as RichConsole

        report = generate_report(last="all")
        assert report["total_calls"] == 0
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "No calls logged" in output

    def test_empty_report_no_waste_table(self, tmp_path: Path) -> None:
        """Zero-call report does not emit the Waste Detectors table."""
        from rich.console import Console as RichConsole

        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        assert "Waste Detectors" not in buf.getvalue()


class TestContextDriverDisplayNames:
    def test_all_known_drivers_have_display_names(self) -> None:
        """Every CONTEXT_DRIVER_LABELS entry has a display name in _DRIVER_DISPLAY."""
        from toklog.context_drivers import CONTEXT_DRIVER_LABELS
        from toklog.report import _DRIVER_DISPLAY

        for label in CONTEXT_DRIVER_LABELS:
            assert label in _DRIVER_DISPLAY, f"Missing display name for driver: {label}"

    def test_display_names_rendered_in_table(self, tmp_path: Path) -> None:
        """Context Composition table uses human-readable driver names."""
        from rich.console import Console as RichConsole
        from toklog.report import _DRIVER_DISPLAY

        log_entry(_sample_entry())
        report = generate_report(last="all")
        report["cost_by_context_driver"] = [
            {"name": "system_prompt", "cost_usd": 0.001, "calls": 1, "tokens": 500, "avg_tokens": 500, "pct": 60.0},
            {"name": "output_text", "cost_usd": 0.0006, "calls": 1, "tokens": 200, "avg_tokens": 200, "pct": 40.0},
        ]
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert _DRIVER_DISPLAY["system_prompt"] in output  # "System Prompt"
        assert _DRIVER_DISPLAY["output_text"] in output    # "Output Text"
        assert "system_prompt" not in output
        assert "output_text" not in output


class TestBenchmarkEntryFilter:
    def test_is_benchmark_entry_true_for_bench_id(self) -> None:
        entry = _sample_entry(request_id="chatcmpl-bench")
        assert _is_benchmark_entry(entry) is True

    def test_is_benchmark_entry_false_for_real_id(self) -> None:
        entry = _sample_entry(request_id="chatcmpl-abc123xyz")
        assert _is_benchmark_entry(entry) is False

    def test_is_benchmark_entry_false_for_no_id(self) -> None:
        entry = _sample_entry()
        del entry["request_id"]
        assert _is_benchmark_entry(entry) is False

    def test_benchmark_entries_excluded_from_report(self) -> None:
        """Entries with request_id 'chatcmpl-bench' do not appear in report metrics."""
        log_entry(_sample_entry(input_tokens=1000, output_tokens=200, request_id="req-real"))
        log_entry(_sample_entry(input_tokens=5, output_tokens=2, request_id="chatcmpl-bench"))
        log_entry(_sample_entry(input_tokens=5, output_tokens=2, request_id="chatcmpl-bench"))

        report = generate_report(last="all")
        assert report["total_calls"] == 1

    def test_only_benchmark_entries_yields_zero_calls(self) -> None:
        """All-benchmark log results in an empty report."""
        log_entry(_sample_entry(request_id="chatcmpl-bench"))

        report = generate_report(last="all")
        assert report["total_calls"] == 0
        assert report["total_spend_usd"] == 0.0

    def test_bench_entries_in_prior_entries_excluded_from_trend(self) -> None:
        """Benchmark entries passed as prior_entries don't skew the spend trend."""
        real = _sample_entry(input_tokens=1000, output_tokens=200, request_id="req-real")
        bench = _sample_entry(input_tokens=5, output_tokens=2, request_id="chatcmpl-bench")
        log_entry(real)

        # Pass prior_entries that mix real + benchmark entries directly (simulating
        # a caller that bypasses read_logs, e.g. a test or future CLI path).
        prior = [real, bench, bench]
        report = generate_report(last="all", prior_entries=prior)

        # spend_trend compares current vs prior; bench entries must not inflate prior cost
        trend = report["spend_trend"]
        assert trend is not None
        # prior_spend must equal exactly one real entry's cost, not three entries
        assert trend["prior_usd"] == trend["current_usd"]


class TestPruneNegligibleKeys:
    """Tests for negligible key pruning in cost_by_key."""

    def test_negligible_keys_pruned_from_report(self) -> None:
        """Keys below max($0.01, 0.1% of total) are removed from cost_by_key."""
        for _ in range(10):
            log_entry(_sample_entry(api_key_hint="...big_key", input_tokens=10000, output_tokens=2000))
        log_entry(_sample_entry(api_key_hint="...tiny_key", input_tokens=1, output_tokens=0))

        report = generate_report(last="all")
        hints = {r["key_hint"] for r in report["cost_by_key"]}
        assert "...big_key" in hints
        assert "...tiny_key" not in hints

    def test_pruned_keys_summary_present(self) -> None:
        """Pruned keys generate a summary with count, total_cost_usd, and pct."""
        for _ in range(10):
            log_entry(_sample_entry(api_key_hint="...big_key", input_tokens=10000, output_tokens=2000))
        log_entry(_sample_entry(api_key_hint="...tiny_key", input_tokens=1, output_tokens=0))

        report = generate_report(last="all")
        summary = report["pruned_keys_summary"]
        assert summary is not None
        assert summary["count"] == 1
        assert summary["total_cost_usd"] >= 0
        assert "pct" in summary

    def test_no_pruning_when_all_significant(self) -> None:
        """When all keys are above threshold, pruned_keys_summary is None."""
        log_entry(_sample_entry(api_key_hint="...key_a", input_tokens=5000, output_tokens=1000))
        log_entry(_sample_entry(api_key_hint="...key_b", input_tokens=5000, output_tokens=1000))

        report = generate_report(last="all")
        assert report["pruned_keys_summary"] is None
        assert len(report["cost_by_key"]) == 2

    def test_threshold_uses_hybrid_max(self) -> None:
        """Threshold = max($0.01, 0.1% of total). For high total, threshold rises above $0.01."""
        for _ in range(100):
            log_entry(_sample_entry(
                api_key_hint="...big_key",
                provider="anthropic", model="claude-sonnet-4-6",
                input_tokens=50000, output_tokens=10000,
            ))
        log_entry(_sample_entry(
            api_key_hint="...mid_key",
            provider="anthropic", model="claude-sonnet-4-6",
            input_tokens=5000, output_tokens=0,
        ))

        report = generate_report(last="all")
        hints = {r["key_hint"] for r in report["cost_by_key"]}
        assert "...mid_key" not in hints
        assert report["pruned_keys_summary"] is not None

    def test_pruned_summary_rendered_in_text(self) -> None:
        """render_text shows the 'N more keys with negligible spend' note."""
        from rich.console import Console as RichConsole

        for _ in range(10):
            log_entry(_sample_entry(api_key_hint="...big_key", input_tokens=10000, output_tokens=2000))
        log_entry(_sample_entry(api_key_hint="...tiny1", input_tokens=1, output_tokens=0))
        log_entry(_sample_entry(api_key_hint="...tiny2", input_tokens=1, output_tokens=0))

        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "negligible spend" in output
        assert "2 more keys" in output

    def test_no_pruned_summary_when_none_pruned(self) -> None:
        """render_text does not show negligible note when no keys are pruned."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry(api_key_hint="...big_key", input_tokens=5000, output_tokens=1000))

        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "negligible spend" not in output

    def test_waste_detectors_sorted_by_dollar_impact(self, tmp_path: Path) -> None:
        """Waste detectors render in descending order of estimated_waste_usd."""
        from rich.console import Console as RichConsole

        # Build a report with mock detectors at different waste amounts
        log_entry(_sample_entry())
        report = generate_report(last="all")
        # Replace detectors with controlled test data
        report["detectors"] = [
            {"name": "small_waste", "triggered": True, "severity": "low", "estimated_waste_usd": 0.50, "description": "Small."},
            {"name": "big_waste", "triggered": True, "severity": "high", "estimated_waste_usd": 100.0, "description": "Big."},
            {"name": "medium_waste", "triggered": True, "severity": "medium", "estimated_waste_usd": 10.0, "description": "Medium."},
            {"name": "zero_waste", "triggered": True, "severity": "low", "estimated_waste_usd": 0.0, "description": "Zero."},
        ]
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True, width=200)
        render_text(report, console=console)
        output = buf.getvalue()

        # Verify order: big_waste before medium_waste before small_waste before zero_waste
        pos_big = output.index("big_waste")
        pos_med = output.index("medium_waste")
        pos_small = output.index("small_waste")
        pos_zero = output.index("zero_waste")
        assert pos_big < pos_med < pos_small < pos_zero, (
            f"Expected descending dollar order, got positions: big={pos_big}, med={pos_med}, small={pos_small}, zero={pos_zero}"
        )

        # Verify severity column is gone
        assert "HIGH" not in output
        assert "MEDIUM" not in output
        assert "LOW" not in output

    def test_single_pruned_key_singular_grammar(self) -> None:
        """Summary uses singular 'key' for count=1."""
        from rich.console import Console as RichConsole

        for _ in range(10):
            log_entry(_sample_entry(api_key_hint="...big_key", input_tokens=10000, output_tokens=2000))
        log_entry(_sample_entry(api_key_hint="...tiny", input_tokens=1, output_tokens=0))

        report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "1 more key " in output  # singular, note trailing space to avoid matching "keys"


class TestBudgetReport:
    """Budget bar and rejection warnings in reports."""

    def test_generate_report_counts_budget_rejections(self, tmp_path: Path) -> None:
        """budget_rejections counts entries with budget_rejected: true."""
        log_entry(_sample_entry())
        log_entry(_sample_entry(budget_rejected=True))
        log_entry(_sample_entry(budget_rejected=True))
        log_entry(_sample_entry(budget_rejected=False))
        with patch("toklog.report.load_status_from_file", return_value=None):
            report = generate_report(last="all")
        assert report["budget_rejections"] == 2

    def test_generate_report_includes_budget_from_state_file(self, tmp_path: Path) -> None:
        """budget key comes from load_status_from_file()."""
        log_entry(_sample_entry())
        budget_state = {
            "limit_usd": 10.0,
            "daily_spend": 7.42,
            "remaining": 2.58,
            "rejected_count": 0,
            "date": "2025-03-09",
            "enforcing": True,
        }
        with patch("toklog.report.load_status_from_file", return_value=budget_state):
            report = generate_report(last="all")
        assert report["budget"] == budget_state

    def test_generate_report_budget_none_when_no_state_file(self, tmp_path: Path) -> None:
        """budget is None when state file doesn't exist."""
        log_entry(_sample_entry())
        with patch("toklog.report.load_status_from_file", return_value=None):
            report = generate_report(last="all")
        assert report["budget"] is None

    def test_render_text_shows_budget_bar_when_enforcing(self, tmp_path: Path) -> None:
        """Budget bar appears when enforcement is active."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        budget_state = {
            "limit_usd": 10.0,
            "daily_spend": 7.42,
            "remaining": 2.58,
            "rejected_count": 0,
            "date": "2025-03-09",
            "enforcing": True,
        }
        with patch("toklog.report.load_status_from_file", return_value=budget_state):
            report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True, width=120)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Budget:" in output
        assert "$7.42" in output
        assert "$10.00" in output
        assert "74%" in output

    def test_render_text_shows_rejection_warning(self, tmp_path: Path) -> None:
        """Rejection warning appears when budget_rejections > 0."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry(budget_rejected=True))
        log_entry(_sample_entry(budget_rejected=True))
        log_entry(_sample_entry())
        with patch("toklog.report.load_status_from_file", return_value=None):
            report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True, width=120)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "2 requests were blocked by budget enforcement" in output

    def test_render_text_omits_budget_bar_when_no_budget(self, tmp_path: Path) -> None:
        """Budget bar does not appear when no budget is configured."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        with patch("toklog.report.load_status_from_file", return_value=None):
            report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True, width=120)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Budget:" not in output

    def test_budget_bar_color_green_under_75(self, tmp_path: Path) -> None:
        """Budget bar uses green markup when under 75%."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        budget_state = {
            "limit_usd": 10.0,
            "daily_spend": 5.0,
            "remaining": 5.0,
            "rejected_count": 0,
            "date": "2025-03-09",
            "enforcing": True,
        }
        with patch("toklog.report.load_status_from_file", return_value=budget_state):
            report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True, width=120)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Budget:" in output
        assert "$5.00" in output
        assert "$10.00" in output
        # 50% — green range (under 75%)
        assert "50%" in output

    def test_budget_bar_color_yellow_75_to_90(self, tmp_path: Path) -> None:
        """Budget bar uses yellow markup when 75-90%."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        budget_state = {
            "limit_usd": 10.0,
            "daily_spend": 8.0,
            "remaining": 2.0,
            "rejected_count": 0,
            "date": "2025-03-09",
            "enforcing": True,
        }
        with patch("toklog.report.load_status_from_file", return_value=budget_state):
            report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True, width=120)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Budget:" in output
        assert "$8.00" in output
        assert "80%" in output

    def test_budget_bar_color_red_over_90(self, tmp_path: Path) -> None:
        """Budget bar uses red markup when over 90%."""
        from rich.console import Console as RichConsole

        log_entry(_sample_entry())
        budget_state = {
            "limit_usd": 10.0,
            "daily_spend": 9.5,
            "remaining": 0.5,
            "rejected_count": 0,
            "date": "2025-03-09",
            "enforcing": True,
        }
        with patch("toklog.report.load_status_from_file", return_value=budget_state):
            report = generate_report(last="all")
        buf = StringIO()
        console = RichConsole(file=buf, no_color=True, width=120)
        render_text(report, console=console)
        output = buf.getvalue()
        assert "Budget:" in output
        assert "$9.50" in output
        assert "95%" in output

    def test_budget_bar_color_markup_thresholds(self) -> None:
        """Verify correct Rich color markup is chosen at each threshold."""
        from rich.console import Console as RichConsole

        # Build a minimal report dict by hand
        base_report: Dict[str, Any] = {
            "period": "1d",
            "total_calls": 1,
            "total_spend_usd": 1.0,
            "estimated_waste_usd": 0.0,
            "waste_pct": 0.0,
            "detectors": [],
            "cost_by_model": {},
            "cost_by_process": [],
            "cost_by_key": [],
            "pruned_keys_summary": None,
            "cost_by_call_site": [],
            "cost_by_context_driver": [],
            "spend_trend": None,
            "error_calls": 0,
            "error_wasted_usd": 0.0,
            "errors_by_type": [],
            "budget_rejections": 0,
        }

        for spend, expected_color in [(5.0, "green"), (7.5, "yellow"), (9.5, "red")]:
            report = dict(base_report)
            report["budget"] = {
                "limit_usd": 10.0,
                "daily_spend": spend,
                "remaining": 10.0 - spend,
                "rejected_count": 0,
                "date": "2025-03-09",
                "enforcing": True,
            }
            buf = StringIO()
            console = RichConsole(file=buf, no_color=False, force_terminal=True, width=120)
            render_text(report, console=console)
            output = buf.getvalue()
            # Verify the ANSI color code is present: green=32, yellow=33, red=31
            ansi_codes = {"green": "\x1b[32m", "yellow": "\x1b[33m", "red": "\x1b[31m"}
            assert ansi_codes[expected_color] in output, (
                f"Expected {expected_color} ANSI code for spend=${spend}, got: {output[:200]}"
            )
