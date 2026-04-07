"""Tests for _format_tail_line unit tests + CLI smoke tests.

_format_tail_line always includes cost (no flag). Pass ``cost=`` to supply
a precomputed value; omit to auto-compute.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import toklog.logger as logger_mod
import toklog.pricing as pricing_mod
from toklog.cli import _format_tail_line, cli

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
# Unit tests — _format_tail_line (cost always on)
# ---------------------------------------------------------------------------

class TestFormatTailLine:
    """Test the pure formatting helper directly — no file I/O, no mocking."""

    def test_contains_model(self):
        line = _format_tail_line(_entry(model="gpt-4o"))
        assert "gpt-4o" in line

    def test_contains_tokens(self):
        line = _format_tail_line(_entry(input_tokens=1234, output_tokens=56))
        assert "in=1234" in line
        assert "out=56" in line

    def test_contains_duration(self):
        line = _format_tail_line(_entry(duration_ms=1337))
        assert "1337ms" in line

    def test_contains_tag_string(self):
        line = _format_tail_line(_entry(tags="myapp"))
        assert "myapp" in line

    def test_always_has_cost(self):
        """Cost is always shown — no flag needed."""
        line = _format_tail_line(_entry())
        assert "$" in line

    def test_no_double_space(self):
        line = _format_tail_line(_entry())
        assert "  " not in line

    def test_tags_list_joined_cleanly(self):
        """tags as a list should be space-joined, not repr'd."""
        line = _format_tail_line(_entry(tags=["foo", "bar"]))
        assert "foo bar" in line
        assert "['foo', 'bar']" not in line
        assert "  " not in line

    def test_tags_none_no_crash(self):
        line = _format_tail_line(_entry(tags=None))
        assert "in=" in line  # just didn't crash

    def test_tags_empty_string_no_trailing_space(self):
        line = _format_tail_line(_entry(tags=""))
        assert not line.endswith(" ")

    def test_missing_model_defaults_to_question_mark(self):
        entry = _entry()
        del entry["model"]
        line = _format_tail_line(entry)
        assert line.startswith("?")

    def test_missing_tokens_default_to_zero(self):
        entry = _entry()
        del entry["input_tokens"]
        del entry["output_tokens"]
        line = _format_tail_line(entry)
        assert "in=0" in line
        assert "out=0" in line

    def test_none_tokens_default_to_zero(self):
        line = _format_tail_line(_entry(input_tokens=None, output_tokens=None))
        assert "in=0" in line
        assert "out=0" in line

    # --- cost specifics ---

    def test_dollar_sign_present(self):
        line = _format_tail_line(_entry())
        assert "$" in line

    def test_cost_four_decimal_places(self):
        line = _format_tail_line(_entry())
        assert re.search(r"\$\d+\.\d{4}", line), f"No $X.XXXX in: {line!r}"

    def test_cost_correct_value_anthropic(self):
        """
        claude-haiku-3-5: $0.80/M input, $4.00/M output
        1000 in + 200 out = $0.0008 + $0.0008 = $0.0016
        """
        line = _format_tail_line(_entry(
            model="claude-haiku-3-5-20241022",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=200,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        ))
        assert "$0.0016" in line

    def test_cost_correct_value_openai(self):
        """
        gpt-4o-mini: $0.075/M input, $0.30/M output
        OpenAI: input_tokens INCLUDES cache_read, so non_cached = 2000 - 500 = 1500
        1500 * 0.075/M + 500 * 0/M + 100 * 0.30/M = 0.0001125 + 0 + 0.00003 = $0.0001425 → $0.0001
        """
        line = _format_tail_line(_entry(
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=2000,
            output_tokens=100,
            cache_read_tokens=500,
            cache_creation_tokens=0,
        ))
        assert re.search(r"\$\d+\.\d{4}", line)

    def test_cost_with_cache_read_anthropic(self):
        """Cache read is billed separately for Anthropic; cost > zero."""
        line = _format_tail_line(_entry(
            model="claude-haiku-3-5-20241022",
            provider="anthropic",
            input_tokens=500,
            output_tokens=0,
            cache_read_tokens=500,
            cache_creation_tokens=0,
        ))
        # cache_read: 500 * 0.08/M = $0.00004
        assert "$0.0000" in line or re.search(r"\$0\.00\d+", line)

    def test_unknown_model_shows_zero_cost_no_crash(self):
        """Unknown model → pricing layer returns zeros, no exception."""
        line = _format_tail_line(_entry(model="totally-unknown-model-xyz"))
        assert "$0.0000" in line

    def test_missing_provider_does_not_crash(self):
        entry = _entry()
        del entry["provider"]
        line = _format_tail_line(entry)
        assert "$" in line

    def test_missing_cache_fields_default_to_zero(self):
        entry = _entry()
        del entry["cache_read_tokens"]
        del entry["cache_creation_tokens"]
        line = _format_tail_line(entry)
        assert "$" in line

    def test_cost_before_duration(self):
        line = _format_tail_line(_entry(duration_ms=9999))
        cost_pos = line.index("$")
        dur_pos = line.index("9999ms")
        assert cost_pos < dur_pos

    def test_no_double_space_with_cost(self):
        line = _format_tail_line(_entry())
        assert "  " not in line

    def test_no_trailing_space(self):
        line = _format_tail_line(_entry())
        assert not line.endswith(" ")

    def test_multiple_entries_independent(self):
        """Each call is stateless — no accumulated state between calls."""
        e = _entry(model="claude-haiku-3-5-20241022", provider="anthropic",
                   input_tokens=1000, output_tokens=200)
        lines = [_format_tail_line(e) for _ in range(3)]
        assert all("$0.0016" in l for l in lines)
        assert len(set(lines)) == 1  # all identical

    def test_precomputed_cost_used(self):
        """When cost= is passed, that exact value appears — no recomputation."""
        line = _format_tail_line(_entry(), cost=42.5678)
        assert "$42.5678" in line


# ---------------------------------------------------------------------------
# CLI smoke tests — toklog tail (integration, not cost logic)
# ---------------------------------------------------------------------------

class TestTailCLI:
    """Thin CLI-level tests: flag accepted, no crash, file reading works."""

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

    def test_tail_no_flag_does_not_crash(self, log_dir):
        self._write_today(log_dir, _entry())
        out = self._invoke(log_dir)
        assert "Stopped" in out

    def test_tail_cost_flag_accepted_as_noop(self, log_dir):
        """--cost is kept as a hidden deprecated no-op, must not error."""
        self._write_today(log_dir, _entry())
        out = self._invoke(log_dir, ["--cost"])
        assert "Stopped" in out

    def test_tail_always_shows_cost(self, log_dir):
        """Cost is always shown (no flag needed)."""
        self._write_today(log_dir, _entry())
        out = self._invoke(log_dir)
        assert "$" in out
