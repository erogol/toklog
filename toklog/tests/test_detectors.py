"""Tests for the detectors module."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from toklog.detectors import (
    _classify_cache_writes,
    _effective_input,
    _entry_cost,
    detect_cache_miss,
    detect_cache_write_churn,
    detect_high_spend_process,
    detect_model_downgrade_opportunity,
    detect_output_truncation,
    detect_thinking_overhead,
    detect_tool_schema_bloat,
    detect_unbounded_context,
    run_all,
)
from toklog.pricing import get_cache_prices


@pytest.fixture(autouse=True)
def _deterministic_pricing():
    """Disable live LiteLLM pricing so tests use hardcoded PRICING table only."""
    with patch("toklog.pricing._live_cache_loaded", True), \
         patch("toklog.pricing._live_cache", None):
        yield


def _base_entry(**overrides: Any) -> Dict[str, Any]:
    """Create a base log entry with sensible defaults."""
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
        "tool_calls_made": None,
        "tags": "test",
        "streaming": False,
        "duration_ms": 1000,
        "error": False,
        "error_type": None,
        "request_id": "req_1",
        "total_message_chars": 1000,
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# _effective_input
# ---------------------------------------------------------------------------


class TestEffectiveInput:
    def test_openai_includes_cache_read_in_input(self) -> None:
        """OpenAI: input_tokens already includes cache_read, so only add cache_creation."""
        entry = _base_entry(provider="openai", input_tokens=1000, cache_read_tokens=500, cache_creation_tokens=0)
        assert _effective_input(entry) == 1000  # 1000 + 0 (cache_read already in input)

    def test_openai_adds_cache_creation(self) -> None:
        entry = _base_entry(provider="openai", input_tokens=1000, cache_read_tokens=0, cache_creation_tokens=200)
        assert _effective_input(entry) == 1200

    def test_anthropic_adds_both_cache_fields(self) -> None:
        """Anthropic: input_tokens excludes cache tokens — add both."""
        entry = _base_entry(provider="anthropic", input_tokens=3, cache_read_tokens=17000, cache_creation_tokens=22000)
        assert _effective_input(entry) == 39003

    def test_null_fields(self) -> None:
        entry = _base_entry(provider="openai", input_tokens=None, cache_read_tokens=None, cache_creation_tokens=None)
        assert _effective_input(entry) == 0


# ---------------------------------------------------------------------------
# _entry_cost (cache-aware)
# ---------------------------------------------------------------------------


class TestEntryCost:
    def test_uses_cost_usd_if_present(self) -> None:
        entry = _base_entry(cost_usd=5.0)
        assert _entry_cost(entry) == 5.0

    def test_openai_no_cache(self) -> None:
        """OpenAI with no cache: simple input + output."""
        entry = _base_entry(provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=200)
        # gpt-4o: input=0.00125/1K, output=0.005/1K
        expected = 1000 * 0.00125 / 1000 + 200 * 0.005 / 1000
        assert abs(_entry_cost(entry) - expected) < 1e-10

    def test_anthropic_with_cache(self) -> None:
        """Anthropic with cache tokens — cost must include cache read and write."""
        entry = _base_entry(
            provider="anthropic", model="claude-sonnet-4-6",
            input_tokens=3, output_tokens=50,
            cache_read_tokens=17000, cache_creation_tokens=22000,
        )
        cost = _entry_cost(entry)
        # claude-sonnet-4-6: input=0.003/1K, output=0.015/1K
        # cache_read=0.003*0.1=0.0003/1K, cache_write=0.003*1.25=0.00375/1K
        # 3*0.003/1000 + 17000*0.0003/1000 + 22000*0.00375/1000 + 50*0.015/1000
        expected = (3 * 0.003 + 17000 * 0.0003 + 22000 * 0.00375 + 50 * 0.015) / 1000
        assert abs(cost - expected) < 1e-10

    def test_openai_with_cache_read(self) -> None:
        """OpenAI cache: input_tokens includes cache_read — non-cached billed at full rate."""
        entry = _base_entry(
            provider="openai", model="gpt-4o",
            input_tokens=1000, output_tokens=100,
            cache_read_tokens=500,
        )
        cost = _entry_cost(entry)
        # 500 non-cached at full price + 500 cached at half price + output
        expected = (500 * 0.00125 / 1000) + (500 * 0.000625 / 1000) + (100 * 0.005 / 1000)
        assert abs(cost - expected) < 1e-10


# ---------------------------------------------------------------------------
# Cache Miss
# ---------------------------------------------------------------------------


class TestCacheMissDetector:
    def test_triggers_when_same_hash_no_cache(self) -> None:
        """Should trigger when >80% calls share a hash and >90% have no cache."""
        entries = [_base_entry(system_prompt_hash="samehash", cache_read_tokens=0) for _ in range(10)]
        result = detect_cache_miss(entries)
        assert result.triggered is True
        assert result.severity == "high"
        assert result.estimated_waste_usd > 0

    def test_waste_bounded_by_spend(self) -> None:
        """Waste should use min(effective_input) as sp_tokens, so waste <= real spend."""
        entries = [
            _base_entry(
                system_prompt_hash="samehash", cache_read_tokens=0,
                input_tokens=500 + i * 200, model="gpt-4o",
            )
            for i in range(10)
        ]
        result = detect_cache_miss(entries)
        assert result.triggered is True
        from toklog.detectors import _input_price_per_token
        price = _input_price_per_token("gpt-4o")
        expected = 500 * 0.5 * price * 10
        assert abs(result.estimated_waste_usd - round(expected, 4)) < 0.001
        total_input_cost = sum((500 + i * 200) * price for i in range(10))
        assert result.estimated_waste_usd <= total_input_cost

    def test_effective_input_used_for_cacheable_filter(self) -> None:
        """Anthropic entries with low input_tokens but high cache tokens should be eligible."""
        entries = [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                system_prompt_hash="samehash",
                input_tokens=3, cache_read_tokens=0, cache_creation_tokens=0,
            )
            for _ in range(10)
        ]
        # effective_input = 3 → below 500 threshold → not cacheable → not triggered
        result = detect_cache_miss(entries)
        assert result.triggered is False

        # With substantial cache_creation, effective_input = 3 + 0 + 5000 = 5003
        entries_big = [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                system_prompt_hash="samehash",
                input_tokens=3, cache_read_tokens=0, cache_creation_tokens=5000,
            )
            for _ in range(10)
        ]
        # But these have cache activity (cache_creation > 0), so >90% check filters them
        result_big = detect_cache_miss(entries_big)
        assert result_big.triggered is False  # cache_creation means cache IS being used

    def test_no_trigger_with_cache_reads(self) -> None:
        entries = [_base_entry(system_prompt_hash="samehash", cache_read_tokens=500) for _ in range(10)]
        result = detect_cache_miss(entries)
        assert result.triggered is False

    def test_no_trigger_with_cache_creation(self) -> None:
        entries = [_base_entry(system_prompt_hash="samehash", cache_read_tokens=0, cache_creation_tokens=1000) for _ in range(10)]
        result = detect_cache_miss(entries)
        assert result.triggered is False

    def test_no_trigger_mixed_read_and_creation(self) -> None:
        entries = (
            [_base_entry(system_prompt_hash="samehash", cache_read_tokens=0, cache_creation_tokens=1000)]
            + [_base_entry(system_prompt_hash="samehash", cache_read_tokens=800, cache_creation_tokens=0) for _ in range(9)]
        )
        result = detect_cache_miss(entries)
        assert result.triggered is False

    def test_no_trigger_diverse_hashes(self) -> None:
        entries = [_base_entry(system_prompt_hash=f"hash_{i}") for i in range(10)]
        result = detect_cache_miss(entries)
        assert result.triggered is False

    def test_empty_data(self) -> None:
        result = detect_cache_miss([])
        assert result.triggered is False

    def test_multiple_features_both_flagged(self) -> None:
        entries = [
            _base_entry(system_prompt_hash="hash_A", cache_read_tokens=0) for _ in range(5)
        ] + [
            _base_entry(system_prompt_hash="hash_B", cache_read_tokens=0) for _ in range(5)
        ]
        result = detect_cache_miss(entries)
        assert result.triggered is True
        assert "hash_A" in result.details["flagged_hashes"]
        assert "hash_B" in result.details["flagged_hashes"]


# ---------------------------------------------------------------------------
# Cache Write Churn
# ---------------------------------------------------------------------------


class TestCacheWriteChurn:
    def test_triggers_high_creation_ratio(self) -> None:
        """Should trigger when creation_ratio > 0.4 in a session."""
        entries = [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                system_prompt_hash="session1",
                input_tokens=3, cache_read_tokens=5000, cache_creation_tokens=5000,
                timestamp=f"2025-03-09T10:{i:02d}:00.000Z",
            )
            for i in range(5)
        ]
        result = detect_cache_write_churn(entries)
        assert result.triggered is True
        assert result.estimated_waste_usd > 0
        assert len(result.details["flagged_sessions"]) == 1
        assert result.details["flagged_sessions"][0]["creation_ratio"] == 0.5

    def test_triggers_high_absolute_waste(self) -> None:
        """Should trigger when absolute waste > $0.50 even if ratio is moderate."""
        entries = [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                system_prompt_hash="session1",
                input_tokens=3, cache_read_tokens=200000, cache_creation_tokens=50000,
                timestamp=f"2025-03-09T10:{i:02d}:00.000Z",
            )
            for i in range(5)
        ]
        result = detect_cache_write_churn(entries)
        # creation_ratio = 250000 / (1000000 + 250000) = 0.2 < 0.4
        # but absolute waste should be > $0.50 due to large churn tokens
        assert result.triggered is True

    def test_no_trigger_healthy_caching(self) -> None:
        """First call creates cache, rest read — healthy pattern."""
        entries = [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                system_prompt_hash="session1",
                input_tokens=3, cache_read_tokens=0, cache_creation_tokens=10000,
                timestamp="2025-03-09T10:00:00.000Z",
            ),
        ] + [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                system_prompt_hash="session1",
                input_tokens=3, cache_read_tokens=10000, cache_creation_tokens=0,
                timestamp=f"2025-03-09T10:{i:02d}:00.000Z",
            )
            for i in range(1, 10)
        ]
        result = detect_cache_write_churn(entries)
        assert result.triggered is False

    def test_no_trigger_openai(self) -> None:
        """OpenAI caching is automatic — only Anthropic sessions are eligible."""
        entries = [
            _base_entry(
                provider="openai", model="gpt-4o",
                system_prompt_hash="session1",
                input_tokens=1000, cache_read_tokens=500, cache_creation_tokens=500,
                timestamp=f"2025-03-09T10:{i:02d}:00.000Z",
            )
            for i in range(5)
        ]
        result = detect_cache_write_churn(entries)
        assert result.triggered is False

    def test_no_trigger_few_calls(self) -> None:
        entries = [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                system_prompt_hash="session1",
                input_tokens=3, cache_read_tokens=0, cache_creation_tokens=10000,
            )
            for _ in range(2)
        ]
        result = detect_cache_write_churn(entries)
        assert result.triggered is False

    def test_empty_data(self) -> None:
        result = detect_cache_write_churn([])
        assert result.triggered is False


# ---------------------------------------------------------------------------
# Output Truncation
# ---------------------------------------------------------------------------


class TestOutputTruncationDetector:
    def test_triggers_when_output_hits_ceiling(self) -> None:
        entries = [_base_entry(max_tokens_set=1000, output_tokens=960)]
        result = detect_output_truncation(entries)
        assert result.triggered is True
        assert result.severity == "high"
        assert result.details["truncated_calls"] == 1
        assert result.estimated_waste_usd == 0

    def test_potential_retry_cost_present(self) -> None:
        """Triggered results should include a potential_retry_cost detail."""
        entries = [_base_entry(max_tokens_set=1000, output_tokens=960)]
        result = detect_output_truncation(entries)
        assert result.triggered is True
        assert result.details["potential_retry_cost"] > 0

    def test_triggers_exact_ceiling(self) -> None:
        entries = [_base_entry(max_tokens_set=1000, output_tokens=1000)]
        result = detect_output_truncation(entries)
        assert result.triggered is True

    def test_no_trigger_below_ceiling(self) -> None:
        entries = [_base_entry(max_tokens_set=4096, output_tokens=100) for _ in range(5)]
        result = detect_output_truncation(entries)
        assert result.triggered is False

    def test_no_trigger_no_max_tokens(self) -> None:
        entries = [_base_entry(max_tokens_set=None, output_tokens=1000) for _ in range(5)]
        result = detect_output_truncation(entries)
        assert result.triggered is False

    def test_no_waste_figure_for_truncated_calls(self) -> None:
        entries = [
            _base_entry(max_tokens_set=1000, output_tokens=960, model="gpt-4o"),
            _base_entry(max_tokens_set=4096, output_tokens=100, model="gpt-4o"),
        ]
        result = detect_output_truncation(entries)
        assert result.triggered is True
        assert result.details["truncated_calls"] == 1
        assert result.estimated_waste_usd == 0

    def test_empty_data(self) -> None:
        result = detect_output_truncation([])
        assert result.triggered is False


# ---------------------------------------------------------------------------
# Tool Schema Bloat
# ---------------------------------------------------------------------------


class TestToolSchemaBloatDetector:
    def test_triggers_zero_use(self) -> None:
        entries = [
            _base_entry(tool_count=5, tool_schema_tokens=800, tool_calls_made=0)
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        assert result.triggered is True
        assert result.severity == "high"
        assert result.details["zero_use_calls"] == 5

    def test_triggers_extreme_ratio_large_context(self) -> None:
        entries = [
            _base_entry(tool_count=5, tool_schema_tokens=1500, input_tokens=2500, tool_calls_made=2)
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        assert result.triggered is True
        assert result.severity == "medium"
        assert result.details["extreme_ratio_calls"] == 5

    def test_uses_effective_input_for_ratio(self) -> None:
        """Anthropic: schema ratio should be computed against effective input (including cache)."""
        # input_tokens=3, cache_read=40000 → effective=40003
        # schema_tokens=9000 → ratio = 9000/40003 ≈ 0.22 → NOT bloated
        entries = [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                tool_count=100, tool_schema_tokens=9000, tool_calls_made=2,
                input_tokens=3, cache_read_tokens=40000,
            )
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        assert result.details["extreme_ratio_calls"] == 0

    def test_no_trigger_extreme_ratio_small_context(self) -> None:
        entries = [
            _base_entry(tool_count=5, tool_schema_tokens=600, input_tokens=10, tool_calls_made=2)
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        assert result.triggered is False
        assert result.details["extreme_ratio_calls"] == 0

    def test_no_trigger_when_tools_used(self) -> None:
        entries = [
            _base_entry(tool_count=5, tool_schema_tokens=800, tool_calls_made=3, input_tokens=5000)
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        assert result.triggered is False

    def test_no_trigger_small_schema_zero_use(self) -> None:
        entries = [
            _base_entry(tool_count=5, tool_schema_tokens=200, tool_calls_made=0)
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        assert result.triggered is False

    def test_no_trigger_legacy_none(self) -> None:
        entries = [
            _base_entry(tool_count=5, tool_schema_tokens=800, tool_calls_made=None, input_tokens=5000)
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        assert result.triggered is False

    def test_no_trigger_no_tools(self) -> None:
        entries = [_base_entry(tool_schema_tokens=0) for _ in range(5)]
        result = detect_tool_schema_bloat(entries)
        assert result.triggered is False

    def test_empty_data(self) -> None:
        result = detect_tool_schema_bloat([])
        assert result.triggered is False


# ---------------------------------------------------------------------------
# Unbounded Context Growth
# ---------------------------------------------------------------------------


class TestUnboundedContextDetector:
    def test_triggers_monotonic_growth(self) -> None:
        entries = []
        tokens = 1000
        for i in range(8):
            entries.append(
                _base_entry(
                    system_prompt_hash="session1",
                    input_tokens=tokens,
                    timestamp=f"2025-03-09T10:{i:02d}:00.000Z",
                )
            )
            tokens = int(tokens * 1.15)
        result = detect_unbounded_context(entries)
        assert result.triggered is True
        assert result.severity == "high"

    def test_tracks_effective_input_for_anthropic(self) -> None:
        """Anthropic: context growth should be visible via cache_read growth."""
        entries = [
            _base_entry(
                provider="anthropic", model="claude-sonnet-4-6",
                system_prompt_hash="session1",
                input_tokens=3, cache_read_tokens=10000 + i * 5000,
                timestamp=f"2025-03-09T10:{i:02d}:00.000Z",
            )
            for i in range(8)
        ]
        # effective: 10003, 15003, 20003, 25003, 30003, 35003, 40003, 45003
        # Growth is monotonic and last/first = 45003/10003 = 4.5 > 1.5
        result = detect_unbounded_context(entries)
        assert result.triggered is True

    def test_no_trigger_stable_context(self) -> None:
        entries = [
            _base_entry(
                system_prompt_hash="session1",
                input_tokens=1000,
                timestamp=f"2025-03-09T10:{i:02d}:00.000Z",
            )
            for i in range(8)
        ]
        result = detect_unbounded_context(entries)
        assert result.triggered is False

    def test_triggers_with_plateau(self) -> None:
        token_values = [1000, 1200, 1200, 1500, 1800]
        entries = [
            _base_entry(
                system_prompt_hash="session1",
                input_tokens=t,
                timestamp=f"2025-03-09T10:{i:02d}:00.000Z",
            )
            for i, t in enumerate(token_values)
        ]
        result = detect_unbounded_context(entries)
        assert result.triggered is True

    def test_no_trigger_few_calls(self) -> None:
        entries = [
            _base_entry(system_prompt_hash="session1", input_tokens=i * 1000)
            for i in range(1, 4)
        ]
        result = detect_unbounded_context(entries)
        assert result.triggered is False

    def test_empty_data(self) -> None:
        result = detect_unbounded_context([])
        assert result.triggered is False


# ---------------------------------------------------------------------------
# High-Spend Process
# ---------------------------------------------------------------------------


class TestHighSpendProcess:
    def _entry_with_site(self, site: str, **kwargs: Any) -> Dict[str, Any]:
        parts = site.split(":")
        cs = {"file": parts[0], "function": parts[1], "line": int(parts[2])}
        return _base_entry(call_site=cs, **kwargs)

    def test_triggers_high_calls_by_call_site(self) -> None:
        entries = [self._entry_with_site("app.py:run:10") for _ in range(51)]
        result = detect_high_spend_process(entries)
        assert result.triggered is True
        assert result.severity == "low"
        call_site_flags = [s for s in result.details["flagged_sites"] if s["type"] == "call_site"]
        assert len(call_site_flags) >= 1
        assert call_site_flags[0]["calls"] == 51

    def test_triggers_high_calls_by_process(self) -> None:
        entries = [_base_entry(program="my-service", call_site=None) for _ in range(51)]
        result = detect_high_spend_process(entries)
        assert result.triggered is True
        process_flags = [s for s in result.details["flagged_sites"] if s["type"] == "process"]
        assert len(process_flags) >= 1
        assert process_flags[0]["calls"] == 51

    def test_triggers_high_spend(self) -> None:
        entries = [self._entry_with_site("app.py:run:10", cost_usd=21.0)]
        result = detect_high_spend_process(entries)
        assert result.triggered is True
        call_site_flags = [s for s in result.details["flagged_sites"] if s["type"] == "call_site"]
        assert len(call_site_flags) >= 1
        assert call_site_flags[0]["cost_usd"] > 20.0

    def test_no_trigger_below_thresholds(self) -> None:
        entries = [self._entry_with_site("app.py:run:10") for _ in range(49)]
        result = detect_high_spend_process(entries)
        call_site_flags = [s for s in result.details["flagged_sites"] if s["type"] == "call_site"]
        assert len(call_site_flags) == 0

    def test_no_trigger_no_call_site_few_calls(self) -> None:
        entries = [_base_entry(call_site=None, program=f"svc-{i}") for i in range(10)]
        result = detect_high_spend_process(entries)
        assert result.triggered is False

    def test_empty_data(self) -> None:
        result = detect_high_spend_process([])
        assert result.triggered is False


# ---------------------------------------------------------------------------
# Model Downgrade Opportunity
# ---------------------------------------------------------------------------


class TestModelDowngradeOpportunity:
    def test_triggers_for_expensive_model(self) -> None:
        """Should flag Opus calls with potential Sonnet savings."""
        entries = [
            _base_entry(provider="anthropic", model="claude-opus-4-6", input_tokens=1000, output_tokens=500)
            for _ in range(5)
        ]
        result = detect_model_downgrade_opportunity(entries)
        assert result.triggered is True
        assert result.estimated_waste_usd == 0  # observation only
        assert len(result.details["comparisons"]) == 1
        assert result.details["comparisons"][0]["alternative"] == "claude-sonnet"
        assert result.details["comparisons"][0]["potential_savings"] > 0

    def test_no_trigger_for_cheap_model(self) -> None:
        entries = [
            _base_entry(provider="openai", model="gpt-4o-mini", input_tokens=100, output_tokens=50)
            for _ in range(5)
        ]
        result = detect_model_downgrade_opportunity(entries)
        assert result.triggered is False

    def test_empty_data(self) -> None:
        result = detect_model_downgrade_opportunity([])
        assert result.triggered is False


# ---------------------------------------------------------------------------
# Thinking Overhead
# ---------------------------------------------------------------------------


class TestThinkingOverhead:
    def test_triggers_heavy_thinking(self) -> None:
        """Should trigger when thinking > 80% of output and > 500 tokens."""
        entries = [
            _base_entry(
                model="claude-opus-4-6", output_tokens=1000,
                thinking_tokens=900,
            )
        ]
        result = detect_thinking_overhead(entries)
        assert result.triggered is True
        assert result.estimated_waste_usd == 0  # observation only
        assert result.details["heavy_thinking_calls"] == 1
        assert result.details["total_thinking_tokens"] == 900
        assert result.details["total_thinking_cost"] > 0

    def test_no_trigger_moderate_thinking(self) -> None:
        """Should NOT trigger when thinking is < 80% of output."""
        entries = [
            _base_entry(model="claude-opus-4-6", output_tokens=1000, thinking_tokens=500)
        ]
        result = detect_thinking_overhead(entries)
        assert result.triggered is False

    def test_no_trigger_small_thinking(self) -> None:
        """Should NOT trigger when thinking < 500 tokens."""
        entries = [
            _base_entry(model="claude-opus-4-6", output_tokens=100, thinking_tokens=90)
        ]
        result = detect_thinking_overhead(entries)
        assert result.triggered is False

    def test_empty_data(self) -> None:
        result = detect_thinking_overhead([])
        assert result.triggered is False


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_returns_nine_results(self) -> None:
        entries = [_base_entry() for _ in range(5)]
        results = run_all(entries)
        assert len(results) == 10  # Added cost_spike detector

    def test_null_fields_no_crash(self) -> None:
        """Entries with null fields should not crash detectors."""
        entries = [
            _base_entry(
                input_tokens=None,
                output_tokens=None,
                system_prompt_hash=None,
                max_tokens_set=None,
                tags=None,
                tool_schema_tokens=None,
                cache_read_tokens=None,
            )
            for _ in range(5)
        ]
        results = run_all(entries)
        assert len(results) == 10  # Added cost_spike detector
        for r in results:
            assert isinstance(r.triggered, bool)

    def test_detector_names(self) -> None:
        """Verify the expected detector names are returned."""
        entries = [_base_entry() for _ in range(5)]
        results = run_all(entries)
        names = {r.name for r in results}
        assert names == {
            "cache_miss_opportunity",
            "cache_write_churn",
            "output_truncation",
            "tool_schema_bloat",
            "unbounded_context_growth",
            "high_spend_process",
            "model_downgrade_opportunity",
            "thinking_overhead",
            "credential_sharing",
            "cost_spike",
        }


# ---------------------------------------------------------------------------
# SPEC 2026-09-30: thread-aware cache-write churn classification
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-6"


def _anthro_entry(**overrides: Any) -> Dict[str, Any]:
    overrides.setdefault("model", _MODEL)
    overrides.setdefault("provider", "anthropic")
    return _base_entry(**overrides)


class TestClassifyCacheWritesGoodCaching:
    def test_growing_conversation_good_caching_no_waste(self) -> None:
        """Each call reads the full previous prefix and writes only the new turn."""
        entries = [
            _anthro_entry(
                system_prompt_hash="h1", input_tokens=3,
                cache_read_tokens=0, cache_creation_tokens=1000, total_message_chars=500,
                timestamp="2025-03-09T10:00:00.000Z",
            ),
            _anthro_entry(
                system_prompt_hash="h1", input_tokens=3,
                cache_read_tokens=1000, cache_creation_tokens=200, total_message_chars=700,
                timestamp="2025-03-09T10:00:30.000Z",
            ),
            _anthro_entry(
                system_prompt_hash="h1", input_tokens=3,
                cache_read_tokens=1200, cache_creation_tokens=200, total_message_chars=900,
                timestamp="2025-03-09T10:01:00.000Z",
            ),
            _anthro_entry(
                system_prompt_hash="h1", input_tokens=3,
                cache_read_tokens=1400, cache_creation_tokens=200, total_message_chars=1100,
                timestamp="2025-03-09T10:01:30.000Z",
            ),
        ]
        result = detect_cache_write_churn(entries)
        assert result.triggered is False
        assert result.estimated_waste_usd == 0.0
        assert result.details["live_churn_tokens"] == 0

    def test_two_interleaved_threads_both_cache_well(self) -> None:
        """Two parallel conversations sharing one namespace, each caching well → $0."""
        entries = [
            _anthro_entry(system_prompt_hash="h2", input_tokens=3,
                           cache_read_tokens=0, cache_creation_tokens=1000, total_message_chars=500,
                           timestamp="2025-03-09T10:00:00.000Z"),
            _anthro_entry(system_prompt_hash="h2", input_tokens=3,
                           cache_read_tokens=0, cache_creation_tokens=800, total_message_chars=300,
                           timestamp="2025-03-09T10:00:10.000Z"),
            _anthro_entry(system_prompt_hash="h2", input_tokens=3,
                           cache_read_tokens=1000, cache_creation_tokens=200, total_message_chars=700,
                           timestamp="2025-03-09T10:00:20.000Z"),
            _anthro_entry(system_prompt_hash="h2", input_tokens=3,
                           cache_read_tokens=800, cache_creation_tokens=100, total_message_chars=450,
                           timestamp="2025-03-09T10:00:30.000Z"),
            _anthro_entry(system_prompt_hash="h2", input_tokens=3,
                           cache_read_tokens=1200, cache_creation_tokens=200, total_message_chars=900,
                           timestamp="2025-03-09T10:00:40.000Z"),
            _anthro_entry(system_prompt_hash="h2", input_tokens=3,
                           cache_read_tokens=900, cache_creation_tokens=100, total_message_chars=600,
                           timestamp="2025-03-09T10:00:50.000Z"),
        ]
        result = detect_cache_write_churn(entries)
        assert result.triggered is False
        assert result.estimated_waste_usd == 0.0

    def test_ttl_cadence_job_no_waste_advisory_present(self) -> None:
        """A 20-min cadence job with cr=0 each call rewrites after TTL expiry, not live churn."""
        entries = [
            _anthro_entry(system_prompt_hash="h3", input_tokens=3,
                           cache_read_tokens=0, cache_creation_tokens=5000, total_message_chars=500,
                           timestamp="2025-03-09T10:00:00.000Z"),
            _anthro_entry(system_prompt_hash="h3", input_tokens=3,
                           cache_read_tokens=0, cache_creation_tokens=5000, total_message_chars=500,
                           timestamp="2025-03-09T10:20:00.000Z"),
            _anthro_entry(system_prompt_hash="h3", input_tokens=3,
                           cache_read_tokens=0, cache_creation_tokens=5000, total_message_chars=500,
                           timestamp="2025-03-09T10:40:00.000Z"),
        ]
        result = detect_cache_write_churn(entries)
        assert result.triggered is False
        assert result.estimated_waste_usd == 0.0
        assert result.details["ttl_rewrite_tokens"] == 10000
        assert result.details["ttl_rewrite_usd"] > 0
        assert "5-min cache expired" in result.description
        assert "Not counted as waste" in result.description


class TestClassifyCacheWritesLimitation:
    """REVIEW_FIXES.md finding 3: message-size/effective-prompt thread matching
    is an inference, not conversation identity. Two independent, healthy
    conversations sharing one namespace can still be mis-threaded together,
    producing an inferred churn estimate although neither real conversation
    wasted any cache tokens. This is a documented, accepted limitation of the
    approved inference — the fixture below pins the CURRENT inferred number
    so a future change to the inference is visible, not to assert it is zero.
    """

    def test_two_independent_conversations_false_positive_is_documented(self) -> None:
        """Four calls, same namespace, two real independent conversations A/B:
        A1 t=0 msg=500 cr=0 cc=1000; B1 t=10 msg=700 cr=0 cc=1600;
        A2 t=20 msg=600 cr=1000 cc=200; B2 t=30 msg=800 cr=1600 cc=100.

        Neither real conversation (A1->A2, B1->B2) ever rewrote its own
        cache: A2 reads exactly what A1 wrote, B2 reads exactly what B1
        wrote. The detector cannot see conversation identity — it only sees
        growing message length and growing effective prompt length — so it
        currently threads B1 under a candidate later claimed by A2/B2's msg
        ordering and reports 1000 live-churn tokens. This is a known false
        positive, not proof of real waste.
        """
        entries = [
            _anthro_entry(system_prompt_hash="limA", input_tokens=3,
                           cache_read_tokens=0, cache_creation_tokens=1000, total_message_chars=500,
                           timestamp="2025-03-09T10:00:00.000Z"),  # A1
            _anthro_entry(system_prompt_hash="limA", input_tokens=3,
                           cache_read_tokens=0, cache_creation_tokens=1600, total_message_chars=700,
                           timestamp="2025-03-09T10:00:10.000Z"),  # B1
            _anthro_entry(system_prompt_hash="limA", input_tokens=3,
                           cache_read_tokens=1000, cache_creation_tokens=200, total_message_chars=600,
                           timestamp="2025-03-09T10:00:20.000Z"),  # A2 (reads exactly A1's write)
            _anthro_entry(system_prompt_hash="limA", input_tokens=3,
                           cache_read_tokens=1600, cache_creation_tokens=100, total_message_chars=800,
                           timestamp="2025-03-09T10:00:30.000Z"),  # B2 (reads exactly B1's write)
        ]
        result = detect_cache_write_churn(entries)

        # Current inferred result: pinned, not a target to tune toward or away from.
        assert result.details["live_churn_tokens"] == 1000
        assert result.triggered is True
        assert result.estimated_waste_usd > 0

        # The visible dollar estimate must carry the uncertainty qualification
        # right next to it — not just a bare number.
        assert "Possible live-cache rewrites inferred from message-size matching" in result.description
        assert "Independent conversations can cause false positives" in result.description
        assert "does not confirm avoidable savings" in result.description
        assert f"${result.estimated_waste_usd:.4f}" in result.description


class TestClassifyCacheWritesLiveChurn:
    def test_live_rewrite_exact_token_split(self) -> None:
        """gap 60s, previous L=100k, now cr=40k cc=70k → 60k live churn, 10k incremental."""
        entries = [
            _anthro_entry(system_prompt_hash="h4", input_tokens=3,
                           cache_read_tokens=0, cache_creation_tokens=100000, total_message_chars=1000,
                           timestamp="2025-03-09T10:00:00.000Z"),
            _anthro_entry(system_prompt_hash="h4", input_tokens=3,
                           cache_read_tokens=40000, cache_creation_tokens=70000, total_message_chars=1100,
                           timestamp="2025-03-09T10:01:00.000Z"),
            _anthro_entry(system_prompt_hash="h4", input_tokens=3,
                           cache_read_tokens=110000, cache_creation_tokens=1000, total_message_chars=1300,
                           timestamp="2025-03-09T10:01:30.000Z"),
        ]
        result = detect_cache_write_churn(entries)
        assert result.details["live_churn_tokens"] == 60000
        assert result.details["incremental_tokens"] == 11000  # 10k (call2) + 1k (call3)
        prices = get_cache_prices(_MODEL, "anthropic")
        delta = (prices["cache_write"] - prices["cache_read"]) / 1000.0
        assert result.estimated_waste_usd == pytest.approx(round(60000 * delta, 4))
        assert result.triggered is True

    def test_mixed_models_same_hash_separate_namespaces(self) -> None:
        """Same system_prompt_hash, two models → priced separately, two flagged namespaces."""
        entries = [
            # Model A: claude-sonnet-4-6
            _anthro_entry(model="claude-sonnet-4-6", system_prompt_hash="shared",
                           input_tokens=3, cache_read_tokens=0, cache_creation_tokens=10000,
                           total_message_chars=100, timestamp="2025-03-09T10:00:00.000Z"),
            _anthro_entry(model="claude-sonnet-4-6", system_prompt_hash="shared",
                           input_tokens=3, cache_read_tokens=4000, cache_creation_tokens=7000,
                           total_message_chars=110, timestamp="2025-03-09T10:01:00.000Z"),
            _anthro_entry(model="claude-sonnet-4-6", system_prompt_hash="shared",
                           input_tokens=3, cache_read_tokens=11000, cache_creation_tokens=500,
                           total_message_chars=120, timestamp="2025-03-09T10:01:30.000Z"),
            # Model B: claude-opus-4-6, same hash → different namespace
            _anthro_entry(model="claude-opus-4-6", system_prompt_hash="shared",
                           input_tokens=3, cache_read_tokens=0, cache_creation_tokens=20000,
                           total_message_chars=200, timestamp="2025-03-09T10:00:05.000Z"),
            _anthro_entry(model="claude-opus-4-6", system_prompt_hash="shared",
                           input_tokens=3, cache_read_tokens=8000, cache_creation_tokens=14000,
                           total_message_chars=210, timestamp="2025-03-09T10:01:05.000Z"),
            _anthro_entry(model="claude-opus-4-6", system_prompt_hash="shared",
                           input_tokens=3, cache_read_tokens=22000, cache_creation_tokens=500,
                           total_message_chars=220, timestamp="2025-03-09T10:01:35.000Z"),
        ]
        result = detect_cache_write_churn(entries)
        assert result.details["live_churn_tokens"] == 6000 + 12000
        sessions = {s["model"]: s for s in result.details["flagged_sessions"]}
        assert set(sessions.keys()) == {"claude-sonnet-4-6", "claude-opus-4-6"}
        assert sessions["claude-sonnet-4-6"]["churn_tokens"] == 6000
        assert sessions["claude-opus-4-6"]["churn_tokens"] == 12000
        price_a = get_cache_prices("claude-sonnet-4-6", "anthropic")
        price_b = get_cache_prices("claude-opus-4-6", "anthropic")
        delta_a = (price_a["cache_write"] - price_a["cache_read"]) / 1000.0
        delta_b = (price_b["cache_write"] - price_b["cache_read"]) / 1000.0
        assert sessions["claude-sonnet-4-6"]["waste_usd"] == pytest.approx(round(6000 * delta_a, 4))
        assert sessions["claude-opus-4-6"]["waste_usd"] == pytest.approx(round(12000 * delta_b, 4))

    def test_one_hour_ttl_survives_read_only_call(self) -> None:
        """A read-only call between a 1h write and a 20-min-gap rewrite keeps the 1h TTL."""
        entries = [
            _anthro_entry(
                system_prompt_hash="h6", input_tokens=3,
                cache_read_tokens=0, cache_creation_tokens=50000, total_message_chars=100,
                timestamp="2025-03-09T10:00:00.000Z",
                raw_usage={"cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 50000}},
            ),
            _anthro_entry(
                system_prompt_hash="h6", input_tokens=3,
                cache_read_tokens=50000, cache_creation_tokens=0, total_message_chars=150,
                timestamp="2025-03-09T10:10:00.000Z",
            ),
            _anthro_entry(
                system_prompt_hash="h6", input_tokens=3,
                cache_read_tokens=10000, cache_creation_tokens=45000, total_message_chars=200,
                timestamp="2025-03-09T10:30:00.000Z",
            ),
        ]
        result = detect_cache_write_churn(entries)
        assert result.details["live_churn_tokens"] == 40000
        assert result.details["ttl_rewrite_tokens"] == 0
        assert result.triggered is True

    def test_mixed_5m_1h_write_then_rewrite_is_ambiguous(self) -> None:
        """A write split across both TTLs makes the next rewrite ambiguous, not waste."""
        entries = [
            _anthro_entry(
                system_prompt_hash="h7", input_tokens=3,
                cache_read_tokens=0, cache_creation_tokens=50000, total_message_chars=100,
                timestamp="2025-03-09T10:00:00.000Z",
                raw_usage={"cache_creation": {"ephemeral_5m_input_tokens": 30000, "ephemeral_1h_input_tokens": 20000}},
            ),
            _anthro_entry(
                system_prompt_hash="h7", input_tokens=3,
                cache_read_tokens=10000, cache_creation_tokens=42000, total_message_chars=110,
                timestamp="2025-03-09T10:01:00.000Z",
            ),
            _anthro_entry(
                system_prompt_hash="h7", input_tokens=3,
                cache_read_tokens=52000, cache_creation_tokens=1000, total_message_chars=120,
                timestamp="2025-03-09T10:01:30.000Z",
            ),
        ]
        result = detect_cache_write_churn(entries)
        assert result.details["ambiguous_tokens"] == 40000
        assert result.details["live_churn_tokens"] == 0
        assert result.details["ttl_rewrite_tokens"] == 0
        assert result.estimated_waste_usd == 0.0

    def test_gap_over_60_min_is_unclassified(self) -> None:
        """A rewrite after more than 60 minutes is neither waste nor an advisory."""
        entries = [
            _anthro_entry(
                system_prompt_hash="h8", input_tokens=3,
                cache_read_tokens=0, cache_creation_tokens=50000, total_message_chars=100,
                timestamp="2025-03-09T10:00:00.000Z",
            ),
            _anthro_entry(
                system_prompt_hash="h8", input_tokens=3,
                cache_read_tokens=10000, cache_creation_tokens=45000, total_message_chars=110,
                timestamp="2025-03-09T11:01:40.000Z",  # gap = 3700s > 3600s
            ),
            _anthro_entry(
                system_prompt_hash="h8", input_tokens=3,
                cache_read_tokens=55000, cache_creation_tokens=1000, total_message_chars=120,
                timestamp="2025-03-09T11:02:10.000Z",
            ),
        ]
        result = detect_cache_write_churn(entries)
        assert result.details["unclassified_write_tokens"] == 40000
        assert result.details["live_churn_tokens"] == 0
        assert result.estimated_waste_usd == 0.0
        assert "5-min cache expired" not in result.description


class TestClassifyCacheWritesRobustness:
    def test_missing_total_message_chars_or_bad_timestamp_not_classified(self) -> None:
        """Missing total_message_chars or an unparseable timestamp: excluded, no crash."""
        entries = [
            _anthro_entry(
                system_prompt_hash="h9", cache_creation_tokens=5000,
                total_message_chars=None,
                timestamp="2025-03-09T10:00:00.000Z",
            ),
            _anthro_entry(
                system_prompt_hash="h9", cache_creation_tokens=5000,
                timestamp="not-a-timestamp",
            ),
        ]
        rows = _classify_cache_writes(entries)
        assert rows == {}
        result = detect_cache_write_churn(entries)
        assert result.triggered is False
        assert result.details["flagged_sessions"] == []


class TestToolSchemaBloatCacheReadPrice:
    def test_cache_read_covers_schema_uses_cache_read_price(self) -> None:
        """cr >= schema_tokens → schema priced at the cache-read rate, not full input."""
        entries = [
            _anthro_entry(
                tool_count=5, tool_schema_tokens=800, tool_calls_made=0,
                cache_read_tokens=1000,
            )
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        prices = get_cache_prices(_MODEL, "anthropic")
        expected = 5 * 800 * (prices["cache_read"] / 1000.0)
        assert result.estimated_waste_usd == pytest.approx(round(expected, 4))

    def test_no_cache_read_uses_input_price(self) -> None:
        """cr == 0 → schema still priced at the full input rate."""
        entries = [
            _anthro_entry(
                tool_count=5, tool_schema_tokens=800, tool_calls_made=0,
                cache_read_tokens=0,
            )
            for _ in range(5)
        ]
        result = detect_tool_schema_bloat(entries)
        input_price = 0.003 / 1000.0  # claude-sonnet-4-6 input price
        expected = 5 * 800 * input_price
        assert result.estimated_waste_usd == pytest.approx(round(expected, 4))
