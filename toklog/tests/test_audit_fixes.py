"""Tests for adapter audit fixes.

Covers:
1. Gemini cache cost computation — promptTokenCount INCLUDES cachedContentTokenCount
2. Anthropic total_tokens computation — should be derived, not always None
3. Cost computation parity between report._compute_cost and detectors._entry_cost
4. Adapter apply_to_entry field consistency across all three providers
5. Edge cases: None tokens, zero tokens, missing provider, unknown provider
6. Shared compute_cost_components in pricing.py
7. Warning log on cache_read > input_tokens inconsistency
"""

from __future__ import annotations

import logging

import pytest
from toklog.adapters.openai import UsageResult as OpenAIUsageResult
from toklog.adapters.anthropic import (
    AnthropicUsageResult,
    AnthropicEventHandler,
    extract_from_response as anthropic_extract,
)
from toklog.adapters.gemini import (
    GeminiUsageResult,
    GeminiEventHandler,
    extract_from_response as gemini_extract,
)
from toklog.report import _compute_cost, _compute_cost_components
from toklog.detectors import _entry_cost
from toklog.pricing import compute_cost_components, get_price, get_cache_prices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    *,
    provider: str,
    model: str,
    input_tokens: int | None = 500,
    output_tokens: int | None = 100,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> dict:
    entry: dict = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }
    return entry


# ===========================================================================
# 1. Gemini cache cost: cache_read MUST be subtracted from input
# ===========================================================================

class TestGeminiCacheCostDeduction:
    """Gemini's promptTokenCount INCLUDES cachedContentTokenCount.
    Cost functions must subtract cache_read from input before pricing."""

    def test_report_compute_cost_subtracts_cache_from_input(self) -> None:
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=1000, output_tokens=100,
            cache_read_tokens=400,
        )
        price = get_price("gemini-2.5-flash")
        assert price is not None
        cache_prices = get_cache_prices("gemini-2.5-flash", "gemini")

        expected = (
            (1000 - 400) * price["input"] / 1000.0
            + 400 * cache_prices["cache_read"] / 1000.0
            + 100 * price["output"] / 1000.0
        )
        assert abs(_compute_cost(entry) - expected) < 1e-10

    def test_detectors_entry_cost_subtracts_cache_from_input(self) -> None:
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=1000, output_tokens=100,
            cache_read_tokens=400,
        )
        price = get_price("gemini-2.5-flash")
        assert price is not None
        cache_prices = get_cache_prices("gemini-2.5-flash", "gemini")

        expected = (
            (1000 - 400) * price["input"] / 1000.0
            + 400 * cache_prices["cache_read"] / 1000.0
            + 100 * price["output"] / 1000.0
        )
        assert abs(_entry_cost(entry) - expected) < 1e-10

    def test_gemini_cache_cost_less_than_without_deduction(self) -> None:
        """Sanity: with cache deduction, cost must be strictly less than
        pricing all input_tokens at full rate + cache at cache rate."""
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=1000, output_tokens=100,
            cache_read_tokens=400,
        )
        price = get_price("gemini-2.5-flash")
        assert price is not None
        cache_prices = get_cache_prices("gemini-2.5-flash", "gemini")

        # Wrong (double-billing): full input + cache_read
        wrong_cost = (
            1000 * price["input"] / 1000.0
            + 400 * cache_prices["cache_read"] / 1000.0
            + 100 * price["output"] / 1000.0
        )
        actual = _compute_cost(entry)
        assert actual < wrong_cost, f"cost {actual} should be < wrong {wrong_cost}"

    def test_gemini_no_cache_tokens_same_as_simple(self) -> None:
        """Zero cache tokens: Gemini cost == simple input*rate + output*rate."""
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=500, output_tokens=200,
            cache_read_tokens=0,
        )
        price = get_price("gemini-2.5-flash")
        assert price is not None
        expected = 500 * price["input"] / 1000.0 + 200 * price["output"] / 1000.0
        assert abs(_compute_cost(entry) - expected) < 1e-10

    def test_gemini_all_cached_input_zero_base_cost(self) -> None:
        """If ALL input is cached, non-cached input cost should be 0."""
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=1000, output_tokens=50,
            cache_read_tokens=1000,
        )
        components = _compute_cost_components(entry)
        assert abs(components["input"]) < 1e-10, "non-cached input should be $0"
        assert components["cache_read"] > 0, "cache_read cost should be >0"

    def test_gemini_cache_read_exceeds_input_no_negative(self) -> None:
        """Edge: cache_read > input_tokens (shouldn't happen, but no crash/negative)."""
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=100, output_tokens=50,
            cache_read_tokens=200,  # more than input
        )
        components = _compute_cost_components(entry)
        assert components["input"] >= 0, "input cost must never be negative"


# ===========================================================================
# 2. report._compute_cost and detectors._entry_cost parity
# ===========================================================================

class TestCostComputationParity:
    """Both functions must produce identical costs for the same entry."""

    @pytest.mark.parametrize("provider,model,cache_read,cache_creation", [
        ("openai", "gpt-4o", 400, 0),
        ("anthropic", "claude-sonnet-4-6", 200, 300),
        ("gemini", "gemini-2.5-flash", 400, 0),
        ("gemini", "gemini-2.5-pro", 500, 0),
        ("openai", "gpt-4o", 0, 0),
        ("anthropic", "claude-sonnet-4-6", 0, 0),
        ("gemini", "gemini-2.5-flash", 0, 0),
    ])
    def test_report_and_detector_agree(
        self, provider: str, model: str, cache_read: int, cache_creation: int,
    ) -> None:
        entry = _make_entry(
            provider=provider, model=model,
            input_tokens=1000, output_tokens=200,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )
        report_cost = _compute_cost(entry)
        detector_cost = _entry_cost(entry)
        assert abs(report_cost - detector_cost) < 1e-10, (
            f"{provider}/{model}: report={report_cost} != detector={detector_cost}"
        )


# ===========================================================================
# 3. Anthropic total_tokens computation
# ===========================================================================

class TestAnthropicTotalTokens:
    """Anthropic total_tokens should be computed from input+output, not None."""

    def test_total_tokens_computed_from_input_output(self) -> None:
        result = AnthropicUsageResult(
            input_tokens=500, output_tokens=100,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=0,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["total_tokens"] == 600

    def test_total_tokens_computed_with_large_values(self) -> None:
        result = AnthropicUsageResult(
            input_tokens=100_000, output_tokens=8_000,
            cache_read_tokens=50_000, cache_creation_tokens=10_000,
            thinking_tokens=2000, tool_calls_made=3,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        # total_tokens = input + output (cache tokens NOT added)
        assert entry["total_tokens"] == 108_000

    def test_total_tokens_none_when_input_missing(self) -> None:
        result = AnthropicUsageResult(
            input_tokens=None, output_tokens=100,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=None,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["total_tokens"] is None

    def test_total_tokens_none_when_output_missing(self) -> None:
        result = AnthropicUsageResult(
            input_tokens=500, output_tokens=None,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=None,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["total_tokens"] is None

    def test_total_tokens_none_when_both_missing(self) -> None:
        result = AnthropicUsageResult(
            input_tokens=None, output_tokens=None,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=None,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["total_tokens"] is None

    def test_total_tokens_zero_when_both_zero(self) -> None:
        result = AnthropicUsageResult(
            input_tokens=0, output_tokens=0,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=0,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["total_tokens"] == 0

    def test_computed_total_property_matches_apply(self) -> None:
        """The property and apply_to_entry must agree."""
        result = AnthropicUsageResult(
            input_tokens=1234, output_tokens=567,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=0,
        )
        assert result.computed_total_tokens == 1234 + 567
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["total_tokens"] == result.computed_total_tokens

    def test_streaming_handler_produces_total_tokens(self) -> None:
        """Anthropic streaming handler also produces computed total_tokens."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 200},
            },
        })
        handler.handle({
            "type": "message_delta",
            "usage": {"output_tokens": 80},
            "delta": {"stop_reason": "end_turn"},
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["total_tokens"] == 280

    def test_non_streaming_extract_produces_total_tokens(self) -> None:
        """anthropic.extract_from_response also produces computed total_tokens."""
        body = {
            "id": "msg_123",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 300, "output_tokens": 50},
        }
        result = anthropic_extract(body)
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["total_tokens"] == 350

    def test_dataclass_total_tokens_field_still_none(self) -> None:
        """The raw dataclass field total_tokens remains None (it's what the API reports).
        Only computed_total_tokens / apply_to_entry derive a value."""
        result = AnthropicUsageResult(
            input_tokens=500, output_tokens=100,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=0,
        )
        assert result.total_tokens is None  # raw field
        assert result.computed_total_tokens == 600  # property


# ===========================================================================
# 4. apply_to_entry field consistency across all three adapters
# ===========================================================================

COMMON_ENTRY_KEYS = {
    "input_tokens", "output_tokens", "total_tokens",
    "cache_read_tokens", "cache_creation_tokens",
    "thinking_tokens", "raw_usage", "usage_source", "usage_status",
}


class TestApplyToEntryConsistency:
    """All adapters must write the same core set of fields."""

    def _get_openai_entry(self) -> dict:
        result = OpenAIUsageResult(
            input_tokens=100, output_tokens=50, total_tokens=150,
            cache_read_tokens=0, cache_creation_tokens=0,
            reasoning_tokens=None, raw_usage={},
            usage_source="provider_response",
            model_served="gpt-4o", request_id="req_1",
            response_id=None, tool_calls_made=0,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        return entry

    def _get_anthropic_entry(self) -> dict:
        result = AnthropicUsageResult(
            input_tokens=100, output_tokens=50,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=0,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        return entry

    def _get_gemini_entry(self) -> dict:
        result = GeminiUsageResult(
            input_tokens=100, output_tokens=50, total_tokens=150,
            thinking_tokens=None, cache_read_tokens=0,
            tool_calls_made=0, raw_usage={},
            usage_source="provider_response",
            model_served=None, request_id=None,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        return entry

    def test_all_adapters_write_common_keys(self) -> None:
        openai_entry = self._get_openai_entry()
        anthropic_entry = self._get_anthropic_entry()
        gemini_entry = self._get_gemini_entry()

        for key in COMMON_ENTRY_KEYS:
            assert key in openai_entry, f"OpenAI missing '{key}'"
            assert key in anthropic_entry, f"Anthropic missing '{key}'"
            assert key in gemini_entry, f"Gemini missing '{key}'"

    def test_total_tokens_not_none_for_all_when_usage_present(self) -> None:
        """When input/output are provided, total_tokens should be set for ALL adapters."""
        openai_entry = self._get_openai_entry()
        anthropic_entry = self._get_anthropic_entry()
        gemini_entry = self._get_gemini_entry()

        assert openai_entry["total_tokens"] is not None, "OpenAI total_tokens is None"
        assert anthropic_entry["total_tokens"] is not None, "Anthropic total_tokens is None"
        assert gemini_entry["total_tokens"] is not None, "Gemini total_tokens is None"


# ===========================================================================
# 5. Edge cases for cost computation
# ===========================================================================

class TestCostEdgeCases:
    """Edge cases that should not crash or produce wrong results."""

    def test_none_tokens_treated_as_zero_cost(self) -> None:
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=None, output_tokens=None,
        )
        assert _compute_cost(entry) == 0.0
        assert _entry_cost(entry) == 0.0

    def test_unknown_provider_with_known_model(self) -> None:
        """Unknown provider falls into 'else' (Anthropic-like) branch."""
        entry = _make_entry(
            provider="together", model="gemini-2.5-flash",
            input_tokens=1000, output_tokens=100,
            cache_read_tokens=0,
        )
        # Should not crash — falls through to default branch
        cost = _compute_cost(entry)
        assert cost > 0

    def test_gemini_cache_write_always_zero(self) -> None:
        """Gemini cost components never have cache_write > 0."""
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=1000, output_tokens=100,
            cache_read_tokens=400,
            cache_creation_tokens=999,  # even if set, should not appear
        )
        components = _compute_cost_components(entry)
        assert components["cache_write"] == 0.0

    def test_openai_cache_write_always_zero(self) -> None:
        """OpenAI cost components never have cache_write > 0."""
        entry = _make_entry(
            provider="openai", model="gpt-4o",
            input_tokens=1000, output_tokens=100,
            cache_read_tokens=400,
            cache_creation_tokens=999,
        )
        components = _compute_cost_components(entry)
        assert components["cache_write"] == 0.0

    def test_anthropic_cache_write_nonzero_when_present(self) -> None:
        """Anthropic DOES charge for cache creation."""
        entry = _make_entry(
            provider="anthropic", model="claude-sonnet-4-6",
            input_tokens=500, output_tokens=100,
            cache_read_tokens=200,
            cache_creation_tokens=300,
        )
        components = _compute_cost_components(entry)
        assert components["cache_write"] > 0

    def test_cost_components_keys_consistent(self) -> None:
        """All three providers should produce the same component key set."""
        openai_entry = _make_entry(provider="openai", model="gpt-4o")
        anthropic_entry = _make_entry(provider="anthropic", model="claude-sonnet-4-6")
        gemini_entry = _make_entry(provider="gemini", model="gemini-2.5-flash")

        openai_keys = set(_compute_cost_components(openai_entry).keys())
        anthropic_keys = set(_compute_cost_components(anthropic_entry).keys())
        gemini_keys = set(_compute_cost_components(gemini_entry).keys())

        assert openai_keys == anthropic_keys == gemini_keys, (
            f"Mismatched cost component keys: "
            f"openai={openai_keys}, anthropic={anthropic_keys}, gemini={gemini_keys}"
        )


# ===========================================================================
# 6. Shared compute_cost_components in pricing.py
# ===========================================================================

class TestSharedComputeCostComponents:
    """The shared function in pricing.py is the single source of truth."""

    def test_direct_call_matches_report_wrapper(self) -> None:
        """pricing.compute_cost_components == report._compute_cost_components."""
        entry = _make_entry(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=1000, output_tokens=100,
            cache_read_tokens=400,
        )
        direct = compute_cost_components(
            provider="gemini", model="gemini-2.5-flash",
            input_tokens=1000, output_tokens=100,
            cache_read=400, cache_creation=0,
        )
        wrapper = _compute_cost_components(entry)
        assert direct == wrapper

    def test_direct_call_matches_detector_cost(self) -> None:
        """pricing.compute_cost_components sum == detectors._entry_cost."""
        direct = compute_cost_components(
            provider="anthropic", model="claude-sonnet-4-6",
            input_tokens=500, output_tokens=100,
            cache_read=200, cache_creation=300,
        )
        entry = _make_entry(
            provider="anthropic", model="claude-sonnet-4-6",
            input_tokens=500, output_tokens=100,
            cache_read_tokens=200, cache_creation_tokens=300,
        )
        assert abs(sum(direct.values()) - _entry_cost(entry)) < 1e-10

    def test_unknown_model_returns_empty(self) -> None:
        result = compute_cost_components(
            provider="openai", model="totally-unknown-model-xyz",
            input_tokens=1000, output_tokens=100,
            cache_read=0, cache_creation=0,
        )
        # get_price returns {"input": 0, "output": 0} for unknown, not None
        # so components should be returned with zero costs
        assert sum(result.values()) == 0.0


# ===========================================================================
# 7. Warning log on cache_read > input_tokens
# ===========================================================================

class TestCacheReadWarning:
    """When cache_read > input_tokens, a warning should be logged."""

    def test_warns_on_inconsistent_cache(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="toklog.pricing"):
            result = compute_cost_components(
                provider="gemini", model="gemini-2.5-flash",
                input_tokens=100, output_tokens=50,
                cache_read=200, cache_creation=0,
            )
        assert result["input"] == 0.0, "input cost clamped to 0"
        assert any("data inconsistency" in r.message for r in caplog.records), (
            "Expected warning about cache_read > input_tokens"
        )

    def test_no_warning_when_consistent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="toklog.pricing"):
            compute_cost_components(
                provider="gemini", model="gemini-2.5-flash",
                input_tokens=1000, output_tokens=50,
                cache_read=400, cache_creation=0,
            )
        assert not any("data inconsistency" in r.message for r in caplog.records)

    def test_no_warning_for_anthropic(self, caplog: pytest.LogCaptureFixture) -> None:
        """Anthropic doesn't include cache in input, so no subtraction happens."""
        with caplog.at_level(logging.WARNING, logger="toklog.pricing"):
            compute_cost_components(
                provider="anthropic", model="claude-sonnet-4-6",
                input_tokens=100, output_tokens=50,
                cache_read=200, cache_creation=0,
            )
        assert not any("data inconsistency" in r.message for r in caplog.records)
