"""Tests for context driver classification."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from toklog.context_drivers import (
    CONTEXT_DRIVER_LABELS,
    aggregate_context_drivers,
    classify_context_driver,
    decompose_context_drivers,
    decompose_context_tokens,
    decompose_output_drivers,
)


def _entry(**kwargs: Any) -> Dict[str, Any]:
    """Build a minimal entry dict with optional field overrides."""
    base: Dict[str, Any] = {
        "input_tokens": 2000,
        "tool_schema_tokens": 0,
        "has_tool_results": False,
        "system_prompt_chars": 0,
        "total_message_chars": 5000,
        "tool_result_chars": 0,
        "has_code_blocks": False,
        "has_structured_data": False,
    }
    base.update(kwargs)
    return base


class TestClassifyContextDriver:
    def test_minimal_by_input_tokens(self) -> None:
        entry = _entry(input_tokens=400, total_message_chars=None)
        assert classify_context_driver(entry) == "minimal"

    def test_minimal_by_total_chars(self) -> None:
        entry = _entry(input_tokens=None, total_message_chars=1500)
        assert classify_context_driver(entry) == "minimal"

    def test_tool_outputs_dominant(self) -> None:
        entry = _entry(total_message_chars=10000, tool_result_chars=4000)
        assert classify_context_driver(entry) == "tool_outputs"

    def test_system_prompt_dominant(self) -> None:
        entry = _entry(total_message_chars=10000, system_prompt_chars=3500)
        assert classify_context_driver(entry) == "system_prompt"

    def test_tool_schemas_dominant(self) -> None:
        entry = _entry(tool_schema_tokens=500)
        assert classify_context_driver(entry) == "tool_schemas"

    def test_code_dominant(self) -> None:
        entry = _entry(has_code_blocks=True)
        assert classify_context_driver(entry) == "code"

    def test_structured_data_dominant(self) -> None:
        entry = _entry(has_structured_data=True)
        assert classify_context_driver(entry) == "structured_data"

    def test_prose_fallback(self) -> None:
        entry = _entry()
        assert classify_context_driver(entry) == "prose"

    def test_tool_outputs_beats_code(self) -> None:
        """tool_outputs has higher priority than code."""
        entry = _entry(
            total_message_chars=10000,
            tool_result_chars=4000,
            has_code_blocks=True,
        )
        assert classify_context_driver(entry) == "tool_outputs"

    def test_system_prompt_beats_code(self) -> None:
        """system_prompt has higher priority than code."""
        entry = _entry(
            total_message_chars=10000,
            system_prompt_chars=3500,
            has_code_blocks=True,
        )
        assert classify_context_driver(entry) == "system_prompt"

    def test_old_entry_no_new_fields(self) -> None:
        """Old entries missing new fields must not crash and return a valid label."""
        old_entry: Dict[str, Any] = {
            "input_tokens": 3000,
            "tool_schema_tokens": 50,
            "has_tool_results": False,
        }
        label = classify_context_driver(old_entry)
        assert label in CONTEXT_DRIVER_LABELS

    def test_old_entry_with_tool_results(self) -> None:
        """Old entries with has_tool_results=True classify as tool_outputs."""
        old_entry: Dict[str, Any] = {
            "input_tokens": 3000,
            "has_tool_results": True,
        }
        assert classify_context_driver(old_entry) == "tool_outputs"

    def test_old_entry_minimal_by_tokens(self) -> None:
        """Old entries with few input_tokens classify as minimal."""
        old_entry: Dict[str, Any] = {"input_tokens": 200}
        assert classify_context_driver(old_entry) == "minimal"

    def test_zero_total_chars_no_division_error(self) -> None:
        """total_message_chars=0 must not raise."""
        entry = _entry(total_message_chars=0, input_tokens=5000)
        label = classify_context_driver(entry)
        assert label in CONTEXT_DRIVER_LABELS

    def test_tool_result_fraction_below_threshold(self) -> None:
        """tool_result_chars just below threshold should NOT classify as tool_outputs."""
        entry = _entry(total_message_chars=10000, tool_result_chars=3400)
        # 3400/10000 = 0.34 < 0.35
        assert classify_context_driver(entry) != "tool_outputs"

    def test_cached_entry_not_minimal(self) -> None:
        """Entry with tiny input_tokens but large cache_read is NOT minimal."""
        entry = _entry(
            input_tokens=3,
            cache_read_tokens=44698,
            cache_creation_tokens=0,
            total_message_chars=None,
        )
        assert classify_context_driver(entry) != "minimal"

    def test_effective_input_includes_cache_creation(self) -> None:
        """cache_creation_tokens count toward effective context size."""
        entry = _entry(
            input_tokens=10,
            cache_read_tokens=0,
            cache_creation_tokens=44698,
            total_message_chars=None,
        )
        assert classify_context_driver(entry) != "minimal"

    def test_minimal_when_cache_plus_input_still_small(self) -> None:
        """Entry stays minimal when effective_input (including cache) is still < 500."""
        entry = _entry(
            input_tokens=3,
            cache_read_tokens=100,
            cache_creation_tokens=0,
            total_message_chars=None,
        )
        assert classify_context_driver(entry) == "minimal"

    def test_cached_driver_when_cache_read_dominant(self) -> None:
        """cache_read > 50% of effective_input → 'cache_read'."""
        entry = _entry(
            input_tokens=3,
            cache_read_tokens=44698,
            cache_creation_tokens=0,
            total_message_chars=None,
        )
        assert classify_context_driver(entry) == "cache_read"

    def test_cached_driver_threshold_exact(self) -> None:
        """cache_read at exactly 51% of effective_input → 'cache_read'."""
        entry = _entry(
            input_tokens=490,
            cache_read_tokens=510,
            cache_creation_tokens=0,
            total_message_chars=None,
        )
        assert classify_context_driver(entry) == "cache_read"

    def test_not_cached_when_cache_read_minority(self) -> None:
        """cache_read < 50% of effective_input → not 'cache_read'."""
        entry = _entry(
            input_tokens=2000,
            cache_read_tokens=100,
            cache_creation_tokens=0,
        )
        assert classify_context_driver(entry) != "cache_read"

    def test_cached_beats_code_signal(self) -> None:
        """'cache_read' fires before content-type signals like has_code_blocks."""
        entry = _entry(
            input_tokens=3,
            cache_read_tokens=44698,
            cache_creation_tokens=0,
            total_message_chars=None,
            has_code_blocks=True,
        )
        assert classify_context_driver(entry) == "cache_read"


class TestAggregateContextDrivers:
    def test_empty_returns_empty(self) -> None:
        assert aggregate_context_drivers([], []) == []

    def test_aggregates_costs(self) -> None:
        entries = [
            _entry(has_code_blocks=True),
            _entry(has_code_blocks=True),
            _entry(has_structured_data=True),
        ]
        costs = [0.10, 0.20, 0.05]
        result = aggregate_context_drivers(entries, costs)
        by_name = {r["name"]: r for r in result}
        assert "code" in by_name
        assert by_name["code"]["calls"] == 2
        assert abs(by_name["code"]["cost_usd"] - 0.30) < 1e-9
        assert "structured_data" in by_name
        assert by_name["structured_data"]["calls"] == 1

    def test_pct_sums_to_100(self) -> None:
        entries = [_entry(has_code_blocks=True), _entry()]
        costs = [0.70, 0.30]
        result = aggregate_context_drivers(entries, costs)
        total_pct = sum(r["pct"] for r in result)
        assert abs(total_pct - 100.0) < 0.2  # rounding tolerance

    def test_sorted_by_cost_descending(self) -> None:
        entries = [
            _entry(input_tokens=200),     # minimal
            _entry(has_code_blocks=True), # code
        ]
        costs = [0.01, 0.99]
        result = aggregate_context_drivers(entries, costs)
        assert result[0]["name"] == "code"
        assert result[1]["name"] == "minimal"

    def test_tokens_aggregated(self) -> None:
        entries = [_entry(has_code_blocks=True), _entry(has_code_blocks=True)]
        costs = [0.10, 0.10]
        tokens = [1000, 2000]
        result = aggregate_context_drivers(entries, costs, entry_tokens=tokens)
        assert result[0]["tokens"] == 3000

    def test_output_shape(self) -> None:
        entries = [_entry()]
        result = aggregate_context_drivers(entries, [0.05])
        assert len(result) == 1
        row = result[0]
        assert set(row.keys()) == {"name", "cost_usd", "calls", "tokens", "avg_tokens", "pct"}


# ---------------------------------------------------------------------------
# Helpers for composition tests
# ---------------------------------------------------------------------------

def _comp_costs(
    input: float = 0.0,
    cache_read: float = 0.0,
    cache_write: float = 0.0,
    output: float = 0.0,
) -> Dict[str, float]:
    return {"input": input, "cache_read": cache_read, "cache_write": cache_write, "output": output}


def _comp_entry(**kwargs: Any) -> Dict[str, Any]:
    """Entry with char fields for composition decomposition."""
    base: Dict[str, Any] = {
        "total_message_chars": 10000,
        "system_prompt_chars": 0,
        "tool_result_chars": 0,
    }
    base.update(kwargs)
    return base


class TestDecomposeContextDrivers:
    def test_empty_component_costs_returns_empty(self) -> None:
        entry = _comp_entry()
        assert decompose_context_drivers(entry, {}) == {}

    def test_zero_total_cost_returns_empty(self) -> None:
        entry = _comp_entry()
        assert decompose_context_drivers(entry, _comp_costs()) == {}

    def test_pure_cache_read(self) -> None:
        entry = _comp_entry()
        fracs = decompose_context_drivers(entry, _comp_costs(cache_read=1.0))
        assert set(fracs.keys()) == {"cache_read"}
        assert abs(fracs["cache_read"] - 1.0) < 1e-9

    def test_pure_cache_write(self) -> None:
        entry = _comp_entry()
        fracs = decompose_context_drivers(entry, _comp_costs(cache_write=1.0))
        assert set(fracs.keys()) == {"cache_write"}
        assert abs(fracs["cache_write"] - 1.0) < 1e-9

    def test_pure_output(self) -> None:
        entry = _comp_entry()
        fracs = decompose_context_drivers(entry, _comp_costs(output=1.0))
        # No sub-fields → all output goes to output_text
        assert set(fracs.keys()) == {"output_text"}

    def test_input_splits_by_char_proportion(self) -> None:
        """system_chars=3000/10000 → 30% system_prompt of input fraction."""
        entry = _comp_entry(system_prompt_chars=3000)
        fracs = decompose_context_drivers(entry, _comp_costs(input=1.0))
        assert "system_prompt" in fracs
        assert abs(fracs["system_prompt"] - 0.30) < 1e-9
        assert "conversation" in fracs
        assert abs(fracs["conversation"] - 0.70) < 1e-9

    def test_tool_outputs_fraction(self) -> None:
        entry = _comp_entry(tool_result_chars=4000)
        fracs = decompose_context_drivers(entry, _comp_costs(input=1.0))
        assert "tool_outputs" in fracs
        assert abs(fracs["tool_outputs"] - 0.40) < 1e-9

    def test_fractions_sum_to_one(self) -> None:
        entry = _comp_entry(system_prompt_chars=2000, tool_result_chars=3000)
        costs = _comp_costs(input=0.50, cache_read=0.20, cache_write=0.10, output=0.20)
        fracs = decompose_context_drivers(entry, costs)
        assert abs(sum(fracs.values()) - 1.0) < 1e-6

    def test_no_char_data_goes_to_unattributed(self) -> None:
        entry = _comp_entry(total_message_chars=0)
        fracs = decompose_context_drivers(entry, _comp_costs(input=1.0))
        assert "unattributed" in fracs
        assert "conversation" not in fracs

    def test_system_chars_exceeds_total_no_negative(self) -> None:
        """Data bug: system_chars > total_chars — no negative fractions."""
        entry = _comp_entry(system_prompt_chars=15000, total_message_chars=10000)
        fracs = decompose_context_drivers(entry, _comp_costs(input=1.0))
        for v in fracs.values():
            assert v >= 0.0

    def test_mixed_all_drivers(self) -> None:
        entry = _comp_entry(system_prompt_chars=2000, tool_result_chars=3000)
        costs = _comp_costs(input=0.40, cache_read=0.30, cache_write=0.15, output=0.15)
        fracs = decompose_context_drivers(entry, costs)
        assert "cache_read" in fracs
        assert "cache_write" in fracs
        # output decomposes to output_text when no sub-fields
        assert "output_text" in fracs
        assert "output" not in fracs
        assert "system_prompt" in fracs
        assert "tool_outputs" in fracs
        assert "conversation" in fracs
        assert abs(sum(fracs.values()) - 1.0) < 1e-6


class TestDecomposeContextTokens:
    def test_empty_entry_returns_empty(self) -> None:
        assert decompose_context_tokens(_comp_entry()) == {}

    def test_cache_read_tokens_go_to_cached(self) -> None:
        entry = _comp_entry(cache_read_tokens=5000)
        counts = decompose_context_tokens(entry)
        assert counts.get("cache_read") == 5000

    def test_cache_creation_tokens_go_to_cache_write(self) -> None:
        entry = _comp_entry(cache_creation_tokens=2000)
        counts = decompose_context_tokens(entry)
        assert counts.get("cache_write") == 2000

    def test_output_tokens_go_to_output_text(self) -> None:
        entry = _comp_entry(output_tokens=400)
        counts = decompose_context_tokens(entry)
        # No sub-fields → all output goes to output_text
        assert counts.get("output_text") == 400
        assert "output" not in counts

    def test_input_tokens_split_by_char_shares(self) -> None:
        """system_chars=3000/10000=30% → system_prompt gets 30% of input_tokens."""
        entry = _comp_entry(input_tokens=1000, system_prompt_chars=3000)
        counts = decompose_context_tokens(entry)
        assert counts.get("system_prompt") == 300
        assert counts.get("conversation") == 700

    def test_input_with_tool_results(self) -> None:
        entry = _comp_entry(input_tokens=1000, tool_result_chars=4000)
        counts = decompose_context_tokens(entry)
        assert counts.get("tool_outputs") == 400
        assert counts.get("conversation") == 600

    def test_no_char_data_goes_to_unattributed(self) -> None:
        entry = _comp_entry(input_tokens=500, total_message_chars=0)
        counts = decompose_context_tokens(entry)
        assert counts.get("unattributed") == 500
        assert "conversation" not in counts

    def test_all_driver_types_present(self) -> None:
        entry = _comp_entry(
            input_tokens=1000,
            cache_read_tokens=5000,
            cache_creation_tokens=500,
            output_tokens=200,
            system_prompt_chars=2000,
            tool_result_chars=3000,
        )
        counts = decompose_context_tokens(entry)
        assert counts["cache_read"] == 5000
        assert counts["cache_write"] == 500
        # No sub-fields → output goes to output_text
        assert counts["output_text"] == 200
        assert "output" not in counts
        assert counts["system_prompt"] == 200   # 20% of 1000
        assert counts["tool_outputs"] == 300    # 30% of 1000
        assert counts["conversation"] == 500    # 50% of 1000

    def test_system_chars_exceeds_total_no_negative(self) -> None:
        entry = _comp_entry(input_tokens=1000, system_prompt_chars=15000, total_message_chars=10000)
        counts = decompose_context_tokens(entry)
        for v in counts.values():
            assert v >= 0


class TestAggregateContextComposition:
    """Tests for aggregate_context_drivers in composition mode (entry_component_costs provided)."""

    def test_empty_returns_empty(self) -> None:
        assert aggregate_context_drivers([], [], entry_component_costs=[]) == []

    def test_output_shape_includes_avg_tokens(self) -> None:
        entries = [_comp_entry()]
        costs = [1.0]
        comp = [_comp_costs(output=1.0)]
        result = aggregate_context_drivers(entries, costs, entry_component_costs=comp)
        assert len(result) == 1
        assert set(result[0].keys()) == {"name", "cost_usd", "calls", "tokens", "avg_tokens", "pct"}

    def test_cost_fractionally_distributed(self) -> None:
        """50% cache_read, 50% input (all conversation) → two drivers, equal cost."""
        entry = _comp_entry()
        costs = [1.0]
        comp = [_comp_costs(input=0.50, cache_read=0.50)]
        result = aggregate_context_drivers(entries=[entry], entry_costs=costs, entry_component_costs=comp)
        by_name = {r["name"]: r for r in result}
        assert "cache_read" in by_name
        assert "conversation" in by_name
        assert abs(by_name["cache_read"]["cost_usd"] - 0.50) < 1e-6
        assert abs(by_name["conversation"]["cost_usd"] - 0.50) < 1e-6

    def test_pct_sums_to_100(self) -> None:
        entries = [_comp_entry(system_prompt_chars=3000), _comp_entry()]
        costs = [0.60, 0.40]
        comp = [_comp_costs(input=0.60), _comp_costs(input=0.40)]
        result = aggregate_context_drivers(entries, costs, entry_component_costs=comp)
        total_pct = sum(r["pct"] for r in result)
        assert abs(total_pct - 100.0) < 0.5

    def test_calls_attributed_to_dominant_driver(self) -> None:
        """Entry with 90% cache_read → dominant is 'cache_read', call goes there."""
        entry = _comp_entry()
        costs = [1.0]
        comp = [_comp_costs(cache_read=0.90, output=0.10)]
        result = aggregate_context_drivers(entries=[entry], entry_costs=costs, entry_component_costs=comp)
        by_name = {r["name"]: r for r in result}
        assert by_name["cache_read"]["calls"] == 1
        assert by_name.get("output", {}).get("calls", 0) == 0

    def test_avg_tokens_computed(self) -> None:
        """avg_tokens is derived from actual token fields, not the entry_tokens parameter."""
        entries = [_comp_entry(output_tokens=3000)]
        costs = [1.0]
        comp = [_comp_costs(output=1.0)]
        result = aggregate_context_drivers(entries, costs, entry_component_costs=comp)
        # No sub-fields → output_text is dominant (100% of cost), output_tokens=3000
        assert result[0]["name"] == "output_text"
        assert result[0]["tokens"] == 3000
        assert result[0]["avg_tokens"] == 3000

    def test_zero_cost_entry_no_crash(self) -> None:
        """Entry with all-zero component costs doesn't crash or add phantom rows."""
        entries = [_comp_entry()]
        costs = [0.0]
        comp = [_comp_costs()]
        result = aggregate_context_drivers(entries, costs, entry_component_costs=comp)
        assert result == []

    def test_sorted_by_cost_descending(self) -> None:
        entries = [_comp_entry(), _comp_entry(system_prompt_chars=3000)]
        costs = [0.10, 0.90]
        comp = [_comp_costs(output=0.10), _comp_costs(input=0.90)]
        result = aggregate_context_drivers(entries, costs, entry_component_costs=comp)
        for i in range(len(result) - 1):
            assert result[i]["cost_usd"] >= result[i + 1]["cost_usd"]

    def test_tokens_accumulate_across_entries(self) -> None:
        """cache_read_tokens from two entries sum correctly under 'cached' driver."""
        entries = [
            _comp_entry(cache_read_tokens=5000),
            _comp_entry(cache_read_tokens=3000),
        ]
        costs = [0.60, 0.40]
        comp = [_comp_costs(cache_read=0.60), _comp_costs(cache_read=0.40)]
        result = aggregate_context_drivers(entries, costs, entry_component_costs=comp)
        by_name = {r["name"]: r for r in result}
        assert by_name["cache_read"]["tokens"] == 8000

    def test_tokens_independent_of_dominant_driver(self) -> None:
        """output tokens accumulate even when 'cached' is the dominant cost driver."""
        entry = _comp_entry(cache_read_tokens=10000, output_tokens=200)
        costs = [1.0]
        comp = [_comp_costs(cache_read=0.90, output=0.10)]
        result = aggregate_context_drivers(entries=[entry], entry_costs=costs, entry_component_costs=comp)
        by_name = {r["name"]: r for r in result}
        assert by_name["cache_read"]["calls"] == 1
        # No sub-fields → output goes to output_text
        assert by_name["output_text"]["tokens"] == 200

    def test_output_never_appears_in_results(self) -> None:
        """The old 'output' label must never appear; output_text is the fallback."""
        entries = [_comp_entry(output_tokens=500)]
        costs = [1.0]
        comp = [_comp_costs(output=1.0)]
        result = aggregate_context_drivers(entries, costs, entry_component_costs=comp)
        names = {r["name"] for r in result}
        assert "output" not in names
        assert "output_text" in names


class TestDecomposeOutputDrivers:
    def test_no_sub_fields_all_to_output_text(self) -> None:
        """Entry with only output_tokens, no sub-fields → all to output_text."""
        entry = {"output_tokens": 1000}
        parts = decompose_output_drivers(entry)
        assert parts == {"output_text": 1000}

    def test_zero_output_returns_empty(self) -> None:
        assert decompose_output_drivers({"output_tokens": 0}) == {}
        assert decompose_output_drivers({}) == {}

    def test_thinking_splits_correctly(self) -> None:
        """thinking_tokens=500, output_tokens=1000 → thinking=500, output_text=500."""
        entry = {"output_tokens": 1000, "thinking_tokens": 500}
        parts = decompose_output_drivers(entry)
        assert parts.get("thinking") == 500
        assert parts.get("output_text") == 500
        assert "tool_calls" not in parts
        assert "output_code" not in parts

    def test_output_code_chars_converted(self) -> None:
        """output_code_chars=400 → output_code=100 (400/4)."""
        entry = {"output_tokens": 500, "output_code_chars": 400}
        parts = decompose_output_drivers(entry)
        assert parts.get("output_code") == 100
        assert parts.get("output_text") == 400

    def test_tool_call_output_tokens(self) -> None:
        entry = {"output_tokens": 1000, "tool_call_output_tokens": 200}
        parts = decompose_output_drivers(entry)
        assert parts.get("tool_calls") == 200
        assert parts.get("output_text") == 800

    def test_all_four_categories(self) -> None:
        """All four sub-categories populated correctly."""
        entry = {
            "output_tokens": 1000,
            "thinking_tokens": 300,
            "tool_call_output_tokens": 200,
            "output_code_chars": 400,  # → 100 tokens
        }
        parts = decompose_output_drivers(entry)
        assert parts["thinking"] == 300
        assert parts["tool_calls"] == 200
        assert parts["output_code"] == 100
        assert parts["output_text"] == 400
        assert sum(parts.values()) == 1000

    def test_clamping_when_sub_categories_exceed_total(self) -> None:
        """Sub-categories exceeding total are scaled down proportionally."""
        entry = {
            "output_tokens": 100,
            "thinking_tokens": 200,
            "tool_call_output_tokens": 200,
        }
        parts = decompose_output_drivers(entry)
        assert sum(parts.values()) <= 100
        assert parts.get("thinking", 0) >= 0
        assert parts.get("tool_calls", 0) >= 0
        assert parts.get("output_text", 0) >= 0

    def test_output_code_chars_zero_omitted(self) -> None:
        entry = {"output_tokens": 500, "output_code_chars": 0}
        parts = decompose_output_drivers(entry)
        assert "output_code" not in parts
        assert parts.get("output_text") == 500


class TestDecomposeContextDriversOutputSubcategories:
    """Tests for output sub-category decomposition in decompose_context_drivers."""

    def test_output_cost_with_thinking_splits(self) -> None:
        """output_cost is split proportionally by output sub-categories."""
        entry = _comp_entry(output_tokens=1000, thinking_tokens=500)
        costs = _comp_costs(output=1.0)
        fracs = decompose_context_drivers(entry, costs)
        # thinking=500, output_text=500 → each gets 50%
        assert abs(fracs.get("thinking", 0) - 0.5) < 1e-9
        assert abs(fracs.get("output_text", 0) - 0.5) < 1e-9
        assert "output" not in fracs

    def test_backward_compat_no_sub_fields(self) -> None:
        """Old entries without new fields → output cost goes entirely to output_text."""
        entry = _comp_entry(output_tokens=500)
        costs = _comp_costs(output=1.0)
        fracs = decompose_context_drivers(entry, costs)
        assert abs(fracs.get("output_text", 0) - 1.0) < 1e-9
        assert "output" not in fracs

    def test_fractions_sum_with_output_subcategories(self) -> None:
        entry = _comp_entry(
            output_tokens=1000,
            thinking_tokens=300,
            tool_call_output_tokens=200,
            system_prompt_chars=2000,
        )
        costs = _comp_costs(input=0.40, cache_read=0.30, cache_write=0.10, output=0.20)
        fracs = decompose_context_drivers(entry, costs)
        assert abs(sum(fracs.values()) - 1.0) < 1e-6
        assert "output" not in fracs


class TestDecomposeContextTokensOutputSubcategories:
    """Tests for output sub-category decomposition in decompose_context_tokens."""

    def test_output_splits_into_subcategories(self) -> None:
        entry = _comp_entry(
            output_tokens=1000,
            thinking_tokens=400,
            tool_call_output_tokens=100,
        )
        counts = decompose_context_tokens(entry)
        assert counts.get("thinking") == 400
        assert counts.get("tool_calls") == 100
        assert counts.get("output_text") == 500
        assert "output" not in counts

    def test_backward_compat_output_to_output_text(self) -> None:
        entry = _comp_entry(output_tokens=700)
        counts = decompose_context_tokens(entry)
        assert counts.get("output_text") == 700
        assert "output" not in counts

    def test_output_code_chars_in_token_counts(self) -> None:
        entry = _comp_entry(output_tokens=500, output_code_chars=400)
        counts = decompose_context_tokens(entry)
        assert counts.get("output_code") == 100
        assert counts.get("output_text") == 400


class TestThinkingInputDriver:
    """Tests for thinking_input_chars → thinking_input driver."""

    def test_thinking_input_in_labels(self) -> None:
        assert "thinking_input" in CONTEXT_DRIVER_LABELS

    def test_decompose_context_drivers_thinking_input(self) -> None:
        """thinking_input_chars carve out a proportional share of input cost."""
        # _comp_entry default: total_message_chars=10000
        # thinking_input_chars=4000 → total_input_chars=14000
        # thinking_input fraction = 4000/14000, conversation = 10000/14000
        entry = _comp_entry(thinking_input_chars=4000)
        fracs = decompose_context_drivers(entry, _comp_costs(input=1.0))
        assert "thinking_input" in fracs
        assert abs(fracs["thinking_input"] - 4000 / 14000) < 1e-9
        assert "conversation" in fracs
        assert abs(fracs["conversation"] - 10000 / 14000) < 1e-9

    def test_decompose_context_drivers_thinking_input_fracs_sum_to_one(self) -> None:
        entry = _comp_entry(thinking_input_chars=2000, system_prompt_chars=1000)
        costs = _comp_costs(input=0.60, cache_read=0.20, output=0.20)
        fracs = decompose_context_drivers(entry, costs)
        assert abs(sum(fracs.values()) - 1.0) < 1e-6

    def test_decompose_context_drivers_no_thinking_input_unchanged(self) -> None:
        """When thinking_input_chars=0, behavior is identical to old code."""
        entry = _comp_entry(system_prompt_chars=3000)
        fracs = decompose_context_drivers(entry, _comp_costs(input=1.0))
        assert "thinking_input" not in fracs
        assert abs(fracs["system_prompt"] - 0.30) < 1e-9
        assert abs(fracs["conversation"] - 0.70) < 1e-9

    def test_decompose_context_tokens_thinking_input(self) -> None:
        """thinking_input_chars → thinking_input token count via proportion."""
        # _comp_entry default: total_message_chars=10000
        # thinking_input_chars=2000 → total=12000
        # thinking_input = round(2000/12000 * 1000) = 167
        # conversation = round(10000/12000 * 1000) = 833
        entry = _comp_entry(input_tokens=1000, thinking_input_chars=2000)
        counts = decompose_context_tokens(entry)
        assert counts.get("thinking_input") == round(2000 / 12000 * 1000)
        assert counts.get("conversation") == round(10000 / 12000 * 1000)

    def test_decompose_context_tokens_no_thinking_input_unchanged(self) -> None:
        """No thinking_input_chars → no change to existing token decomposition."""
        entry = _comp_entry(input_tokens=1000, system_prompt_chars=3000)
        counts = decompose_context_tokens(entry)
        assert "thinking_input" not in counts
        assert counts.get("system_prompt") == 300
        assert counts.get("conversation") == 700

    def test_decompose_context_tokens_only_thinking_input(self) -> None:
        """Only thinking_input_chars, no regular message chars → 100% to thinking_input."""
        entry = _comp_entry(
            input_tokens=500,
            total_message_chars=0,
            thinking_input_chars=1000,
        )
        counts = decompose_context_tokens(entry)
        assert counts.get("thinking_input") == 500
        assert "conversation" not in counts
        assert "unattributed" not in counts
