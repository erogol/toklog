"""Tests for the pricing module."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import pytest

import toklog.pricing as pricing_mod
from toklog.pricing import (
    _load_cache,
    _normalize_litellm,
    _normalize_model_name,
    _save_cache,
    get_cache_prices,
    get_price,
    refresh_pricing,
)


@pytest.fixture(autouse=True)
def _isolate_live_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent live cache from affecting tests — force fallback to hardcoded table."""
    monkeypatch.setattr(pricing_mod, "_live_cache", None)
    monkeypatch.setattr(pricing_mod, "_live_cache_loaded", True)


class TestGetPrice:
    def test_exact_match(self) -> None:
        """Exact model name returns correct pricing."""
        price = get_price("gpt-4o")
        assert price is not None
        assert price["input"] == 0.00125
        assert price["output"] == 0.005

    def test_prefix_match(self) -> None:
        """Model name with date suffix falls back to prefix match."""
        price = get_price("gpt-4o-2025-01-15")
        assert price is not None
        assert price["input"] == 0.00125
        assert price["output"] == 0.005

    def test_longest_prefix_wins(self) -> None:
        """gpt-4o-mini prefix should win over gpt-4o for mini variants."""
        price = get_price("gpt-4o-mini-2024-07-18")
        assert price is not None
        assert price["input"] == 0.000075
        assert price["output"] == 0.0003

    def test_unknown_model_returns_zeros(self) -> None:
        """Unknown model returns zero pricing."""
        price = get_price("unknown-xyz")
        assert price is not None
        assert price["input"] == 0
        assert price["output"] == 0


class TestGetCachePrices:
    def test_anthropic_multipliers(self) -> None:
        """Anthropic cache prices use 0.1x read, 1.25x write."""
        # claude-sonnet input = 0.003 per 1K
        prices = get_cache_prices("claude-sonnet-4-6", "anthropic")
        assert abs(prices["cache_read"] - 0.003 * 0.1) < 1e-10
        assert abs(prices["cache_write"] - 0.003 * 1.25) < 1e-10

    def test_openai_multipliers(self) -> None:
        """OpenAI cache prices use 0.5x read, 1.0x write."""
        # gpt-4o input = 0.00125 per 1K
        prices = get_cache_prices("gpt-4o", "openai")
        assert abs(prices["cache_read"] - 0.00125 * 0.5) < 1e-10
        assert abs(prices["cache_write"] - 0.00125 * 1.0) < 1e-10

    def test_unknown_model_returns_zeros(self) -> None:
        """Unknown model has zero base price, so cache prices are also zero."""
        prices = get_cache_prices("no-such-model", "anthropic")
        assert prices["cache_read"] == 0.0
        assert prices["cache_write"] == 0.0

    def test_unknown_provider_uses_defaults(self) -> None:
        """Unknown provider falls back to DEFAULT_CACHE_MULTIPLIERS (0.5x read, 1.0x write)."""
        prices = get_cache_prices("claude-sonnet-4-6", "unknown-provider")
        input_price = 0.003
        assert abs(prices["cache_read"] - input_price * 0.5) < 1e-10
        assert abs(prices["cache_write"] - input_price * 1.0) < 1e-10


class TestNormalizeLitellm:
    def test_per_1k_conversion(self) -> None:
        """Per-token costs are multiplied by 1000 to get per-1K."""
        raw: Dict = {"gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}}
        result = _normalize_litellm(raw)
        assert abs(result["gpt-4o"]["input"] - 0.0025) < 1e-10
        assert abs(result["gpt-4o"]["output"] - 0.01) < 1e-10

    def test_provider_prefixed_keys_skipped(self) -> None:
        """Keys with provider prefix like 'openai/gpt-4o' are skipped (not supported)."""
        raw: Dict = {"openai/gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}}
        result = _normalize_litellm(raw)
        assert "gpt-4o" not in result
        assert "openai/gpt-4o" not in result

    def test_cache_fields_included(self) -> None:
        """cache_read and cache_write are included when present in source."""
        raw: Dict = {
            "claude-sonnet": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_read_input_token_cost": 0.0000003,
                "cache_creation_input_token_cost": 0.00000375,
            }
        }
        result = _normalize_litellm(raw)
        assert "cache_read" in result["claude-sonnet"]
        assert "cache_write" in result["claude-sonnet"]
        assert abs(result["claude-sonnet"]["cache_read"] - 0.0003) < 1e-10
        assert abs(result["claude-sonnet"]["cache_write"] - 0.00375) < 1e-10

    def test_skip_missing_costs(self) -> None:
        """Entries missing input or output cost are skipped."""
        raw: Dict = {
            "model-no-cost": {"max_tokens": 1000},
            "model-no-output": {"input_cost_per_token": 0.001},
        }
        result = _normalize_litellm(raw)
        assert "model-no-cost" not in result
        assert "model-no-output" not in result


class TestCacheFile:
    def test_save_load_fresh(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Saved cache is loaded back and reported as fresh."""
        cache_file = str(tmp_path / "pricing_cache.json")
        monkeypatch.setattr(pricing_mod, "_get_cache_path", lambda: cache_file)
        data: Dict = {"gpt-4o": {"input": 0.00125, "output": 0.005}}
        _save_cache(data)
        loaded, is_fresh = _load_cache()
        assert is_fresh
        assert loaded == data

    def test_load_stale(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cache older than 24h returns data with is_fresh=False."""
        cache_file = str(tmp_path / "pricing_cache.json")
        monkeypatch.setattr(pricing_mod, "_get_cache_path", lambda: cache_file)
        old_time = time.time() - 90000  # >24h ago
        data: Dict = {"gpt-4o": {"input": 0.00125, "output": 0.005}}
        with open(cache_file, "w") as f:
            json.dump({"fetched_at": old_time, "models": data}, f)
        loaded, is_fresh = _load_cache()
        assert not is_fresh
        assert loaded == data

    def test_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing cache file returns (None, False)."""
        cache_file = str(tmp_path / "no_such_file.json")
        monkeypatch.setattr(pricing_mod, "_get_cache_path", lambda: cache_file)
        loaded, is_fresh = _load_cache()
        assert loaded is None
        assert not is_fresh

    def test_corrupt_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Corrupt JSON cache returns (None, False)."""
        cache_file = str(tmp_path / "bad_cache.json")
        monkeypatch.setattr(pricing_mod, "_get_cache_path", lambda: cache_file)
        with open(cache_file, "w") as f:
            f.write("not valid json!!")
        loaded, is_fresh = _load_cache()
        assert loaded is None
        assert not is_fresh


class TestEnsureLiveCacheStaleMiddleTier:
    def test_uses_stale_when_fetch_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When network fetch fails but stale cache exists, stale data is used."""
        stale_data: Dict = {"gpt-4o": {"input": 0.00125, "output": 0.005}}

        # Override autouse fixture to let _ensure_live_cache actually run
        monkeypatch.setattr(pricing_mod, "_live_cache_loaded", False)
        monkeypatch.setattr(pricing_mod, "_live_cache", None)
        monkeypatch.setattr(pricing_mod, "_load_cache", lambda: (stale_data, False))

        def _fail_fetch() -> Dict:
            raise OSError("network fail")

        monkeypatch.setattr(pricing_mod, "_fetch_litellm_pricing", _fail_fetch)

        pricing_mod._ensure_live_cache()
        assert pricing_mod._live_cache == stale_data


class TestGetPriceWithLiveCache:
    def test_prefers_live_over_hardcoded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Live cache price takes precedence over hardcoded table."""
        live_data: Dict = {"gpt-4o": {"input": 9.99, "output": 9.99}}
        monkeypatch.setattr(pricing_mod, "_live_cache", live_data)
        price = get_price("gpt-4o")
        assert price is not None
        assert price["input"] == 9.99

    def test_falls_back_to_hardcoded_on_miss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model missing from live cache falls back to hardcoded table."""
        monkeypatch.setattr(pricing_mod, "_live_cache", {})
        price = get_price("gpt-4o")
        assert price is not None
        assert price["input"] == 0.00125

    def test_prefix_match_in_live_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prefix matching works against live cache entries."""
        live_data: Dict = {"gpt-4o": {"input": 9.99, "output": 9.99}}
        monkeypatch.setattr(pricing_mod, "_live_cache", live_data)
        price = get_price("gpt-4o-2025-01-15")
        assert price is not None
        assert price["input"] == 9.99


class TestGetCachePricesWithLiveCache:
    def test_direct_cache_prices_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses direct cache_read/cache_write from live entry when present."""
        live_data: Dict = {
            "claude-sonnet-4-6": {
                "input": 0.003,
                "output": 0.015,
                "cache_read": 0.0003,
                "cache_write": 0.00375,
            }
        }
        monkeypatch.setattr(pricing_mod, "_live_cache", live_data)
        prices = get_cache_prices("claude-sonnet-4-6", "anthropic")
        assert abs(prices["cache_read"] - 0.0003) < 1e-10
        assert abs(prices["cache_write"] - 0.00375) < 1e-10

    def test_multiplier_fallback_when_no_direct_prices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to multipliers when live entry lacks cache_read/cache_write."""
        live_data: Dict = {"claude-sonnet-4-6": {"input": 0.003, "output": 0.015}}
        monkeypatch.setattr(pricing_mod, "_live_cache", live_data)
        prices = get_cache_prices("claude-sonnet-4-6", "anthropic")
        assert abs(prices["cache_read"] - 0.003 * 0.1) < 1e-10
        assert abs(prices["cache_write"] - 0.003 * 1.25) < 1e-10


class TestRefreshPricing:
    def test_success_returns_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """refresh_pricing returns the number of models fetched on success."""
        data: Dict = {
            "gpt-4o": {"input": 0.00125, "output": 0.005},
            "gpt-4o-mini": {"input": 0.000075, "output": 0.0003},
        }
        monkeypatch.setattr(pricing_mod, "_fetch_litellm_pricing", lambda: data)
        monkeypatch.setattr(pricing_mod, "_save_cache", lambda d: None)
        count = refresh_pricing()
        assert count == 2

    def test_failure_returns_minus_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """refresh_pricing returns -1 when fetch fails."""
        def _fail() -> Dict:
            raise OSError("network fail")

        monkeypatch.setattr(pricing_mod, "_fetch_litellm_pricing", _fail)
        count = refresh_pricing()
        assert count == -1


class TestNormalizeModelName:
    def test_strips_date_suffix(self) -> None:
        assert _normalize_model_name("gpt-5.4-2026-03-05") == "gpt-5.4"

    def test_no_suffix_unchanged(self) -> None:
        assert _normalize_model_name("gpt-5.4") == "gpt-5.4"

    def test_anthropic_yyyymmdd_unchanged(self) -> None:
        """Anthropic date format (YYYYMMDD without dashes) is not stripped."""
        assert _normalize_model_name("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"

    def test_gpt4_turbo_with_date(self) -> None:
        assert _normalize_model_name("gpt-4-turbo-2024-04-09") == "gpt-4-turbo"

    def test_unknown_model_passthrough(self) -> None:
        assert _normalize_model_name("unknown") == "unknown"


class TestPricingCLI:
    def test_refresh_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI pricing refresh command outputs model count on success."""
        from click.testing import CliRunner

        from toklog.cli import cli

        data: Dict = {"gpt-4o": {"input": 0.00125, "output": 0.005}}
        monkeypatch.setattr(pricing_mod, "_fetch_litellm_pricing", lambda: data)
        monkeypatch.setattr(pricing_mod, "_save_cache", lambda d: None)

        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "refresh"])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_refresh_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI pricing refresh command exits non-zero on fetch failure."""
        from click.testing import CliRunner

        from toklog.cli import cli

        def _fail() -> Dict:
            raise OSError("network fail")

        monkeypatch.setattr(pricing_mod, "_fetch_litellm_pricing", _fail)

        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "refresh"])
        assert result.exit_code != 0
