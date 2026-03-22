"""Model pricing table with live LiteLLM data and hardcoded fallback."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import urllib.request
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Prices in USD per 1K tokens
PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.00125, "output": 0.005},
    "gpt-4o-mini": {"input": 0.000075, "output": 0.0003},
    "gpt-4o-2024-11-20": {"input": 0.00125, "output": 0.005},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-5": {"input": 0.00125, "output": 0.01},
    "gpt-5.1": {"input": 0.00125, "output": 0.01},
    "gpt-5.2": {"input": 0.00175, "output": 0.014},
    "gpt-5.4": {"input": 0.0025, "output": 0.015},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    "o4-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic (short aliases + full dated names)
    "claude-opus-4-6": {"input": 0.005, "output": 0.025},
    "claude-opus-4-5": {"input": 0.005, "output": 0.025},
    "claude-opus": {"input": 0.005, "output": 0.025},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
    "claude-sonnet": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5": {"input": 0.0008, "output": 0.004},
    "claude-haiku-3-5": {"input": 0.0008, "output": 0.004},
    "claude-haiku": {"input": 0.0008, "output": 0.004},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    # Google
    "gemini-3.1-pro": {"input": 0.002, "output": 0.012},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    # Together / Open-weight
    "deepseek-v3.1": {"input": 0.0006, "output": 0.0017},
    "deepseek-ai/DeepSeek-V3.1": {"input": 0.0006, "output": 0.0017},
    "qwen3-235b": {"input": 0.0002, "output": 0.0006},
    "qwen3-235b-a22b": {"input": 0.0002, "output": 0.0006},
    "qwen3-235b-thinking": {"input": 0.00065, "output": 0.003},
    "qwen3.5-397b": {"input": 0.0006, "output": 0.0036},
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": {"input": 0.0002, "output": 0.0006},
    "Qwen/Qwen3-235B-A22B-Thinking-2507": {"input": 0.00065, "output": 0.003},
    "Qwen/Qwen3.5-397B-A17B": {"input": 0.0006, "output": 0.0036},
    "minimax-m2.5": {"input": 0.0003, "output": 0.0012},
    "MiniMaxAI/MiniMax-M2.5": {"input": 0.0003, "output": 0.0012},
    "glm-5": {"input": 0.001, "output": 0.0032},
    "zai-org/GLM-5": {"input": 0.001, "output": 0.0032},
}

PRICING_LAST_UPDATED = "2025-07-19"

_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _normalize_model_name(model: str) -> str:
    """Strip trailing date suffix (e.g. -2026-03-05) so API-resolved names group with request names."""
    return _DATE_SUFFIX_RE.sub("", model)

CACHE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "anthropic": {"read": 0.1, "write": 1.25},
    "openai": {"read": 0.5, "write": 1.0},
}
DEFAULT_CACHE_MULTIPLIERS: Dict[str, float] = {"read": 0.5, "write": 1.0}

_LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

_live_cache: Optional[Dict[str, Dict[str, float]]] = None
_live_cache_loaded: bool = False
_warned_models: set = set()


def _get_cache_path() -> str:
    return os.path.expanduser("~/.toklog/pricing_cache.json")


_SUPPORTED_PREFIXES = ("gpt-", "o1", "o3", "o4", "claude-", "text-embedding", "chatgpt")


def _is_supported_model(key: str) -> bool:
    """Only keep OpenAI and Anthropic models — skip provider-prefixed, fine-tuned, and image models."""
    if "/" in key:
        return False
    return any(key.startswith(p) for p in _SUPPORTED_PREFIXES)


def _normalize_litellm(raw: Dict) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for key, val in raw.items():
        if not _is_supported_model(key):
            continue
        if not isinstance(val, dict):
            continue
        input_cost = val.get("input_cost_per_token")
        output_cost = val.get("output_cost_per_token")
        if input_cost is None or output_cost is None:
            continue
        normalized_key = key
        entry: Dict[str, float] = {
            "input": input_cost * 1000,
            "output": output_cost * 1000,
        }
        cache_read = val.get("cache_read_input_token_cost")
        cache_write = val.get("cache_creation_input_token_cost")
        if cache_read is not None:
            entry["cache_read"] = cache_read * 1000
        if cache_write is not None:
            entry["cache_write"] = cache_write * 1000
        result[normalized_key] = entry
    return result


def _fetch_litellm_pricing() -> Dict[str, Dict[str, float]]:
    with urllib.request.urlopen(_LITELLM_URL, timeout=15) as resp:
        raw = json.loads(resp.read().decode())
    return _normalize_litellm(raw)


def _save_cache(data: Dict[str, Dict[str, float]]) -> None:
    cache_path = _get_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": time.time(), "models": data}, f)


def _load_cache() -> Tuple[Optional[Dict[str, Dict[str, float]]], bool]:
    cache_path = _get_cache_path()
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("models")
        fetched_at = payload.get("fetched_at", 0)
        is_fresh = (time.time() - fetched_at) < 86400  # 24 hours
        return data, is_fresh
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None, False


def _ensure_live_cache() -> None:
    global _live_cache, _live_cache_loaded
    if _live_cache_loaded:
        return
    _live_cache_loaded = True

    cached_data, is_fresh = _load_cache()
    if is_fresh and cached_data is not None:
        _live_cache = cached_data
        return

    try:
        fetched = _fetch_litellm_pricing()
        _save_cache(fetched)
        _live_cache = fetched
        return
    except Exception:
        pass

    if cached_data is not None:
        _live_cache = cached_data
        return

    _live_cache = None


def _lookup(model: str, table: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    if model in table:
        return table[model]
    best_match: Optional[str] = None
    for known in table:
        if model.startswith(known) and (best_match is None or len(known) > len(best_match)):
            best_match = known
    return table[best_match] if best_match is not None else None


def refresh_pricing() -> int:
    """Force-fetch latest pricing from LiteLLM. Returns model count, or -1 on failure."""
    global _live_cache, _live_cache_loaded, _warned_models
    try:
        data = _fetch_litellm_pricing()
        _save_cache(data)
        _live_cache = data
        _live_cache_loaded = True
        _warned_models = set()
        return len(data)
    except Exception:
        return -1


def get_price(model: str) -> Optional[Dict[str, float]]:
    """Return pricing for a model. Falls back to prefix match, then returns zeros with warning."""
    _ensure_live_cache()
    if _live_cache is not None:
        result = _lookup(model, _live_cache)
        if result is not None:
            return result
    result = _lookup(model, PRICING)
    if result is not None:
        return result
    if model not in _warned_models:
        _warned_models.add(model)
        print(f"toklog: unknown model '{model}', cost will be $0", file=sys.stderr)
    return {"input": 0, "output": 0}


def get_cache_prices(model: str, provider: str) -> Dict[str, float]:
    """Return cache read and write prices per 1K tokens for a model/provider pair."""
    _ensure_live_cache()
    if _live_cache is not None:
        live_entry = _lookup(model, _live_cache)
        if live_entry is not None and "cache_read" in live_entry and "cache_write" in live_entry:
            return {
                "cache_read": live_entry["cache_read"],
                "cache_write": live_entry["cache_write"],
            }
    base = get_price(model)
    input_price = base["input"] if base is not None else 0.0
    multipliers = CACHE_MULTIPLIERS.get(provider, DEFAULT_CACHE_MULTIPLIERS)
    return {
        "cache_read": input_price * multipliers["read"],
        "cache_write": input_price * multipliers["write"],
    }


# Providers where input_tokens INCLUDES cache_read tokens in the API response.
# OpenAI: prompt_tokens includes cached_tokens
# Gemini: promptTokenCount includes cachedContentTokenCount
_CACHE_INCLUDED_IN_INPUT: frozenset[str] = frozenset({"openai", "gemini"})


def compute_cost_components(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
) -> Dict[str, float]:
    """Compute cost breakdown for a single API call.

    Returns dict with keys: input, cache_read, cache_write, output.
    Returns empty dict when the model has no known pricing.

    This is the single source of truth for cost computation — both
    report.py and detectors.py delegate here.
    """
    price = get_price(model)
    if price is None:
        return {}

    cache_prices = get_cache_prices(model, provider)

    if provider in _CACHE_INCLUDED_IN_INPUT:
        # input_tokens includes cache_read; subtract to avoid double-billing
        non_cached_input = input_tokens - cache_read
        if non_cached_input < 0:
            logger.warning(
                "cache_read_tokens (%d) > input_tokens (%d) for %s/%s — data inconsistency",
                cache_read, input_tokens, provider, model,
            )
            non_cached_input = 0
        return {
            "input": non_cached_input * price["input"] / 1000.0,
            "cache_read": cache_read * cache_prices["cache_read"] / 1000.0,
            "cache_write": 0.0,
            "output": output_tokens * price["output"] / 1000.0,
        }

    if provider not in ("anthropic",):
        logger.debug(
            "Unknown provider %r — assuming cache tokens are excluded from input (Anthropic convention)",
            provider,
        )

    # Anthropic & others: input_tokens EXCLUDES cache tokens
    return {
        "input": input_tokens * price["input"] / 1000.0,
        "cache_read": cache_read * cache_prices["cache_read"] / 1000.0,
        "cache_write": cache_creation * cache_prices["cache_write"] / 1000.0,
        "output": output_tokens * price["output"] / 1000.0,
    }
