"""Context driver classification: identify what structural content drives token cost."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

# Priority-ordered category labels
CONTEXT_DRIVER_LABELS = [
    "minimal",
    "cache_read",
    "tool_outputs",
    "thinking_input",
    "system_prompt",
    "tool_schemas",
    "code",
    "structured_data",
    "prose",
    # Output sub-categories (decomposed from output_tokens)
    "thinking",
    "tool_calls",
    "output_code",
    "output_text",
]


def classify_context_driver(entry: dict) -> str:
    """Classify an entry by its dominant context cost driver.

    Uses structural metadata only — no content is read. Falls back gracefully
    when new fields are absent (old log entries).
    """
    input_tokens: Optional[int] = entry.get("input_tokens")
    cache_read: int = entry.get("cache_read_tokens") or 0
    cache_creation: int = entry.get("cache_creation_tokens") or 0
    total_chars: Optional[int] = entry.get("total_message_chars")
    system_chars: int = entry.get("system_prompt_chars") or 0
    tool_result_chars: int = entry.get("tool_result_chars") or 0
    tool_schema_tokens: int = entry.get("tool_schema_tokens") or 0
    has_code: bool = entry.get("has_code_blocks") or False
    has_data: bool = entry.get("has_structured_data") or False

    # Effective input includes cached tokens — reflects actual context size.
    # With Anthropic prompt caching, input_tokens is only the uncached delta;
    # cache_read/cache_creation hold the bulk of the context.
    effective_input: Optional[int] = None
    if input_tokens is not None:
        effective_input = input_tokens + cache_read + cache_creation

    # minimal: small request — use whichever size signal is available
    if effective_input is not None and effective_input < 500:
        return "minimal"
    if total_chars is not None and total_chars < 2000:
        return "minimal"

    # cache_read: majority of context is served from prompt cache
    if effective_input and cache_read / effective_input > 0.5:
        return "cache_read"

    # fraction-based checks require total_chars
    if total_chars and total_chars > 0:
        if tool_result_chars / total_chars > 0.35:
            return "tool_outputs"
        if system_chars / total_chars > 0.30:
            return "system_prompt"

    # tool_schemas: large tool definitions — works even without new fields
    if tool_schema_tokens > 300:
        return "tool_schemas"

    # content-type signals
    if has_code:
        return "code"
    if has_data:
        return "structured_data"

    # has_tool_results on old entries (new fields absent) — best we can do
    if total_chars is None and entry.get("has_tool_results"):
        return "tool_outputs"

    return "prose"


def _char_shares(entry: dict) -> "tuple[float, float, float]":
    """Return (sys_share, tool_share, conv_share) of total_message_chars.

    All three are in [0, 1] and sum to 1.0. Returns (0, 0, 1) when no char data.
    """
    total_chars: int = entry.get("total_message_chars") or 0
    if total_chars <= 0:
        return (0.0, 0.0, 1.0)
    system_chars: int = entry.get("system_prompt_chars") or 0
    tool_result_chars: int = entry.get("tool_result_chars") or 0
    sys_share = min(system_chars / total_chars, 1.0)
    tool_share = min(tool_result_chars / total_chars, max(0.0, 1.0 - sys_share))
    conv_share = max(0.0, 1.0 - sys_share - tool_share)
    return (sys_share, tool_share, conv_share)


def decompose_output_drivers(entry: dict) -> Dict[str, int]:
    """Break output_tokens into thinking, tool_calls, output_code, output_text sub-categories.

    Uses best-effort estimation from new output fields. Falls back to output_text
    when sub-fields are absent (backward compatible with old log entries).
    """
    total_output = entry.get("output_tokens") or 0
    if total_output <= 0:
        return {}

    thinking = entry.get("thinking_tokens") or 0
    tool_calls = entry.get("tool_call_output_tokens") or 0

    output_code_chars = entry.get("output_code_chars") or 0
    code = int(output_code_chars / 4) if output_code_chars > 0 else 0

    # Clamp so sub-categories don't exceed total
    used = thinking + tool_calls + code
    if used > total_output:
        scale = total_output / used if used > 0 else 0
        thinking = int(thinking * scale)
        tool_calls = int(tool_calls * scale)
        code = int(code * scale)

    text = max(0, total_output - thinking - tool_calls - code)

    counts: Dict[str, int] = {}
    if thinking > 0:
        counts["thinking"] = thinking
    if tool_calls > 0:
        counts["tool_calls"] = tool_calls
    if code > 0:
        counts["output_code"] = code
    if text > 0:
        counts["output_text"] = text
    return counts


def decompose_context_drivers(
    entry: dict,
    component_costs: Dict[str, float],
) -> Dict[str, float]:
    """Decompose one entry into {driver: cost_fraction} proportional attributions.

    Uses actual per-component costs (not token counts) so the fractions reflect
    real spend, not volume. Fractions sum to approximately 1.0; zero-fraction
    drivers are omitted.

    Drivers produced:
      cache_read   — cost from prompt-cache reads
      cache_write  — cost from writing to prompt cache
      thinking     — cost from reasoning/thinking output tokens
      tool_calls   — cost from tool call JSON output tokens
      output_code  — cost from code blocks in model's visible response
      output_text  — cost from remaining prose in model's visible response
      system_prompt — share of input cost proportional to system_chars / total_chars
      tool_outputs  — share of input cost proportional to tool_result_chars / total_chars
      conversation  — remainder of input cost (user messages and history)
      unattributed  — input cost when character breakdown is unavailable (old entries)
    """
    total = sum(component_costs.values())
    if total <= 0:
        return {}

    fracs: Dict[str, float] = {}

    cache_read_cost = component_costs.get("cache_read", 0.0)
    cache_write_cost = component_costs.get("cache_write", 0.0)
    output_cost = component_costs.get("output", 0.0)
    input_cost = component_costs.get("input", 0.0)

    if cache_read_cost > 0:
        fracs["cache_read"] = cache_read_cost / total
    if cache_write_cost > 0:
        fracs["cache_write"] = cache_write_cost / total
    if output_cost > 0:
        output_parts = decompose_output_drivers(entry)
        total_output_tokens = sum(output_parts.values())
        if total_output_tokens > 0:
            for driver, tok in output_parts.items():
                fracs[driver] = (tok / total_output_tokens) * (output_cost / total)
        else:
            fracs["output_text"] = output_cost / total  # fallback

    input_frac = input_cost / total
    if input_frac > 0:
        total_chars: int = entry.get("total_message_chars") or 0
        thinking_in_chars: int = entry.get("thinking_input_chars") or 0
        total_input_chars = total_chars + thinking_in_chars
        if total_input_chars > 0:
            if thinking_in_chars > 0:
                fracs["thinking_input"] = (thinking_in_chars / total_input_chars) * input_frac
            if total_chars > 0:
                msg_frac = (total_chars / total_input_chars) * input_frac
                sys_share, tool_share, conv_share = _char_shares(entry)
                if sys_share > 0:
                    fracs["system_prompt"] = sys_share * msg_frac
                if tool_share > 0:
                    fracs["tool_outputs"] = tool_share * msg_frac
                if conv_share > 0:
                    fracs["conversation"] = conv_share * msg_frac
        else:
            fracs["unattributed"] = input_frac

    # Drop negligible floating-point noise
    return {k: v for k, v in fracs.items() if v > 1e-9}


def decompose_context_tokens(entry: dict) -> Dict[str, int]:
    """Return {driver: token_count} for one entry using actual token fields.

    Token attribution:
      cache_read   — cache_read_tokens (exact)
      cache_write  — cache_creation_tokens (exact)
      thinking     — estimated reasoning token count
      tool_calls   — estimated tool call JSON token count
      output_code  — estimated code block token count
      output_text  — remaining output tokens
      system_prompt — input_tokens × system_chars/total_chars
      tool_outputs  — input_tokens × tool_result_chars/total_chars
      conversation  — input_tokens × remainder char share
      unattributed  — input_tokens when no char breakdown available
    """
    input_tok: int = entry.get("input_tokens") or 0
    cache_read_tok: int = entry.get("cache_read_tokens") or 0
    cache_creation_tok: int = entry.get("cache_creation_tokens") or 0
    output_tok: int = entry.get("output_tokens") or 0

    counts: Dict[str, int] = {}

    if cache_read_tok > 0:
        counts["cache_read"] = cache_read_tok
    if cache_creation_tok > 0:
        counts["cache_write"] = cache_creation_tok
    if output_tok > 0:
        output_parts = decompose_output_drivers(entry)
        if output_parts:
            counts.update(output_parts)
        else:
            counts["output_text"] = output_tok  # fallback

    if input_tok > 0:
        total_chars: int = entry.get("total_message_chars") or 0
        thinking_in_chars: int = entry.get("thinking_input_chars") or 0
        total_input_chars = total_chars + thinking_in_chars
        if total_input_chars > 0:
            if thinking_in_chars > 0:
                counts["thinking_input"] = round((thinking_in_chars / total_input_chars) * input_tok)
            if total_chars > 0:
                msg_tok = round((total_chars / total_input_chars) * input_tok)
                if msg_tok > 0:
                    sys_share, tool_share, conv_share = _char_shares(entry)
                    if sys_share > 0:
                        counts["system_prompt"] = round(sys_share * msg_tok)
                    if tool_share > 0:
                        counts["tool_outputs"] = round(tool_share * msg_tok)
                    if conv_share > 0:
                        counts["conversation"] = round(conv_share * msg_tok)
        else:
            counts["unattributed"] = input_tok

    return counts


def aggregate_context_drivers(
    entries: List[Dict[str, Any]],
    entry_costs: List[float],
    entry_tokens: Optional[List[int]] = None,
    entry_component_costs: Optional[List[Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Aggregate context driver costs across all entries.

    When entry_component_costs is provided, uses fractional cost decomposition
    (composition mode): each entry's cost is split proportionally across all
    contributing drivers. Calls and tokens are attributed to the dominant driver.

    When entry_component_costs is absent, falls back to single-label classification
    via classify_context_driver (legacy mode).

    Returns list of rows sorted by cost descending, shape:
      {name, cost_usd, calls, tokens, avg_tokens, pct}
    """
    if not entries:
        return []

    total_cost = sum(entry_costs)
    cost_totals: Dict[str, float] = defaultdict(float)
    call_totals: Dict[str, int] = defaultdict(int)
    token_totals: Dict[str, int] = defaultdict(int)

    for i, entry in enumerate(entries):
        cost = entry_costs[i]
        tokens = entry_tokens[i] if entry_tokens is not None else 0

        if entry_component_costs is not None:
            fracs = decompose_context_drivers(entry, entry_component_costs[i])
            for driver, frac in fracs.items():
                cost_totals[driver] += cost * frac
            if fracs:
                dominant = max(fracs, key=lambda k: fracs[k])
                call_totals[dominant] += 1
                # Token attribution: actual token counts per driver, not winner-takes-all
                for driver, tok in decompose_context_tokens(entry).items():
                    token_totals[driver] += tok
        else:
            label = classify_context_driver(entry)
            cost_totals[label] += cost
            call_totals[label] += 1
            token_totals[label] += tokens

    result = [
        {
            "name": driver,
            "cost_usd": round(cost_totals[driver], 6),
            "calls": call_totals.get(driver, 0),
            "tokens": token_totals.get(driver, 0),
            "avg_tokens": (
                token_totals.get(driver, 0) // call_totals[driver]
                if call_totals.get(driver, 0) > 0
                else 0
            ),
            "pct": round(cost_totals[driver] / total_cost * 100, 1) if total_cost > 0 else 0.0,
        }
        for driver in cost_totals
    ]
    result.sort(key=lambda x: -x["cost_usd"])
    return result
