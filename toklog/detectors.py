"""Waste pattern detectors run at report time.

Design rules (enforced here, not just documented):
- estimated_waste_usd must always be <= actual spend of the flagged entries
- No magic numbers without an explanatory comment
- Detectors that can't produce a reliable waste figure use estimated_waste_usd=0
  and flag the issue as an observation only
- Coverage gaps (e.g. proxy calls lacking system_prompt_hash) are surfaced
  in detector details so users know the signal is partial
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from toklog.pricing import (
    _normalize_model_name,
    compute_cost_components,
    get_cache_prices,
    get_price,
)


@dataclass
class DetectorResult:
    name: str
    triggered: bool
    severity: str  # "high", "medium", "low"
    estimated_waste_usd: float
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


def run_all(entries: List[Dict[str, Any]]) -> List[DetectorResult]:
    """Run all detectors and return results."""
    return [
        detect_cache_miss(entries),
        detect_cache_write_churn(entries),
        detect_output_truncation(entries),
        detect_tool_schema_bloat(entries),
        detect_unbounded_context(entries),
        detect_high_spend_process(entries),
        detect_model_downgrade_opportunity(entries),
        detect_thinking_overhead(entries),
        detect_credential_sharing(entries),
        detect_cost_spike(entries),
    ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _effective_input(entry: Dict[str, Any]) -> int:
    """Total context tokens sent to the model, including cached portions.

    Provider-aware: OpenAI's input_tokens INCLUDES cache_read_tokens already,
    so we only add cache_creation_tokens. Anthropic's input_tokens EXCLUDES
    both cache fields, so we add both.
    """
    it = entry.get("input_tokens") or 0
    cr = entry.get("cache_read_tokens") or 0
    cc = entry.get("cache_creation_tokens") or 0
    if entry.get("provider") == "openai":
        # OpenAI: input_tokens already contains cache_read_tokens
        return it + cc
    # Anthropic & others: input_tokens excludes all cache tokens
    return it + cr + cc


def _entry_cost(entry: Dict[str, Any]) -> float:
    """Compute full cost for a single entry, accounting for cache tokens.

    Uses cost_usd if pre-computed, otherwise delegates to the shared
    compute_cost_components() in pricing.py.
    """
    cost = entry.get("cost_usd")
    if cost is not None:
        return float(cost)
    components = compute_cost_components(
        provider=entry.get("provider", ""),
        model=entry.get("model", ""),
        input_tokens=entry.get("input_tokens") or 0,
        output_tokens=entry.get("output_tokens") or 0,
        cache_read=entry.get("cache_read_tokens") or 0,
        cache_creation=entry.get("cache_creation_tokens") or 0,
    )
    return sum(components.values())


def _input_price_per_token(model: str) -> float:
    price = get_price(model)
    if price is None:
        return 0.0
    return price["input"] / 1000.0


def _output_price_per_token(model: str) -> float:
    price = get_price(model)
    if price is None:
        return 0.0
    return price["output"] / 1000.0


# ---------------------------------------------------------------------------
# Detector 1: Cache Miss Opportunity
# ---------------------------------------------------------------------------


def detect_cache_miss(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Fires when the same system prompt appears in ≥5 calls with no cache activity.

    Waste formula: system_prompt_tokens * cache_discount * input_price per uncached call
    where system_prompt_tokens = min(effective_input in group).

    Why min(effective_input)? The first call in a session (smallest context) is the
    closest proxy for system-prompt-only cost we have. It undercounts if the system
    prompt is large and conversation starts small, but it will never exceed actual
    spend — making it a conservative, safe lower bound.

    effective_input = input_tokens + cache_read + cache_creation (provider-aware)
    so this works correctly for both Anthropic (input excludes cache) and OpenAI
    (input includes cache_read).

    Coverage gap: only calls with system_prompt_hash are eligible.
    """
    if not entries:
        return DetectorResult(
            name="cache_miss_opportunity",
            triggered=False,
            severity="high",
            estimated_waste_usd=0.0,
            description="No data to analyze.",
            details={},
        )

    eligible = [e for e in entries if e.get("system_prompt_hash") is not None]
    coverage_pct = round(len(eligible) / len(entries) * 100, 1) if entries else 0.0

    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in eligible:
        by_hash[e["system_prompt_hash"]].append(e)

    waste = 0.0
    flagged_hashes: List[str] = []

    for h, calls in by_hash.items():
        # Require ≥5 calls with non-trivial context (≥500 effective tokens) to avoid noise
        cacheable = [c for c in calls if _effective_input(c) >= 500]
        if len(cacheable) < 5:
            continue
        no_cache = sum(
            1 for c in cacheable
            if (c.get("cache_read_tokens") or 0) == 0
            and (c.get("cache_creation_tokens") or 0) == 0
        )
        # Only flag if >90% of eligible calls have no cache activity
        if no_cache / len(cacheable) <= 0.9:
            continue

        flagged_hashes.append(h)

        # Estimate cacheable tokens as min(effective_input) in group.
        # Conservative lower bound — see docstring.
        sp_tokens = min(_effective_input(c) for c in cacheable)
        for c in cacheable:
            provider = c.get("provider", "openai")
            # Anthropic: 90% discount on cached reads
            # OpenAI: 50% discount on cached reads
            discount = 0.9 if provider == "anthropic" else 0.5
            price = _input_price_per_token(c.get("model", ""))
            waste += sp_tokens * discount * price

    triggered = len(flagged_hashes) > 0
    return DetectorResult(
        name="cache_miss_opportunity",
        triggered=triggered,
        severity="high",
        estimated_waste_usd=round(waste, 4),
        description=(
            f"System prompt repeated across calls with no caching. "
            f"Estimated savings: ${waste:.4f}. "
            f"({coverage_pct}% of calls have system_prompt_hash.)"
            if triggered
            else f"Caching looks fine or no eligible calls. "
                 f"({coverage_pct}% of calls have system_prompt_hash.)"
        ),
        details={
            "flagged_hashes": flagged_hashes,
            "flagged_coverage_pct": coverage_pct,
        },
    )


# ---------------------------------------------------------------------------
# Detector 2: Cache Write Churn
# ---------------------------------------------------------------------------


def detect_cache_write_churn(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Fires when prompt cache is repeatedly recreated instead of being read.

    Within a session (same system_prompt_hash), after the initial cache creation,
    subsequent writes are churn — paying cache_write price for tokens that should
    cost cache_read price.

    Trigger conditions:
    - Same system_prompt_hash with ≥3 calls
    - creation_ratio > 0.4 (unhealthy cache pattern) OR absolute waste > $0.50

    creation_ratio = total_cache_creation / (total_cache_creation + total_cache_read)
    A healthy session has ratio < 0.1 (first call writes, rest read).

    Waste formula: churn_tokens * (cache_write_price - cache_read_price)
    where churn_tokens = cache_creation after the first creation in each session.
    This is conservative: it's the difference between what was paid and the minimum
    that would have been paid if caching worked optimally.

    Only Anthropic calls are eligible — OpenAI prompt caching is automatic and not
    user-controllable.
    """
    if not entries:
        return DetectorResult(
            name="cache_write_churn",
            triggered=False,
            severity="high",
            estimated_waste_usd=0.0,
            description="No data to analyze.",
            details={},
        )

    eligible = [
        e for e in entries
        if e.get("system_prompt_hash") is not None
        and e.get("provider") == "anthropic"
    ]
    coverage_pct = round(len(eligible) / len(entries) * 100, 1) if entries else 0.0

    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in eligible:
        by_hash[e["system_prompt_hash"]].append(e)

    waste = 0.0
    flagged_sessions: List[Dict[str, Any]] = []

    for h, calls in by_hash.items():
        if len(calls) < 3:
            continue

        sorted_calls = sorted(calls, key=lambda c: c.get("timestamp", ""))

        total_cr = sum(c.get("cache_read_tokens") or 0 for c in sorted_calls)
        total_cc = sum(c.get("cache_creation_tokens") or 0 for c in sorted_calls)
        total_cache = total_cr + total_cc

        if total_cache == 0:
            continue

        creation_ratio = total_cc / total_cache

        # Compute churn tokens: all cache_creation after the first creation call
        churn_tokens = 0
        first_seen = False
        for c in sorted_calls:
            cc = c.get("cache_creation_tokens") or 0
            if cc > 0:
                if first_seen:
                    churn_tokens += cc
                else:
                    first_seen = True

        if churn_tokens == 0:
            continue

        model = sorted_calls[0].get("model", "")
        cache_prices = get_cache_prices(model, "anthropic")
        write_price = cache_prices["cache_write"] / 1000.0
        read_price = cache_prices["cache_read"] / 1000.0
        session_waste = churn_tokens * (write_price - read_price)

        # Flag if creation ratio is unhealthy (>40%) OR absolute waste is significant (>$0.50)
        if creation_ratio > 0.4 or session_waste > 0.50:
            waste += session_waste
            flagged_sessions.append({
                "hash": h,
                "calls": len(calls),
                "creation_ratio": round(creation_ratio, 2),
                "churn_tokens": churn_tokens,
                "waste_usd": round(session_waste, 4),
            })

    flagged_sessions.sort(key=lambda x: -x["waste_usd"])
    triggered = len(flagged_sessions) > 0
    return DetectorResult(
        name="cache_write_churn",
        triggered=triggered,
        severity="high",
        estimated_waste_usd=round(waste, 4),
        description=(
            f"{len(flagged_sessions)} session(s) recreating prompt cache instead of reading it. "
            f"Estimated waste: ${waste:.4f}. "
            f"Use consistent cache_control breakpoints to avoid re-creation. "
            f"({coverage_pct}% of calls are Anthropic with system_prompt_hash.)"
            if triggered
            else f"No cache write churn detected. "
                 f"({coverage_pct}% of calls are Anthropic with system_prompt_hash.)"
        ),
        details={
            "flagged_sessions": flagged_sessions,
            "flagged_coverage_pct": coverage_pct,
        },
    )


# ---------------------------------------------------------------------------
# Detector 3: Output Truncation
# ---------------------------------------------------------------------------


def detect_output_truncation(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Fires when output_tokens >= max_tokens_set * 0.95 — the model likely hit the ceiling.

    Why 0.95? Models occasionally produce output within a few tokens of the ceiling
    without being truncated. Using 95% catches genuine truncation while ignoring
    near-misses. We don't use 1.0 because off-by-one rounding can occur.

    Waste figure: $0. The call cost has already been paid (sunk cost). The waste is
    the *potential* retry cost and the bad user experience — not a dollar amount we
    can reliably calculate without knowing whether the user retried.

    Reports potential_retry_cost as an informational detail.

    Coverage gap: only calls with both max_tokens_set and output_tokens are eligible.
    """
    if not entries:
        return DetectorResult(
            name="output_truncation",
            triggered=False,
            severity="low",
            estimated_waste_usd=0,
            description="No data to analyze.",
            details={},
        )

    truncated = [
        e for e in entries
        if e.get("max_tokens_set") and e.get("output_tokens")
        and e["output_tokens"] >= e["max_tokens_set"] * 0.95
    ]
    triggered = len(truncated) > 0

    # Informational: how much it would cost if these calls were retried
    retry_cost = round(sum(_entry_cost(e) for e in truncated), 4) if triggered else 0

    return DetectorResult(
        name="output_truncation",
        triggered=triggered,
        severity="high" if triggered else "low",
        # No waste figure — sunk cost, retry cost unknown
        estimated_waste_usd=0,
        description=(
            f"{len(truncated)} call(s) hit the max_tokens ceiling and were likely truncated. "
            "Responses may be incomplete — consider raising max_tokens."
            if triggered
            else "No output truncation detected."
        ),
        details={
            "truncated_calls": len(truncated),
            "potential_retry_cost": retry_cost,
        },
    )


# ---------------------------------------------------------------------------
# Detector 4: Tool Schema Bloat
# ---------------------------------------------------------------------------


def detect_tool_schema_bloat(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Fires when tool schemas waste input tokens.

    Primary signal: tools loaded but never called (tool_calls_made == 0).
    This is genuine waste — you paid to send schemas the model never used.
    Waste = full schema token cost.

    Secondary signal: schema tokens > 50% of effective input AND effective input > 2000.
    Why 50%? If schemas are the majority of what you're sending, there's likely
    room to trim. Why 2000 token floor? First-turn calls naturally have tiny inputs
    (just the user message) with a large schema — the ratio looks extreme but isn't
    waste. 2000 tokens represents a conversation with real context built up.
    Waste = 50% of schema cost (rough estimate of the trimmable portion).

    Uses _effective_input (input + cache tokens) so the ratio is correct for
    Anthropic where input_tokens excludes cached portions.
    """
    if not entries:
        return DetectorResult(
            name="tool_schema_bloat",
            triggered=False,
            severity="medium",
            estimated_waste_usd=0.0,
            description="No data to analyze.",
            details={},
        )

    zero_use_entries: List[Dict[str, Any]] = []
    extreme_ratio_entries: List[Dict[str, Any]] = []
    max_ratio = 0.0

    for e in entries:
        tool_count = e.get("tool_count") or 0
        schema_tokens = e.get("tool_schema_tokens") or 0
        tool_calls_made = e.get("tool_calls_made")

        if tool_count <= 0:
            continue

        # Primary: schema sent but no tools called — unambiguous waste
        if schema_tokens > 500 and tool_calls_made == 0:
            zero_use_entries.append(e)
            continue

        # Secondary: schema dominates a substantial context (see docstring for thresholds)
        eff_input = _effective_input(e)
        if schema_tokens > 0 and eff_input > 2000:
            ratio = schema_tokens / eff_input
            max_ratio = max(max_ratio, ratio)
            if ratio > 0.5:
                extreme_ratio_entries.append(e)

    waste = 0.0
    for e in zero_use_entries:
        schema_tokens = e.get("tool_schema_tokens") or 0
        price = _input_price_per_token(e.get("model", ""))
        waste += schema_tokens * price
    for e in extreme_ratio_entries:
        schema_tokens = e.get("tool_schema_tokens") or 0
        price = _input_price_per_token(e.get("model", ""))
        # Conservative estimate: assume ~50% of schema could be trimmed
        waste += schema_tokens * price * 0.5

    triggered = len(zero_use_entries) > 0 or len(extreme_ratio_entries) > 0
    severity = "high" if len(zero_use_entries) > 0 else "medium"

    return DetectorResult(
        name="tool_schema_bloat",
        triggered=triggered,
        severity=severity,
        estimated_waste_usd=round(waste, 4),
        description=(
            f"{len(zero_use_entries)} call(s) loaded tools that were never used; "
            f"{len(extreme_ratio_entries)} call(s) have schema > 50% of input (in substantial contexts). "
            "Consider trimming unused tool definitions."
            if triggered
            else "Tool schema sizes look reasonable."
        ),
        details={
            "zero_use_calls": len(zero_use_entries),
            "extreme_ratio_calls": len(extreme_ratio_entries),
            "max_ratio": round(max_ratio, 2),
        },
    )


# ---------------------------------------------------------------------------
# Detector 5: Unbounded Context Growth
# ---------------------------------------------------------------------------


def detect_unbounded_context(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Fires when effective context grows monotonically within a session.

    Groups by system_prompt_hash: ≥3 of 4 consecutive pairs grow AND
    last > first * 1.5.

    Uses _effective_input (input + cache tokens) so growth is visible even when
    Anthropic's input_tokens stays small (cached delta) while cache_read grows.

    Why these thresholds?
    - ≥3 of 4 pairs: tolerates one non-growing step (e.g. a short tool result)
    - 1.5x overall: filters out stable conversations with minor variance

    Waste formula: (last - first) * input_price * 0.5
    The 0.5 is a rough estimate of how much of the growth could have been
    avoided with compaction.

    Coverage gap: only system_prompt_hash calls are eligible.
    """
    if not entries:
        return DetectorResult(
            name="unbounded_context_growth",
            triggered=False,
            severity="high",
            estimated_waste_usd=0.0,
            description="No data to analyze.",
            details={},
        )

    eligible = [e for e in entries if e.get("system_prompt_hash") is not None]
    coverage_pct = round(len(eligible) / len(entries) * 100, 1) if entries else 0.0

    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in eligible:
        by_hash[e["system_prompt_hash"]].append(e)

    waste = 0.0
    flagged_hashes: List[str] = []

    for h, calls in by_hash.items():
        sorted_calls = sorted(calls, key=lambda c: c.get("timestamp", ""))
        if len(sorted_calls) < 5:
            continue

        context_sizes = [_effective_input(c) for c in sorted_calls]

        flagged = False
        for i in range(len(context_sizes) - 4):
            window = context_sizes[i: i + 5]
            growing_pairs = sum(1 for j in range(4) if window[j + 1] > window[j])
            if growing_pairs >= 3 and window[4] > window[0] * 1.5:
                flagged = True
                break

        if flagged:
            flagged_hashes.append(h)
            first_tokens = context_sizes[0]
            last_tokens = context_sizes[-1]
            model = sorted_calls[0].get("model", "")
            price = _input_price_per_token(model)
            # Rough estimate: assume ~50% of growth could have been avoided
            waste += (last_tokens - first_tokens) * price * 0.5

    triggered = len(flagged_hashes) > 0
    return DetectorResult(
        name="unbounded_context_growth",
        triggered=triggered,
        severity="high",
        estimated_waste_usd=round(waste, 4),
        description=(
            f"Context growing without bound in {len(flagged_hashes)} session(s). "
            f"Estimated avoidable cost (rough): ${waste:.4f}. "
            f"Consider compaction or summarization. "
            f"({coverage_pct}% of calls have system_prompt_hash.)"
            if triggered
            else f"No unbounded context growth detected. "
                 f"({coverage_pct}% of calls have system_prompt_hash.)"
        ),
        details={
            "flagged_hashes": flagged_hashes,
            "flagged_coverage_pct": coverage_pct,
        },
    )


# ---------------------------------------------------------------------------
# Detector 6: High-Spend Process
# ---------------------------------------------------------------------------


def detect_high_spend_process(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Groups calls by process/call_site. Flags >50 calls or >$20 spend.

    This is an OBSERVATION detector, not a waste detector. High spend from a
    process is not inherently wasteful — it may be doing exactly what it should.
    This detector surfaces visibility: "these processes are your biggest spenders."

    No waste figure is produced. estimated_waste_usd = 0 always.

    Thresholds:
    - >50 calls: a process making >50 LLM calls is worth knowing about
    - >$20 spend: arbitrary but pragmatic — anything above $20/period is notable
    """
    if not entries:
        return DetectorResult(
            name="high_spend_process",
            triggered=False,
            severity="low",
            estimated_waste_usd=0,
            description="No data to analyze.",
            details={"flagged_sites": []},
        )

    by_process: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0})
    for e in entries:
        key = e.get("program") or e.get("tags") or "<unknown>"
        by_process[key]["calls"] += 1
        by_process[key]["cost_usd"] += _entry_cost(e)

    by_site: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0})
    for e in entries:
        cs = e.get("call_site")
        if cs is None:
            continue
        key = f"{cs['file']}:{cs['function']}:{cs['line']}"
        by_site[key]["calls"] += 1
        by_site[key]["cost_usd"] += _entry_cost(e)

    flagged_sites: List[Dict[str, Any]] = []
    for k, v in by_process.items():
        if v["calls"] > 50 or v["cost_usd"] > 20.0:
            flagged_sites.append({
                "call_site": k,
                "type": "process",
                "calls": v["calls"],
                "cost_usd": round(v["cost_usd"], 4),
            })
    for k, v in by_site.items():
        if v["calls"] > 50 or v["cost_usd"] > 20.0:
            flagged_sites.append({
                "call_site": k,
                "type": "call_site",
                "calls": v["calls"],
                "cost_usd": round(v["cost_usd"], 4),
            })
    flagged_sites.sort(key=lambda x: -x["cost_usd"])

    triggered = len(flagged_sites) > 0
    return DetectorResult(
        name="high_spend_process",
        triggered=triggered,
        severity="low",
        estimated_waste_usd=0,
        description=(
            f"{len(flagged_sites)} high-spend process(es) detected (>50 calls or >$20). "
            "Review whether call volume or spend is expected."
            if triggered
            else "No high-spend processes detected."
        ),
        details={"flagged_sites": flagged_sites},
    )


# ---------------------------------------------------------------------------
# Detector 7: Model Downgrade Opportunity (observation)
# ---------------------------------------------------------------------------

# Model tiers: maps model name prefixes to (tier_name, tier_rank).
# Lower rank = cheaper. Used to identify when an expensive model is used
# where a cheaper one might suffice.
_MODEL_TIERS: Dict[str, tuple[str, int]] = {
    "claude-opus": ("opus", 3),
    "claude-sonnet": ("sonnet", 2),
    "claude-haiku": ("haiku", 1),
    "gpt-4o-mini": ("gpt-4o-mini", 1),
    "gpt-4.1-nano": ("gpt-4.1-nano", 0),
    "gpt-4.1-mini": ("gpt-4.1-mini", 1),
    "gpt-4.1": ("gpt-4.1", 2),
    "gpt-4o": ("gpt-4o", 2),
    "gpt-5": ("gpt-5", 3),
    "o1": ("o1", 3),
    "o3-mini": ("o3-mini", 2),
    "o4-mini": ("o4-mini", 2),
}


def _model_tier(model: str) -> tuple[str, int] | None:
    """Return (tier_name, tier_rank) for a model, or None if unknown."""
    for prefix, tier in _MODEL_TIERS.items():
        if model.startswith(prefix):
            return tier
    return None


def _cheaper_alternative(model: str) -> str | None:
    """Return the name of a cheaper same-family alternative, or None."""
    if model.startswith("claude-opus"):
        return "claude-sonnet"
    if model.startswith("claude-sonnet"):
        return "claude-haiku"
    if model.startswith("gpt-4o") and not model.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"
    if model.startswith("gpt-4.1") and not model.startswith("gpt-4.1-mini") and not model.startswith("gpt-4.1-nano"):
        return "gpt-4.1-mini"
    if model.startswith("gpt-5"):
        return "gpt-4.1"
    return None


def detect_model_downgrade_opportunity(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Observation: shows potential savings from using cheaper models.

    Groups calls by model. For each expensive model (tier rank ≥ 2), computes
    what the cost would have been with the next cheaper alternative. Reports
    the delta as informational — not waste, since quality requirements are unknown.

    estimated_waste_usd = 0 always (observation only).
    """
    if not entries:
        return DetectorResult(
            name="model_downgrade_opportunity",
            triggered=False,
            severity="low",
            estimated_waste_usd=0,
            description="No data to analyze.",
            details={},
        )

    comparisons: List[Dict[str, Any]] = []

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_model[_normalize_model_name(e.get("model", "unknown"))].append(e)

    for model, calls in by_model.items():
        tier = _model_tier(model)
        if tier is None or tier[1] < 2:
            continue
        cheaper = _cheaper_alternative(model)
        if cheaper is None:
            continue
        cheaper_price = get_price(cheaper)
        if cheaper_price is None:
            continue

        actual_cost = sum(_entry_cost(e) for e in calls)
        # Re-compute cost with cheaper model's prices
        alt_cost = 0.0
        for e in calls:
            eff_in = _effective_input(e)
            out_t = e.get("output_tokens") or 0
            alt_cost += (
                eff_in * cheaper_price["input"] / 1000.0
                + out_t * cheaper_price["output"] / 1000.0
            )

        savings = actual_cost - alt_cost
        if savings > 0.01:
            comparisons.append({
                "model": model,
                "alternative": cheaper,
                "calls": len(calls),
                "actual_cost": round(actual_cost, 4),
                "alternative_cost": round(alt_cost, 4),
                "potential_savings": round(savings, 4),
            })

    comparisons.sort(key=lambda x: -x["potential_savings"])
    triggered = len(comparisons) > 0
    total_savings = sum(c["potential_savings"] for c in comparisons)

    return DetectorResult(
        name="model_downgrade_opportunity",
        triggered=triggered,
        severity="low",
        estimated_waste_usd=0,
        description=(
            f"If cheaper models were used for {sum(c['calls'] for c in comparisons)} call(s), "
            f"potential savings: ${total_savings:.4f}. "
            "Review whether quality requirements justify the model choice."
            if triggered
            else "No downgrade opportunities found."
        ),
        details={"comparisons": comparisons},
    )


# ---------------------------------------------------------------------------
# Detector 8: Thinking Overhead (observation)
# ---------------------------------------------------------------------------


def detect_thinking_overhead(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Observation: flags calls where thinking/reasoning tokens dominate output cost.

    Fires when thinking_tokens > 80% of output_tokens AND thinking_tokens > 500.
    This indicates the model is spending most of its output budget on reasoning.

    estimated_waste_usd = 0 (observation — thinking may be needed for accuracy).
    Reports total thinking cost for awareness.
    """
    if not entries:
        return DetectorResult(
            name="thinking_overhead",
            triggered=False,
            severity="low",
            estimated_waste_usd=0,
            description="No data to analyze.",
            details={},
        )

    heavy_thinking: List[Dict[str, Any]] = []
    total_thinking_cost = 0.0

    for e in entries:
        thinking = e.get("thinking_tokens") or 0
        output = e.get("output_tokens") or 0
        if thinking <= 500 or output <= 0:
            continue
        # thinking_tokens > 80% of output_tokens
        if thinking / output > 0.8:
            cost = thinking * _output_price_per_token(e.get("model", ""))
            total_thinking_cost += cost
            heavy_thinking.append(e)

    triggered = len(heavy_thinking) > 0
    total_thinking_tokens = sum(e.get("thinking_tokens") or 0 for e in heavy_thinking)

    return DetectorResult(
        name="thinking_overhead",
        triggered=triggered,
        severity="low",
        estimated_waste_usd=0,
        description=(
            f"{len(heavy_thinking)} call(s) spent >80% of output on thinking/reasoning "
            f"({total_thinking_tokens:,} thinking tokens, ${total_thinking_cost:.4f}). "
            "Consider disabling extended thinking if not needed for accuracy."
            if triggered
            else "No thinking overhead detected."
        ),
        details={
            "heavy_thinking_calls": len(heavy_thinking),
            "total_thinking_tokens": total_thinking_tokens,
            "total_thinking_cost": round(total_thinking_cost, 4),
        },
    )


# ---------------------------------------------------------------------------
# Detector 9: Credential Sharing (observation)
# ---------------------------------------------------------------------------

# Hints that indicate missing or OAuth credentials — skip these since they
# don't represent a static key being shared.
_NON_STATIC_HINTS = {None, "", "(unset)"}


def detect_credential_sharing(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Flags when the same static API key is used by multiple client identities.

    Groups entries by api_key_hint (ignoring None, "(unset)", and OAuth tokens
    that lack a hint). For each static key used by more than one (program, tag)
    pair, emits a warning.

    This is an OBSERVATION detector — no waste figure is produced.
    estimated_waste_usd = 0 always.

    Why only static keys? OAuth tokens rotate and are expected to appear across
    clients. Static keys shared across programs signal a credential hygiene issue.
    """
    if not entries:
        return DetectorResult(
            name="credential_sharing",
            triggered=False,
            severity="low",
            estimated_waste_usd=0,
            description="No data to analyze.",
            details={"shared_keys": []},
        )

    # Map each api_key_hint → set of (program, tag) clients
    key_to_clients: Dict[str, set] = defaultdict(set)
    for e in entries:
        hint = e.get("api_key_hint")
        if hint in _NON_STATIC_HINTS:
            continue
        # Skip bracket-wrapped hints — those are tags, not static keys
        if hint.startswith("[") and hint.endswith("]"):
            continue
        prog = e.get("program") or "(unknown)"
        tag = e.get("tags")
        client = f"{prog} [{tag}]" if tag else prog
        key_to_clients[hint].add(client)

    shared_keys: List[Dict[str, Any]] = []
    for hint, clients in sorted(key_to_clients.items()):
        if len(clients) > 1:
            shared_keys.append({
                "key_hint": hint,
                "clients": sorted(clients),
            })

    triggered = len(shared_keys) > 0
    descriptions = []
    for sk in shared_keys:
        client_list = ", ".join(sk["clients"])
        descriptions.append(f"key {sk['key_hint']} used by {client_list}")

    return DetectorResult(
        name="credential_sharing",
        triggered=triggered,
        severity="medium" if triggered else "low",
        estimated_waste_usd=0,
        description=(
            f"Shared credential{'s' if len(shared_keys) != 1 else ''}: "
            + "; ".join(descriptions)
            if triggered
            else "No shared credentials detected."
        ),
        details={"shared_keys": shared_keys},
    )


# ---------------------------------------------------------------------------
# Detector 10: Cost Spike (anomaly detection)
# ---------------------------------------------------------------------------

# Minimum entries in a session (or globally) to compute meaningful stats.
_SPIKE_MIN_SAMPLE = 5

# Tukey fence multiplier for "far outlier" detection.
# k=3 is standard for far outliers (k=1.5 for mild).
_SPIKE_FENCE_K = 3.0

# Max spike entries to include in details (keep report readable).
_SPIKE_MAX_DETAILS = 10


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile (0–100) using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def detect_cost_spike(entries: List[Dict[str, Any]]) -> DetectorResult:
    """Flags individual requests whose cost is a far outlier within their session.

    Uses Tukey fences (Q3 + k×IQR) rather than mean/stddev because LLM cost
    distributions are heavily right-skewed. Computes baselines per
    system_prompt_hash (session). Sessions with fewer than _SPIKE_MIN_SAMPLE
    entries are folded into a global baseline.

    Waste = spike_cost - session_P75 (the excess above expected variance),
    capped at the spike's actual cost per the design rule.
    """
    if not entries:
        return DetectorResult(
            name="cost_spike",
            triggered=False,
            severity="low",
            estimated_waste_usd=0.0,
            description="No data to analyze.",
            details={"spike_count": 0, "spikes": [], "threshold_multiplier": _SPIKE_FENCE_K},
        )

    # Compute cost per entry, filter out zero-cost (unknown models).
    entry_costs: List[tuple] = []  # (index, cost, session_hash)
    for i, e in enumerate(entries):
        c = _entry_cost(e)
        if c > 0:
            h = e.get("system_prompt_hash") or None
            entry_costs.append((i, c, h))

    if len(entry_costs) < _SPIKE_MIN_SAMPLE:
        return DetectorResult(
            name="cost_spike",
            triggered=False,
            severity="low",
            estimated_waste_usd=0.0,
            description=f"Too few entries with known cost ({len(entry_costs)}) for spike detection.",
            details={"spike_count": 0, "spikes": [], "threshold_multiplier": _SPIKE_FENCE_K},
        )

    # Group costs by session hash.
    session_costs: Dict[str, List[float]] = defaultdict(list)
    for _, c, h in entry_costs:
        session_costs[h or "__global__"].append(c)

    # Build baselines: per-session if enough samples, else fold into global pool.
    global_costs: List[float] = []
    session_baselines: Dict[str, tuple] = {}  # hash → (median, q3, fence)

    for h, costs in session_costs.items():
        if len(costs) >= _SPIKE_MIN_SAMPLE:
            q1 = _percentile(costs, 25)
            q3 = _percentile(costs, 75)
            iqr = q3 - q1
            median = _percentile(costs, 50)
            fence = q3 + _SPIKE_FENCE_K * iqr
            session_baselines[h] = (median, q3, fence)
        else:
            global_costs.extend(costs)

    # Compute global baseline from entries that didn't form their own session.
    global_baseline: tuple | None = None
    if len(global_costs) >= _SPIKE_MIN_SAMPLE:
        q1 = _percentile(global_costs, 25)
        q3 = _percentile(global_costs, 75)
        iqr = q3 - q1
        median = _percentile(global_costs, 50)
        fence = q3 + _SPIKE_FENCE_K * iqr
        global_baseline = (median, q3, fence)

    # Fallback: if global pool is too small, compute from ALL entries combined.
    # This catches cases where most entries belong to sessions (and have their
    # own baselines) but a few orphan entries need something to compare against.
    all_costs = [c for _, c, _ in entry_costs]
    fallback_baseline: tuple | None = None
    if global_baseline is None and len(all_costs) >= _SPIKE_MIN_SAMPLE:
        q1 = _percentile(all_costs, 25)
        q3 = _percentile(all_costs, 75)
        iqr = q3 - q1
        median = _percentile(all_costs, 50)
        fence = q3 + _SPIKE_FENCE_K * iqr
        fallback_baseline = (median, q3, fence)

    # Scan entries for spikes.
    spikes: List[Dict[str, Any]] = []
    total_waste = 0.0

    for idx, cost, h in entry_costs:
        session_key = h or "__global__"
        baseline = session_baselines.get(session_key) or global_baseline or fallback_baseline
        if baseline is None:
            continue

        median, q3, fence = baseline
        if cost > fence and fence > 0:
            # Waste = excess above Q3 (the "expected high end"), capped at entry cost.
            excess = min(cost - q3, cost)
            multiplier = round(cost / median, 1) if median > 0 else 0.0
            total_waste += excess
            spikes.append({
                "index": idx,
                "cost_usd": round(cost, 4),
                "session_median_usd": round(median, 4),
                "session_q3_usd": round(q3, 4),
                "fence_usd": round(fence, 4),
                "excess_usd": round(excess, 4),
                "multiplier": multiplier,
                "model": entries[idx].get("model", "?"),
                "session_hash": h,
                "timestamp": entries[idx].get("timestamp", ""),
            })

    triggered = len(spikes) > 0

    # Sort by excess descending, cap details.
    spikes.sort(key=lambda s: s["excess_usd"], reverse=True)
    spike_count = len(spikes)
    top_spikes = spikes[:_SPIKE_MAX_DETAILS]

    if triggered:
        top_cost = top_spikes[0]["cost_usd"]
        top_mult = top_spikes[0]["multiplier"]
        description = (
            f"{spike_count} request{'s' if spike_count != 1 else ''} with anomalous cost detected. "
            f"Worst: ${top_cost:.2f} ({top_mult}x session median). "
            f"Estimated avoidable cost: ${total_waste:.2f}."
        )
    else:
        description = "No cost spikes detected."

    return DetectorResult(
        name="cost_spike",
        triggered=triggered,
        severity="high" if triggered else "low",
        estimated_waste_usd=round(total_waste, 4),
        description=description,
        details={
            "spike_count": spike_count,
            "spikes": top_spikes,
            "threshold_multiplier": _SPIKE_FENCE_K,
        },
    )
