"""Tests for detect_cost_spike — anomaly detector for per-request cost outliers.

Flags requests where cost is significantly above the session average,
indicating runaway prompts, accidental model upgrades, or agent loops
that suddenly bloat context.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import toklog.pricing as pricing_mod
from toklog.detectors import detect_cost_spike, _entry_cost


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stable_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pricing_mod, "_live_cache", None)
    monkeypatch.setattr(pricing_mod, "_live_cache_loaded", True)


def _entry(
    model: str = "claude-sonnet-4-6",
    provider: str = "anthropic",
    input_tokens: int = 5000,
    output_tokens: int = 500,
    cache_read: int = 0,
    cache_creation: int = 0,
    system_prompt_hash: str = "abc123",
    timestamp: str = "2026-04-07T10:00:00Z",
    **kw: Any,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "model": model,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "system_prompt_hash": system_prompt_hash,
        "timestamp": timestamp,
        "duration_ms": 1000,
        "error": False,
    }
    d.update(kw)
    return d


def _make_session(
    n: int,
    input_tokens: int = 5000,
    output_tokens: int = 500,
    hash: str = "session1",
    **kw: Any,
) -> List[Dict[str, Any]]:
    """Create n uniform entries for a session."""
    return [
        _entry(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            system_prompt_hash=hash,
            timestamp=f"2026-04-07T10:{i:02d}:00Z",
            **kw,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------

class TestCostSpikeEmpty:
    def test_empty_entries(self):
        result = detect_cost_spike([])
        assert not result.triggered
        assert result.estimated_waste_usd == 0.0
        assert result.name == "cost_spike"

    def test_single_entry_no_spike(self):
        """A single request can't be an outlier — nothing to compare against."""
        result = detect_cost_spike([_entry()])
        assert not result.triggered

    def test_two_entries_no_spike(self):
        """Too few entries for meaningful statistics."""
        entries = _make_session(2)
        result = detect_cost_spike(entries)
        assert not result.triggered


# ---------------------------------------------------------------------------
# No spikes — uniform sessions
# ---------------------------------------------------------------------------

class TestCostSpikeUniform:
    def test_uniform_session_no_spike(self):
        """All entries identical cost → no outliers."""
        entries = _make_session(20)
        result = detect_cost_spike(entries)
        assert not result.triggered

    def test_minor_variance_no_spike(self):
        """Small cost variance should not trigger."""
        entries = []
        for i in range(20):
            # Vary input tokens slightly: 5000 ± 500
            entries.append(_entry(
                input_tokens=5000 + (i % 5) * 100,
                output_tokens=500,
                system_prompt_hash="sess1",
                timestamp=f"2026-04-07T10:{i:02d}:00Z",
            ))
        result = detect_cost_spike(entries)
        assert not result.triggered


# ---------------------------------------------------------------------------
# Clear spikes
# ---------------------------------------------------------------------------

class TestCostSpikeTriggers:
    def test_single_spike_detected(self):
        """One request 10x more expensive than the rest → spike."""
        entries = _make_session(10, input_tokens=5000, output_tokens=500)
        # Add a spike: 50x the input tokens
        entries.append(_entry(
            input_tokens=250000,
            output_tokens=5000,
            system_prompt_hash="session1",
            timestamp="2026-04-07T10:10:00Z",
        ))
        result = detect_cost_spike(entries)
        assert result.triggered
        assert result.estimated_waste_usd > 0
        assert result.details["spike_count"] >= 1

    def test_spike_across_sessions(self):
        """Spikes in different sessions are both caught."""
        entries = _make_session(10, hash="s1")
        entries.append(_entry(
            input_tokens=200000, output_tokens=5000,
            system_prompt_hash="s1",
            timestamp="2026-04-07T10:10:00Z",
        ))
        entries += _make_session(10, hash="s2")
        entries.append(_entry(
            input_tokens=200000, output_tokens=5000,
            system_prompt_hash="s2",
            timestamp="2026-04-07T10:10:00Z",
        ))
        result = detect_cost_spike(entries)
        assert result.triggered
        assert result.details["spike_count"] >= 2

    def test_multiple_spikes_same_session(self):
        """Multiple outliers in one session are each flagged."""
        entries = _make_session(15, input_tokens=5000, output_tokens=500)
        for i in range(3):
            entries.append(_entry(
                input_tokens=200000,
                output_tokens=5000,
                system_prompt_hash="session1",
                timestamp=f"2026-04-07T11:{i:02d}:00Z",
            ))
        result = detect_cost_spike(entries)
        assert result.triggered
        assert result.details["spike_count"] >= 3


# ---------------------------------------------------------------------------
# Waste estimation
# ---------------------------------------------------------------------------

class TestCostSpikeWaste:
    def test_waste_is_excess_over_median(self):
        """Waste = spike_cost - session_median, not the full spike cost."""
        entries = _make_session(10, input_tokens=5000, output_tokens=500)
        spike = _entry(
            input_tokens=200000, output_tokens=5000,
            system_prompt_hash="session1",
            timestamp="2026-04-07T10:10:00Z",
        )
        entries.append(spike)

        result = detect_cost_spike(entries)
        assert result.triggered

        # Waste should be less than the full spike cost
        spike_cost = _entry_cost(spike)
        assert result.estimated_waste_usd < spike_cost
        assert result.estimated_waste_usd > 0

    def test_waste_never_exceeds_spike_cost(self):
        """Design rule: estimated_waste_usd <= actual spend of flagged entries."""
        entries = _make_session(20, input_tokens=5000, output_tokens=500)
        spike = _entry(
            input_tokens=300000, output_tokens=10000,
            system_prompt_hash="session1",
            timestamp="2026-04-07T10:20:00Z",
        )
        entries.append(spike)

        result = detect_cost_spike(entries)
        if result.triggered:
            spike_cost = _entry_cost(spike)
            assert result.estimated_waste_usd <= spike_cost + 0.001  # float tolerance


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCostSpikeEdgeCases:
    def test_entries_without_hash_use_global_baseline(self):
        """Entries without system_prompt_hash should still be checked against global stats."""
        entries = _make_session(10, hash="s1")
        # Add a spike with no hash
        entries.append(_entry(
            input_tokens=200000, output_tokens=5000,
            system_prompt_hash=None,
            timestamp="2026-04-07T10:10:00Z",
        ))
        result = detect_cost_spike(entries)
        # Should still detect the spike using global baseline
        assert result.triggered

    def test_all_zero_cost_entries(self):
        """All entries with unknown model (zero cost) → no spikes."""
        entries = _make_session(10, hash="s1", model="totally-unknown-xyz")
        result = detect_cost_spike(entries)
        assert not result.triggered

    def test_mixed_models_in_session(self):
        """A session switching from cheap to expensive model should detect the spike."""
        entries = _make_session(10, hash="s1", model="claude-haiku-3-5-20241022",
                                input_tokens=5000, output_tokens=500)
        # Suddenly switch to opus with massive context
        entries.append(_entry(
            model="claude-opus-4-6",
            input_tokens=100000,
            output_tokens=5000,
            system_prompt_hash="s1",
            timestamp="2026-04-07T10:10:00Z",
        ))
        result = detect_cost_spike(entries)
        assert result.triggered

    def test_severity_is_high(self):
        """Cost spikes are high severity — they represent unexpected burns."""
        entries = _make_session(10)
        entries.append(_entry(
            input_tokens=300000, output_tokens=10000,
            system_prompt_hash="session1",
            timestamp="2026-04-07T10:10:00Z",
        ))
        result = detect_cost_spike(entries)
        if result.triggered:
            assert result.severity == "high"


# ---------------------------------------------------------------------------
# Details structure
# ---------------------------------------------------------------------------

class TestCostSpikeDetails:
    def test_details_contain_expected_fields(self):
        entries = _make_session(10)
        entries.append(_entry(
            input_tokens=300000, output_tokens=10000,
            system_prompt_hash="session1",
            timestamp="2026-04-07T10:10:00Z",
        ))
        result = detect_cost_spike(entries)
        assert "spike_count" in result.details
        assert "spikes" in result.details
        assert "threshold_multiplier" in result.details

    def test_spike_entry_details(self):
        """Each spike in details should have cost, session hash, and how much above median."""
        entries = _make_session(10)
        entries.append(_entry(
            input_tokens=300000, output_tokens=10000,
            system_prompt_hash="session1",
            timestamp="2026-04-07T10:10:00Z",
        ))
        result = detect_cost_spike(entries)
        if result.triggered and result.details["spikes"]:
            spike = result.details["spikes"][0]
            assert "cost_usd" in spike
            assert "session_median_usd" in spike
            assert "multiplier" in spike
            assert "model" in spike
