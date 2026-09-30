"""Tests for detect_cost_spike — anomaly detector for per-request cost outliers.

Flags requests where cost is significantly above the session average,
indicating runaway prompts, accidental model upgrades, or agent loops
that suddenly bloat context.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import toklog.pricing as pricing_mod
from toklog.detectors import detect_cost_spike, _entry_cost, _classify_cache_writes, _counted_live_churn_usd


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


# ---------------------------------------------------------------------------
# SPEC 2026-09-30: subtract counted live cache-write churn from spike excess
# ---------------------------------------------------------------------------

class TestCostSpikeChurnOverlap:
    """detect_cost_spike must not double-count dollars _counted_live_churn_usd
    already attributes to cache_write_churn. We patch _counted_live_churn_usd
    directly so each case isolates the C-rule arithmetic in detectors.py:
    excess = max(0.0, min(cost - q3, cost) - counted.get(idx, 0.0))."""

    def _spike_session(self, hash_: str) -> tuple:
        """5 uniform baseline entries + 1 clear spike entry. Returns (entries, spike_idx)."""
        entries = _make_session(5, hash=hash_)
        entries.append(_entry(
            input_tokens=250000, output_tokens=5000,
            system_prompt_hash=hash_,
            timestamp="2026-04-07T10:10:00Z",
        ))
        return entries, len(entries) - 1

    def test_spike_fully_covered_by_churn_is_dropped(self, monkeypatch):
        """When counted churn on the spike call equals its full excess, the spike
        is dropped entirely — those dollars were already reported as churn."""
        import toklog.detectors as det

        entries, idx = self._spike_session("churnA")
        baseline = detect_cost_spike(entries)
        spike = next(s for s in baseline.details["spikes"] if s["index"] == idx)
        full_excess = spike["excess_usd"]
        assert full_excess > 0

        monkeypatch.setattr(det, "_counted_live_churn_usd", lambda entries: {idx: full_excess})
        result = det.detect_cost_spike(entries)
        assert not any(s["index"] == idx for s in result.details["spikes"])

    def test_spike_partly_churn_is_reduced(self, monkeypatch):
        """When counted churn covers only part of the excess, the spike stays,
        with excess_usd reduced by exactly the counted churn dollars."""
        import toklog.detectors as det

        entries, idx = self._spike_session("churnB")
        baseline = detect_cost_spike(entries)
        spike = next(s for s in baseline.details["spikes"] if s["index"] == idx)
        full_excess = spike["excess_usd"]
        partial = full_excess / 2.0

        monkeypatch.setattr(det, "_counted_live_churn_usd", lambda entries: {idx: partial})
        result = det.detect_cost_spike(entries)
        reduced = next(s for s in result.details["spikes"] if s["index"] == idx)
        assert reduced["excess_usd"] == pytest.approx(round(full_excess - partial, 4))
        assert reduced["excess_usd"] > 0

    def test_churn_in_small_namespace_not_subtracted(self, monkeypatch):
        """_counted_live_churn_usd already excludes namespaces with < 3
        classifiable calls, so an empty churn map leaves the spike's excess
        untouched — nothing gets subtracted."""
        import toklog.detectors as det

        entries, idx = self._spike_session("churnC")
        baseline = detect_cost_spike(entries)
        spike = next(s for s in baseline.details["spikes"] if s["index"] == idx)
        full_excess = spike["excess_usd"]

        monkeypatch.setattr(det, "_counted_live_churn_usd", lambda entries: {})
        result = det.detect_cost_spike(entries)
        unchanged = next(s for s in result.details["spikes"] if s["index"] == idx)
        assert unchanged["excess_usd"] == pytest.approx(full_excess)


class TestCostSpikeChurnOverlapRealClassifier:
    """REVIEW_FIXES.md finding 4: the three tests above mock
    _counted_live_churn_usd, so they never exercise the real classifier
    wiring or the <3-call exclusion end to end. These two tests call the
    real _classify_cache_writes / _counted_live_churn_usd — no mocks — and
    check detect_cost_spike's subtraction against that real output.
    """

    def test_real_classifier_live_churn_reduces_real_spike(self) -> None:
        """A >=3-classifiable-call namespace where the spike call itself
        rewrites a just-cached prefix: the real classifier assigns it a
        nonzero live_churn_usd, and detect_cost_spike's excess_usd is
        reduced by exactly that real counted amount."""
        hash_ = "realchurnA"
        baseline = [
            _entry(
                input_tokens=3, output_tokens=50,
                cache_read_tokens=0, cache_creation_tokens=1000,
                system_prompt_hash=hash_,
                total_message_chars=500,
                timestamp=f"2026-04-07T10:{i:02d}:00Z",
            )
            for i in range(5)
        ]
        spike_entry = _entry(
            input_tokens=3, output_tokens=5000,
            cache_read_tokens=500, cache_creation_tokens=300000,
            system_prompt_hash=hash_,
            total_message_chars=600,
            timestamp="2026-04-07T10:05:00Z",  # 60s after the last baseline call
        )
        entries = baseline + [spike_entry]
        idx = len(entries) - 1

        # Real classifier: the spike's own thread had last_L=1000 (from the
        # last baseline write) and cr=500 < last_L, within the 300s live-churn
        # TTL, so miss = last_L - cr = 500 tokens counted as live churn.
        rows = _classify_cache_writes(entries)
        assert rows[idx]["live_churn_tokens"] == 500

        counted = _counted_live_churn_usd(entries)
        assert idx in counted
        real_churn_usd = counted[idx]
        assert real_churn_usd > 0

        result = detect_cost_spike(entries)
        spike = next(s for s in result.details["spikes"] if s["index"] == idx)

        # Independently derive the pre-subtraction excess from real cost and
        # the real session Q3 (all 5 baseline entries share one identical
        # cost, so Q3 == that baseline cost exactly).
        cost = _entry_cost(spike_entry)
        baseline_cost = _entry_cost(baseline[0])
        full_excess = min(cost - baseline_cost, cost)
        assert full_excess > real_churn_usd  # the spike is not fully absorbed by churn

        assert spike["excess_usd"] == pytest.approx(round(full_excess - real_churn_usd, 4))
        assert spike["excess_usd"] > 0

    def test_real_classifier_small_namespace_spike_not_subtracted(self) -> None:
        """Positive inferred churn in a two-call namespace is not subtracted."""
        baseline = [
            _entry(
                input_tokens=100, output_tokens=50,
                cache_read_tokens=0, cache_creation_tokens=0,
                system_prompt_hash="realbaselineB",
                total_message_chars=200,
                timestamp=f"2026-04-07T11:{i:02d}:00Z",
            )
            for i in range(8)
        ]
        first_write = _entry(
            input_tokens=3, output_tokens=50,
            cache_read_tokens=0, cache_creation_tokens=1000,
            system_prompt_hash="realsmallB",
            total_message_chars=500,
            timestamp="2026-04-07T11:09:00Z",
        )
        spike_entry = _entry(
            input_tokens=250000, output_tokens=5000,
            cache_read_tokens=500, cache_creation_tokens=300000,
            system_prompt_hash="realsmallB",
            total_message_chars=900,
            timestamp="2026-04-07T11:10:00Z",
        )
        entries = baseline + [first_write, spike_entry]
        idx = len(entries) - 1

        rows = _classify_cache_writes(entries)
        assert rows[idx]["live_churn_tokens"] == 500
        assert rows[idx]["live_churn_usd"] > 0
        assert sum(row["namespace"] == rows[idx]["namespace"] for row in rows.values()) == 2
        counted = _counted_live_churn_usd(entries)
        assert idx not in counted
        assert idx - 1 not in counted

        result = detect_cost_spike(entries)
        spike = next(s for s in result.details["spikes"] if s["index"] == idx)

        cost = _entry_cost(spike_entry)
        baseline_cost = _entry_cost(baseline[0])
        full_excess = min(cost - baseline_cost, cost)
        assert spike["excess_usd"] == pytest.approx(round(full_excess, 4))
