"""Tests for TailSpikeTracker — real-time rolling spike detection for tl tail.

The tracker maintains per-session cost statistics and flags individual
requests whose cost exceeds the Tukey fence (Q3 + k*IQR) for their session.
"""

from __future__ import annotations

import pytest

from toklog.cli import TailSpikeTracker


class TestTrackerBasics:
    def test_empty_tracker_no_spike(self):
        t = TailSpikeTracker()
        is_spike, info = t.check(0.05, "sess1")
        assert not is_spike

    def test_first_n_entries_never_spike(self):
        """Below minimum sample size, nothing is flagged."""
        t = TailSpikeTracker(min_sample=5)
        for i in range(4):
            is_spike, _ = t.check(0.05, "sess1")
            assert not is_spike

    def test_uniform_costs_no_spike(self):
        """Identical costs → IQR=0, fence=Q3 — no outliers possible."""
        t = TailSpikeTracker(min_sample=5)
        for _ in range(20):
            is_spike, _ = t.check(0.05, "sess1")
            assert not is_spike

    def test_slight_variance_no_spike(self):
        """Small natural variance should not trigger."""
        t = TailSpikeTracker(min_sample=5)
        costs = [0.05, 0.06, 0.04, 0.055, 0.045, 0.05, 0.06, 0.04]
        for c in costs:
            is_spike, _ = t.check(c, "sess1")
        # None of these should be spikes
        results = [t.check(c, "sess1")[0] for c in costs]
        assert not any(results)


class TestTrackerSpikes:
    def test_obvious_spike_detected(self):
        """A cost 10x the norm should be flagged."""
        t = TailSpikeTracker(min_sample=5)
        for _ in range(10):
            t.check(0.05, "sess1")
        is_spike, info = t.check(0.50, "sess1")
        assert is_spike
        assert info["multiplier"] > 1.0
        assert info["median"] == pytest.approx(0.05, abs=0.01)

    def test_spike_info_contains_fields(self):
        t = TailSpikeTracker(min_sample=5)
        for _ in range(10):
            t.check(0.03, "sess1")
        is_spike, info = t.check(0.90, "sess1")
        assert is_spike
        assert "multiplier" in info
        assert "median" in info
        assert "fence" in info

    def test_spike_multiplier_accuracy(self):
        t = TailSpikeTracker(min_sample=5)
        for _ in range(10):
            t.check(0.10, "sess1")
        # Spike at 1.0 — should be 10x median
        is_spike, info = t.check(1.0, "sess1")
        assert is_spike
        assert info["multiplier"] == pytest.approx(10.0, rel=0.1)


class TestTrackerSessions:
    def test_independent_sessions(self):
        """Each session hash has its own baseline."""
        t = TailSpikeTracker(min_sample=5)
        # Session 1: cheap
        for _ in range(10):
            t.check(0.01, "cheap")
        # Session 2: expensive (normal for this session)
        for _ in range(10):
            t.check(1.00, "expensive")
        # $1.00 is NOT a spike in session "expensive"
        is_spike, _ = t.check(1.00, "expensive")
        assert not is_spike
        # But $1.00 IS a spike in session "cheap"
        is_spike, _ = t.check(1.00, "cheap")
        assert is_spike

    def test_none_hash_uses_global(self):
        """Entries with no session hash are tracked in a global pool."""
        t = TailSpikeTracker(min_sample=5)
        for _ in range(10):
            t.check(0.05, None)
        is_spike, _ = t.check(0.50, None)
        assert is_spike

    def test_empty_string_hash_uses_global(self):
        t = TailSpikeTracker(min_sample=5)
        for _ in range(10):
            t.check(0.05, "")
        is_spike, _ = t.check(0.50, "")
        assert is_spike


class TestTrackerEdgeCases:
    def test_zero_cost_entries_skipped(self):
        """Zero-cost entries (unknown models) should not corrupt the baseline."""
        t = TailSpikeTracker(min_sample=5)
        for _ in range(10):
            t.check(0.0, "sess1")  # unknown model → $0
        # Now a real cost comes in — should not be flagged as spike
        # because the baseline has no real data
        is_spike, _ = t.check(0.05, "sess1")
        assert not is_spike

    def test_spike_after_zero_cost_warmup(self):
        """Mix of zero and real costs — baseline from real costs only."""
        t = TailSpikeTracker(min_sample=5)
        for _ in range(5):
            t.check(0.0, "sess1")
        for _ in range(10):
            t.check(0.05, "sess1")
        is_spike, _ = t.check(0.50, "sess1")
        assert is_spike

    def test_custom_fence_multiplier(self):
        """Lower fence_k → more sensitive to spikes."""
        t = TailSpikeTracker(min_sample=5, fence_k=1.5)
        for _ in range(10):
            t.check(0.05, "sess1")
        # With k=1.5 (mild outlier), a 3x cost might trigger
        is_spike, _ = t.check(0.15, "sess1")
        # With uniform data IQR≈0, fence≈Q3≈0.05, so 0.15 > 0.05 → spike
        assert is_spike

    def test_custom_min_sample(self):
        """Higher min_sample delays spike detection."""
        t = TailSpikeTracker(min_sample=10)
        for _ in range(9):
            t.check(0.05, "sess1")
        # Still below min_sample — shouldn't flag
        is_spike, _ = t.check(0.50, "sess1")
        assert not is_spike
        # Now hit 10 regular entries (the spike above was added too, but
        # adding one more regular to be safe)
        t.check(0.05, "sess1")
        # NOW a spike should be detected
        is_spike, _ = t.check(0.50, "sess1")
        assert is_spike

    def test_tracker_is_stateful(self):
        """Tracker accumulates — early entries affect later detection."""
        t = TailSpikeTracker(min_sample=5)
        # Build baseline
        for _ in range(10):
            t.check(0.05, "sess1")
        # First spike
        is_spike1, _ = t.check(0.50, "sess1")
        assert is_spike1
        # Build more baseline — spike gets absorbed over time
        for _ in range(100):
            t.check(0.05, "sess1")
        # The same $0.50 should still be a spike (baseline hasn't shifted much)
        is_spike2, _ = t.check(0.50, "sess1")
        assert is_spike2


class TestTrackerSlidingWindow:
    def test_window_bounds_memory(self):
        """After window fills, oldest entries are evicted."""
        t = TailSpikeTracker(min_sample=5, window=20)
        for _ in range(50):
            t.check(0.05, "sess1")
        # Internal list should be capped at 20
        assert len(t._sessions[None if not "sess1" else "sess1"]) <= 20

    def test_window_doesnt_break_detection(self):
        """Spikes still detected after window eviction."""
        t = TailSpikeTracker(min_sample=5, window=20)
        for _ in range(50):
            t.check(0.05, "sess1")
        is_spike, _ = t.check(0.50, "sess1")
        assert is_spike

    def test_small_window(self):
        """Window smaller than min_sample still works (min_sample reached first)."""
        t = TailSpikeTracker(min_sample=5, window=10)
        for _ in range(15):
            t.check(0.05, "sess1")
        is_spike, _ = t.check(0.50, "sess1")
        assert is_spike


class TestTrackerSpikeCount:
    def test_spike_count_starts_at_zero(self):
        t = TailSpikeTracker()
        assert t.spike_count == 0

    def test_spike_count_increments(self):
        t = TailSpikeTracker(min_sample=5)
        for _ in range(10):
            t.check(0.05, "sess1")
        t.check(0.50, "sess1")
        assert t.spike_count == 1
        t.check(0.60, "sess1")
        assert t.spike_count == 2

    def test_non_spikes_dont_increment(self):
        t = TailSpikeTracker(min_sample=5)
        for _ in range(10):
            t.check(0.05, "sess1")
        assert t.spike_count == 0
