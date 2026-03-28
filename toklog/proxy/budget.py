"""Daily budget enforcement for the TokLog proxy.

Tracks cumulative spend per calendar day in memory.  When spend exceeds
a configured limit, check() returns denied so the proxy can return 429
without forwarding upstream.

State is persisted to ~/.toklog/budget_state.json on every mutation so
other processes (tl proxy status, tl report) can read it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _today() -> str:
    """Current local date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def _default_state_path() -> Path:
    return Path.home() / ".toklog" / "budget_state.json"


class BudgetTracker:
    """In-memory daily spend tracker with optional hard limit."""

    def __init__(self, limit_usd: float | None = None, state_path: Path | None = None):
        self._limit_usd = limit_usd
        self._daily_spend: float = 0.0
        self._rejected_count: int = 0
        self._current_date: str = _today()
        self._state_path: Path = state_path or _default_state_path()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_reset(self) -> None:
        """Reset counters if the calendar day has changed."""
        today = _today()
        if today != self._current_date:
            self._daily_spend = 0.0
            self._rejected_count = 0
            self._current_date = today

    def _write_state(self) -> None:
        """Atomically write current state to the state file."""
        data = self.status()
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")

        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: tmp file + rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._state_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, str(self._state_path))
        except Exception:
            # Best effort — never crash the proxy over state file I/O
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check whether the next request is allowed.

        Returns:
            (True, {}) if allowed.
            (False, info_dict) if budget exceeded — info_dict has fields
            for the 429 response body.
        """
        self._maybe_reset()

        if self._limit_usd is None:
            return True, {}

        if self._daily_spend < self._limit_usd:
            return True, {}

        # Over budget — reject
        self._rejected_count += 1
        self._write_state()

        return False, {
            "type": "budget_exceeded",
            "message": (
                f"TokLog daily budget of ${self._limit_usd:.2f} exceeded "
                f"(spent: ${self._daily_spend:.2f}). "
                f"Remove budget: edit proxy.budget_usd in ~/.toklog/config.json "
                f"or restart without --budget."
            ),
            "daily_spend": round(self._daily_spend, 4),
            "budget_limit": self._limit_usd,
            "rejected_count": self._rejected_count,
        }

    def record(self, cost_usd: float) -> None:
        """Record spend from a completed request."""
        self._maybe_reset()
        self._daily_spend += cost_usd
        self._write_state()

    def status(self) -> Dict[str, Any]:
        """Current budget state as a dict."""
        self._maybe_reset()
        enforcing = self._limit_usd is not None
        remaining = (
            max(0.0, self._limit_usd - self._daily_spend)
            if self._limit_usd is not None
            else None
        )
        return {
            "limit_usd": self._limit_usd,
            "daily_spend": round(self._daily_spend, 4),
            "remaining": round(remaining, 4) if remaining is not None else None,
            "rejected_count": self._rejected_count,
            "date": self._current_date,
            "enforcing": enforcing,
        }


# ------------------------------------------------------------------
# Module-level singleton and convenience functions
# ------------------------------------------------------------------

_tracker: BudgetTracker = BudgetTracker(limit_usd=None)


def configure(limit_usd: float | None, state_path: Path | None = None) -> None:
    """Replace the active budget tracker.  Called at daemon startup."""
    global _tracker
    _tracker = BudgetTracker(limit_usd=limit_usd, state_path=state_path)
    # Write initial state so tl proxy status works immediately
    _tracker._write_state()


def check() -> Tuple[bool, Dict[str, Any]]:
    """Delegate to the active tracker."""
    return _tracker.check()


def record(cost_usd: float) -> None:
    """Delegate to the active tracker."""
    _tracker.record(cost_usd)


def status() -> Dict[str, Any]:
    """Delegate to the active tracker."""
    return _tracker.status()


def load_status_from_file(path: Path | None = None) -> Optional[Dict[str, Any]]:
    """Read budget state from the state file (for cross-process consumption).

    Returns None if the file doesn't exist or is unreadable.
    """
    state_path = path or _default_state_path()
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
