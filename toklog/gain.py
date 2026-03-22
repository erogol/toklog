"""Cumulative gain tracking — waste detected over time."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_GAIN_FILE = Path.home() / ".toklog" / "gain.json"


def load_gain() -> Dict[str, Any]:
    try:
        return json.loads(_GAIN_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_gain(data: Dict[str, Any]) -> None:
    _GAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _GAIN_FILE.write_text(json.dumps(data, indent=2, default=str))


def update_gain(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Called automatically by 'toklog report'. Records a daily snapshot.
    Idempotent: calling multiple times on the same day replaces the snapshot,
    does NOT accumulate totals multiple times.

    'toklog gain' only READS gain.json — it never calls update_gain().
    Accumulation happens via 'toklog report', not 'toklog gain'.
    """
    gain = load_gain()
    now = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if "first_seen" not in gain:
        gain["first_seen"] = now

    gain["last_updated"] = now

    # Daily snapshot — replace today's entry rather than appending.
    # This makes update_gain() idempotent within a day.
    snapshots = gain.get("snapshots", [])
    snapshots = [s for s in snapshots if s.get("date") != today]
    snapshots.append(
        {
            "date": today,
            "spend": report.get("total_spend_usd", 0.0),
            "waste": report.get("estimated_waste_usd", 0.0),
            "calls": report.get("total_calls", 0),
        }
    )
    gain["snapshots"] = snapshots[-90:]  # keep 90 days

    # Recompute totals from snapshots — never accumulate incrementally.
    # This prevents double-counting from multiple report runs.
    gain["total_calls_observed"] = sum(s["calls"] for s in gain["snapshots"])
    gain["total_spend_observed_usd"] = round(
        sum(s["spend"] for s in gain["snapshots"]), 4
    )
    gain["total_waste_detected_usd"] = round(
        sum(s["waste"] for s in gain["snapshots"]), 4
    )

    # Per-detector: take the latest report's values (not cumulative sum)
    by_det: Dict[str, float] = {}
    for det in report.get("detectors", []):
        if det["triggered"]:
            by_det[det["name"]] = round(det["estimated_waste_usd"], 4)
    gain["by_detector"] = by_det

    save_gain(gain)
    return gain
