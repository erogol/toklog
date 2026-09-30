"""JSONL local logger with date-partitioned files."""

from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_LOG_DIR = os.path.join(os.path.expanduser("~"), ".toklog", "logs")
_lock = threading.Lock()
_dir_ensured = False

# Background write queue used by log_entry_async() so the proxy event loop
# is never blocked on disk I/O.  The writer thread is a daemon so it dies
# with the process without needing explicit shutdown.
_write_queue: queue.SimpleQueue[str] = queue.SimpleQueue()


def _writer() -> None:
    """Daemon thread: drain _write_queue and append to today's log file."""
    while True:
        line = _write_queue.get()
        try:
            _ensure_log_dir()
            with open(_today_file(), "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass  # never crash the background thread


_writer_thread = threading.Thread(target=_writer, daemon=True, name="toklog-log-writer")
_writer_thread.start()


def get_log_dir() -> str:
    return _LOG_DIR


def _ensure_log_dir() -> None:
    global _dir_ensured
    if _dir_ensured:
        return
    os.makedirs(_LOG_DIR, exist_ok=True)
    _dir_ensured = True


def _today_file() -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(_LOG_DIR, f"{date_str}.jsonl")


def log_entry(entry: Dict[str, Any]) -> None:
    """Append a single JSON entry to today's log file. Thread-safe, blocking."""
    try:
        _ensure_log_dir()
        line = json.dumps(entry, default=str) + "\n"
        with _lock:
            with open(_today_file(), "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        # Never break the user's app
        pass


def log_entry_async(entry: Dict[str, Any]) -> None:
    """Enqueue a JSON entry for background write. Returns immediately (non-blocking)."""
    try:
        line = json.dumps(entry, default=str) + "\n"
        _write_queue.put(line)
    except Exception:
        pass


# Request IDs that identify synthetic benchmark/test traffic — excluded from all reads.
# This is a backward-compat filter for data logged before the benchmark defaulted to
# TOKLOG_DISABLED=1. Can be removed once logs containing these IDs have aged out.
_BENCH_REQUEST_IDS: frozenset[str] = frozenset({"chatcmpl-bench"})


def _is_benchmark_entry(entry: dict) -> bool:
    """Return True for synthetic benchmark entries that carry no real cost."""
    return entry.get("request_id") in _BENCH_REQUEST_IDS


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp string into an aware UTC datetime.

    A trailing 'Z' is treated as '+00:00'. A naive value (no tz info) is
    assumed to already be UTC. Returns None when value is missing or does
    not parse — callers must keep the entry in that case, not drop it.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalize_bound(value: Optional[datetime]) -> Optional[datetime]:
    """Normalize a start_date/end_date bound to an aware UTC datetime.

    A naive bound (no tzinfo) is treated as already being in UTC. An
    aware bound is converted to UTC with astimezone. None passes through
    unchanged. Entry timestamps from _parse_ts are always aware UTC, so
    comparing them against a naive bound raises TypeError unless the
    bound is normalized first.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def read_logs(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> list:
    """Read all log entries within date range. Returns list of dicts."""
    start_date = _normalize_bound(start_date)
    end_date = _normalize_bound(end_date)
    entries: list = []
    log_dir = Path(_LOG_DIR)
    if not log_dir.exists():
        return entries

    for filepath in sorted(log_dir.glob("*.jsonl")):
        # Extract date from filename
        try:
            file_date = datetime.strptime(filepath.stem, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        if start_date and file_date.date() < start_date.date():
            continue
        if end_date and file_date.date() > end_date.date():
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if _is_benchmark_entry(entry):
                            continue
                        entry_ts = _parse_ts(entry.get("timestamp"))
                        if entry_ts is not None:
                            if start_date and entry_ts < start_date:
                                continue
                            if end_date and entry_ts > end_date:
                                continue
                        entries.append(entry)
        except OSError:
            continue

    return entries


