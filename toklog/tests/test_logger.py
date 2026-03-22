"""Tests for the logger module."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

import toklog.logger as logger_mod
from toklog.logger import _is_benchmark_entry, log_entry, read_logs


@pytest.fixture(autouse=True)
def tmp_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect logs to a temp directory."""
    log_dir = str(tmp_path / "logs")
    monkeypatch.setattr(logger_mod, "_LOG_DIR", log_dir)
    monkeypatch.setattr(logger_mod, "_dir_ensured", False)
    return tmp_path


def _sample_entry(**overrides: dict) -> dict:
    entry = {
        "timestamp": "2025-03-09T10:00:00.000Z",
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "max_tokens_set": 4096,
        "system_prompt_hash": "abc123",
        "tool_count": 0,
        "tool_schema_tokens": 0,
        "tags": None,
        "streaming": False,
        "duration_ms": 1000,
        "error": False,
        "error_type": None,
        "request_id": "req_1",
    }
    entry.update(overrides)
    return entry


class TestLogEntry:
    def test_creates_directory(self, tmp_path: Path) -> None:
        """Log directory is created automatically on first write."""
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()
        log_entry(_sample_entry())
        assert log_dir.exists()

    def test_valid_json(self, tmp_path: Path) -> None:
        """Each logged line is valid JSON."""
        log_entry(_sample_entry())
        log_dir = tmp_path / "logs"
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text().strip()
        parsed = json.loads(content)
        assert parsed["provider"] == "openai"

    def test_all_fields_present(self, tmp_path: Path) -> None:
        """All schema fields are present in logged entry."""
        log_entry(_sample_entry())
        log_dir = tmp_path / "logs"
        files = list(log_dir.glob("*.jsonl"))
        content = files[0].read_text().strip()
        parsed = json.loads(content)

        expected_fields = {
            "timestamp", "provider", "model", "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_creation_tokens", "max_tokens_set",
            "system_prompt_hash", "tool_count", "tool_schema_tokens", "tags",
            "streaming", "duration_ms", "error", "error_type", "request_id",
        }
        assert set(parsed.keys()) == expected_fields

    def test_append_multiple(self, tmp_path: Path) -> None:
        """Multiple entries append to the same file."""
        log_entry(_sample_entry(model="gpt-4o"))
        log_entry(_sample_entry(model="gpt-4o-mini"))
        log_dir = tmp_path / "logs"
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 2

    def test_thread_safety(self, tmp_path: Path) -> None:
        """Write from multiple threads, verify no corruption."""
        errors: List[str] = []

        def writer(idx: int) -> None:
            try:
                for _ in range(20):
                    log_entry(_sample_entry(request_id=f"req_{idx}"))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all entries are valid JSON
        log_dir = tmp_path / "logs"
        total_entries = 0
        for f in log_dir.glob("*.jsonl"):
            for line in f.read_text().strip().splitlines():
                json.loads(line)  # Should not raise
                total_entries += 1
        assert total_entries == 100  # 5 threads * 20 entries


class TestReadLogs:
    def test_read_with_date_range(self, tmp_path: Path) -> None:
        """Read logs filters by date range."""
        log_entry(_sample_entry())
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, tzinfo=timezone.utc)
        entries = read_logs(start_date=start, end_date=end)
        assert len(entries) >= 1

    def test_read_empty(self, tmp_path: Path) -> None:
        """Reading with no log files returns empty list."""
        entries = read_logs()
        assert entries == []

    def test_read_all(self, tmp_path: Path) -> None:
        """read_logs with no date range returns everything."""
        log_entry(_sample_entry())
        log_entry(_sample_entry())
        entries = read_logs()
        assert len(entries) == 2

    def test_bench_entries_excluded_by_read_logs(self) -> None:
        """read_logs silently drops entries with a known benchmark request_id."""
        log_entry(_sample_entry(request_id="req-real"))
        log_entry(_sample_entry(request_id="chatcmpl-bench"))
        log_entry(_sample_entry(request_id="chatcmpl-bench"))
        entries = read_logs()
        assert len(entries) == 1
        assert entries[0]["request_id"] == "req-real"

    def test_all_bench_entries_returns_empty(self) -> None:
        """read_logs returns empty list when every entry is a benchmark entry."""
        log_entry(_sample_entry(request_id="chatcmpl-bench"))
        entries = read_logs()
        assert entries == []


class TestIsBenchmarkEntry:
    def test_known_id_is_benchmark(self) -> None:
        assert _is_benchmark_entry({"request_id": "chatcmpl-bench"}) is True

    def test_real_id_is_not_benchmark(self) -> None:
        assert _is_benchmark_entry({"request_id": "chatcmpl-abc123"}) is False

    def test_missing_id_is_not_benchmark(self) -> None:
        assert _is_benchmark_entry({}) is False
