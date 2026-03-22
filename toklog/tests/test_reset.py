"""Tests for `toklog reset` CLI command."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

import toklog.logger as logger_mod
from toklog.cli import cli


@pytest.fixture()
def isolated_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(logger_mod, "_LOG_DIR", str(log_dir))
    monkeypatch.setattr(logger_mod, "_dir_ensured", False)
    return log_dir


def _write_log(log_dir: Path, name: str, content: str = '{"x":1}\n') -> Path:
    p = log_dir / name
    p.write_text(content)
    return p


class TestReset:
    def test_no_logs_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No log directory → informative message, no crash."""
        missing = str(tmp_path / "nonexistent")
        monkeypatch.setattr(logger_mod, "_LOG_DIR", missing)
        result = CliRunner().invoke(cli, ["reset"])
        assert result.exit_code == 0
        assert "nothing to reset" in result.output.lower()

    def test_empty_log_dir(self, isolated_log_dir: Path) -> None:
        """Empty log directory → nothing to reset."""
        result = CliRunner().invoke(cli, ["reset"])
        assert result.exit_code == 0
        assert "nothing to reset" in result.output.lower()

    def test_lists_files_before_confirm(self, isolated_log_dir: Path) -> None:
        """Files are listed before the confirmation prompt."""
        _write_log(isolated_log_dir, "2025-01-01.jsonl")
        _write_log(isolated_log_dir, "2025-01-02.jsonl")
        result = CliRunner().invoke(cli, ["reset"], input="n\n")
        assert "2025-01-01.jsonl" in result.output
        assert "2025-01-02.jsonl" in result.output

    def test_abort_on_no(self, isolated_log_dir: Path) -> None:
        """Answering 'n' aborts without deleting files."""
        _write_log(isolated_log_dir, "2025-01-01.jsonl")
        result = CliRunner().invoke(cli, ["reset"], input="n\n")
        assert result.exit_code == 0
        assert "aborted" in result.output.lower()
        assert (isolated_log_dir / "2025-01-01.jsonl").exists()

    def test_deletes_on_yes(self, isolated_log_dir: Path) -> None:
        """Answering 'y' deletes all .jsonl files."""
        _write_log(isolated_log_dir, "2025-01-01.jsonl")
        _write_log(isolated_log_dir, "2025-01-02.jsonl")
        result = CliRunner().invoke(cli, ["reset"], input="y\n")
        assert result.exit_code == 0
        assert "deleted 2" in result.output.lower()
        assert not list(isolated_log_dir.glob("*.jsonl"))

    def test_non_jsonl_files_untouched(self, isolated_log_dir: Path) -> None:
        """Non-.jsonl files in the log dir are not deleted."""
        _write_log(isolated_log_dir, "2025-01-01.jsonl")
        other = isolated_log_dir / "notes.txt"
        other.write_text("keep me")
        CliRunner().invoke(cli, ["reset"], input="y\n")
        assert other.exists()
