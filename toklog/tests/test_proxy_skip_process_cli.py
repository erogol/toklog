"""Tests for toklog proxy skip-process add/remove/list CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from toklog.proxy.cli import proxy_cli


@pytest.fixture()
def skip_file(tmp_path: Path, monkeypatch) -> Path:
    """Patch the skip_processes file location into tmp_path."""
    sf = tmp_path / "skip_processes"
    # Monkey-patch Path.home() won't work cleanly; patch the CLI's lookup path via env
    # Instead, we rely on monkeypatching builtins.open indirectly — easier to just
    # patch the actual Path.home() to return tmp_path.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path / ".toklog" / "skip_processes"


def test_add_creates_file(skip_file: Path):
    runner = CliRunner()
    result = runner.invoke(proxy_cli, ["skip-process", "add", "pytest"])
    assert result.exit_code == 0, result.output
    assert skip_file.exists()
    assert "pytest" in skip_file.read_text()


def test_add_is_idempotent(skip_file: Path):
    runner = CliRunner()
    runner.invoke(proxy_cli, ["skip-process", "add", "pytest"])
    result = runner.invoke(proxy_cli, ["skip-process", "add", "pytest"])
    assert result.exit_code == 0
    lines = [l for l in skip_file.read_text().splitlines() if l == "pytest"]
    assert len(lines) == 1, "duplicate entry should not be written"


def test_add_multiple_patterns(skip_file: Path):
    runner = CliRunner()
    runner.invoke(proxy_cli, ["skip-process", "add", "pytest"])
    runner.invoke(proxy_cli, ["skip-process", "add", "myapp"])
    text = skip_file.read_text()
    assert "pytest" in text
    assert "myapp" in text


def test_remove_existing(skip_file: Path):
    runner = CliRunner()
    runner.invoke(proxy_cli, ["skip-process", "add", "pytest"])
    result = runner.invoke(proxy_cli, ["skip-process", "remove", "pytest"])
    assert result.exit_code == 0, result.output
    assert "pytest" not in skip_file.read_text()


def test_remove_missing(skip_file: Path):
    runner = CliRunner()
    result = runner.invoke(proxy_cli, ["skip-process", "remove", "nonexistent"])
    assert result.exit_code == 0
    assert "Not found" in result.output or "empty" in result.output.lower()


def test_list_empty(skip_file: Path):
    runner = CliRunner()
    result = runner.invoke(proxy_cli, ["skip-process", "list"])
    assert result.exit_code == 0
    assert "No skip patterns" in result.output


def test_list_entries(skip_file: Path):
    runner = CliRunner()
    runner.invoke(proxy_cli, ["skip-process", "add", "pytest"])
    runner.invoke(proxy_cli, ["skip-process", "add", "myapp"])
    result = runner.invoke(proxy_cli, ["skip-process", "list"])
    assert result.exit_code == 0
    assert "pytest" in result.output
    assert "myapp" in result.output
