"""Tests for the CLI categories commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import toklog.classify as classify_mod
from toklog.cli import cli


@pytest.fixture(autouse=True)
def tmp_categories_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect categories file to a temp path for test isolation."""
    f = tmp_path / "categories.json"
    monkeypatch.setattr(classify_mod, "_CATEGORIES_FILE", f)
    return f


class TestCategoriesCli:
    def test_add_then_list(self) -> None:
        """Adding categories and listing them shows the added names."""
        runner = CliRunner()
        result = runner.invoke(cli, ["categories", "add", "search", "coding"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["categories", "list"])
        assert result.exit_code == 0
        assert "search" in result.output
        assert "coding" in result.output

    def test_add_dedup(self) -> None:
        """Adding the same category twice only stores it once."""
        runner = CliRunner()
        runner.invoke(cli, ["categories", "add", "search"])
        runner.invoke(cli, ["categories", "add", "search"])
        result = runner.invoke(cli, ["categories", "list"])
        assert result.exit_code == 0
        assert result.output.count("search") == 1

    def test_remove(self) -> None:
        """Removing a category by name leaves the rest intact."""
        runner = CliRunner()
        runner.invoke(cli, ["categories", "add", "search", "coding"])
        result = runner.invoke(cli, ["categories", "remove", "search"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["categories", "list"])
        assert "search" not in result.output
        assert "coding" in result.output

    def test_clear(self) -> None:
        """Clear removes all categories."""
        runner = CliRunner()
        runner.invoke(cli, ["categories", "add", "search", "coding"])
        result = runner.invoke(cli, ["categories", "clear"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["categories", "list"])
        assert "No categories" in result.output

    def test_list_when_empty(self) -> None:
        """Listing with no categories prints a helpful message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["categories", "list"])
        assert result.exit_code == 0
        assert "No categories" in result.output

    def test_add_empty_string_rejected(self) -> None:
        """Adding an empty string is rejected; no category is stored."""
        runner = CliRunner()
        result = runner.invoke(cli, ["categories", "add", ""])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["categories", "list"])
        assert "No categories" in result.output
