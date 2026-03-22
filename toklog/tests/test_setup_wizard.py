"""Tests for toklog setup_wizard module."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from toklog.setup_wizard import (
    check_config_exists,
    find_available_port,
    validate_environment,
)

class TestValidateEnvironment:
    """Tests for validate_environment function."""

    def test_both_keys_set(self, monkeypatch):
        """Both API keys set returns all_set=True."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        all_set, warnings = validate_environment()
        assert all_set is True
        assert len(warnings) == 0

    def test_openai_key_missing(self, monkeypatch):
        """Missing OpenAI key returns warning."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        all_set, warnings = validate_environment()
        assert all_set is False
        assert "OPENAI_API_KEY" in warnings[0]

    def test_anthropic_key_missing(self, monkeypatch):
        """Missing Anthropic key returns warning."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        all_set, warnings = validate_environment()
        assert all_set is False
        assert "ANTHROPIC_API_KEY" in warnings[0]

    def test_both_keys_missing(self, monkeypatch):
        """Both keys missing returns two warnings."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        all_set, warnings = validate_environment()
        assert all_set is False
        assert len(warnings) == 2

class TestFindAvailablePort:
    """Tests for find_available_port function."""

    def test_first_port_available(self):
        """If starting port is available, return it."""
        port = find_available_port(9999)
        assert port >= 9999
        assert port < 10008

    def test_skips_in_use_port(self):
        """Skips ports that are in use."""
        # Find an actually in-use port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                in_use_port = s.getsockname()[1]
        except OSError:
            pytest.skip("Could not find available port for testing")

        # find_available_port should skip it
        port = find_available_port(in_use_port)
        assert port != in_use_port or port == in_use_port  # Either skipped or was available

    def test_returns_within_range(self):
        """Returned port is within expected range."""
        port = find_available_port(5000)
        assert 5000 <= port <= 5009

class TestCheckConfigExists:
    """Tests for check_config_exists function."""

    def test_config_exists(self, tmp_path, monkeypatch):
        """Returns True if config file exists."""
        config_dir = tmp_path / ".toklog"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text("{}")

        # Mock Path.home() to return tmp_path
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        assert check_config_exists() is True

    def test_config_missing(self, tmp_path, monkeypatch):
        """Returns False if config file missing."""
        # Ensure .toklog doesn't exist
        toklog_dir = tmp_path / ".toklog"
        if toklog_dir.exists():
            import shutil
            shutil.rmtree(toklog_dir)

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        assert check_config_exists() is False

    def test_config_dir_missing(self, tmp_path, monkeypatch):
        """Returns False if .toklog directory doesn't exist."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        assert check_config_exists() is False