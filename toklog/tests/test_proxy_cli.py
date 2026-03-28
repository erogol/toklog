"""Unit tests for toklog/proxy/cli.py — proxy start, background, and budget."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from toklog.proxy.cli import proxy_start, proxy_status


# ---------------------------------------------------------------------------
# proxy start --background: log file creation
# ---------------------------------------------------------------------------


def test_background_start_creates_log_file(tmp_path: Path) -> None:
    """Background start should redirect child stdout/stderr to ~/.toklog/proxy.log."""
    log_path = tmp_path / "proxy.log"

    fake_popen = MagicMock()
    fake_pid_file = tmp_path / "proxy.pid"

    # Simulate the daemon writing the PID file quickly
    def _start_popen(*args, **kwargs):
        fake_pid_file.write_text("99999")
        return MagicMock()

    runner = CliRunner()
    with (
        patch("toklog.proxy.daemon.status", return_value={"running": False}),
        patch("toklog.proxy.daemon._PID_FILE", fake_pid_file),
        patch("subprocess.Popen", side_effect=_start_popen) as mock_popen,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        result = runner.invoke(proxy_start, ["--port", "4007", "--background"])

    assert result.exit_code == 0, result.output
    assert "Proxy started in background" in result.output
    # Log path should be shown in output (rich may wrap long paths)
    assert "proxy.log" in result.output.replace("\n", "")

    # Verify Popen was called with stdout and stderr file handles
    assert mock_popen.call_count == 1
    _, kwargs = mock_popen.call_args
    assert "stdout" in kwargs, "stdout should be redirected to log file"
    assert "stderr" in kwargs, "stderr should be redirected to log file"
    # Both should be the same file object
    assert kwargs["stdout"] is kwargs["stderr"]
    assert kwargs["start_new_session"] is True


def test_background_start_does_not_start_if_already_running() -> None:
    """If proxy is already running, start should be a no-op."""
    runner = CliRunner()
    with patch(
        "toklog.proxy.daemon.status",
        return_value={"running": True, "pid": 12345, "port": 4007},
    ):
        result = runner.invoke(proxy_start, ["--background"])

    assert result.exit_code == 0
    assert "already running" in result.output


# ---------------------------------------------------------------------------
# proxy start --budget: CLI option threading
# ---------------------------------------------------------------------------


def test_budget_option_accepted_foreground() -> None:
    """--budget option is accepted and passed to start() in foreground mode."""
    runner = CliRunner()
    with (
        patch("toklog.proxy.daemon.status", return_value={"running": False}),
        patch("toklog.proxy.daemon.start") as mock_start,
    ):
        result = runner.invoke(proxy_start, ["--budget", "5.00"])

    assert result.exit_code == 0, result.output
    assert "budget: $5.00/day" in result.output
    mock_start.assert_called_once_with(port=4007, budget_usd=5.0)


def test_budget_passed_to_subprocess_in_background(tmp_path: Path) -> None:
    """--budget value is threaded through to subprocess args in background mode."""
    fake_pid_file = tmp_path / "proxy.pid"

    def _start_popen(*args, **kwargs):
        fake_pid_file.write_text("99999")
        return MagicMock()

    runner = CliRunner()
    with (
        patch("toklog.proxy.daemon.status", return_value={"running": False}),
        patch("toklog.proxy.daemon._PID_FILE", fake_pid_file),
        patch("subprocess.Popen", side_effect=_start_popen) as mock_popen,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        result = runner.invoke(
            proxy_start, ["--port", "4007", "--background", "--budget", "10.0"]
        )

    assert result.exit_code == 0, result.output
    assert "budget: $10.00/day" in result.output

    # Verify --budget appears in subprocess command
    popen_args = mock_popen.call_args[0][0]  # positional arg 0 = cmd list
    assert "--budget" in popen_args
    budget_idx = popen_args.index("--budget")
    assert popen_args[budget_idx + 1] == "10.0"


def test_background_no_budget_omits_flag(tmp_path: Path) -> None:
    """When --budget is not set, subprocess args should NOT contain --budget."""
    fake_pid_file = tmp_path / "proxy.pid"

    def _start_popen(*args, **kwargs):
        fake_pid_file.write_text("99999")
        return MagicMock()

    runner = CliRunner()
    with (
        patch("toklog.proxy.daemon.status", return_value={"running": False}),
        patch("toklog.proxy.daemon._PID_FILE", fake_pid_file),
        patch("subprocess.Popen", side_effect=_start_popen) as mock_popen,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        result = runner.invoke(proxy_start, ["--port", "4007", "--background"])

    assert result.exit_code == 0, result.output
    popen_args = mock_popen.call_args[0][0]
    assert "--budget" not in popen_args


# ---------------------------------------------------------------------------
# proxy status: budget display
# ---------------------------------------------------------------------------


def test_proxy_status_shows_budget_when_enforcing(tmp_path: Path) -> None:
    """proxy status shows budget bar when budget state file has enforcing=True."""
    state_file = tmp_path / "budget_state.json"
    state_file.write_text(json.dumps({
        "enforcing": True,
        "daily_spend": 3.50,
        "limit_usd": 10.0,
        "remaining": 6.50,
        "rejected_count": 0,
        "date": "2025-01-01",
    }))

    runner = CliRunner()
    with (
        patch(
            "toklog.proxy.daemon.status",
            return_value={"running": True, "pid": 123, "port": 4007},
        ),
        patch(
            "toklog.proxy.budget.load_status_from_file",
            return_value=json.loads(state_file.read_text()),
        ),
    ):
        result = runner.invoke(proxy_status)

    assert result.exit_code == 0, result.output
    assert "Budget: $3.50 / $10.00 (35%)" in result.output


def test_proxy_status_no_budget_when_not_enforcing() -> None:
    """proxy status omits budget line when enforcing is False."""
    runner = CliRunner()
    with (
        patch(
            "toklog.proxy.daemon.status",
            return_value={"running": True, "pid": 123, "port": 4007},
        ),
        patch(
            "toklog.proxy.budget.load_status_from_file",
            return_value={"enforcing": False, "daily_spend": 0, "limit_usd": None},
        ),
    ):
        result = runner.invoke(proxy_status)

    assert result.exit_code == 0, result.output
    assert "Budget:" not in result.output
