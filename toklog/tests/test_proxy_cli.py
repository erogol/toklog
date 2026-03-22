"""Unit tests for toklog/proxy/cli.py — proxy start background log file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from toklog.proxy.cli import proxy_start


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
