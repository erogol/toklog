"""Tests for the doctor health-check module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from toklog.doctor import (
    CheckResult,
    _check_config,
    _check_environment,
    _check_logging,
    _check_proxy,
    _check_traffic,
    _format_bytes,
    _mask_key,
    run_doctor_checks,
)

class TestMaskKey:
    def test_long_key(self) -> None:
        assert _mask_key("sk-proj-abcdefghijklmnop") == "sk-pr...mnop"

    def test_short_key(self) -> None:
        assert _mask_key("sk-abcdef") == "sk-...ef"

    def test_exact_12(self) -> None:
        assert _mask_key("123456789012") == "123...12"

    def test_longer_than_12(self) -> None:
        result = _mask_key("1234567890123")
        assert result == "12345...0123"

class TestFormatBytes:
    def test_bytes(self) -> None:
        assert _format_bytes(500) == "500 B"

    def test_kilobytes(self) -> None:
        assert _format_bytes(2048) == "2.0 KB"

    def test_megabytes(self) -> None:
        assert _format_bytes(50 * 1024 * 1024) == "50.0 MB"

    def test_gigabytes(self) -> None:
        assert _format_bytes(3 * 1024 * 1024 * 1024) == "3.0 GB"

    def test_zero(self) -> None:
        assert _format_bytes(0) == "0 B"

class TestCheckEnvironment:
    def test_all_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-testkey1234")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-testkey5678")
        results = _check_environment()
        assert len(results) == 2
        assert all(r.status == "pass" for r in results)

    def test_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        results = _check_environment()
        assert len(results) == 2
        assert all(r.status == "fail" for r in results)
        assert all(r.fix for r in results)

    def test_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
        results = _check_environment()
        assert results[0].status == "fail"
        assert results[1].status == "pass"

    def test_masked_in_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnop")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz")
        results = _check_environment()
        assert "abcdefghijklmnop" not in results[0].message
        assert "sk-pr" in results[0].message

class TestCheckConfig:
    def test_valid_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "proxy": {"port": 4007, "host": "127.0.0.1", "enabled": True},
            "logging": {"retention_days": 30, "max_file_size_mb": 100, "enabled": True},
            "skip_processes": [],
            "features": {
                "anomaly_detection": True,
                "daily_digest": True,
                "auto_start": True,
            },
        }))
        monkeypatch.setattr(
            "toklog.config.get_config_path", lambda: config_path
        )
        results = _check_config()
        assert all(r.status == "pass" for r in results)
        assert any("4007" in r.message for r in results)
        assert any("30" in r.message for r in results)

    def test_missing_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "config.json"
        monkeypatch.setattr(
            "toklog.config.get_config_path", lambda: config_path
        )
        results = _check_config()
        assert len(results) == 1
        assert results[0].status == "fail"
        assert "not found" in results[0].message

    def test_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text("{invalid json")
        monkeypatch.setattr(
            "toklog.config.get_config_path", lambda: config_path
        )
        results = _check_config()
        assert any(r.status == "fail" for r in results)

class TestCheckProxy:
    def test_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "proxy": {"port": 4007, "host": "127.0.0.1", "enabled": True},
            "logging": {"retention_days": 30, "max_file_size_mb": 100, "enabled": True},
            "skip_processes": [],
            "features": {"anomaly_detection": True, "daily_digest": True, "auto_start": True},
        }))
        monkeypatch.setattr(
            "toklog.config.get_config_path", lambda: config_path
        )
        monkeypatch.setattr(
            "toklog.proxy.daemon.status",
            lambda: {"running": True, "pid": 1234},
        )
        results = _check_proxy()
        assert any(r.status == "pass" and "Daemon running" in r.message for r in results)

    def test_not_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "proxy": {"port": 4007, "host": "127.0.0.1", "enabled": True},
            "logging": {"retention_days": 30, "max_file_size_mb": 100, "enabled": True},
            "skip_processes": [],
            "features": {"anomaly_detection": True, "daily_digest": True, "auto_start": True},
        }))
        monkeypatch.setattr(
            "toklog.config.get_config_path", lambda: config_path
        )
        monkeypatch.setattr(
            "toklog.proxy.daemon.status",
            lambda: {"running": False},
        )
        results = _check_proxy()
        assert any(r.status == "fail" and "not running" in r.message for r in results)

class TestCheckLogging:
    def test_with_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "2025-03-16.jsonl").write_text("{}")
        monkeypatch.setattr(
            "toklog.logger.get_log_dir", lambda: str(log_dir)
        )
        results = _check_logging()
        assert any(r.status == "pass" and "log files" in r.message for r in results)

    def test_empty_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        monkeypatch.setattr(
            "toklog.logger.get_log_dir", lambda: str(log_dir)
        )
        results = _check_logging()
        assert any(r.status == "warn" and "No log files" in r.message for r in results)

    def test_no_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(
            "toklog.logger.get_log_dir", lambda: str(log_dir)
        )
        results = _check_logging()
        assert any(r.status == "fail" and "missing" in r.message for r in results)

def _make_entry(**overrides: object) -> dict:
    """Create a minimal log entry for traffic tests."""
    entry = {
        "timestamp": "2026-03-20T10:00:00.000Z",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "input_tokens": 100,
        "output_tokens": 50,
        "error": False,
        "error_type": None,
    }
    entry.update(overrides)
    return entry


class TestCheckTraffic:
    def test_no_traffic_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No entries in last 24h produces a warning."""
        monkeypatch.setattr("toklog.doctor.read_logs", lambda **kw: [])
        results = _check_traffic()
        assert len(results) == 1
        assert results[0].status == "warn"
        assert "No traffic" in results[0].message

    def test_healthy_traffic_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All successful calls produce a pass."""
        entries = [_make_entry() for _ in range(100)]
        monkeypatch.setattr("toklog.doctor.read_logs", lambda **kw: entries)
        results = _check_traffic()
        assert len(results) == 1
        assert results[0].status == "pass"
        assert "healthy" in results[0].message

    def test_high_401_rate_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """401 errors above 1% threshold produce a fail with provider context."""
        entries = [_make_entry() for _ in range(90)]
        entries += [_make_entry(error=True, error_type="http_401", provider="anthropic") for _ in range(10)]
        monkeypatch.setattr("toklog.doctor.read_logs", lambda **kw: entries)
        results = _check_traffic()
        fail_results = [r for r in results if r.status == "fail"]
        assert len(fail_results) == 1
        assert "401" in fail_results[0].message
        assert "anthropic" in fail_results[0].message
        assert "key" in fail_results[0].fix.lower()

    def test_high_404_rate_fails_with_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """404 errors above threshold show the bad model names."""
        entries = [_make_entry() for _ in range(90)]
        entries += [_make_entry(error=True, error_type="http_404", model="claude-3-opus-20240229") for _ in range(10)]
        monkeypatch.setattr("toklog.doctor.read_logs", lambda **kw: entries)
        results = _check_traffic()
        fail_results = [r for r in results if r.status == "fail"]
        assert len(fail_results) == 1
        assert "404" in fail_results[0].message
        assert "claude-3-opus-20240229" in fail_results[0].message

    def test_low_error_rate_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Errors below threshold still produce a pass."""
        entries = [_make_entry() for _ in range(200)]
        entries += [_make_entry(error=True, error_type="http_401") for _ in range(1)]  # 0.5%
        monkeypatch.setattr("toklog.doctor.read_logs", lambda **kw: entries)
        results = _check_traffic()
        # Should pass — no fail or warn for the 401
        fail_results = [r for r in results if r.status == "fail"]
        assert len(fail_results) == 0
        assert any("healthy" in r.message for r in results)

    def test_transient_errors_warn_not_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """529/503 above threshold produce warn, not fail."""
        entries = [_make_entry() for _ in range(80)]
        entries += [_make_entry(error=True, error_type="http_529") for _ in range(20)]  # 20%
        monkeypatch.setattr("toklog.doctor.read_logs", lambda **kw: entries)
        results = _check_traffic()
        warn_results = [r for r in results if r.status == "warn"]
        assert any("529" in r.message for r in warn_results)
        fail_results = [r for r in results if r.status == "fail"]
        assert len(fail_results) == 0

    def test_multiple_error_types(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple error types above threshold each produce their own result."""
        entries = [_make_entry() for _ in range(80)]
        entries += [_make_entry(error=True, error_type="http_401") for _ in range(10)]
        entries += [_make_entry(error=True, error_type="http_404", model="bad-model") for _ in range(10)]
        monkeypatch.setattr("toklog.doctor.read_logs", lambda **kw: entries)
        results = _check_traffic()
        fail_results = [r for r in results if r.status == "fail"]
        assert len(fail_results) == 2  # one for 401, one for 404


class TestRunDoctorChecks:
    def test_returns_sections_and_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setattr(
            "toklog.config.get_config_path",
            lambda: Path("/nonexistent/config.json"),
        )
        sections, fail_count = run_doctor_checks()
        assert len(sections) == 5
        assert sections[0][0] == "Environment"
        assert sections[4][0] == "Traffic"
        assert fail_count > 0  # Config missing

class TestDoctorCLI:
    def test_exit_0_all_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setattr(
            "toklog.doctor.run_doctor_checks",
            lambda: (
                [
                    ("Environment", [CheckResult("pass", "All set", "")]),
                    ("Config", [CheckResult("pass", "All good", "")]),
                    ("Proxy", [CheckResult("pass", "Running", "")]),
                    ("Logging", [CheckResult("pass", "OK", "")]),
                    ("Traffic", [CheckResult("pass", "Traffic healthy", "")]),
                ],
                0,
            ),
        )
        from toklog.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0

    def test_exit_1_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "toklog.doctor.run_doctor_checks",
            lambda: (
                [
                    ("Environment", [CheckResult("fail", "Missing key", "export KEY=...")]),
                ],
                1,
            ),
        )
        from toklog.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 1