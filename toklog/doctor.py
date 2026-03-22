"""Health check for TokLog proxy installation."""

from __future__ import annotations

import os
import socket
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Literal, NamedTuple, Tuple

from toklog.logger import read_logs


class CheckResult(NamedTuple):
    status: Literal["pass", "fail", "warn"]
    message: str
    fix: str  # empty string if no fix needed


def _mask_key(key: str) -> str:
    """Mask an API key: show first 5 + last 4 chars."""
    if len(key) <= 12:
        return key[:3] + "..." + key[-2:]
    return key[:5] + "..." + key[-4:]


def _format_bytes(n: int) -> str:
    """Human-readable byte size."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


def _check_environment() -> List[CheckResult]:
    """Check API keys are set and non-empty."""
    results: List[CheckResult] = []
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var, "")
        if not val:
            results.append(CheckResult(
                "fail",
                f"{var} not set",
                f"export {var}=<your-key>",
            ))
        elif not val.strip():
            results.append(CheckResult(
                "fail",
                f"{var} is empty",
                f"export {var}=<your-key>",
            ))
        else:
            results.append(CheckResult(
                "pass",
                f"{var} is set ({_mask_key(val)})",
                "",
            ))
    return results


def _check_config() -> List[CheckResult]:
    """Check config file exists, is valid JSON, and passes schema validation."""
    import json
    from toklog.config import get_config_path, load_config

    results: List[CheckResult] = []
    path = get_config_path()

    if not path.exists():
        results.append(CheckResult(
            "fail",
            f"Config not found: {path}",
            "toklog setup",
        ))
        return results

    results.append(CheckResult("pass", f"Config exists: {path}", ""))

    try:
        config = load_config(config_path=path, validate=True)
        results.append(CheckResult("pass", "Config is valid JSON", ""))
    except json.JSONDecodeError as e:
        results.append(CheckResult(
            "fail",
            "Config is not valid JSON",
            f"Fix syntax in {path}",
        ))
        return results
    except Exception as e:
        results.append(CheckResult(
            "fail",
            f"Config validation failed: {str(e)[:50]}",
            "toklog setup",
        ))
        return results

    # Check specific values
    port = config.get("proxy", {}).get("port", "?")
    results.append(CheckResult("pass", f"Config port: {port}", ""))

    retention = config.get("logging", {}).get("retention_days", "?")
    results.append(CheckResult("pass", f"Config retention: {retention} days", ""))

    skip_count = len(config.get("skip_processes", []))
    results.append(CheckResult("pass", f"Skip patterns: {skip_count} configured", ""))

    return results


def _check_proxy() -> List[CheckResult]:
    """Check proxy daemon: running, TCP reachable, shell profile."""
    from toklog.config import get_config_path, load_config
    from toklog.proxy.daemon import status
    from toklog.proxy.setup import _default_profile

    results: List[CheckResult] = []

    # Check daemon running
    try:
        st = status()
        if st["running"]:
            results.append(CheckResult(
                "pass",
                f"Daemon running (PID {st['pid']})",
                "",
            ))
        else:
            results.append(CheckResult(
                "fail",
                "Daemon not running",
                "toklog proxy start --background",
            ))
            return results  # Can't check TCP if not running
    except Exception:
        results.append(CheckResult(
            "fail",
            "Cannot get daemon status",
            "Check proxy daemon installation",
        ))
        return results

    # Check TCP reachable
    try:
        config = load_config(validate=False)
        port = config.get("proxy", {}).get("port", 4007)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("127.0.0.1", port))

        results.append(CheckResult(
            "pass",
            f"Listening on 127.0.0.1:{port}",
            "",
        ))
    except Exception as e:
        results.append(CheckResult(
            "fail",
            f"Cannot reach proxy: {str(e)[:40]}",
            "toklog proxy logs",
        ))

    # Check shell profile
    try:
        profile = _default_profile()
        if profile.exists():
            content = profile.read_text()
            if "OPENAI_BASE_URL" in content and "ANTHROPIC_BASE_URL" in content:
                results.append(CheckResult(
                    "pass",
                    "Shell profile has env vars",
                    "",
                ))
            else:
                results.append(CheckResult(
                    "warn",
                    "Shell profile missing env vars",
                    "toklog setup --auto-start",
                ))
        else:
            results.append(CheckResult(
                "warn",
                "Shell profile not found",
                "Open a new terminal or source ~/.bashrc",
            ))
    except Exception:
        results.append(CheckResult(
            "warn",
            "Could not check shell profile",
            "Manually verify OPENAI_BASE_URL and ANTHROPIC_BASE_URL in shell",
        ))

    return results


def _check_logging() -> List[CheckResult]:
    """Check logging: directory exists, files present, disk usage."""
    from toklog.logger import get_log_dir

    results: List[CheckResult] = []
    log_dir = Path(get_log_dir())

    if not log_dir.exists():
        results.append(CheckResult(
            "fail",
            f"Log directory missing: {log_dir}",
            "toklog init",
        ))
        return results

    results.append(CheckResult("pass", f"Log directory exists: {log_dir}", ""))

    jsonl_files = sorted(log_dir.glob("*.jsonl"))
    if not jsonl_files:
        results.append(CheckResult(
            "warn",
            "No log files yet",
            "Logs appear after first API call through proxy",
        ))
        return results

    total_size = sum(f.stat().st_size for f in jsonl_files)
    results.append(CheckResult(
        "pass",
        f"{len(jsonl_files)} log files ({_format_bytes(total_size)})",
        "",
    ))

    # Check latest log date
    latest = jsonl_files[-1].stem  # YYYY-MM-DD
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if latest == today:
        label = "today"
    else:
        try:
            latest_dt = datetime.strptime(latest, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_ago = (datetime.now(timezone.utc) - latest_dt).days
            label = f"{days_ago} days ago"
        except ValueError:
            label = latest
    results.append(CheckResult("pass", f"Latest log: {label}", ""))

    # Warn if > 5 GB
    if total_size > 5 * 1024 * 1024 * 1024:
        results.append(CheckResult(
            "warn",
            f"Disk usage: {_format_bytes(total_size)}",
            "toklog reset",
        ))

    return results


def _check_traffic() -> List[CheckResult]:
    """Check recent traffic for error patterns that indicate misconfiguration."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=1)

    try:
        entries = read_logs(start_date=start, end_date=now)
    except Exception:
        return [CheckResult("warn", "Could not read logs for traffic check", "")]

    if not entries:
        return [CheckResult(
            "warn",
            "No traffic in last 24h",
            "Is traffic routed through the proxy? Check OPENAI_BASE_URL / ANTHROPIC_BASE_URL",
        )]

    total = len(entries)
    error_entries = [e for e in entries if e.get("error") is True]

    if not error_entries:
        return [CheckResult("pass", f"Traffic healthy: {total} calls, 0 errors", "")]

    # Count errors by type and collect context
    type_counts: Counter = Counter()
    type_models: dict = {}
    type_providers: dict = {}
    for e in error_entries:
        etype = e.get("error_type") or "unknown"
        type_counts[etype] += 1
        if etype not in type_models:
            type_models[etype] = set()
            type_providers[etype] = set()
        model = e.get("model")
        provider = e.get("provider")
        if model:
            type_models[etype].add(model)
        if provider:
            type_providers[etype].add(provider)

    results: List[CheckResult] = []

    # Diagnosis rules — ordered by severity
    _DIAGNOSIS = [
        # (error_types, threshold_pct, status, message_fn, fix)
        (
            {"http_401"},
            1.0,
            "fail",
            lambda cnt, providers, models: (
                f"{cnt} auth failures (401) — "
                f"provider{'s' if len(providers) > 1 else ''}: {', '.join(sorted(providers)) or 'unknown'}"
            ),
            "Check/rotate API key for the affected provider",
        ),
        (
            {"http_404"},
            1.0,
            "fail",
            lambda cnt, providers, models: (
                f"{cnt} model-not-found errors (404) — "
                f"models: {', '.join(sorted(models)[:5]) or 'unknown'}"
            ),
            "Check model names — these may be deprecated or mistyped",
        ),
        (
            {"http_400"},
            1.0,
            "warn",
            lambda cnt, providers, models: f"{cnt} bad requests (400)",
            "Check client configuration — requests are malformed",
        ),
        (
            {"http_529", "http_503"},
            5.0,
            "warn",
            lambda cnt, providers, models: (
                f"{cnt} provider overload errors (529/503)"
            ),
            "Provider is overloaded — transient, monitor and retry",
        ),
    ]

    matched_types: set = set()
    for error_types, threshold, status, msg_fn, fix in _DIAGNOSIS:
        count = sum(type_counts.get(et, 0) for et in error_types)
        if count == 0:
            continue
        rate = count / total * 100
        if rate >= threshold:
            # Collect context from all matching error types
            models: set = set()
            providers: set = set()
            for et in error_types:
                models.update(type_models.get(et, set()))
                providers.update(type_providers.get(et, set()))
            results.append(CheckResult(status, msg_fn(count, providers, models), fix))
            matched_types.update(error_types)

    # Summary for unmatched / below-threshold errors
    total_errors = len(error_entries)
    error_rate = total_errors / total * 100
    if not results:
        results.append(CheckResult(
            "pass",
            f"Traffic healthy: {total} calls, {total_errors} errors ({error_rate:.1f}%)",
            "",
        ))
    else:
        # Add overall context line
        results.insert(0, CheckResult(
            "pass" if error_rate < 5 else "warn",
            f"{total} calls, {total_errors} errors ({error_rate:.1f}%)",
            "",
        ))

    return results


def run_doctor_checks() -> Tuple[List[Tuple[str, List[CheckResult]]], int]:
    """Run all doctor checks.

    Returns:
        (sections, fail_count) where sections is a list of
        (section_name, checks) tuples, and fail_count is the
        number of non-pass checks.
    """
    sections = [
        ("Environment", _check_environment()),
        ("Config", _check_config()),
        ("Proxy", _check_proxy()),
        ("Logging", _check_logging()),
        ("Traffic", _check_traffic()),
    ]

    fail_count = sum(
        1 for _, checks in sections
        for check in checks
        if check.status in ("fail", "warn")
    )

    return sections, fail_count
