"""Process management for the toklog proxy daemon."""

from __future__ import annotations

import os
import signal
from pathlib import Path

_PID_FILE = Path.home() / ".toklog" / "proxy.pid"
DEFAULT_PORT = int(os.environ.get("TOKLOG_PROXY_PORT", "4007"))
DEFAULT_HOST = "127.0.0.1"


def start(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST) -> None:
    """Start the proxy in the foreground. Called by launchd/systemd or background fork."""
    import uvicorn

    from toklog.config import load_config, migrate_from_skip_processes_file
    from toklog.proxy.server import app

    # One-time migration from old skip_processes file to config.json
    migrate_from_skip_processes_file()

    # Load config; use file values if present, otherwise defaults
    config = load_config(validate=False)  # Validate after we get port/host
    config_port = config.get("proxy", {}).get("port", DEFAULT_PORT)
    config_host = config.get("proxy", {}).get("host", DEFAULT_HOST)

    # CLI args override config file
    final_port = port if port != DEFAULT_PORT else config_port
    final_host = host if host != DEFAULT_HOST else config_host

    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    try:
        uvicorn.run(app, host=final_host, port=final_port, log_level="warning", access_log=False)
    finally:
        _PID_FILE.unlink(missing_ok=True)


def stop() -> bool:
    """Send SIGTERM to the running daemon. Returns True if signal was sent."""
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        _PID_FILE.unlink(missing_ok=True)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        _PID_FILE.unlink(missing_ok=True)
        return False


def status() -> dict:
    """Return daemon status: {"running": bool, "pid": int?, "port": int?}."""
    if not _PID_FILE.exists():
        return {"running": False}
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0: check process existence without signalling
        return {"running": True, "pid": pid, "port": DEFAULT_PORT}
    except (ValueError, ProcessLookupError, PermissionError):
        _PID_FILE.unlink(missing_ok=True)
        return {"running": False}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TokLog proxy daemon")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    args = parser.parse_args()
    start(port=args.port, host=args.host)
