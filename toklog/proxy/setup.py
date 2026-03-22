"""Shell profile injection and auto-start management for the TokLog proxy."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

_MARKER_START = "# TokLog HTTP Proxy — start"
_MARKER_END = "# TokLog HTTP Proxy — end"

_BLOCK_TEMPLATE = (
    "\n"
    "# TokLog HTTP Proxy — start\n"
    "export OPENAI_BASE_URL=http://127.0.0.1:{port}/openai\n"
    "export ANTHROPIC_BASE_URL=http://127.0.0.1:{port}/anthropic\n"
    "# TokLog HTTP Proxy — end\n"
)


def _default_profile() -> Path:
    shell = os.environ.get("SHELL", "")
    home = Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    if "fish" in shell:
        return home / ".config" / "fish" / "config.fish"
    return home / ".bashrc"


def inject_shell_profile(profile: Path | None = None, port: int = 4007) -> Path:
    """Add proxy env var block to the shell profile. Idempotent."""
    target = profile or _default_profile()
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    if _MARKER_START in existing:
        return target  # already present — no-op
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(_BLOCK_TEMPLATE.format(port=port))
    return target


def remove_shell_profile(profile: Path | None = None) -> bool:
    """Remove proxy env var block from the shell profile. Returns True if removed."""
    target = profile or _default_profile()
    if not target.exists():
        return False
    text = target.read_text(encoding="utf-8")
    if _MARKER_START not in text:
        return False

    lines = text.splitlines(keepends=True)
    inside = False
    filtered: list[str] = []
    for line in lines:
        if _MARKER_START in line:
            inside = True
        if not inside:
            filtered.append(line)
        if _MARKER_END in line:
            inside = False

    target.write_text("".join(filtered), encoding="utf-8")
    return True


def install_autostart(port: int = 4007) -> None:
    """Install launchd (macOS) or systemd (Linux) auto-start for the proxy daemon."""
    python = sys.executable
    if platform.system() == "Darwin":
        _install_launchd(python, port)
    elif platform.system() == "Linux":
        _install_systemd(python, port)


def remove_autostart() -> None:
    """Remove launchd/systemd auto-start configuration."""
    if platform.system() == "Darwin":
        plist = Path.home() / "Library" / "LaunchAgents" / "dev.toklog.proxy.plist"
        if plist.exists():
            subprocess.run(["launchctl", "unload", str(plist)], check=False)
            plist.unlink()
    elif platform.system() == "Linux":
        svc = (
            Path.home() / ".config" / "systemd" / "user" / "toklog-proxy.service"
        )
        if svc.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "toklog-proxy"], check=False
            )
            svc.unlink()


def _install_launchd(python: str, port: int) -> None:
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    log = str(Path.home() / ".toklog" / "proxy.log")
    plist = plist_dir / "dev.toklog.proxy.plist"
    plist.write_text(
        dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
              "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
                <key>Label</key><string>dev.toklog.proxy</string>
                <key>ProgramArguments</key>
                <array>
                    <string>{python}</string>
                    <string>-m</string>
                    <string>toklog.proxy.daemon</string>
                    <string>--port</string>
                    <string>{port}</string>
                </array>
                <key>RunAtLoad</key><true/>
                <key>KeepAlive</key><true/>
                <key>StandardOutPath</key><string>{log}</string>
                <key>StandardErrorPath</key><string>{log}</string>
            </dict>
            </plist>
        """),
        encoding="utf-8",
    )
    subprocess.run(["launchctl", "load", str(plist)], check=False)


def _install_systemd(python: str, port: int) -> None:
    svc_dir = Path.home() / ".config" / "systemd" / "user"
    svc_dir.mkdir(parents=True, exist_ok=True)
    svc = svc_dir / "toklog-proxy.service"
    svc.write_text(
        dedent(f"""\
            [Unit]
            Description=TokLog HTTP Proxy
            After=network.target

            [Service]
            Type=simple
            ExecStart={python} -m toklog.proxy.daemon --port {port}
            Restart=on-failure
            RestartSec=5
            StandardOutput=append:%h/.toklog/proxy.log
            StandardError=append:%h/.toklog/proxy.log

            [Install]
            WantedBy=default.target
        """),
        encoding="utf-8",
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "toklog-proxy"], check=False)
