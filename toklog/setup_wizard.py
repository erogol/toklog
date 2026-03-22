"""Interactive setup wizard for TokLog proxy."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import click
from rich.console import Console


def validate_environment() -> tuple[bool, list[str]]:
    """Check required environment variables. Returns (all_set, warnings)."""
    warnings = []

    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY not set")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        warnings.append("ANTHROPIC_API_KEY not set")

    all_set = len(warnings) == 0
    return all_set, warnings


def find_available_port(start_port: int = 4007) -> int:
    """Find an available port starting from start_port.

    Tries up to 10 ports. If all busy, returns the original port
    (user will get error when trying to start).
    """
    port = start_port
    while port < start_port + 10:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            port += 1
    # All ports busy; return the original
    return start_port


def check_config_exists() -> bool:
    """Return True if config file already exists."""
    return (Path.home() / ".toklog" / "config.json").exists()


def confirm_overwrite_config() -> bool:
    """Ask user if they want to overwrite existing config."""
    return click.confirm("Config file already exists. Overwrite?", default=False)


def _backup_config() -> None:
    """Backup existing config file."""
    import shutil

    config_path = Path.home() / ".toklog" / "config.json"
    if config_path.exists():
        backup_path = config_path.with_stem(config_path.stem + ".backup")
        shutil.copy(config_path, backup_path)


def _get_proxy_status() -> dict:
    """Get proxy daemon status."""
    from toklog.proxy.daemon import status

    return status()


def _start_proxy_background(port: int) -> bool:
    """Start proxy in background. Returns True if successful."""
    try:
        subprocess.Popen(
            [sys.executable, "-m", "toklog.proxy.daemon", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give it a moment to start
        time.sleep(0.5)
        return _get_proxy_status()["running"]
    except Exception:
        return False


def run_interactive_setup(
    non_interactive: bool = False,
    port_override: int | None = None,
    auto_start_override: bool | None = None,
    dry_run: bool = False,
) -> bool:
    """Run the interactive setup wizard. Returns True if successful.

    Args:
        non_interactive: Skip interactive questions (use defaults + overrides)
        port_override: Specific port to use (skips port question)
        auto_start_override: Specific auto_start choice (skips question)
        dry_run: Show what would happen without making changes

    Returns:
        True if setup succeeded, False if cancelled or errored
    """
    console = Console()

    # Step 1: Environment validation
    console.print("[cyan]Checking environment...[/cyan]")
    all_set, warnings = validate_environment()

    if warnings:
        for warn in warnings:
            console.print(f"[yellow]⚠[/yellow] {warn}")

    if not all_set:
        console.print("[yellow]Proceeding without all keys...[/yellow]")

    console.print()

    # Step 2: Check if config already exists
    if check_config_exists():
        console.print("[cyan]Config file already exists[/cyan]")
        if not confirm_overwrite_config():
            console.print("[yellow]Setup cancelled[/yellow]")
            return False
        console.print("[yellow]Backing up old config...[/yellow]")
        _backup_config()
        console.print()

    # Step 3: Interactive questions (or use defaults in non-interactive mode)
    console.print("[cyan]Configuring TokLog proxy...[/cyan]")

    from toklog.config import create_config_interactive

    config = create_config_interactive(port_override=port_override)
    config_port = config["proxy"]["port"]

    if dry_run:
        console.print("[dim]Dry run mode: not making changes[/dim]")
        return True

    console.print()

    # Step 4: Setup shell profile
    console.print("[cyan]Setting up shell profile...[/cyan]")
    try:
        from toklog.proxy.setup import inject_shell_profile

        path = inject_shell_profile(port=config_port)
        console.print(f"[green]✓[/green] Env vars written to {path}")
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Could not update shell profile: {e}[/yellow]")

    # Step 5: Setup auto-start (if requested)
    if config["features"]["auto_start"]:
        console.print("[cyan]Configuring auto-start...[/cyan]")
        try:
            from toklog.proxy.setup import install_autostart

            install_autostart(port=config_port)
            console.print("[green]✓[/green] Auto-start configured")
        except Exception as e:
            console.print(f"[yellow]⚠ Warning: Could not install auto-start: {e}[/yellow]")

    # Step 6: Start proxy
    console.print("[cyan]Starting proxy...[/cyan]")
    try:
        st = _get_proxy_status()
        if st["running"]:
            console.print(f"[green]●[/green] Proxy already running (PID {st['pid']})")
        else:
            if _start_proxy_background(config_port):
                console.print(f"[green]✓[/green] Proxy running on 127.0.0.1:{config_port}")
            else:
                console.print(
                    f"[yellow]⚠ Proxy may not have started. Try: toklog proxy start --background[/yellow]"
                )
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Could not start proxy: {e}[/yellow]")

    # Step 7: Success summary
    console.print()
    console.print("[green]Setup complete![/green]")
    console.print(f"[cyan]OPENAI_BASE_URL[/cyan]=http://127.0.0.1:{config_port}/openai")
    console.print(f"[cyan]ANTHROPIC_BASE_URL[/cyan]=http://127.0.0.1:{config_port}/anthropic")
    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print("  1. Open a new terminal (or: source ~/.bashrc)")
    console.print("  2. Run: toklog proxy status")
    console.print("  3. Run: toklog doctor (coming soon)")

    return True