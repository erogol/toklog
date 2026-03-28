"""CLI subcommand group: toklog proxy start|stop|status|setup|uninstall."""

from __future__ import annotations

import click
from rich.console import Console


@click.group("proxy")
def proxy_cli() -> None:
    """Manage the local HTTP proxy daemon."""


@proxy_cli.command("start")
@click.option("--port", default=4007, show_default=True, help="Port to listen on.")
@click.option("--background", is_flag=True, help="Fork into background.")
@click.option("--budget", default=None, type=float, help="Daily spend limit in USD.")
def proxy_start(port: int, background: bool, budget: float | None) -> None:
    """Start the TokLog HTTP proxy."""
    from toklog.proxy.daemon import start, status

    st = status()
    if st["running"]:
        Console().print(
            f"[yellow]Proxy already running[/yellow] (PID {st['pid']}, port {st['port']})"
        )
        return

    if background:
        import os
        import subprocess
        import sys
        import time as _time
        from pathlib import Path

        from toklog.proxy.daemon import _PID_FILE

        log_path = Path.home() / ".toklog" / "proxy.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fd = open(log_path, "a")  # noqa: SIM115 — file left open intentionally for child
        cmd = [sys.executable, "-m", "toklog.proxy.daemon", "--port", str(port)]
        if budget is not None:
            cmd.extend(["--budget", str(budget)])
        subprocess.Popen(
            cmd,
            env={**os.environ, "TOKLOG_PROXY_PORT": str(port)},
            start_new_session=True,
            stdout=log_fd,
            stderr=log_fd,
        )
        log_fd.close()
        # Poll until the daemon writes its PID file (up to 2s)
        deadline = _time.monotonic() + 2.0
        while _time.monotonic() < deadline:
            if _PID_FILE.exists():
                break
            _time.sleep(0.1)
        budget_msg = f" (budget: ${budget:.2f}/day)" if budget is not None else ""
        Console().print(f"[green]Proxy started in background on port {port}{budget_msg}.[/green]")
        Console().print(f"  Logs: {log_path}")
    else:
        budget_msg = f" (budget: ${budget:.2f}/day)" if budget is not None else ""
        Console().print(f"[green]Starting TokLog proxy on 127.0.0.1:{port}{budget_msg}...[/green]")
        start(port=port, budget_usd=budget)


@proxy_cli.command("stop")
def proxy_stop() -> None:
    """Stop the running proxy daemon."""
    from toklog.proxy.daemon import stop

    if stop():
        Console().print("[green]Proxy stopped.[/green]")
    else:
        Console().print("[yellow]No proxy running.[/yellow]")


@proxy_cli.command("status")
def proxy_status() -> None:
    """Show proxy daemon status."""
    from toklog.proxy.daemon import status as _status

    st = _status()
    console = Console()
    if st["running"]:
        port = st["port"]
        console.print(f"[green]● Proxy running[/green] — PID {st['pid']}, port {port}")
        console.print(f"  OPENAI_BASE_URL=http://127.0.0.1:{port}/openai")
        console.print(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{port}/anthropic")

        from toklog.proxy.budget import load_status_from_file

        budget_state = load_status_from_file()
        if budget_state and budget_state.get("enforcing"):
            spent = budget_state["daily_spend"]
            limit = budget_state["limit_usd"]
            pct = (spent / limit * 100) if limit > 0 else 0
            console.print(f"  Budget: ${spent:.2f} / ${limit:.2f} ({pct:.0f}%)")
    else:
        console.print("[red]● Proxy not running[/red]")
        console.print("  Run: toklog proxy start --background")


@proxy_cli.command("setup")
@click.option("--port", default=None, type=int, help="Port for proxy (default: 4007).")
@click.option("--auto-start", is_flag=True, help="Install launchd/systemd auto-start.")
def proxy_setup(port: int | None, auto_start: bool) -> None:
    """Interactive setup: create config, start proxy, configure auto-start."""
    from pathlib import Path

    from toklog.config import create_config_interactive
    from toklog.proxy.daemon import start, status
    from toklog.proxy.setup import inject_shell_profile, install_autostart

    console = Console()

    # Create config file interactively
    console.print("[cyan]Configuring TokLog proxy...[/cyan]")
    config = create_config_interactive(port_override=port)
    config_port = config["proxy"]["port"]
    console.print(f"[green]✓[/green] Config saved to ~/.toklog/config.json")

    # Inject env vars into shell profile
    path = inject_shell_profile(port=config_port)
    console.print(f"[green]✓[/green] Env vars written to {path}")

    # Setup auto-start if requested
    if auto_start:
        install_autostart(port=config_port)
        console.print("[green]✓[/green] Auto-start configured")
    else:
        # Offer to start the proxy now
        if click.confirm("Start proxy now?", default=True):
            import subprocess
            import sys

            subprocess.Popen(
                [sys.executable, "-m", "toklog.proxy.daemon", "--port", str(config_port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    # Show next steps
    console.print("\n[green]Setup complete![/green]")
    console.print(f"[cyan]OPENAI_BASE_URL[/cyan]=http://127.0.0.1:{config_port}/openai")
    console.print(f"[cyan]ANTHROPIC_BASE_URL[/cyan]=http://127.0.0.1:{config_port}/anthropic")
    console.print("\n[dim]Next steps:[/dim]")
    console.print("  1. Open a new terminal (or: source ~/.bashrc)")
    console.print("  2. Run: toklog proxy status")
    console.print("  3. Run: toklog doctor (when available)")


@proxy_cli.group("skip-process")
def skip_process_group() -> None:
    """Manage the process skip list (~/.toklog/skip_processes)."""


@skip_process_group.command("add")
@click.argument("pattern")
def skip_process_add(pattern: str) -> None:
    """Add PATTERN to the process skip list (substring match against cmdline)."""
    from pathlib import Path

    skip_file = Path.home() / ".toklog" / "skip_processes"
    skip_file.parent.mkdir(parents=True, exist_ok=True)
    existing = skip_file.read_text(encoding="utf-8").splitlines() if skip_file.exists() else []
    if pattern in existing:
        Console().print(f"[yellow]Already present:[/yellow] {pattern}")
        return
    with open(skip_file, "a", encoding="utf-8") as f:
        f.write(pattern + "\n")
    Console().print(f"[green]Added:[/green] {pattern}")


@skip_process_group.command("remove")
@click.argument("pattern")
def skip_process_remove(pattern: str) -> None:
    """Remove PATTERN from the process skip list."""
    from pathlib import Path

    skip_file = Path.home() / ".toklog" / "skip_processes"
    if not skip_file.exists():
        Console().print("[yellow]Skip list is empty.[/yellow]")
        return
    lines = skip_file.read_text(encoding="utf-8").splitlines()
    if pattern not in lines:
        Console().print(f"[yellow]Not found:[/yellow] {pattern}")
        return
    skip_file.write_text(
        "\n".join(l for l in lines if l != pattern) + "\n", encoding="utf-8"
    )
    Console().print(f"[green]Removed:[/green] {pattern}")


@skip_process_group.command("list")
def skip_process_list() -> None:
    """List all process skip patterns."""
    from pathlib import Path

    skip_file = Path.home() / ".toklog" / "skip_processes"
    console = Console()
    if not skip_file.exists():
        console.print("[dim]No skip patterns configured.[/dim]")
        return
    lines = [
        l for l in skip_file.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    if not lines:
        console.print("[dim]No skip patterns configured.[/dim]")
        return
    for line in lines:
        console.print(f"  {line}")


@proxy_cli.command("uninstall")
def proxy_uninstall() -> None:
    """Stop daemon, remove env vars, remove auto-start."""
    from toklog.proxy.daemon import stop
    from toklog.proxy.setup import remove_autostart, remove_shell_profile

    console = Console()
    stop()
    removed = remove_shell_profile()
    remove_autostart()
    if removed:
        console.print("[green]✓[/green] Proxy uninstalled and env vars removed.")
    else:
        console.print("[yellow]Env vars were not found in shell profile.[/yellow]")
