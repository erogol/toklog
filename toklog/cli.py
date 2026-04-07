"""Click CLI: toklog report, toklog init, toklog tail."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import click
from rich.console import Console

from toklog.classify import load_categories, save_categories
from toklog.logger import get_log_dir, read_logs
from toklog.proxy.cli import proxy_cli
from toklog.report import generate_report, render_json, render_text

log = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """TokLog - LLM token waste detector."""
    pass


cli.add_command(proxy_cli)


@cli.command("setup")
@click.option("--port", type=int, default=None, help="Port for proxy (default: find available).")
@click.option("--auto-start/--no-auto-start", default=None, help="Enable/disable auto-start.")
@click.option("--dry-run", is_flag=True, help="Show what would happen without making changes.")
def setup_command(port: int | None, auto_start: bool | None, dry_run: bool) -> None:
    """Interactive setup wizard for TokLog proxy."""
    from toklog.setup_wizard import run_interactive_setup

    success = run_interactive_setup(
        port_override=port,
        auto_start_override=auto_start,
        dry_run=dry_run,
    )

    if not success:
        sys.exit(1)


@cli.command("doctor")
def doctor_command() -> None:
    """Health check for TokLog proxy installation."""
    from rich.console import Console

    from toklog.doctor import run_doctor_checks

    console = Console()
    console.print("[cyan]Checking TokLog proxy installation...[/cyan]\n")

    sections, fail_count = run_doctor_checks()

    for section_name, checks in sections:
        console.print(f"[bold]{section_name}[/bold]")
        for check in checks:
            if check.status == "pass":
                symbol = "[green]✓[/green]"
            elif check.status == "fail":
                symbol = "[red]✗[/red]"
            else:  # warn
                symbol = "[yellow]⚠[/yellow]"

            console.print(f"  {symbol} {check.message}")
            if check.fix:
                console.print(f"      → Fix: {check.fix}")
        console.print()

    console.print("[bold]Summary[/bold]")
    console.print("─" * 40)
    if fail_count == 0:
        console.print("Status: All checks passed ✓")
        return

    console.print(f"Status: {fail_count} problem(s) found")
    sys.exit(1)


@cli.command()
def init() -> None:
    """Initialize TokLog log directory."""
    log_dir = get_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    click.echo(f"TokLog initialized. Logs will be written to {log_dir}/")


@cli.command()
@click.option(
    "--last",
    default="7d",
    type=click.Choice(["1d", "7d", "30d", "90d", "all"]),
    help="Time range for report.",
)
@click.option(
    "--format",
    "fmt",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def report(last: str, fmt: str) -> None:
    """Generate a waste report from logged LLM calls."""
    _DURATION_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
    now = datetime.now(timezone.utc)
    prior_entries = None
    if last in _DURATION_DAYS:
        n_days = _DURATION_DAYS[last]
        prior_end = now - timedelta(days=n_days)
        prior_start = prior_end - timedelta(days=n_days)
        prior_entries = read_logs(prior_start, prior_end)
    data = generate_report(last=last, prior_entries=prior_entries)
    if fmt == "json":
        render_json(data)
    else:
        render_text(data)

    try:
        from toklog.gain import update_gain

        update_gain(data)
    except Exception:
        pass  # never break the report for gain tracking failures


@cli.group()
def pricing() -> None:
    """Manage pricing data."""
    pass


@pricing.command("refresh")
def pricing_refresh() -> None:
    """Fetch latest pricing from LiteLLM and update local cache."""
    from toklog.pricing import refresh_pricing

    count = refresh_pricing()
    if count >= 0:
        click.echo(f"Pricing updated: {count} models loaded.")
    else:
        click.echo("Failed to fetch pricing data.", err=True)
        sys.exit(1)


@cli.group()
def categories() -> None:
    """Manage use case categories for classification."""
    pass


@categories.command("add")
@click.argument("names", nargs=-1, required=True)
def categories_add(names: tuple) -> None:
    """Add one or more categories (duplicates and empty strings are ignored)."""
    existing = load_categories()
    existing_names = {c["name"] for c in existing}
    added = []
    rejected = []
    for name in names:
        if not name.strip():
            rejected.append(repr(name))
            continue
        if name not in existing_names:
            existing.append({"name": name})
            existing_names.add(name)
            added.append(name)
    save_categories(existing)
    if added:
        click.echo(f"Added: {', '.join(added)}")
    if rejected:
        click.echo(f"Rejected (empty): {', '.join(rejected)}")


@categories.command("list")
def categories_list() -> None:
    """List current categories."""
    cats = load_categories()
    if not cats:
        click.echo("No categories defined.")
        return
    for c in cats:
        click.echo(c["name"])


@categories.command("remove")
@click.argument("name")
def categories_remove(name: str) -> None:
    """Remove a category by name."""
    cats = load_categories()
    new_cats = [c for c in cats if c["name"] != name]
    save_categories(new_cats)
    if len(new_cats) < len(cats):
        click.echo(f"Removed: {name}")
    else:
        click.echo(f"Category not found: {name}")


@categories.command("clear")
def categories_clear() -> None:
    """Remove all categories."""
    save_categories([])
    click.echo("All categories cleared.")


@cli.command()
@click.option("--graph", is_flag=True, help="Show ASCII spend graph (last 30 days).")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def gain(graph: bool, as_json: bool) -> None:
    """Show cumulative waste detected since first install."""
    from toklog.gain import load_gain

    gain_data = load_gain()

    if as_json:
        click.echo(json.dumps(gain_data, indent=2))
        return

    console = Console()
    first = (gain_data.get("first_seen") or "unknown")[:10]
    total_waste = gain_data.get("total_waste_detected_usd", 0.0)
    total_spend = gain_data.get("total_spend_observed_usd", 0.0)
    total_calls = gain_data.get("total_calls_observed", 0)
    pct = (total_waste / total_spend * 100) if total_spend > 0 else 0

    console.print(f"\n[bold]TokLog Savings[/bold] (since {first})")
    console.print(
        f"  Waste opportunity found: [bold green]${total_waste:.2f}[/bold green]"
        f" ({pct:.1f}% of ${total_spend:.2f} observed)"
    )
    console.print(f"  Total calls observed: {total_calls:,}")

    if graph:
        snapshots = gain_data.get("snapshots", [])[-30:]
        if snapshots:
            _render_ascii_graph(console, snapshots)

    by_det = gain_data.get("by_detector", {})
    if by_det:
        console.print("\n  [dim]By detector:[/dim]")
        for det, amt in sorted(by_det.items(), key=lambda x: -x[1]):
            console.print(f"    {det}: ${amt:.2f}")

    if not gain_data:
        console.print(
            "\n  [dim]No data yet. Run [bold]toklog report[/bold] first.[/dim]"
        )


def _render_ascii_graph(console: Console, snapshots: list) -> None:
    """Render a simple ASCII bar graph of daily waste."""
    if not snapshots:
        return
    max_waste = max((s.get("waste", 0) for s in snapshots), default=0)
    if max_waste == 0:
        return
    height = 6
    console.print("\n  [dim]Daily waste (last 30 days):[/dim]")
    for row in range(height, 0, -1):
        threshold = max_waste * row / height
        line = "  "
        for s in snapshots:
            w = s.get("waste", 0)
            line += "█" if w >= threshold else " "
        console.print(f"[dim]{line}[/dim]")
    labels = "  " + "".join(s["date"][-2:] if i % 5 == 0 else " " for i, s in enumerate(snapshots))
    console.print(f"[dim]{labels}[/dim]")


@cli.command()
@click.option(
    "--last",
    default="7d",
    type=click.Choice(["1d", "7d", "30d", "90d", "all"]),
    help="Time range for report.",
)
@click.option("--output", "-o", default=None, help="Output file path.")
@click.option("--open", "auto_open", is_flag=True, help="Open in browser after generating.")
def share(last: str, output: str, auto_open: bool) -> None:
    """Generate a shareable self-contained HTML report."""
    from toklog.gain import load_gain
    from toklog.report import generate_report
    from toklog.share import generate_html, save_html

    console = Console()
    with console.status("Generating report..."):
        report_data = generate_report(last=last)
        gain_data = load_gain()
        html = generate_html(report_data, gain_data)
        out_path = save_html(html, output)

    console.print(f"✓ Report saved → [bold]{out_path}[/bold]")
    console.print("  Share it: open the .html file in any browser, no server needed.")

    if auto_open:
        import webbrowser

        webbrowser.open(f"file://{out_path}")


@cli.command()
def reset() -> None:
    """Delete all log data (fresh start)."""
    log_dir = get_log_dir()
    if not os.path.isdir(log_dir):
        click.echo("No log directory found — nothing to reset.")
        return

    files = sorted(f for f in os.listdir(log_dir) if f.endswith(".jsonl"))
    if not files:
        click.echo("No log files found — nothing to reset.")
        return

    click.echo(f"This will permanently delete {len(files)} log file(s) in {log_dir}/:")
    for f in files:
        path = os.path.join(log_dir, f)
        size = os.path.getsize(path)
        click.echo(f"  {f}  ({size:,} bytes)")

    if not click.confirm("\nAre you sure you want to delete all log data?", default=False):
        click.echo("Aborted.")
        return

    for f in files:
        os.remove(os.path.join(log_dir, f))
    click.echo(f"Deleted {len(files)} file(s). Log data has been reset.")


# ---------------------------------------------------------------------------
# Real-time spike tracker for tl tail
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0–100) using linear interpolation.

    Shared between :class:`TailSpikeTracker` and ``detectors.detect_cost_spike``.
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


# Maximum costs to keep per session — sliding window to bound memory.
_SPIKE_WINDOW = 200


class TailSpikeTracker:
    """Rolling per-session cost anomaly detector for live tail output.

    Maintains a sliding window of costs per session hash and uses Tukey
    fences (Q3 + k*IQR) to flag outliers — same math as the report-time
    ``detect_cost_spike`` detector.

    Usage::

        tracker = TailSpikeTracker()
        is_spike, info = tracker.check(cost, session_hash)
        if is_spike:
            print(f"SPIKE! {info['multiplier']:.1f}x median")
    """

    def __init__(self, *, min_sample: int = 5, fence_k: float = 3.0,
                 window: int = _SPIKE_WINDOW) -> None:
        self._min_sample = min_sample
        self._fence_k = fence_k
        self._window = window
        # session_hash → list of non-zero costs (bounded by window)
        self._sessions: dict[str | None, list[float]] = {}
        self.spike_count: int = 0

    def check(self, cost: float, session_hash: str | None) -> tuple[bool, dict]:
        """Record a cost and check whether it's a spike.

        Returns ``(is_spike, info)`` where *info* contains ``multiplier``,
        ``median``, and ``fence`` when a spike is detected, or an empty dict
        otherwise.

        Zero-cost entries (unknown models) are ignored for baseline
        computation but still checked.
        """
        key = session_hash or None

        # Initialize session bucket if needed
        if key not in self._sessions:
            self._sessions[key] = []

        costs = self._sessions[key]

        # Not enough data yet — record and move on
        if len(costs) < self._min_sample:
            if cost > 0:
                costs.append(cost)
            return False, {}

        # Compute Tukey fence from accumulated costs
        q1 = _percentile(costs, 25)
        q3 = _percentile(costs, 75)
        iqr = q3 - q1
        fence = q3 + self._fence_k * iqr
        median = _percentile(costs, 50)

        # Record the cost AFTER computing the fence (don't let the spike
        # pollute its own baseline on the first check)
        if cost > 0:
            costs.append(cost)
            # Sliding window — drop oldest to bound memory
            if len(costs) > self._window:
                costs[:] = costs[-self._window:]

        # Check if this cost exceeds the fence
        if cost > fence and fence > 0:
            multiplier = round(cost / median, 1) if median > 0 else 0.0
            self.spike_count += 1
            return True, {
                "multiplier": multiplier,
                "median": round(median, 4),
                "fence": round(fence, 4),
            }

        return False, {}


def _compute_entry_cost(entry: dict) -> float:
    """Compute cost for a single log entry. Returns 0.0 on failure."""
    from toklog.pricing import compute_cost_components

    try:
        components = compute_cost_components(
            provider=entry.get("provider", ""),
            model=entry.get("model", ""),
            input_tokens=entry.get("input_tokens") or 0,
            output_tokens=entry.get("output_tokens") or 0,
            cache_read=entry.get("cache_read_tokens") or 0,
            cache_creation=entry.get("cache_creation_tokens") or 0,
        )
        return sum(components.values())
    except Exception:
        log.debug("Failed to compute cost for %s", entry.get("model", "?"), exc_info=True)
        return 0.0


def _format_tail_line(entry: dict, *, cost: float | None = None) -> str:
    """Format a single JSONL log entry for tail display.

    Cost is always shown. Pass a precomputed ``cost`` to avoid redundant
    pricing lookups; when *None* the cost is computed from the entry.

    Returns a string like::

        claude-haiku-3-5 in=1000 out=200 $0.0016 500ms myapp
    """
    model = entry.get("model", "?")
    tokens_in = entry.get("input_tokens") or 0
    tokens_out = entry.get("output_tokens") or 0
    dur = entry.get("duration_ms") or 0
    raw_tags = entry.get("tags")
    tag = " ".join(raw_tags) if isinstance(raw_tags, list) else (str(raw_tags) if raw_tags else "")

    if cost is None:
        cost = _compute_entry_cost(entry)

    parts: list[str] = [f"in={tokens_in}", f"out={tokens_out}", f"${cost:.4f}", f"{dur}ms"]
    if tag:
        parts.append(tag)

    return f"{model} " + " ".join(parts)


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration like '5m 32s' or '1h 12m'."""
    s = max(int(seconds), 0)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _render_tail_header(total_cost: float, req_count: int, elapsed: float,
                        spike_count: int = 0) -> str:
    """Render the sticky header line for tl tail.

    Returns a plain string (no Rich markup) suitable for any output target.
    """
    cost_min = total_cost / (elapsed / 60) if elapsed >= 1 else 0.0
    cost_req = total_cost / req_count if req_count > 0 else 0.0
    dur = _format_duration(elapsed)

    parts = [
        f"${total_cost:.2f}",
        f"{req_count} reqs",
        f"${cost_min:.2f}/min",
        f"${cost_req:.2f}/req",
        dur,
    ]
    if spike_count > 0:
        parts.append(f"🔺{spike_count} spike{'s' if spike_count != 1 else ''}")

    return " │ ".join(parts)


def _tail_current_date() -> datetime:
    """Return current UTC datetime. Extracted for test mocking (midnight rollover)."""
    return datetime.now(timezone.utc)


# Header update interval (seconds) when idle — avoids pointless redraws.
_HEADER_IDLE_INTERVAL = 5.0


@cli.command()
@click.option("--cost", "show_cost", is_flag=True, default=False, hidden=True,
              help="Deprecated — cost is always shown.")
def tail(show_cost: bool) -> None:
    """Live tail of today's JSONL log file with running cost totals."""
    console = Console()
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    log_dir_path = get_log_dir()

    def _log_filepath() -> str:
        date_str = _tail_current_date().strftime("%Y-%m-%d")
        return os.path.join(log_dir_path, f"{date_str}.jsonl")

    filepath = _log_filepath()

    if not os.path.exists(filepath):
        console.print(f"[yellow]No log file for today. Waiting...[/yellow]")

    console.print(f"[dim]Tailing {filepath} (Ctrl+C to stop)[/dim]")

    # Running totals since tail started
    total_cost = 0.0
    req_count = 0
    start_time = time.monotonic()
    last_header_time = 0.0
    tracker = TailSpikeTracker()

    def _write_header(force: bool = False) -> None:
        nonlocal last_header_time
        now = time.monotonic()
        if not force and (now - last_header_time) < _HEADER_IDLE_INTERVAL:
            return
        last_header_time = now
        elapsed = now - start_time
        header = _render_tail_header(total_cost, req_count, elapsed,
                                     spike_count=tracker.spike_count)
        if is_tty:
            # Save cursor, move to row 1 col 1, clear line, print, restore cursor
            sys.stdout.write(f"\033[s\033[1;1H\033[2K{header}\033[u")
            sys.stdout.flush()
        # Non-TTY: header printed only in the final Stopped line

    # Reserve the header line in TTY mode
    if is_tty:
        sys.stdout.write("\n")
        sys.stdout.flush()

    try:
        pos = 0
        if os.path.exists(filepath):
            pos = os.path.getsize(filepath)

        _write_header(force=True)

        while True:
            # Midnight rollover: recompute filepath each iteration
            new_filepath = _log_filepath()
            if new_filepath != filepath:
                filepath = new_filepath
                pos = 0

            if not os.path.exists(filepath):
                time.sleep(1)
                _write_header()
                continue

            size = os.path.getsize(filepath)
            # File truncated (shouldn't happen, but be safe)
            if size < pos:
                pos = 0

            if size > pos:
                with open(filepath, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for raw_line in f:
                        raw_line = raw_line.strip()
                        if raw_line:
                            try:
                                entry = json.loads(raw_line)
                                err = entry.get("error", False)
                                status = "[red]ERR[/red]" if err else "[green]OK[/green]"

                                cost = _compute_entry_cost(entry)
                                total_cost += cost
                                req_count += 1

                                session_hash = entry.get("system_prompt_hash")
                                is_spike, spike_info = tracker.check(cost, session_hash)

                                line = _format_tail_line(entry, cost=cost)
                                if is_spike:
                                    mult = spike_info.get("multiplier", 0)
                                    console.print(
                                        f"🔺{status} [bold red]{line} ({mult}x median)[/bold red]"
                                    )
                                else:
                                    console.print(f"{status} {line}")

                                _write_header(force=True)
                            except json.JSONDecodeError:
                                console.print(f"[dim]{raw_line}[/dim]")
                    pos = f.tell()

            _write_header()
            time.sleep(0.5)
    except KeyboardInterrupt:
        elapsed = time.monotonic() - start_time
        header = _render_tail_header(total_cost, req_count, elapsed,
                                     spike_count=tracker.spike_count)
        console.print(f"\n[dim]Stopped. {header}[/dim]")
