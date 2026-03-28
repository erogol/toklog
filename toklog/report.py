"""Report data aggregation and rendering."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table

from toklog.context_drivers import aggregate_context_drivers
from toklog.detectors import run_all
from toklog.logger import _is_benchmark_entry, read_logs
from toklog.pricing import _normalize_model_name, compute_cost_components
from toklog.proxy.budget import load_status_from_file

_PERIOD_LABELS: Dict[str, str] = {
    "1d": "Last 1 Day",
    "7d": "Last 7 Days",
    "30d": "Last 30 Days",
    "90d": "Last 90 Days",
    "all": "All Time",
}

_DRIVER_DISPLAY: Dict[str, str] = {
    "cache_read": "Cache Reads",
    "cache_write": "Cache Writes",
    "system_prompt": "System Prompt",
    "tool_outputs": "Tool Outputs",
    "tool_schemas": "Tool Schemas",
    "thinking_input": "Thinking Input",
    "thinking": "Thinking",
    "tool_calls": "Tool Call JSON",
    "output_code": "Output Code",
    "output_text": "Output Text",
    "conversation": "Conversation",
    "unattributed": "Unattributed",
    "minimal": "Minimal",
    "code": "Code",
    "structured_data": "Structured Data",
    "prose": "Prose",
}


def _fmt_usd(amount: float) -> str:
    """Format a USD amount: 4 decimal places for sub-cent amounts, 2 otherwise."""
    if amount == 0.0:
        return "$0.00"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.2f}"


def _parse_duration(duration: str) -> Optional[datetime]:
    """Parse duration string like '7d', '30d' into a start datetime."""
    if duration == "all":
        return None
    unit = duration[-1]
    try:
        value = int(duration[:-1])
    except ValueError:
        return None
    if unit == "d":
        return datetime.now(timezone.utc) - timedelta(days=value)
    return None


def _classify_key_hint(hint: str) -> str:
    """Classify an api_key_hint as 'api key', 'tag', or 'unknown'."""
    if hint == "(unset)":
        return "unknown"
    if hint.startswith("[") and hint.endswith("]"):
        return "tag"
    return "api key"


def _compute_tokens(entry: dict) -> int:
    """Total tokens for an entry: input + output (excluding cache tokens)."""
    return (entry.get("input_tokens") or 0) + (entry.get("output_tokens") or 0)


def _compute_cost_components(entry: Dict[str, Any]) -> Dict[str, float]:
    """Break down the cost of one entry into per-driver components.

    Returns dict with keys: input, cache_read, cache_write, output.
    Returns empty dict when the model is unknown (cost cannot be computed).
    """
    return compute_cost_components(
        provider=entry.get("provider", ""),
        model=entry.get("model", ""),
        input_tokens=entry.get("input_tokens") or 0,
        output_tokens=entry.get("output_tokens") or 0,
        cache_read=entry.get("cache_read_tokens") or 0,
        cache_creation=entry.get("cache_creation_tokens") or 0,
    )


def _compute_cost(entry: Dict[str, Any]) -> float:
    """Compute cost for a single log entry, accounting for cache tokens."""
    return sum(_compute_cost_components(entry).values())


def compute_spend_trend(
    current_entries: List[Dict[str, Any]],
    prior_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute spend trend comparing current period to prior period."""
    current_usd = round(sum(_compute_cost(e) for e in current_entries), 4)
    prior_usd = round(sum(_compute_cost(e) for e in prior_entries), 4)
    change_usd = round(current_usd - prior_usd, 4)
    change_pct = round((change_usd / prior_usd) * 100, 1) if prior_usd > 0 else None

    if change_pct is None:
        direction = "up" if current_usd > 0 else "flat"
    elif change_pct > 5:
        direction = "up"
    elif change_pct < -5:
        direction = "down"
    else:
        direction = "flat"

    return {
        "current_usd": current_usd,
        "prior_usd": prior_usd,
        "change_usd": change_usd,
        "change_pct": change_pct,
        "direction": direction,
    }


def generate_report(
    last: str = "7d",
    prior_entries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate report data dict."""
    start_date = _parse_duration(last)
    end_date = datetime.now(timezone.utc)
    entries = read_logs(start_date=start_date, end_date=end_date)
    if prior_entries is not None:
        prior_entries = [e for e in prior_entries if not _is_benchmark_entry(e)]

    # Run detectors
    detector_results = run_all(entries)

    # Compute aggregates
    total_calls = len(entries)
    budget_rejections = sum(1 for e in entries if e.get("budget_rejected") is True)
    total_spend = sum(_compute_cost(e) for e in entries)
    total_waste = sum(d.estimated_waste_usd for d in detector_results if d.triggered)

    # Cost by model
    cost_by_model: Dict[str, float] = defaultdict(float)
    calls_by_model: Dict[str, int] = defaultdict(int)
    tokens_by_model: Dict[str, int] = defaultdict(int)
    for e in entries:
        model = _normalize_model_name(e.get("model", "unknown"))
        cost_by_model[model] += _compute_cost(e)
        calls_by_model[model] += 1
        tokens_by_model[model] += _compute_tokens(e)

    # Cost by process
    cost_by_process_raw: Dict[str, Dict] = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0, "tokens": 0})
    for e in entries:
        prog = e.get("program") or e.get("tags") or "(unknown)"
        cost_by_process_raw[prog]["calls"] += 1
        cost_by_process_raw[prog]["cost_usd"] += _compute_cost(e)
        cost_by_process_raw[prog]["tokens"] += _compute_tokens(e)

    # Cost by call site
    _DYNAMIC_FILES = {"<string>", "<stdin>", ""}
    cost_by_call_site_raw: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0, "tokens": 0})
    for e in entries:
        cs = e.get("call_site")
        tag = e.get("tags") or ""

        if cs is None or cs.get("file") in _DYNAMIC_FILES:
            key = f"[{tag}]" if tag else "<unknown>"
        else:
            key = f"{cs['file']}:{cs['function']}:{cs['line']}"

        cost_by_call_site_raw[key]["calls"] += 1
        cost_by_call_site_raw[key]["cost_usd"] += _compute_cost(e)
        cost_by_call_site_raw[key]["tokens"] += _compute_tokens(e)

    # Cost by API key
    cost_by_key_raw: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0})
    for e in entries:
        key_hint = e.get("api_key_hint") or "(unset)"
        cost_by_key_raw[key_hint]["calls"] += 1
        cost_by_key_raw[key_hint]["cost_usd"] += _compute_cost(e)
    # Threshold: a key is significant if cost >= max($0.01, 0.1% of total spend)
    significance_threshold = max(0.01, 0.001 * total_spend) if total_spend > 0 else 0.01
    cost_by_key = []
    pruned_keys_count = 0
    pruned_keys_cost = 0.0
    for k, v in sorted(cost_by_key_raw.items(), key=lambda x: -x[1]["cost_usd"]):
        row = {
            "key_hint": k,
            "type": _classify_key_hint(k),
            "calls": v["calls"],
            "cost_usd": round(float(v["cost_usd"]), 4),
            "pct": round(v["cost_usd"] / total_spend * 100, 1) if total_spend > 0 else 0.0,
        }
        if v["cost_usd"] >= significance_threshold:
            cost_by_key.append(row)
        else:
            pruned_keys_count += 1
            pruned_keys_cost += v["cost_usd"]
    pruned_keys_summary = None
    if pruned_keys_count > 0:
        pruned_keys_summary = {
            "count": pruned_keys_count,
            "total_cost_usd": round(pruned_keys_cost, 4),
            "pct": round(pruned_keys_cost / total_spend * 100, 1) if total_spend > 0 else 0.0,
        }

    sorted_call_sites = sorted(cost_by_call_site_raw.items(), key=lambda x: -x[1]["cost_usd"])[:15]
    cost_by_call_site = [
        {
            "call_site": k,
            "calls": v["calls"],
            "cost_usd": round(float(v["cost_usd"]), 4),
            "tokens": v["tokens"],
            "pct": round(v["cost_usd"] / total_spend * 100, 1) if total_spend > 0 else 0.0,
        }
        for k, v in sorted_call_sites
    ]

    # Error call aggregation
    # An error entry is "likely billed" when the API returned token counts before failing.
    # Entries with zero tokens failed pre-call (connection error, auth failure, etc.) — not billed.
    error_calls = sum(1 for e in entries if e.get("error") is True)
    errors_by_type_raw: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        if e.get("error") is not True:
            continue
        etype = e.get("error_type") or "UnknownError"
        model = _normalize_model_name(e.get("model") or "unknown")
        ts = e.get("timestamp") or ""
        likely_billed = (e.get("input_tokens") or 0) > 0 or (e.get("output_tokens") or 0) > 0
        if etype not in errors_by_type_raw:
            errors_by_type_raw[etype] = {
                "count": 0,
                "billed_count": 0,
                "unbilled_count": 0,
                "wasted_usd": 0.0,
                "models": set(),
                "last_seen": "",
            }
        errors_by_type_raw[etype]["count"] += 1
        if likely_billed:
            errors_by_type_raw[etype]["billed_count"] += 1
            errors_by_type_raw[etype]["wasted_usd"] += _compute_cost(e)
        else:
            errors_by_type_raw[etype]["unbilled_count"] += 1
        errors_by_type_raw[etype]["models"].add(model)
        if ts > errors_by_type_raw[etype]["last_seen"]:
            errors_by_type_raw[etype]["last_seen"] = ts
    errors_by_type = [
        {
            "error_type": etype,
            "count": v["count"],
            "billed_count": v["billed_count"],
            "unbilled_count": v["unbilled_count"],
            "wasted_usd": round(v["wasted_usd"], 4),
            "models": sorted(v["models"]),
            "last_seen": v["last_seen"],
        }
        for etype, v in sorted(errors_by_type_raw.items(), key=lambda x: -x[1]["count"])
    ]
    error_wasted_usd = round(sum(r["wasted_usd"] for r in errors_by_type), 4)

    # Cost by context driver (composition)
    # Compute component costs once; derive total cost from them to avoid redundant pricing lookups.
    entry_component_costs_list = [_compute_cost_components(e) for e in entries]
    entry_costs_list = [sum(c.values()) for c in entry_component_costs_list]
    entry_tokens_list = [_compute_tokens(e) for e in entries]
    cost_by_context_driver: List[Dict[str, Any]] = []
    try:
        cost_by_context_driver = aggregate_context_drivers(
            entries,
            entry_costs_list,
            entry_tokens=entry_tokens_list,
            entry_component_costs=entry_component_costs_list,
        )
    except Exception:
        cost_by_context_driver = []

    return {
        "period": last,
        "total_calls": total_calls,
        "total_spend_usd": round(total_spend, 4),
        "estimated_waste_usd": round(total_waste, 4),
        "waste_pct": round((total_waste / total_spend * 100) if total_spend > 0 else 0, 1),
        "detectors": [
            {
                "name": d.name,
                "triggered": d.triggered,
                "severity": d.severity,
                "estimated_waste_usd": d.estimated_waste_usd,
                "description": d.description,
                "details": d.details,
            }
            for d in detector_results
        ],
        "cost_by_model": {
            m: {"cost_usd": round(c, 4), "calls": calls_by_model[m], "tokens": tokens_by_model[m]}
            for m, c in sorted(cost_by_model.items(), key=lambda x: -x[1])
        },
        "cost_by_process": [
            {
                "process": prog,
                "calls": v["calls"],
                "tokens": v["tokens"],
                "avg_tokens": v["tokens"] // v["calls"] if v["calls"] > 0 else 0,
                "cost_usd": round(float(v["cost_usd"]), 4),
                "pct": round(v["cost_usd"] / total_spend * 100, 1) if total_spend > 0 else 0.0,
            }
            for prog, v in sorted(cost_by_process_raw.items(), key=lambda x: -x[1]["cost_usd"])
        ],
        "cost_by_key": cost_by_key,
        "pruned_keys_summary": pruned_keys_summary,
        "cost_by_call_site": cost_by_call_site,
        "cost_by_context_driver": cost_by_context_driver,
        "spend_trend": compute_spend_trend(entries, prior_entries) if prior_entries is not None else None,
        "error_calls": error_calls,
        "error_wasted_usd": error_wasted_usd,
        "errors_by_type": errors_by_type,
        "budget_rejections": budget_rejections,
        "budget": load_status_from_file(),
    }


def render_text(report: Dict[str, Any], console: Optional[Console] = None) -> None:
    """Render report as rich text to console."""
    if console is None:
        console = Console()

    period_label = _PERIOD_LABELS.get(report["period"], report["period"])

    # Zero-data early return
    if report["total_calls"] == 0:
        console.print(
            Panel(
                "No calls logged in this period.\n\n"
                "If this is a new install, run [bold]toklog init[/bold] to set up the log directory.\n"
                "Try a wider time range: [bold]toklog report --last 30d[/bold]",
                title=f"TokLog Report – {period_label}",
                style="dim",
            )
        )
        return

    # Summary panel
    # Separate billed errors (provider charged) from unbilled (auth/connection failures).
    # Only billed errors belong in the headline — unbilled are infra noise.
    error_calls = report.get("error_calls", 0)
    error_wasted_usd = report.get("error_wasted_usd", 0.0)
    total_calls = report["total_calls"]
    errors_by_type = report.get("errors_by_type", [])
    billed_errors = sum(e.get("billed_count", 0) for e in errors_by_type)
    unbilled_errors = sum(e.get("unbilled_count", 0) for e in errors_by_type)
    successful_calls = total_calls - error_calls

    error_rate_str = ""
    if billed_errors > 0:
        error_rate_str = f"\nBilled failures: {billed_errors} ({_fmt_usd(error_wasted_usd)} wasted)"
    if unbilled_errors > 0:
        error_rate_str += f"\nUnbilled failures: {unbilled_errors} (auth/connection — not charged)"

    summary = (
        f"Total calls: {total_calls} ({successful_calls} successful)\n"
        f"Total spend: {_fmt_usd(report['total_spend_usd'])}\n"
        f"Estimated waste: {_fmt_usd(report['estimated_waste_usd'])}\n"
        f"Waste: {report['waste_pct']}%"
        f"{error_rate_str}"
    )
    console.print(Panel(summary, title=f"TokLog Report – {period_label}"))

    # Budget bar — only when enforcement is active
    budget = report.get("budget")
    if budget is not None and budget.get("enforcing"):
        limit = budget.get("limit_usd", 0)
        spent = budget.get("daily_spend", 0)
        pct = (spent / limit * 100) if limit > 0 else 0
        filled = int(round(pct / 10))
        filled = min(filled, 10)
        bar = "█" * filled + "░" * (10 - filled)
        if pct > 90:
            color = "red"
        elif pct >= 75:
            color = "yellow"
        else:
            color = "green"
        console.print(
            f"[{color}]Budget: {_fmt_usd(spent)} / {_fmt_usd(limit)} {bar} {pct:.0f}%[/{color}]"
        )

    # Rejection warning
    budget_rejections = report.get("budget_rejections", 0)
    if budget_rejections > 0:
        console.print(
            f"[bold yellow]⚠ {budget_rejections} request{'s' if budget_rejections != 1 else ''}"
            f" {'were' if budget_rejections != 1 else 'was'} blocked by budget enforcement during this period.[/bold yellow]"
        )

    trend = report.get("spend_trend")
    if trend is not None:
        emoji = {"up": "📈", "down": "📉", "flat": "➡️"}.get(trend["direction"], "➡️")
        pct_str = f" ({trend['change_pct']:+.1f}%)" if trend["change_pct"] is not None else ""
        console.print(
            f"{emoji} Spend trend: {_fmt_usd(trend['prior_usd'])} → {_fmt_usd(trend['current_usd'])}"
            f" (Δ${trend['change_usd']:+.4f}{pct_str})"
        )

    # Failed calls — split into billed (cost you money) and unbilled (infra noise)
    billed_rows = [r for r in errors_by_type if r.get("billed_count", 0) > 0]
    unbilled_rows = [r for r in errors_by_type if r.get("unbilled_count", 0) > 0]

    if billed_rows:
        err_table = Table(title=f"Billed Failures ({billed_errors})")
        err_table.add_column("Error Type", style="red")
        err_table.add_column("Count", justify="right")
        err_table.add_column("Wasted $", justify="right")
        err_table.add_column("Models")
        err_table.add_column("Last Seen")
        for row in billed_rows:
            err_table.add_row(
                row["error_type"],
                str(row["billed_count"]),
                _fmt_usd(row.get("wasted_usd", 0.0)),
                ", ".join(row["models"]),
                row["last_seen"][:19].replace("T", " ") if row["last_seen"] else "-",
            )
        console.print(err_table)

    if unbilled_rows:
        # Compact summary for unbilled errors — these didn't cost anything
        unbilled_summary = ", ".join(
            f"{r['error_type']} ×{r['unbilled_count']}" for r in unbilled_rows
        )
        console.print(
            f"[dim]Unbilled failures ({unbilled_errors}): {unbilled_summary}[/dim]"
        )

    # Detector results — sorted by dollar impact descending
    triggered = [d for d in report["detectors"] if d["triggered"]]
    triggered.sort(key=lambda d: d["estimated_waste_usd"], reverse=True)
    if triggered:
        table = Table(title="Waste Detectors")
        table.add_column("Waste $", justify="right", style="bold")
        table.add_column("Detector")
        table.add_column("Description")
        for d in triggered:
            table.add_row(
                _fmt_usd(d["estimated_waste_usd"]),
                d["name"],
                d["description"],
            )
        console.print(table)
    else:
        console.print("[green]No waste patterns detected.[/green]")

    # Context composition — shown before models to explain where tokens go
    context_driver_rows = report.get("cost_by_context_driver", [])
    if context_driver_rows:
        cd_table = Table(title="Context Composition")
        cd_table.add_column("Driver")
        cd_table.add_column("Calls", justify="right")
        cd_table.add_column("Tokens", justify="right")
        cd_table.add_column("Avg Tokens", justify="right")
        cd_table.add_column("Cost", justify="right")
        cd_table.add_column("%", justify="right")
        for row in context_driver_rows:
            display_name = _DRIVER_DISPLAY.get(row["name"], row["name"].replace("_", " ").title())
            cd_table.add_row(
                display_name,
                str(row["calls"]),
                f"{row.get('tokens', 0):,}",
                f"{row.get('avg_tokens', 0):,}",
                _fmt_usd(row["cost_usd"]),
                f"{row['pct']:.1f}%",
            )
        console.print(cd_table)

    # Top models
    cost_by_model = report.get("cost_by_model", {})
    if cost_by_model:
        model_table = Table(title="Top Models by Spend")
        model_table.add_column("Model")
        model_table.add_column("Calls", justify="right")
        model_table.add_column("Tokens", justify="right")
        model_table.add_column("Cost", justify="right")
        for model, data in cost_by_model.items():
            model_table.add_row(model, str(data["calls"]), f"{data.get('tokens', 0):,}", _fmt_usd(data["cost_usd"]))
        console.print(model_table)

    # Cost by process
    cost_by_process = report.get("cost_by_process", [])
    if cost_by_process:
        prog_table = Table(title="Cost by Process")
        prog_table.add_column("Process")
        prog_table.add_column("Calls", justify="right")
        prog_table.add_column("Tokens", justify="right")
        prog_table.add_column("Avg/Call", justify="right")
        prog_table.add_column("Cost", justify="right")
        prog_table.add_column("%", justify="right")
        for row in cost_by_process:
            avg = f"{row['avg_tokens']:,}" if row["avg_tokens"] > 0 else "\u2014"
            prog_table.add_row(
                row["process"],
                str(row["calls"]),
                f"{row['tokens']:,}",
                avg,
                _fmt_usd(row["cost_usd"]),
                f"{row['pct']:.1f}%",
            )
        console.print(prog_table)

    # Cost by API key
    cost_by_key = report.get("cost_by_key", [])
    if len(cost_by_key) > 0:
        key_table = Table(title="Cost by API Key")
        key_table.add_column("API Key")
        key_table.add_column("Type")
        key_table.add_column("Calls", justify="right")
        key_table.add_column("Cost", justify="right")
        key_table.add_column("%", justify="right")
        for row in cost_by_key:
            key_table.add_row(rich_escape(row["key_hint"]), row["type"], str(row["calls"]), _fmt_usd(row["cost_usd"]), f"{row.get('pct', 0.0):.1f}%")
        console.print(key_table)
        pruned = report.get("pruned_keys_summary")
        if pruned:
            console.print(
                f"  [dim]{pruned['count']} more key{'s' if pruned['count'] != 1 else ''}"
                f" with negligible spend ({_fmt_usd(pruned['total_cost_usd'])} total, <0.1%)[/dim]"
            )

    # Cost by call site — only shown when at least one real file-based call site exists
    call_site_rows = report.get("cost_by_call_site", [])
    if any(not r["call_site"].startswith(("<", "[")) for r in call_site_rows):
        cs_table = Table(title="Cost by Call Site")
        cs_table.add_column("Call Site")
        cs_table.add_column("Calls", justify="right")
        cs_table.add_column("Tokens", justify="right")
        cs_table.add_column("Cost", justify="right")
        cs_table.add_column("%", justify="right")
        for row in call_site_rows:
            cs_table.add_row(
                rich_escape(row["call_site"]),
                str(row["calls"]),
                f"{row.get('tokens', 0):,}",
                _fmt_usd(row["cost_usd"]),
                f"{row['pct']:.1f}%",
            )
        console.print(cs_table)


def render_json(report: Dict[str, Any]) -> None:
    """Render report as JSON to stdout."""
    print(json.dumps(report, indent=2))
