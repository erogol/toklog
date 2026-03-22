"""toklog/share.py — Generate self-contained HTML report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_REPORTS_DIR = Path.home() / ".toklog" / "reports"


def generate_html(
    report: Dict[str, Any], gain: Optional[Dict[str, Any]] = None
) -> str:
    """Render report data as self-contained HTML string."""
    try:
        from jinja2 import Environment, PackageLoader

        env = Environment(loader=PackageLoader("toklog", "templates"))
        template = env.get_template("report.html.j2")
    except ImportError:
        raise RuntimeError("jinja2 required: pip install toklog[share]")

    return template.render(
        report=report,
        gain=gain or {},
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        report_json=json.dumps(report),
        gain_json=json.dumps(gain or {}),
    )


def save_html(html: str, filename: Optional[str] = None) -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filename = f"{date_str}.html"
    out = _REPORTS_DIR / filename
    out.write_text(html, encoding="utf-8")
    return out
