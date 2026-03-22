"""Tests for toklog/share.py — HTML report generation."""

from __future__ import annotations

import json
import re
from unittest.mock import patch

import pytest


@pytest.fixture()
def sample_report() -> dict:
    return {
        "period": "7d",
        "total_calls": 150,
        "total_spend_usd": 5.25,
        "estimated_waste_usd": 1.10,
        "waste_pct": 20.9,
        "detectors": [
            {
                "name": "cache_miss_opportunity",
                "triggered": True,
                "severity": "high",
                "estimated_waste_usd": 1.10,
                "description": "Repeated identical prompts without caching.",
                "details": None,
            },
            {
                "name": "tool_schema_bloat",
                "triggered": False,
                "severity": "low",
                "estimated_waste_usd": 0.0,
                "description": "Tool schemas are small.",
                "details": None,
            },
        ],
        "cost_by_tag": {"job:test": 5.25},
        "cost_by_model": {
            "claude-opus-4-6": {"cost_usd": 4.00, "calls": 100, "tokens": 500000},
            "gpt-4o": {"cost_usd": 1.25, "calls": 50, "tokens": 100000},
        },
        "cost_by_call_site": [
            {"call_site": "agent/run.py:main:42", "calls": 100, "tokens": 500000, "cost_usd": 4.00, "pct": 76.2},
        ],
        "cost_by_use_case": [
            {"name": "code_generation", "calls": 100, "tokens": 500000, "cost_usd": 4.00, "pct": 76.2},
        ],
        "spend_trend": None,
    }


@pytest.fixture()
def tmp_reports_dir(tmp_path):
    with patch("toklog.share._REPORTS_DIR", tmp_path):
        yield tmp_path


def test_generate_html_produces_valid_html(sample_report):
    pytest.importorskip("jinja2")
    from toklog.share import generate_html

    html = generate_html(sample_report)
    assert "<!DOCTYPE html>" in html
    assert "TokLog Report" in html
    assert "$5.25" in html
    assert "$1.10" in html


def test_generate_html_embeds_json_data(sample_report):
    pytest.importorskip("jinja2")
    from toklog.share import generate_html

    html = generate_html(sample_report, gain={"total_waste_detected_usd": 9.99})
    assert "const reportData" in html
    assert "const gainData" in html
    # Verify the embedded JSON is valid
    m = re.search(r"const reportData = ({.*?});", html, re.DOTALL)
    assert m is not None
    parsed = json.loads(m.group(1))
    assert parsed["total_calls"] == 150


def test_save_html_writes_file(sample_report, tmp_reports_dir):
    pytest.importorskip("jinja2")
    from toklog.share import generate_html, save_html

    html = generate_html(sample_report)
    out = save_html(html, "test-report.html")

    assert out.exists()
    assert out.suffix == ".html"
    content = out.read_text(encoding="utf-8")
    assert "TokLog" in content
