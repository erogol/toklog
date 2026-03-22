"""Tests for toklog.classify module — BM25+ classifier."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

import toklog.classify as classify_mod
from toklog.classify import (
    BM25PlusClassifier,
    DEFAULT_CATEGORIES,
    CATEGORY_KEYWORDS,
    _classify_single,
    classify_entries,
    classify_entries_async,
    load_categories,
    save_categories,
    simple_stem,
    tokenize,
)


class TestLoadSave:
    def test_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cats = [{"name": "search", "description": "search queries"}]
        monkeypatch.setattr(classify_mod, "_CATEGORIES_FILE", tmp_path / "cats.json")
        save_categories(cats)
        loaded = load_categories()
        assert loaded == cats

    def test_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(classify_mod, "_CATEGORIES_FILE", tmp_path / "missing.json")
        assert load_categories() == []

    def test_creates_parent_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = tmp_path / "nested" / "dir" / "cats.json"
        monkeypatch.setattr(classify_mod, "_CATEGORIES_FILE", f)
        save_categories([])
        assert f.exists()


class TestTokenizer:
    def test_tokenize_basic(self) -> None:
        tokens = tokenize("Hello World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_tokenize_empty(self) -> None:
        assert tokenize("") == []

    def test_simple_stem(self) -> None:
        assert simple_stem("running") == "runn"
        assert simple_stem("implementation") == "implementa"
        assert simple_stem("go") == "go"  # too short to stem


class TestBM25PlusClassifier:
    def test_init_builds_docs(self) -> None:
        clf = BM25PlusClassifier(DEFAULT_CATEGORIES, CATEGORY_KEYWORDS)
        assert len(clf.cat_names) == 6
        assert len(clf.cat_docs_stemmed) == 6

    def test_classify_returns_string(self) -> None:
        clf = BM25PlusClassifier(DEFAULT_CATEGORIES, CATEGORY_KEYWORDS)
        result = clf.classify("Write a Python function to validate emails")
        assert isinstance(result, str)
        assert result in [c["name"] for c in DEFAULT_CATEGORIES] + ["uncategorized"]


class TestClassifySingle:
    def test_classifies_code_generation(self) -> None:
        result = _classify_single("Write a Python function to validate emails", DEFAULT_CATEGORIES)
        assert result == "code_generation"

    def test_classifies_summarization(self) -> None:
        result = _classify_single("Summarize this meeting transcript", DEFAULT_CATEGORIES)
        assert result == "summarization"

    def test_classifies_chat(self) -> None:
        result = _classify_single("Tell me a joke", DEFAULT_CATEGORIES)
        assert result == "chat"

    def test_classifies_research(self) -> None:
        result = _classify_single("Compare PostgreSQL vs MongoDB", DEFAULT_CATEGORIES)
        assert result == "research"

    def test_classifies_data_extraction(self) -> None:
        result = _classify_single("Extract all email addresses from this text", DEFAULT_CATEGORIES)
        assert result == "data_extraction"

    def test_classifies_tool_use(self) -> None:
        result = _classify_single("Send an email to the team about the meeting", DEFAULT_CATEGORIES)
        assert result == "tool_use"

    def test_first_verb_detection(self) -> None:
        result = _classify_single("Implement a binary search tree", DEFAULT_CATEGORIES)
        assert result == "code_generation"

    def test_first_verb_summarize(self) -> None:
        result = _classify_single("Summarize the key findings from this report", DEFAULT_CATEGORIES)
        assert result == "summarization"


class TestClassifyEntries:
    def _entries(self, previews: List[Any]) -> List[Dict[str, Any]]:
        return [{"user_message_preview": p} for p in previews]

    def test_bm25_grouping(self) -> None:
        cats = [
            {"name": "code", "description": "python programming software code"},
            {"name": "story", "description": "fiction novel story narrative"},
        ]
        entries = self._entries(["write python code function", "write a fiction story"])
        costs = [0.01, 0.01]
        result = classify_entries(entries, cats, costs)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_no_previews_returns_uncategorized(self) -> None:
        cats = [{"name": "test", "description": "test"}]
        entries = [{"user_message_preview": None}, {"input_tokens": 100}]
        costs = [0.01, 0.01]
        result = classify_entries(entries, cats, costs)
        assert len(result) == 1
        assert result[0]["name"] == "(uncategorized)"
        assert result[0]["calls"] == 2
        assert abs(result[0]["cost_usd"] - 0.02) < 1e-9
        assert result[0]["pct"] == 100.0

    def test_no_categories_uses_defaults(self) -> None:
        entries = self._entries(["Write a Python function"])
        costs = [0.05]
        result = classify_entries(entries, [], costs)
        assert len(result) > 0
        names = [r["name"] for r in result]
        assert "code_generation" in names

    def test_default_categories_work(self) -> None:
        entries = self._entries([
            "Write a Python function to validate emails",
            "Summarize this meeting transcript",
            "Tell me a joke",
            "Compare PostgreSQL vs MongoDB",
            "Extract all email addresses from this text",
            "Send an email to the team",
        ])
        costs = [0.01] * 6
        result = classify_entries(entries, [], costs)
        assert len(result) > 0
        names = {r["name"] for r in result}
        # At least several categories should be represented
        assert len(names) >= 3

    def test_cost_aggregation(self) -> None:
        entries = self._entries([
            "Write a Python class",
            "Implement a REST API endpoint",
        ])
        costs = [0.10, 0.20]
        result = classify_entries(entries, DEFAULT_CATEGORIES, costs)
        total_cost = sum(r["cost_usd"] for r in result)
        assert abs(total_cost - 0.30) < 0.001

    def test_pct_sums_to_100(self) -> None:
        entries = self._entries([
            "Write a function",
            "Summarize this document",
            "Tell me a joke",
        ])
        costs = [0.10, 0.20, 0.30]
        result = classify_entries(entries, DEFAULT_CATEGORIES, costs)
        total_pct = sum(r["pct"] for r in result)
        assert abs(total_pct - 100.0) < 1.0  # rounding tolerance

    def test_sorted_by_cost_desc(self) -> None:
        entries = self._entries(["Tell me a joke", "Write a function"])
        costs = [0.50, 0.01]
        result = classify_entries(entries, DEFAULT_CATEGORIES, costs)
        costs_out = [r["cost_usd"] for r in result]
        assert costs_out == sorted(costs_out, reverse=True)


class TestClassifyEntriesAsync:
    @pytest.mark.asyncio
    async def test_async_returns_same_as_sync(self) -> None:
        entries = [
            {"user_message_preview": "Write a Python function to validate emails"},
            {"user_message_preview": "Summarize this meeting transcript"},
        ]
        costs = [0.10, 0.20]
        sync_result = classify_entries(entries, DEFAULT_CATEGORIES, costs)
        async_result = await classify_entries_async(entries, DEFAULT_CATEGORIES, costs)
        assert sync_result == async_result

    @pytest.mark.asyncio
    async def test_async_with_empty_categories(self) -> None:
        entries = [{"user_message_preview": "Write a Python function"}]
        costs = [0.05]
        result = await classify_entries_async(entries, [], costs)
        assert len(result) > 0
