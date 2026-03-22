"""Tests for compress_history()."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

openai = pytest.importorskip("openai", reason="openai not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def _make_openai_mock(summary_text: str = "A summary.") -> MagicMock:
    choice = MagicMock()
    choice.message.content = summary_text
    response = MagicMock()
    response.choices = [choice]
    client_instance = MagicMock()
    client_instance.chat.completions.create.return_value = response
    return client_instance


# ---------------------------------------------------------------------------
# TestCompressHistory
# ---------------------------------------------------------------------------

class TestCompressHistory:
    def test_no_compression_when_fits(self):
        """Returns unchanged when total tokens already fit."""
        from toklog.compress import compress_history

        msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
        result = compress_history(msgs, max_tokens=100_000)
        assert result == msgs

    def test_empty_messages(self):
        from toklog.compress import compress_history

        assert compress_history([]) == []

    def test_only_system_messages(self):
        """Only system messages — conversation is empty, no compression needed."""
        from toklog.compress import compress_history

        msgs = [_msg("system", "You are helpful.")]
        result = compress_history(msgs, max_tokens=100_000)
        assert result == msgs

    def test_keep_recent_exceeds_conversation(self):
        """keep_recent >= len(conversation) → return unchanged."""
        from toklog.compress import compress_history

        msgs = [_msg("user", "a"), _msg("assistant", "b")]
        result = compress_history(msgs, keep_recent=10, max_tokens=1)
        assert result == msgs

    def test_keep_recent_zero(self):
        """keep_recent=0 summarizes ALL conversation messages."""
        from toklog.compress import compress_history

        msgs = [_msg("user", "x " * 500), _msg("assistant", "y " * 500)]
        mock_client = _make_openai_mock("full summary")

        with patch("openai.OpenAI", return_value=mock_client):
            result = compress_history(msgs, max_tokens=10, keep_recent=0)

        # Result must have no recent tail from original conversation
        non_system = [m for m in result if m["role"] != "system"]
        assert non_system == []
        # Summary injected
        system_msgs = [m for m in result if m["role"] == "system"]
        assert any("[Summary" in m["content"] for m in system_msgs)

    def test_system_messages_preserved_verbatim(self):
        """System messages always appear first and unchanged."""
        from toklog.compress import compress_history

        sys_msg = _msg("system", "You are helpful.")
        conversation = [_msg("user", "word " * 300), _msg("assistant", "word " * 300)]
        msgs = [sys_msg] + conversation
        mock_client = _make_openai_mock("summary text")

        with patch("openai.OpenAI", return_value=mock_client):
            result = compress_history(msgs, max_tokens=50, keep_recent=1)

        assert result[0] == sys_msg

    def test_recent_messages_preserved_verbatim(self):
        """The last keep_recent conversation messages are unchanged."""
        from toklog.compress import compress_history

        old = [_msg("user", "old " * 300)]
        recent = [_msg("user", "recent message"), _msg("assistant", "recent reply")]
        mock_client = _make_openai_mock("summary")

        with patch("openai.OpenAI", return_value=mock_client):
            result = compress_history(old + recent, max_tokens=50, keep_recent=2)

        assert result[-2] == recent[0]
        assert result[-1] == recent[1]

    def test_old_messages_summarized(self):
        """Old messages are replaced by a summary system message."""
        from toklog.compress import compress_history

        old = [_msg("user", "tell me about cats " * 100)]
        recent = [_msg("assistant", "ok")]
        mock_client = _make_openai_mock("cats summary")

        with patch("openai.OpenAI", return_value=mock_client):
            result = compress_history(old + recent, max_tokens=50, keep_recent=1)

        # old message must not appear verbatim
        assert old[0] not in result
        # summary message must be present
        summary_msgs = [m for m in result if "[Summary" in m.get("content", "")]
        assert len(summary_msgs) == 1
        assert "cats summary" in summary_msgs[0]["content"]

    def test_summary_message_format(self):
        """Summary message has role=system and correct prefix."""
        from toklog.compress import compress_history

        msgs = [_msg("user", "word " * 400), _msg("assistant", "response")]
        mock_client = _make_openai_mock("the summary")

        with patch("openai.OpenAI", return_value=mock_client):
            result = compress_history(msgs, max_tokens=50, keep_recent=1)

        summary_msg = next(m for m in result if "[Summary" in m.get("content", ""))
        assert summary_msg["role"] == "system"
        assert summary_msg["content"].startswith("[Summary of earlier conversation]\n")

    def test_api_key_forwarded(self):
        """api_key is passed to openai.OpenAI()."""
        from toklog.compress import compress_history

        msgs = [_msg("user", "word " * 400), _msg("assistant", "reply")]
        mock_client = _make_openai_mock("summary")

        with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
            compress_history(msgs, max_tokens=50, keep_recent=1, api_key="sk-test")

        mock_cls.assert_called_once_with(api_key="sk-test")

    def test_summarizer_max_tokens_set(self):
        """max_tokens is passed to chat.completions.create."""
        from toklog.compress import compress_history

        msgs = [_msg("user", "word " * 400), _msg("assistant", "reply")]
        mock_client = _make_openai_mock("summary")

        with patch("openai.OpenAI", return_value=mock_client):
            compress_history(msgs, max_tokens=200, keep_recent=1)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert "max_tokens" in call_kwargs.kwargs or "max_tokens" in (call_kwargs.args[0] if call_kwargs.args else {})
        # Verify max_tokens was passed
        create_kwargs = call_kwargs.kwargs
        assert "max_tokens" in create_kwargs


# ---------------------------------------------------------------------------
# TestTokenCounting
# ---------------------------------------------------------------------------

class TestTokenCounting:
    def test_fallback_without_tiktoken(self, monkeypatch: pytest.MonkeyPatch):
        """Falls back to len//4 when tiktoken is unavailable."""
        import sys

        monkeypatch.setitem(sys.modules, "tiktoken", None)  # type: ignore[arg-type]

        # Re-import the function after patching
        import importlib
        import toklog.compress as compress_mod
        importlib.reload(compress_mod)

        text = "hello world " * 10  # 120 chars → 30 tokens
        result = compress_mod._count_tokens(text)
        assert result == len(text) // 4

        # Restore
        importlib.reload(compress_mod)

    def test_multimodal_content(self):
        """_extract_text handles list-of-dict multimodal content."""
        from toklog.compress import _extract_text

        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": "..."},
            {"type": "text", "text": "world"},
        ]
        result = _extract_text(content)
        assert "hello" in result
        assert "world" in result


# ---------------------------------------------------------------------------
# TestImport
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_from_package(self):
        """compress_history is importable from the top-level package."""
        from toklog import compress_history

        assert callable(compress_history)
