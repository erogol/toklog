"""Tests for the wrapper module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

openai = pytest.importorskip("openai", reason="openai not installed")

import toklog.logger as logger_mod
from toklog.wrapper import _PACKAGE_DIR, _get_call_site


@pytest.fixture(autouse=True)
def tmp_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect logs to a temp directory."""
    log_dir = str(tmp_path / "logs")
    monkeypatch.setattr(logger_mod, "_LOG_DIR", log_dir)
    monkeypatch.setattr(logger_mod, "_dir_ensured", False)
    return tmp_path


def _read_log_entries(tmp_path: Path) -> List[dict]:
    """Read all log entries from temp log dir."""
    entries: List[dict] = []
    log_dir = tmp_path / "logs"
    if not log_dir.exists():
        return entries
    for f in log_dir.glob("*.jsonl"):
        for line in f.read_text().strip().splitlines():
            entries.append(json.loads(line))
    return entries


def _make_openai_client() -> Any:
    """Create a mock OpenAI client that passes isinstance checks."""
    import openai

    client = MagicMock(spec=openai.OpenAI)
    client._toklog_wrapped = False

    # Set up the chat.completions.create chain
    completions = MagicMock()
    chat = MagicMock()
    chat.completions = completions
    client.chat = chat

    # Mock response
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_tokens_details = None
    response = MagicMock()
    response.usage = usage
    response.id = "chatcmpl-abc123"
    response._request_id = "req_abc123"
    response.model = "gpt-4o"
    completions.create.return_value = response

    return client


def _make_anthropic_client() -> Any:
    """Create a mock Anthropic client that passes isinstance checks."""
    import anthropic

    client = MagicMock(spec=anthropic.Anthropic)
    client._toklog_wrapped = False

    messages_api = MagicMock()
    client.messages = messages_api

    usage = MagicMock()
    usage.input_tokens = 200
    usage.output_tokens = 80
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    response = MagicMock()
    response.usage = usage
    response.id = "msg_abc123"
    response.model = "claude-3-5-sonnet-20241022"
    messages_api.create.return_value = response

    return client


class TestOpenAIWrapper:
    def test_wrap_and_call(self, tmp_path: Path) -> None:
        """Wrap an OpenAI client, call create(), verify log is written."""
        from toklog import wrap

        client = _make_openai_client()
        wrapped = wrap(client)

        result = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )

        # Response should pass through
        assert result.usage.prompt_tokens == 100

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["provider"] == "openai"
        assert e["model"] == "gpt-4o"
        assert e["input_tokens"] == 100
        assert e["output_tokens"] == 50
        assert e["error"] is False
        assert e["system_prompt_hash"] is not None

    def test_idempotency(self) -> None:
        """Wrapping twice should not double-wrap."""
        from toklog import wrap

        client = _make_openai_client()
        wrapped1 = wrap(client)
        wrapped2 = wrap(wrapped1)
        assert wrapped1 is wrapped2

    def test_default_tags(self, tmp_path: Path) -> None:
        """default_tags should apply when no per-request tag is set."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client, default_tags="my-feature")

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tags"] == "my-feature"

    def test_per_request_tag_overrides_default(self, tmp_path: Path) -> None:
        """X-TB-Tag header should override default_tags."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client, default_tags="default-tag")

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            extra_headers={"X-TB-Tag": "override-tag"},
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tags"] == "override-tag"

    def test_error_propagates_and_is_logged(self, tmp_path: Path) -> None:
        """If create() raises, exception propagates AND error is logged."""
        from toklog import wrap

        client = _make_openai_client()
        client.chat.completions.create.side_effect = RuntimeError("rate limit")
        wrap(client)

        with pytest.raises(RuntimeError, match="rate limit"):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["error"] is True
        assert entries[0]["error_type"] == "RuntimeError"

    def test_streaming(self, tmp_path: Path) -> None:
        """Test streaming response captures usage from final chunk."""
        from toklog import wrap

        client = _make_openai_client()

        # Create mock chunks
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.model = None

        chunk2 = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 150
        usage.completion_tokens = 75
        chunk2.usage = usage
        chunk2.model = "gpt-4o"

        client.chat.completions.create.return_value = iter([chunk1, chunk2])
        wrap(client)

        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) == 2

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["streaming"] is True
        assert entries[0]["input_tokens"] == 150
        assert entries[0]["output_tokens"] == 75

    def test_openai_streaming_cache_read_tokens(self, tmp_path: Path) -> None:
        """OpenAI streaming: cache_read_tokens extracted from prompt_tokens_details."""
        from toklog import wrap

        client = _make_openai_client()

        chunk = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 200
        usage.completion_tokens = 30
        ptd = MagicMock()
        ptd.cached_tokens = 80
        usage.prompt_tokens_details = ptd
        usage.completion_tokens_details = None
        chunk.usage = usage
        chunk.model = "gpt-4o"

        client.chat.completions.create.return_value = iter([chunk])
        wrap(client)

        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        list(stream)

        entries = _read_log_entries(tmp_path)
        assert entries[0]["cache_read_tokens"] == 80

    def test_openai_multiple_system_messages(self, tmp_path: Path) -> None:
        """Hash is computed from all system messages, not just the first."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        )

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        h = entries[0]["system_prompt_hash"]
        assert h is not None

        # Hash of two system msgs should differ from hash of just one
        from toklog.wrapper import _hash_system_prompt

        single_hash = _hash_system_prompt("You are helpful.")
        assert h != single_hash

    def test_xtb_tag_stripped_before_forwarding(self, tmp_path: Path) -> None:
        """X-TB-Tag header should be stripped before calling original create."""
        from toklog import wrap

        client = _make_openai_client()
        original_create = client.chat.completions.create
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            extra_headers={"X-TB-Tag": "my-tag", "Authorization": "Bearer xxx"},
        )

        # The original create should have been called without X-TB-Tag
        call_kwargs = original_create.call_args[1]
        headers = call_kwargs.get("extra_headers", {})
        assert "X-TB-Tag" not in headers
        assert headers.get("Authorization") == "Bearer xxx"

        # But the log should still have the tag
        entries = _read_log_entries(tmp_path)
        assert entries[0]["tags"] == "my-tag"


class TestAnthropicWrapper:
    def test_wrap_and_call(self, tmp_path: Path) -> None:
        """Wrap an Anthropic client, call create(), verify log is written."""
        from toklog import wrap

        client = _make_anthropic_client()
        wrapped = wrap(client)

        result = wrapped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result.usage.input_tokens == 200

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["provider"] == "anthropic"
        assert e["model"] == "claude-3-5-sonnet-20241022"
        assert e["input_tokens"] == 200
        assert e["output_tokens"] == 80
        assert e["system_prompt_hash"] is not None
        assert e["max_tokens_set"] == 1024

    def test_idempotency(self) -> None:
        """Wrapping twice should not double-wrap."""
        from toklog import wrap

        client = _make_anthropic_client()
        wrapped1 = wrap(client)
        wrapped2 = wrap(wrapped1)
        assert wrapped1 is wrapped2

    def test_error_propagates_and_is_logged(self, tmp_path: Path) -> None:
        """If create() raises, exception propagates AND error is logged."""
        from toklog import wrap

        client = _make_anthropic_client()
        client.messages.create.side_effect = ConnectionError("timeout")
        wrap(client)

        with pytest.raises(ConnectionError, match="timeout"):
            client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
            )

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["error"] is True
        assert entries[0]["error_type"] == "ConnectionError"

    def test_streaming(self, tmp_path: Path) -> None:
        """Test Anthropic streaming captures usage."""
        from toklog import wrap

        client = _make_anthropic_client()

        event1 = MagicMock()
        event1.type = "content_block_delta"
        event1.usage = None
        # Remove message attribute to avoid false positive
        del event1.message

        event2 = MagicMock()
        event2.type = "message_stop"
        event2.usage = None
        msg_usage = MagicMock()
        msg_usage.input_tokens = 300
        msg_usage.output_tokens = 120
        msg_usage.cache_read_input_tokens = 10
        msg_usage.cache_creation_input_tokens = 5
        event2.message = MagicMock()
        event2.message.usage = msg_usage
        event2.message.model = "claude-3-5-sonnet-20241022"

        client.messages.create.return_value = iter([event1, event2])
        wrap(client)

        stream = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        events = list(stream)
        assert len(events) == 2

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["streaming"] is True
        assert entries[0]["input_tokens"] == 300
        assert entries[0]["output_tokens"] == 120
        assert entries[0]["cache_read_tokens"] == 10
        assert entries[0]["cache_creation_tokens"] == 5

    def test_stream_method_interception(self, tmp_path: Path) -> None:
        """messages.stream() context manager should log usage on exit."""
        from toklog import wrap

        client = _make_anthropic_client()

        # Set up mock for messages.stream()
        mock_stream = MagicMock()

        # Mock the text_stream property
        mock_stream.text_stream = iter(["Hello", " world"])

        # Mock get_final_message()
        final_msg = MagicMock()
        final_msg.id = "msg_stream_123"
        final_msg.model = "claude-3-5-sonnet-20241022"
        usage = MagicMock()
        usage.input_tokens = 400
        usage.output_tokens = 150
        usage.cache_read_input_tokens = 20
        usage.cache_creation_input_tokens = 10
        final_msg.usage = usage
        mock_stream.get_final_message.return_value = final_msg

        # Make it work as a context manager
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        client.messages.stream = MagicMock(return_value=mock_stream)
        wrap(client)

        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            # Access text_stream to simulate real usage
            text = list(stream.text_stream)
            assert text == ["Hello", " world"]

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["provider"] == "anthropic"
        assert e["streaming"] is True
        assert e["input_tokens"] == 400
        assert e["output_tokens"] == 150
        assert e["cache_read_tokens"] == 20
        assert e["cache_creation_tokens"] == 10
        assert e["request_id"] == "msg_stream_123"

    def test_xtb_tag_stripped_before_forwarding(self, tmp_path: Path) -> None:
        """X-TB-Tag header should be stripped before calling Anthropic create."""
        from toklog import wrap

        client = _make_anthropic_client()
        original_create = client.messages.create
        wrap(client)

        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            extra_headers={"X-TB-Tag": "my-tag"},
        )

        call_kwargs = original_create.call_args[1]
        assert "extra_headers" not in call_kwargs or "X-TB-Tag" not in call_kwargs.get(
            "extra_headers", {}
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tags"] == "my-tag"


class TestToolCallsMade:
    def test_openai_non_streaming_with_tool_calls(self, tmp_path: Path) -> None:
        """OpenAI non-streaming: tool_calls_made reflects number of tool calls."""
        from toklog import wrap

        client = _make_openai_client()
        response = client.chat.completions.create.return_value
        mock_tc_1 = MagicMock()
        mock_tc_2 = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = [mock_tc_1, mock_tc_2]
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tool_calls_made"] == 2

    def test_openai_non_streaming_no_tool_calls(self, tmp_path: Path) -> None:
        """OpenAI non-streaming: tool_calls_made is 0 when tool_calls is None."""
        from toklog import wrap

        client = _make_openai_client()
        response = client.chat.completions.create.return_value
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tool_calls_made"] == 0

    def test_anthropic_non_streaming_with_tool_calls(self, tmp_path: Path) -> None:
        """Anthropic non-streaming: tool_calls_made counts tool_use blocks."""
        from toklog import wrap

        client = _make_anthropic_client()
        response = client.messages.create.return_value
        response.content = [
            MagicMock(type="text"),
            MagicMock(type="tool_use"),
            MagicMock(type="tool_use"),
        ]
        wrap(client)

        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tool_calls_made"] == 2

    def test_anthropic_non_streaming_no_tool_calls(self, tmp_path: Path) -> None:
        """Anthropic non-streaming: tool_calls_made is 0 when no tool_use blocks."""
        from toklog import wrap

        client = _make_anthropic_client()
        response = client.messages.create.return_value
        response.content = [MagicMock(type="text")]
        wrap(client)

        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tool_calls_made"] == 0


class TestModelFromResponse:
    """Verify model is read from response object, not just request kwargs."""

    def test_openai_non_streaming_model_from_response(self, tmp_path: Path) -> None:
        """OpenAI non-streaming: model in log entry comes from response.model."""
        from toklog import wrap

        client = _make_openai_client()
        # Override the response model to simulate server returning a different alias
        client.chat.completions.create.return_value.model = "gpt-4o-2024-08-06"
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["model"] == "gpt-4o-2024-08-06"

    def test_openai_streaming_model_from_chunk(self, tmp_path: Path) -> None:
        """OpenAI streaming: model in log entry comes from chunk.model."""
        from toklog import wrap

        client = _make_openai_client()

        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.model = None

        chunk2 = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 20
        chunk2.usage = usage
        chunk2.model = "gpt-4o-2024-08-06"

        client.chat.completions.create.return_value = iter([chunk1, chunk2])
        wrap(client)

        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        list(stream)

        entries = _read_log_entries(tmp_path)
        assert entries[0]["model"] == "gpt-4o-2024-08-06"

    def test_anthropic_non_streaming_model_from_response(self, tmp_path: Path) -> None:
        """Anthropic non-streaming: model in log entry comes from response.model."""
        from toklog import wrap

        client = _make_anthropic_client()
        client.messages.create.return_value.model = "claude-3-5-sonnet-20241022"
        wrap(client)

        client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["model"] == "claude-3-5-sonnet-20241022"

    def test_anthropic_streaming_model_from_event_message(self, tmp_path: Path) -> None:
        """Anthropic stream=True: model comes from event.message.model."""
        from toklog import wrap

        client = _make_anthropic_client()

        event1 = MagicMock()
        event1.type = "content_block_delta"
        event1.usage = None
        del event1.message

        event2 = MagicMock()
        event2.type = "message_start"
        event2.usage = None
        msg_usage = MagicMock()
        msg_usage.input_tokens = 100
        msg_usage.output_tokens = 40
        msg_usage.cache_read_input_tokens = 0
        msg_usage.cache_creation_input_tokens = 0
        event2.message = MagicMock()
        event2.message.usage = msg_usage
        event2.message.model = "claude-3-5-sonnet-20241022"

        client.messages.create.return_value = iter([event1, event2])
        wrap(client)

        stream = client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        list(stream)

        entries = _read_log_entries(tmp_path)
        assert entries[0]["model"] == "claude-3-5-sonnet-20241022"

    def test_anthropic_stream_method_model_from_final_message(self, tmp_path: Path) -> None:
        """Anthropic messages.stream(): model comes from get_final_message().model."""
        from toklog import wrap

        client = _make_anthropic_client()

        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["Hi"])

        final_msg = MagicMock()
        final_msg.id = "msg_xyz"
        final_msg.model = "claude-3-5-sonnet-20241022"
        usage = MagicMock()
        usage.input_tokens = 50
        usage.output_tokens = 10
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0
        final_msg.usage = usage
        mock_stream.get_final_message.return_value = final_msg

        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        client.messages.stream = MagicMock(return_value=mock_stream)
        wrap(client)

        with client.messages.stream(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        ):
            pass

        entries = _read_log_entries(tmp_path)
        assert entries[0]["model"] == "claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# Async helpers and fixtures
# ---------------------------------------------------------------------------


async def _async_iter(items: list) -> Any:
    """Async generator that yields items one by one."""
    for item in items:
        yield item


def _make_openai_async_client() -> Any:
    """Create a mock AsyncOpenAI client that passes isinstance checks."""
    import openai

    client = MagicMock(spec=openai.AsyncOpenAI)
    client._toklog_wrapped = False

    completions = MagicMock()
    chat = MagicMock()
    chat.completions = completions
    client.chat = chat

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_tokens_details = None
    response = MagicMock()
    response.usage = usage
    response.id = "chatcmpl-async-abc123"
    response._request_id = "req_async_abc123"
    response.model = "gpt-4o"
    completions.create = AsyncMock(return_value=response)

    return client


def _make_anthropic_async_client() -> Any:
    """Create a mock AsyncAnthropic client that passes isinstance checks."""
    import anthropic

    client = MagicMock(spec=anthropic.AsyncAnthropic)
    client._toklog_wrapped = False

    messages_api = MagicMock()
    client.messages = messages_api

    usage = MagicMock()
    usage.input_tokens = 200
    usage.output_tokens = 80
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    response = MagicMock()
    response.usage = usage
    response.id = "msg_async_abc123"
    response.model = "claude-3-5-sonnet-20241022"
    messages_api.create = AsyncMock(return_value=response)

    return client


# ---------------------------------------------------------------------------
# Async OpenAI tests
# ---------------------------------------------------------------------------


class TestOpenAIAsyncWrapper:
    @pytest.mark.asyncio
    async def test_wrap_and_call(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_openai_async_client()
        wrapped = wrap(client)

        result = await wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )

        assert result.usage.prompt_tokens == 100

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["provider"] == "openai"
        assert e["model"] == "gpt-4o"
        assert e["input_tokens"] == 100
        assert e["output_tokens"] == 50
        assert e["error"] is False
        assert e["system_prompt_hash"] is not None

    @pytest.mark.asyncio
    async def test_idempotency(self) -> None:
        from toklog import wrap

        client = _make_openai_async_client()
        wrapped1 = wrap(client)
        wrapped2 = wrap(wrapped1)
        assert wrapped1 is wrapped2

    @pytest.mark.asyncio
    async def test_error_propagates(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_openai_async_client()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("rate limit"))
        wrap(client)

        with pytest.raises(RuntimeError, match="rate limit"):
            await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["error"] is True
        assert entries[0]["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_streaming(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_openai_async_client()

        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.model = None

        chunk2 = MagicMock()
        usage = MagicMock()
        usage.prompt_tokens = 150
        usage.completion_tokens = 75
        chunk2.usage = usage
        chunk2.model = "gpt-4o"

        client.chat.completions.create = AsyncMock(
            return_value=_async_iter([chunk1, chunk2])
        )
        wrap(client)

        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        chunks = [chunk async for chunk in stream]
        assert len(chunks) == 2

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["streaming"] is True
        assert entries[0]["input_tokens"] == 150
        assert entries[0]["output_tokens"] == 75


# ---------------------------------------------------------------------------
# Async Anthropic tests
# ---------------------------------------------------------------------------


class TestAnthropicAsyncWrapper:
    @pytest.mark.asyncio
    async def test_wrap_and_call(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_anthropic_async_client()
        wrapped = wrap(client)

        result = await wrapped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result.usage.input_tokens == 200

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["provider"] == "anthropic"
        assert e["model"] == "claude-3-5-sonnet-20241022"
        assert e["input_tokens"] == 200
        assert e["output_tokens"] == 80
        assert e["system_prompt_hash"] is not None
        assert e["max_tokens_set"] == 1024

    @pytest.mark.asyncio
    async def test_idempotency(self) -> None:
        from toklog import wrap

        client = _make_anthropic_async_client()
        wrapped1 = wrap(client)
        wrapped2 = wrap(wrapped1)
        assert wrapped1 is wrapped2

    @pytest.mark.asyncio
    async def test_error_propagates(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_anthropic_async_client()
        client.messages.create = AsyncMock(side_effect=ConnectionError("timeout"))
        wrap(client)

        with pytest.raises(ConnectionError, match="timeout"):
            await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
            )

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["error"] is True
        assert entries[0]["error_type"] == "ConnectionError"

    @pytest.mark.asyncio
    async def test_streaming(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_anthropic_async_client()

        event1 = MagicMock()
        event1.type = "content_block_delta"
        event1.usage = None
        del event1.message

        event2 = MagicMock()
        event2.type = "message_stop"
        event2.usage = None
        msg_usage = MagicMock()
        msg_usage.input_tokens = 300
        msg_usage.output_tokens = 120
        msg_usage.cache_read_input_tokens = 10
        msg_usage.cache_creation_input_tokens = 5
        event2.message = MagicMock()
        event2.message.usage = msg_usage
        event2.message.model = "claude-3-5-sonnet-20241022"

        client.messages.create = AsyncMock(
            return_value=_async_iter([event1, event2])
        )
        wrap(client)

        stream = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        events = [e async for e in stream]
        assert len(events) == 2

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["streaming"] is True
        assert entries[0]["input_tokens"] == 300
        assert entries[0]["output_tokens"] == 120
        assert entries[0]["cache_read_tokens"] == 10
        assert entries[0]["cache_creation_tokens"] == 5


# ---------------------------------------------------------------------------
# Async tool calls
# ---------------------------------------------------------------------------


class TestAsyncToolCallsMade:
    @pytest.mark.asyncio
    async def test_openai_async_with_tool_calls(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_openai_async_client()
        response = client.chat.completions.create.return_value
        mock_tc_1 = MagicMock()
        mock_tc_2 = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = [mock_tc_1, mock_tc_2]
        wrap(client)

        await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tool_calls_made"] == 2

    @pytest.mark.asyncio
    async def test_anthropic_async_with_tool_calls(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_anthropic_async_client()
        response = client.messages.create.return_value
        response.content = [
            MagicMock(type="text"),
            MagicMock(type="tool_use"),
            MagicMock(type="tool_use"),
        ]
        wrap(client)

        await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["tool_calls_made"] == 2


# ---------------------------------------------------------------------------
# Async model from response
# ---------------------------------------------------------------------------


class TestAsyncModelFromResponse:
    @pytest.mark.asyncio
    async def test_openai_async_non_streaming(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_openai_async_client()
        client.chat.completions.create.return_value.model = "gpt-4o-2024-08-06"
        wrap(client)

        await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["model"] == "gpt-4o-2024-08-06"

    @pytest.mark.asyncio
    async def test_anthropic_async_non_streaming(self, tmp_path: Path) -> None:
        from toklog import wrap

        client = _make_anthropic_async_client()
        client.messages.create.return_value.model = "claude-3-5-sonnet-20241022"
        wrap(client)

        await client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["model"] == "claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# Call site attribution tests
# ---------------------------------------------------------------------------


class TestCallSite:
    def test_returns_external_frame(self) -> None:
        """Should return file/function/line of first external frame."""
        internal = MagicMock()
        internal.filename = os.path.join(_PACKAGE_DIR, "wrapper.py")
        internal.function = "wrapped_create"
        internal.lineno = 42

        external = MagicMock()
        external.filename = "/home/user/myapp/app.py"
        external.function = "my_function"
        external.lineno = 10

        with patch("inspect.stack", return_value=[internal, external]):
            result = _get_call_site()

        assert result is not None
        assert result["function"] == "my_function"
        assert result["line"] == 10
        assert "app.py" in result["file"]

    def test_returns_none_on_error(self) -> None:
        """Should return None when inspect.stack raises."""
        with patch("inspect.stack", side_effect=RuntimeError("boom")):
            result = _get_call_site()
        assert result is None

    def test_skips_internal_frames(self) -> None:
        """Should return None when all frames are inside the package."""
        internal1 = MagicMock()
        internal1.filename = os.path.join(_PACKAGE_DIR, "wrapper.py")
        internal1.function = "f1"
        internal1.lineno = 1

        internal2 = MagicMock()
        internal2.filename = os.path.join(_PACKAGE_DIR, "report.py")
        internal2.function = "f2"
        internal2.lineno = 2

        with patch("inspect.stack", return_value=[internal1, internal2]):
            result = _get_call_site()
        assert result is None

    def test_integration_call_site_in_entry(self, tmp_path: Path) -> None:
        """Wrapped create() call should include call_site in logged entry."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert "call_site" in entries[0]
        cs = entries[0]["call_site"]
        # call_site may be None if detection fails, but in normal test runs it should
        # be populated with the test file as the caller
        if cs is not None:
            assert "file" in cs
            assert "function" in cs
            assert "line" in cs

    def test_skips_site_packages_frames(self) -> None:
        """Should skip frames from site-packages directories."""
        site_pkg = MagicMock()
        site_pkg.filename = "/usr/local/lib/python3.11/site-packages/somelib/module.py"
        site_pkg.function = "call_api"
        site_pkg.lineno = 99

        user = MagicMock()
        user.filename = "/home/user/myproject/main.py"
        user.function = "run"
        user.lineno = 5

        with patch("inspect.stack", return_value=[site_pkg, user]):
            result = _get_call_site()

        assert result is not None
        assert result["function"] == "run"
        assert result["line"] == 5

    def test_skips_dist_packages_frames(self) -> None:
        """Should skip frames from dist-packages directories."""
        dist_pkg = MagicMock()
        dist_pkg.filename = "/usr/lib/python3/dist-packages/requests/adapters.py"
        dist_pkg.function = "send"
        dist_pkg.lineno = 200

        user = MagicMock()
        user.filename = "/app/service.py"
        user.function = "fetch"
        user.lineno = 15

        with patch("inspect.stack", return_value=[dist_pkg, user]):
            result = _get_call_site()

        assert result is not None
        assert result["function"] == "fetch"

    def test_integration_call_site_skips_libraries_when_present(self, tmp_path: Path) -> None:
        """Wrapped create() call site skips site-packages frames and returns user code."""
        from toklog import wrap
        from unittest.mock import patch

        user_frame = MagicMock()
        user_frame.filename = "/home/user/myapp/pipeline.py"
        user_frame.function = "run_inference"
        user_frame.lineno = 77

        lib_frame = MagicMock()
        lib_frame.filename = "/usr/local/lib/python3.11/site-packages/httpx/client.py"
        lib_frame.function = "send"
        lib_frame.lineno = 300

        # Stack: internal toklog frame → lib frame → user frame
        internal_frame = MagicMock()
        internal_frame.filename = os.path.join(_PACKAGE_DIR, "wrapper.py")
        internal_frame.function = "wrapped_create"
        internal_frame.lineno = 42

        with patch("inspect.stack", return_value=[internal_frame, lib_frame, user_frame]):
            result = _get_call_site()

        assert result is not None
        assert result["function"] == "run_inference"
        assert result["line"] == 77
        assert "pipeline.py" in result["file"]


class TestUserMessagePreview:
    def test_openai_multiple_user_msgs_last_one(self, tmp_path: Path) -> None:
        """Multiple user messages → last one is captured as preview."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "second message"},
            ],
        )

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["user_message_preview"] == "second message"

    def test_anthropic_content_blocks(self, tmp_path: Path) -> None:
        """Anthropic content blocks (list of dicts) → text extracted."""
        from toklog import wrap

        client = _make_anthropic_client()
        wrap(client)

        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello from block"},
                        {"type": "image", "source": {}},
                    ],
                }
            ],
        )

        entries = _read_log_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["user_message_preview"] == "hello from block"

    def test_truncation_at_word_boundary(self, tmp_path: Path) -> None:
        """Message >200 chars is truncated at nearest word boundary after char 150."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)

        # Build a string that's well over 200 chars with a space between 150 and 200
        long_msg = "word " * 50  # 250 chars
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": long_msg}],
        )

        entries = _read_log_entries(tmp_path)
        preview = entries[0]["user_message_preview"]
        assert preview is not None
        assert len(preview) <= 200
        # Should end at a word boundary (space removed by rfind, not in middle of word)
        assert not preview.endswith(" ")

    def test_no_user_message_returns_none(self, tmp_path: Path) -> None:
        """No user message in messages → preview is None."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are helpful."}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["user_message_preview"] is None

    def test_log_preview_false_returns_none(self, tmp_path: Path) -> None:
        """log_preview=False → user_message_preview is None regardless of messages."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client, log_preview=False)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "some content"}],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["user_message_preview"] is None

    def test_empty_messages_returns_none(self, tmp_path: Path) -> None:
        """Empty messages list → preview is None."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
        )

        entries = _read_log_entries(tmp_path)
        assert entries[0]["user_message_preview"] is None


class TestContextSignals:
    def test_context_signal_fields_present(self, tmp_path: Path) -> None:
        """All context signal fields are logged on every entry."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
        entries = _read_log_entries(tmp_path)
        entry = entries[0]
        for field in ("system_prompt_chars", "total_message_chars", "tool_result_chars",
                      "has_code_blocks", "has_structured_data", "thinking_input_chars"):
            assert field in entry, f"missing field: {field}"

    def test_thinking_input_chars_anthropic(self, tmp_path: Path) -> None:
        """thinking_input_chars counts thinking blocks echoed back in Anthropic messages."""
        from toklog import wrap

        thinking_text = "t" * 400
        client = _make_anthropic_client()
        wrap(client)
        client.messages.create(
            model="claude-opus-4-6",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": thinking_text, "signature": "sig"},
                    {"type": "text", "text": "answer"},
                ]},
                {"role": "user", "content": "follow-up"},
            ],
        )
        entries = _read_log_entries(tmp_path)
        assert entries[0]["thinking_input_chars"] == 400

    def test_tool_result_chars_counted(self, tmp_path: Path) -> None:
        """tool_result_chars accumulates content from tool role messages."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "run it"},
                {"role": "tool", "content": "x" * 500},
            ],
        )
        entries = _read_log_entries(tmp_path)
        assert entries[0]["tool_result_chars"] == 500

    def test_has_code_blocks_detected(self, tmp_path: Path) -> None:
        """has_code_blocks is True when any message contains triple backticks."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "here is code:\n```python\nprint('hi')\n```"}],
        )
        entries = _read_log_entries(tmp_path)
        assert entries[0]["has_code_blocks"] is True

    def test_has_code_blocks_false_when_no_code(self, tmp_path: Path) -> None:
        """has_code_blocks is False for plain text messages."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "what is 2 + 2?"}],
        )
        entries = _read_log_entries(tmp_path)
        assert entries[0]["has_code_blocks"] is False

    def test_system_prompt_chars_openai(self, tmp_path: Path) -> None:
        """system_prompt_chars counts the role=system message for OpenAI."""
        from toklog import wrap

        client = _make_openai_client()
        wrap(client)
        system_text = "You are a helpful assistant." * 10
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": "hi"},
            ],
        )
        entries = _read_log_entries(tmp_path)
        assert entries[0]["system_prompt_chars"] == len(system_text)


class TestKeyHint:
    """Tests for the _key_hint helper and api_key_hint logging."""

    def test_returns_last_8_chars(self) -> None:
        from toklog.wrapper import _key_hint

        # "sk-proj-12345678" → last 8 = "12345678"
        assert _key_hint("sk-proj-12345678") == "...12345678"

    def test_returns_none_for_none(self) -> None:
        from toklog.wrapper import _key_hint

        assert _key_hint(None) is None

    def test_returns_none_for_short_key(self) -> None:
        from toklog.wrapper import _key_hint

        assert _key_hint("ab") is None
        assert _key_hint("abc") is None

    def test_returns_none_for_non_string(self) -> None:
        from toklog.wrapper import _key_hint

        assert _key_hint(12345) is None  # type: ignore[arg-type]
        assert _key_hint([]) is None  # type: ignore[arg-type]

    def test_strips_whitespace(self) -> None:
        from toklog.wrapper import _key_hint

        assert _key_hint("  sk-abcdefgh  ") == "...abcdefgh"

    def test_openai_entry_has_key_hint(self, tmp_path: Path) -> None:
        """api_key_hint logged for OpenAI client with api_key set."""
        from toklog import wrap

        client = _make_openai_client()
        client.api_key = "sk-test-abcdef12345678"
        wrap(client)
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        entries = _read_log_entries(tmp_path)
        assert entries[0]["api_key_hint"] == "...12345678"

    def test_openai_entry_key_hint_none_when_no_key(self, tmp_path: Path) -> None:
        """api_key_hint is None when client has no api_key."""
        from toklog import wrap

        client = _make_openai_client()
        client.api_key = None
        wrap(client)
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        entries = _read_log_entries(tmp_path)
        assert entries[0]["api_key_hint"] is None

    def test_anthropic_entry_has_key_hint(self, tmp_path: Path) -> None:
        """api_key_hint logged for Anthropic client with api_key set."""
        from toklog import wrap

        client = _make_anthropic_client()
        client.api_key = "sk-ant-abcdef12345678"
        wrap(client)
        client.messages.create(
            model="claude-opus-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
        entries = _read_log_entries(tmp_path)
        assert entries[0]["api_key_hint"] == "...12345678"
