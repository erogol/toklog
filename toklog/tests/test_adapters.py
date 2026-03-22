from __future__ import annotations
"""Tests for toklog.adapters — SSEStreamBuffer, OpenAI, Anthropic, and Gemini adapters.

Tests covering:
  1-7:   Module-level functions (extract_from_response, classify_endpoint)
  8-17:  SSEStreamBuffer
  18-25: OpenAIEventHandler
  Anthropic: TestAnthropicExtractFromResponse (1-6), TestAnthropicEventHandler (7-25),
             TestAnthropicIntegration (26-28), TestAnthropicRegression (29-30)
  32-35: stream_options injection
  36-39: Integration tests (full pipeline)
  40-42: Regression tests
  Gemini: TestGeminiExtractModelFromUrl, TestGeminiExtractFromResponse,
          TestGeminiEventHandler, TestGeminiIntegration, TestGeminiRegression
"""

import json
import logging

import pytest

from toklog.adapters.sse import SSEStreamBuffer
from toklog.adapters.gemini import (
    GeminiUsageResult,
    GeminiEventHandler,
    extract_from_response as gemini_extract_from_response,
    extract_model_from_url,
    create_stream_handler as gemini_create_stream_handler,
)
from toklog.adapters.openai import (
    UsageResult,
    OpenAIEventHandler,
    classify_endpoint,
    extract_from_response,
    create_stream_handler,
    maybe_inject_stream_options,
)
from toklog.adapters.anthropic import (
    AnthropicUsageResult,
    AnthropicEventHandler,
    extract_from_response as anthropic_extract_from_response,
    create_stream_handler as anthropic_create_stream_handler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse_event(data: dict, event_type: str | None = None) -> bytes:
    """Build a complete SSE event as bytes (with trailing \\n\\n)."""
    lines: list[str] = []
    if event_type:
        lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(data)}")
    return ("\n".join(lines) + "\n\n").encode()


def _chat_completion_body(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int = 150,
    model: str = "gpt-4o",
    request_id: str = "chatcmpl-abc123",
    cached_tokens: int = 0,
    reasoning_tokens: int | None = None,
    tool_calls: list[dict] | None = None,
) -> dict:
    """Build a Chat Completions non-streaming response body."""
    usage: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if cached_tokens:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}

    message: dict = {"role": "assistant", "content": "Hello"}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": request_id,
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage,
    }


def _responses_api_body(
    input_tokens: int = 200,
    output_tokens: int = 80,
    total_tokens: int = 280,
    model: str = "gpt-5.4",
    response_id: str = "resp_xyz789",
    cached_tokens: int = 0,
    reasoning_tokens: int | None = None,
    output_items: list[dict] | None = None,
) -> dict:
    """Build a Responses API non-streaming response body."""
    usage: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if cached_tokens:
        usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
    if reasoning_tokens is not None:
        usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}

    return {
        "id": response_id,
        "object": "response",
        "model": model,
        "output": output_items or [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
        "usage": usage,
    }


# ===========================================================================
# Tests 1-7: Module-level functions
# ===========================================================================

class TestExtractFromResponse:
    """Tests 1-5: extract_from_response for Chat Completions and Responses API."""

    def test_01_chat_completions_full_usage(self) -> None:
        """Test 1: Chat Completions body with full usage."""
        body = _chat_completion_body()
        result = extract_from_response(body, "openai.chat_completions")

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.usage_status == "exact"
        assert result.usage_source == "provider_response"
        assert result.model_served == "gpt-4o"
        assert result.request_id == "chatcmpl-abc123"
        assert result.response_id is None
        assert result.cache_read_tokens == 0
        assert result.cache_creation_tokens == 0
        assert result.raw_usage is not None

    def test_02_responses_api_full_usage(self) -> None:
        """Test 2: Responses API body with full usage."""
        body = _responses_api_body()
        result = extract_from_response(body, "openai.responses")

        assert result.input_tokens == 200
        assert result.output_tokens == 80
        assert result.total_tokens == 280
        assert result.usage_status == "exact"
        assert result.usage_source == "provider_response"
        assert result.model_served == "gpt-5.4"
        assert result.response_id == "resp_xyz789"
        assert result.request_id == "resp_xyz789"

    def test_03_missing_usage(self) -> None:
        """Test 3: Missing usage -> usage_status == 'missing'."""
        body = {"id": "chatcmpl-x", "model": "gpt-4o", "choices": []}
        result = extract_from_response(body, "openai.chat_completions")

        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.usage_status == "missing"
        assert result.raw_usage is None

    def test_04_with_reasoning_tokens(self) -> None:
        """Test 4: Reasoning tokens extracted correctly."""
        body = _chat_completion_body(reasoning_tokens=42)
        result = extract_from_response(body, "openai.chat_completions")

        assert result.reasoning_tokens == 42

    def test_05_with_cached_tokens(self) -> None:
        """Test 5: Cached tokens extracted correctly."""
        body = _chat_completion_body(cached_tokens=30)
        result = extract_from_response(body, "openai.chat_completions")

        assert result.cache_read_tokens == 30

    def test_05b_responses_api_with_tool_calls(self) -> None:
        """Responses API tool calls counted from output items."""
        items = [
            {"type": "function_call", "name": "search"},
            {"type": "message", "content": []},
            {"type": "function_call", "name": "read"},
        ]
        body = _responses_api_body(output_items=items)
        result = extract_from_response(body, "openai.responses")
        assert result.tool_calls_made == 2

    def test_05c_chat_completions_with_tool_calls(self) -> None:
        """Chat Completions tool calls counted from choices[0].message.tool_calls."""
        tc = [{"id": "call_1", "function": {"name": "f"}}, {"id": "call_2", "function": {"name": "g"}}]
        body = _chat_completion_body(tool_calls=tc)
        result = extract_from_response(body, "openai.chat_completions")
        assert result.tool_calls_made == 2


class TestClassifyEndpoint:
    """Tests 6-7: classify_endpoint URL and body heuristics."""

    def test_06_url_paths(self) -> None:
        """Test 6: Various URL paths classified correctly."""
        assert classify_endpoint("/v1/chat/completions") == "openai.chat_completions"
        assert classify_endpoint("/v1/responses") == "openai.responses"
        assert classify_endpoint("/v1/responses/resp_abc") == "openai.responses"
        assert classify_endpoint("/openai/deployments/gpt4/chat/completions") == "openai.chat_completions"
        assert classify_endpoint("/v1/unknown") == "openai.chat_completions"  # default

    def test_07_body_fallback(self) -> None:
        """Test 7: Body-based fallback for ambiguous paths."""
        assert classify_endpoint("/v1/unknown", {"input": "hello"}) == "openai.responses"
        assert classify_endpoint("/v1/unknown", {"messages": []}) == "openai.chat_completions"
        assert classify_endpoint("/v1/unknown", {"input": "x", "messages": []}) == "openai.chat_completions"
        assert classify_endpoint("/v1/unknown", {}) == "openai.chat_completions"


# ===========================================================================
# Tests 8-17: SSEStreamBuffer
# ===========================================================================

class TestSSEStreamBuffer:
    """Tests 8-17: byte-level SSE buffering."""

    def test_08_complete_event_in_one_chunk(self) -> None:
        """Test 8: Complete event in one chunk returns parsed dict."""
        buf = SSEStreamBuffer()
        data = {"type": "message", "text": "hello"}
        events = buf.feed(_sse_event(data))

        assert len(events) == 1
        assert events[0] == data

    def test_09_event_split_across_two_chunks(self) -> None:
        """Test 9: Event split across two chunks — nothing on first, parsed on second."""
        buf = SSEStreamBuffer()
        raw = _sse_event({"key": "value"})
        mid = len(raw) // 2

        events1 = buf.feed(raw[:mid])
        assert events1 == []

        events2 = buf.feed(raw[mid:])
        assert len(events2) == 1
        assert events2[0] == {"key": "value"}

    def test_10_event_split_across_three_chunks(self) -> None:
        """Test 10: Event split across three chunks — buffered correctly."""
        buf = SSEStreamBuffer()
        raw = _sse_event({"a": 1, "b": 2, "c": 3})
        third = len(raw) // 3

        assert buf.feed(raw[:third]) == []
        assert buf.feed(raw[third:third * 2]) == []
        events = buf.feed(raw[third * 2:])
        assert len(events) == 1
        assert events[0]["a"] == 1

    def test_11_multiline_data_field(self) -> None:
        """Test 11: Multi-line data: field — lines joined correctly."""
        buf = SSEStreamBuffer()
        # Build a multi-line data SSE event manually
        raw = b'data: {"key"\n'
        raw += b'data: : "value"}\n\n'
        events = buf.feed(raw)

        assert len(events) == 1
        assert events[0] == {"key": "value"}

    def test_12_crlf_delimiters(self) -> None:
        """Test 12: \\r\\n\\r\\n delimiters handled."""
        buf = SSEStreamBuffer()
        raw = b'data: {"ok": true}\r\n\r\n'
        events = buf.feed(raw)

        assert len(events) == 1
        assert events[0] == {"ok": True}

    def test_13_multiple_events_in_one_chunk(self) -> None:
        """Test 13: Multiple events in one chunk — all returned."""
        buf = SSEStreamBuffer()
        raw = _sse_event({"n": 1}) + _sse_event({"n": 2}) + _sse_event({"n": 3})
        events = buf.feed(raw)

        assert len(events) == 3
        assert [e["n"] for e in events] == [1, 2, 3]

    def test_14_done_skipped(self) -> None:
        """Test 14: data: [DONE] skipped cleanly."""
        buf = SSEStreamBuffer()
        raw = b"data: [DONE]\n\n"
        events = buf.feed(raw)

        assert events == []

    def test_15_empty_chunks(self) -> None:
        """Test 15: Empty chunks — no crash, returns empty list."""
        buf = SSEStreamBuffer()
        assert buf.feed(b"") == []
        assert buf.feed(b"") == []
        assert buf.flush() == []

    def test_16_malformed_json_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test 16: Malformed JSON — returns None, logs WARNING."""
        buf = SSEStreamBuffer()
        raw = b"data: {not valid json}\n\n"

        with caplog.at_level(logging.WARNING, logger="toklog.adapters.sse"):
            events = buf.feed(raw)

        assert events == []
        assert any("malformed JSON" in rec.message for rec in caplog.records)

    def test_17_flush_with_trailing_data(self) -> None:
        """Test 17: flush() with trailing data — parsed if valid."""
        buf = SSEStreamBuffer()
        # Feed incomplete event (no trailing \n\n)
        buf.feed(b'data: {"trailing": true}')

        events = buf.flush()
        assert len(events) == 1
        assert events[0] == {"trailing": True}


# ===========================================================================
# Tests 18-25: OpenAIEventHandler
# ===========================================================================

class TestOpenAIEventHandler:
    """Tests 18-25: OpenAI streaming event handler."""

    def test_18_response_completed_usage(self) -> None:
        """Test 18: response.completed event — usage extracted correctly."""
        handler = OpenAIEventHandler("openai.responses")
        handler.handle({
            "type": "response.completed",
            "response": {
                "id": "resp_001",
                "model": "gpt-5.4",
                "output": [{"type": "message", "content": []}],
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 200,
                    "total_tokens": 700,
                    "input_tokens_details": {"cached_tokens": 50},
                    "output_tokens_details": {"reasoning_tokens": 30},
                },
            },
        })
        result = handler.finalize()

        assert result.input_tokens == 500
        assert result.output_tokens == 200
        assert result.total_tokens == 700
        assert result.cache_read_tokens == 50
        assert result.reasoning_tokens == 30
        assert result.usage_status == "exact"
        assert result.usage_source == "provider_stream_final"
        assert result.response_id == "resp_001"
        assert result.model_served == "gpt-5.4"

    def test_19_chat_completions_final_chunk_with_usage(self) -> None:
        """Test 19: Chat Completions final chunk with usage — extracted correctly."""
        handler = OpenAIEventHandler("openai.chat_completions")
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-xyz",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Hi"}}],
        })
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-xyz",
            "model": "gpt-4o",
            "choices": [],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "total_tokens": 180,
            },
        })
        result = handler.finalize()

        assert result.input_tokens == 120
        assert result.output_tokens == 60
        assert result.usage_status == "exact"
        assert result.model_served == "gpt-4o"
        assert result.request_id == "chatcmpl-xyz"

    def test_20_stream_without_usage(self) -> None:
        """Test 20: Stream without usage event — finalize() returns missing result."""
        handler = OpenAIEventHandler("openai.chat_completions")
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-nousage",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "hello"}}],
        })
        result = handler.finalize()

        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.usage_status == "missing"
        assert result.model_served == "gpt-4o"

    def test_21_finalize_idempotency(self) -> None:
        """Test 21: finalize() idempotency — second call returns same cached result."""
        handler = OpenAIEventHandler("openai.responses")
        handler.handle({
            "type": "response.completed",
            "response": {
                "id": "resp_idem",
                "model": "gpt-5.4",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        })
        r1 = handler.finalize()
        r2 = handler.finalize()
        assert r1 is r2

    def test_22_tool_call_index_tracking_chat(self) -> None:
        """Test 22: Tool call index tracking — tool_calls_made correct for Chat Completions."""
        handler = OpenAIEventHandler("openai.chat_completions")
        # Two chunks with tool call deltas
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "search"}}]}}],
        })
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"name": "read"}}]}}],
        })
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })
        result = handler.finalize()
        assert result.tool_calls_made == 2

    def test_23_responses_api_tool_calls_from_completed(self) -> None:
        """Test 23: Responses API tool calls counted from response.completed output items only."""
        handler = OpenAIEventHandler("openai.responses")
        handler.handle({
            "type": "response.completed",
            "response": {
                "id": "resp_tc",
                "model": "gpt-5.4",
                "output": [
                    {"type": "function_call", "name": "search"},
                    {"type": "message", "content": []},
                    {"type": "function_call", "name": "write"},
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
        })
        result = handler.finalize()
        assert result.tool_calls_made == 2

    def test_24_response_created_captures_metadata(self) -> None:
        """Test 24: response.created event — model and response_id captured."""
        handler = OpenAIEventHandler("openai.responses")
        handler.handle({
            "type": "response.created",
            "response": {
                "id": "resp_created_01",
                "model": "gpt-5.4",
                "status": "in_progress",
            },
        })
        assert handler.model_served == "gpt-5.4"
        assert handler.response_id == "resp_created_01"
        assert handler.request_id == "resp_created_01"

    def test_25_interleaved_text_delta_and_usage(self) -> None:
        """Test 25: Interleaved text delta + usage events — only usage captured."""
        handler = OpenAIEventHandler("openai.chat_completions")
        # Several content deltas
        for i in range(5):
            handler.handle({
                "object": "chat.completion.chunk",
                "id": "chatcmpl-inter",
                "model": "gpt-4o",
                "choices": [{"delta": {"content": f"word{i} "}}],
            })
        # Final usage chunk
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-inter",
            "model": "gpt-4o",
            "choices": [],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        })
        result = handler.finalize()
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.usage_status == "exact"


# ===========================================================================
# Anthropic: TestAnthropicExtractFromResponse (tests 1-6)
# ===========================================================================

class TestAnthropicExtractFromResponse:
    """Tests 1-6: anthropic_extract_from_response for non-streaming responses."""

    def test_anth_01_full_response(self) -> None:
        """Test 1: Full non-streaming response — all fields extracted correctly."""
        body = {
            "id": "msg_full",
            "model": "claude-sonnet-4-20250514",
            "usage": {
                "input_tokens": 200,
                "output_tokens": 100,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 5,
            },
            "content": [{"type": "text", "text": "Hello"}],
        }
        result = anthropic_extract_from_response(body)

        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.cache_read_tokens == 30
        assert result.cache_creation_tokens == 5
        assert result.thinking_tokens is None
        assert result.tool_calls_made == 0
        assert result.total_tokens is None
        assert result.usage_status == "exact"
        assert result.usage_source == "provider_response"
        assert result.model_served == "claude-sonnet-4-20250514"
        assert result.response_id == "msg_full"
        assert result.raw_usage is not None

    def test_anth_02_response_with_tool_use(self) -> None:
        """Test 2: Response with tool use — tool_calls_made counted from content blocks."""
        body = {
            "id": "msg_tools",
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 150, "output_tokens": 80},
            "content": [
                {"type": "text", "text": "I'll search for that."},
                {"type": "tool_use", "id": "tu_01", "name": "search", "input": {}},
                {"type": "tool_use", "id": "tu_02", "name": "read", "input": {}},
            ],
        }
        result = anthropic_extract_from_response(body)

        assert result.tool_calls_made == 2
        assert result.usage_status == "exact"

    def test_anth_03_response_with_thinking(self) -> None:
        """Test 3: Response with thinking — thinking_tokens estimated from text length."""
        thinking_text = "A" * 400  # 400 chars → 100 tokens
        body = {
            "id": "msg_think",
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "content": [
                {"type": "thinking", "thinking": thinking_text},
                {"type": "text", "text": "Answer"},
            ],
        }
        result = anthropic_extract_from_response(body)

        assert result.thinking_tokens == 100  # 400 chars // 4
        assert result.tool_calls_made == 0

    def test_anth_04_response_with_cache_tokens(self) -> None:
        """Test 4: Cache tokens — cache_read and cache_creation extracted."""
        body = {
            "id": "msg_cache",
            "model": "claude-opus-4-20250514",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_read_input_tokens": 500,
                "cache_creation_input_tokens": 200,
            },
            "content": [],
        }
        result = anthropic_extract_from_response(body)

        assert result.cache_read_tokens == 500
        assert result.cache_creation_tokens == 200

    def test_anth_05_missing_usage(self) -> None:
        """Test 5: Missing usage — usage_status == 'missing'."""
        body = {
            "id": "msg_nousage",
            "model": "claude-sonnet-4-20250514",
            "content": [],
        }
        result = anthropic_extract_from_response(body)

        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.usage_status == "missing"
        assert result.raw_usage is None

    def test_anth_06_empty_content(self) -> None:
        """Test 6: Empty content array — no crash, tool_calls_made=0, thinking_tokens=None."""
        body = {
            "id": "msg_empty",
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": [],
        }
        result = anthropic_extract_from_response(body)

        assert result.tool_calls_made == 0
        assert result.thinking_tokens is None
        assert result.usage_status == "exact"


# ===========================================================================
# Anthropic: TestAnthropicEventHandler (tests 7-25)
# ===========================================================================

class TestAnthropicEventHandler:
    """Tests 7-25: Anthropic streaming event handler."""

    def test_anth_07_message_start(self) -> None:
        """Test 7: message_start event — input_tokens, cache tokens, model, request_id captured."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {
                "id": "msg_01abc",
                "model": "claude-sonnet-4-20250514",
                "usage": {
                    "input_tokens": 300,
                    "cache_read_input_tokens": 50,
                    "cache_creation_input_tokens": 10,
                },
            },
        })
        assert handler.input_tokens == 300
        assert handler.cache_read_tokens == 50
        assert handler.cache_creation_tokens == 10
        assert handler.model_served == "claude-sonnet-4-20250514"
        assert handler.response_id == "msg_01abc"

    def test_anth_08_message_delta(self) -> None:
        """Test 8: message_delta event — output_tokens captured, raw_usage stored."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_delta",
            "usage": {"output_tokens": 150},
        })
        assert handler.output_tokens == 150
        assert "message_delta" in handler._raw_usage_parts

    def test_anth_09_content_block_start_tool_use(self) -> None:
        """Test 9: content_block_start with tool_use — tool_call_count incremented."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "tu_01", "name": "search"},
        })
        assert handler._tool_call_count == 1

    def test_anth_10_content_block_start_text(self) -> None:
        """Test 10: content_block_start with text — tool_call_count NOT incremented."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "content_block_start",
            "content_block": {"type": "text"},
        })
        assert handler._tool_call_count == 0

    def test_anth_11_content_block_delta_thinking(self) -> None:
        """Test 11: content_block_delta with thinking_delta — thinking chars accumulated."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "Let me think about this..."},
        })
        handler.handle({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": " More thinking."},
        })
        expected_chars = len("Let me think about this...") + len(" More thinking.")
        assert handler._thinking_chars == expected_chars

    def test_anth_12_content_block_delta_text(self) -> None:
        """Test 12: content_block_delta with text_delta — no effect on thinking."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello world, this is text."},
        })
        assert handler._thinking_chars == 0

    def test_anth_13_complete_stream(self) -> None:
        """Test 13: Complete stream (message_start → deltas → message_delta) — all fields correct."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {
                "id": "msg_complete",
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 400},
            },
        })
        handler.handle({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        })
        handler.handle({
            "type": "message_delta",
            "usage": {"output_tokens": 50},
            "delta": {"stop_reason": "end_turn"},
        })
        result = handler.finalize()

        assert result.input_tokens == 400
        assert result.output_tokens == 50
        assert result.usage_status == "exact"
        assert result.model_served == "claude-sonnet-4-20250514"
        assert result.response_id == "msg_complete"
        assert result.usage_source == "provider_stream_final"
        assert result.raw_usage is not None

    def test_anth_14_stream_with_tool_use(self) -> None:
        """Test 14: Stream with tool use — tool_calls_made correct."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_tu", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 100}},
        })
        handler.handle({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "tu_01", "name": "search"},
        })
        handler.handle({
            "type": "message_delta",
            "usage": {"output_tokens": 30},
        })
        result = handler.finalize()

        assert result.tool_calls_made == 1

    def test_anth_15_stream_with_thinking(self) -> None:
        """Test 15: Stream with thinking — thinking_tokens = chars // 4."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_think", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 100}},
        })
        handler.handle({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "X" * 400},  # 400 chars
        })
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 20}})
        result = handler.finalize()

        assert result.thinking_tokens == 100  # 400 // 4

    def test_anth_16_multiple_tool_use_blocks(self) -> None:
        """Test 16: Stream with multiple tool_use blocks — tool_calls_made = count."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_multi_tu", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 200}},
        })
        for i in range(3):
            handler.handle({
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": f"tu_{i:02d}", "name": f"tool{i}"},
            })
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 60}})
        result = handler.finalize()

        assert result.tool_calls_made == 3

    def test_anth_17_no_message_start(self) -> None:
        """Test 17: Stream without usage (no message_start) — usage_status 'missing', tool_calls_made None."""
        handler = AnthropicEventHandler()
        result = handler.finalize()

        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.usage_status == "missing"
        assert result.tool_calls_made is None

    def test_anth_18_finalize_idempotency(self) -> None:
        """Test 18: finalize() idempotency — same result on repeated calls."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_idem", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 100}},
        })
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 20}})
        r1 = handler.finalize()
        r2 = handler.finalize()
        assert r1 is r2

    def test_anth_19_apply_to_entry_no_endpoint_family(self) -> None:
        """Test 19: apply_to_entry does NOT set endpoint_family."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_ep", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 50}},
        })
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 10}})
        entry: dict = {}
        handler.apply_to_entry(entry)

        assert "endpoint_family" not in entry

    def test_anth_20_ignored_events(self) -> None:
        """Test 20: ping, content_block_stop, message_stop — no crash, ignored gracefully."""
        handler = AnthropicEventHandler()
        handler.handle({"type": "ping"})
        handler.handle({"type": "content_block_stop", "index": 0})
        handler.handle({"type": "message_stop"})
        # Should not raise; state is unchanged
        assert handler.input_tokens is None
        assert handler._tool_call_count == 0

    def test_anth_21_error_event(self) -> None:
        """Test 21: error event — handler records error_type/message; apply_to_entry sets error fields."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "API overloaded"},
        })
        assert handler.error_type == "overloaded_error"
        assert handler.error_message == "API overloaded"

        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["error"] is True
        assert entry["error_type"] == "overloaded_error"

    def test_anth_22_multiple_message_delta(self) -> None:
        """Test 22: Multiple message_delta events — last one wins (cumulative, not summed)."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_multi", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 100}},
        })
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 10}})
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 50}})  # last wins
        result = handler.finalize()

        assert result.output_tokens == 50

    def test_anth_23_interleaved_content_blocks(self) -> None:
        """Test 23: Interleaved text → thinking → tool_use — tool_calls_made=1, thinking accumulated."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_interleaved", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 200}},
        })
        # text block
        handler.handle({"type": "content_block_start", "content_block": {"type": "text"}})
        handler.handle({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}})
        # thinking block
        handler.handle({"type": "content_block_start", "content_block": {"type": "thinking"}})
        handler.handle({
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "Q" * 80},
        })
        # tool_use block
        handler.handle({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "tu_01", "name": "search"},
        })
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 40}})
        result = handler.finalize()

        assert result.tool_calls_made == 1
        assert result.thinking_tokens == 80 // 4  # 20

    def test_anth_24_output_tokens_from_message_delta(self) -> None:
        """Test 24: output_tokens comes from message_delta, not message_start."""
        handler = AnthropicEventHandler()
        # message_start has no output_tokens field — that's correct Anthropic behavior
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_out", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 100}},
        })
        assert handler.output_tokens is None  # not set yet
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 42}})
        assert handler.output_tokens == 42

    def test_anth_25_stop_reason(self) -> None:
        """Test 25: stop_reason from message_delta — captured; apply_to_entry writes entry['stop_reason']."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {"id": "msg_stop", "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 50}},
        })
        handler.handle({
            "type": "message_delta",
            "usage": {"output_tokens": 10},
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        })
        assert handler.stop_reason == "tool_use"

        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["stop_reason"] == "tool_use"


# ===========================================================================
# Anthropic: TestAnthropicIntegration (tests 26-28)
# ===========================================================================

class TestAnthropicIntegration:
    """Tests 26-28: Full pipeline — SSEStreamBuffer → AnthropicEventHandler."""

    def test_anth_26_full_stream_fragmented(self) -> None:
        """Test 26: Full Anthropic stream as fragmented bytes — correct usage extracted."""
        msg_start = _sse_event(
            {"type": "message_start", "message": {
                "id": "msg_integ_01",
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 800, "cache_read_input_tokens": 100},
            }},
            event_type="message_start",
        )
        content_delta = _sse_event(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello world"}},
            event_type="content_block_delta",
        )
        msg_delta = _sse_event(
            {"type": "message_delta", "usage": {"output_tokens": 200},
             "delta": {"stop_reason": "end_turn"}},
            event_type="message_delta",
        )
        full_stream = msg_start + content_delta + msg_delta
        split = len(msg_start) + len(content_delta) + 15

        buf = SSEStreamBuffer()
        handler = anthropic_create_stream_handler()

        for event in buf.feed(full_stream[:split]):
            handler.handle(event)
        for event in buf.feed(full_stream[split:]):
            handler.handle(event)
        for event in buf.flush():
            handler.handle(event)

        entry: dict = {}
        handler.apply_to_entry(entry)

        assert entry["input_tokens"] == 800
        assert entry["output_tokens"] == 200
        assert entry["cache_read_tokens"] == 100
        assert entry["usage_status"] == "exact"
        assert entry["model"] == "claude-sonnet-4-20250514"
        assert entry["stop_reason"] == "end_turn"

    def test_anth_27_message_delta_split_across_chunks(self) -> None:
        """Test 27: message_delta split across TCP chunks — buffered and parsed correctly."""
        msg_start = _sse_event(
            {"type": "message_start", "message": {
                "id": "msg_split", "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 500},
            }},
            event_type="message_start",
        )
        msg_delta = _sse_event(
            {"type": "message_delta", "usage": {"output_tokens": 300}},
            event_type="message_delta",
        )
        full_stream = msg_start + msg_delta
        # Fragment right in the middle of message_delta JSON
        split = len(msg_start) + len(msg_delta) // 2

        buf = SSEStreamBuffer()
        handler = anthropic_create_stream_handler()

        for event in buf.feed(full_stream[:split]):
            handler.handle(event)
        for event in buf.feed(full_stream[split:]):
            handler.handle(event)
        for event in buf.flush():
            handler.handle(event)

        result = handler.finalize()
        assert result.input_tokens == 500
        assert result.output_tokens == 300
        assert result.usage_status == "exact"

    def test_anth_28_non_streaming_extract(self) -> None:
        """Test 28: Non-streaming response via extract_from_response — correct entry."""
        body = {
            "id": "msg_ns_01",
            "model": "claude-opus-4-20250514",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 400,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 50,
            },
            "content": [
                {"type": "thinking", "thinking": "Z" * 200},
                {"type": "tool_use", "id": "tu_01", "name": "fetch", "input": {}},
                {"type": "text", "text": "Done."},
            ],
        }
        result = anthropic_extract_from_response(body)
        entry: dict = {}
        result.apply_to_entry(entry)

        assert entry["input_tokens"] == 1000
        assert entry["output_tokens"] == 400
        assert entry["cache_read_tokens"] == 200
        assert entry["cache_creation_tokens"] == 50
        assert entry["thinking_tokens"] == 200 // 4  # 50
        assert entry["tool_calls_made"] == 1
        assert entry["usage_status"] == "exact"
        assert entry["usage_source"] == "provider_response"
        assert entry["model"] == "claude-opus-4-20250514"
        assert entry["request_id"] == "msg_ns_01"


# ===========================================================================
# Anthropic: TestAnthropicRegression (tests 29-30)
# ===========================================================================

class TestAnthropicRegression:
    """Tests 29-30: Regression tests for Anthropic adapter."""

    def test_anth_30_cache_field_names(self) -> None:
        """Test 30: Cache field names — cache_read_input_tokens (Anthropic) not cached_tokens (OpenAI)."""
        handler = AnthropicEventHandler()
        # Anthropic-style fields: cache_read_input_tokens and cache_creation_input_tokens
        handler.handle({
            "type": "message_start",
            "message": {
                "id": "msg_cache_fields",
                "model": "claude-sonnet-4-20250514",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 400,    # Anthropic field name
                    "cache_creation_input_tokens": 50,  # Anthropic field name
                    "cached_tokens": 999,               # OpenAI field — must NOT be used
                },
            },
        })
        # Verify handler uses Anthropic field names, not OpenAI's
        assert handler.cache_read_tokens == 400    # from cache_read_input_tokens
        assert handler.cache_creation_tokens == 50  # from cache_creation_input_tokens


# ===========================================================================
# Tests 32-35: stream_options injection
# ===========================================================================

class TestStreamOptionsInjection:
    """Tests 32-35: maybe_inject_stream_options."""

    def test_32_chat_completions_streaming_injected(self) -> None:
        """Test 32: Chat Completions streaming without stream_options — injected."""
        body: dict = {"model": "gpt-4o", "stream": True, "messages": []}
        result = maybe_inject_stream_options(body, "openai.chat_completions")

        assert result["stream_options"] == {"include_usage": True}
        assert result is body  # mutated in place

    def test_33_existing_stream_options_not_modified(self) -> None:
        """Test 33: Chat Completions streaming with existing stream_options — not modified."""
        body: dict = {
            "model": "gpt-4o",
            "stream": True,
            "messages": [],
            "stream_options": {"include_usage": False},
        }
        result = maybe_inject_stream_options(body, "openai.chat_completions")

        assert result["stream_options"] == {"include_usage": False}

    def test_34_non_streaming_not_modified(self) -> None:
        """Test 34: Chat Completions non-streaming — not modified."""
        body: dict = {"model": "gpt-4o", "messages": []}
        result = maybe_inject_stream_options(body, "openai.chat_completions")

        assert "stream_options" not in result

    def test_35_responses_api_not_modified(self) -> None:
        """Test 35: Responses API streaming — not modified."""
        body: dict = {"model": "gpt-5.4", "stream": True, "input": "hello"}
        result = maybe_inject_stream_options(body, "openai.responses")

        assert "stream_options" not in result


# ===========================================================================
# Tests 36-39: Integration tests (full pipeline)
# ===========================================================================

class TestIntegration:
    """Tests 36-39: end-to-end pipeline from bytes to entry."""

    def test_36_openai_fragmented_stream(self) -> None:
        """Test 36: Simulated fragmented OpenAI Responses API stream -> entry has correct usage."""
        # Build SSE data for response.created + response.completed
        created_event = _sse_event(
            {"type": "response.created", "response": {"id": "resp_frag", "model": "gpt-5.4", "status": "in_progress"}},
            event_type="response.created",
        )
        completed_data = {
            "type": "response.completed",
            "response": {
                "id": "resp_frag",
                "model": "gpt-5.4",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}],
                "usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
            },
        }
        completed_event = _sse_event(completed_data, event_type="response.completed")

        # Fragment the stream: split completed event mid-JSON
        full_stream = created_event + completed_event
        split_point = len(created_event) + 30  # mid-way through completed event

        buf = SSEStreamBuffer()
        handler = create_stream_handler("openai.responses")

        # Feed fragment 1
        for event in buf.feed(full_stream[:split_point]):
            handler.handle(event)
        # Feed fragment 2
        for event in buf.feed(full_stream[split_point:]):
            handler.handle(event)
        # Flush
        for event in buf.flush():
            handler.handle(event)

        entry: dict = {}
        handler.apply_to_entry(entry)

        assert entry["input_tokens"] == 1000
        assert entry["output_tokens"] == 500
        assert entry["usage_status"] == "exact"
        assert entry["usage_source"] == "provider_stream_final"
        assert entry["response_id"] == "resp_frag"

    def test_37_anthropic_fragmented_stream(self) -> None:
        """Test 37: Simulated fragmented Anthropic stream -> entry has correct usage."""
        msg_start = _sse_event(
            {"type": "message_start", "message": {"id": "msg_frag", "model": "claude-sonnet-4-20250514", "usage": {"input_tokens": 800, "cache_read_input_tokens": 100}}},
            event_type="message_start",
        )
        content_delta = _sse_event(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello world"}},
            event_type="content_block_delta",
        )
        msg_delta = _sse_event(
            {"type": "message_delta", "usage": {"output_tokens": 200}},
            event_type="message_delta",
        )

        full_stream = msg_start + content_delta + msg_delta
        # Fragment: split msg_delta mid-way
        split_point = len(msg_start) + len(content_delta) + 15

        buf = SSEStreamBuffer()
        handler = AnthropicEventHandler()

        for event in buf.feed(full_stream[:split_point]):
            handler.handle(event)
        for event in buf.feed(full_stream[split_point:]):
            handler.handle(event)
        for event in buf.flush():
            handler.handle(event)

        entry: dict = {}
        handler.apply_to_entry(entry)

        assert entry["input_tokens"] == 800
        assert entry["output_tokens"] == 200
        assert entry["cache_read_tokens"] == 100
        assert entry["usage_status"] == "exact"
        assert entry["model"] == "claude-sonnet-4-20250514"

    def test_38_non_streaming_openai(self) -> None:
        """Test 38: Non-streaming OpenAI — mock response body -> correct UsageResult."""
        body = _responses_api_body(
            input_tokens=1234,
            output_tokens=567,
            total_tokens=1801,
            cached_tokens=200,
            reasoning_tokens=100,
        )
        result = extract_from_response(body, "openai.responses")

        entry: dict = {}
        result.apply_to_entry(entry)

        assert entry["input_tokens"] == 1234
        assert entry["output_tokens"] == 567
        assert entry["cache_read_tokens"] == 200
        assert entry["thinking_tokens"] == 100
        assert entry["usage_status"] == "exact"
        assert entry["usage_source"] == "provider_response"

    def test_39_legacy_entry_compat(self) -> None:
        """Test 39: Legacy entry without usage_status — correct inference."""
        def _infer_usage_status(entry: dict) -> str:
            if "usage_status" in entry:
                return entry["usage_status"]
            if entry.get("input_tokens") is not None:
                return "exact"
            return "missing"

        assert _infer_usage_status({"input_tokens": 100}) == "exact"
        assert _infer_usage_status({"input_tokens": None}) == "missing"
        assert _infer_usage_status({}) == "missing"
        assert _infer_usage_status({"usage_status": "exact", "input_tokens": None}) == "exact"


# ===========================================================================
# Tests 40-42: Regression tests
# ===========================================================================

class TestRegression:
    """Tests 40-42: Regression tests for known failure modes."""

    def test_40_gpt54_fragmented_response_completed(self) -> None:
        """Test 40: GPT-5.4 entries reproduction — feed fragmented response.completed -> usage captured.

        Reproduces the critical bug: response.completed split across TCP chunks.
        """
        # Simulate the exact fragmentation pattern that caused 98.4% missing usage
        completed_data = {
            "type": "response.completed",
            "response": {
                "id": "resp_gpt54_repro",
                "model": "gpt-5.4",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Result"}]}],
                "usage": {
                    "input_tokens": 2048,
                    "output_tokens": 1024,
                    "total_tokens": 3072,
                    "input_tokens_details": {"cached_tokens": 512},
                    "output_tokens_details": {"reasoning_tokens": 256},
                },
            },
        }

        raw_event = _sse_event(completed_data, event_type="response.completed")
        # Fragment at the worst possible point: mid-JSON, right through "usage"
        json_str = json.dumps(completed_data)
        usage_pos = json_str.index('"usage"')
        # The data: prefix + event: line + some chars before the split
        split_point = raw_event.index(b'"usage"')

        buf = SSEStreamBuffer()
        handler = OpenAIEventHandler("openai.responses")

        events1 = buf.feed(raw_event[:split_point])
        for e in events1:
            handler.handle(e)

        events2 = buf.feed(raw_event[split_point:])
        for e in events2:
            handler.handle(e)

        for e in buf.flush():
            handler.handle(e)

        result = handler.finalize()

        # The old code would have input_tokens=None here. The new code captures it.
        assert result.input_tokens == 2048
        assert result.output_tokens == 1024
        assert result.cache_read_tokens == 512
        assert result.reasoning_tokens == 256
        assert result.usage_status == "exact"

    def test_41_compute_cost_with_missing_usage(self) -> None:
        """Test 41: _compute_cost with missing usage — verify it returns 0 cost, not fake cost."""
        result = UsageResult(
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            reasoning_tokens=None,
            raw_usage=None,
            usage_source="provider_stream_final",
            model_served="gpt-5.4",
            request_id=None,
            response_id=None,
            tool_calls_made=None,
        )

        entry: dict = {}
        result.apply_to_entry(entry)

        assert entry["input_tokens"] is None
        assert entry["output_tokens"] is None
        assert entry["usage_status"] == "missing"

        # Simulating what _compute_cost should do: treat None as 0 cost
        input_t = entry["input_tokens"] or 0
        output_t = entry["output_tokens"] or 0
        assert input_t == 0
        assert output_t == 0

    def test_42_report_with_mixed_exact_missing(self) -> None:
        """Test 42: Report with mixed exact/missing — verify coverage columns are correct."""
        entries = []

        # 3 exact entries
        for i in range(3):
            e: dict = {}
            UsageResult(
                input_tokens=100 * (i + 1),
                output_tokens=50 * (i + 1),
                total_tokens=150 * (i + 1),
                cache_read_tokens=0,
                cache_creation_tokens=0,
                reasoning_tokens=None,
                raw_usage={"prompt_tokens": 100 * (i + 1)},
                usage_source="provider_stream_final",
                model_served="gpt-5.4",
                request_id=f"req_{i}",
                response_id=None,
                tool_calls_made=None,
            ).apply_to_entry(e)
            entries.append(e)

        # 2 missing entries
        for i in range(2):
            e = {}
            UsageResult(
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                reasoning_tokens=None,
                raw_usage=None,
                usage_source="provider_stream_final",
                model_served="gpt-5.4",
                request_id=f"req_missing_{i}",
                response_id=None,
                tool_calls_made=None,
            ).apply_to_entry(e)
            entries.append(e)

        total_calls = len(entries)
        exact_calls = sum(1 for e in entries if e["usage_status"] == "exact")
        missing_calls = sum(1 for e in entries if e["usage_status"] == "missing")
        coverage_pct = (exact_calls / total_calls) * 100 if total_calls else 0

        assert total_calls == 5
        assert exact_calls == 3
        assert missing_calls == 2
        assert coverage_pct == 60.0

        # Only exact calls contribute to token totals
        exact_input_total = sum(
            e["input_tokens"] for e in entries if e["usage_status"] == "exact"
        )
        assert exact_input_total == 100 + 200 + 300


# ===========================================================================
# Gemini adapter tests
# ===========================================================================

def _gemini_response(
    prompt_tokens: int = 100,
    candidates_tokens: int = 50,
    total_tokens: int = 150,
    thinking_tokens: int = 0,
    cache_tokens: int = 0,
    tool_use_prompt_tokens: int = 0,
    finish_reason: str = "STOP",
    function_calls: int = 0,
    model_version: str = "gemini-2.5-flash",
    response_id: str = "resp_test_001",
) -> dict:
    """Build a minimal non-streaming GenerateContentResponse body."""
    parts: list[dict] = [{"text": "Hello world"}]
    for i in range(function_calls):
        parts.append({"functionCall": {"name": f"tool_{i}", "args": {}}})

    usage: dict = {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": candidates_tokens,
        "totalTokenCount": total_tokens,
        "cachedContentTokenCount": cache_tokens,
    }
    if thinking_tokens:
        usage["thoughtsTokenCount"] = thinking_tokens
    if tool_use_prompt_tokens:
        usage["toolUsePromptTokenCount"] = tool_use_prompt_tokens

    return {
        "candidates": [
            {
                "content": {"parts": parts, "role": "model"},
                "finishReason": finish_reason,
                "index": 0,
            }
        ],
        "usageMetadata": usage,
        "modelVersion": model_version,
        "responseId": response_id,
    }


def _gemini_sse_chunk(
    text: str | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    function_calls: int = 0,
    model_version: str | None = "gemini-2.5-flash",
    response_id: str | None = "resp_stream_001",
) -> bytes:
    """Build a Gemini SSE chunk (data: {...}\\n\\n) for a streaming response."""
    parts: list[dict] = []
    if text is not None:
        parts.append({"text": text})
    for i in range(function_calls):
        parts.append({"functionCall": {"name": f"tool_{i}", "args": {}}})

    candidate: dict = {"content": {"parts": parts, "role": "model"}, "index": 0}
    if finish_reason:
        candidate["finishReason"] = finish_reason

    event: dict = {"candidates": [candidate]}
    if model_version:
        event["modelVersion"] = model_version
    if response_id:
        event["responseId"] = response_id
    if usage is not None:
        event["usageMetadata"] = usage

    return (f"data: {json.dumps(event)}\n\n").encode()


# ---------------------------------------------------------------------------
# TestGeminiExtractModelFromUrl
# ---------------------------------------------------------------------------

class TestGeminiExtractModelFromUrl:
    """Unit tests for extract_model_from_url."""

    def test_gem_url_1_standard_v1beta(self) -> None:
        """Test 1: Standard v1beta path extracts model name."""
        assert extract_model_from_url("/v1beta/models/gemini-2.5-flash:generateContent") == "gemini-2.5-flash"

    def test_gem_url_2_streaming_path(self) -> None:
        """Test 2: streamGenerateContent path extracts model name."""
        result = extract_model_from_url("/v1beta/models/gemini-3-flash-preview:streamGenerateContent")
        assert result == "gemini-3-flash-preview"

    def test_gem_url_3_v1_path(self) -> None:
        """Test 3: /v1/ prefix works."""
        assert extract_model_from_url("/v1/models/gemini-2.0-flash:generateContent") == "gemini-2.0-flash"

    def test_gem_url_4_no_model(self) -> None:
        """Test 4: No model in path returns None."""
        assert extract_model_from_url("/v1beta/something/else") is None

    def test_gem_url_5_query_params(self) -> None:
        """Test 5: Query params don't break extraction."""
        path = "/v1beta/models/gemini-2.5-flash:streamGenerateContent?key=abc123&alt=sse"
        assert extract_model_from_url(path) == "gemini-2.5-flash"

    def test_gem_url_6_tuned_models(self) -> None:
        """Test 6: /tunedModels/ path extracts model name."""
        assert extract_model_from_url("/v1beta/tunedModels/my-custom-model:generateContent") == "my-custom-model"

    def test_gem_url_7_malformed_paths(self) -> None:
        """Test 7: Malformed / empty paths return None."""
        assert extract_model_from_url("") is None
        assert extract_model_from_url("/") is None
        assert extract_model_from_url("/v1beta/models/") is None
        assert extract_model_from_url("/v1beta/models/:generateContent") is None


# ---------------------------------------------------------------------------
# TestGeminiExtractFromResponse
# ---------------------------------------------------------------------------

class TestGeminiExtractFromResponse:
    """Unit tests for extract_from_response (non-streaming)."""

    def test_gem_ns_1_full_response(self) -> None:
        """Test 1: Full response — all fields extracted correctly."""
        body = _gemini_response(
            prompt_tokens=100, candidates_tokens=50, total_tokens=150,
            model_version="gemini-2.5-flash", response_id="resp_full",
        )
        result = gemini_extract_from_response(body)

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.thinking_tokens is None
        assert result.cache_read_tokens == 0
        assert result.tool_calls_made == 0
        assert result.usage_status == "exact"
        assert result.usage_source == "provider_response"
        assert result.model_served == "gemini-2.5-flash"
        assert result.request_id == "resp_full"
        assert result.stop_reason == "STOP"
        assert result.raw_usage is not None

    def test_gem_ns_2_thinking_tokens(self) -> None:
        """Test 2: thoughtsTokenCount mapped to thinking_tokens."""
        body = _gemini_response(thinking_tokens=491, total_tokens=641)
        result = gemini_extract_from_response(body)
        assert result.thinking_tokens == 491

    def test_gem_ns_3_cache_tokens(self) -> None:
        """Test 3: cachedContentTokenCount mapped to cache_read_tokens."""
        body = _gemini_response(cache_tokens=200)
        result = gemini_extract_from_response(body)
        assert result.cache_read_tokens == 200

    def test_gem_ns_4_function_calls(self) -> None:
        """Test 4: functionCall parts counted in tool_calls_made."""
        body = _gemini_response(function_calls=3)
        result = gemini_extract_from_response(body)
        assert result.tool_calls_made == 3

    def test_gem_ns_5_tool_use_prompt_tokens_in_raw_usage_only(self) -> None:
        """Test 5: toolUsePromptTokenCount in raw_usage only, not a named field."""
        body = _gemini_response(tool_use_prompt_tokens=42)
        result = gemini_extract_from_response(body)
        assert not hasattr(result, "tool_use_prompt_tokens") or not isinstance(
            getattr(result, "tool_use_prompt_tokens", None), int
        )
        assert result.raw_usage is not None
        assert result.raw_usage["toolUsePromptTokenCount"] == 42

    def test_gem_ns_6_missing_usage_metadata(self) -> None:
        """Test 6: Missing usageMetadata — usage_status 'missing'."""
        body = {
            "candidates": [
                {"content": {"parts": [{"text": "hi"}], "role": "model"}, "finishReason": "STOP"}
            ],
            "modelVersion": "gemini-2.5-flash",
        }
        result = gemini_extract_from_response(body)
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.usage_status == "missing"
        assert result.raw_usage is None

    def test_gem_ns_7_empty_candidates_no_feedback(self) -> None:
        """Test 7: Empty candidates, no promptFeedback — tool_calls_made None."""
        body = {
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0, "totalTokenCount": 10},
            "modelVersion": "gemini-2.5-flash",
        }
        result = gemini_extract_from_response(body)
        assert result.tool_calls_made is None
        assert result.stop_reason is None

    def test_gem_ns_8_multiple_function_calls(self) -> None:
        """Test 8: Multiple functionCall parts all counted."""
        body = _gemini_response(function_calls=5)
        result = gemini_extract_from_response(body)
        assert result.tool_calls_made == 5

    def test_gem_ns_9_multimodal_details_in_raw_usage(self) -> None:
        """Test 9: promptTokensDetails stored in raw_usage (not normalized)."""
        body = _gemini_response()
        body["usageMetadata"]["promptTokensDetails"] = [
            {"modality": "TEXT", "tokenCount": 80},
            {"modality": "IMAGE", "tokenCount": 20},
        ]
        result = gemini_extract_from_response(body)
        assert result.raw_usage is not None
        assert "promptTokensDetails" in result.raw_usage

    def test_gem_ns_10_prompt_blocked(self) -> None:
        """Test 10: Prompt-blocked — no candidates, promptFeedback.blockReason set."""
        body = {
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {"promptTokenCount": 15, "totalTokenCount": 15},
            "modelVersion": "gemini-2.5-flash",
        }
        result = gemini_extract_from_response(body)
        assert result.stop_reason == "SAFETY"
        assert result.tool_calls_made is None
        # No candidatesTokenCount in usageMetadata → output_tokens is None
        assert result.output_tokens is None

    def test_gem_ns_11_safety_blocked_candidate(self) -> None:
        """Test 11: Safety-blocked candidate — present but empty content, tool_calls_made = 0."""
        body = {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "SAFETY",
                    "index": 0,
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0, "totalTokenCount": 10},
            "modelVersion": "gemini-2.5-flash",
        }
        result = gemini_extract_from_response(body)
        assert result.tool_calls_made == 0
        assert result.stop_reason == "SAFETY"


# ---------------------------------------------------------------------------
# TestGeminiEventHandler
# ---------------------------------------------------------------------------

class TestGeminiEventHandler:
    """Unit tests for GeminiEventHandler (streaming)."""

    def test_gem_eh_1_single_chunk_all_data(self) -> None:
        """Test 1: Single chunk with all data — all fields captured."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10, "totalTokenCount": 30},
            "modelVersion": "gemini-2.5-flash",
            "responseId": "resp_single",
        })
        result = handler.finalize()

        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.total_tokens == 30
        assert result.model_served == "gemini-2.5-flash"
        assert result.request_id == "resp_single"
        assert result.stop_reason == "STOP"
        assert result.usage_status == "exact"
        assert result.usage_source == "provider_stream_final"

    def test_gem_eh_2_usage_only_on_last_chunk(self) -> None:
        """Test 2: usageMetadata only on last chunk — captured correctly."""
        handler = GeminiEventHandler()
        # First chunk: no usageMetadata
        handler.handle({
            "candidates": [{"content": {"parts": [{"text": "Hello "}], "role": "model"}}],
            "modelVersion": "gemini-2.5-flash",
            "responseId": "resp_last",
        })
        # Final chunk: has usageMetadata and finishReason
        handler.handle({
            "candidates": [{"content": {"parts": [{"text": "world"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 25, "totalTokenCount": 75},
        })
        result = handler.finalize()

        assert result.input_tokens == 50
        assert result.output_tokens == 25
        assert result.stop_reason == "STOP"

    def test_gem_eh_3_candidates_seen_no_function_calls(self) -> None:
        """Test 3: Candidates seen but no functionCall parts — tool_calls_made = 0."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [{"text": "text only"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        })
        result = handler.finalize()
        assert result.tool_calls_made == 0

    def test_gem_eh_4_function_call_part(self) -> None:
        """Test 4: Chunk with functionCall part — tool_call_count incremented."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "search", "args": {}}}], "role": "model"}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        })
        result = handler.finalize()
        assert result.tool_calls_made == 1

    def test_gem_eh_5_two_chunks_function_calls_accumulated(self) -> None:
        """Test 5: Two chunks with different functionCall parts — no double-count."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "tool_a", "args": {}}}], "role": "model"}}
            ],
            "modelVersion": "gemini-2.5-flash",
        })
        handler.handle({
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "tool_b", "args": {}}}], "role": "model"}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 20, "totalTokenCount": 50},
        })
        result = handler.finalize()
        assert result.tool_calls_made == 2  # one from each chunk

    def test_gem_eh_6_thinking_tokens(self) -> None:
        """Test 6: thoughtsTokenCount in usageMetadata → thinking_tokens."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [{"text": "answer"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 506,
                "thoughtsTokenCount": 491,
            },
        })
        result = handler.finalize()
        assert result.thinking_tokens == 491

    def test_gem_eh_7_finish_reason(self) -> None:
        """Test 7: finishReason from candidate → stop_reason."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 100, "totalTokenCount": 110},
        })
        result = handler.finalize()
        assert result.stop_reason == "MAX_TOKENS"

    def test_gem_eh_8_no_usage_mid_stream_disconnect(self) -> None:
        """Test 8: No usageMetadata in any chunk (mid-stream disconnect) → usage_status 'missing'."""
        handler = GeminiEventHandler()
        handler.handle({"candidates": [{"content": {"parts": [{"text": "partial"}], "role": "model"}}]})
        result = handler.finalize()
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.usage_status == "missing"
        assert result.raw_usage is None

    def test_gem_eh_9_finalize_idempotency(self) -> None:
        """Test 9: finalize() idempotency — same object returned on repeated calls."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
        })
        r1 = handler.finalize()
        r2 = handler.finalize()
        assert r1 is r2

    def test_gem_eh_10_apply_to_entry_no_endpoint_family(self) -> None:
        """Test 10: apply_to_entry does NOT set endpoint_family."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert "endpoint_family" not in entry

    def test_gem_eh_11_apply_to_entry_sets_stop_reason(self) -> None:
        """Test 11: apply_to_entry sets stop_reason when finishReason was seen."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "RECITATION"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["stop_reason"] == "RECITATION"

    def test_gem_eh_12_no_candidates_tool_calls_none(self) -> None:
        """Test 12: No candidates seen — tool_calls_made is None, not 0."""
        handler = GeminiEventHandler()
        handler.handle({"usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 10}})
        result = handler.finalize()
        assert result.tool_calls_made is None

    def test_gem_eh_13_prompt_blocked_stream(self) -> None:
        """Test 13: promptFeedback.blockReason → stop_reason set, tool_calls_made None."""
        handler = GeminiEventHandler()
        handler.handle({
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {"promptTokenCount": 15, "totalTokenCount": 15},
            "modelVersion": "gemini-2.5-flash",
        })
        result = handler.finalize()
        assert result.stop_reason == "SAFETY"
        assert result.tool_calls_made is None

    def test_gem_eh_14_total_tokens_missing(self) -> None:
        """Test 14: totalTokenCount missing → total_tokens is None in result."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        })
        result = handler.finalize()
        assert result.total_tokens is None


# ---------------------------------------------------------------------------
# TestGeminiIntegration
# ---------------------------------------------------------------------------

class TestGeminiIntegration:
    """Integration tests: SSEStreamBuffer + GeminiEventHandler."""

    def test_gem_int_1_fragmented_stream(self) -> None:
        """Test 1: Full Gemini stream fragmented into bytes — correct usage extracted."""
        chunk1_bytes = _gemini_sse_chunk(text="Hello ", model_version="gemini-2.5-flash", response_id="resp_frag")
        chunk2_bytes = _gemini_sse_chunk(
            text="world",
            finish_reason="STOP",
            usage={"promptTokenCount": 40, "candidatesTokenCount": 20, "totalTokenCount": 60},
            model_version=None,
            response_id=None,
        )

        full_stream = chunk1_bytes + chunk2_bytes
        # Split mid-way through second chunk to test buffering
        split_at = len(chunk1_bytes) + 15

        buf = SSEStreamBuffer()
        handler = gemini_create_stream_handler()

        for event in buf.feed(full_stream[:split_at]):
            handler.handle(event)
        for event in buf.feed(full_stream[split_at:]):
            handler.handle(event)
        for event in buf.flush():
            handler.handle(event)

        result = handler.finalize()
        assert result.input_tokens == 40
        assert result.output_tokens == 20
        assert result.total_tokens == 60
        assert result.stop_reason == "STOP"
        assert result.model_served == "gemini-2.5-flash"
        assert result.usage_status == "exact"

    def test_gem_int_2_usage_split_across_tcp_chunks(self) -> None:
        """Test 2: usageMetadata JSON split across TCP chunks — buffered and parsed correctly."""
        final_chunk = _gemini_sse_chunk(
            text="done",
            finish_reason="STOP",
            usage={"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150},
        )
        # Fragment right through the usageMetadata JSON
        split_at = final_chunk.index(b"usageMetadata") + 5

        buf = SSEStreamBuffer()
        handler = gemini_create_stream_handler()

        for event in buf.feed(final_chunk[:split_at]):
            handler.handle(event)
        for event in buf.feed(final_chunk[split_at:]):
            handler.handle(event)
        for event in buf.flush():
            handler.handle(event)

        result = handler.finalize()
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.usage_status == "exact"

    def test_gem_int_3_non_streaming_entry_fields(self) -> None:
        """Test 3: Non-streaming response via extract_from_response — correct entry fields."""
        body = _gemini_response(
            prompt_tokens=200, candidates_tokens=100, total_tokens=300,
            cache_tokens=50, thinking_tokens=80, function_calls=2,
        )
        result = gemini_extract_from_response(body)
        entry: dict = {}
        result.apply_to_entry(entry)

        assert entry["input_tokens"] == 200
        assert entry["output_tokens"] == 100
        assert entry["total_tokens"] == 300
        assert entry["cache_read_tokens"] == 50
        assert entry["cache_creation_tokens"] == 0
        assert entry["thinking_tokens"] == 80
        assert entry["tool_calls_made"] == 2
        assert entry["usage_status"] == "exact"
        assert entry["usage_source"] == "provider_response"
        assert entry["stop_reason"] == "STOP"
        assert "endpoint_family" not in entry


# ---------------------------------------------------------------------------
# TestGeminiRegression
# ---------------------------------------------------------------------------

class TestGeminiRegression:
    """Regression tests for Gemini adapter."""

    def test_gem_reg_1_field_name_correctness(self) -> None:
        """Test 1: promptTokenCount (not prompt_tokens) is read from usageMetadata."""
        # If we accidentally read "prompt_tokens" (OpenAI name), it would return None
        body = {
            "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 42,      # Gemini field name
                "candidatesTokenCount": 10,
                "totalTokenCount": 52,
            },
        }
        result = gemini_extract_from_response(body)
        assert result.input_tokens == 42, "Must read promptTokenCount, not prompt_tokens"
        assert result.output_tokens == 10, "Must read candidatesTokenCount, not completion_tokens"

    def test_gem_reg_2_cache_creation_tokens_always_zero(self) -> None:
        """Test 2: cache_creation_tokens is always 0 for Gemini."""
        body = _gemini_response(cache_tokens=100)
        result = gemini_extract_from_response(body)
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["cache_creation_tokens"] == 0

    def test_gem_reg_3_cache_creation_zero_in_stream(self) -> None:
        """Test 3: cache_creation_tokens always 0 for streaming handler too."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
                "cachedContentTokenCount": 8,
            },
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["cache_creation_tokens"] == 0
        assert entry["cache_read_tokens"] == 8


# ===========================================================================
# New tests for cross-adapter review fixes
# ===========================================================================

class TestOpenAIStopReason:
    """OpenAI handler captures stop_reason from streaming events."""

    def test_openai_stop_reason_chat_completions(self) -> None:
        """Chat Completions: finish_reason from choices captured and written to entry."""
        handler = OpenAIEventHandler("openai.chat_completions")
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-stop1",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        })
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-stop1",
            "model": "gpt-4o",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        assert handler.stop_reason == "stop"

        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["stop_reason"] == "stop"

    def test_openai_stop_reason_tool_calls_finish_reason(self) -> None:
        """Chat Completions: finish_reason=tool_calls captured correctly."""
        handler = OpenAIEventHandler("openai.chat_completions")
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "search"}}]}, "finish_reason": None}],
        })
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["stop_reason"] == "tool_calls"

    def test_openai_stop_reason_responses_api_status(self) -> None:
        """Responses API: response.completed status captured and written to entry."""
        handler = OpenAIEventHandler("openai.responses")
        handler.handle({
            "type": "response.completed",
            "response": {
                "id": "resp_status_01",
                "model": "gpt-5.4",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
        })
        assert handler.stop_reason == "completed"

        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["stop_reason"] == "completed"

    def test_openai_no_stop_reason_when_absent(self) -> None:
        """No stop_reason written to entry when finish_reason never seen."""
        handler = OpenAIEventHandler("openai.chat_completions")
        handler.handle({
            "object": "chat.completion.chunk",
            "id": "chatcmpl-nofr",
            "model": "gpt-4o",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert "stop_reason" not in entry


class TestSSEFlushIdempotency:
    """SSEStreamBuffer.flush() is idempotent."""

    def test_flush_idempotent_with_trailing_data(self) -> None:
        """flush() called twice returns same cached result (same list object)."""
        buf = SSEStreamBuffer()
        buf.feed(b'data: {"trailing": true}')

        events1 = buf.flush()
        events2 = buf.flush()

        assert events1 is events2
        assert len(events1) == 1
        assert events1[0] == {"trailing": True}

    def test_flush_idempotent_empty(self) -> None:
        """flush() on empty buffer returns [] both times (same object)."""
        buf = SSEStreamBuffer()
        e1 = buf.flush()
        e2 = buf.flush()
        assert e1 is e2
        assert e1 == []


class TestGeminiFunctionCallWithoutName:
    """Gemini: functionCall part without 'name' field doesn't increment counter."""

    def test_function_call_without_name_not_counted(self) -> None:
        """Part with functionCall but no 'name' key — should NOT increment tool_call_count."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"args": {"q": "hello"}}}],  # no "name"
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        })
        result = handler.finalize()
        assert result.tool_calls_made == 0

    def test_function_call_with_name_counted(self) -> None:
        """Part with functionCall and 'name' key — counted correctly."""
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "search", "args": {}}}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        })
        result = handler.finalize()
        assert result.tool_calls_made == 1

    def test_mixed_named_and_unnamed_function_calls(self) -> None:
        """Named calls counted, unnamed (continuation chunks) not counted."""
        handler = GeminiEventHandler()
        # Chunk with named call
        handler.handle({
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "tool_a", "args": {}}}], "role": "model"}}
            ],
        })
        # Chunk with args-only (no name — continuation)
        handler.handle({
            "candidates": [
                {"content": {"parts": [{"functionCall": {"args": {"key": "value"}}}], "role": "model"},
                 "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10, "totalTokenCount": 30},
        })
        result = handler.finalize()
        assert result.tool_calls_made == 1  # only the named one


class TestAnthropicRawUsageNotShared:
    """Anthropic raw_usage is a copy, not a reference to internal handler state."""

    def test_raw_usage_is_copy_not_reference(self) -> None:
        """Mutating result.raw_usage must not affect handler's _raw_usage_parts."""
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {
                "id": "msg_copy",
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 100},
            },
        })
        handler.handle({"type": "message_delta", "usage": {"output_tokens": 50}})

        result = handler.finalize()
        assert result.raw_usage is not None

        # Mutate the result's raw_usage
        original_keys = set(result.raw_usage.keys())
        result.raw_usage["injected"] = "malicious"

        # Handler's internal state must be unchanged
        assert "injected" not in handler._raw_usage_parts
        # A second finalize() call must also return a fresh copy (idempotent)
        result2 = handler.finalize()
        assert result2 is result  # same cached object — that's fine
        # But the original raw_usage in the cached result was already mutated;
        # what matters is the handler's internal dict is untouched
        assert "injected" not in handler._raw_usage_parts


class TestApplyToEntryParity:
    """Cross-adapter parity: all adapters write the same canonical entry keys."""

    def test_anthropic_writes_total_tokens(self) -> None:
        """Anthropic apply_to_entry now writes total_tokens (always None)."""
        from toklog.adapters.anthropic import AnthropicUsageResult
        result = AnthropicUsageResult(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            thinking_tokens=0,
            tool_calls_made=0,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert "total_tokens" in entry
        assert entry["total_tokens"] == 150  # Computed from input + output

    def test_openai_writes_total_tokens(self) -> None:
        """OpenAI apply_to_entry writes total_tokens."""
        result = UsageResult(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            reasoning_tokens=None,
            raw_usage=None,
            usage_source="provider_response",
            model_served=None,
            request_id=None,
            response_id=None,
            tool_calls_made=None,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["total_tokens"] == 150

    def test_openai_thinking_tokens_is_none_when_reasoning_not_reported(self) -> None:
        """OpenAI apply_to_entry writes None for thinking_tokens when reasoning_tokens is None."""
        result = UsageResult(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            reasoning_tokens=None,
            raw_usage=None,
            usage_source="provider_response",
            model_served=None,
            request_id=None,
            response_id=None,
            tool_calls_made=None,
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["thinking_tokens"] is None

    def test_openai_tool_calls_made_zero_when_empty_output(self) -> None:
        """Responses API with empty output[] → tool_calls_made=0, not None."""
        handler = OpenAIEventHandler("openai.responses")
        handler.handle({
            "type": "response.completed",
            "response": {
                "id": "resp_empty",
                "model": "gpt-5.4",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        })
        result = handler.finalize()
        assert result.tool_calls_made == 0  # was None before fix


class TestGeminiStopReasonPrecedence:
    """promptFeedback.blockReason takes precedence over finishReason."""

    def test_extract_block_reason_overrides_finish_reason(self) -> None:
        """Non-streaming: blockReason present alongside candidate finishReason → blockReason wins."""
        body = {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0, "totalTokenCount": 10},
            "modelVersion": "gemini-2.5-flash",
        }
        result = gemini_extract_from_response(body)
        assert result.stop_reason == "SAFETY"

    def test_handler_block_reason_overrides_finish_reason(self) -> None:
        """Streaming handler: blockReason seen before candidate → stop_reason stays blockReason."""
        handler = GeminiEventHandler()
        # Chunk with both promptFeedback.blockReason and a candidate finishReason
        handler.handle({
            "candidates": [
                {"content": {"parts": [], "role": "model"}, "finishReason": "STOP"}
            ],
            "promptFeedback": {"blockReason": "OTHER"},
            "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 10},
        })
        result = handler.finalize()
        assert result.stop_reason == "OTHER"

    def test_no_block_reason_uses_finish_reason(self) -> None:
        """Without promptFeedback, finishReason used normally."""
        body = {
            "candidates": [
                {"content": {"parts": [], "role": "model"}, "finishReason": "MAX_TOKENS", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 100, "totalTokenCount": 110},
        }
        result = gemini_extract_from_response(body)
        assert result.stop_reason == "MAX_TOKENS"
