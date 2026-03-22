"""Tests for adapter consistency fixes.

Covers:
1. stop_reason: all adapters should capture stop_reason internally
   (not rely on server.py to manually extract it from response bodies)
2. Anthropic response_id: msg_... should be stored as response_id,
   not request_id (which is semantically wrong)
3. __init__.py re-exports: all public adapter types should be importable
   from toklog.adapters directly
"""

from __future__ import annotations

import pytest


# ===========================================================================
# 1. stop_reason captured by all adapters, not just Gemini
# ===========================================================================

class TestStopReasonInAdapters:
    """Every adapter should capture stop_reason via apply_to_entry
    for both streaming and non-streaming paths."""

    # ---- OpenAI non-streaming: Chat Completions ----

    def test_openai_non_streaming_chat_stop_reason(self) -> None:
        from toklog.adapters.openai import extract_from_response
        body = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = extract_from_response(body, "openai.chat_completions")
        assert result.stop_reason == "stop"
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["stop_reason"] == "stop"

    def test_openai_non_streaming_chat_no_stop_reason(self) -> None:
        """No choices → no stop_reason → entry should not have key."""
        from toklog.adapters.openai import extract_from_response
        body = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = extract_from_response(body, "openai.chat_completions")
        assert result.stop_reason is None
        entry: dict = {}
        result.apply_to_entry(entry)
        assert "stop_reason" not in entry

    # ---- OpenAI non-streaming: Responses API ----

    def test_openai_non_streaming_responses_stop_reason(self) -> None:
        from toklog.adapters.openai import extract_from_response
        body = {
            "id": "resp_abc",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = extract_from_response(body, "openai.responses")
        assert result.stop_reason == "completed"
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["stop_reason"] == "completed"

    # ---- OpenAI streaming: Chat Completions ----

    def test_openai_streaming_chat_stop_reason(self) -> None:
        from toklog.adapters.openai import OpenAIEventHandler
        handler = OpenAIEventHandler(endpoint="openai.chat_completions")
        # Chunk with finish_reason
        handler.handle({
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["stop_reason"] == "stop"

    # ---- Anthropic non-streaming ----

    def test_anthropic_non_streaming_stop_reason(self) -> None:
        from toklog.adapters.anthropic import extract_from_response
        body = {
            "id": "msg_123",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = extract_from_response(body)
        assert result.stop_reason == "end_turn"
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["stop_reason"] == "end_turn"

    def test_anthropic_non_streaming_no_stop_reason(self) -> None:
        from toklog.adapters.anthropic import extract_from_response
        body = {
            "id": "msg_123",
            "model": "claude-sonnet-4-6",
            "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = extract_from_response(body)
        assert result.stop_reason is None
        entry: dict = {}
        result.apply_to_entry(entry)
        assert "stop_reason" not in entry

    def test_anthropic_non_streaming_tool_use_stop(self) -> None:
        from toklog.adapters.anthropic import extract_from_response
        body = {
            "id": "msg_123",
            "model": "claude-sonnet-4-6",
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "t1", "name": "calc", "input": {}}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = extract_from_response(body)
        assert result.stop_reason == "tool_use"

    # ---- Anthropic streaming ----

    def test_anthropic_streaming_stop_reason(self) -> None:
        from toklog.adapters.anthropic import AnthropicEventHandler
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {
                "id": "msg_123", "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 10},
            },
        })
        handler.handle({
            "type": "message_delta",
            "usage": {"output_tokens": 5},
            "delta": {"stop_reason": "end_turn"},
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["stop_reason"] == "end_turn"

    # ---- Gemini non-streaming (already works) ----

    def test_gemini_non_streaming_stop_reason(self) -> None:
        from toklog.adapters.gemini import extract_from_response
        body = {
            "candidates": [{
                "content": {"parts": [{"text": "hello"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        result = extract_from_response(body)
        assert result.stop_reason == "STOP"
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["stop_reason"] == "STOP"

    # ---- Gemini streaming (already works) ----

    def test_gemini_streaming_stop_reason(self) -> None:
        from toklog.adapters.gemini import GeminiEventHandler
        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{
                "content": {"parts": [{"text": "hello"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry["stop_reason"] == "STOP"


# ===========================================================================
# 2. Anthropic response_id: msg_... should be response_id, not request_id
# ===========================================================================

class TestAnthropicResponseId:
    """Anthropic's message id (msg_...) is a response identifier.
    It should be stored as response_id for parity with OpenAI."""

    def test_non_streaming_has_response_id(self) -> None:
        from toklog.adapters.anthropic import extract_from_response
        body = {
            "id": "msg_abc123",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = extract_from_response(body)
        assert result.response_id == "msg_abc123"
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry.get("response_id") == "msg_abc123"

    def test_streaming_has_response_id(self) -> None:
        from toklog.adapters.anthropic import AnthropicEventHandler
        handler = AnthropicEventHandler()
        handler.handle({
            "type": "message_start",
            "message": {
                "id": "msg_stream_456",
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 10},
            },
        })
        handler.handle({
            "type": "message_delta",
            "usage": {"output_tokens": 5},
            "delta": {"stop_reason": "end_turn"},
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry.get("response_id") == "msg_stream_456"

    def test_non_streaming_no_id_no_response_id(self) -> None:
        """When id is missing, response_id should not be set."""
        from toklog.adapters.anthropic import extract_from_response
        body = {
            "model": "claude-sonnet-4-6",
            "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = extract_from_response(body)
        assert result.response_id is None
        entry: dict = {}
        result.apply_to_entry(entry)
        assert "response_id" not in entry

    def test_request_id_from_header_not_body(self) -> None:
        """Anthropic's x-request-id header is the real request_id.
        The body id is the response_id. These must be separate."""
        from toklog.adapters.anthropic import AnthropicUsageResult
        result = AnthropicUsageResult(
            input_tokens=100, output_tokens=50,
            cache_read_tokens=0, cache_creation_tokens=0,
            thinking_tokens=None, tool_calls_made=0,
            request_id="req-header-abc",
            response_id="msg_body_123",
        )
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry.get("request_id") == "req-header-abc"
        assert entry.get("response_id") == "msg_body_123"

    def test_response_id_parity_with_openai(self) -> None:
        """Both OpenAI and Anthropic results should have a response_id field."""
        from toklog.adapters.openai import UsageResult as OpenAIUsageResult
        from toklog.adapters.anthropic import AnthropicUsageResult

        assert "response_id" in OpenAIUsageResult.__dataclass_fields__, "OpenAI missing response_id"
        assert "response_id" in AnthropicUsageResult.__dataclass_fields__, "Anthropic missing response_id"



# ===========================================================================
# 4. server.py _extract_non_streaming: stop_reason should come from adapter
# ===========================================================================

class TestServerStopReasonDelegation:
    """After adapters handle stop_reason internally,
    _extract_non_streaming should NOT need manual stop_reason extraction."""

    def test_openai_chat_completions_stop_via_adapter(self) -> None:
        """Verify that extract_from_response captures finish_reason for chat completions."""
        from toklog.adapters.openai import extract_from_response
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = extract_from_response(body, "openai.chat_completions")
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["stop_reason"] == "length"

    def test_openai_responses_api_stop_via_adapter(self) -> None:
        from toklog.adapters.openai import extract_from_response
        body = {
            "id": "resp_test",
            "object": "response",
            "model": "gpt-4o",
            "status": "incomplete",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        result = extract_from_response(body, "openai.responses")
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["stop_reason"] == "incomplete"

    def test_anthropic_stop_via_adapter(self) -> None:
        from toklog.adapters.anthropic import extract_from_response
        body = {
            "id": "msg_test",
            "model": "claude-sonnet-4-6",
            "stop_reason": "max_tokens",
            "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = extract_from_response(body)
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry["stop_reason"] == "max_tokens"

    def test_openai_multiple_choices_first_stop_reason(self) -> None:
        """With multiple choices, stop_reason comes from first choice."""
        from toklog.adapters.openai import extract_from_response
        body = {
            "id": "chatcmpl-multi",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "a"}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "b"}, "finish_reason": "length"},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        result = extract_from_response(body, "openai.chat_completions")
        assert result.stop_reason == "stop"  # first choice wins
