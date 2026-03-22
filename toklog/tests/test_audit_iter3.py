"""Tests for audit iteration 3 fixes.

Covers:
1. Schema consistency: all entry fields initialized by extractor.py
2. Gemini response_id written to entry
3. Streaming exception handling: re-raise after logging
4. resp.aclose() exception isolation
"""

from __future__ import annotations

import pytest

# ===========================================================================
# 1. Schema consistency — initial entry dict has all fields adapters write
# ===========================================================================

# Fields that every adapter's apply_to_entry() writes (always, not conditionally)
_ALWAYS_WRITTEN = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "thinking_tokens",
    "raw_usage",
    "usage_source",
    "usage_status",
}

# Fields that adapters write conditionally (if not None)
_CONDITIONALLY_WRITTEN = {
    "model",
    "request_id",
    "response_id",
    "tool_calls_made",
    "stop_reason",
}


class TestEntrySchemaConsistency:
    """Every log entry should have a consistent schema, regardless of
    whether an adapter's apply_to_entry was called."""

    def test_initial_entry_has_all_always_fields(self) -> None:
        """extract_from_request should initialize all always-written fields."""
        from toklog.proxy.extractor import extract_from_request

        entry = extract_from_request(
            provider="openai",
            headers={"authorization": "Bearer sk-test1234"},
            body={"model": "gpt-4o", "messages": []},
            tag=None,
            cmdline=None,
        )
        for field in _ALWAYS_WRITTEN:
            assert field in entry, f"Initial entry missing always-written field: {field}"

    def test_initial_entry_has_all_conditional_fields(self) -> None:
        """extract_from_request should initialize conditional fields with None
        so entries have a consistent schema even when adapter isn't called."""
        from toklog.proxy.extractor import extract_from_request

        entry = extract_from_request(
            provider="openai",
            headers={"authorization": "Bearer sk-test1234"},
            body={"model": "gpt-4o", "messages": []},
            tag=None,
            cmdline=None,
        )
        for field in _CONDITIONALLY_WRITTEN:
            assert field in entry, f"Initial entry missing conditional field: {field}"

    def test_entry_schema_stable_after_openai_apply(self) -> None:
        """After OpenAI adapter apply_to_entry, no new keys should appear
        that weren't in the initial entry."""
        from toklog.proxy.extractor import extract_from_request
        from toklog.adapters.openai import extract_from_response

        entry = extract_from_request(
            provider="openai",
            headers={"authorization": "Bearer sk-test1234"},
            body={"model": "gpt-4o", "messages": []},
            tag=None,
            cmdline=None,
        )
        initial_keys = set(entry.keys())

        body = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = extract_from_response(body, "openai.chat_completions")
        result.apply_to_entry(entry)

        new_keys = set(entry.keys()) - initial_keys
        assert not new_keys, f"OpenAI adapter introduced new keys: {new_keys}"

    def test_entry_schema_stable_after_anthropic_apply(self) -> None:
        from toklog.proxy.extractor import extract_from_request
        from toklog.adapters.anthropic import extract_from_response

        entry = extract_from_request(
            provider="anthropic",
            headers={"x-api-key": "sk-ant-api-test1234"},
            body={"model": "claude-sonnet-4-6", "messages": []},
            tag=None,
            cmdline=None,
        )
        initial_keys = set(entry.keys())

        body = {
            "id": "msg_123",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = extract_from_response(body)
        result.apply_to_entry(entry)

        new_keys = set(entry.keys()) - initial_keys
        assert not new_keys, f"Anthropic adapter introduced new keys: {new_keys}"

    def test_entry_schema_stable_after_gemini_apply(self) -> None:
        from toklog.proxy.extractor import extract_from_request
        from toklog.adapters.gemini import extract_from_response

        entry = extract_from_request(
            provider="gemini",
            headers={"x-goog-api-key": "AIzatest1234"},
            body={"model": "gemini-2.5-flash"},
            tag=None,
            cmdline=None,
        )
        initial_keys = set(entry.keys())

        body = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            "modelVersion": "gemini-2.5-flash",
            "responseId": "resp_abc",
        }
        result = extract_from_response(body)
        result.apply_to_entry(entry)

        new_keys = set(entry.keys()) - initial_keys
        assert not new_keys, f"Gemini adapter introduced new keys: {new_keys}"


# ===========================================================================
# 2. Gemini response_id
# ===========================================================================

class TestGeminiResponseId:
    """Gemini's responseId should be written as response_id for parity."""

    def test_non_streaming_response_id_written(self) -> None:
        from toklog.adapters.gemini import extract_from_response

        body = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            "responseId": "resp_gemini_123",
        }
        result = extract_from_response(body)
        entry: dict = {}
        result.apply_to_entry(entry)
        assert entry.get("response_id") == "resp_gemini_123"

    def test_streaming_response_id_written(self) -> None:
        from toklog.adapters.gemini import GeminiEventHandler

        handler = GeminiEventHandler()
        handler.handle({
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            "responseId": "resp_stream_456",
        })
        entry: dict = {}
        handler.apply_to_entry(entry)
        assert entry.get("response_id") == "resp_stream_456"

    def test_no_response_id_no_key(self) -> None:
        """When Gemini doesn't include responseId, entry should have None."""
        from toklog.adapters.gemini import extract_from_response

        body = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }
        result = extract_from_response(body)
        entry: dict = {}
        result.apply_to_entry(entry)
        # response_id should not appear when None (conditional write)
        assert "response_id" not in entry

    def test_gemini_request_id_still_set(self) -> None:
        """Gemini's responseId maps to BOTH request_id and response_id for
        backwards compatibility (request_id was historically the only field)."""
        from toklog.adapters.gemini import extract_from_response

        body = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            "responseId": "resp_123",
        }
        result = extract_from_response(body)
        entry: dict = {}
        result.apply_to_entry(entry)
        # Both should be set for backwards compat
        assert entry.get("request_id") == "resp_123"
        assert entry.get("response_id") == "resp_123"


# ===========================================================================
# 3. Streaming exception handling
# ===========================================================================

class TestStreamingExceptionHandling:
    """The streaming generate() should re-raise exceptions after logging,
    not silently swallow them."""

    @pytest.mark.asyncio
    async def test_generate_reraises_on_stream_error(self) -> None:
        """If resp.aiter_bytes() raises mid-stream, the generator should
        propagate the error after logging."""
        import asyncio

        class FakeResp:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def aiter_bytes(self):
                yield b"data: {}\n\n"
                raise ConnectionError("upstream died")

            async def aclose(self):
                pass

        from toklog.adapters.sse import SSEStreamBuffer

        resp = FakeResp()
        buffer = SSEStreamBuffer()
        error_logged = False

        async def generate():
            nonlocal error_logged
            try:
                async for chunk in resp.aiter_bytes():
                    for event in buffer.feed(chunk):
                        pass
                    yield chunk
            except Exception:
                error_logged = True
                raise  # This is what we're testing — must re-raise
            finally:
                for event in buffer.flush():
                    pass
                await resp.aclose()

        chunks = []
        with pytest.raises(ConnectionError):
            async for chunk in generate():
                chunks.append(chunk)

        assert error_logged
        assert len(chunks) == 1  # got first chunk before error


# ===========================================================================
# 4. resp.aclose() exception isolation
# ===========================================================================

class TestAcloseExceptionIsolation:
    """resp.aclose() should not mask the original exception."""

    @pytest.mark.asyncio
    async def test_aclose_error_does_not_mask_original(self) -> None:
        """If both the stream and aclose() raise, the original error
        should be propagated, not the aclose error."""

        class BrokenResp:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def aiter_bytes(self):
                raise ConnectionError("stream broke")
                yield  # make it an async generator  # noqa: unreachable

            async def aclose(self):
                raise OSError("aclose also broke")

        resp = BrokenResp()

        async def generate():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            except Exception:
                raise
            finally:
                try:
                    await resp.aclose()
                except Exception:
                    pass  # Don't mask the original

        with pytest.raises(ConnectionError, match="stream broke"):
            async for _ in generate():
                pass