# toklog/adapters/anthropic.py

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnthropicUsageResult:
    """Normalized usage extracted from an Anthropic response."""
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int
    cache_creation_tokens: int
    thinking_tokens: int | None  # Estimated from thinking text chars // 4 (Anthropic doesn't report separately)
    tool_calls_made: int | None
    total_tokens: int | None = None  # Always None — Anthropic doesn't report it; field exists for parity with OpenAI
    raw_usage: dict | None = None
    usage_source: str = "provider_response"
    model_served: str | None = None
    request_id: str | None = None
    response_id: str | None = None  # msg_... message ID from API response
    stop_reason: str | None = None  # end_turn, tool_use, max_tokens, etc.

    @property
    def usage_status(self) -> str:
        if self.input_tokens is not None and self.output_tokens is not None:
            return "exact"
        return "missing"

    @property
    def computed_total_tokens(self) -> int | None:
        """Compute total from input + output since Anthropic doesn't report it.

        Note: this is input_tokens + output_tokens, which EXCLUDES cache tokens
        (Anthropic's input_tokens does not include cache_read/cache_creation).
        Returns None when either component is missing rather than falling back
        to an ambiguous value.
        """
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None

    def apply_to_entry(self, entry: dict) -> None:
        """Write normalized fields into a log entry dict."""
        entry["input_tokens"] = self.input_tokens
        entry["output_tokens"] = self.output_tokens
        entry["total_tokens"] = self.computed_total_tokens
        entry["cache_read_tokens"] = self.cache_read_tokens
        entry["cache_creation_tokens"] = self.cache_creation_tokens
        entry["thinking_tokens"] = self.thinking_tokens
        entry["raw_usage"] = self.raw_usage
        entry["usage_source"] = self.usage_source
        entry["usage_status"] = self.usage_status
        if self.model_served:
            entry["model"] = self.model_served
        if self.request_id:
            entry["request_id"] = self.request_id
        if self.response_id:
            entry["response_id"] = self.response_id
            # Dual-write: also set request_id for backwards compatibility
            # with existing log entries and downstream code that reads request_id.
            if not self.request_id:
                entry["request_id"] = self.response_id
        if self.tool_calls_made is not None:
            entry["tool_calls_made"] = self.tool_calls_made
        if self.stop_reason is not None:
            entry["stop_reason"] = self.stop_reason


def extract_from_response(body: dict) -> AnthropicUsageResult:
    """Extract usage from a non-streaming Anthropic Messages API response body."""
    usage = body.get("usage") or {}
    content = body.get("content") or []

    thinking_chars = sum(
        len(block.get("thinking") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "thinking"
    )

    tool_calls = sum(
        1 for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    )

    return AnthropicUsageResult(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        thinking_tokens=thinking_chars // 4 if thinking_chars > 0 else None,
        tool_calls_made=tool_calls,
        raw_usage=usage if usage else None,
        usage_source="provider_response",
        model_served=body.get("model"),
        response_id=body.get("id"),  # msg_... is a response identifier
        stop_reason=body.get("stop_reason"),
    )


def create_stream_handler() -> AnthropicEventHandler:
    """Factory: create a fresh AnthropicEventHandler for a new stream."""
    return AnthropicEventHandler()


class AnthropicEventHandler:
    """Handles parsed SSE events for Anthropic Messages API.

    Receives dicts from SSEStreamBuffer and accumulates usage state.

    Events handled:
    - message_start        -> input_tokens, cache tokens, model, response_id
    - content_block_start  -> tool_calls_made counter (type == "tool_use")
    - content_block_delta  -> thinking char accumulation (type == "thinking_delta")
    - message_delta        -> output_tokens, stop_reason
    - error                -> error_type, error_message
    """

    def __init__(self) -> None:
        self._finalized_result: AnthropicUsageResult | None = None

        self.model_served: str | None = None
        self.response_id: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cache_read_tokens: int = 0
        self.cache_creation_tokens: int = 0
        self._thinking_chars: int = 0
        self._tool_call_count: int = 0
        self._raw_usage_parts: dict = {}
        self.error_type: str | None = None
        self.error_message: str | None = None
        self.stop_reason: str | None = None

    def handle(self, event: dict) -> None:
        """Process a single parsed SSE event dict."""
        if self._finalized_result is not None:
            logger.warning("AnthropicEventHandler.handle() called after finalize() — event dropped")
            return

        event_type = event.get("type")
        if event_type == "message_start":
            self._handle_message_start(event)
        elif event_type == "content_block_start":
            self._handle_content_block_start(event)
        elif event_type == "content_block_delta":
            self._handle_content_block_delta(event)
        elif event_type == "message_delta":
            self._handle_message_delta(event)
        elif event_type == "error":
            self._handle_error(event)
        # message_stop, ping, content_block_stop: no action needed

    def finalize(self) -> AnthropicUsageResult:
        """Called when the stream ends. Returns accumulated usage.

        Idempotent: caches the result on first call.
        tool_calls_made is None when input_tokens is None (no message_start received).
        """
        if self._finalized_result is not None:
            return self._finalized_result

        self._finalized_result = AnthropicUsageResult(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens,
            thinking_tokens=self._thinking_chars // 4 if self._thinking_chars > 0 else None,
            tool_calls_made=self._tool_call_count if self.input_tokens is not None else None,
            raw_usage=dict(self._raw_usage_parts) if self._raw_usage_parts else None,
            usage_source="provider_stream_final",
            model_served=self.model_served,
            response_id=self.response_id,
            stop_reason=self.stop_reason,
        )
        return self._finalized_result

    def apply_to_entry(self, entry: dict) -> None:
        """Finalize and write accumulated streaming state to a log entry.

        Does NOT set endpoint_family — that must be set by integration code.
        """
        self.finalize().apply_to_entry(entry)
        if self.error_type is not None:
            entry["error"] = True
            entry["error_type"] = self.error_type

    def _handle_message_start(self, event: dict) -> None:
        message = event.get("message") or {}
        self.model_served = message.get("model")
        self.response_id = message.get("id")
        usage = message.get("usage") or {}
        self.input_tokens = usage.get("input_tokens")
        self.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
        self.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
        self._raw_usage_parts["message_start"] = usage

    def _handle_content_block_start(self, event: dict) -> None:
        block = event.get("content_block") or {}
        if block.get("type") == "tool_use":
            self._tool_call_count += 1

    def _handle_content_block_delta(self, event: dict) -> None:
        delta = event.get("delta") or {}
        if delta.get("type") == "thinking_delta":
            self._thinking_chars += len(delta.get("thinking", ""))

    def _handle_message_delta(self, event: dict) -> None:
        """Note: Anthropic only sends cache tokens in message_start.
        We intentionally skip cache token extraction from message_delta."""
        usage = event.get("usage") or {}
        if "output_tokens" in usage:
            self.output_tokens = usage["output_tokens"]
        self._raw_usage_parts["message_delta"] = usage
        stop_reason = event.get("delta", {}).get("stop_reason")
        if stop_reason is not None:
            self.stop_reason = stop_reason

    def _handle_error(self, event: dict) -> None:
        error = event.get("error") or {}
        self.error_type = error.get("type")
        self.error_message = error.get("message")
