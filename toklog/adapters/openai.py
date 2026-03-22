# toklog/adapters/openai.py

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UsageResult:
    """Normalized usage extracted from an OpenAI response."""
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cache_read_tokens: int
    cache_creation_tokens: int  # always 0 for OpenAI
    reasoning_tokens: int | None
    raw_usage: dict | None      # always a JSON-parsed dict, never an SDK object
    usage_source: str            # "provider_response" | "provider_stream_final"
    model_served: str | None
    request_id: str | None
    response_id: str | None
    tool_calls_made: int | None
    stop_reason: str | None = None  # finish_reason (chat) or status (responses)

    @property
    def usage_status(self) -> str:
        if self.input_tokens is not None and self.output_tokens is not None:
            return "exact"
        return "missing"

    def apply_to_entry(self, entry: dict) -> None:
        """Write normalized fields into a log entry dict."""
        entry["input_tokens"] = self.input_tokens
        entry["output_tokens"] = self.output_tokens
        entry["total_tokens"] = self.total_tokens
        entry["cache_read_tokens"] = self.cache_read_tokens
        entry["cache_creation_tokens"] = self.cache_creation_tokens
        entry["thinking_tokens"] = self.reasoning_tokens
        entry["raw_usage"] = self.raw_usage
        entry["usage_source"] = self.usage_source
        entry["usage_status"] = self.usage_status
        if self.model_served:
            entry["model"] = self.model_served
        if self.request_id:
            entry["request_id"] = self.request_id
        if self.response_id:
            entry["response_id"] = self.response_id
        if self.tool_calls_made is not None:
            entry["tool_calls_made"] = self.tool_calls_made
        if self.stop_reason is not None:
            entry["stop_reason"] = self.stop_reason


def classify_endpoint(path: str, body: dict | None = None) -> str:
    """Classify an OpenAI request as chat_completions or responses.

    Args:
        path: URL path (e.g. "/v1/chat/completions", "/v1/responses")
        body: Optional request body for ambiguous cases

    Returns:
        "openai.chat_completions" or "openai.responses"
    """
    if "/chat/completions" in path:
        return "openai.chat_completions"
    if "/responses" in path:
        return "openai.responses"
    # Heuristic fallback: Responses API uses "input", Chat uses "messages"
    if body:
        if "input" in body and "messages" not in body:
            return "openai.responses"
        if "messages" in body:
            return "openai.chat_completions"
    return "openai.chat_completions"  # default


def extract_from_response(body: dict, endpoint: str) -> UsageResult:
    """Extract usage from a non-streaming response body (JSON dict).

    Works for both Chat Completions and Responses API response objects.
    """
    usage = body.get("usage") or {}

    if endpoint == "openai.responses":
        return _extract_responses_usage(usage, body, "provider_response")
    else:
        return _extract_chat_usage(usage, body, "provider_response")


def create_stream_handler(endpoint: str) -> OpenAIEventHandler:
    """Create a streaming event handler for the given endpoint family."""
    return OpenAIEventHandler(endpoint)


def maybe_inject_stream_options(request_body: dict, endpoint: str) -> dict:
    """Inject stream_options.include_usage=true for Chat Completions streaming.

    Only mutates if:
    - endpoint is openai.chat_completions
    - stream is truthy
    - stream_options is not already set

    Returns the (possibly modified) request body.
    """
    if endpoint != "openai.chat_completions":
        return request_body
    if not request_body.get("stream"):
        return request_body
    if "stream_options" in request_body:
        return request_body  # respect user's explicit setting

    request_body["stream_options"] = {"include_usage": True}
    return request_body


# --- Internal helpers ---

def _extract_responses_usage(usage: dict, body: dict, source: str) -> UsageResult:
    """Extract from Responses API usage dict."""
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}

    output_items = body.get("output") or []
    tc_count = sum(
        1 for item in output_items
        if isinstance(item, dict) and item.get("type") == "function_call"
    )

    return UsageResult(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        cache_read_tokens=input_details.get("cached_tokens", 0),
        cache_creation_tokens=0,
        reasoning_tokens=output_details.get("reasoning_tokens"),
        raw_usage=usage if usage else None,
        usage_source=source,
        model_served=body.get("model"),
        request_id=body.get("id"),
        response_id=body.get("id"),
        tool_calls_made=tc_count,
        stop_reason=body.get("status"),  # "completed", "incomplete", etc.
    )


def _extract_chat_usage(usage: dict, body: dict, source: str) -> UsageResult:
    """Extract from Chat Completions usage dict."""
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}

    tc_count = None
    stop_reason: str | None = None
    choices = body.get("choices") or []
    if choices:
        tc = choices[0].get("message", {}).get("tool_calls") or []
        tc_count = len(tc)
        stop_reason = choices[0].get("finish_reason")

    return UsageResult(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cache_read_tokens=prompt_details.get("cached_tokens", 0),
        cache_creation_tokens=0,
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        raw_usage=usage if usage else None,
        usage_source=source,
        model_served=body.get("model"),
        request_id=body.get("id"),
        response_id=None,  # Chat Completions don't have retrievable response_id
        tool_calls_made=tc_count,
        stop_reason=stop_reason,
    )


class OpenAIEventHandler:
    """Handles parsed SSE events for OpenAI endpoints.

    Receives dicts from SSEStreamBuffer and accumulates usage state.
    Supports both Chat Completions and Responses API event shapes.
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._finalized_result: UsageResult | None = None

        # Accumulated state
        self.model_served: str | None = None
        self.request_id: str | None = None
        self.response_id: str | None = None
        self.stop_reason: str | None = None
        self.tool_call_indices: set[int] = set()
        self._usage_result: UsageResult | None = None

    def handle(self, event: dict) -> None:
        """Process a single parsed SSE event dict."""
        if self._finalized_result is not None:
            logger.warning("OpenAIEventHandler.handle() called after finalize() — event dropped")
            return

        event_type = event.get("type")

        # --- Responses API events ---
        if event_type == "response.completed":
            self._handle_response_completed(event)
            return

        if event_type == "response.created":
            resp = event.get("response") or {}
            if resp.get("model"):
                self.model_served = resp["model"]
            if resp.get("id"):
                self.response_id = resp["id"]
                self.request_id = resp["id"]
            return

        # --- Chat Completions events ---
        if event.get("object") == "chat.completion.chunk":
            self._handle_chat_chunk(event)
            return

        # Generic: capture model from any event
        if event.get("model") and not self.model_served:
            self.model_served = event["model"]

    def finalize(self) -> UsageResult:
        """Called when the stream ends. Returns captured or missing usage.

        Idempotent: caches the result on first call and returns
        the same result on subsequent calls.
        """
        if self._finalized_result is not None:
            return self._finalized_result

        if self._usage_result is not None:
            self._finalized_result = self._usage_result
        else:
            self._finalized_result = UsageResult(
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                reasoning_tokens=None,
                raw_usage=None,
                usage_source="provider_stream_final",
                model_served=self.model_served,
                request_id=self.request_id,
                response_id=self.response_id,
                tool_calls_made=None,
            )

        # Handler always owns stop_reason — override what the result may have
        if self.stop_reason is not None:
            self._finalized_result.stop_reason = self.stop_reason

        return self._finalized_result

    def apply_to_entry(self, entry: dict) -> None:
        """Finalize and write accumulated streaming state to a log entry."""
        self.finalize().apply_to_entry(entry)

    def _handle_response_completed(self, data: dict) -> None:
        """Handle Responses API response.completed — the main usage carrier.

        Tool calls are counted from the output items in this event.
        No incremental counter is needed — response.completed carries all output items.
        """
        resp = data.get("response") or {}
        usage = resp.get("usage") or {}

        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}

        # Count tool calls from the complete output list
        output_items = resp.get("output") or []
        tc_count = sum(
            1 for item in output_items
            if isinstance(item, dict) and item.get("type") == "function_call"
        )

        status = resp.get("status")
        if status is not None:
            self.stop_reason = status

        self._usage_result = UsageResult(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            cache_read_tokens=input_details.get("cached_tokens", 0),
            cache_creation_tokens=0,
            reasoning_tokens=output_details.get("reasoning_tokens"),
            raw_usage=usage if usage else None,
            usage_source="provider_stream_final",
            model_served=resp.get("model") or self.model_served,
            request_id=resp.get("id") or self.request_id,
            response_id=resp.get("id") or self.response_id,
            tool_calls_made=tc_count,
        )

    def _handle_chat_chunk(self, data: dict) -> None:
        """Handle a Chat Completions streaming chunk."""
        if data.get("model"):
            self.model_served = data["model"]
        if data.get("id"):
            self.request_id = data["id"]

        # Track tool call indices for counting; capture finish_reason
        for choice in (data.get("choices") or []):
            delta = choice.get("delta") or {}
            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index")
                if idx is not None:
                    self.tool_call_indices.add(idx)
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                self.stop_reason = finish_reason

        # Usage is only on the final chunk (when stream_options.include_usage=true)
        usage = data.get("usage")
        if usage:
            prompt_details = usage.get("prompt_tokens_details") or {}
            completion_details = usage.get("completion_tokens_details") or {}

            self._usage_result = UsageResult(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                cache_read_tokens=prompt_details.get("cached_tokens", 0),
                cache_creation_tokens=0,
                reasoning_tokens=completion_details.get("reasoning_tokens"),
                raw_usage=usage,
                usage_source="provider_stream_final",
                model_served=self.model_served,
                request_id=self.request_id,
                response_id=None,
                tool_calls_made=len(self.tool_call_indices) if self.tool_call_indices else 0,
            )
