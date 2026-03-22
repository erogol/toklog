# toklog/adapters/gemini.py

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Matches /models/{name} or /tunedModels/{name} in URL paths.
# Group 1: "models" or "tunedModels". Group 2: the model name.
_MODEL_RE = re.compile(r"/(models|tunedModels)/([^/:]+)")


@dataclass
class GeminiUsageResult:
    """Normalized usage extracted from a Gemini API response."""

    input_tokens: int | None          # promptTokenCount
    output_tokens: int | None         # candidatesTokenCount
    total_tokens: int | None          # totalTokenCount
    thinking_tokens: int | None       # thoughtsTokenCount (explicit); None when absent
    cache_read_tokens: int            # cachedContentTokenCount (default 0)
    tool_calls_made: int | None       # count of functionCall parts; None if no candidates seen
    raw_usage: dict | None            # original usageMetadata dict (includes toolUsePromptTokenCount)
    usage_source: str                 # "provider_response" | "provider_stream_final"
    model_served: str | None          # modelVersion
    request_id: str | None            # responseId (backwards compat)
    response_id: str | None = None   # responseId (canonical)
    stop_reason: str | None = None    # finishReason or promptFeedback.blockReason

    @property
    def usage_status(self) -> str:
        if self.input_tokens is not None and self.output_tokens is not None:
            return "exact"
        return "missing"

    def apply_to_entry(self, entry: dict) -> None:
        """Write normalized fields into a log entry dict.

        Does NOT set endpoint_family — that must be set by integration code.
        """
        entry["input_tokens"] = self.input_tokens
        entry["output_tokens"] = self.output_tokens
        entry["cache_read_tokens"] = self.cache_read_tokens
        entry["cache_creation_tokens"] = 0       # Gemini doesn't have cache creation
        entry["thinking_tokens"] = self.thinking_tokens
        entry["raw_usage"] = self.raw_usage
        entry["usage_source"] = self.usage_source
        entry["total_tokens"] = self.total_tokens
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


def extract_model_from_url(path: str) -> str | None:
    """Extract model name from a Gemini API URL path.

    Handles both /models/{name} and /tunedModels/{name} variants.

    Examples:
        /v1beta/models/gemini-2.5-flash:generateContent  ->  gemini-2.5-flash
        /v1beta/models/gemini-3-flash-preview:streamGenerateContent  ->  gemini-3-flash-preview
        /v1/models/gemini-2.0-flash:generateContent  ->  gemini-2.0-flash
        /v1beta/tunedModels/my-tuned-model:generateContent  ->  my-tuned-model
        /v1beta/something/else  ->  None
    """
    match = _MODEL_RE.search(path)
    return match.group(2) if match else None


def extract_from_response(body: dict) -> GeminiUsageResult:
    """Extract usage from a non-streaming Gemini generateContent response body."""
    usage = body.get("usageMetadata") or {}
    candidates = body.get("candidates") or []

    stop_reason: str | None = None
    tool_calls_made: int | None

    # promptFeedback.blockReason takes precedence over finishReason
    prompt_feedback = body.get("promptFeedback") or {}
    block_reason = prompt_feedback.get("blockReason")
    if block_reason:
        stop_reason = block_reason

    if candidates:
        candidate = candidates[0]
        # Only use finishReason if no promptFeedback block
        if stop_reason is None:
            stop_reason = candidate.get("finishReason")
        parts = (candidate.get("content") or {}).get("parts") or []
        tool_calls_made = sum(1 for p in parts if isinstance(p, dict) and "functionCall" in p)
    else:
        tool_calls_made = None  # No candidates seen

    return GeminiUsageResult(
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
        total_tokens=usage.get("totalTokenCount"),
        thinking_tokens=usage.get("thoughtsTokenCount"),
        cache_read_tokens=usage.get("cachedContentTokenCount", 0),
        tool_calls_made=tool_calls_made,
        raw_usage=usage if usage else None,
        usage_source="provider_response",
        model_served=body.get("modelVersion"),
        request_id=body.get("responseId"),
        response_id=body.get("responseId"),
        stop_reason=stop_reason,
    )


def create_stream_handler() -> "GeminiEventHandler":
    """Factory: create a fresh GeminiEventHandler for a new stream."""
    return GeminiEventHandler()


class GeminiEventHandler:
    """Handles parsed SSE chunks for the Gemini streamGenerateContent API.

    Receives GenerateContentResponse dicts from SSEStreamBuffer.

    Unlike OpenAI/Anthropic, Gemini SSE has no event type field —
    every data payload is a GenerateContentResponse object.

    Accumulates:
    - Latest usageMetadata (typically only in the final chunk; last wins)
    - tool call count from functionCall parts across all chunks
    - model version and response ID from the first chunk that has them
    - finishReason from any candidate in any chunk
    - promptFeedback.blockReason when no candidates (safety-blocked prompt)
    """

    def __init__(self) -> None:
        self._finalized_result: GeminiUsageResult | None = None
        self.model_served: str | None = None
        self.request_id: str | None = None
        self.response_id: str | None = None
        self.stop_reason: str | None = None
        self._prompt_blocked: bool = False
        self._latest_usage: dict | None = None
        self._tool_call_count: int = 0
        self._seen_candidates: bool = False

    def handle(self, event: dict) -> None:
        """Process a parsed SSE chunk (a GenerateContentResponse dict)."""
        if self._finalized_result is not None:
            logger.warning("GeminiEventHandler.handle() called after finalize() — event dropped")
            return

        # Capture model and response ID from first chunk that has them
        if event.get("modelVersion") and not self.model_served:
            self.model_served = event["modelVersion"]
        if event.get("responseId") and not self.request_id:
            self.request_id = event["responseId"]
            self.response_id = event["responseId"]

        # Keep latest usageMetadata (typically only on final chunk; last wins)
        if "usageMetadata" in event:
            self._latest_usage = event["usageMetadata"]

        # Check for prompt-level block — takes precedence over finishReason
        if "promptFeedback" in event:
            block_reason = (event["promptFeedback"] or {}).get("blockReason")
            if block_reason:
                self._prompt_blocked = True
                self.stop_reason = block_reason

        # Process candidates
        candidates = event.get("candidates") or []
        for candidate in candidates:
            self._seen_candidates = True

            # Capture finish reason (unless overridden by promptFeedback.blockReason)
            if candidate.get("finishReason") and not self._prompt_blocked:
                self.stop_reason = candidate["finishReason"]

            # Count function calls in this chunk's parts.
            # Only count parts that have a "name" field to avoid double-counting
            # continuation chunks that send args without the function name.
            parts = (candidate.get("content") or {}).get("parts") or []
            for part in parts:
                if isinstance(part, dict) and "functionCall" in part and part["functionCall"].get("name"):
                    self._tool_call_count += 1

    def finalize(self) -> GeminiUsageResult:
        """Build and cache the final GeminiUsageResult. Idempotent."""
        if self._finalized_result is not None:
            return self._finalized_result

        usage = self._latest_usage or {}

        self._finalized_result = GeminiUsageResult(
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            total_tokens=usage.get("totalTokenCount"),
            thinking_tokens=usage.get("thoughtsTokenCount"),
            cache_read_tokens=usage.get("cachedContentTokenCount", 0),
            tool_calls_made=self._tool_call_count if self._seen_candidates else None,
            raw_usage=dict(usage) if usage else None,
            usage_source="provider_stream_final",
            model_served=self.model_served,
            request_id=self.request_id,
            response_id=self.response_id,
            stop_reason=self.stop_reason,
        )
        return self._finalized_result

    def apply_to_entry(self, entry: dict) -> None:
        """Finalize and write accumulated streaming state to a log entry.

        Does NOT set endpoint_family — that must be set by integration code.
        """
        self.finalize().apply_to_entry(entry)
