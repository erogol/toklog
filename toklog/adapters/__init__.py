"""LLM provider adapters for usage extraction and streaming handling."""

from toklog.adapters.openai import (
    UsageResult as OpenAIUsageResult,
    OpenAIEventHandler,
    extract_from_response as openai_extract,
)
from toklog.adapters.anthropic import (
    AnthropicUsageResult,
    AnthropicEventHandler,
    extract_from_response as anthropic_extract,
)
from toklog.adapters.gemini import (
    GeminiUsageResult,
    GeminiEventHandler,
    extract_from_response as gemini_extract,
)

__all__ = [
    # Result dataclasses
    "OpenAIUsageResult",
    "AnthropicUsageResult",
    "GeminiUsageResult",
    # Streaming event handlers
    "OpenAIEventHandler",
    "AnthropicEventHandler",
    "GeminiEventHandler",
    # Non-streaming extraction
    "openai_extract",
    "anthropic_extract",
    "gemini_extract",
]
