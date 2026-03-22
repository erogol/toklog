"""TokLog - LLM token waste detector SDK."""

from __future__ import annotations

from typing import Any, Optional

from toklog.classify import classify_entries_async

__version__ = "0.1.0"


def wrap(
    client: Any,
    default_tags: Optional[str] = None,
    log_preview: bool = True,
) -> Any:
    """Wrap an OpenAI or Anthropic client to log all LLM calls.

    Supports sync and async clients for OpenAI and Anthropic.

    Usage:
        import openai
        import anthropic
        from toklog import wrap

        # Sync
        client = wrap(openai.OpenAI())
        # Use client as normal - all calls are logged

        # Async
        async_client = wrap(openai.AsyncOpenAI())
        response = await async_client.chat.completions.create(...)

        # Anthropic (sync + async both supported)
        anth = wrap(anthropic.Anthropic())
        anth_async = wrap(anthropic.AsyncAnthropic())
    """
    from toklog.wrapper import wrap as _wrap

    return _wrap(client, default_tags=default_tags, log_preview=log_preview)


def log_raw(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    max_tokens_set: int | None = None,
    system_prompt_hash: str | None = None,
    tool_count: int = 0,
    tool_schema_tokens: int = 0,
    tags: str | None = None,
    streaming: bool = False,
    duration_ms: int | None = None,
    error: bool = False,
    error_type: str | None = None,
    request_id: str | None = None,
    caller: str | None = None,
) -> None:
    """Log an LLM call manually (for apps that can't use wrap()).

    Never raises — silently swallows errors to avoid breaking the caller.
    """
    try:
        from datetime import datetime, timezone
        from toklog.logger import log_entry

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "streaming": streaming,
            "error": error,
        }
        if max_tokens_set is not None:
            entry["max_tokens_set"] = max_tokens_set
        if system_prompt_hash is not None:
            entry["system_prompt_hash"] = system_prompt_hash
        if tool_count:
            entry["tool_count"] = tool_count
        if tool_schema_tokens:
            entry["tool_schema_tokens"] = tool_schema_tokens
        if tags is not None:
            entry["tags"] = tags
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        if error_type is not None:
            entry["error_type"] = error_type
        if request_id is not None:
            entry["request_id"] = request_id
        if caller is not None:
            entry["call_site"] = {"file": caller, "function": "", "line": 0}
        else:
            from toklog.wrapper import _get_call_site
            cs = _get_call_site()
            if cs is not None:
                entry["call_site"] = cs

        log_entry(entry)
    except Exception:
        pass


def compress_history(
    messages: list,
    max_tokens: int = 8000,
    keep_recent: int = 5,
    summarizer_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> list:
    """Compress a message history by summarizing old messages.

    Partitions messages into system messages (preserved verbatim at the top)
    and conversation messages. Keeps the most recent `keep_recent` conversation
    messages unchanged. Older conversation messages are summarized into a single
    system message when they would exceed the token budget.

    Args:
        messages: List of chat messages (dicts with 'role' and 'content').
        max_tokens: Target token budget. Returns unchanged if already fits.
        keep_recent: Number of recent conversation messages to keep verbatim.
        summarizer_model: OpenAI model used for summarization.
        api_key: Optional OpenAI API key (uses env var if None).

    Returns:
        Compressed list of messages, or the original list if no compression needed.
    """
    from toklog.compress import compress_history as _compress_history

    return _compress_history(
        messages,
        max_tokens=max_tokens,
        keep_recent=keep_recent,
        summarizer_model=summarizer_model,
        api_key=api_key,
    )
