"""History compression for LLM conversations."""

from __future__ import annotations

from typing import Any, Optional


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
        return " ".join(parts)
    return str(content)


def _count_messages_tokens(messages: list[dict[str, Any]], model: str) -> int:
    total = 0
    for msg in messages:
        text = _extract_text(msg.get("content", ""))
        total += _count_tokens(text, model) + 4
    return total


def _summarize(
    messages: list[dict[str, Any]],
    model: str,
    max_summary_tokens: int,
    api_key: Optional[str],
) -> str:
    import openai

    lines = [f"{m['role']}: {_extract_text(m.get('content', ''))}" for m in messages]
    conversation_text = "\n".join(lines)

    kwargs: dict[str, Any] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key

    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_summary_tokens,
        messages=[
            {
                "role": "system",
                "content": "Summarize the following conversation concisely, preserving key facts and decisions.",
            },
            {"role": "user", "content": conversation_text},
        ],
    )
    return response.choices[0].message.content or ""


def compress_history(
    messages: list[dict[str, Any]],
    max_tokens: int = 8000,
    keep_recent: int = 5,
    summarizer_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> list[dict[str, Any]]:
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
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conversation = [m for m in messages if m.get("role") != "system"]

    keep_recent = max(0, keep_recent)

    if len(conversation) <= keep_recent:
        return messages

    recent = conversation[-keep_recent:] if keep_recent > 0 else []
    old = conversation[:-keep_recent] if keep_recent > 0 else conversation[:]

    fixed_tokens = _count_messages_tokens(system_msgs + recent, summarizer_model)
    if fixed_tokens >= max_tokens:
        return messages

    remaining = max_tokens - fixed_tokens
    if _count_messages_tokens(old, summarizer_model) <= remaining:
        return messages

    max_summary_tokens = max(remaining - 10, 50)
    summary = _summarize(old, summarizer_model, max_summary_tokens, api_key)
    summary_msg: dict[str, Any] = {
        "role": "system",
        "content": f"[Summary of earlier conversation]\n{summary}",
    }
    return system_msgs + [summary_msg] + recent
