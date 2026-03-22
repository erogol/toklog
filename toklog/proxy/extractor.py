"""Extract JSONL fields from HTTP request/response bodies."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, List, Optional

from toklog.wrapper import _key_hint


# Known prefixes for static (non-rotating) LLM API keys
_STATIC_KEY_PREFIXES = ("sk-ant-api", "sk-", "r8_", "gsk_", "hf_", "xai-", "AIza")
# Anthropic OAuth access tokens start with sk-ant-oat — they rotate and must
# NOT be treated as stable static keys. Check these BEFORE _STATIC_KEY_PREFIXES.
_OAUTH_KEY_PREFIXES = ("sk-ant-oat",)


def _key_hint_from_headers(headers: Any, tag: Optional[str] = None) -> Optional[str]:
    """Extract a safe API key identifier from request headers.

    Static API keys (with known prefixes like sk-ant-, sk-, r8_) are identified
    by their last 8 chars. OAuth/rotating bearer tokens are skipped — the tag
    is used instead to avoid polluting the breakdown with short-lived tokens.
    """
    if not headers:
        return f"[{tag}]" if tag else None
    try:
        # Authorization: Bearer <token> (OpenAI, some Anthropic OAuth paths)
        auth = headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if any(token.startswith(p) for p in _OAUTH_KEY_PREFIXES):
                return f"[{tag}]" if tag else "[OAuth]"
            if any(token.startswith(p) for p in _STATIC_KEY_PREFIXES):
                return _key_hint(token)
            return f"[{tag}]" if tag else "[OAuth]"
        # x-api-key: <key> — used by Anthropic SDK for both static keys AND
        # OAuth tokens (sk-ant-oat... tokens also arrive here via auth_token=)
        key = headers.get("x-api-key") or ""
        if key:
            if any(key.startswith(p) for p in _OAUTH_KEY_PREFIXES):
                return f"[{tag}]" if tag else "[OAuth]"
            if any(key.startswith(p) for p in _STATIC_KEY_PREFIXES):
                return _key_hint(key)
            return f"[{tag}]" if tag else "[OAuth]"
        # x-goog-api-key: <key> — used by Gemini SDK
        goog_key = headers.get("x-goog-api-key") or ""
        if goog_key:
            if any(goog_key.startswith(p) for p in _STATIC_KEY_PREFIXES):
                return _key_hint(goog_key)
            return f"[{tag}]" if tag else None
    except Exception:
        pass
    return f"[{tag}]" if tag else None


def _derive_program(cmdline: Optional[str], tag: Optional[str]) -> Optional[str]:
    """Derive a short program name from the process command line or tag."""
    if cmdline:
        # Rule 1: python3 -m module.sub → first segment of module
        m = re.search(r"-m\s+(\S+)", cmdline)
        if m:
            return m.group(1).split(".")[0]
        # Rule 2: last token ends with .py or .js → strip path and extension
        tokens = cmdline.split()
        if tokens and (tokens[-1].endswith(".py") or tokens[-1].endswith(".js")):
            import os.path
            return os.path.splitext(os.path.basename(tokens[-1]))[0]
        # Rule 3: first non-interpreter token
        for t in tokens:
            if t not in ("python3", "python", "node"):
                return t
    if tag is not None:
        return tag
    return None


def extract_from_request(provider: str, body: dict, tag: Optional[str], headers: Any = None, *, cmdline: Optional[str] = None) -> dict:
    """Build a partial JSONL entry from a parsed request body."""
    tools: List[Any] = body.get("tools") or []
    # Responses API uses "input" instead of "messages"
    messages: List[Any] = body.get("messages") or body.get("input") or []
    # Responses API allows "input" to be a plain string — normalise to list
    if isinstance(messages, str):
        messages = []

    # System prompt:
    #   Anthropic: top-level "system"
    #   OpenAI Chat Completions: role=system message
    #   OpenAI Responses API: top-level "instructions"
    system: Any = body.get("system") or body.get("instructions")
    if system is None:
        sys_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        system = "\n".join(str(s) for s in sys_parts) if sys_parts else None

    # Responses API uses "max_output_tokens" instead of "max_tokens"
    max_tokens_set = body.get("max_tokens") or body.get("max_output_tokens")

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "provider": provider,
        "model": body.get("model", "unknown"),
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "max_tokens_set": max_tokens_set,
        "system_prompt_hash": _hash_system(system),
        "tool_count": len(tools),
        "tool_schema_tokens": len(json.dumps(tools, default=str)) // 4 if tools else 0,
        "tool_names": _extract_tool_names(tools),
        "has_tool_results": _has_tool_results(messages),
        "tool_calls_made": None,
        "program": _derive_program(cmdline, tag),
        "tags": tag,
        "streaming": body.get("stream", False),
        "duration_ms": None,
        "error": False,
        "error_type": None,
        "request_id": None,
        "response_id": None,
        "stop_reason": None,
        "total_tokens": None,
        "raw_usage": None,
        "usage_source": None,
        "usage_status": None,
        "call_site": None,
        "user_message_preview": None,  # disabled by default — proxy sees full prompts
        "assistant_preview": None,
        "thinking_tokens": None,
        "api_key_hint": _key_hint_from_headers(headers, tag=tag),
        "instrumentation": "proxy",
        "proxy_client_addr": "127.0.0.1",
        **_extract_context_signals(provider, body),
    }



def _hash_system(content: Any) -> Optional[str]:
    if not content:
        return None
    text = str(content).strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_tool_names(tools: list) -> list:
    names = []
    for t in tools:
        if isinstance(t, dict):
            name = t.get("name") or (t.get("function") or {}).get("name")
            if name:
                names.append(name)
    return names


def _has_tool_results(messages: list) -> bool:
    return any(
        isinstance(m, dict) and (
            m.get("role") in ("tool", "tool_result")
            # Responses API uses type=function_call_output instead of role=tool
            or m.get("type") == "function_call_output"
        )
        for m in messages
    )


def _content_text(content: Any) -> str:
    """Stringify a message content value (str, list-of-blocks, or None) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text") or "" for b in content if isinstance(b, dict))
    return str(content)


def _is_json_heavy(text: str) -> bool:
    """True if text looks like a large JSON/structured-data blob."""
    if len(text) < 200:
        return False
    json_chars = sum(1 for c in text if c in '{}[]:,"')
    return json_chars / len(text) > 0.08


def _extract_context_signals(provider: str, body: dict) -> dict:
    """Single-pass scan of messages to extract structural cost-driver signals."""
    # Responses API uses "input" instead of "messages"
    messages: list = body.get("messages") or body.get("input") or []
    # Responses API allows "input" to be a plain string — normalise to list
    if isinstance(messages, str):
        messages = []

    # System prompt:
    #   Anthropic: top-level "system"
    #   OpenAI Responses API: top-level "instructions"
    #   OpenAI Chat Completions: role=system in messages
    system_text = ""
    raw_system = body.get("system") or body.get("instructions")
    if raw_system:
        system_text = raw_system if isinstance(raw_system, str) else str(raw_system)
    if not system_text:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_text = _content_text(msg.get("content"))
                break

    system_prompt_chars = len(system_text)
    total_message_chars = len(system_text)
    tool_result_chars = 0
    thinking_input_chars = 0
    has_code_blocks = "```" in system_text
    has_structured_data = _is_json_heavy(system_text)

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue  # already counted
        # OpenAI Responses API: reasoning items fed back as input context
        if msg.get("type") == "reasoning":
            for s in (msg.get("summary") or []):
                if isinstance(s, dict):
                    thinking_input_chars += len(s.get("text") or "")
            continue
        is_tool_result = (
            role in ("tool", "tool_result")
            or msg.get("type") == "function_call_output"
        )
        # Responses API tool results carry their content in "output", not "content"
        raw_content = msg.get("content") if not is_tool_result else (
            msg.get("content") or msg.get("output")
        )
        text = _content_text(raw_content)
        total_message_chars += len(text)
        if is_tool_result:
            tool_result_chars += len(text)
        if not has_code_blocks and "```" in text:
            has_code_blocks = True
        if not has_structured_data and _is_json_heavy(text):
            has_structured_data = True
        # Anthropic: thinking blocks in assistant messages echoed back as context
        if isinstance(raw_content, list):
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_input_chars += len(block.get("thinking") or "")

    return {
        "system_prompt_chars": system_prompt_chars,
        "total_message_chars": total_message_chars,
        "tool_result_chars": tool_result_chars,
        "has_code_blocks": has_code_blocks,
        "has_structured_data": has_structured_data,
        "thinking_input_chars": thinking_input_chars,
    }
