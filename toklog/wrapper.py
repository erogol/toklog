"""Intercept create() methods per provider."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sysconfig
import time
from datetime import datetime, timezone
from typing import Any, Optional

from toklog.logger import log_entry

_PACKAGE_DIR = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

# Standard library root (e.g. /usr/lib/python3.11)
_STDLIB_PREFIX = os.path.normpath(sysconfig.get_paths()["stdlib"])

_DYNAMIC_FILENAMES = {"<string>", "<stdin>", "<console>"}


def _extract_output_code_chars(text: str) -> int:
    """Count chars of code block content (excluding delimiters) in fenced blocks."""
    if not text:
        return 0
    try:
        matches = re.findall(r'```[^\n]*\n([\s\S]*?)```', text)
        return sum(len(m) for m in matches)
    except Exception:
        return 0


def _set_openai_output_fields(entry: dict, response: Any) -> None:
    """Extract output decomposition fields from an OpenAI non-streaming response."""
    try:
        details = getattr(getattr(response, "usage", None), "completion_tokens_details", None)
        entry["thinking_tokens"] = getattr(details, "reasoning_tokens", None) if details is not None else None
    except Exception:
        pass
    try:
        tc_list = response.choices[0].message.tool_calls or []
        total_chars = 0
        for tc in tc_list:
            try:
                args = tc.function.arguments
                total_chars += len(args) if isinstance(args, str) else len(json.dumps(args))
            except Exception:
                pass
        entry["tool_call_output_tokens"] = total_chars // 4 if total_chars > 0 else None
    except Exception:
        pass
    try:
        content = getattr(
            getattr(getattr(response, "choices", [None])[0], "message", None), "content", None
        ) or ""
        chars = _extract_output_code_chars(content)
        entry["output_code_chars"] = chars if chars > 0 else None
    except Exception:
        pass


def _set_anthropic_output_fields(entry: dict, response: Any) -> None:
    """Extract output decomposition fields from an Anthropic non-streaming response."""
    content = getattr(response, "content", None) or []
    try:
        thinking_chars = sum(
            len(getattr(b, "thinking", "") or "")
            for b in content if getattr(b, "type", None) == "thinking"
        )
        entry["thinking_tokens"] = thinking_chars // 4 if thinking_chars > 0 else None
    except Exception:
        pass
    try:
        tool_chars = sum(
            len(json.dumps(b.input))
            for b in content if getattr(b, "type", None) == "tool_use" and hasattr(b, "input")
        )
        entry["tool_call_output_tokens"] = tool_chars // 4 if tool_chars > 0 else None
    except Exception:
        pass
    try:
        code_chars = sum(
            _extract_output_code_chars(b.text)
            for b in content if getattr(b, "type", None) == "text" and hasattr(b, "text")
        )
        entry["output_code_chars"] = code_chars if code_chars > 0 else None
    except Exception:
        pass


def _get_call_site() -> Optional[dict]:
    """Return the first user-code frame outside this package, stdlib, and installed packages."""
    try:
        pkg_prefix = _PACKAGE_DIR + os.sep
        for frame_info in inspect.stack(context=0):
            filename = os.path.normpath(frame_info.filename)
            if filename.startswith(pkg_prefix):
                continue
            if filename in _DYNAMIC_FILENAMES or os.path.basename(filename).startswith("<"):
                continue
            # Skip Python standard library
            if filename.startswith(_STDLIB_PREFIX + os.sep):
                continue
            # Skip third-party installed packages
            filename_fwd = filename.replace("\\", "/")
            if "site-packages" in filename_fwd or "dist-packages" in filename_fwd:
                continue
            relpath = os.path.relpath(filename)
            # Count leading ".." components
            parts = relpath.replace("\\", "/").split("/")
            leading_dotdot = 0
            for p in parts:
                if p == "..":
                    leading_dotdot += 1
                else:
                    break
            if leading_dotdot > 3:
                relpath = os.path.basename(filename)
            return {
                "file": relpath,
                "function": frame_info.function,
                "line": frame_info.lineno,
            }
        return None
    except Exception:
        return None


def wrap(
    client: Any,
    default_tags: Optional[str] = None,
    log_preview: bool = True,
) -> Any:
    """Wrap an OpenAI or Anthropic client to log all LLM calls.

    Patches the instance (not the class). Idempotent.
    """
    if getattr(client, "_toklog_wrapped", False):
        return client

    try:
        import openai

        if isinstance(client, openai.OpenAI):
            _wrap_openai(client, default_tags, log_preview)
            client._toklog_wrapped = True
            return client
        if isinstance(client, openai.AsyncOpenAI):
            _wrap_openai_async(client, default_tags, log_preview)
            client._toklog_wrapped = True
            return client
    except ImportError:
        pass

    try:
        import anthropic

        if isinstance(client, anthropic.Anthropic):
            _wrap_anthropic(client, default_tags, log_preview)
            client._toklog_wrapped = True
            return client
        if isinstance(client, anthropic.AsyncAnthropic):
            _wrap_anthropic_async(client, default_tags, log_preview)
            client._toklog_wrapped = True
            return client
    except ImportError:
        pass

    raise ImportError(
        "toklog requires either 'openai' or 'anthropic' package. "
        "Supported types: OpenAI, AsyncOpenAI, Anthropic, AsyncAnthropic. "
        "Install with: pip install toklog[openai]"
    )


def _hash_system_prompt(content: Any) -> Optional[str]:
    """Hash system prompt content to a short hex string."""
    if content is None:
        return None
    if isinstance(content, list):
        # Anthropic content blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    else:
        text = str(content)
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _estimate_tool_tokens(tools: Any) -> int:
    """Estimate token count of tool schemas."""
    if not tools:
        return 0
    try:
        return len(json.dumps(tools, default=str)) // 4
    except Exception:
        return 0


def _extract_tag(kwargs: dict, default_tags: Optional[str]) -> Optional[str]:
    """Extract tag from extra_headers or fall back to default."""
    headers = kwargs.get("extra_headers") or {}
    tag = headers.get("X-TB-Tag")
    if tag is not None:
        return tag
    return default_tags


def _truncate_preview(text: str, max_len: int) -> str:
    """Truncate text at word boundary, up to max_len chars."""
    if len(text) <= max_len:
        return text
    start_search = int(max_len * 0.75)
    space_idx = text.rfind(" ", start_search, max_len)
    if space_idx != -1:
        return text[:space_idx]
    return text[:max_len]


def _extract_user_preview(kwargs: dict) -> Optional[str]:
    """Extract and truncate the last user message from kwargs for preview logging."""
    messages = kwargs.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text = " ".join(parts)
        elif isinstance(content, str):
            text = content
        else:
            continue
        if not text:
            return None
        return _truncate_preview(text, 200)
    return None


def _extract_tool_names(kwargs: dict) -> list:
    """Extract tool names from the tools schema passed to the API call."""
    tools = kwargs.get("tools") or []
    names = []
    for t in tools:
        if isinstance(t, dict):
            # OpenAI format: {"type": "function", "function": {"name": "..."}}
            name = t.get("name") or (t.get("function") or {}).get("name")
            if name:
                names.append(name)
    return names


def _extract_has_tool_results(kwargs: dict) -> bool:
    """Check if messages contain any tool or tool_result role (mid-agent-loop signal)."""
    messages = kwargs.get("messages") or []
    return any(
        isinstance(m, dict) and m.get("role") in ("tool", "tool_result")
        for m in messages
    )


def _content_text(content: Any) -> str:
    """Stringify a message content value (str, list-of-blocks, or None) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text") or "")
        return " ".join(parts)
    return str(content)


def _is_json_heavy(text: str) -> bool:
    """True if text looks like a large JSON/structured-data blob."""
    if len(text) < 200:
        return False
    json_chars = sum(1 for c in text if c in '{}[]:,"')
    return json_chars / len(text) > 0.08


def _extract_context_signals(provider: str, kwargs: dict) -> dict:
    """Single-pass scan of messages to extract structural cost-driver signals."""
    messages = kwargs.get("messages") or []

    # System prompt: Anthropic uses top-level "system" kwarg; OpenAI uses role=system in messages
    system_text = ""
    if provider == "anthropic":
        raw_system = kwargs.get("system")
        if raw_system:
            system_text = str(raw_system) if not isinstance(raw_system, str) else raw_system
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
            continue  # already counted above
        content = msg.get("content")
        text = _content_text(content)
        total_message_chars += len(text)
        if role in ("tool", "tool_result"):
            tool_result_chars += len(text)
        if not has_code_blocks and "```" in text:
            has_code_blocks = True
        if not has_structured_data and _is_json_heavy(text):
            has_structured_data = True
        # Anthropic: thinking blocks in assistant messages echoed back as context
        if isinstance(content, list):
            for block in content:
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


def _extract_assistant_preview(response: Any) -> Optional[str]:
    """Extract and truncate the assistant reply text from the response object."""
    try:
        # OpenAI: response.choices[0].message.content
        choices = getattr(response, "choices", None)
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg:
                text = getattr(msg, "content", None) or ""
                if text and isinstance(text, str):
                    return _truncate_preview(text, 300)
        # Anthropic: response.content is a list of blocks
        content = getattr(response, "content", None)
        if isinstance(content, list):
            parts = [
                b.text for b in content
                if hasattr(b, "type") and b.type == "text" and hasattr(b, "text")
            ]
            text = " ".join(parts)
            if text:
                return _truncate_preview(text, 300)
    except Exception:
        pass
    return None


def _key_hint(api_key: Any) -> Optional[str]:
    """Return a safe identifier for an API key: last 8 chars prefixed with '...'."""
    if not api_key or not isinstance(api_key, str):
        return None
    key = api_key.strip()
    if len(key) < 4:
        return None
    return "..." + key[-8:]


def _extract_key_hint(client: Any) -> Optional[str]:
    """Extract API key hint from client, checking api_key and auth_token."""
    for attr in ("api_key", "auth_token"):
        val = getattr(client, attr, None)
        hint = _key_hint(val)
        if hint is not None:
            return hint
    return None


def _build_base_entry(
    provider: str,
    model: str,
    kwargs: dict,
    default_tags: Optional[str],
    start_time: float,
    log_preview: bool = True,
    api_key_hint: Optional[str] = None,
) -> dict:
    """Build the common log entry fields from request kwargs."""
    tools = kwargs.get("tools")
    entry: dict = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z",
        "provider": provider,
        "model": model,
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "max_tokens_set": kwargs.get("max_tokens"),
        "system_prompt_hash": None,
        "tool_count": len(tools) if tools else 0,
        "tool_schema_tokens": _estimate_tool_tokens(tools),
        "tool_calls_made": None,
        "tags": _extract_tag(kwargs, default_tags),
        "streaming": kwargs.get("stream", False),
        "duration_ms": None,
        "error": False,
        "error_type": None,
        "request_id": None,
        "api_key_hint": api_key_hint,
        "call_site": _get_call_site(),
        "user_message_preview": _extract_user_preview(kwargs) if log_preview else None,
        "tool_names": _extract_tool_names(kwargs) if log_preview else [],
        "has_tool_results": _extract_has_tool_results(kwargs),
        "assistant_preview": None,
        "thinking_tokens": None,
        "tool_call_output_tokens": None,
        "output_code_chars": None,
        **_extract_context_signals(provider, kwargs),
    }
    return entry


def _wrap_openai(client: Any, default_tags: Optional[str], log_preview: bool) -> None:
    """Patch client.chat.completions.create on the instance."""
    completions = client.chat.completions
    original_create = completions.create
    hint = _extract_key_hint(client)

    def _extract_system_hash(kwargs: dict) -> Optional[str]:
        messages = kwargs.get("messages", [])
        system_parts = [
            m.get("content", "") for m in messages if m.get("role") == "system"
        ]
        if not system_parts:
            return None
        return _hash_system_prompt("\n".join(str(p) for p in system_parts))

    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        model = kwargs.get("model", "unknown")
        entry = _build_base_entry("openai", model, kwargs, default_tags, start, log_preview, api_key_hint=hint)
        entry["system_prompt_hash"] = _extract_system_hash(kwargs)

        is_stream = kwargs.get("stream", False)

        if is_stream:
            # Inject stream_options to get usage in final chunk
            stream_opts = kwargs.get("stream_options") or {}
            stream_opts["include_usage"] = True
            kwargs["stream_options"] = stream_opts

        # Strip TokLog headers before forwarding
        headers = kwargs.get("extra_headers")
        if headers and "X-TB-Tag" in headers:
            headers = {k: v for k, v in headers.items() if not k.startswith("X-TB-")}
            if headers:
                kwargs["extra_headers"] = headers
            else:
                kwargs.pop("extra_headers", None)

        try:
            response = original_create(*args, **kwargs)
        except Exception as exc:
            entry["duration_ms"] = int((time.monotonic() - start) * 1000)
            entry["error"] = True
            entry["error_type"] = type(exc).__name__
            log_entry(entry)
            raise

        if is_stream:
            return _wrap_openai_stream(response, entry, start)

        # Non-streaming: read usage directly
        entry["duration_ms"] = int((time.monotonic() - start) * 1000)
        if hasattr(response, "usage") and response.usage is not None:
            entry["input_tokens"] = response.usage.prompt_tokens
            entry["output_tokens"] = response.usage.completion_tokens
            entry["cache_read_tokens"] = getattr(
                response.usage, "prompt_tokens_details", None
            )
            if entry["cache_read_tokens"] is not None:
                entry["cache_read_tokens"] = getattr(
                    entry["cache_read_tokens"], "cached_tokens", 0
                ) or 0
            else:
                entry["cache_read_tokens"] = 0

        try:
            tc = response.choices[0].message.tool_calls
            entry["tool_calls_made"] = len(tc) if tc else 0
        except (AttributeError, TypeError, IndexError):
            pass

        # Override model from response if available
        if getattr(response, "model", None):
            entry["model"] = response.model

        # Request ID from response headers or model attribute
        entry["request_id"] = getattr(response, "_request_id", None) or getattr(
            response, "id", None
        )
        if log_preview:
            entry["assistant_preview"] = _extract_assistant_preview(response)
        _set_openai_output_fields(entry, response)
        log_entry(entry)
        return response

    completions.create = wrapped_create


def _wrap_openai_stream(stream: Any, entry: dict, start: float) -> Any:
    """Wrap an OpenAI streaming response to capture usage from final chunk."""

    class StreamWrapper:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __iter__(self) -> Any:
            return self._iterate()

        def __enter__(self) -> "StreamWrapper":
            if hasattr(self._inner, "__enter__"):
                self._inner.__enter__()
            return self

        def __exit__(self, *args: Any) -> None:
            if hasattr(self._inner, "__exit__"):
                self._inner.__exit__(*args)

        def _iterate(self) -> Any:
            tool_call_indices: set = set()
            saw_chunks = False
            _thinking_tokens = None
            _tool_arg_chars = 0
            _text_parts: list = []
            try:
                for chunk in self._inner:
                    saw_chunks = True
                    # Check for usage in final chunk
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        entry["input_tokens"] = getattr(
                            chunk.usage, "prompt_tokens", None
                        )
                        entry["output_tokens"] = getattr(
                            chunk.usage, "completion_tokens", None
                        )
                        try:
                            ptd = getattr(chunk.usage, "prompt_tokens_details", None)
                            if ptd is not None:
                                entry["cache_read_tokens"] = getattr(ptd, "cached_tokens", 0) or 0
                        except Exception:
                            pass
                        try:
                            details = getattr(chunk.usage, "completion_tokens_details", None)
                            if details is not None:
                                rt = getattr(details, "reasoning_tokens", None)
                                if rt is not None:
                                    _thinking_tokens = rt
                        except Exception:
                            pass
                    if getattr(chunk, "model", None):
                        entry["model"] = chunk.model
                    try:
                        tcs = chunk.choices[0].delta.tool_calls
                        if tcs:
                            for tc in tcs:
                                if hasattr(tc, "index"):
                                    tool_call_indices.add(tc.index)
                                try:
                                    args = tc.function.arguments
                                    if args:
                                        _tool_arg_chars += len(args)
                                except Exception:
                                    pass
                    except (AttributeError, TypeError, IndexError):
                        pass
                    try:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            _text_parts.append(delta_content)
                    except (AttributeError, TypeError, IndexError):
                        pass
                    yield chunk
            finally:
                if saw_chunks:
                    entry["tool_calls_made"] = len(tool_call_indices)
                entry["duration_ms"] = int((time.monotonic() - start) * 1000)
                try:
                    entry["thinking_tokens"] = _thinking_tokens
                except Exception:
                    pass
                try:
                    entry["tool_call_output_tokens"] = _tool_arg_chars // 4 if _tool_arg_chars > 0 else None
                except Exception:
                    pass
                try:
                    full_text = "".join(_text_parts)
                    chars = _extract_output_code_chars(full_text)
                    entry["output_code_chars"] = chars if chars > 0 else None
                except Exception:
                    pass
                log_entry(entry)

    return StreamWrapper(stream)


def _wrap_anthropic(client: Any, default_tags: Optional[str], log_preview: bool) -> None:
    """Patch client.messages.create on the instance."""
    messages_api = client.messages
    original_create = messages_api.create
    hint = _extract_key_hint(client)

    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        model = kwargs.get("model", "unknown")
        entry = _build_base_entry("anthropic", model, kwargs, default_tags, start, log_preview, api_key_hint=hint)
        entry["system_prompt_hash"] = _hash_system_prompt(kwargs.get("system"))
        # Anthropic uses max_tokens as required param
        entry["max_tokens_set"] = kwargs.get("max_tokens")

        is_stream = kwargs.get("stream", False)

        # Strip TokLog headers before forwarding
        headers = kwargs.get("extra_headers")
        if headers and "X-TB-Tag" in headers:
            headers = {k: v for k, v in headers.items() if not k.startswith("X-TB-")}
            if headers:
                kwargs["extra_headers"] = headers
            else:
                kwargs.pop("extra_headers", None)

        try:
            response = original_create(*args, **kwargs)
        except Exception as exc:
            entry["duration_ms"] = int((time.monotonic() - start) * 1000)
            entry["error"] = True
            entry["error_type"] = type(exc).__name__
            log_entry(entry)
            raise

        if is_stream:
            return _wrap_anthropic_stream(response, entry, start)

        # Non-streaming
        entry["duration_ms"] = int((time.monotonic() - start) * 1000)
        if hasattr(response, "usage") and response.usage is not None:
            entry["input_tokens"] = getattr(response.usage, "input_tokens", None)
            entry["output_tokens"] = getattr(response.usage, "output_tokens", None)
            entry["cache_read_tokens"] = getattr(
                response.usage, "cache_read_input_tokens", 0
            ) or 0
            entry["cache_creation_tokens"] = getattr(
                response.usage, "cache_creation_input_tokens", 0
            ) or 0
        try:
            entry["tool_calls_made"] = sum(
                1 for b in response.content if getattr(b, "type", None) == "tool_use"
            )
        except (AttributeError, TypeError):
            pass

        if getattr(response, "model", None):
            entry["model"] = response.model
        entry["request_id"] = getattr(response, "id", None)
        if log_preview:
            entry["assistant_preview"] = _extract_assistant_preview(response)
        _set_anthropic_output_fields(entry, response)
        log_entry(entry)
        return response

    messages_api.create = wrapped_create

    # Also patch messages.stream() which returns a MessageStream context manager
    original_stream = getattr(messages_api, "stream", None)
    if original_stream is not None:

        def wrapped_stream(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            model = kwargs.get("model", "unknown")
            entry = _build_base_entry("anthropic", model, kwargs, default_tags, start, log_preview, api_key_hint=hint)
            entry["system_prompt_hash"] = _hash_system_prompt(kwargs.get("system"))
            entry["max_tokens_set"] = kwargs.get("max_tokens")
            entry["streaming"] = True

            # Strip TokLog headers before forwarding
            headers = kwargs.get("extra_headers")
            if headers and "X-TB-Tag" in headers:
                headers = {
                    k: v for k, v in headers.items() if not k.startswith("X-TB-")
                }
                if headers:
                    kwargs["extra_headers"] = headers
                else:
                    kwargs.pop("extra_headers", None)

            try:
                stream_obj = original_stream(*args, **kwargs)
            except Exception as exc:
                entry["duration_ms"] = int((time.monotonic() - start) * 1000)
                entry["error"] = True
                entry["error_type"] = type(exc).__name__
                log_entry(entry)
                raise

            return _MessageStreamProxy(stream_obj, entry, start)

        messages_api.stream = wrapped_stream


class _MessageStreamProxy:
    """Proxy for Anthropic MessageStream that captures usage on exit."""

    def __init__(self, inner: Any, entry: dict, start: float) -> None:
        self._inner = inner
        self._entered: Any = None
        self._entry = entry
        self._start = start

    def __enter__(self) -> "_MessageStreamProxy":
        # MessageStreamManager.__enter__ returns a MessageStream
        self._entered = self._inner.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        target = self._entered if self._entered is not None else self._inner
        try:
            msg = target.get_final_message()
            if msg is not None and hasattr(msg, "usage") and msg.usage is not None:
                self._entry["input_tokens"] = getattr(
                    msg.usage, "input_tokens", None
                )
                self._entry["output_tokens"] = getattr(
                    msg.usage, "output_tokens", None
                )
                self._entry["cache_read_tokens"] = getattr(
                    msg.usage, "cache_read_input_tokens", 0
                ) or 0
                self._entry["cache_creation_tokens"] = getattr(
                    msg.usage, "cache_creation_input_tokens", 0
                ) or 0
            if getattr(msg, "model", None):
                self._entry["model"] = msg.model
            self._entry["request_id"] = getattr(msg, "id", None)
            try:
                self._entry["tool_calls_made"] = sum(
                    1 for b in msg.content if getattr(b, "type", None) == "tool_use"
                )
            except (AttributeError, TypeError):
                pass
            _set_anthropic_output_fields(self._entry, msg)
        except Exception:
            pass
        self._entry["duration_ms"] = int((time.monotonic() - self._start) * 1000)
        log_entry(self._entry)
        self._inner.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        # After __enter__, proxy to the entered stream (MessageStream)
        if self._entered is not None:
            return getattr(self._entered, name)
        return getattr(self._inner, name)


def _wrap_anthropic_stream(stream: Any, entry: dict, start: float) -> Any:
    """Wrap an Anthropic streaming response to capture usage from message_stop."""

    class StreamWrapper:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __iter__(self) -> Any:
            return self._iterate()

        def __enter__(self) -> "StreamWrapper":
            if hasattr(self._inner, "__enter__"):
                self._inner.__enter__()
            return self

        def __exit__(self, *args: Any) -> None:
            if hasattr(self._inner, "__exit__"):
                self._inner.__exit__(*args)

        def _iterate(self) -> Any:
            tool_call_count = 0
            saw_events = False
            _thinking_chars = 0
            _tool_input_chars = 0
            _text_parts: list = []
            try:
                for event in self._inner:
                    saw_events = True
                    # Look for final message with usage
                    if hasattr(event, "type") and event.type == "message_stop":
                        pass
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        usage = event.message.usage
                        entry["input_tokens"] = getattr(usage, "input_tokens", None)
                        entry["output_tokens"] = getattr(usage, "output_tokens", None)
                        entry["cache_read_tokens"] = getattr(
                            usage, "cache_read_input_tokens", 0
                        ) or 0
                        entry["cache_creation_tokens"] = getattr(
                            usage, "cache_creation_input_tokens", 0
                        ) or 0
                        if getattr(event.message, "model", None):
                            entry["model"] = event.message.model
                    # Also check usage directly on the event (content_block_delta etc.)
                    if hasattr(event, "usage") and event.usage is not None:
                        usage = event.usage
                        if hasattr(usage, "input_tokens"):
                            entry["input_tokens"] = usage.input_tokens
                        if hasattr(usage, "output_tokens"):
                            entry["output_tokens"] = usage.output_tokens
                    try:
                        if getattr(event, "type", None) == "content_block_start":
                            cb = getattr(event, "content_block", None)
                            if cb is not None and getattr(cb, "type", None) == "tool_use":
                                tool_call_count += 1
                    except (AttributeError, TypeError):
                        pass
                    # Accumulate for output decomposition
                    try:
                        if getattr(event, "type", None) == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta is not None:
                                delta_type = getattr(delta, "type", None)
                                if delta_type == "thinking_delta":
                                    _thinking_chars += len(getattr(delta, "thinking", "") or "")
                                elif delta_type == "text_delta":
                                    _text_parts.append(getattr(delta, "text", "") or "")
                                elif delta_type == "input_json_delta":
                                    _tool_input_chars += len(getattr(delta, "partial_json", "") or "")
                    except Exception:
                        pass
                    yield event
            finally:
                if saw_events:
                    entry["tool_calls_made"] = tool_call_count
                entry["duration_ms"] = int((time.monotonic() - start) * 1000)
                try:
                    entry["thinking_tokens"] = _thinking_chars // 4 if _thinking_chars > 0 else None
                except Exception:
                    pass
                try:
                    entry["tool_call_output_tokens"] = _tool_input_chars // 4 if _tool_input_chars > 0 else None
                except Exception:
                    pass
                try:
                    full_text = "".join(_text_parts)
                    chars = _extract_output_code_chars(full_text)
                    entry["output_code_chars"] = chars if chars > 0 else None
                except Exception:
                    pass
                log_entry(entry)

    return StreamWrapper(stream)


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


def _wrap_openai_async(client: Any, default_tags: Optional[str], log_preview: bool) -> None:
    """Patch client.chat.completions.create on an AsyncOpenAI instance."""
    completions = client.chat.completions
    original_create = completions.create
    hint = _extract_key_hint(client)

    def _extract_system_hash(kwargs: dict) -> Optional[str]:
        messages = kwargs.get("messages", [])
        system_parts = [
            m.get("content", "") for m in messages if m.get("role") == "system"
        ]
        if not system_parts:
            return None
        return _hash_system_prompt("\n".join(str(p) for p in system_parts))

    async def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        model = kwargs.get("model", "unknown")
        entry = _build_base_entry("openai", model, kwargs, default_tags, start, log_preview, api_key_hint=hint)
        entry["system_prompt_hash"] = _extract_system_hash(kwargs)

        is_stream = kwargs.get("stream", False)

        if is_stream:
            stream_opts = kwargs.get("stream_options") or {}
            stream_opts["include_usage"] = True
            kwargs["stream_options"] = stream_opts

        # Strip TokLog headers before forwarding
        headers = kwargs.get("extra_headers")
        if headers and "X-TB-Tag" in headers:
            headers = {k: v for k, v in headers.items() if not k.startswith("X-TB-")}
            if headers:
                kwargs["extra_headers"] = headers
            else:
                kwargs.pop("extra_headers", None)

        try:
            response = await original_create(*args, **kwargs)
        except Exception as exc:
            entry["duration_ms"] = int((time.monotonic() - start) * 1000)
            entry["error"] = True
            entry["error_type"] = type(exc).__name__
            log_entry(entry)
            raise

        if is_stream:
            return _wrap_openai_stream_async(response, entry, start)

        # Non-streaming: read usage directly
        entry["duration_ms"] = int((time.monotonic() - start) * 1000)
        if hasattr(response, "usage") and response.usage is not None:
            entry["input_tokens"] = response.usage.prompt_tokens
            entry["output_tokens"] = response.usage.completion_tokens
            entry["cache_read_tokens"] = getattr(
                response.usage, "prompt_tokens_details", None
            )
            if entry["cache_read_tokens"] is not None:
                entry["cache_read_tokens"] = getattr(
                    entry["cache_read_tokens"], "cached_tokens", 0
                ) or 0
            else:
                entry["cache_read_tokens"] = 0

        try:
            tc = response.choices[0].message.tool_calls
            entry["tool_calls_made"] = len(tc) if tc else 0
        except (AttributeError, TypeError, IndexError):
            pass

        if getattr(response, "model", None):
            entry["model"] = response.model

        entry["request_id"] = getattr(response, "_request_id", None) or getattr(
            response, "id", None
        )
        if log_preview:
            entry["assistant_preview"] = _extract_assistant_preview(response)
        _set_openai_output_fields(entry, response)
        log_entry(entry)
        return response

    completions.create = wrapped_create


def _wrap_openai_stream_async(stream: Any, entry: dict, start: float) -> Any:
    """Wrap an async OpenAI streaming response to capture usage from final chunk."""

    class AsyncStreamWrapper:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __aiter__(self) -> Any:
            return self._iterate()

        async def __aenter__(self) -> "AsyncStreamWrapper":
            if hasattr(self._inner, "__aenter__"):
                await self._inner.__aenter__()
            return self

        async def __aexit__(self, *args: Any) -> None:
            if hasattr(self._inner, "__aexit__"):
                await self._inner.__aexit__(*args)

        async def _iterate(self) -> Any:
            tool_call_indices: set = set()
            saw_chunks = False
            _thinking_tokens = None
            _tool_arg_chars = 0
            _text_parts: list = []
            try:
                async for chunk in self._inner:
                    saw_chunks = True
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        entry["input_tokens"] = getattr(
                            chunk.usage, "prompt_tokens", None
                        )
                        entry["output_tokens"] = getattr(
                            chunk.usage, "completion_tokens", None
                        )
                        try:
                            ptd = getattr(chunk.usage, "prompt_tokens_details", None)
                            if ptd is not None:
                                entry["cache_read_tokens"] = getattr(ptd, "cached_tokens", 0) or 0
                        except Exception:
                            pass
                        try:
                            details = getattr(chunk.usage, "completion_tokens_details", None)
                            if details is not None:
                                rt = getattr(details, "reasoning_tokens", None)
                                if rt is not None:
                                    _thinking_tokens = rt
                        except Exception:
                            pass
                    if getattr(chunk, "model", None):
                        entry["model"] = chunk.model
                    try:
                        tcs = chunk.choices[0].delta.tool_calls
                        if tcs:
                            for tc in tcs:
                                if hasattr(tc, "index"):
                                    tool_call_indices.add(tc.index)
                                try:
                                    args = tc.function.arguments
                                    if args:
                                        _tool_arg_chars += len(args)
                                except Exception:
                                    pass
                    except (AttributeError, TypeError, IndexError):
                        pass
                    try:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            _text_parts.append(delta_content)
                    except (AttributeError, TypeError, IndexError):
                        pass
                    yield chunk
            finally:
                if saw_chunks:
                    entry["tool_calls_made"] = len(tool_call_indices)
                entry["duration_ms"] = int((time.monotonic() - start) * 1000)
                try:
                    entry["thinking_tokens"] = _thinking_tokens
                except Exception:
                    pass
                try:
                    entry["tool_call_output_tokens"] = _tool_arg_chars // 4 if _tool_arg_chars > 0 else None
                except Exception:
                    pass
                try:
                    full_text = "".join(_text_parts)
                    chars = _extract_output_code_chars(full_text)
                    entry["output_code_chars"] = chars if chars > 0 else None
                except Exception:
                    pass
                log_entry(entry)

    return AsyncStreamWrapper(stream)


def _wrap_anthropic_async(client: Any, default_tags: Optional[str], log_preview: bool) -> None:
    """Patch client.messages.create on an AsyncAnthropic instance."""
    messages_api = client.messages
    original_create = messages_api.create
    hint = _extract_key_hint(client)

    async def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        model = kwargs.get("model", "unknown")
        entry = _build_base_entry("anthropic", model, kwargs, default_tags, start, log_preview, api_key_hint=hint)
        entry["system_prompt_hash"] = _hash_system_prompt(kwargs.get("system"))
        entry["max_tokens_set"] = kwargs.get("max_tokens")

        is_stream = kwargs.get("stream", False)

        # Strip TokLog headers before forwarding
        headers = kwargs.get("extra_headers")
        if headers and "X-TB-Tag" in headers:
            headers = {k: v for k, v in headers.items() if not k.startswith("X-TB-")}
            if headers:
                kwargs["extra_headers"] = headers
            else:
                kwargs.pop("extra_headers", None)

        try:
            response = await original_create(*args, **kwargs)
        except Exception as exc:
            entry["duration_ms"] = int((time.monotonic() - start) * 1000)
            entry["error"] = True
            entry["error_type"] = type(exc).__name__
            log_entry(entry)
            raise

        if is_stream:
            return _wrap_anthropic_stream_async(response, entry, start)

        # Non-streaming
        entry["duration_ms"] = int((time.monotonic() - start) * 1000)
        if hasattr(response, "usage") and response.usage is not None:
            entry["input_tokens"] = getattr(response.usage, "input_tokens", None)
            entry["output_tokens"] = getattr(response.usage, "output_tokens", None)
            entry["cache_read_tokens"] = getattr(
                response.usage, "cache_read_input_tokens", 0
            ) or 0
            entry["cache_creation_tokens"] = getattr(
                response.usage, "cache_creation_input_tokens", 0
            ) or 0
        try:
            entry["tool_calls_made"] = sum(
                1 for b in response.content if getattr(b, "type", None) == "tool_use"
            )
        except (AttributeError, TypeError):
            pass

        if getattr(response, "model", None):
            entry["model"] = response.model
        entry["request_id"] = getattr(response, "id", None)
        if log_preview:
            entry["assistant_preview"] = _extract_assistant_preview(response)
        _set_anthropic_output_fields(entry, response)
        log_entry(entry)
        return response

    messages_api.create = wrapped_create

    # Also patch messages.stream() with async proxy
    original_stream = getattr(messages_api, "stream", None)
    if original_stream is not None:

        def wrapped_stream(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            model = kwargs.get("model", "unknown")
            entry = _build_base_entry("anthropic", model, kwargs, default_tags, start, log_preview, api_key_hint=hint)
            entry["system_prompt_hash"] = _hash_system_prompt(kwargs.get("system"))
            entry["max_tokens_set"] = kwargs.get("max_tokens")
            entry["streaming"] = True

            # Strip TokLog headers before forwarding
            headers = kwargs.get("extra_headers")
            if headers and "X-TB-Tag" in headers:
                headers = {
                    k: v for k, v in headers.items() if not k.startswith("X-TB-")
                }
                if headers:
                    kwargs["extra_headers"] = headers
                else:
                    kwargs.pop("extra_headers", None)

            try:
                stream_obj = original_stream(*args, **kwargs)
            except Exception as exc:
                entry["duration_ms"] = int((time.monotonic() - start) * 1000)
                entry["error"] = True
                entry["error_type"] = type(exc).__name__
                log_entry(entry)
                raise

            return _AsyncMessageStreamProxy(stream_obj, entry, start)

        messages_api.stream = wrapped_stream


def _wrap_anthropic_stream_async(stream: Any, entry: dict, start: float) -> Any:
    """Wrap an async Anthropic streaming response to capture usage from events."""

    class AsyncStreamWrapper:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __aiter__(self) -> Any:
            return self._iterate()

        async def __aenter__(self) -> "AsyncStreamWrapper":
            if hasattr(self._inner, "__aenter__"):
                await self._inner.__aenter__()
            return self

        async def __aexit__(self, *args: Any) -> None:
            if hasattr(self._inner, "__aexit__"):
                await self._inner.__aexit__(*args)

        async def _iterate(self) -> Any:
            tool_call_count = 0
            saw_events = False
            _thinking_chars = 0
            _tool_input_chars = 0
            _text_parts: list = []
            try:
                async for event in self._inner:
                    saw_events = True
                    if hasattr(event, "type") and event.type == "message_stop":
                        pass
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        usage = event.message.usage
                        entry["input_tokens"] = getattr(usage, "input_tokens", None)
                        entry["output_tokens"] = getattr(usage, "output_tokens", None)
                        entry["cache_read_tokens"] = getattr(
                            usage, "cache_read_input_tokens", 0
                        ) or 0
                        entry["cache_creation_tokens"] = getattr(
                            usage, "cache_creation_input_tokens", 0
                        ) or 0
                        if getattr(event.message, "model", None):
                            entry["model"] = event.message.model
                    if hasattr(event, "usage") and event.usage is not None:
                        usage = event.usage
                        if hasattr(usage, "input_tokens"):
                            entry["input_tokens"] = usage.input_tokens
                        if hasattr(usage, "output_tokens"):
                            entry["output_tokens"] = usage.output_tokens
                    try:
                        if getattr(event, "type", None) == "content_block_start":
                            cb = getattr(event, "content_block", None)
                            if cb is not None and getattr(cb, "type", None) == "tool_use":
                                tool_call_count += 1
                    except (AttributeError, TypeError):
                        pass
                    # Accumulate for output decomposition
                    try:
                        if getattr(event, "type", None) == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta is not None:
                                delta_type = getattr(delta, "type", None)
                                if delta_type == "thinking_delta":
                                    _thinking_chars += len(getattr(delta, "thinking", "") or "")
                                elif delta_type == "text_delta":
                                    _text_parts.append(getattr(delta, "text", "") or "")
                                elif delta_type == "input_json_delta":
                                    _tool_input_chars += len(getattr(delta, "partial_json", "") or "")
                    except Exception:
                        pass
                    yield event
            finally:
                if saw_events:
                    entry["tool_calls_made"] = tool_call_count
                entry["duration_ms"] = int((time.monotonic() - start) * 1000)
                try:
                    entry["thinking_tokens"] = _thinking_chars // 4 if _thinking_chars > 0 else None
                except Exception:
                    pass
                try:
                    entry["tool_call_output_tokens"] = _tool_input_chars // 4 if _tool_input_chars > 0 else None
                except Exception:
                    pass
                try:
                    full_text = "".join(_text_parts)
                    chars = _extract_output_code_chars(full_text)
                    entry["output_code_chars"] = chars if chars > 0 else None
                except Exception:
                    pass
                log_entry(entry)

    return AsyncStreamWrapper(stream)


class _AsyncMessageStreamProxy:
    """Async proxy for Anthropic AsyncMessageStream that captures usage on exit."""

    def __init__(self, inner: Any, entry: dict, start: float) -> None:
        self._inner = inner
        self._entered: Any = None
        self._entry = entry
        self._start = start

    async def __aenter__(self) -> "_AsyncMessageStreamProxy":
        self._entered = await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        target = self._entered if self._entered is not None else self._inner
        try:
            msg = target.get_final_message()
            if msg is not None and hasattr(msg, "usage") and msg.usage is not None:
                self._entry["input_tokens"] = getattr(msg.usage, "input_tokens", None)
                self._entry["output_tokens"] = getattr(msg.usage, "output_tokens", None)
                self._entry["cache_read_tokens"] = getattr(
                    msg.usage, "cache_read_input_tokens", 0
                ) or 0
                self._entry["cache_creation_tokens"] = getattr(
                    msg.usage, "cache_creation_input_tokens", 0
                ) or 0
            if getattr(msg, "model", None):
                self._entry["model"] = msg.model
            self._entry["request_id"] = getattr(msg, "id", None)
            try:
                self._entry["tool_calls_made"] = sum(
                    1 for b in msg.content if getattr(b, "type", None) == "tool_use"
                )
            except (AttributeError, TypeError):
                pass
            _set_anthropic_output_fields(self._entry, msg)
        except Exception:
            pass
        self._entry["duration_ms"] = int((time.monotonic() - self._start) * 1000)
        log_entry(self._entry)
        await self._inner.__aexit__(*args)

    def get_final_message(self) -> Any:
        target = self._entered if self._entered is not None else self._inner
        return target.get_final_message()

    def __getattr__(self, name: str) -> Any:
        if self._entered is not None:
            return getattr(self._entered, name)
        return getattr(self._inner, name)
