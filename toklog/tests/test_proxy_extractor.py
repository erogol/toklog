"""Unit tests for toklog/proxy/extractor.py."""

from __future__ import annotations

import json

from toklog.proxy.extractor import (
    _derive_program,
    _hash_system,
    _has_tool_results,
    _key_hint_from_headers,
    extract_from_request,
)


# ---------------------------------------------------------------------------
# extract_from_request
# ---------------------------------------------------------------------------


def test_extract_openai_basic():
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }
    entry = extract_from_request("openai", body, tag=None)

    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-4o"
    assert entry["max_tokens_set"] == 100
    assert entry["streaming"] is False
    assert entry["instrumentation"] == "proxy"
    assert entry["call_site"] is None
    assert entry["user_message_preview"] is None


def test_extract_anthropic_basic():
    body = {
        "model": "claude-opus-4-6",
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 512,
        "stream": True,
    }
    entry = extract_from_request("anthropic", body, tag="job:test")

    assert entry["provider"] == "anthropic"
    assert entry["model"] == "claude-opus-4-6"
    assert entry["streaming"] is True
    assert entry["tags"] == "job:test"
    assert entry["system_prompt_hash"] is not None
    assert len(entry["system_prompt_hash"]) == 16


def test_extract_tools():
    body = {
        "model": "gpt-4o",
        "messages": [],
        "tools": [
            {"type": "function", "function": {"name": "search"}},
            {"type": "function", "function": {"name": "write"}},
        ],
    }
    entry = extract_from_request("openai", body, tag=None)

    assert entry["tool_count"] == 2
    assert entry["tool_names"] == ["search", "write"]
    assert entry["tool_schema_tokens"] > 0


def test_extract_openai_system_from_messages():
    """OpenAI embeds system in messages; extractor must find it."""
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ],
    }
    entry = extract_from_request("openai", body, tag=None)

    assert entry["system_prompt_hash"] is not None


def test_extract_has_tool_results():
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Use the tool"},
            {"role": "tool", "content": "result"},
        ],
    }
    entry = extract_from_request("openai", body, tag=None)
    assert entry["has_tool_results"] is True


def test_extract_has_tool_results_responses_api():
    """Responses API uses type=function_call_output instead of role=tool."""
    body = {
        "model": "gpt-5.4",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "run it"}]},
            {"type": "function_call_output", "call_id": "call_1", "output": "done"},
        ],
    }
    entry = extract_from_request("openai", body, tag=None)
    assert entry["has_tool_results"] is True


def test_extract_responses_api_body():
    """extract_from_request handles Responses API body shape correctly."""
    body = {
        "model": "gpt-5.4",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ],
        "instructions": "You are a helpful assistant.",
        "max_output_tokens": 512,
        "tools": [
            {"type": "function", "name": "search", "description": "search", "parameters": {}},
        ],
        "stream": True,
    }
    entry = extract_from_request("openai", body, tag=None)

    assert entry["model"] == "gpt-5.4"
    assert entry["max_tokens_set"] == 512
    assert entry["system_prompt_hash"] is not None  # extracted from instructions
    assert entry["tool_count"] == 1
    assert entry["tool_names"] == ["search"]
    assert entry["streaming"] is True
    assert entry["has_tool_results"] is False


def test_extract_responses_api_system_hash_matches_instructions():
    """system_prompt_hash for Responses API matches hash of the instructions string."""
    instructions = "Be concise and precise."
    body = {"model": "gpt-5.4", "input": [], "instructions": instructions}
    entry = extract_from_request("openai", body, tag=None)
    assert entry["system_prompt_hash"] == _hash_system(instructions)


def test_context_signals_responses_api_input():
    """Context signals use 'input' items when 'messages' is absent (Responses API)."""
    body = {
        "model": "gpt-5.4",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello world"}]},
        ],
        "instructions": "Be brief.",
    }
    entry = extract_from_request("openai", body, tag=None)
    # total_message_chars should include the user message text + instructions
    assert entry["total_message_chars"] > 0
    assert entry["system_prompt_chars"] == len("Be brief.")


def test_context_signal_tool_result_chars_responses_api():
    """tool_result_chars counts function_call_output items in Responses API input."""
    output_text = "x" * 150
    body = {
        "model": "gpt-5.4",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "run it"}]},
            {"type": "function_call_output", "call_id": "c1", "output": output_text},
        ],
    }
    entry = extract_from_request("openai", body, tag=None)
    assert entry["tool_result_chars"] == len(output_text)


def test_extract_empty_body():
    entry = extract_from_request("openai", {}, tag=None)
    assert entry["model"] == "unknown"
    assert entry["tool_count"] == 0
    assert entry["has_tool_results"] is False




# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_hash_system_none():
    assert _hash_system(None) is None


def test_hash_system_empty_string():
    assert _hash_system("") is None
    assert _hash_system("   ") is None


def test_hash_system_returns_16_hex():
    h = _hash_system("You are helpful.")
    assert h is not None
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_has_tool_results_false():
    assert _has_tool_results([{"role": "user", "content": "hi"}]) is False


def test_has_tool_results_true():
    assert _has_tool_results([{"role": "tool_result", "content": "ok"}]) is True


# ---------------------------------------------------------------------------
# context signal fields
# ---------------------------------------------------------------------------


def test_context_signal_fields_present():
    """All context signal fields appear in extract_from_request output."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    entry = extract_from_request("openai", body, tag=None)
    for field in ("system_prompt_chars", "total_message_chars", "tool_result_chars",
                  "has_code_blocks", "has_structured_data", "thinking_input_chars"):
        assert field in entry, f"missing field: {field}"


def test_context_signal_tool_result_chars():
    """tool_result_chars counts content of tool role messages."""
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "do it"},
            {"role": "tool", "content": "a" * 300},
        ],
    }
    entry = extract_from_request("openai", body, tag=None)
    assert entry["tool_result_chars"] == 300


def test_context_signal_has_code_blocks():
    """has_code_blocks is True when any message contains triple backticks."""
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "```python\nprint('hi')\n```"}],
    }
    entry = extract_from_request("openai", body, tag=None)
    assert entry["has_code_blocks"] is True


def test_context_signal_system_prompt_chars_anthropic():
    """system_prompt_chars reflects the top-level Anthropic system field."""
    system_text = "You are a helpful assistant."
    body = {
        "model": "claude-opus-4-6",
        "system": system_text,
        "messages": [{"role": "user", "content": "hi"}],
    }
    entry = extract_from_request("anthropic", body, tag=None)
    assert entry["system_prompt_chars"] == len(system_text)


def test_context_signal_system_prompt_chars_openai():
    """system_prompt_chars reflects a role=system message for OpenAI."""
    system_text = "Be concise."
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "hi"},
        ],
    }
    entry = extract_from_request("openai", body, tag=None)
    assert entry["system_prompt_chars"] == len(system_text)


# ---------------------------------------------------------------------------
# _key_hint_from_headers
# ---------------------------------------------------------------------------


class TestKeyHintFromHeaders:
    def test_openai_bearer_static_key(self) -> None:
        headers = {"authorization": "Bearer sk-proj-abcdef12345678"}
        assert _key_hint_from_headers(headers) == "...12345678"

    def test_anthropic_x_api_key(self) -> None:
        headers = {"x-api-key": "sk-ant-abcdef12345678"}
        assert _key_hint_from_headers(headers) == "...12345678"

    def test_returns_none_when_no_headers(self) -> None:
        assert _key_hint_from_headers(None) is None
        assert _key_hint_from_headers({}) is None

    def test_returns_none_when_no_auth_headers(self) -> None:
        headers = {"content-type": "application/json"}
        assert _key_hint_from_headers(headers) is None

    def test_extract_from_request_includes_key_hint(self) -> None:
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        headers = {"authorization": "Bearer sk-test-abcdefgh12345678"}
        entry = extract_from_request("openai", body, tag=None, headers=headers)
        assert entry["api_key_hint"] == "...12345678"

    def test_extract_from_request_no_headers_gives_none(self) -> None:
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        entry = extract_from_request("openai", body, tag=None)
        assert entry["api_key_hint"] is None

    # --- OAuth / rotating token tests ---

    def test_oauth_jwt_bearer_token_returns_oauth(self) -> None:
        """OAuth JWT tokens (eyJ prefix) return '[OAuth]' when no tag provided."""
        headers = {"authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"}
        assert _key_hint_from_headers(headers) == "[OAuth]"

    def test_non_static_bearer_token_returns_oauth(self) -> None:
        """Any non-static bearer token returns '[OAuth]'."""
        headers = {"authorization": "Bearer some-opaque-session-token-abc123"}
        assert _key_hint_from_headers(headers) == "[OAuth]"

    def test_static_key_sk_ant_prefix(self) -> None:
        headers = {"authorization": "Bearer sk-ant-api03-abcdef12345678"}
        assert _key_hint_from_headers(headers) == "...12345678"

    def test_static_key_r8_prefix(self) -> None:
        headers = {"authorization": "Bearer r8_abcdefghij12345678"}
        assert _key_hint_from_headers(headers) == "...12345678"

    def test_static_key_gsk_prefix(self) -> None:
        headers = {"authorization": "Bearer gsk_abcdefgh12345678"}
        assert _key_hint_from_headers(headers) == "...12345678"

    def test_no_headers_returns_none(self) -> None:
        assert _key_hint_from_headers(None) is None
        assert _key_hint_from_headers({}) is None

    def test_no_auth_headers_returns_none(self) -> None:
        headers = {"content-type": "application/json"}
        assert _key_hint_from_headers(headers) is None

    def test_bearer_checked_before_x_api_key(self) -> None:
        """Authorization: Bearer is checked first; rotating token returns tag fallback."""
        headers = {
            "x-api-key": "sk-ant-abcdef12345678",
            "authorization": "Bearer eyJhbG.rotating.token",
        }
        # Bearer is checked first, token is not static → [tag] or None
        assert _key_hint_from_headers(headers, tag="myapp") == "[myapp]"
        assert _key_hint_from_headers(headers) == "[OAuth]"


# ---------------------------------------------------------------------------
# tag fallback for api_key_hint
# ---------------------------------------------------------------------------


def test_extract_from_request_no_auth_header_falls_back_to_tag():
    """When no auth headers are present, api_key_hint falls back to [tag]."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    entry = extract_from_request("openai", body, tag="myapp")
    assert entry["api_key_hint"] == "[myapp]"


def test_extract_from_request_no_auth_no_tag_stays_none():
    """When no auth headers and no tag, api_key_hint remains None."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    entry = extract_from_request("openai", body, tag=None)
    assert entry["api_key_hint"] is None


def test_extract_from_request_static_key_preferred_over_tag():
    """When a static Bearer key is present, it produces a hint."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    headers = {"authorization": "Bearer sk-test-abcdefgh12345678"}
    entry = extract_from_request("openai", body, tag="myapp", headers=headers)
    assert entry["api_key_hint"] == "...12345678"


def test_extract_from_request_oauth_bearer_returns_tag_fallback():
    """OAuth bearer token produces [tag] api_key_hint when tag is set."""
    body = {"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]}
    headers = {"authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig"}
    entry = extract_from_request("anthropic", body, tag="myapp", headers=headers)
    assert entry["api_key_hint"] == "[myapp]"
    assert entry["tags"] == "myapp"  # tag still available for report grouping


# ---------------------------------------------------------------------------
# OAuth vs static key distinction
# ---------------------------------------------------------------------------


def test_oauth_jwt_bearer_returns_tag_fallback():
    """OAuth/JWT bearer token returns [tag] fallback, not a key hint."""
    headers = {"authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"}
    assert _key_hint_from_headers(headers, tag="myapp") == "[myapp]"


def test_oauth_jwt_bearer_no_tag_returns_oauth():
    """OAuth/JWT bearer token with no tag returns '[OAuth]'."""
    headers = {"authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"}
    assert _key_hint_from_headers(headers) == "[OAuth]"


def test_static_sk_ant_bearer_returns_hint():
    """Static sk-ant- key in Authorization header returns key hint."""
    headers = {"authorization": "Bearer sk-ant-api03-abcdef12345678"}
    assert _key_hint_from_headers(headers) == "...12345678"


def test_static_sk_bearer_returns_hint():
    """Static sk- key in Authorization header returns key hint."""
    headers = {"authorization": "Bearer sk-proj-abcdefgh12345678"}
    assert _key_hint_from_headers(headers) == "...12345678"


def test_x_api_key_still_works():
    """x-api-key header still returns key hint when no Authorization header."""
    headers = {"x-api-key": "sk-ant-api03-abcdef12345678"}
    assert _key_hint_from_headers(headers) == "...12345678"


# ---------------------------------------------------------------------------
# thinking_tokens — output capture
# ---------------------------------------------------------------------------


def test_extract_from_request_thinking_tokens_none():
    """thinking_tokens starts as None in new entries."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    entry = extract_from_request("openai", body, tag=None)
    assert entry["thinking_tokens"] is None


# ---------------------------------------------------------------------------
# thinking_input_chars — input capture
# ---------------------------------------------------------------------------


def test_thinking_input_chars_zero_by_default():
    """thinking_input_chars is 0 when no thinking blocks are present."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    entry = extract_from_request("openai", body, tag=None)
    assert entry["thinking_input_chars"] == 0


def test_thinking_input_chars_anthropic_thinking_blocks():
    """Anthropic thinking blocks echoed back in messages count as thinking_input_chars."""
    thinking_text = "t" * 500
    body = {
        "model": "claude-opus-4-6",
        "messages": [
            {"role": "user", "content": "what is the meaning of life?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": thinking_text, "signature": "sig"},
                {"type": "text", "text": "42"},
            ]},
            {"role": "user", "content": "why?"},
        ],
    }
    entry = extract_from_request("anthropic", body, tag=None)
    assert entry["thinking_input_chars"] == 500


def test_thinking_input_chars_multiple_thinking_blocks():
    """Multiple thinking blocks across messages are summed."""
    body = {
        "model": "claude-opus-4-6",
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "a" * 100, "signature": "s1"},
                {"type": "text", "text": "ans1"},
            ]},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "b" * 200, "signature": "s2"},
                {"type": "text", "text": "ans2"},
            ]},
        ],
    }
    entry = extract_from_request("anthropic", body, tag=None)
    assert entry["thinking_input_chars"] == 300


def test_thinking_input_chars_redacted_thinking_not_counted():
    """redacted_thinking blocks are not counted (content unknown)."""
    body = {
        "model": "claude-opus-4-6",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "ok"},
            ]},
        ],
    }
    entry = extract_from_request("anthropic", body, tag=None)
    assert entry["thinking_input_chars"] == 0


def test_thinking_input_chars_openai_responses_api_reasoning():
    """OpenAI Responses API reasoning items in input[] contribute thinking_input_chars."""
    summary_text = "r" * 300
    body = {
        "model": "o3",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "think hard"}]},
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": summary_text}],
            },
        ],
    }
    entry = extract_from_request("openai", body, tag=None)
    assert entry["thinking_input_chars"] == 300


def test_thinking_input_chars_reasoning_not_counted_in_total_message_chars():
    """Reasoning item chars do not inflate total_message_chars (kept separate)."""
    summary_text = "r" * 300
    user_text = "think hard"
    body = {
        "model": "o3",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": summary_text}],
            },
        ],
    }
    entry = extract_from_request("openai", body, tag=None)
    # total_message_chars should NOT include the reasoning summary chars
    assert entry["thinking_input_chars"] == 300
    assert entry["total_message_chars"] == len(user_text)


# ---------------------------------------------------------------------------
# _derive_program
# ---------------------------------------------------------------------------


class TestDeriveProgram:
    def test_python_m_module(self) -> None:
        assert _derive_program("python3 -m myapp serve", None) == "myapp"

    def test_python_m_dotted_module(self) -> None:
        assert _derive_program("python3 -m myapp.core.agent", None) == "myapp"

    def test_python_script(self) -> None:
        assert _derive_program("python3 benchmark_proxy.py", None) == "benchmark_proxy"

    def test_node_js_script(self) -> None:
        assert _derive_program("node project/index.js", None) == "index"

    def test_cmdline_none_tag_fallback(self) -> None:
        assert _derive_program(None, "myapp") == "myapp"

    def test_cmdline_none_tag_none(self) -> None:
        assert _derive_program(None, None) is None

    def test_extract_from_request_sets_program(self) -> None:
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        entry = extract_from_request("openai", body, tag=None, cmdline="python3 -m myapp serve")
        assert entry["program"] == "myapp"
