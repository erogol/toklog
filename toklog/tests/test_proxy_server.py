"""Integration tests for toklog/proxy/server.py using Starlette TestClient."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from toklog.proxy.server import ProcessInfo, app


# ---------------------------------------------------------------------------
# Helpers — fake httpx responses
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal httpx.Response-like object for mocking client.send()."""

    def __init__(
        self,
        status_code: int,
        content: bytes,
        headers: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self._content = content
        self.headers = httpx.Headers(headers or {"content-type": "application/json"})

    async def aread(self) -> bytes:
        return self._content

    async def aclose(self) -> None:
        pass

    async def aiter_bytes(self, chunk_size: int | None = None):
        yield self._content


class _FakeSendClient:
    """Fake httpx.AsyncClient that returns a pre-configured response."""

    def __init__(self, fake_resp: _FakeResponse) -> None:
        self._fake_resp = fake_resp

    def build_request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
        return MagicMock(spec=httpx.Request)

    async def send(self, request: Any, *, stream: bool = False) -> _FakeResponse:
        return self._fake_resp


@pytest.fixture(autouse=True)
def _reset_proxy_client():
    """Reset the module-level _client between tests to avoid state leakage."""
    import toklog.proxy.server as srv

    original = srv._client
    srv._client = None
    yield
    srv._client = original


@pytest.fixture(autouse=True)
def _reset_tag_cache():
    """Clear the port→tag cache between tests to prevent stale-port collisions."""
    import toklog.proxy.server as srv

    srv._tag_cache.clear()
    yield
    srv._tag_cache.clear()


@pytest.fixture(autouse=True)
def _reset_skip_patterns_cache():
    """Clear _parse_skip_patterns lru_cache so monkeypatch env changes take effect."""
    from toklog.proxy.server import _parse_skip_patterns

    _parse_skip_patterns.cache_clear()
    yield
    _parse_skip_patterns.cache_clear()


@pytest.fixture(autouse=True)
def _reset_skip_proc_cache():
    """Reset _skip_proc_cache between tests so file changes are picked up."""
    import toklog.proxy.server as srv

    srv._skip_proc_cache = None
    yield
    srv._skip_proc_cache = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_unknown_prefix_returns_404():
    client = TestClient(app)
    resp = client.post("/unknown/v1/models", json={})
    assert resp.status_code == 404


def test_missing_remainder_returns_404():
    """Path /openai (no remainder after prefix) should 404."""
    client = TestClient(app)
    resp = client.post("/openai", json={})
    assert resp.status_code == 404


def test_openai_route_logs_and_returns():
    """Non-streaming OpenAI request is forwarded, logged, response returned."""
    fake_body = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)

    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        resp = tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer sk-test"},
        )

    assert resp.status_code == 200
    assert len(logged) == 1
    e = logged[0]
    assert e["provider"] == "openai"
    assert e["model"] == "gpt-4o"
    assert e["input_tokens"] == 10
    assert e["output_tokens"] == 5
    assert e["instrumentation"] == "proxy"
    assert e["duration_ms"] is not None


def test_anthropic_route_logs_correctly():
    """Anthropic endpoint is routed and logged with correct provider."""
    fake_body = {
        "id": "msg_xyz",
        "type": "message",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "Hello"}],
        "usage": {"input_tokens": 15, "output_tokens": 8},
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)

    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        resp = tc.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
            headers={"x-api-key": "sk-ant-test"},
        )

    assert resp.status_code == 200
    assert len(logged) == 1
    e = logged[0]
    assert e["provider"] == "anthropic"
    assert e["input_tokens"] == 15
    assert e["output_tokens"] == 8


def test_x_tb_skip_suppresses_logging():
    """X-TB-Skip: true header must forward the request but write no log entry."""
    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)

    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": []},
            headers={"x-tb-skip": "true"},
        )

    assert len(logged) == 0


def test_streaming_response_passes_through_and_logs():
    """Streaming SSE response is passed through, usage captured from final event."""
    final_event = {
        "object": "chat.completion.chunk",
        "id": "chatcmpl-stream",
        "model": "gpt-4o",
        "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    }
    sse_content = b"data: " + json.dumps(final_event).encode() + b"\n\ndata: [DONE]\n\n"
    fake_resp = _FakeResponse(
        200,
        sse_content,
        headers={"content-type": "text/event-stream"},
    )
    fake_client = _FakeSendClient(fake_resp)

    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        resp = tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [], "stream": True},
        )

    assert resp.status_code == 200
    assert len(logged) == 1
    e = logged[0]
    assert e["streaming"] is True
    assert e["input_tokens"] == 20
    assert e["output_tokens"] == 8


def test_process_tag_auto_applied():
    """When no X-TB-Tag header, tag is derived from connecting process via /proc lookup."""
    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append), \
         patch("toklog.proxy.server._lookup_process_info",
               return_value=ProcessInfo(tag="myapp", cmdline="python3 -m myapp")):
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert len(logged) == 1
    assert logged[0]["tags"] == "myapp"


def test_x_tb_tag_header_overrides_process_lookup():
    """Explicit X-TB-Tag header takes priority over auto process lookup."""
    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append), \
         patch("toklog.proxy.server._lookup_process_info",
               return_value=ProcessInfo(tag="myapp", cmdline="python3 -m myapp")):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": []},
            headers={"x-tb-tag": "custom-tag"},
        )

    assert len(logged) == 1
    assert logged[0]["tags"] == "custom-tag"


def test_process_tag_lookup_failure_does_not_break():
    """If _lookup_process_tag returns None, request still logs (untagged)."""
    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append), \
         patch("toklog.proxy.server._lookup_process_info",
               return_value=ProcessInfo(tag=None, cmdline=None)):
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert len(logged) == 1
    assert logged[0]["tags"] is None


def test_skip_models_suppresses_logging(monkeypatch):
    """TOKLOG_SKIP_MODELS matching the request model must suppress logging."""
    monkeypatch.setenv("TOKLOG_SKIP_MODELS", "gpt-4o,haiku")

    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append), \
         patch("toklog.proxy.server._lookup_process_info",
               return_value=ProcessInfo(tag=None, cmdline=None)):
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert len(logged) == 0


def test_skip_models_does_not_suppress_non_matching(monkeypatch):
    """TOKLOG_SKIP_MODELS must not suppress a model that doesn't match."""
    monkeypatch.setenv("TOKLOG_SKIP_MODELS", "haiku")

    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append), \
         patch("toklog.proxy.server._lookup_process_info",
               return_value=ProcessInfo(tag=None, cmdline=None)):
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert len(logged) == 1


def test_resolve_process_tag_real_socket():
    """_resolve_process_tag returns a ProcessInfo for the current process using a real TCP socket."""
    import socket as _socket
    from toklog.proxy.server import _resolve_process_tag

    server_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    try:
        client_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        client_sock.connect(("127.0.0.1", server_sock.getsockname()[1]))
        accepted, _ = server_sock.accept()
        try:
            client_port = client_sock.getsockname()[1]
            info = _resolve_process_tag(client_port)
            # Tag must resolve to something (current process is identifiable)
            assert info.tag is not None
            assert isinstance(info.tag, str)
            assert len(info.tag) > 0
            # Cmdline must also be populated
            assert info.cmdline is not None
            assert isinstance(info.cmdline, str)
            assert len(info.cmdline) > 0
        finally:
            accepted.close()
            client_sock.close()
    finally:
        server_sock.close()


def test_toklog_disabled_env_suppresses_logging(monkeypatch):
    """TOKLOG_DISABLED=1 must not write any log entries."""
    monkeypatch.setenv("TOKLOG_DISABLED", "1")

    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)

    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert len(logged) == 0


def test_no_log_body_param_suppresses_logging():
    """no-log: true in request body must suppress logging (LiteLLM-compatible)."""
    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [], "no-log": True},
        )

    assert len(logged) == 0


def test_no_log_body_param_stripped_before_forwarding():
    """no-log key must be removed from the body sent to the upstream API."""
    received_bodies: list[bytes] = []

    class _CaptureSendClient:
        def build_request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
            received_bodies.append(kwargs.get("content", b""))
            return MagicMock(spec=httpx.Request)

        async def send(self, request: Any, *, stream: bool = False) -> _FakeResponse:
            return _FakeResponse(200, b"{}")

    with patch("toklog.proxy.server._get_client", return_value=_CaptureSendClient()), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [], "no-log": True},
        )

    assert received_bodies
    forwarded = json.loads(received_bodies[0])
    assert "no-log" not in forwarded


def test_skip_process_suppresses_logging(tmp_path):
    """Process matching a skip pattern must not be logged."""
    import toklog.proxy.server as srv

    skip_file = tmp_path / "skip_processes"
    skip_file.write_text("pytest\n")

    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    # Patch load_config to return empty skip_processes so it falls back to skip_processes file
    def mock_load_config(**kwargs):
        return {"skip_processes": []}

    with patch("toklog.proxy.server._SKIP_PROCESSES_FILE", skip_file), \
         patch("toklog.config.load_config", mock_load_config), \
         patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append), \
         patch("toklog.proxy.server._lookup_process_info",
               return_value=ProcessInfo(tag="myapp", cmdline="python3 -m pytest tests/")):
        # Reset cache to force reloading
        srv._skip_proc_cache = None
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert len(logged) == 0


def test_skip_process_no_match_logs_normally(tmp_path):
    """Process not matching any skip pattern is logged as normal."""
    skip_file = tmp_path / "skip_processes"
    skip_file.write_text("pytest\n")

    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._SKIP_PROCESSES_FILE", skip_file), \
         patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append), \
         patch("toklog.proxy.server._lookup_process_info",
               return_value=ProcessInfo(tag="myapp", cmdline="python3 -m myapp serve")):
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert len(logged) == 1


def test_stream_options_injected_for_chat_completions():
    """Streaming chat completions request gets stream_options.include_usage injected."""
    received_bodies: list[bytes] = []

    class _CaptureSendClient:
        def build_request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
            received_bodies.append(kwargs.get("content", b""))
            return MagicMock(spec=httpx.Request)

        async def send(self, request: Any, *, stream: bool = False) -> _FakeResponse:
            sse = b"data: [DONE]\n\n"
            return _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})

    with patch("toklog.proxy.server._get_client", return_value=_CaptureSendClient()), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        tc.post(
            "/openai/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [], "stream": True},
        )

    assert received_bodies
    forwarded = json.loads(received_bodies[0])
    assert forwarded.get("stream_options", {}).get("include_usage") is True


def test_stream_options_not_injected_for_responses_api():
    """Streaming Responses API request must NOT get stream_options.include_usage injected."""
    received_bodies: list[bytes] = []

    class _CaptureSendClient:
        def build_request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
            received_bodies.append(kwargs.get("content", b""))
            return MagicMock(spec=httpx.Request)

        async def send(self, request: Any, *, stream: bool = False) -> _FakeResponse:
            completed_event = {
                "type": "response.completed",
                "response": {
                    "id": "resp_abc",
                    "model": "gpt-5.4",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            }
            sse = b"data: " + json.dumps(completed_event).encode() + b"\n\ndata: [DONE]\n\n"
            return _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})

    with patch("toklog.proxy.server._get_client", return_value=_CaptureSendClient()), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        tc.post(
            "/openai/v1/responses",
            json={"model": "gpt-5.4", "input": [], "stream": True},
        )

    assert received_bodies
    forwarded = json.loads(received_bodies[0])
    # stream_options must not have been injected (or must not have include_usage)
    stream_opts = forwarded.get("stream_options") or {}
    assert "include_usage" not in stream_opts


def test_responses_api_stream_logs_usage():
    """Streaming Responses API response.completed event is parsed for usage."""
    completed_event = {
        "type": "response.completed",
        "response": {
            "id": "resp_abc",
            "model": "gpt-5.4",
            "usage": {
                "input_tokens": 30,
                "output_tokens": 12,
                "input_tokens_details": {"cached_tokens": 4},
            },
        },
    }
    sse_content = b"data: " + json.dumps(completed_event).encode() + b"\n\ndata: [DONE]\n\n"
    fake_resp = _FakeResponse(
        200,
        sse_content,
        headers={"content-type": "text/event-stream"},
    )
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/v1/responses",
            json={"model": "gpt-5.4", "input": [], "stream": True},
        )

    assert len(logged) == 1
    e = logged[0]
    assert e["input_tokens"] == 30
    assert e["output_tokens"] == 12
    assert e["cache_read_tokens"] == 4
    assert e["request_id"] == "resp_abc"
    assert e["model"] == "gpt-5.4"


def test_toklog_skip_process_tag_skips_proc_scan(monkeypatch):
    """TOKLOG_SKIP_PROCESS_TAG=1 must not call _lookup_process_info."""
    monkeypatch.setenv("TOKLOG_SKIP_PROCESS_TAG", "1")

    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry"), \
         patch("toklog.proxy.server._lookup_process_info") as mock_lookup:
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    mock_lookup.assert_not_called()


def test_toklog_disabled_skips_proc_scan(monkeypatch):
    """TOKLOG_DISABLED=1 must not call _lookup_process_info (wasteful scan)."""
    monkeypatch.setenv("TOKLOG_DISABLED", "1")

    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry"), \
         patch("toklog.proxy.server._lookup_process_info") as mock_lookup:
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    mock_lookup.assert_not_called()


def test_streaming_upstream_error_flagged():
    """Streaming response with HTTP 401 must flag entry as error."""
    error_body = b'{"error": {"message": "Incorrect API key", "type": "authentication_error"}}'
    fake_resp = _FakeResponse(
        401,
        error_body,
        headers={"content-type": "application/json"},
    )
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-5.4", "messages": [], "stream": True},
        )

    assert len(logged) == 1
    e = logged[0]
    assert e["error"] is True
    assert e["error_type"] == "http_401"


def test_streaming_upstream_error_still_streams():
    """Error response body must still be streamed back to the client."""
    error_body = b'{"error": {"message": "Unauthorized", "type": "auth_error"}}'
    fake_resp = _FakeResponse(
        401,
        error_body,
        headers={"content-type": "application/json"},
    )
    fake_client = _FakeSendClient(fake_resp)

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        resp = tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-5.4", "messages": [], "stream": True},
        )

    assert resp.status_code == 401
    assert b"Unauthorized" in resp.content


def test_streaming_success_no_error():
    """Streaming 200 response must not flag entry as error."""
    sse_content = b"data: [DONE]\n\n"
    fake_resp = _FakeResponse(
        200,
        sse_content,
        headers={"content-type": "text/event-stream"},
    )
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [], "stream": True},
        )

    assert len(logged) == 1
    e = logged[0]
    assert e["error"] is False
    assert e["error_type"] is None


def test_non_streaming_upstream_error():
    """Non-streaming response with HTTP 500 must flag entry as error."""
    error_body = b'{"error": {"message": "Internal server error", "type": "server_error"}}'
    fake_resp = _FakeResponse(
        500,
        error_body,
        headers={"content-type": "application/json"},
    )
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        resp = tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert resp.status_code == 500
    assert len(logged) == 1
    e = logged[0]
    assert e["error"] is True
    assert e["error_type"] == "http_500"


def test_skip_process_file_missing_logs_normally():
    """When skip_processes file does not exist, all processes are logged."""
    import toklog.proxy.server as srv
    from pathlib import Path

    nonexistent = Path("/tmp/toklog_test_skip_processes_does_not_exist")
    nonexistent.unlink(missing_ok=True)

    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._SKIP_PROCESSES_FILE", nonexistent), \
         patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append), \
         patch("toklog.proxy.server._lookup_process_info",
               return_value=ProcessInfo(tag="myapp", cmdline="python3 app.py")):
        tc = TestClient(app)
        tc.post("/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert len(logged) == 1


# ---------------------------------------------------------------------------
# OpenAI adapter integration tests
# ---------------------------------------------------------------------------


def test_openai_chat_completions_non_streaming_extracts_usage():
    """POST to /openai/chat/completions: adapter extracts full usage from response."""
    fake_body = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 5},
        },
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        resp = tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer sk-test"},
        )

    assert resp.status_code == 200
    assert len(logged) == 1
    e = logged[0]
    assert e["input_tokens"] == 20
    assert e["output_tokens"] == 10
    assert e["total_tokens"] == 30
    assert e["cache_read_tokens"] == 5
    assert e["thinking_tokens"] is None
    assert e["usage_source"] == "provider_response"
    assert e["usage_status"] == "exact"
    assert e["endpoint_family"] == "openai.chat_completions"


def test_openai_responses_api_non_streaming_extracts_usage():
    """POST to /openai/v1/responses: adapter extracts usage and counts tool calls from output items."""
    fake_body = {
        "id": "resp_abc",
        "object": "response",
        "model": "gpt-5.4",
        "output": [
            {"type": "message", "content": [{"type": "text", "text": "Hi"}]},
            {"type": "function_call", "name": "get_weather"},
        ],
        "usage": {
            "input_tokens": 15,
            "output_tokens": 8,
            "total_tokens": 23,
        },
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/v1/responses",
            json={"model": "gpt-5.4", "input": [{"role": "user", "content": "Hi"}]},
        )

    assert len(logged) == 1
    e = logged[0]
    assert e["endpoint_family"] == "openai.responses"
    assert e["input_tokens"] == 15
    assert e["output_tokens"] == 8
    assert e["total_tokens"] == 23
    assert e["usage_source"] == "provider_response"
    assert e["usage_status"] == "exact"
    assert e["tool_calls_made"] == 1


def test_openai_chat_completions_streaming_extracts_usage():
    """Streaming chat completions: usage extracted from the final usage chunk."""
    first_chunk = {
        "object": "chat.completion.chunk",
        "id": "chatcmpl-stream",
        "model": "gpt-4o",
        "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}],
    }
    usage_chunk = {
        "object": "chat.completion.chunk",
        "id": "chatcmpl-stream",
        "model": "gpt-4o",
        "choices": [],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 6,
            "total_tokens": 18,
        },
    }
    sse = (
        b"data: " + json.dumps(first_chunk).encode() + b"\n\n"
        + b"data: " + json.dumps(usage_chunk).encode() + b"\n\n"
        + b"data: [DONE]\n\n"
    )
    fake_resp = _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [], "stream": True},
        )

    assert len(logged) == 1
    e = logged[0]
    assert e["input_tokens"] == 12
    assert e["output_tokens"] == 6
    assert e["total_tokens"] == 18
    assert e["usage_source"] == "provider_stream_final"


def test_openai_streaming_stop_reason_captured():
    """Streaming chunk with finish_reason sets stop_reason in the logged entry."""
    chunk = {
        "object": "chat.completion.chunk",
        "id": "chatcmpl-stop",
        "model": "gpt-4o",
        "choices": [{"delta": {"content": "Done"}, "finish_reason": "stop"}],
    }
    sse = b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"
    fake_resp = _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [], "stream": True},
        )

    assert len(logged) == 1
    assert logged[0]["stop_reason"] == "stop"


# ---------------------------------------------------------------------------
# Anthropic adapter integration tests
# ---------------------------------------------------------------------------


def test_anthropic_non_streaming_extracts_usage():
    """POST to /anthropic/v1/messages: adapter extracts full usage including cache tokens."""
    fake_body = {
        "id": "msg_xyz",
        "type": "message",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "Hello"}],
        "usage": {
            "input_tokens": 15,
            "output_tokens": 8,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2,
        },
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        resp = tc.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
            headers={"x-api-key": "sk-ant-test"},
        )

    assert resp.status_code == 200
    assert len(logged) == 1
    e = logged[0]
    assert e["input_tokens"] == 15
    assert e["output_tokens"] == 8
    assert e["cache_read_tokens"] == 4
    assert e["cache_creation_tokens"] == 2
    assert e["endpoint_family"] == "anthropic.messages"
    assert e["usage_source"] == "provider_response"
    assert e["usage_status"] == "exact"


def test_anthropic_streaming_extracts_usage():
    """Anthropic SSE stream: input tokens from message_start, output from message_delta."""
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_stream",
            "model": "claude-opus-4-6",
            "usage": {
                "input_tokens": 20,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 0,
            },
        },
    }
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 12},
    }
    message_stop = {"type": "message_stop"}
    sse = (
        b"event: message_start\ndata: " + json.dumps(message_start).encode() + b"\n\n"
        + b"event: message_delta\ndata: " + json.dumps(message_delta).encode() + b"\n\n"
        + b"event: message_stop\ndata: " + json.dumps(message_stop).encode() + b"\n\n"
    )
    fake_resp = _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
                "stream": True,
            },
            headers={"x-api-key": "sk-ant-test"},
        )

    assert len(logged) == 1
    e = logged[0]
    assert e["input_tokens"] == 20
    assert e["output_tokens"] == 12
    assert e["cache_read_tokens"] == 5
    assert e["endpoint_family"] == "anthropic.messages"


def test_anthropic_streaming_thinking_tokens_estimated():
    """thinking_delta chars // 4 are reported as thinking_tokens in the log entry."""
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_think",
            "model": "claude-opus-4-6",
            "usage": {"input_tokens": 10},
        },
    }
    content_block_delta = {
        "type": "content_block_delta",
        "delta": {
            "type": "thinking_delta",
            "thinking": "a" * 40,  # 40 chars -> 40 // 4 = 10 thinking tokens
        },
    }
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 5},
    }
    sse = (
        b"event: message_start\ndata: " + json.dumps(message_start).encode() + b"\n\n"
        + b"event: content_block_delta\ndata: " + json.dumps(content_block_delta).encode() + b"\n\n"
        + b"event: message_delta\ndata: " + json.dumps(message_delta).encode() + b"\n\n"
    )
    fake_resp = _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "messages": [],
                "max_tokens": 100,
                "stream": True,
            },
        )

    assert len(logged) == 1
    assert logged[0]["thinking_tokens"] == 10  # 40 chars // 4


# ---------------------------------------------------------------------------
# Gemini adapter integration tests
# ---------------------------------------------------------------------------


def test_gemini_non_streaming_extracts_usage():
    """POST to /gemini/.../generateContent: adapter extracts usageMetadata fields."""
    fake_body = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 10,
            "totalTokenCount": 35,
            "thoughtsTokenCount": 3,
        },
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        resp = tc.post(
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
        )

    assert resp.status_code == 200
    assert len(logged) == 1
    e = logged[0]
    assert e["input_tokens"] == 25
    assert e["output_tokens"] == 10
    assert e["total_tokens"] == 35
    assert e["thinking_tokens"] == 3
    assert e["endpoint_family"] == "gemini.generate_content"
    assert e["usage_source"] == "provider_response"
    assert e["usage_status"] == "exact"


def test_gemini_streaming_extracts_usage():
    """Gemini SSE stream: usage extracted from the final chunk's usageMetadata."""
    chunk1 = {
        "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
    }
    chunk2 = {
        "candidates": [
            {
                "content": {"parts": [{"text": " world"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 5,
            "totalTokenCount": 20,
        },
    }
    sse = (
        b"data: " + json.dumps(chunk1).encode() + b"\n\n"
        + b"data: " + json.dumps(chunk2).encode() + b"\n\n"
    )
    fake_resp = _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
        )

    assert len(logged) == 1
    e = logged[0]
    assert e["input_tokens"] == 15
    assert e["output_tokens"] == 5
    assert e["total_tokens"] == 20
    assert e["usage_source"] == "provider_stream_final"
    assert e["endpoint_family"] == "gemini.generate_content"


def test_gemini_model_extracted_from_url():
    """Gemini: entry model is set from the URL path, not from the request body."""
    # No modelVersion in response — adapter won't overwrite the URL-extracted model
    fake_body = {
        "candidates": [],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
        },
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": []},
        )

    assert len(logged) == 1
    assert logged[0]["model"] == "gemini-2.5-flash"


def test_gemini_streaming_detected_from_url():
    """streamGenerateContent in URL triggers streaming path even without stream:true in body."""
    chunk = {
        "candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3, "totalTokenCount": 11},
    }
    sse = b"data: " + json.dumps(chunk).encode() + b"\n\n"
    fake_resp = _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        # Intentionally no "stream": True in body — streaming inferred from URL
        tc.post(
            "/gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            json={"contents": []},
        )

    assert len(logged) == 1
    e = logged[0]
    # usage_source="provider_stream_final" proves the streaming handler path was used
    assert e["usage_source"] == "provider_stream_final"
    assert e["input_tokens"] == 8
    assert e["output_tokens"] == 3


def test_gemini_valid_request_passes_through():
    """Gemini provider is in _UPSTREAM — valid requests must not return 404."""
    fake_resp = _FakeResponse(200, b"{}")
    fake_client = _FakeSendClient(fake_resp)

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        resp = tc.post(
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": []},
        )

    assert resp.status_code == 200


def test_gemini_streaming_entry_has_streaming_true():
    """Gemini streaming request must set entry['streaming'] = True in the log entry."""
    chunk = {
        "candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3, "totalTokenCount": 11},
    }
    sse = b"data: " + json.dumps(chunk).encode() + b"\n\n"
    fake_resp = _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            json={"contents": []},
        )

    assert len(logged) == 1
    assert logged[0]["streaming"] is True


def test_gemini_streaming_injects_alt_sse():
    """Gemini streaming must append alt=sse to the upstream URL."""
    captured_urls: list[str] = []

    class _CaptureUrlClient:
        def build_request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            return MagicMock(spec=httpx.Request)

        async def send(self, request: Any, *, stream: bool = False) -> _FakeResponse:
            sse = b"data: [DONE]\n\n"
            return _FakeResponse(200, sse, headers={"content-type": "text/event-stream"})

    with patch("toklog.proxy.server._get_client", return_value=_CaptureUrlClient()), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        tc.post(
            "/gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            json={"contents": []},
        )

    assert captured_urls, "build_request was not called"
    assert "alt=sse" in captured_urls[0], f"alt=sse missing from URL: {captured_urls[0]}"


def test_openai_non_streaming_stop_reason():
    """Non-streaming OpenAI chat completion: finish_reason must be logged as stop_reason."""
    fake_body = {
        "id": "chatcmpl-stop",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert len(logged) == 1
    assert logged[0]["stop_reason"] == "stop"


def test_anthropic_non_streaming_stop_reason():
    """Non-streaming Anthropic messages: stop_reason must be logged in the entry."""
    fake_body = {
        "id": "msg_xyz",
        "type": "message",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 15, "output_tokens": 8},
    }
    fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
    fake_client = _FakeSendClient(fake_resp)
    logged: list[dict] = []

    with patch("toklog.proxy.server._get_client", return_value=fake_client), \
         patch("toklog.proxy.server.log_entry", side_effect=logged.append):
        tc = TestClient(app)
        tc.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
            headers={"x-api-key": "sk-ant-test"},
        )

    assert len(logged) == 1
    assert logged[0]["stop_reason"] == "end_turn"


# --- OpenAI /v1 prefix deduplication ---


def test_openai_v1_prefix_dedup():
    """Client path /openai/v1/chat/completions must not produce double /v1/v1 upstream."""
    captured_urls: list[str] = []

    class _CaptureUrlClient:
        def build_request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            return MagicMock(spec=httpx.Request)

        async def send(self, request: Any, *, stream: bool = False) -> _FakeResponse:
            return _FakeResponse(200, b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}')

    with patch("toklog.proxy.server._get_client", return_value=_CaptureUrlClient()), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        tc.post(
            "/openai/v1/chat/completions",
            json={"model": "gpt-4o", "messages": []},
            headers={"Authorization": "Bearer test"},
        )

    assert captured_urls, "build_request was not called"
    assert "/v1/v1/" not in captured_urls[0], f"Double /v1 in URL: {captured_urls[0]}"
    assert "api.openai.com/v1/chat/completions" in captured_urls[0]


def test_openai_without_v1_prefix_still_works():
    """Client path /openai/chat/completions (SDK default) still routes correctly."""
    captured_urls: list[str] = []

    class _CaptureUrlClient:
        def build_request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            return MagicMock(spec=httpx.Request)

        async def send(self, request: Any, *, stream: bool = False) -> _FakeResponse:
            return _FakeResponse(200, b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}')

    with patch("toklog.proxy.server._get_client", return_value=_CaptureUrlClient()), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        tc.post(
            "/openai/chat/completions",
            json={"model": "gpt-4o", "messages": []},
            headers={"Authorization": "Bearer test"},
        )

    assert captured_urls, "build_request was not called"
    assert "api.openai.com/v1/chat/completions" in captured_urls[0]


def test_openai_v1_responses_dedup():
    """Client path /openai/v1/responses must not produce double /v1/v1."""
    captured_urls: list[str] = []

    class _CaptureUrlClient:
        def build_request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            return MagicMock(spec=httpx.Request)

        async def send(self, request: Any, *, stream: bool = False) -> _FakeResponse:
            return _FakeResponse(200, b'{"id":"resp_1","model":"gpt-4o","usage":{"input_tokens":1,"output_tokens":1}}')

    with patch("toklog.proxy.server._get_client", return_value=_CaptureUrlClient()), \
         patch("toklog.proxy.server.log_entry"):
        tc = TestClient(app)
        tc.post(
            "/openai/v1/responses",
            json={"model": "gpt-4o", "input": []},
            headers={"Authorization": "Bearer test"},
        )

    assert captured_urls, "build_request was not called"
    assert "/v1/v1/" not in captured_urls[0], f"Double /v1 in URL: {captured_urls[0]}"
    assert "api.openai.com/v1/responses" in captured_urls[0]


# ---------------------------------------------------------------------------
# Budget kill switch integration
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    """Tests for budget check (429 rejection) and cost recording."""

    @pytest.fixture(autouse=True)
    def _reset_budget(self):
        """Reset budget tracker between tests so budget state doesn't leak."""
        import toklog.proxy.budget as budget_mod
        budget_mod.configure(limit_usd=None)
        yield
        budget_mod.configure(limit_usd=None)

    def test_budget_exceeded_returns_429(self, tmp_path):
        """When budget is exceeded, proxy returns 429 without forwarding upstream."""
        import toklog.proxy.budget as budget_mod

        state_file = tmp_path / "budget_state.json"
        budget_mod.configure(limit_usd=1.0, state_path=state_file)
        budget_mod.record(2.0)  # over budget

        logged: list[dict] = []

        with patch("toklog.proxy.server.log_entry", side_effect=logged.append):
            tc = TestClient(app)
            resp = tc.post(
                "/openai/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer sk-test"},
            )

        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["type"] == "budget_exceeded"
        assert body["error"]["budget_limit"] == 1.0
        assert body["error"]["daily_spend"] == 2.0
        # Should log the rejection
        assert len(logged) == 1
        assert logged[0]["budget_rejected"] is True
        assert logged[0]["provider"] == "openai"

    def test_budget_exceeded_does_not_forward_upstream(self, tmp_path):
        """429 budget rejection must NOT contact the upstream provider."""
        import toklog.proxy.budget as budget_mod

        budget_mod.configure(limit_usd=0.01, state_path=tmp_path / "state.json")
        budget_mod.record(1.0)

        send_called = False

        class _SpyClient:
            def build_request(self, *a, **kw):
                return MagicMock(spec=httpx.Request)

            async def send(self, *a, **kw):
                nonlocal send_called
                send_called = True
                return _FakeResponse(200, b'{}')

        with patch("toklog.proxy.server._get_client", return_value=_SpyClient()), \
             patch("toklog.proxy.server.log_entry"):
            tc = TestClient(app)
            resp = tc.post(
                "/anthropic/v1/messages",
                json={"model": "claude-sonnet-4-5", "messages": []},
                headers={"x-api-key": "sk-ant-api-test"},
            )

        assert resp.status_code == 429
        assert not send_called, "Upstream was contacted despite budget exceeded"

    def test_under_budget_forwards_normally(self, tmp_path):
        """When under budget, requests forward normally."""
        import toklog.proxy.budget as budget_mod

        budget_mod.configure(limit_usd=100.0, state_path=tmp_path / "state.json")

        fake_body = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
        fake_client = _FakeSendClient(fake_resp)

        with patch("toklog.proxy.server._get_client", return_value=fake_client), \
             patch("toklog.proxy.server.log_entry"):
            tc = TestClient(app)
            resp = tc.post(
                "/openai/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer sk-test"},
            )

        assert resp.status_code == 200

    def test_no_budget_set_forwards_normally(self):
        """With no budget configured, all requests pass through."""
        fake_body = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
        fake_client = _FakeSendClient(fake_resp)

        with patch("toklog.proxy.server._get_client", return_value=fake_client), \
             patch("toklog.proxy.server.log_entry"):
            tc = TestClient(app)
            resp = tc.post(
                "/openai/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer sk-test"},
            )

        assert resp.status_code == 200

    def test_cost_recorded_after_non_streaming_response(self, tmp_path):
        """After a non-streaming response, cost is recorded in the budget tracker."""
        import toklog.proxy.budget as budget_mod

        state_file = tmp_path / "budget_state.json"
        budget_mod.configure(limit_usd=100.0, state_path=state_file)

        fake_body = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }
        fake_resp = _FakeResponse(200, json.dumps(fake_body).encode())
        fake_client = _FakeSendClient(fake_resp)

        with patch("toklog.proxy.server._get_client", return_value=fake_client), \
             patch("toklog.proxy.server.log_entry"):
            tc = TestClient(app)
            tc.post(
                "/openai/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer sk-test"},
            )

        s = budget_mod.status()
        assert s["daily_spend"] > 0, "Cost was not recorded in budget tracker"

    def test_429_body_provider_agnostic(self, tmp_path):
        """Budget 429 works for all providers (test with Gemini path)."""
        import toklog.proxy.budget as budget_mod

        budget_mod.configure(limit_usd=0.01, state_path=tmp_path / "state.json")
        budget_mod.record(1.0)

        with patch("toklog.proxy.server.log_entry"):
            tc = TestClient(app)
            resp = tc.post(
                "/gemini/v1beta/models/gemini-pro:generateContent",
                json={"contents": [{"parts": [{"text": "Hi"}]}]},
            )

        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["type"] == "budget_exceeded"

    def test_get_request_not_blocked_by_budget(self, tmp_path):
        """GET requests (e.g. /openai/models) pass through even when over budget."""
        import toklog.proxy.budget as budget_mod

        budget_mod.configure(limit_usd=0.01, state_path=tmp_path / "state.json")
        budget_mod.record(1.0)  # over budget

        fake_resp = _FakeResponse(
            200, json.dumps({"data": [{"id": "gpt-4o"}]}).encode()
        )
        fake_client = _FakeSendClient(fake_resp)

        with patch("toklog.proxy.server._get_client", return_value=fake_client), \
             patch("toklog.proxy.server.log_entry"):
            tc = TestClient(app)
            resp = tc.get(
                "/openai/models",
                headers={"Authorization": "Bearer sk-test"},
            )

        assert resp.status_code == 200, "GET request was blocked by budget"

    def test_error_response_does_not_crash_budget(self, tmp_path):
        """4xx upstream responses don't crash the budget recording code path."""
        import toklog.proxy.budget as budget_mod

        budget_mod.configure(limit_usd=100.0, state_path=tmp_path / "state.json")

        fake_resp = _FakeResponse(400, json.dumps({"error": {"message": "bad"}}).encode())
        fake_client = _FakeSendClient(fake_resp)

        with patch("toklog.proxy.server._get_client", return_value=fake_client), \
             patch("toklog.proxy.server.log_entry"):
            tc = TestClient(app)
            resp = tc.post(
                "/openai/chat/completions",
                json={"model": "gpt-4o", "messages": []},
                headers={"Authorization": "Bearer sk-test"},
            )

        assert resp.status_code == 400
        s = budget_mod.status()
        assert isinstance(s["daily_spend"], float)  # didn't crash
