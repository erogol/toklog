"""Starlette ASGI proxy — routes /openai/*, /anthropic/*, and /gemini/* to upstream LLM APIs."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import time
from pathlib import Path
from typing import AsyncIterator, NamedTuple

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from toklog.logger import log_entry_async as log_entry
from toklog.adapters.sse import SSEStreamBuffer
from toklog.adapters import openai as openai_adapter
from toklog.adapters import anthropic as anthropic_adapter
from toklog.adapters import gemini as gemini_adapter
from toklog.proxy.extractor import extract_from_request


class ProcessInfo(NamedTuple):
    tag: str | None
    cmdline: str | None  # space-joined /proc/{pid}/cmdline, first 256 chars

# Upstream base URLs — must match the SDK defaults for correct path appending
_UPSTREAM: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}

# Path prefixes already included in _UPSTREAM base URLs.
# If a client path starts with one of these, strip it to avoid duplication.
# e.g. client sends /openai/v1/chat/completions → remainder "v1/chat/completions"
#      upstream base already has /v1 → strip to "chat/completions"
_UPSTREAM_PREFIX_DEDUP: dict[str, str] = {
    "openai": "v1/",
}

# Headers that must not be forwarded to the client (hop-by-hop)
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "content-length", "content-encoding",
})

# Module-level shared client — reused across requests for connection pooling
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)
    return _client


# Port → (ProcessInfo, timestamp) cache to avoid scanning /proc on every request
_tag_cache: dict[int, tuple[ProcessInfo, float]] = {}
_TAG_CACHE_TTL = 30.0  # seconds

# Process skip list file and TTL cache
_SKIP_PROCESSES_FILE = Path.home() / ".toklog" / "skip_processes"
_skip_proc_cache: tuple[frozenset[str], float] | None = None
_SKIP_PROC_CACHE_TTL = 30.0  # seconds


def _get_skip_processes() -> frozenset[str]:
    """Load process skip patterns from config.json or skip_processes file (TTL-cached).
    
    Tries config first; if it has patterns, uses those. Otherwise, falls back to the
    old skip_processes file for backward compatibility.
    """
    global _skip_proc_cache
    now = time.monotonic()
    if _skip_proc_cache is not None:
        patterns, ts = _skip_proc_cache
        if now - ts < _SKIP_PROC_CACHE_TTL:
            return patterns

    # Try to load from config first; only use it if non-empty
    try:
        from toklog.config import load_config
        config = load_config(validate=False)
        patterns_list = config.get("skip_processes", [])
        if patterns_list:  # Only use config if it has patterns
            patterns = frozenset(str(p).lower() for p in patterns_list if p)
            _skip_proc_cache = (patterns, now)
            return patterns
    except Exception:
        # Fallback to old file if config loading fails
        pass

    # Fallback: read old skip_processes file for backward compat
    if not _SKIP_PROCESSES_FILE.exists():
        _skip_proc_cache = (frozenset(), now)
        return frozenset()
    lines = _SKIP_PROCESSES_FILE.read_text(encoding="utf-8").splitlines()
    patterns = frozenset(
        line.strip().lower()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    )
    _skip_proc_cache = (patterns, now)
    return patterns


def _resolve_process_tag(client_port: int) -> ProcessInfo:
    """Map a TCP client port to a ProcessInfo by scanning /proc/net/tcp + /proc/{pid}/environ."""
    try:
        port_hex = format(client_port, "04X")
        inode: str | None = None

        for tcp_path in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                with open(tcp_path) as f:
                    for line in f:
                        cols = line.split()
                        if len(cols) < 10:
                            continue
                        addr_parts = cols[1].split(":")
                        if len(addr_parts) < 2:
                            continue
                        if addr_parts[1] == port_hex:
                            inode = cols[9]
                            break
            except OSError:
                continue
            if inode:
                break

        if not inode:
            return ProcessInfo(tag=None, cmdline=None)

        socket_link = f"socket:[{inode}]"
        pid: int | None = None

        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                for fd_entry in os.scandir(f"/proc/{entry.name}/fd"):
                    try:
                        if os.readlink(fd_entry.path) == socket_link:
                            pid = int(entry.name)
                            break
                    except OSError:
                        continue
            except OSError:
                continue
            if pid:
                break

        if not pid:
            return ProcessInfo(tag=None, cmdline=None)

        # Read cmdline (null-separated args)
        cmdline: str | None = None
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw_cmd = f.read().decode(errors="replace")
            cmdline = " ".join(raw_cmd.split("\0")).strip()[:256] or None
        except OSError:
            pass

        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read().decode(errors="replace")
        env = {k: v for item in raw.split("\0") if "=" in item for k, v in [item.split("=", 1)]}

        # TOKLOG_TAG overrides everything — lets callers self-identify.
        explicit_tag = env.get("TOKLOG_TAG", "")
        if explicit_tag:
            return ProcessInfo(tag=explicit_tag, cmdline=cmdline)

        user = env.get("USER", "")
        if not user:
            return ProcessInfo(tag=None, cmdline=cmdline)
        return ProcessInfo(tag=user, cmdline=cmdline)

    except Exception:
        return ProcessInfo(tag=None, cmdline=None)


def _lookup_process_info(client_port: int) -> ProcessInfo:
    """Cached wrapper around _resolve_process_tag."""
    now = time.monotonic()
    if client_port in _tag_cache:
        info, ts = _tag_cache[client_port]
        if now - ts < _TAG_CACHE_TTL:
            return info
    info = _resolve_process_tag(client_port)
    _tag_cache[client_port] = (info, now)
    # Evict stale entries
    stale = [p for p, (_, ts) in _tag_cache.items() if now - ts >= _TAG_CACHE_TTL]
    for p in stale:
        del _tag_cache[p]
    return info


@functools.lru_cache(maxsize=16)
def _parse_skip_patterns(raw: str) -> frozenset[str]:
    """Parse TOKLOG_SKIP_MODELS value into a frozenset of lowercase substrings."""
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _create_stream_handler(provider: str, endpoint_family: str | None):
    """Create the appropriate streaming event handler for the provider."""
    if provider == "openai" and endpoint_family:
        return openai_adapter.create_stream_handler(endpoint_family)
    elif provider == "anthropic":
        return anthropic_adapter.create_stream_handler()
    elif provider == "gemini":
        return gemini_adapter.create_stream_handler()
    return None


def _extract_non_streaming(entry: dict, provider: str, endpoint_family: str | None, body: dict) -> None:
    """Extract usage from a non-streaming response using the appropriate adapter.

    Each adapter's extract_from_response() captures all fields including
    stop_reason, and apply_to_entry() writes them to the log entry.
    """
    if provider == "openai" and endpoint_family:
        result = openai_adapter.extract_from_response(body, endpoint_family)
        result.apply_to_entry(entry)
    elif provider == "anthropic":
        result = anthropic_adapter.extract_from_response(body)
        result.apply_to_entry(entry)
    elif provider == "gemini":
        result = gemini_adapter.extract_from_response(body)
        result.apply_to_entry(entry)


async def proxy_handler(request: Request) -> Response:
    """Route, forward, log, and return all LLM API requests."""
    # Parse provider from URL prefix: /openai/... or /anthropic/...
    path = request.url.path.lstrip("/")
    parts = path.split("/", 1)
    if len(parts) < 2 or not parts[1] or parts[0] not in _UPSTREAM:
        return Response("Not found", status_code=404)

    provider, remainder = parts[0], parts[1]

    # Strip duplicate path prefix (e.g. /openai/v1/... when upstream already has /v1)
    dedup = _UPSTREAM_PREFIX_DEDUP.get(provider)
    if dedup and remainder.startswith(dedup):
        remainder = remainder[len(dedup):]

    upstream_url = f"{_UPSTREAM[provider]}/{remainder}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    # Build forwarded headers — strip proxy-specific and hop-by-hop headers
    fw_headers: dict[str, str] = {}
    for k, v in request.headers.items():
        if k.lower() in ("host", "content-length", "x-tb-tag", "x-tb-skip",
                         "transfer-encoding"):
            continue
        fw_headers[k] = v
    fw_headers["host"] = _UPSTREAM[provider].split("//")[1].split("/")[0]

    disabled = os.environ.get("TOKLOG_DISABLED") == "1"
    skip_proc_tag = os.environ.get("TOKLOG_SKIP_PROCESS_TAG") == "1"

    info = ProcessInfo(tag=None, cmdline=None)
    if request.client and not disabled and not skip_proc_tag:
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(None, _lookup_process_info, request.client.port)
    tag = request.headers.get("x-tb-tag") or info.tag
    skip = request.headers.get("x-tb-skip", "").lower() == "true"

    start = time.monotonic()
    body_bytes = await request.body()

    try:
        body: dict = json.loads(body_bytes) if body_bytes else {}
    except (json.JSONDecodeError, ValueError):
        body = {}

    # no-log body param (LiteLLM-compatible per-request opt-out); strip before forwarding
    no_log = bool(body.pop("no-log", False))
    if no_log:
        body_bytes = json.dumps(body).encode()

    # Model-based skip: TOKLOG_SKIP_MODELS=haiku,opus-4-5 (comma-separated substrings)
    skip_patterns = _parse_skip_patterns(os.environ.get("TOKLOG_SKIP_MODELS", ""))
    if provider == "gemini":
        _model = (gemini_adapter.extract_model_from_url(remainder) or "").lower()
    else:
        _model = (body.get("model") or "").lower()
    model_skipped = bool(skip_patterns and any(p in _model for p in skip_patterns))

    # Process-based skip: ~/.toklog/skip_processes
    skip_procs = _get_skip_processes()
    process_skipped = bool(
        skip_procs and info.cmdline and any(p in info.cmdline.lower() for p in skip_procs)
    )

    should_log = not skip and not no_log and not disabled and not model_skipped and not process_skipped

    # Streaming detection — Gemini uses URL path, others use body
    if provider == "gemini":
        is_stream = "streamGenerateContent" in remainder
    else:
        is_stream = body.get("stream", False)

    # Gemini streaming requires alt=sse so the response is SSE format (not newline-delimited JSON)
    if provider == "gemini" and is_stream:
        sep = "&" if "?" in upstream_url else "?"
        if "alt=sse" not in upstream_url:
            upstream_url += f"{sep}alt=sse"

    # Classify endpoint and inject stream_options where needed
    endpoint_family: str | None = None
    if provider == "openai":
        endpoint_family = openai_adapter.classify_endpoint(remainder, body)
        if is_stream and endpoint_family == "openai.chat_completions":
            had_opts = "stream_options" in body
            body = openai_adapter.maybe_inject_stream_options(body, endpoint_family)
            if not had_opts and "stream_options" in body:
                body_bytes = json.dumps(body).encode()
    elif provider == "anthropic":
        endpoint_family = "anthropic.messages"
    elif provider == "gemini":
        endpoint_family = "gemini.generate_content"

    entry: dict = extract_from_request(provider, body, tag, request.headers, cmdline=info.cmdline) if should_log else {}
    if should_log and provider == "gemini":
        url_model = gemini_adapter.extract_model_from_url(remainder)
        if url_model:
            entry["model"] = url_model
    if should_log and provider == "gemini" and is_stream:
        entry["streaming"] = True

    client = _get_client()
    prepped = client.build_request(
        request.method, upstream_url, headers=fw_headers, content=body_bytes
    )

    try:
        resp = await client.send(prepped, stream=True)
    except Exception as exc:
        if should_log:
            entry["duration_ms"] = int((time.monotonic() - start) * 1000)
            entry["error"] = True
            entry["error_type"] = type(exc).__name__
            log_entry(entry)
        raise

    if is_stream:
        resp_headers = _filter_response_headers(resp.headers)

        # Flag upstream HTTP errors before the generator runs so the entry
        # gets the error flag regardless of what the SSE parser finds.
        if should_log and resp.status_code >= 400:
            entry["error"] = True
            entry["error_type"] = f"http_{resp.status_code}"

        async def generate() -> AsyncIterator[bytes]:
            buffer = SSEStreamBuffer()
            handler = _create_stream_handler(provider, endpoint_family)
            try:
                async for chunk in resp.aiter_bytes():
                    if should_log and handler is not None:
                        for event in buffer.feed(chunk):
                            handler.handle(event)
                    yield chunk
            except Exception as exc:
                if should_log:
                    entry["error"] = True
                    entry["error_type"] = type(exc).__name__
                raise  # Propagate so client sees connection error, not truncated stream
            finally:
                if should_log and handler is not None:
                    for event in buffer.flush():
                        handler.handle(event)
                    handler.apply_to_entry(entry)
                    if endpoint_family:
                        entry["endpoint_family"] = endpoint_family
                try:
                    await resp.aclose()
                except Exception:
                    pass  # Don't mask the original exception
                if should_log:
                    entry["duration_ms"] = int((time.monotonic() - start) * 1000)
                    log_entry(entry)

        return StreamingResponse(
            generate(), status_code=resp.status_code, headers=resp_headers
        )

    # Non-streaming: buffer full response, extract usage, log
    try:
        content = await resp.aread()
    finally:
        await resp.aclose()

    entry["duration_ms"] = int((time.monotonic() - start) * 1000)

    if should_log:
        if resp.status_code >= 400:
            entry["error"] = True
            entry["error_type"] = f"http_{resp.status_code}"
        ct = resp.headers.get("content-type", "")
        if ct.startswith("application/json") and content:
            try:
                response_body = json.loads(content)
                _extract_non_streaming(entry, provider, endpoint_family, response_body)
            except (json.JSONDecodeError, ValueError):
                pass
        if endpoint_family:
            entry["endpoint_family"] = endpoint_family
        log_entry(entry)

    return Response(
        content=content,
        status_code=resp.status_code,
        headers=_filter_response_headers(resp.headers),
    )


app = Starlette(
    routes=[
        Route(
            "/{path:path}",
            proxy_handler,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        ),
    ]
)
