# toklog/adapters/sse.py

import json
import logging

logger = logging.getLogger(__name__)


class SSEStreamBuffer:
    """Buffers raw SSE bytes and yields complete, parsed events.

    Handles the byte-level concerns that affect ALL providers going through
    the proxy: TCP chunk fragmentation, event boundary detection, multi-line
    data payloads, and [DONE] filtering.

    Usage:
        buf = SSEStreamBuffer()
        for chunk in resp.aiter_bytes():
            for event in buf.feed(chunk):
                handler.handle(event)  # event is a parsed dict
        for event in buf.flush():
            handler.handle(event)
    """

    def __init__(self) -> None:
        self._buffer = b""
        self._flushed: list[dict] | None = None

    def feed(self, chunk: bytes) -> list[dict]:
        """Feed raw bytes. Returns a list of parsed event dicts (may be empty).

        Must not be called after flush().
        """
        if self._flushed is not None:
            raise RuntimeError("Cannot feed() after flush() — buffer already finalized")
        self._buffer += chunk
        return self._drain_complete_events()

    def flush(self) -> list[dict]:
        """Call when the stream ends. Parses any trailing data in the buffer. Idempotent."""
        if self._flushed is not None:
            return self._flushed
        events: list[dict] = []
        if self._buffer.strip():
            parsed = self._parse_event(self._buffer)
            if parsed is not None:
                events.append(parsed)
            self._buffer = b""
        self._flushed = events
        return events

    def _drain_complete_events(self) -> list[dict]:
        """Split buffer on SSE event boundaries and parse complete events."""
        events: list[dict] = []
        while True:
            # SSE events are delimited by a blank line (\n\n or \r\n\r\n)
            idx_nn = self._buffer.find(b"\n\n")
            idx_rn = self._buffer.find(b"\r\n\r\n")

            if idx_nn == -1 and idx_rn == -1:
                return events  # no complete event yet

            if idx_nn == -1:
                idx, delim_len = idx_rn, 4
            elif idx_rn == -1:
                idx, delim_len = idx_nn, 2
            else:
                if idx_nn <= idx_rn:
                    idx, delim_len = idx_nn, 2
                else:
                    idx, delim_len = idx_rn, 4

            event_bytes = self._buffer[:idx]
            self._buffer = self._buffer[idx + delim_len:]

            if event_bytes.strip():
                parsed = self._parse_event(event_bytes)
                if parsed is not None:
                    events.append(parsed)

    def _parse_event(self, event_bytes: bytes) -> dict | None:
        """Parse a single SSE event into a dict. Returns None on skip/failure."""
        data_parts: list[bytes] = []

        for line in event_bytes.split(b"\n"):
            line = line.rstrip(b"\r")
            if line.startswith(b"data: "):
                data_parts.append(line[6:])
            elif line.startswith(b"data:"):
                data_parts.append(line[5:])
            # Ignore event:, id:, retry:, comment lines

        if not data_parts:
            return None

        payload = b"\n".join(data_parts).strip()
        if payload == b"[DONE]":
            return None

        try:
            return json.loads(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "SSEStreamBuffer: malformed JSON in complete SSE event: %s (payload: %s)",
                exc,
                payload[:200],
            )
            return None
