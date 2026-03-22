"""Unit tests for toklog/proxy/setup.py — shell profile injection/removal."""

from __future__ import annotations

from pathlib import Path

import pytest

from toklog.proxy.setup import (
    _MARKER_END,
    _MARKER_START,
    inject_shell_profile,
    remove_shell_profile,
)


@pytest.fixture()
def profile(tmp_path) -> Path:
    return tmp_path / ".bashrc"


def test_inject_creates_block(profile):
    out = inject_shell_profile(profile=profile, port=4007)

    assert out == profile
    text = profile.read_text()
    assert _MARKER_START in text
    assert _MARKER_END in text
    assert "OPENAI_BASE_URL=http://127.0.0.1:4007/openai" in text
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:4007/anthropic" in text


def test_inject_custom_port(profile):
    inject_shell_profile(profile=profile, port=9999)

    text = profile.read_text()
    assert "9999" in text


def test_inject_is_idempotent(profile):
    inject_shell_profile(profile=profile)
    inject_shell_profile(profile=profile)  # second call

    text = profile.read_text()
    assert text.count(_MARKER_START) == 1


def test_inject_appends_to_existing_content(profile):
    profile.write_text("# existing content\n")
    inject_shell_profile(profile=profile)

    text = profile.read_text()
    assert text.startswith("# existing content\n")
    assert _MARKER_START in text


def test_remove_deletes_block(profile):
    inject_shell_profile(profile=profile)
    result = remove_shell_profile(profile=profile)

    assert result is True
    text = profile.read_text()
    assert _MARKER_START not in text
    assert _MARKER_END not in text
    assert "OPENAI_BASE_URL" not in text


def test_remove_preserves_surrounding_content(profile):
    profile.write_text("# before\n")
    inject_shell_profile(profile=profile)

    with open(profile, "a") as f:
        f.write("# after\n")

    remove_shell_profile(profile=profile)
    text = profile.read_text()
    assert "# before" in text
    assert "# after" in text


def test_remove_returns_false_when_nothing_to_remove(profile):
    profile.write_text("# no proxy block here\n")
    result = remove_shell_profile(profile=profile)
    assert result is False


def test_remove_returns_false_when_file_missing(tmp_path):
    missing = tmp_path / "nonexistent_profile"
    result = remove_shell_profile(profile=missing)
    assert result is False
