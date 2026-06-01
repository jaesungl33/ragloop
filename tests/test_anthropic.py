"""Unit tests for the Anthropic provider with the SDK client mocked."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("anthropic")

from ragloop.llm import _retry  # noqa: E402
from ragloop.llm.anthropic_provider import AnthropicProvider  # noqa: E402


class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, *blocks):
        self.content = list(blocks)


def _provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # no real call at init
    return AnthropicProvider(model="claude-test")


def test_complete_joins_text_blocks(monkeypatch):
    p = _provider(monkeypatch)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _Resp(_TextBlock("Refund window is 30 days "), _TextBlock("[source:refunds:0]."))

    p._client.messages.create = fake_create
    out = p.complete("system text", "the question")
    assert out == "Refund window is 30 days [source:refunds:0]."
    assert captured["model"] == "claude-test"
    assert captured["system"] == "system text"


def test_complete_retries_on_transient_error(monkeypatch):
    p = _provider(monkeypatch)
    p._transient = (ValueError,)  # treat ValueError as transient for the test
    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("rate limited")
        return _Resp(_TextBlock("ok"))

    p._client.messages.create = flaky_create
    assert p.complete("s", "p") == "ok"
    assert calls["n"] == 2


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AnthropicProvider()
