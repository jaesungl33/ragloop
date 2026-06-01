"""Unit tests for the Ollama provider with the network mocked out."""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragloop.llm import _retry, ollama_provider  # noqa: E402
from ragloop.llm.ollama_provider import OllamaProvider  # noqa: E402


def _no_sleep(monkeypatch):
    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)


def _fake_response(content: str):
    payload = json.dumps({"message": {"content": content}}).encode("utf-8")
    return io.BytesIO(payload)


def test_complete_parses_message_content(monkeypatch):
    def fake_urlopen(req, timeout=None):
        body = _fake_response("Refunds are 30 days [source:refunds:0].")
        body.__enter__ = lambda s: s        # support the `with` statement
        body.__exit__ = lambda s, *a: False
        return body

    monkeypatch.setattr(ollama_provider.urllib.request, "urlopen", fake_urlopen)
    out = OllamaProvider(model="test").complete("system", "prompt")
    assert out == "Refunds are 30 days [source:refunds:0]."


def test_complete_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection refused")
        body = _fake_response("ok")
        body.__enter__ = lambda s: s
        body.__exit__ = lambda s, *a: False
        return body

    monkeypatch.setattr(ollama_provider.urllib.request, "urlopen", flaky_urlopen)
    _no_sleep(monkeypatch)
    out = OllamaProvider(model="test", max_retries=2).complete("s", "p")
    assert out == "ok"
    assert calls["n"] == 2  # failed once, retried, succeeded


def test_complete_raises_helpful_error_when_unreachable(monkeypatch):
    def dead_urlopen(req, timeout=None):
        raise urllib.error.URLError("no server")

    monkeypatch.setattr(ollama_provider.urllib.request, "urlopen", dead_urlopen)
    _no_sleep(monkeypatch)
    with pytest.raises(RuntimeError, match="Ollama"):
        OllamaProvider(model="test", max_retries=1).complete("s", "p")
