"""Tests for config parsing and the provider/retriever factories."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragloop.config import Config, _build_llm, _build_retriever, build_from_config  # noqa: E402


def test_defaults():
    c = Config()
    assert c.llm_provider == "anthropic"
    assert c.retriever_backend == "chroma"
    assert c.top_k == 5
    assert c.max_attempts == 2
    assert c.critic_fail_closed is False


def test_from_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "llm_provider: ollama\n"
        "top_k: 7\n"
        "max_attempts: 3\n"
        "critic_fail_closed: true\n"
        "llm:\n  model: llama3.2:3b\n"
    )
    c = Config.from_yaml(str(p))
    assert c.llm_provider == "ollama"
    assert c.top_k == 7
    assert c.max_attempts == 3
    assert c.critic_fail_closed is True
    assert c.llm["model"] == "llama3.2:3b"


def test_build_llm_ollama_does_not_connect():
    llm = _build_llm(Config(llm_provider="ollama", llm={"model": "x"}))
    assert llm.__class__.__name__ == "OllamaProvider"


def test_build_llm_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _build_llm(Config(llm_provider="anthropic"))


def test_build_llm_unknown_raises():
    with pytest.raises(ValueError):
        _build_llm(Config(llm_provider="bogus"))


def test_build_retriever_unknown_raises():
    with pytest.raises(ValueError):
        _build_retriever(Config(retriever_backend="bogus"))


def test_build_from_config_wires_a_runnable_loop():
    pytest.importorskip("chromadb")
    cfg = Config(
        llm_provider="ollama",            # constructed but never called here
        llm={"model": "x"},
        retriever_backend="chroma",
        retriever={"collection": "ragloop_cfgtest"},
        top_k=4,
        max_attempts=3,
        critic_fail_closed=True,
    )
    loop = build_from_config(cfg=cfg)
    assert loop.max_attempts == 3
    assert loop.deps.k == 4
    assert loop.deps.fail_closed is True
