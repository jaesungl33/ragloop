"""Offline test of the full graph using in-memory fakes (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from quickstart import EchoLLM, InMemoryRetriever  # noqa: E402

from ragloop import Deps, Document, RagLoop  # noqa: E402


def _loop():
    r = InMemoryRetriever()
    r.add([
        Document("policy:0", "Customers may request a refund within 30 days of purchase."),
        Document("policy:1", "Shipping is free on orders over $50."),
    ])
    return RagLoop(Deps(retriever=r, llm=EchoLLM(), k=3), max_attempts=2)


def test_answer_is_grounded():
    result = _loop().ask("What is the refund window?")
    assert result["grounded"] is True
    assert "30 days" in result["answer"]


def test_sources_returned():
    result = _loop().ask("refund")
    assert any(s.startswith("policy:") for s in result["sources"])


def test_attempts_bounded():
    result = _loop().ask("anything")
    assert result["attempts"] <= 2
