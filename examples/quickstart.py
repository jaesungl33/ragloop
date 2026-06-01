"""End-to-end example with zero external dependencies.

Demonstrates the two extension points -- a custom Retriever and a custom
LLMProvider -- by using in-memory fakes. Run: ``python examples/quickstart.py``.
This is also how the test suite exercises the loop without network calls.
"""
from __future__ import annotations

from ragloop import Deps, Document, LLMProvider, RagLoop, Retriever


class InMemoryRetriever(Retriever):
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    def add(self, documents: list[Document]) -> None:
        for d in documents:
            self._docs[d.id] = d

    def semantic_search(self, query: str, k: int = 5) -> list[Document]:
        terms = set(query.lower().split())
        scored = [
            Document(d.id, d.text, d.metadata, score=len(terms & set(d.text.lower().split())))
            for d in self._docs.values()
        ]
        scored.sort(key=lambda d: d.score or 0, reverse=True)
        return scored[:k]

    def keyword_search(self, query: str, k: int = 5) -> list[Document]:
        hits = [d for d in self._docs.values() if query.lower() in d.text.lower()]
        return hits[:k]

    def get_chunk(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)


class EchoLLM(LLMProvider):
    """Deterministic stand-in: planner returns the query, generator quotes a
    source, critic always approves. Replace with a real provider in practice."""

    def complete(self, system: str, prompt: str) -> str:
        if "decompose" in system:
            return '["refund policy"]'
        if "grade" in system:
            return '{"grounded": true, "reason": "supported"}'
        return "Refunds are accepted within 30 days [source:policy:0]."


if __name__ == "__main__":
    retriever = InMemoryRetriever()
    retriever.add([
        Document("policy:0", "Customers may request a refund within 30 days of purchase."),
        Document("policy:1", "Shipping is free on orders over $50."),
    ])
    loop = RagLoop(Deps(retriever=retriever, llm=EchoLLM(), k=3), max_attempts=2)
    print(loop.ask("What is the refund window?"))
