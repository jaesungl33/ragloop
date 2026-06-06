"""One-shot RAG baselines: single embed -> top-k retrieve -> generate.

Two variants, no planner / critic / retry in either:

- ``BaselineRAG``  — a *careful* one-shot pipeline whose prompt already tells
  the model to answer only from the sources and to decline otherwise. This is
  the honest, hard-to-beat reference.
- ``NaiveRAG``     — a *naive* one-shot pipeline that just stuffs the context
  and asks the question, with no grounding guard. This is what a lot of quick
  RAG demos actually ship, and it is what the self-correcting loop most clearly
  improves on.

Both reuse the Retriever / LLMProvider interfaces without modification.
"""
from __future__ import annotations

import time

from ragloop import LLMProvider, Retriever

_GROUNDED_SYSTEM = (
    "Answer the question using ONLY the provided sources. Cite each claim "
    "inline as [source:ID]. If the sources do not contain the answer, say "
    "so plainly. Do not use outside knowledge."
)

# No grounding guard, no decline instruction -- the common naive setup.
_NAIVE_SYSTEM = "You are a helpful assistant. Answer the user's question using the context provided."


class BaselineRAG:
    """Careful one-shot: embed -> top-k -> grounded generate. No critic/retry."""

    _SYSTEM = _GROUNDED_SYSTEM

    def __init__(self, retriever: Retriever, llm: LLMProvider, k: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def ask(self, query: str) -> dict:
        t0 = time.perf_counter()
        docs = self.retriever.semantic_search(query, k=self.k)
        context = "\n\n".join(f"[source:{d.id}] {d.text}" for d in docs)
        prompt = f"Sources:\n{context}\n\nQuestion: {query}"
        answer = self.llm.complete(self._SYSTEM, prompt)
        latency = time.perf_counter() - t0
        return {
            "answer": answer,
            "sources": [d.id for d in docs],
            "contexts": [d.text for d in docs],
            "latency_s": latency,
            "retries": 0,
            "attempts": 1,
        }


class NaiveRAG(BaselineRAG):
    """Naive one-shot: same retrieval, but no grounding guard in the prompt."""

    _SYSTEM = _NAIVE_SYSTEM
