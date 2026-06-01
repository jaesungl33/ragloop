"""Naive RAG baseline: single embed -> top-k retrieve -> generate.

No planner, no critic, no retry loop. Serves as a reference point to measure
how much the self-correcting RagLoop adds over a plain one-shot pipeline.
Reuses the existing Retriever and LLMProvider interfaces without modification.
"""
from __future__ import annotations

import time

from ragloop import LLMProvider, Retriever

_SYSTEM = (
    "Answer the question using ONLY the provided sources. Cite each claim "
    "inline as [source:ID]. If the sources do not contain the answer, say "
    "so plainly. Do not use outside knowledge."
)


class BaselineRAG:
    """Single embed -> top-k -> generate; no planner, no critic, no retry."""

    def __init__(self, retriever: Retriever, llm: LLMProvider, k: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def ask(self, query: str) -> dict:
        t0 = time.perf_counter()
        docs = self.retriever.semantic_search(query, k=self.k)
        context = "\n\n".join(f"[source:{d.id}] {d.text}" for d in docs)
        prompt = f"Sources:\n{context}\n\nQuestion: {query}"
        answer = self.llm.complete(_SYSTEM, prompt)
        latency = time.perf_counter() - t0
        return {
            "answer": answer,
            "sources": [d.id for d in docs],
            "contexts": [d.text for d in docs],
            "latency_s": latency,
            "retries": 0,
            "attempts": 1,
        }
