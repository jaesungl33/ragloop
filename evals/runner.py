"""Evals runner: compare BaselineRAG vs RagLoop over the bundled question set.

Usage::

    python -m evals.runner           # tries ANTHROPIC_API_KEY; falls back to offline
    python -m evals.runner --offline # force in-memory fake LLM (no API key needed)

Results are written to evals/results.json so report.py can be run separately.

Metrics
-------
citation_accuracy   Pure-Python; always runs. Fraction of [source:ID] citations
                    in the answer that are actually present in the retrieved set.
faithfulness        LLM-judged (RAGAS). Measures grounding in context, NOT
                    factual truth. Requires ragas + API key.
answer_relevancy    LLM-judged (RAGAS). How well the answer addresses the question.
context_precision   LLM-judged (RAGAS). Fraction of retrieved chunks that matter.
context_recall      LLM-judged (RAGAS). Fraction of relevant chunks retrieved.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "examples"))

from ragloop import Deps, LLMProvider, RagLoop  # noqa: E402

from .baseline import BaselineRAG  # noqa: E402
from .corpus import DOCS, QUESTIONS  # noqa: E402

_CITE_RE = re.compile(r"\[source:([^\]]+)\]")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunResult:
    pipeline: str
    question: str
    answer: str
    sources: List[str]
    contexts: List[str]
    ground_truth: str
    latency_s: float
    token_cost: int
    retries: int
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    citation_accuracy: Optional[float] = None


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

class TracingLLM(LLMProvider):
    """Wraps any LLMProvider to track approximate token usage between resets."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.call_count: int = 0
        self.total_chars: int = 0

    def complete(self, system: str, prompt: str) -> str:
        response = self._inner.complete(system, prompt)
        self.call_count += 1
        self.total_chars += len(system) + len(prompt) + len(response)
        return response

    def reset(self) -> None:
        self.call_count = 0
        self.total_chars = 0

    @property
    def token_cost(self) -> int:
        return max(1, self.total_chars // 4)


# ---------------------------------------------------------------------------
# Offline LLM
# ---------------------------------------------------------------------------

class _OfflineLLM(LLMProvider):
    """Deterministic fake for --offline mode.

    Unlike EchoLLM from quickstart.py, this one actually reads the context
    passed to the generator and cites a real retrieved source — producing
    non-trivial citation_accuracy values without any API call.
    """

    def complete(self, system: str, prompt: str) -> str:
        if "decompose" in system:
            last_line = prompt.strip().splitlines()[-1][:80].replace('"', '\\"')
            return f'["{last_line}"]'
        if "grade" in system:
            return '{"grounded": true, "reason": "supported"}'
        # Generator: find sources in context and cite the first one
        sources = _CITE_RE.findall(prompt)
        if sources:
            src = sources[0]
            match = re.search(
                rf"\[source:{re.escape(src)}\]\s*([^.!?]+[.!?])", prompt
            )
            snippet = match.group(1).strip() if match else "See the source for details."
            return f"{snippet} [source:{src}]"
        return "The provided sources do not contain information about this question."


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    name: str,
    ask_fn,
    questions: list,
    tracing_llm: TracingLLM,
    retriever,
) -> List[RunResult]:
    results: List[RunResult] = []
    for q in questions:
        tracing_llm.reset()
        t0 = time.perf_counter()
        raw = ask_fn(q["question"])
        latency = time.perf_counter() - t0

        # Baseline returns contexts directly; RagLoop returns source IDs only —
        # reconstruct context text via get_chunk so RAGAS has what it needs.
        if "contexts" in raw:
            contexts: List[str] = raw["contexts"]
        else:
            contexts = []
            for sid in raw.get("sources", []):
                doc = retriever.get_chunk(sid)
                if doc:
                    contexts.append(doc.text)

        results.append(RunResult(
            pipeline=name,
            question=q["question"],
            answer=raw["answer"],
            sources=raw.get("sources", []),
            contexts=contexts,
            ground_truth=q["ground_truth"],
            latency_s=latency,
            token_cost=tracing_llm.token_cost,
            retries=max(0, raw.get("attempts", 1) - 1),
        ))
    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_citation_accuracy(results: List[RunResult]) -> None:
    for r in results:
        cited = set(_CITE_RE.findall(r.answer))
        if not cited:
            r.citation_accuracy = 1.0
        else:
            r.citation_accuracy = len(cited & set(r.sources)) / len(cited)


def _score_with_ragas(results: List[RunResult]) -> None:
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print("[ragas] package not installed — skipping LLM-judged metrics")
        return

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        print("[ragas] no API key found — skipping LLM-judged metrics")
        return

    data = {
        "question": [r.question for r in results],
        "answer": [r.answer for r in results],
        "contexts": [r.contexts for r in results],
        "ground_truth": [r.ground_truth for r in results],
    }
    try:
        dataset = Dataset.from_dict(data)
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = scores.to_pandas()
        for i, r in enumerate(results):
            for attr in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ):
                if attr in df.columns:
                    val = df.iloc[i][attr]
                    try:
                        fval = float(val)
                        if fval == fval:  # skip NaN
                            setattr(r, attr, fval)
                    except (TypeError, ValueError):
                        pass
    except Exception as exc:
        print(f"[ragas] scoring failed: {exc}")


def _score_with_deepeval(results: List[RunResult]) -> None:
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric  # type: ignore
        from deepeval.test_case import LLMTestCase  # type: ignore
    except ImportError:
        print("[deepeval] package not installed — skipping")
        return

    for r in results:
        case = LLMTestCase(
            input=r.question,
            actual_output=r.answer,
            expected_output=r.ground_truth,
            retrieval_context=r.contexts,
        )
        if r.faithfulness is None:
            try:
                m = FaithfulnessMetric(threshold=0.5)
                m.measure(case)
                r.faithfulness = float(m.score)
            except Exception:
                pass
        if r.answer_relevancy is None:
            try:
                m = AnswerRelevancyMetric(threshold=0.5)
                m.measure(case)
                r.answer_relevancy = float(m.score)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm(offline: bool) -> LLMProvider:
    if offline:
        return _OfflineLLM()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        from ragloop.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
        print("[runner] using AnthropicProvider (claude-sonnet-4-6)")
        return AnthropicProvider()
    print("[runner] ANTHROPIC_API_KEY not set — falling back to offline LLM")
    return _OfflineLLM()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from . import report  # local import avoids any circular-import risk

    parser = argparse.ArgumentParser(description="Run ragloop evals")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the in-memory fake LLM (no API key required).",
    )
    args = parser.parse_args()

    # --- shared retriever ---
    # Import here so we can also use InMemoryRetriever without pulling in
    # quickstart at module level (keeps import errors obvious).
    from quickstart import InMemoryRetriever  # type: ignore  # noqa: PLC0415

    retriever = InMemoryRetriever()
    retriever.add(DOCS)
    print(f"[runner] loaded {len(DOCS)} chunks, {len(QUESTIONS)} questions")

    inner_llm = _build_llm(args.offline)
    tracing_llm = TracingLLM(inner_llm)

    # --- baseline ---
    baseline_pipeline = BaselineRAG(retriever=retriever, llm=tracing_llm, k=5)
    print("[runner] running baseline ...")
    baseline_results = run_pipeline(
        "baseline", baseline_pipeline.ask, QUESTIONS, tracing_llm, retriever
    )

    # --- ragloop ---
    loop = RagLoop(Deps(retriever=retriever, llm=tracing_llm, k=5), max_attempts=2)
    print("[runner] running ragloop ...")
    ragloop_results = run_pipeline(
        "ragloop", loop.ask, QUESTIONS, tracing_llm, retriever
    )

    # --- scoring ---
    print("[runner] scoring citation accuracy ...")
    _score_citation_accuracy(baseline_results)
    _score_citation_accuracy(ragloop_results)

    if not args.offline:
        print("[runner] scoring with ragas ...")
        _score_with_ragas(baseline_results)
        _score_with_ragas(ragloop_results)
        print("[runner] scoring with deepeval ...")
        _score_with_deepeval(baseline_results)
        _score_with_deepeval(ragloop_results)

    # --- persist ---
    out_path = Path(__file__).parent / "results.json"
    payload = [dataclasses.asdict(r) for r in baseline_results + ragloop_results]
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[runner] results written to {out_path}")

    # --- report ---
    report.print_table(
        [dataclasses.asdict(r) for r in baseline_results],
        [dataclasses.asdict(r) for r in ragloop_results],
    )


if __name__ == "__main__":
    main()
