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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "examples"))

from ragloop import Deps, LLMProvider, RagLoop  # noqa: E402

from .baseline import BaselineRAG, NaiveRAG  # noqa: E402
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
    sources: list[str]
    contexts: list[str]
    ground_truth: str
    latency_s: float
    token_cost: int
    retries: int
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    citation_accuracy: float | None = None


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
) -> list[RunResult]:
    results: list[RunResult] = []
    for q in questions:
        tracing_llm.reset()
        t0 = time.perf_counter()
        raw = ask_fn(q["question"])
        latency = time.perf_counter() - t0

        # Baseline returns contexts directly; RagLoop returns source IDs only —
        # reconstruct context text via get_chunk so RAGAS has what it needs.
        if "contexts" in raw:
            contexts: list[str] = raw["contexts"]
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

def _score_citation_accuracy(results: list[RunResult]) -> None:
    for r in results:
        cited = set(_CITE_RE.findall(r.answer))
        if not cited:
            r.citation_accuracy = 1.0
        else:
            r.citation_accuracy = len(cited & set(r.sources)) / len(cited)


def _score_with_ragas(results: list[RunResult]) -> None:
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


def _score_with_deepeval(results: list[RunResult]) -> None:
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

def _build_llm(provider: str, model: str | None) -> LLMProvider:
    """provider: 'auto' | 'anthropic' | 'ollama' | 'offline'."""
    if provider == "offline":
        return _OfflineLLM()
    if provider == "ollama":
        from ragloop.llm.ollama_provider import OllamaProvider  # noqa: PLC0415
        m = model or "llama3.2:3b"
        print(f"[runner] using OllamaProvider (local, model={m})")
        return OllamaProvider(model=m)
    if provider == "anthropic" or (provider == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        from ragloop.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
        m = model or "claude-sonnet-4-6"
        print(f"[runner] using AnthropicProvider (model={m})")
        return AnthropicProvider(model=m)
    print("[runner] no real LLM selected/available — falling back to offline LLM")
    return _OfflineLLM()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from . import report  # local import avoids any circular-import risk
    from .metrics import score_deterministic  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run ragloop evals")
    parser.add_argument(
        "--llm",
        choices=["auto", "anthropic", "ollama", "offline"],
        default="auto",
        help="Which LLM to generate answers with. 'auto' uses Anthropic if "
        "ANTHROPIC_API_KEY is set, else offline. 'ollama' runs locally for $0.",
    )
    parser.add_argument("--model", default=None, help="Override the model id for the chosen provider.")
    parser.add_argument(
        "--offline", action="store_true", help="Alias for --llm offline (no API key, no cost)."
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also run the LLM-judged RAGAS/deepeval metrics (needs OPENAI/ANTHROPIC key; costs money).",
    )
    parser.add_argument(
        "--scenario",
        choices=["standard", "hard"],
        default="standard",
        help="'standard' = small clean corpus; 'hard' = adds distractors and "
        "retrieval-hard / trap questions that stress the self-correction loop.",
    )
    parser.add_argument(
        "--naive-baseline",
        action="store_true",
        help="Compare against a NAIVE RAG baseline (no grounding prompt) instead "
        "of the careful grounded one -- the honest ablation for the critic's value.",
    )
    args = parser.parse_args()
    provider = "offline" if args.offline else args.llm

    if args.scenario == "hard":
        from .hard import HARD_DOCS as docs  # noqa: PLC0415
        from .hard import HARD_QUESTIONS as questions  # noqa: PLC0415
    else:
        docs, questions = DOCS, QUESTIONS

    # --- shared retriever: real Chroma vector store, free local embeddings ---
    from ragloop.retrieval.chroma_retriever import ChromaRetriever  # noqa: PLC0415

    retriever = ChromaRetriever(collection=f"ragloop_evals_{args.scenario}")
    retriever.add(docs)
    questions_by_text = {q["question"]: q for q in questions}
    n_unans = sum(1 for q in questions if not q.get("answerable", True))
    print(
        f"[runner] scenario={args.scenario}: loaded {len(docs)} chunks, "
        f"{len(questions)} questions ({n_unans} out-of-corpus) into Chroma"
    )

    inner_llm = _build_llm(provider, args.model)
    tracing_llm = TracingLLM(inner_llm)

    # --- baseline (grounded by default, or naive for the ablation) ---
    baseline_cls = NaiveRAG if args.naive_baseline else BaselineRAG
    baseline_pipeline = baseline_cls(retriever=retriever, llm=tracing_llm, k=5)
    print(f"[runner] running baseline ({baseline_cls.__name__}) ...")
    baseline_results = run_pipeline(
        "baseline", baseline_pipeline.ask, questions, tracing_llm, retriever
    )

    # --- ragloop ---
    loop = RagLoop(Deps(retriever=retriever, llm=tracing_llm, k=5), max_attempts=2)
    print("[runner] running ragloop ...")
    ragloop_results = run_pipeline(
        "ragloop", loop.ask, questions, tracing_llm, retriever
    )

    # --- scoring (deterministic: always, no API cost) ---
    print("[runner] scoring citation accuracy ...")
    _score_citation_accuracy(baseline_results)
    _score_citation_accuracy(ragloop_results)

    if args.judge:
        print("[runner] scoring with ragas (LLM judge) ...")
        _score_with_ragas(baseline_results)
        _score_with_ragas(ragloop_results)
        print("[runner] scoring with deepeval ...")
        _score_with_deepeval(baseline_results)
        _score_with_deepeval(ragloop_results)

    # --- to dicts, then deterministic label-based metrics ---
    base_dicts = [dataclasses.asdict(r) for r in baseline_results]
    rag_dicts = [dataclasses.asdict(r) for r in ragloop_results]
    print("[runner] scoring deterministic metrics (recall@k, decline, similarity) ...")
    score_deterministic(base_dicts, questions_by_text)
    score_deterministic(rag_dicts, questions_by_text)

    # --- persist ---
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(base_dicts + rag_dicts, indent=2))
    print(f"[runner] results written to {out_path}")

    # --- report ---
    report.print_table(base_dicts, rag_dicts)


if __name__ == "__main__":
    main()
