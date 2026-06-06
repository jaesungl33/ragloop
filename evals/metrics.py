"""Deterministic, label-based eval metrics — no LLM judge, no API cost.

These score against the ground-truth labels already in ``corpus.py``
(``relevant_ids``, ``ground_truth``, ``answerable``), so they are objective and
reproducible by anyone with zero API keys. For a self-correcting RAG system the
most telling numbers are:

- **retrieval_recall**   did retrieval surface the gold chunk(s)? (answerable Qs)
- **declined**           did the system refuse to answer?
- **answer_similarity**  cosine similarity of the answer to the ground truth,
                         using a free local embedding model (Chroma's MiniLM).

Aggregated, ``declined`` becomes the headline pair: **hallucination resistance**
(fraction of *unanswerable* questions correctly declined) and **false-decline
rate** (fraction of *answerable* questions wrongly declined).
"""
from __future__ import annotations

import re
from collections.abc import Sequence

# A decline means the system says it *can't answer from the sources* — not an
# incidental "X is not covered" inside an otherwise-complete answer. So we
# require the refusal to be about the sources/information or the inability to
# determine an answer, which avoids false positives on real answers.
_DECLINE_PATTERNS = re.compile(
    r"(?:"
    r"(?:sources?|information|context|provided (?:text|documents?|materials?|information))"
    r"\s+(?:do(?:es)? not|don'?t|cannot)\s+"
    r"(?:contain|mention|specify|include|provide|cover|address|indicate|state|say)"
    r"|no (?:information|mention|reference|details?)\s+(?:about|on|regarding|for|of)"
    r"|not (?:enough|sufficient) information"
    r"|cannot (?:determine|find|answer|provide|tell|confirm)"
    r"|can'?t (?:determine|find|answer|tell)"
    r"|unable to (?:determine|answer|find|provide)"
    r"|(?:do(?:es)? not|don'?t) have (?:enough |sufficient )?(?:information|details)"
    r"|is not (?:specified|mentioned|stated|provided) in the (?:sources?|provided|context|information)"
    r")",
    re.IGNORECASE,
)


def declined(answer: str) -> bool:
    """Heuristic: did the answer say it can't answer from the sources?

    Detects genuine refusals ("the sources do not specify...", "I cannot
    determine...") while ignoring incidental negations inside real answers
    (e.g. "accidents are not covered").
    """
    return bool(_DECLINE_PATTERNS.search(answer or ""))


def retrieval_recall(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float | None:
    """Fraction of gold chunks that appear in the retrieved set. None if no labels."""
    rel = set(relevant_ids)
    if not rel:
        return None
    return len(rel & set(retrieved_ids)) / len(rel)


# --- free local embedding similarity ---------------------------------------

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from chromadb.utils import embedding_functions

        _EMBEDDER = embedding_functions.DefaultEmbeddingFunction()
    return _EMBEDDER


def answer_similarity(answer: str, ground_truth: str) -> float | None:
    """Cosine similarity between answer and ground truth via a local MiniLM model.

    Returns a value in roughly [0, 1]. Falls back to ``None`` if embeddings are
    unavailable (e.g. Chroma not installed).
    """
    if not answer or not ground_truth:
        return None
    try:
        import numpy as np

        vecs = _get_embedder()([answer, ground_truth])
        a, b = np.asarray(vecs[0], dtype=float), np.asarray(vecs[1], dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)
    except Exception:  # pragma: no cover - optional dependency / model load
        return None


def score_deterministic(results: list[dict], questions_by_text: dict) -> None:
    """Mutate each result dict in place, adding deterministic metric fields."""
    for r in results:
        q = questions_by_text.get(r["question"], {})
        answerable = q.get("answerable", True)
        r["answerable"] = answerable
        r["retrieval_recall"] = retrieval_recall(r.get("sources", []), q.get("relevant_ids", []))
        r["declined"] = declined(r.get("answer", ""))
        r["answer_similarity"] = answer_similarity(r.get("answer", ""), r.get("ground_truth", ""))
