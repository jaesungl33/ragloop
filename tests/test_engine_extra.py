"""Tests for the parts that the happy-path suite doesn't cover:
the self-correcting retry loop (the project's whole thesis), the JSON parser,
fusion dedup, and the critic routing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from quickstart import InMemoryRetriever  # noqa: E402

from ragloop import Deps, Document, LLMProvider, RagLoop, Retriever  # noqa: E402
from ragloop.engine.nodes import _parse_json, fuse, retrieve, route_after_critic  # noqa: E402


def _retriever():
    r = InMemoryRetriever()
    r.add([
        Document("policy:0", "Customers may request a refund within 30 days of purchase."),
        Document("policy:1", "Shipping is free on orders over $50."),
    ])
    return r


class _ScriptedLLM(LLMProvider):
    """Critic returns the queued grades in order; generator always cites a source."""

    def __init__(self, grades):
        self._grades = list(grades)
        self.grade_calls = 0
        self.retrieve_rounds = 0

    def complete(self, system: str, prompt: str) -> str:
        if "decompose" in system:
            return '["refund policy"]'
        if "grade" in system:
            g = self._grades[min(self.grade_calls, len(self._grades) - 1)]
            self.grade_calls += 1
            return g
        return "Refunds are accepted within 30 days [source:policy:0]."


def test_retry_loop_recovers_then_stops():
    """Ungrounded on attempt 1 -> back-edge -> grounded on attempt 2."""
    llm = _ScriptedLLM([
        '{"grounded": false, "reason": "needs more evidence"}',
        '{"grounded": true, "reason": "supported"}',
    ])
    loop = RagLoop(Deps(retriever=_retriever(), llm=llm, k=3), max_attempts=3)
    result = loop.ask("What is the refund window?")
    assert result["attempts"] == 2          # it retried exactly once
    assert result["grounded"] is True
    assert llm.grade_calls == 2             # critic ran twice = the loop closed


def test_retry_is_bounded_when_never_grounded():
    """Always-ungrounded must stop at max_attempts, not spin forever."""
    llm = _ScriptedLLM(['{"grounded": false, "reason": "still not grounded"}'])
    loop = RagLoop(Deps(retriever=_retriever(), llm=llm, k=3), max_attempts=2)
    result = loop.ask("What is the refund window?")
    assert result["attempts"] == 2          # capped
    assert result["grounded"] is False      # honestly reports it never grounded


class _UnparseableGradeLLM(LLMProvider):
    """Critic always returns non-JSON, exercising the fail-open/closed path."""

    def complete(self, system: str, prompt: str) -> str:
        if "decompose" in system:
            return '["refund policy"]'
        if "grade" in system:
            return "the answer looks fine to me"  # not JSON
        return "Refunds are accepted within 30 days [source:policy:0]."


def test_critic_fail_open_accepts_unparseable_grade():
    loop = RagLoop(Deps(retriever=_retriever(), llm=_UnparseableGradeLLM(), k=3), max_attempts=2)
    assert loop.ask("refund?")["grounded"] is True   # fail_closed defaults to False


def test_critic_fail_closed_rejects_unparseable_grade():
    deps = Deps(retriever=_retriever(), llm=_UnparseableGradeLLM(), k=3, fail_closed=True)
    result = RagLoop(deps, max_attempts=2).ask("refund?")
    assert result["grounded"] is False               # honest failure, not silent accept
    assert result["attempts"] == 2                   # retried, then stopped at budget


def test_parse_json_tolerates_fences_and_prose():
    assert _parse_json('```json\n{"grounded": true}\n```', {}) == {"grounded": True}
    assert _parse_json('Sure! ["a", "b"] hope that helps', None) == ["a", "b"]
    assert _parse_json("not json at all", {"fallback": 1}) == {"fallback": 1}


def test_fuse_deduplicates_and_ranks_by_score():
    state = {"retrieved": [
        {"id": "a", "text": "same prefix text here", "score": 0.2},
        {"id": "b", "text": "same prefix text here", "score": 0.9},  # dup prefix
        {"id": "c", "text": "a clearly different chunk", "score": 0.5},
    ]}

    class _D:
        k = 5

    out = fuse(state, _D())["retrieved"]
    ids = [d["id"] for d in out]
    # near-identical prefixes collapse to one, and higher score ranks first
    assert "a" in ids or "b" in ids
    assert len([i for i in ids if i in {"a", "b"}]) == 1
    assert ids[0] == "b"  # highest score first


def test_retrieve_always_includes_original_query():
    """Regression: decomposition must not drop the raw-query search (recall safety)."""

    class _Recorder(Retriever):
        def __init__(self):
            self.searched = []

        def add(self, docs):
            pass

        def semantic_search(self, q, k=5):
            self.searched.append(q)
            return []

        def keyword_search(self, q, k=5):
            self.searched.append(q)
            return []

        def get_chunk(self, doc_id):
            return None

    rec = _Recorder()
    state = {"query": "What is the refund window?", "subtasks": ["refund policy", "returns"]}
    retrieve(state, Deps(retriever=rec, llm=None, k=3))  # type: ignore[arg-type]
    assert "What is the refund window?" in rec.searched   # raw query searched
    assert "refund policy" in rec.searched                # sub-tasks still searched


def test_route_after_critic():
    assert route_after_critic({"grade": {"grounded": True}, "attempts": 1, "max_attempts": 3}) == "done"
    assert route_after_critic({"grade": {"grounded": False}, "attempts": 1, "max_attempts": 3}) == "retry"
    # budget exhausted -> stop even if ungrounded
    assert route_after_critic({"grade": {"grounded": False}, "attempts": 3, "max_attempts": 3}) == "done"
