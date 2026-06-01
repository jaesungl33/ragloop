"""The five nodes of the agentic loop.

Each node is a plain function of (state, deps) -> partial state update. Keeping
them free of framework glue makes them unit-testable in isolation and easy to
reason about. ``deps`` carries the retriever and the LLM provider so nothing is
hardwired.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from ..llm.base import LLMProvider
from ..retrieval.base import Document, Retriever
from .state import GraphState


@dataclass
class Deps:
    retriever: Retriever
    llm: LLMProvider
    k: int = 5


def _parse_json(text: str, default: Any) -> Any:
    """Best-effort JSON parse that tolerates code fences and stray prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    start = cleaned.find("{")
    arr = cleaned.find("[")
    if arr != -1 and (start == -1 or arr < start):
        start = arr
    if start == -1:
        return default
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    try:
        return json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return default


# --- Nodes -----------------------------------------------------------------

def plan(state: GraphState, deps: Deps) -> Dict[str, Any]:
    system = (
        "You decompose a user question into 1-4 concrete retrieval sub-tasks. "
        "Simple questions get one sub-task. Respond ONLY with a JSON array of "
        "short search strings, nothing else."
    )
    raw = deps.llm.complete(system, state["query"])
    subtasks = _parse_json(raw, default=[state["query"]])
    if not isinstance(subtasks, list) or not subtasks:
        subtasks = [state["query"]]
    return {"subtasks": [str(s) for s in subtasks][:4]}


def retrieve(state: GraphState, deps: Deps) -> Dict[str, Any]:
    """Agentic step: pick a strategy per sub-task, blend lexical + semantic."""
    seen: Dict[str, Document] = {}
    feedback = state.get("feedback", "")
    # Always search the original question alongside the planner's sub-tasks, so
    # decomposition can only *add* recall, never lose the chunk a direct search
    # would have found. (Benchmarks showed weak planners hurting simple queries.)
    tasks = list(state.get("subtasks") or [])
    if state["query"] not in tasks:
        tasks.insert(0, state["query"])
    for task in tasks:
        # Blend lexical (exact terms) and semantic (meaning) for each task.
        lexical = deps.retriever.keyword_search(task, k=deps.k)
        semantic = deps.retriever.semantic_search(task, k=deps.k)
        for doc in lexical + semantic:
            if doc.id not in seen:
                seen[doc.id] = doc
    docs = list(seen.values())
    # If the critic asked for more, widen by reading full chunks of top hits.
    if feedback:
        for doc in docs[: deps.k]:
            full = deps.retriever.get_chunk(doc.id)
            if full:
                seen[full.id] = full
    return {"retrieved": [d.to_dict() for d in seen.values()]}


def fuse(state: GraphState, deps: Deps) -> Dict[str, Any]:
    """Deduplicate, rank by score, and trim to a diverse top set (MMR-lite)."""
    docs = state.get("retrieved", [])
    # Sort by score desc (None last).
    docs = sorted(docs, key=lambda d: (d.get("score") is not None, d.get("score") or 0), reverse=True)
    # Simple diversity pass: drop near-identical text prefixes.
    kept: List[Dict[str, Any]] = []
    prefixes: set[str] = set()
    for d in docs:
        prefix = (d.get("text") or "")[:120]
        if prefix in prefixes:
            continue
        prefixes.add(prefix)
        kept.append(d)
        if len(kept) >= deps.k:
            break
    return {"retrieved": kept}


def generate(state: GraphState, deps: Deps) -> Dict[str, Any]:
    context_blocks = []
    for d in state.get("retrieved", []):
        context_blocks.append(f"[source:{d['id']}] {d['text']}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"
    system = (
        "Answer the question using ONLY the provided sources. Cite each claim "
        "inline as [source:ID]. If the sources do not contain the answer, say "
        "so plainly. Do not use outside knowledge."
    )
    prompt = f"Sources:\n{context}\n\nQuestion: {state['query']}"
    answer = deps.llm.complete(system, prompt)
    return {"answer": answer}


def critique(state: GraphState, deps: Deps) -> Dict[str, Any]:
    context = "\n\n".join(
        f"[source:{d['id']}] {d['text']}" for d in state.get("retrieved", [])
    )
    system = (
        "You grade whether an answer is fully grounded in the given sources. "
        'Respond ONLY with JSON: {"grounded": true|false, "reason": "<short>"}. '
        "Mark grounded=false if any claim lacks support or a citation is wrong."
    )
    prompt = (
        f"Sources:\n{context}\n\nQuestion: {state['query']}\n\nAnswer:\n{state.get('answer', '')}"
    )
    raw = deps.llm.complete(system, prompt)
    grade = _parse_json(raw, default={"grounded": True, "reason": "unparseable grade; accepted"})
    attempts = state.get("attempts", 0) + 1
    feedback = "" if grade.get("grounded") else grade.get("reason", "answer not grounded")
    return {"grade": grade, "attempts": attempts, "feedback": feedback}


def route_after_critic(state: GraphState) -> str:
    """Conditional edge: loop back to retrieval, or finish."""
    grounded = state.get("grade", {}).get("grounded", True)
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 2)
    if grounded or attempts >= max_attempts:
        return "done"
    return "retry"
