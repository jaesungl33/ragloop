"""Shared state that travels between graph nodes.

LangGraph passes one of these dicts through every node. Each node reads what it
needs and returns a partial update. The ``attempts`` / ``max_attempts`` pair is
what bounds the self-correction loop so it can't run forever.
"""
from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    query: str
    subtasks: list[str]
    retrieved: list[dict[str, Any]]  # serialized Document dicts
    answer: str
    grade: dict[str, Any]            # {"grounded": bool, "reason": str}
    feedback: str                    # critic note fed back into retrieval
    attempts: int
    max_attempts: int
