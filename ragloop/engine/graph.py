"""Assemble the nodes into a LangGraph state machine.

    plan -> retrieve -> fuse -> generate -> critique --(grounded?)--> END
                ^------------------------------------------(retry)----'

The single back-edge from ``critique`` to ``retrieve`` is the whole point: it
is what classic linear RAG cannot express and what makes this self-correcting.
"""
from __future__ import annotations

from functools import partial
from typing import Any

from .nodes import Deps, critique, fuse, generate, plan, retrieve, route_after_critic
from .state import GraphState


def build_graph(deps: Deps):
    from langgraph.graph import END, StateGraph

    g = StateGraph(GraphState)
    g.add_node("plan", partial(plan, deps=deps))
    g.add_node("retrieve", partial(retrieve, deps=deps))
    g.add_node("fuse", partial(fuse, deps=deps))
    g.add_node("generate", partial(generate, deps=deps))
    g.add_node("critique", partial(critique, deps=deps))

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "fuse")
    g.add_edge("fuse", "generate")
    g.add_edge("generate", "critique")
    g.add_conditional_edges(
        "critique",
        route_after_critic,
        {"retry": "retrieve", "done": END},
    )
    return g.compile()


class RagLoop:
    """High-level entry point. Construct once, call :meth:`ask` repeatedly."""

    def __init__(self, deps: Deps, max_attempts: int = 2) -> None:
        self.deps = deps
        self.max_attempts = max_attempts
        self._graph = build_graph(deps)

    def ask(self, query: str) -> dict[str, Any]:
        initial: GraphState = {
            "query": query,
            "attempts": 0,
            "max_attempts": self.max_attempts,
        }
        final = self._graph.invoke(initial)
        return {
            "answer": final.get("answer", ""),
            "grounded": final.get("grade", {}).get("grounded"),
            "attempts": final.get("attempts"),
            "sources": [d["id"] for d in final.get("retrieved", [])],
        }
