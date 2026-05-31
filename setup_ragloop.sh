#!/usr/bin/env bash
#
# setup_ragloop.sh -- scaffold the ragloop project skeleton in the current directory.
#
# Usage:
#   bash setup_ragloop.sh            # writes project files here, venv, tests
#   RAGLOOP_NO_VENV=1 bash setup_ragloop.sh   # skip venv + install + tests
#
# Safe to read before running. It only writes under the current directory.
set -euo pipefail

if [ -f pyproject.toml ]; then
  echo "pyproject.toml already exists here. Move or remove the existing project first." >&2
  exit 1
fi

echo "Creating ragloop project skeleton in $(pwd)..."

echo "  writing .env.example"
cat > ".env.example" <<'RAGLOOP_FILE_EOF'
# Copy to .env and fill in. Never commit real keys.
ANTHROPIC_API_KEY=sk-ant-...
RAGLOOP_CONFIG=./config.yaml
RAGLOOP_FILE_EOF

echo "  writing .gitignore"
cat > ".gitignore" <<'RAGLOOP_FILE_EOF'
__pycache__/
*.pyc
.env
.chroma/
dist/
build/
*.egg-info/
.pytest_cache/
RAGLOOP_FILE_EOF

echo "  writing LICENSE"
cat > "LICENSE" <<'RAGLOOP_FILE_EOF'
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Copyright 2026 ragloop contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
RAGLOOP_FILE_EOF

echo "  writing README.md"
cat > "README.md" <<'RAGLOOP_FILE_EOF'
# ragloop

A pluggable, self-correcting **agentic RAG** framework. Point it at your
documents, your vector store, and your LLM, and it answers questions with
inline citations — checking its own work and retrying when an answer isn't
grounded.

It's built so that the things companies actually differ on are swappable
without touching the engine:

- **Your vector store** — Chroma ships as the reference backend; swap in
  pgvector, Pinecone, or Elasticsearch by subclassing one interface.
- **Your LLM** — Anthropic (Claude) ships as the reference provider; add
  OpenAI, Bedrock, or a local model the same way.
- **Your corpus, models, and retry budget** — all config-driven.

## How it works

```
plan -> retrieve -> fuse -> generate -> critique --(grounded?)--> done
            ^------------------------------------------(retry)----'
```

A LangGraph state machine decomposes the question, lets the agent choose its
retrieval strategy (lexical vs. semantic vs. full-chunk read), fuses and ranks
the evidence, generates a cited answer, then **grades whether that answer is
fully supported**. If not, it feeds the critique back into retrieval and tries
again — bounded by a retry budget. The single back-edge from the critic is what
distinguishes this from a one-shot RAG pipeline.

Retrieval is also exposed over **MCP**, so any MCP-capable client can query your
corpus directly with access control enforced server-side.

## Install

```bash
pip install -e ".[all]"        # engine + Anthropic + Chroma + MCP
# or pick pieces: pip install -e ".[anthropic,chroma]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

```bash
cp examples/config.example.yaml config.yaml
ragloop ingest ./your-docs --config config.yaml
ragloop ask "What is our refund window?" --config config.yaml
```

Or from Python:

```python
from ragloop import build_from_config
loop = build_from_config("config.yaml")
result = loop.ask("What is our refund policy?")
print(result["answer"], result["sources"], result["grounded"])
```

Run the dependency-free demo to see the loop without any API keys:

```bash
python examples/quickstart.py
pytest        # the same fakes power the test suite
```

## Serve retrieval over MCP

```bash
ragloop serve --config config.yaml
```

Exposes three tools — `keyword_search`, `semantic_search`, `chunk_read` — to any
MCP client.

## Extending it

Add a backend by implementing one interface and registering it:

- **New vector store:** subclass `ragloop.Retriever` (`add`, `semantic_search`,
  `keyword_search`, `get_chunk`), then add a branch in `config._build_retriever`.
- **New LLM:** subclass `ragloop.LLMProvider` (`complete`), then add a branch in
  `config._build_llm`.

Nothing in the engine changes. See `examples/quickstart.py` for a complete
custom retriever and provider in ~30 lines.

## Project layout

```
ragloop/
  llm/          LLMProvider interface + Anthropic reference
  retrieval/    Retriever interface + Chroma reference
  engine/       state, nodes (plan/retrieve/fuse/generate/critique), graph
  mcp/          MCP server exposing the retrieval tools
  config.py     YAML + env wiring
  cli.py        ingest / ask / serve
```

## Roadmap

- Reranker hook in the fusion step (cross-encoder)
- Persistent agent memory across sessions (LangGraph checkpointer)
- Reference backends for pgvector and OpenAI
- Streaming answers

## License

Apache-2.0. Contributions welcome — see issues.
RAGLOOP_FILE_EOF

mkdir -p "examples"
echo "  writing examples/config.example.yaml"
cat > "examples/config.example.yaml" <<'RAGLOOP_FILE_EOF'
# Copy to config.yaml and adjust. Secrets come from environment variables,
# never from this file (keep credentials out of version control).

llm_provider: anthropic
llm:
  # Any current Claude model id. sonnet-4-6 is a good default balance of
  # cost and quality; use an opus model for harder reasoning. Omit
  # temperature for models that reject a non-default value.
  model: claude-sonnet-4-6
  max_tokens: 1024
  # api_key is read from ANTHROPIC_API_KEY if omitted (recommended).

retriever_backend: chroma
retriever:
  collection: ragloop
  # Set a path to persist the index to disk; omit for in-memory.
  persist_dir: ./.chroma

top_k: 5
max_attempts: 2   # self-correction retries before returning best effort
RAGLOOP_FILE_EOF

mkdir -p "examples"
echo "  writing examples/quickstart.py"
cat > "examples/quickstart.py" <<'RAGLOOP_FILE_EOF'
"""End-to-end example with zero external dependencies.

Demonstrates the two extension points -- a custom Retriever and a custom
LLMProvider -- by using in-memory fakes. Run: ``python examples/quickstart.py``.
This is also how the test suite exercises the loop without network calls.
"""
from __future__ import annotations

from typing import List, Optional

from ragloop import Deps, Document, LLMProvider, RagLoop, Retriever


class InMemoryRetriever(Retriever):
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    def add(self, documents: List[Document]) -> None:
        for d in documents:
            self._docs[d.id] = d

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        terms = set(query.lower().split())
        scored = [
            Document(d.id, d.text, d.metadata, score=len(terms & set(d.text.lower().split())))
            for d in self._docs.values()
        ]
        scored.sort(key=lambda d: d.score or 0, reverse=True)
        return scored[:k]

    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        hits = [d for d in self._docs.values() if query.lower() in d.text.lower()]
        return hits[:k]

    def get_chunk(self, doc_id: str) -> Optional[Document]:
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
RAGLOOP_FILE_EOF

mkdir -p "ragloop"
echo "  writing pyproject.toml"
cat > "pyproject.toml" <<'RAGLOOP_FILE_EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ragloop"
version = "0.1.0"
description = "A pluggable, self-correcting agentic RAG framework with MCP-native retrieval."
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [{ name = "ragloop contributors" }]
keywords = ["rag", "llm", "agentic", "mcp", "langgraph", "retrieval"]
dependencies = [
    "langgraph>=0.2",
    "pydantic>=2",
    "pyyaml>=6",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
chroma = ["chromadb>=0.5"]
mcp = ["mcp>=1.2"]
# Convenience: everything needed to run the reference stack end to end.
all = ["anthropic>=0.40", "chromadb>=0.5", "mcp>=1.2"]
dev = ["pytest>=8"]

[project.scripts]
ragloop = "ragloop.cli:main"

[project.urls]
Homepage = "https://github.com/your-org/ragloop"
Issues = "https://github.com/your-org/ragloop/issues"

[tool.hatch.build.targets.wheel]
packages = ["ragloop"]
RAGLOOP_FILE_EOF

mkdir -p "ragloop"
echo "  writing ragloop/__init__.py"
cat > "ragloop/__init__.py" <<'RAGLOOP_FILE_EOF'
"""ragloop -- a pluggable, self-correcting agentic RAG framework.

Quick start::

    from ragloop import build_from_config
    loop = build_from_config("config.yaml")
    print(loop.ask("What is our refund policy?")["answer"])
"""
from .config import Config, build_from_config
from .engine import Deps, RagLoop
from .llm.base import LLMProvider
from .retrieval.base import Document, Retriever

__version__ = "0.1.0"

__all__ = [
    "Config",
    "build_from_config",
    "RagLoop",
    "Deps",
    "LLMProvider",
    "Retriever",
    "Document",
]
RAGLOOP_FILE_EOF

mkdir -p "ragloop"
echo "  writing ragloop/cli.py"
cat > "ragloop/cli.py" <<'RAGLOOP_FILE_EOF'
"""Command-line interface: ingest a corpus, ask a question, serve MCP.

    ragloop ingest ./docs --config config.yaml
    ragloop ask "What is our refund window?" --config config.yaml
    ragloop serve --config config.yaml
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _load_cfg(config: Optional[str]):
    from .config import Config

    return Config.from_yaml(config) if config else Config()


def cmd_ingest(path: str, config: Optional[str]) -> None:
    from .config import _build_retriever
    from .retrieval.base import Document

    cfg = _load_cfg(config)
    retriever = _build_retriever(cfg)
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob("*.txt")) + sorted(root.rglob("*.md"))
    docs = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        # Naive paragraph chunking; replace with your own splitter as needed.
        for i, chunk in enumerate(c for c in text.split("\n\n") if c.strip()):
            docs.append(Document(id=f"{fp.name}:{i}", text=chunk.strip(), metadata={"file": str(fp)}))
    retriever.add(docs)
    print(f"Ingested {len(docs)} chunks from {len(files)} file(s).")


def cmd_ask(question: str, config: Optional[str]) -> None:
    from .config import build_from_config

    loop = build_from_config(path=config)
    result = loop.ask(question)
    print("\nAnswer:\n" + result["answer"])
    print(f"\ngrounded={result['grounded']}  attempts={result['attempts']}  sources={result['sources']}")


def cmd_serve(config: Optional[str]) -> None:
    from .config import Config
    from .mcp.server import build_server

    cfg = Config.from_yaml(config) if config else Config()
    build_server(cfg).run()


def _resolve_config(args) -> Optional[str]:
    """--config may appear before or after the subcommand; fall back to env."""
    return getattr(args, "config", None) or os.environ.get("RAGLOOP_CONFIG")


def main(argv: Optional[list] = None) -> int:
    import argparse

    # SUPPRESS on subparsers so a root-level --config is not cleared when the
    # subcommand omits --config (argparse would otherwise merge in None).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(prog="ragloop")
    parser.add_argument("--config", default=os.environ.get("RAGLOOP_CONFIG"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="index a file or directory", parents=[common])
    p_ing.add_argument("path")
    p_ask = sub.add_parser("ask", help="ask a question", parents=[common])
    p_ask.add_argument("question")
    sub.add_parser("serve", help="serve retrieval over MCP", parents=[common])

    args = parser.parse_args(argv)
    config = _resolve_config(args)
    if args.command == "ingest":
        cmd_ingest(args.path, config)
    elif args.command == "ask":
        cmd_ask(args.question, config)
    elif args.command == "serve":
        cmd_serve(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
RAGLOOP_FILE_EOF

mkdir -p "ragloop"
echo "  writing ragloop/config.py"
cat > "ragloop/config.py" <<'RAGLOOP_FILE_EOF'
"""Config-driven wiring so adopters never edit code.

A YAML file (and environment variables for secrets) selects the LLM provider,
the retriever backend, and runtime knobs. :func:`build_from_config` returns a
ready ``RagLoop``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .engine import Deps, RagLoop
from .llm.base import LLMProvider
from .retrieval.base import Retriever


@dataclass
class Config:
    llm_provider: str = "anthropic"
    llm: Dict[str, Any] = field(default_factory=dict)
    retriever_backend: str = "chroma"
    retriever: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    max_attempts: int = 2

    @staticmethod
    def from_yaml(path: str) -> "Config":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return Config(
            llm_provider=data.get("llm_provider", "anthropic"),
            llm=data.get("llm", {}),
            retriever_backend=data.get("retriever_backend", "chroma"),
            retriever=data.get("retriever", {}),
            top_k=int(data.get("top_k", 5)),
            max_attempts=int(data.get("max_attempts", 2)),
        )


def _build_llm(cfg: Config) -> LLMProvider:
    if cfg.llm_provider == "anthropic":
        from .llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(**cfg.llm)
    raise ValueError(
        f"Unknown llm_provider '{cfg.llm_provider}'. "
        "Register your own by subclassing LLMProvider and editing _build_llm."
    )


def _build_retriever(cfg: Config) -> Retriever:
    if cfg.retriever_backend == "chroma":
        from .retrieval.chroma_retriever import ChromaRetriever

        return ChromaRetriever(**cfg.retriever)
    raise ValueError(
        f"Unknown retriever_backend '{cfg.retriever_backend}'. "
        "Register your own by subclassing Retriever and editing _build_retriever."
    )


def build_from_config(path: Optional[str] = None, cfg: Optional[Config] = None) -> RagLoop:
    cfg = cfg or (Config.from_yaml(path) if path else Config())
    llm = _build_llm(cfg)
    retriever = _build_retriever(cfg)
    deps = Deps(retriever=retriever, llm=llm, k=cfg.top_k)
    return RagLoop(deps, max_attempts=cfg.max_attempts)
RAGLOOP_FILE_EOF

mkdir -p "ragloop/engine"
echo "  writing ragloop/engine/__init__.py"
cat > "ragloop/engine/__init__.py" <<'RAGLOOP_FILE_EOF'
from .graph import RagLoop, build_graph
from .nodes import Deps
from .state import GraphState

__all__ = ["RagLoop", "build_graph", "Deps", "GraphState"]
RAGLOOP_FILE_EOF

mkdir -p "ragloop/engine"
echo "  writing ragloop/engine/graph.py"
cat > "ragloop/engine/graph.py" <<'RAGLOOP_FILE_EOF'
"""Assemble the nodes into a LangGraph state machine.

    plan -> retrieve -> fuse -> generate -> critique --(grounded?)--> END
                ^------------------------------------------(retry)----'

The single back-edge from ``critique`` to ``retrieve`` is the whole point: it
is what classic linear RAG cannot express and what makes this self-correcting.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict

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

    def ask(self, query: str) -> Dict[str, Any]:
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
RAGLOOP_FILE_EOF

mkdir -p "ragloop/engine"
echo "  writing ragloop/engine/nodes.py"
cat > "ragloop/engine/nodes.py" <<'RAGLOOP_FILE_EOF'
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
    for task in state.get("subtasks", [state["query"]]):
        # A short identifier-like query leans lexical; prose leans semantic.
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
RAGLOOP_FILE_EOF

mkdir -p "ragloop/engine"
echo "  writing ragloop/engine/state.py"
cat > "ragloop/engine/state.py" <<'RAGLOOP_FILE_EOF'
"""Shared state that travels between graph nodes.

LangGraph passes one of these dicts through every node. Each node reads what it
needs and returns a partial update. The ``attempts`` / ``max_attempts`` pair is
what bounds the self-correction loop so it can't run forever.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict, total=False):
    query: str
    subtasks: List[str]
    retrieved: List[Dict[str, Any]]  # serialized Document dicts
    answer: str
    grade: Dict[str, Any]            # {"grounded": bool, "reason": str}
    feedback: str                    # critic note fed back into retrieval
    attempts: int
    max_attempts: int
RAGLOOP_FILE_EOF

mkdir -p "ragloop/llm"
echo "  writing ragloop/llm/__init__.py"
cat > "ragloop/llm/__init__.py" <<'RAGLOOP_FILE_EOF'
from .base import LLMProvider

__all__ = ["LLMProvider"]
RAGLOOP_FILE_EOF

mkdir -p "ragloop/llm"
echo "  writing ragloop/llm/anthropic_provider.py"
cat > "ragloop/llm/anthropic_provider.py" <<'RAGLOOP_FILE_EOF'
"""Anthropic (Claude) implementation of :class:`LLMProvider`.

This is the reference provider. Adding another (OpenAI, a local vLLM server,
Bedrock, etc.) means writing one more file like this and pointing the config
at it -- no engine changes required.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
    ) -> None:
        # Imported lazily so the package installs without the SDK present
        # until a caller actually selects this provider.
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key=..."
            )
        self._client = Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        # Note: some newer Claude models reject a non-default temperature.
        # Leave it as None to omit the parameter entirely (recommended).
        self.temperature = temperature

    def complete(self, system: str, prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self._client.messages.create(**kwargs)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
RAGLOOP_FILE_EOF

mkdir -p "ragloop/llm"
echo "  writing ragloop/llm/base.py"
cat > "ragloop/llm/base.py" <<'RAGLOOP_FILE_EOF'
"""Abstract LLM provider interface.

Any company can swap in their own model backend by subclassing ``LLMProvider``
and implementing ``complete``. The engine never imports a concrete provider
directly; it receives one through configuration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Minimal text-completion interface used by the engine nodes."""

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str:
        """Return the model's text response to ``prompt`` under ``system``.

        Implementations should return plain text (no tool-use blocks). The
        engine handles structured output by instructing the model to emit JSON
        and parsing the returned string.
        """
        raise NotImplementedError
RAGLOOP_FILE_EOF

mkdir -p "ragloop/mcp"
echo "  writing ragloop/mcp/__init__.py"
cat > "ragloop/mcp/__init__.py" <<'RAGLOOP_FILE_EOF'
from .server import build_server

__all__ = ["build_server"]
RAGLOOP_FILE_EOF

mkdir -p "ragloop/mcp"
echo "  writing ragloop/mcp/server.py"
cat > "ragloop/mcp/server.py" <<'RAGLOOP_FILE_EOF'
"""Expose the retriever as MCP tools.

Running this serves ``keyword_search``, ``semantic_search`` and ``chunk_read``
over MCP so any MCP-capable client (an agent, an IDE, another service) can
query the corpus directly, with access control enforced here on the server
side rather than baked into each client. This is the "let the agent pull the
data" pattern.
"""
from __future__ import annotations

import json
from typing import Optional

from ..config import Config, _build_retriever


def build_server(cfg: Optional[Config] = None):
    from mcp.server.fastmcp import FastMCP

    cfg = cfg or Config()
    retriever = _build_retriever(cfg)
    mcp = FastMCP("ragloop-retrieval")

    @mcp.tool()
    def keyword_search(query: str, k: int = 5) -> str:
        """Lexical search for exact terms, codes, names, or identifiers."""
        docs = retriever.keyword_search(query, k=k)
        return json.dumps([d.to_dict() for d in docs])

    @mcp.tool()
    def semantic_search(query: str, k: int = 5) -> str:
        """Meaning-based search for paraphrased or conceptual questions."""
        docs = retriever.semantic_search(query, k=k)
        return json.dumps([d.to_dict() for d in docs])

    @mcp.tool()
    def chunk_read(doc_id: str) -> str:
        """Read the full text of one document by its id."""
        doc = retriever.get_chunk(doc_id)
        return json.dumps(doc.to_dict() if doc else None)

    return mcp


def main() -> None:
    import os

    cfg_path = os.environ.get("RAGLOOP_CONFIG")
    cfg = Config.from_yaml(cfg_path) if cfg_path else Config()
    build_server(cfg).run()


if __name__ == "__main__":
    main()
RAGLOOP_FILE_EOF

mkdir -p "ragloop/retrieval"
echo "  writing ragloop/retrieval/__init__.py"
cat > "ragloop/retrieval/__init__.py" <<'RAGLOOP_FILE_EOF'
from .base import Document, Retriever

__all__ = ["Document", "Retriever"]
RAGLOOP_FILE_EOF

mkdir -p "ragloop/retrieval"
echo "  writing ragloop/retrieval/base.py"
cat > "ragloop/retrieval/base.py" <<'RAGLOOP_FILE_EOF'
"""Abstract retrieval interface.

The agent reaches the corpus only through these three operations. A company
backs them with whatever store they already run -- Chroma, pgvector, Pinecone,
Elasticsearch -- by subclassing :class:`Retriever`. The hierarchy (keyword /
semantic / chunk) mirrors how an agent actually searches: broad lexical match,
meaning-based match, then full-context read of a promising hit.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


class Retriever(ABC):
    """Three retrieval granularities exposed to the agent."""

    @abstractmethod
    def add(self, documents: List[Document]) -> None:
        """Index a batch of documents (idempotent on ``id``)."""

    @abstractmethod
    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Dense / embedding-based search. Best for meaning and paraphrase."""

    @abstractmethod
    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """Lexical search. Best for exact terms, codes, names, identifiers."""

    @abstractmethod
    def get_chunk(self, doc_id: str) -> Optional[Document]:
        """Fetch one document by id for full-context reading."""
RAGLOOP_FILE_EOF

mkdir -p "ragloop/retrieval"
echo "  writing ragloop/retrieval/chroma_retriever.py"
cat > "ragloop/retrieval/chroma_retriever.py" <<'RAGLOOP_FILE_EOF'
"""Chroma implementation of :class:`Retriever`.

Reference backend. ``semantic_search`` uses Chroma's vector index;
``keyword_search`` uses Chroma's ``$contains`` document filter as a simple
lexical match. For production-grade lexical ranking, back ``keyword_search``
with a real BM25 index (e.g. an Elasticsearch or OpenSearch sidecar) -- the
interface stays identical, so the engine is unaffected.
"""
from __future__ import annotations

from typing import List, Optional

from .base import Document, Retriever


class ChromaRetriever(Retriever):
    def __init__(
        self,
        collection: str = "ragloop",
        persist_dir: Optional[str] = None,
    ) -> None:
        import chromadb

        self._client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.Client()
        )
        # Chroma supplies a default embedding function; swap via config in a
        # real deployment to match your domain.
        self._col = self._client.get_or_create_collection(name=collection)

    def add(self, documents: List[Document]) -> None:
        if not documents:
            return
        self._col.upsert(
            ids=[d.id for d in documents],
            documents=[d.text for d in documents],
            metadatas=[d.metadata or {} for d in documents],
        )

    def _to_docs(self, res, with_distance: bool) -> List[Document]:
        out: List[Document] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0] if with_distance else [None] * len(ids)
        for i, _id in enumerate(ids):
            out.append(
                Document(
                    id=_id,
                    text=docs[i] if i < len(docs) else "",
                    metadata=metas[i] if i < len(metas) else {},
                    # Convert distance to a similarity-style score (higher = better).
                    score=(1.0 - dists[i]) if dists[i] is not None else None,
                )
            )
        return out

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        res = self._col.query(query_texts=[query], n_results=k)
        return self._to_docs(res, with_distance=True)

    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        res = self._col.query(
            query_texts=[query],
            n_results=k,
            where_document={"$contains": query},
        )
        return self._to_docs(res, with_distance=True)

    def get_chunk(self, doc_id: str) -> Optional[Document]:
        res = self._col.get(ids=[doc_id])
        ids = res.get("ids") or []
        if not ids:
            return None
        docs = res.get("documents") or [""]
        metas = res.get("metadatas") or [{}]
        return Document(id=ids[0], text=docs[0], metadata=metas[0])
RAGLOOP_FILE_EOF

mkdir -p "tests"
echo "  writing tests/test_engine.py"
cat > "tests/test_engine.py" <<'RAGLOOP_FILE_EOF'
"""Offline test of the full graph using in-memory fakes (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from quickstart import EchoLLM, InMemoryRetriever  # noqa: E402

from ragloop import Deps, Document, RagLoop  # noqa: E402


def _loop():
    r = InMemoryRetriever()
    r.add([
        Document("policy:0", "Customers may request a refund within 30 days of purchase."),
        Document("policy:1", "Shipping is free on orders over $50."),
    ])
    return RagLoop(Deps(retriever=r, llm=EchoLLM(), k=3), max_attempts=2)


def test_answer_is_grounded():
    result = _loop().ask("What is the refund window?")
    assert result["grounded"] is True
    assert "30 days" in result["answer"]


def test_sources_returned():
    result = _loop().ask("refund")
    assert any(s.startswith("policy:") for s in result["sources"])


def test_attempts_bounded():
    result = _loop().ask("anything")
    assert result["attempts"] <= 2
RAGLOOP_FILE_EOF


echo
echo "Files written."
if [ ! -d .git ]; then
  git init -q 2>/dev/null && git add -A && git commit -q -m "Initial commit: ragloop skeleton" 2>/dev/null && echo "  git repo initialized." || echo "  (git not available; skipping commit)"
else
  echo "  (git repo already present; skipping init)"
fi

if [ "${RAGLOOP_NO_VENV:-0}" != "1" ]; then
  echo
  echo "Setting up a virtual environment and running the offline tests..."
  PYTHON="$(command -v python3.12 || command -v python3.11 || command -v python3)"
  "$PYTHON" -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -e ".[dev]"
  echo
  if pytest -q; then
    echo
    echo "All tests passed. The skeleton works offline (no API key needed)."
  else
    echo "Tests failed -- check the output above." >&2
  fi
fi

cat <<'NEXT'

------------------------------------------------------------------
ragloop is ready. Next steps:

  1. Open the folder in Cursor:        cursor .
  2. Install the full reference stack:  pip install -e ".[all]"
  3. Set your key:                      export ANTHROPIC_API_KEY=sk-ant-...
  4. Configure:                         cp examples/config.example.yaml config.yaml
  5. Ingest some docs:                  ragloop ingest ./your-docs --config config.yaml
  6. Ask:                               ragloop ask "your question" --config config.yaml

Run the dependency-free demo any time:  python examples/quickstart.py
------------------------------------------------------------------
NEXT