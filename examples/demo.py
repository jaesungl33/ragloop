"""Live demo: ingest the sample policy corpus into Chroma and ask real questions.

Requires: pip install -e ".[all]" and ANTHROPIC_API_KEY in the environment.
Run from the repo root: python examples/demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ragloop import Deps, Document, RagLoop
from ragloop.llm.anthropic_provider import AnthropicProvider
from ragloop.retrieval.chroma_retriever import ChromaRetriever

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
PERSIST_DIR = Path(".chroma_demo")
COLLECTION = "ragloop_demo"


def _ingest(retriever: ChromaRetriever) -> None:
    files = sorted(CORPUS_DIR.rglob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found under {CORPUS_DIR}")

    docs: list[Document] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(c for c in text.split("\n\n") if c.strip()):
            docs.append(
                Document(
                    id=f"{fp.stem}:{i}",
                    text=chunk.strip(),
                    metadata={"file": str(fp.relative_to(CORPUS_DIR.parent))},
                )
            )
    retriever.add(docs)
    print(f"Ingested {len(docs)} chunks from {len(files)} file(s) into {PERSIST_DIR}/")


def _print_result(label: str, question: str, result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"{label}: {question}")
    print(f"{'=' * 60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(
        f"\ngrounded={result['grounded']}  "
        f"attempts={result['attempts']}  "
        f"sources={result['sources']}"
    )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export it before running the live demo.", file=sys.stderr)
        sys.exit(1)

    retriever = ChromaRetriever(collection=COLLECTION, persist_dir=str(PERSIST_DIR))
    _ingest(retriever)

    loop = RagLoop(
        Deps(retriever=retriever, llm=AnthropicProvider(), k=5),
        max_attempts=2,
    )

    _print_result(
        "In-corpus",
        "What is the refund window?",
        loop.ask("What is the refund window?"),
    )
    _print_result(
        "Out-of-corpus",
        "Do you offer financing?",
        loop.ask("Do you offer financing?"),
    )


if __name__ == "__main__":
    main()
