"""Command-line interface: ingest a corpus, ask a question, serve MCP.

    ragloop ingest ./docs --config config.yaml
    ragloop ask "What is our refund window?" --config config.yaml
    ragloop serve --config config.yaml
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_cfg(config: str | None):
    from .config import Config

    return Config.from_yaml(config) if config else Config()


def cmd_ingest(path: str, config: str | None) -> None:
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


def cmd_ask(question: str, config: str | None) -> None:
    from .config import build_from_config

    loop = build_from_config(path=config)
    result = loop.ask(question)
    print("\nAnswer:\n" + result["answer"])
    print(f"\ngrounded={result['grounded']}  attempts={result['attempts']}  sources={result['sources']}")


def cmd_serve(config: str | None) -> None:
    from .config import Config
    from .mcp.server import build_server

    cfg = Config.from_yaml(config) if config else Config()
    build_server(cfg).run()


def _resolve_config(args) -> str | None:
    """--config may appear before or after the subcommand; fall back to env."""
    return getattr(args, "config", None) or os.environ.get("RAGLOOP_CONFIG")


def main(argv: list | None = None) -> int:
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
