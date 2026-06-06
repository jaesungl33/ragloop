"""Expose the retriever as MCP tools.

Running this serves ``keyword_search``, ``semantic_search`` and ``chunk_read``
over MCP so any MCP-capable client (an agent, an IDE, another service) can
query the corpus directly. Retrieval lives in one server-side surface -- the
natural place to add auth or access filtering -- rather than being reimplemented
in each client. This is the "let the agent pull the data" pattern.

Note: this reference server does not implement authentication itself; add it
around ``build_server`` (or in front of the transport) for production use.
"""
from __future__ import annotations

import json

from ..config import Config, _build_retriever


def build_server(cfg: Config | None = None):
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
