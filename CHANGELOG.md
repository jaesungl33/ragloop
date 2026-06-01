# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-01

Initial public release.

### Added

- **Self-correcting agentic RAG engine** — a LangGraph state machine
  (`plan → retrieve → fuse → generate → critique`) with a back-edge from the
  critic to retrieval. When an answer isn't fully grounded in its sources, the
  critique is fed back and the loop retries, bounded by a retry budget.
- **Pluggable retrieval** — `Retriever` interface with a **Chroma** reference
  backend (`semantic_search`, `keyword_search`, `get_chunk`).
- **Pluggable LLM** — `LLMProvider` interface with an **Anthropic (Claude)**
  reference provider.
- **MCP server** exposing `keyword_search`, `semantic_search`, and `chunk_read`
  to any MCP-capable client, with access control enforced server-side.
- **CLI** — `ragloop ingest`, `ragloop ask`, `ragloop serve`.
- **Config-driven wiring** via YAML + environment variables; no engine changes
  needed to swap vector store, LLM, model, or retry budget.
- **Examples** — a live demo (`examples/demo.py`) over a 5-document policy
  corpus, and a dependency-free `examples/quickstart.py` that also powers the
  test suite.
- **Evals harness** comparing a baseline pipeline against ragloop.

### Packaging

- Published to PyPI as **`ragloop-agentic`** (the import name stays `ragloop`).
- Apache-2.0 licensed; CI across Python 3.10–3.12; PyPI Trusted Publishing on
  GitHub Releases.

[0.1.0]: https://github.com/jaesungl33/ragloop/releases/tag/v0.1.0
