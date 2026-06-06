# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Harder eval scenario (`--scenario hard`) with distractor chunks and
  retrieval-hard / trap questions, plus a naive-baseline ablation
  (`--naive-baseline`) that isolates the loop's contribution from the prompt.

### Changed

- Benchmarks rewritten around an honest finding: with a modern instruction-tuned
  model, the self-correcting loop *ties* a one-shot baseline on quality (both
  decline ungrounded questions by default) at ~3× the cost. README claims
  tempered to match the evidence; the loop's value is documented as conditional
  (weak models, noisy corpora, multi-hop).

### Fixed

- `declined()` eval metric no longer counts incidental negations (e.g. "accidents
  are not covered") inside complete answers as refusals.

## [0.2.0] - 2026-06-01

### Added

- **Ollama provider** (`llm_provider: ollama`) — run the full loop on a local
  model for zero API cost. Stdlib-only, no new dependency.
- **Retry/backoff** on transient provider errors (both Anthropic and Ollama),
  plus structured `logging` through the engine for observability.
- **Configurable critic failure mode** — `critic_fail_closed` treats an
  unparseable grade as not-grounded (safer for high-stakes use).
- **Deterministic eval metrics** (`evals/metrics.py`): retrieval recall@k,
  hallucination-resistance / false-decline rate, and local-embedding answer
  similarity — no LLM judge, reproducible at $0. Four out-of-corpus questions
  added to the eval set. Published benchmark table in the README.
- `py.typed` marker so downstream users get the type hints.
- Tooling: `ruff` + `mypy` (both clean) and a lint/type CI job; coverage
  reporting via `pytest-cov`.

### Fixed

- `ChromaRetriever.add()` no longer crashes on documents with empty metadata.
- `retrieve()` now always searches the original query alongside the planner's
  sub-tasks, so decomposition can't lose a chunk a direct search would find.

### Changed

- Test suite expanded from 3 to 35 tests (~87% coverage), including the
  self-correcting retry loop, both providers, Chroma, config, and CLI.

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

[0.2.0]: https://github.com/jaesungl33/ragloop/releases/tag/v0.2.0
[0.1.0]: https://github.com/jaesungl33/ragloop/releases/tag/v0.1.0
