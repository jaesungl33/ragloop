# Contributing to ragloop

Thanks for considering a contribution. ragloop aims to stay small, readable,
and easy to extend, so most contributions land quickly.

## Development setup

```bash
git clone https://github.com/<your-username>/ragloop
cd ragloop
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The test suite runs entirely offline with in-memory fakes -- no API key or
vector store required. If `pytest` is green, your environment is ready.

## How the project is organized

- `ragloop/llm/` -- the `LLMProvider` interface and the Anthropic reference.
- `ragloop/retrieval/` -- the `Retriever` interface and the Chroma reference.
- `ragloop/engine/` -- graph state, nodes, and the LangGraph assembly.
- `ragloop/mcp/` -- the MCP retrieval server.

The engine never imports a concrete provider or backend; everything is wired
through `config.py`. Keep it that way -- it's what makes ragloop pluggable.

## Adding a backend

The most useful contributions are new reference backends:

- **New vector store:** subclass `Retriever` (implement `add`,
  `semantic_search`, `keyword_search`, `get_chunk`) and add a branch in
  `config._build_retriever`.
- **New LLM provider:** subclass `LLMProvider` (implement `complete`) and add a
  branch in `config._build_llm`.

`examples/quickstart.py` shows a complete custom retriever and provider in about
30 lines -- use it as a template, and add a small offline test alongside it.

## Pull request checklist

- Tests pass locally (`pytest`) and you added tests for new behavior.
- New backends keep the existing interface unchanged.
- Public functions have a short docstring; comments explain *why*, not *what*.
- One focused change per PR. Link the issue it addresses.

## Reporting bugs and proposing features

Open an issue using the templates. For anything security-sensitive, please do
not open a public issue -- contact a maintainer directly.

## License

By contributing you agree that your contributions are licensed under the
project's Apache-2.0 license.
