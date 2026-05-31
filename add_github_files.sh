#!/usr/bin/env bash
#
# add_github_files.sh -- add CI, contributing guide, and issue/PR templates
# to an existing ragloop project. Run from inside the ragloop/ directory.
set -euo pipefail
if [ ! -f pyproject.toml ] || [ ! -d ragloop ]; then
  echo "Run this from inside your ragloop project directory." >&2
  exit 1
fi
echo "Adding project files..."

mkdir -p ".github/workflows"
echo "  writing .github/workflows/ci.yml"
cat > ".github/workflows/ci.yml" <<'RAGLOOP_FILE_EOF'
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run tests
        # The offline test suite uses in-memory fakes -- no API key needed.
        run: pytest -q
RAGLOOP_FILE_EOF

echo "  writing CONTRIBUTING.md"
cat > "CONTRIBUTING.md" <<'RAGLOOP_FILE_EOF'
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
RAGLOOP_FILE_EOF

echo "  writing CODE_OF_CONDUCT.md"
cat > "CODE_OF_CONDUCT.md" <<'RAGLOOP_FILE_EOF'
# Code of Conduct

This project adopts the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/),
version 2.1.

In short: be respectful, assume good faith, and keep discussion focused on the
work. Harassment or discrimination of any kind is not tolerated.

To report unacceptable behavior, contact the maintainers at
`<your-contact-email>`. Reports will be handled confidentially.
RAGLOOP_FILE_EOF

mkdir -p ".github/ISSUE_TEMPLATE"
echo "  writing .github/ISSUE_TEMPLATE/bug_report.md"
cat > ".github/ISSUE_TEMPLATE/bug_report.md" <<'RAGLOOP_FILE_EOF'
---
name: Bug report
about: Something isn't working as documented
title: "[bug] "
labels: bug
---

**What happened**
A clear description of the bug.

**Steps to reproduce**
1.
2.
3.

**Expected behavior**
What you expected instead.

**Environment**
- ragloop version:
- Python version:
- LLM provider / model:
- Retriever backend:

**Logs / traceback**
```
paste here
```
RAGLOOP_FILE_EOF

mkdir -p ".github/ISSUE_TEMPLATE"
echo "  writing .github/ISSUE_TEMPLATE/feature_request.md"
cat > ".github/ISSUE_TEMPLATE/feature_request.md" <<'RAGLOOP_FILE_EOF'
---
name: Feature request
about: Suggest an improvement or a new backend
title: "[feature] "
labels: enhancement
---

**Problem**
What are you trying to do that ragloop makes hard today?

**Proposed solution**
What you'd like to see. For a new backend, name the store/provider and link its
client library.

**Alternatives considered**
Anything you've tried or ruled out.

**Willing to contribute?**
Are you open to opening a PR for this?
RAGLOOP_FILE_EOF

mkdir -p ".github"
echo "  writing .github/PULL_REQUEST_TEMPLATE.md"
cat > ".github/PULL_REQUEST_TEMPLATE.md" <<'RAGLOOP_FILE_EOF'
## What this does

Brief description of the change.

## Related issue

Closes #

## Checklist

- [ ] `pytest` passes locally
- [ ] Added or updated tests
- [ ] Existing `Retriever` / `LLMProvider` interfaces unchanged (or change is documented)
- [ ] Docstrings added for new public functions
RAGLOOP_FILE_EOF

echo "  writing GOOD_FIRST_ISSUES.md"
cat > "GOOD_FIRST_ISSUES.md" <<'RAGLOOP_FILE_EOF'
# Good first issues (seed list)

Copy each block into a new GitHub issue and apply the `good first issue` label.
These are scoped so a new contributor can finish one in an afternoon, and they
double as your own roadmap.

---

### Add a cross-encoder reranker to the fusion step
**labels:** enhancement, good first issue

`engine/nodes.py:fuse` currently ranks by retrieval score and dedupes by text
prefix (MMR-lite). Add an optional reranker that re-scores the fused candidates
with a cross-encoder (e.g. a sentence-transformers model) before trimming to
top-k. Make it config-toggleable so it stays optional.

---

### Add a pgvector reference retriever
**labels:** enhancement, good first issue

Implement `PgVectorRetriever(Retriever)` in `retrieval/pgvector_retriever.py`
backing `semantic_search` with a pgvector similarity query and `keyword_search`
with Postgres full-text search. Register it in `config._build_retriever` under
`retriever_backend: pgvector`. Add an offline-friendly test.

---

### Add an OpenAI LLM provider
**labels:** enhancement, good first issue

Implement `OpenAIProvider(LLMProvider)` in `llm/openai_provider.py` with a
`complete(system, prompt)` method. Register it in `config._build_llm` under
`llm_provider: openai`. Keep the interface identical to the Anthropic provider.

---

### Persist agent memory across sessions
**labels:** enhancement

Wire a LangGraph checkpointer into `engine/graph.py` so the loop can carry
short- and long-term state between calls instead of starting cold each time.
Expose the store choice through config.

---

### Stream the generated answer
**labels:** enhancement

Add a streaming variant of `RagLoop.ask` that yields answer tokens as they are
produced, for use in chat UIs. Requires a streaming method on `LLMProvider`.
RAGLOOP_FILE_EOF


echo
echo "Done. Review, then commit:"
echo "  git add -A && git commit -m \"Add CI, contributing guide, and templates\""
echo
echo "Then create the GitHub repo and push (see instructions)."
echo "Before committing, edit CODE_OF_CONDUCT.md to add your contact email,"
echo "and the <your-username> placeholders in CONTRIBUTING.md / pyproject.toml."