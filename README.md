# ragloop

[![PyPI](https://img.shields.io/pypi/v/ragloop-agentic.svg)](https://pypi.org/project/ragloop-agentic/)
[![CI](https://github.com/jaesungl33/ragloop/actions/workflows/ci.yml/badge.svg)](https://github.com/jaesungl33/ragloop/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jaesungl33/ragloop/branch/main/graph/badge.svg)](https://codecov.io/gh/jaesungl33/ragloop)
[![Python](https://img.shields.io/pypi/pyversions/ragloop-agentic.svg)](https://pypi.org/project/ragloop-agentic/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A pluggable, self-correcting **agentic RAG** framework. Point it at your
documents, your vector store, and your LLM, and it answers questions with
inline citations — checking its own work and retrying when an answer isn't
grounded.

![ragloop live demo](docs/demo.gif)

*The scripted demo (`examples/demo.py`): a cited, grounded answer for an
in-corpus question — and an honest decline instead of an invented answer when
the corpus doesn't cover the question.*

It's built so that the things companies actually differ on are swappable
without touching the engine:

- **Your vector store** — Chroma ships as the reference backend; swap in
  pgvector, Pinecone, or Elasticsearch by subclassing one interface.
- **Your LLM** — Anthropic (Claude) and a local **Ollama** backend ship as
  reference providers; add OpenAI, Bedrock, or any model the same way.
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
pip install "ragloop-agentic[all]"            # engine + Anthropic + Chroma + MCP
# or pick pieces: pip install "ragloop-agentic[anthropic,chroma]"
# from source:    pip install -e ".[all]"
export ANTHROPIC_API_KEY=sk-ant-...
```

> The distribution is published as **`ragloop-agentic`**; the import name stays
> `ragloop` (`from ragloop import build_from_config`).

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

## Live demo

Clone the repo, install the full stack, set your API key, and run the scripted demo against a small fictional policy corpus (`examples/corpus/`):

```bash
pip install -e ".[all]"
export ANTHROPIC_API_KEY=sk-ant-...
python examples/demo.py
```

The script ingests five policy documents into Chroma (persisted under `.chroma_demo/`), then asks two questions:

1. **In-corpus** — *"What is the refund window?"* — expects a cited answer grounded in the refund policy (e.g. a **30-day** window) with `grounded=True`.
2. **Out-of-corpus** — *"Do you offer financing?"* — expects the model to say the sources do not cover financing rather than inventing terms, typically with `grounded=True` on a decline-style answer.

Each question prints the answer plus metadata:

```
============================================================
In-corpus: What is the refund window?
============================================================

Answer:
Customers may request a refund within 30 days of delivery [source:refunds:0].

grounded=True  attempts=1  sources=['refunds:0', ...]

============================================================
Out-of-corpus: Do you offer financing?
============================================================

Answer:
The provided sources do not mention financing or payment plans.

grounded=True  attempts=1  sources=[...]
```

Exact wording varies by model run; the important part is grounded, cited answers for in-corpus questions and an honest decline for out-of-corpus ones.

## Benchmarks

ragloop ships a reproducible eval harness (`evals/`) that runs a naive one-shot
**baseline** (embed → top-k → generate) against the full **self-correcting
loop** over a labelled policy corpus — 14 questions, 4 of them deliberately
*out-of-corpus*. Scoring is **deterministic and label-based** (no LLM-as-judge),
so anyone can reproduce it with **zero API cost** on a local model:

```bash
pip install -e ".[evals,chroma]"
ollama pull llama3.2:3b          # any local model works
python -m evals.runner --llm ollama --model llama3.2:3b
```

Representative run (local `llama3.2:3b`, real Chroma retrieval):

| Metric | Baseline | RagLoop | What it means |
|---|---:|---:|---|
| **Hallucination resistance** | 1.00 | **1.00** | fraction of *unanswerable* questions correctly declined (↑) |
| **False-decline rate** | 0.00 | **0.00** | answerable questions wrongly refused (↓) |
| **Citation accuracy** | 1.00 | **1.00** | cited IDs that are actually in the retrieved set (↑) |
| **Retrieval recall@k** | 1.00 | 0.90 | gold chunks surfaced (↑) |
| **Answer similarity** | 0.78 | 0.78 | cosine vs. ground truth, local embeddings (↑) |
| Avg latency (s) | 1.8 | 5.1 | wall-clock per question (↓) |
| Avg token cost | 415 | 1203 | approx tokens per question (↓) |
| Avg retries | 0.00 | 0.36 | extra loop iterations (↓) |

**Honest read of these numbers.** On a small, clean corpus both pipelines
already resist hallucination perfectly and cite accurately — so here the loop's
self-correction buys safety you can't see, at a real ~2.8× latency / ~2.9× token
cost. The loop earns that cost on **noisier corpora, weaker retrieval, or
higher-stakes grounding**, where a one-shot baseline *would* answer from a wrong
chunk and the critic catches it. Building this benchmark also surfaced a real
regression — the planner's decomposition could drop the chunk a direct search
would find — which is now fixed (recall 0.80 → 0.90) and guarded by a test. The
remaining recall gap (cross-query score comparability in fusion) is tracked as
future work. See [`evals/README.md`](evals/README.md) for methodology and the
LLM-judged RAGAS metrics (opt-in via `--judge`).

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
