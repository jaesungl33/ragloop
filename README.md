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

It's a **reference implementation** you can read end to end and extend — not a
black box — and it ships with a reproducible eval harness so you can measure
whether the self-correction loop actually helps on *your* data instead of taking
the claim on faith. (Spoiler from my own runs: with a modern model on clean data,
often it doesn't — [see Benchmarks](#benchmarks--does-the-self-correction-actually-help).)

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
architecturally distinguishes this from a one-shot RAG pipeline. Whether that
back-edge *changes the final answer* depends on your model and data — see
[Benchmarks](#benchmarks--does-the-self-correction-actually-help) for an honest
measurement, not a marketing claim.

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

## Benchmarks — does the self-correction actually help?

Short answer, honestly: **on a clean corpus with a modern model, not measurably.**
This section reports that rather than hiding it, because knowing *when* a
technique helps is the whole point of building an eval harness.

ragloop ships a reproducible harness (`evals/`) that runs one-shot baselines
against the full loop over a labelled policy corpus. Scoring is **deterministic
and label-based** (no LLM-as-judge), so anyone can reproduce it at **zero API
cost** on a local model:

```bash
pip install -e ".[evals,chroma]"
ollama pull qwen2.5:7b-instruct
python -m evals.runner --llm ollama --model qwen2.5:7b-instruct --scenario hard
```

Representative run (`qwen2.5:7b`, "hard" corpus with distractor chunks, 16
questions, real Chroma retrieval):

| Metric | Grounded baseline | RagLoop | |
|---|---:|---:|---|
| **Hallucination resistance** | 1.00 | 1.00 | fraction of *unanswerable* questions correctly declined (↑) |
| **False-decline rate** | 0.10 | 0.10 | answerable questions wrongly refused (↓) |
| **Citation accuracy** | 0.94 | 0.94 | cited IDs actually in the retrieved set (↑) |
| **Retrieval recall@k** | 0.90 | 0.90 | gold chunks surfaced (↑) |
| Avg latency (s) | 4.0 | 11.9 | wall-clock per question (↓) |
| Avg token cost | 404 | 1320 | approx tokens per question (↓) |

**What this shows.** The two pipelines are statistically tied on every quality
metric — and the loop costs ~3× the latency and tokens to get there. I ran the
same comparison three ways (clean corpus; hard corpus with distractors; and
against a **naive** baseline with *no* grounding prompt at all) and the result
held every time: hallucination resistance stayed at 1.00 for both.

The reason is worth stating plainly: **modern instruction-tuned models are
cautious by default.** Even a 7B model with no grounding instruction declines to
answer questions the sources don't cover — so the critic's safety net is
*redundant* in this regime. (Building this also caught a false-positive bug in my
own decline metric, now fixed and regression-tested.)

**So when *does* the loop earn its cost?** When the model isn't doing this work
for you: weak or unaligned models that *do* hallucinate, noisy or contradictory
corpora where the right move is to reject a confident-looking wrong chunk, or
multi-hop questions a single retrieval can't satisfy. ragloop is a clean place to
*measure* that for your own data — `python -m evals.runner --help` for the
scenarios, and [`evals/README.md`](evals/README.md) for methodology.

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
