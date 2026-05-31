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
