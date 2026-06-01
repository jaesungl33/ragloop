# ragloop: RAG that checks its own work

*Draft launch writeup — adapt for a blog post, Show HN, or the GitHub "About" section.*

---

## The one-line pitch

**ragloop is a RAG framework with a back-edge.** When the model writes an
answer, a critic grades whether every claim is actually supported by the
retrieved sources. If it isn't, the critique is fed back into retrieval and the
loop tries again — bounded by a retry budget. Classic linear RAG can't express
that; ragloop is built around it.

## The problem

Most RAG pipelines are a straight line: `retrieve → stuff into prompt →
generate`. They retrieve once, answer once, and hope the chunks were good. When
they weren't, the model fills the gap with something plausible and wrong — and
nothing in the pipeline notices. You find out when a user does.

The missing piece isn't a better embedding model. It's a **feedback loop**: a
step that judges the answer against its evidence and can send the system back to
look again.

## How ragloop works

```
plan → retrieve → fuse → generate → critique --(grounded?)--> done
          ^------------------------------------------(retry)----'
```

A LangGraph state machine:

1. **plan** — decompose the question into sub-queries.
2. **retrieve** — the agent picks its strategy: lexical, semantic, or a
   full-chunk read of a promising hit.
3. **fuse** — merge and rank the evidence.
4. **generate** — write an answer with inline `[source:id]` citations.
5. **critique** — grade whether the answer is *fully* grounded in those sources.
   Not grounded? Feed the critique back to **retrieve** and try again.

That single back-edge from the critic is the whole point. It's what turns "a
prompt with some context" into something that self-corrects.

## What it does in practice

Point it at a small policy corpus and ask an in-corpus question:

> **Q:** What is the refund window?
> **A:** The refund window is **30 days** from the delivery date shown on your
> order confirmation `[source:refunds:1]`.
> `grounded=True`

Now ask something the corpus doesn't cover:

> **Q:** Do you offer financing?
> **A:** The provided sources do not contain any information about financing
> options... I'd recommend reaching out directly to the support team.
> `grounded=True`

It declines instead of inventing a financing policy. For a system that's going
to answer real customer questions, an honest "I don't know" is the feature.

## Built to be swapped, not forked

The things companies actually differ on are behind one interface each, so you
never touch the engine:

- **Vector store** — Chroma ships as the reference backend; subclass
  `Retriever` for pgvector, Pinecone, Elasticsearch.
- **LLM** — Anthropic (Claude) ships as the reference provider; subclass
  `LLMProvider` for OpenAI, Bedrock, or a local model.
- **Corpus, model, retry budget** — all config-driven.

Retrieval is also exposed over **MCP**, so any MCP-capable client can query your
corpus with access control enforced server-side.

A complete custom retriever + provider is ~30 lines — see
`examples/quickstart.py`, which runs with zero API keys and also powers the test
suite.

## Try it

```bash
pip install "ragloop-agentic[all]"
export ANTHROPIC_API_KEY=sk-ant-...
python examples/demo.py        # the scripted in-corpus / out-of-corpus demo
```

Or the dependency-free version with no API key:

```bash
python examples/quickstart.py
```

## Roadmap

- Cross-encoder reranker hook in the fusion step
- Persistent agent memory across sessions (LangGraph checkpointer)
- pgvector + OpenAI reference backends
- Streaming answers

Apache-2.0. Issues and PRs welcome.
