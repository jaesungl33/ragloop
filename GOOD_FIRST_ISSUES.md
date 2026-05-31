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
