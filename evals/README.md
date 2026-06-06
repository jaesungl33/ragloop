# ragloop evals

Compares the self-correcting **RagLoop** pipeline against a naive **baseline**
(single embed → top-k → generate, no planner / critic / retry) over a small
bundled corpus of policy documents and 10 questions.

## Install

```bash
pip install -e ".[evals]"
```

The `[evals]` extra adds `ragas`, `deepeval`, and `datasets`. The core package
and existing tests work without it.

## Run

The default metrics are **deterministic and label-based** — no LLM judge — so a
full run costs **$0** on a local model and is reproducible by anyone:

```bash
# $0, fully local: real Chroma retrieval + a local LLM via Ollama
ollama pull llama3.2:3b
python -m evals.runner --llm ollama --model llama3.2:3b

# Offline smoke run — fake LLM, no model needed (CI-friendly)
python -m evals.runner --offline

# Hosted model instead of local
ANTHROPIC_API_KEY=sk-ant-... python -m evals.runner --llm anthropic

# Harder corpus (distractor chunks + retrieval-hard / trap questions)
python -m evals.runner --llm ollama --scenario hard

# Ablation: compare against a NAIVE baseline (no grounding prompt) instead of
# the careful grounded one — isolates how much the loop adds beyond the prompt
python -m evals.runner --llm ollama --scenario hard --naive-baseline

# Add the LLM-judged RAGAS/deepeval metrics (opt-in; needs an OpenAI/Anthropic
# key and costs money — the judge fires many grading calls)
python -m evals.runner --llm ollama --judge

# Re-print the table from a previous run
python -m evals.report [path/to/results.json]
```

> **What these runs show:** with a modern instruction-tuned model, the loop and
> the baseline tie on quality (both decline ungrounded questions by default), and
> the loop costs ~3× latency. The self-correction earns its cost with weaker
> models, noisier/contradictory corpora, or multi-hop retrieval. See the main
> [README Benchmarks](../README.md#benchmarks--does-the-self-correction-actually-help)
> section for the numbers and the honest read.

Retrieval uses a **real Chroma vector store** with free local embeddings, so the
retrieval metrics reflect a real index — not a toy. Results are written to
`evals/results.json`.

## Metrics

**Deterministic (default — no API, no judge):**

| Metric | Notes |
|---|---|
| **Hallucination resistance** | Fraction of *out-of-corpus* questions correctly declined (↑). The headline safety metric. |
| **False-decline rate** | Fraction of answerable questions wrongly refused (↓). |
| **Retrieval recall@k** | `\|retrieved ∩ gold\| / \|gold\|` against labelled `relevant_ids` (↑). |
| **Citation accuracy** | `\|cited ∩ retrieved\| / \|cited\|` (↑). |
| **Answer similarity** | Cosine of answer vs. `ground_truth` using a local MiniLM embedder (↑). |
| **Avg latency / token cost / retries** | Cost of the self-correction loop (↓). |

**LLM-judged (opt-in via `--judge`, needs an API key, costs money):**

| Metric | Tool | Notes |
|---|---|---|
| **Faithfulness** | RAGAS | Grounding in the retrieved context — see caveat |
| **Answer Relevancy** | RAGAS | How directly the answer addresses the question |
| **Context Precision / Recall** | RAGAS | Relevance of retrieved chunks |

> **Faithfulness ≠ factual truth.** A score of 1.0 means every claim is
> supported by a retrieved chunk — not that the chunk is correct. RAGAS/deepeval
> default their judge to OpenAI, which is why these are opt-in and not free.

## Corpus

Ten chunks from the five Northstar Outdoors policy documents in
`examples/corpus/`, plus **four out-of-corpus questions** (`answerable: False`)
the corpus deliberately can't answer — these drive the hallucination-resistance
metric. Two chunks per document keep retrieval non-trivial.

## Adding questions or documents

Edit `evals/corpus.py`. `DOCS` is a plain `list[Document]`; `QUESTIONS` is a
`list[dict]` with keys `question`, `ground_truth`, and `relevant_ids`.
