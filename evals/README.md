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

```bash
# Offline smoke run — no API key needed; LLM-judged metrics are skipped
python -m evals.runner --offline

# Full run — scores all metrics via a real LLM judge
ANTHROPIC_API_KEY=sk-ant-... python -m evals.runner

# Re-print the table from a previous run
python -m evals.report
python -m evals.report path/to/results.json   # custom file
```

Results are written to `evals/results.json` after each run.

## Metrics

| Metric | Tool | Notes |
|---|---|---|
| **Faithfulness** | RAGAS | Grounding in the *retrieved context* — see note below |
| **Answer Relevancy** | RAGAS | How directly the answer addresses the question |
| **Context Precision** | RAGAS | Fraction of retrieved chunks that are actually relevant |
| **Context Recall** | RAGAS | Fraction of relevant chunks that were retrieved |
| **Citation Accuracy** | built-in | `\|cited ∩ retrieved\| / \|cited\|`; always runs offline |
| **Avg Latency (s)** | built-in | Wall-clock time per question |
| **Avg Token Cost** | built-in | Approx tokens (input + output chars ÷ 4) per question |
| **Avg Retries** | built-in | Extra loop iterations beyond the first attempt |

> **Faithfulness ≠ factual truth.** A score of 1.0 means every claim in the
> answer is supported by a retrieved chunk. It says nothing about whether
> that chunk is factually correct. An answer can be faithfully grounded in
> a wrong source and still score 1.0.

LLM-judged metrics (faithfulness, answer relevancy, context precision/recall)
require `ragas>=0.1` and either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
**Citation accuracy always runs**, making it the primary offline signal.

## Corpus

Ten chunks drawn from the five Northstar Outdoors policy documents in
`examples/corpus/` (refunds, shipping, warranty, privacy, support hours).
Two chunks per document keep retrieval non-trivial so precision and recall
are meaningful even with the in-memory fake retriever.

## Adding questions or documents

Edit `evals/corpus.py`. `DOCS` is a plain `list[Document]`; `QUESTIONS` is a
`list[dict]` with keys `question`, `ground_truth`, and `relevant_ids`.
