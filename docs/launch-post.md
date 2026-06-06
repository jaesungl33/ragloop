# I built a self-correcting RAG loop — then measured whether it actually helps

*Draft writeup — adapt for a blog post, Show HN, or r/LocalLLaMA. Put it in your
own voice before posting.*

---

## The hook

I built **ragloop**, an agentic RAG framework whose whole idea is a *back-edge*:
after the model writes a cited answer, a critic grades whether every claim is
actually supported by the retrieved sources, and if not, the critique is fed
back into retrieval and it tries again. Classic linear RAG can't express that.

Then I did the thing most "look at my framework" posts skip: **I built a
reproducible benchmark and tested whether the self-correction actually changes
the answer.** The honest result is more interesting than a win.

## What the benchmark found

Across three setups — a clean policy corpus, a harder corpus seeded with
distractor chunks, and an ablation against a *naive* baseline with no grounding
prompt at all — the self-correcting loop **tied** the one-shot baseline on every
quality metric (hallucination resistance, citation accuracy, retrieval recall),
while costing about **3× the latency and tokens**.

Why? **Modern instruction-tuned models are cautious by default.** Even a 7B model
with no instruction to stay grounded declines to answer questions the sources
don't cover. So on clean data with a capable model, the critic's safety net is
*redundant* — the model is already doing that work.

That's not the result I was hoping for. It's the one I'm publishing, because the
point of an eval harness is to find out, not to confirm.

## So when *is* the loop worth its cost?

When the model isn't doing the work for you:

- **Weak or unaligned models** that genuinely hallucinate.
- **Noisy or contradictory corpora**, where the right move is to reject a
  confident-looking wrong chunk.
- **Multi-hop questions** a single retrieval can't satisfy.

ragloop is a clean place to *measure that for your own data* rather than guess.

## How it works

```
plan → retrieve → fuse → generate → critique --(grounded?)--> done
          ^------------------------------------------(retry)----'
```

A LangGraph state machine: decompose the question, choose a retrieval strategy
(lexical / semantic / full-chunk read), fuse and rank evidence, generate a cited
answer, then grade whether it's grounded — and loop back if not, bounded by a
retry budget.

## The parts I'm actually happy with

- **Swap-not-fork design.** Your vector store and your LLM each sit behind one
  interface. Chroma + Anthropic ship as references; a local **Ollama** backend
  ships too (so the whole thing — including the benchmark — runs for $0).
- **MCP-native retrieval**, so any MCP client can query your corpus.
- **Honest, reproducible evals** — deterministic, label-based, no LLM-as-judge,
  and the harness even caught a false-positive bug in my own decline metric.
- ~88% test coverage, typed, CI across Python 3.10–3.12.

## Try it

```bash
pip install "ragloop-agentic[all]"
python examples/quickstart.py          # no API key
# reproduce the benchmark locally for $0:
ollama pull qwen2.5:7b-instruct
python -m evals.runner --llm ollama --model qwen2.5:7b-instruct --scenario hard
```

Apache-2.0 · https://github.com/jaesungl33/ragloop

Feedback I'd genuinely like: a setting where the self-correction loop *clearly*
beats a one-shot baseline. I couldn't manufacture one without it feeling
contrived — if you can, I want to see it.
