"""Print a side-by-side comparison table of baseline vs RagLoop eval results.

Usage::

    python -m evals.report                       # reads evals/results.json
    python -m evals.report path/to/results.json  # custom results file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# (metric key, display label, True = higher is better)
_METRICS = [
    ("retrieval_recall",   "Retrieval Recall@k",  True),
    ("citation_accuracy",  "Citation Accuracy",   True),
    ("answer_similarity",  "Answer Similarity",   True),
    ("faithfulness",       "Faithfulness",        True),
    ("answer_relevancy",   "Answer Relevancy",    True),
    ("context_precision",  "Context Precision",   True),
    ("context_recall",     "Context Recall",      True),
    ("latency_s",          "Avg Latency (s)",     False),
    ("token_cost",         "Avg Token Cost",      False),
    ("retries",            "Avg Retries",         False),
]

_COL_W = 24
_VAL_W = 11


def _avg(results: list[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _decline_rate(results: list[dict], answerable: bool) -> Optional[float]:
    """Fraction of {answerable|unanswerable} questions the system declined."""
    subset = [r for r in results if r.get("answerable", True) is answerable]
    declined = [1.0 if r.get("declined") else 0.0 for r in subset]
    return sum(declined) / len(declined) if declined else None


def _fmt(val: Optional[float], precision: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{precision}f}"


def _delta(b: Optional[float], r: Optional[float], higher_better: bool) -> str:
    if b is None or r is None:
        return "N/A"
    diff = r - b
    sign = "+" if diff >= 0 else ""
    marker = ""
    if diff != 0:
        marker = " ▲" if (diff > 0) == higher_better else " ▼"
    return f"{sign}{diff:.2f}{marker}"


def print_table(baseline: list[dict], ragloop: list[dict]) -> None:
    sep = "─" * (_COL_W + _VAL_W * 2 + 14)
    header = (
        f"{'Metric':<{_COL_W}}  "
        f"{'Baseline':>{_VAL_W}}  "
        f"{'RagLoop':>{_VAL_W}}  "
        f"{'Δ (rag−base)':>{_VAL_W}}"
    )
    print()
    print(header)
    print(sep)
    for key, label, higher_better in _METRICS:
        b_val = _avg(baseline, key)
        r_val = _avg(ragloop, key)
        if b_val is None and r_val is None:
            continue  # hide metrics that weren't scored (e.g. RAGAS without --judge)
        prec = 3 if key == "latency_s" else 2
        row = (
            f"{label:<{_COL_W}}  "
            f"{_fmt(b_val, prec):>{_VAL_W}}  "
            f"{_fmt(r_val, prec):>{_VAL_W}}  "
            f"{_delta(b_val, r_val, higher_better):>{_VAL_W}}"
        )
        print(row)
    print(sep)

    # --- decline behavior (hallucination resistance) ---
    halluc_b = _decline_rate(baseline, answerable=False)   # higher = better (declined unanswerable)
    halluc_r = _decline_rate(ragloop, answerable=False)
    false_b = _decline_rate(baseline, answerable=True)     # lower = better (didn't refuse answerable)
    false_r = _decline_rate(ragloop, answerable=True)
    if halluc_b is not None or halluc_r is not None:
        print()
        print("Decline behavior (out-of-corpus questions)")
        print(sep)
        print(
            f"{'Hallucination resist.':<{_COL_W}}  "
            f"{_fmt(halluc_b):>{_VAL_W}}  {_fmt(halluc_r):>{_VAL_W}}  "
            f"{_delta(halluc_b, halluc_r, True):>{_VAL_W}}"
        )
        print(
            f"{'False-decline (ans.)':<{_COL_W}}  "
            f"{_fmt(false_b):>{_VAL_W}}  {_fmt(false_r):>{_VAL_W}}  "
            f"{_delta(false_b, false_r, False):>{_VAL_W}}"
        )
        print(sep)
        print("  Hallucination resist. = fraction of unanswerable Qs correctly declined (↑ better)")
        print("  False-decline = fraction of answerable Qs wrongly declined (↓ better)")
    print("  ▲ = improvement   ▼ = regression   (for cost metrics, lower is better)")
    print()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results.json"
    if not path.exists():
        print(f"[report] results file not found: {path}", file=sys.stderr)
        print("[report] run `python -m evals.runner` first", file=sys.stderr)
        sys.exit(1)
    all_results: list[dict] = json.loads(path.read_text())
    baseline = [r for r in all_results if r["pipeline"] == "baseline"]
    ragloop = [r for r in all_results if r["pipeline"] == "ragloop"]
    print_table(baseline, ragloop)


if __name__ == "__main__":
    main()
