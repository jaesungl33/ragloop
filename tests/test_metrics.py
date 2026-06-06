"""Tests for the deterministic eval metrics (no API, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.metrics import declined, retrieval_recall  # noqa: E402


def test_declined_detects_refusals():
    assert declined("The provided sources do not contain information about financing.")
    assert declined("There is no mention of gift cards in the sources.")
    assert declined("I cannot find that in the provided sources.")


def test_declined_false_for_real_answers():
    assert not declined("The refund window is 30 days [source:refunds:0].")
    assert not declined("Standard shipping is free on orders over $50.")


def test_declined_ignores_incidental_negation_in_real_answers():
    # Regression: a complete answer that happens to contain "not covered"
    # must NOT be counted as a decline.
    answer = (
        "Tents are covered under a two-year warranty [source:warranty:0]. "
        "Normal wear, misuse, and damage from accidents are not covered. "
        "Contact support with photos to start a claim."
    )
    assert not declined(answer)


def test_retrieval_recall():
    assert retrieval_recall(["a", "b", "c"], ["a"]) == 1.0
    assert retrieval_recall(["x", "y"], ["a", "b"]) == 0.0
    assert retrieval_recall(["a", "x"], ["a", "b"]) == 0.5
    assert retrieval_recall(["a"], []) is None  # no gold labels -> not applicable
