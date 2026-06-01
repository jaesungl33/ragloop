"""Tests for the dependency-free retry helper (no real sleeping)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragloop.llm._retry import retry_call  # noqa: E402


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    out = retry_call(fn, retries=3, base_delay=0, sleep=lambda _s: None)
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_budget_exhausted():
    def fn():
        raise ValueError("always fails")

    with pytest.raises(ValueError):
        retry_call(fn, retries=2, base_delay=0, sleep=lambda _s: None)


def test_retry_only_catches_listed_exceptions():
    def fn():
        raise KeyError("not retried")

    with pytest.raises(KeyError):
        retry_call(fn, retries=3, base_delay=0, exceptions=(ValueError,), sleep=lambda _s: None)
