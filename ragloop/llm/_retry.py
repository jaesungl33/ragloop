"""Tiny dependency-free retry/backoff helper for transient LLM failures.

Kept in-house (rather than pulling in ``tenacity``) so the package adds no
dependency and the stdlib-only Ollama provider stays stdlib-only.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger("ragloop.llm")

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with exponential backoff on ``exceptions``.

    Retries up to ``retries`` times (so ``retries + 1`` total attempts), sleeping
    ``base_delay * backoff**(n-1)`` seconds between tries. Re-raises the last
    error once the budget is exhausted. ``sleep`` is injectable for testing.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except exceptions as exc:
            attempt += 1
            if attempt > retries:
                log.error("LLM call failed after %d attempts: %s", attempt, exc)
                raise
            delay = base_delay * (backoff ** (attempt - 1))
            log.warning(
                "LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                retries,
                exc,
                delay,
            )
            sleep(delay)
