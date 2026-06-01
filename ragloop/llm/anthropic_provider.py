"""Anthropic (Claude) implementation of :class:`LLMProvider`.

This is the reference provider. Adding another (OpenAI, a local vLLM server,
Bedrock, etc.) means writing one more file like this and pointing the config
at it -- no engine changes required.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ._retry import retry_call
from .base import LLMProvider

log = logging.getLogger("ragloop.llm.anthropic")


def _transient_errors() -> tuple:
    """Anthropic exception types worth retrying, tolerant of SDK versions."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return ()
    names = ("APIConnectionError", "RateLimitError", "InternalServerError", "APITimeoutError")
    return tuple(t for t in (getattr(anthropic, n, None) for n in names) if t is not None)


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> None:
        # Imported lazily so the package installs without the SDK present
        # until a caller actually selects this provider.
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key=..."
            )
        self._client = Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        # Note: some newer Claude models reject a non-default temperature.
        # Leave it as None to omit the parameter entirely (recommended).
        self.temperature = temperature
        self.max_retries = max_retries
        self._transient = _transient_errors() or (Exception,)

    def complete(self, system: str, prompt: str) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        log.debug("anthropic complete: model=%s prompt_chars=%d", self.model, len(prompt))
        resp = retry_call(
            lambda: self._client.messages.create(**kwargs),
            retries=self.max_retries,
            exceptions=self._transient,
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
