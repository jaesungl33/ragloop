"""Anthropic (Claude) implementation of :class:`LLMProvider`.

This is the reference provider. Adding another (OpenAI, a local vLLM server,
Bedrock, etc.) means writing one more file like this and pointing the config
at it -- no engine changes required.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
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

    def complete(self, system: str, prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self._client.messages.create(**kwargs)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
