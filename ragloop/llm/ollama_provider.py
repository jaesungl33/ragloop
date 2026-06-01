"""Ollama implementation of :class:`LLMProvider` — run ragloop on a local model.

A second reference provider alongside Anthropic, proving the point that the
engine is model-agnostic: the same loop runs on a hosted Claude model or a
local Llama/Qwen/Mistral served by Ollama, with no engine changes.

Talks to the Ollama HTTP API (``/api/chat``) using only the standard library,
so it adds no dependency. Start the server with ``ollama serve`` and pull a
model first, e.g. ``ollama pull llama3.2:3b``.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from ._retry import retry_call
from .base import LLMProvider

log = logging.getLogger("ragloop.llm.ollama")


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        temperature: float | None = 0.0,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def complete(self, system: str, prompt: str) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}

        log.debug("ollama complete: model=%s prompt_chars=%d", self.model, len(prompt))
        try:
            body = retry_call(
                lambda: self._post(payload),
                retries=self.max_retries,
                exceptions=(urllib.error.URLError, TimeoutError),
            )
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running "
                f"and the model '{self.model}' pulled? Original error: {exc}"
            ) from exc

        return (body.get("message") or {}).get("content", "")
