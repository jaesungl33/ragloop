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
import urllib.error
import urllib.request
from typing import Optional

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        temperature: Optional[float] = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running "
                f"and the model '{self.model}' pulled? Original error: {exc}"
            ) from exc

        return (body.get("message") or {}).get("content", "")
