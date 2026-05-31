"""Abstract LLM provider interface.

Any company can swap in their own model backend by subclassing ``LLMProvider``
and implementing ``complete``. The engine never imports a concrete provider
directly; it receives one through configuration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Minimal text-completion interface used by the engine nodes."""

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str:
        """Return the model's text response to ``prompt`` under ``system``.

        Implementations should return plain text (no tool-use blocks). The
        engine handles structured output by instructing the model to emit JSON
        and parsing the returned string.
        """
        raise NotImplementedError
