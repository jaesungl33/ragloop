"""Config-driven wiring so adopters never edit code.

A YAML file (and environment variables for secrets) selects the LLM provider,
the retriever backend, and runtime knobs. :func:`build_from_config` returns a
ready ``RagLoop``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engine import Deps, RagLoop
from .llm.base import LLMProvider
from .retrieval.base import Retriever


@dataclass
class Config:
    llm_provider: str = "anthropic"
    llm: dict[str, Any] = field(default_factory=dict)
    retriever_backend: str = "chroma"
    retriever: dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    max_attempts: int = 2
    critic_fail_closed: bool = False

    @staticmethod
    def from_yaml(path: str) -> Config:
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return Config(
            llm_provider=data.get("llm_provider", "anthropic"),
            llm=data.get("llm", {}),
            retriever_backend=data.get("retriever_backend", "chroma"),
            retriever=data.get("retriever", {}),
            top_k=int(data.get("top_k", 5)),
            max_attempts=int(data.get("max_attempts", 2)),
            critic_fail_closed=bool(data.get("critic_fail_closed", False)),
        )


def _build_llm(cfg: Config) -> LLMProvider:
    if cfg.llm_provider == "anthropic":
        from .llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(**cfg.llm)
    if cfg.llm_provider == "ollama":
        from .llm.ollama_provider import OllamaProvider

        return OllamaProvider(**cfg.llm)
    raise ValueError(
        f"Unknown llm_provider '{cfg.llm_provider}'. "
        "Register your own by subclassing LLMProvider and editing _build_llm."
    )


def _build_retriever(cfg: Config) -> Retriever:
    if cfg.retriever_backend == "chroma":
        from .retrieval.chroma_retriever import ChromaRetriever

        return ChromaRetriever(**cfg.retriever)
    raise ValueError(
        f"Unknown retriever_backend '{cfg.retriever_backend}'. "
        "Register your own by subclassing Retriever and editing _build_retriever."
    )


def build_from_config(path: str | None = None, cfg: Config | None = None) -> RagLoop:
    cfg = cfg or (Config.from_yaml(path) if path else Config())
    llm = _build_llm(cfg)
    retriever = _build_retriever(cfg)
    deps = Deps(retriever=retriever, llm=llm, k=cfg.top_k, fail_closed=cfg.critic_fail_closed)
    return RagLoop(deps, max_attempts=cfg.max_attempts)
