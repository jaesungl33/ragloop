"""ragloop -- a pluggable, self-correcting agentic RAG framework.

Quick start::

    from ragloop import build_from_config
    loop = build_from_config("config.yaml")
    print(loop.ask("What is our refund policy?")["answer"])
"""
from .config import Config, build_from_config
from .engine import Deps, RagLoop
from .llm.base import LLMProvider
from .retrieval.base import Document, Retriever

__version__ = "0.1.0"

__all__ = [
    "Config",
    "build_from_config",
    "RagLoop",
    "Deps",
    "LLMProvider",
    "Retriever",
    "Document",
]
