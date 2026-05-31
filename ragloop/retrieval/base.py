"""Abstract retrieval interface.

The agent reaches the corpus only through these three operations. A company
backs them with whatever store they already run -- Chroma, pgvector, Pinecone,
Elasticsearch -- by subclassing :class:`Retriever`. The hierarchy (keyword /
semantic / chunk) mirrors how an agent actually searches: broad lexical match,
meaning-based match, then full-context read of a promising hit.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


class Retriever(ABC):
    """Three retrieval granularities exposed to the agent."""

    @abstractmethod
    def add(self, documents: List[Document]) -> None:
        """Index a batch of documents (idempotent on ``id``)."""

    @abstractmethod
    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Dense / embedding-based search. Best for meaning and paraphrase."""

    @abstractmethod
    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """Lexical search. Best for exact terms, codes, names, identifiers."""

    @abstractmethod
    def get_chunk(self, doc_id: str) -> Optional[Document]:
        """Fetch one document by id for full-context reading."""
