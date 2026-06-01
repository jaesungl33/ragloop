"""Chroma implementation of :class:`Retriever`.

Reference backend. ``semantic_search`` uses Chroma's vector index;
``keyword_search`` uses Chroma's ``$contains`` document filter as a simple
lexical match. For production-grade lexical ranking, back ``keyword_search``
with a real BM25 index (e.g. an Elasticsearch or OpenSearch sidecar) -- the
interface stays identical, so the engine is unaffected.
"""
from __future__ import annotations

from typing import List, Optional

from .base import Document, Retriever


class ChromaRetriever(Retriever):
    def __init__(
        self,
        collection: str = "ragloop",
        persist_dir: Optional[str] = None,
    ) -> None:
        import chromadb

        self._client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.Client()
        )
        # Chroma supplies a default embedding function; swap via config in a
        # real deployment to match your domain.
        self._col = self._client.get_or_create_collection(name=collection)

    def add(self, documents: List[Document]) -> None:
        if not documents:
            return
        # Chroma rejects empty metadata dicts, so guarantee every record carries
        # at least its own id. Caller-supplied metadata is preserved and wins.
        self._col.upsert(
            ids=[d.id for d in documents],
            documents=[d.text for d in documents],
            metadatas=[{"doc_id": d.id, **(d.metadata or {})} for d in documents],
        )

    def _to_docs(self, res, with_distance: bool) -> List[Document]:
        out: List[Document] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0] if with_distance else [None] * len(ids)
        for i, _id in enumerate(ids):
            out.append(
                Document(
                    id=_id,
                    text=docs[i] if i < len(docs) else "",
                    metadata=metas[i] if i < len(metas) else {},
                    # Convert distance to a similarity-style score (higher = better).
                    score=(1.0 - dists[i]) if dists[i] is not None else None,
                )
            )
        return out

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        res = self._col.query(query_texts=[query], n_results=k)
        return self._to_docs(res, with_distance=True)

    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        res = self._col.query(
            query_texts=[query],
            n_results=k,
            where_document={"$contains": query},
        )
        return self._to_docs(res, with_distance=True)

    def get_chunk(self, doc_id: str) -> Optional[Document]:
        res = self._col.get(ids=[doc_id])
        ids = res.get("ids") or []
        if not ids:
            return None
        docs = res.get("documents") or [""]
        metas = res.get("metadatas") or [{}]
        return Document(id=ids[0], text=docs[0], metadata=metas[0])
