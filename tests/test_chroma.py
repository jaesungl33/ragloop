"""Integration test for the Chroma retriever against a real (in-memory) Chroma.

Skips automatically if chromadb isn't installed, so the core suite stays
dependency-light.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("chromadb")

from ragloop import Document  # noqa: E402
from ragloop.retrieval.chroma_retriever import ChromaRetriever  # noqa: E402


@pytest.fixture
def store():
    r = ChromaRetriever(collection="ragloop_test")
    r.add([
        Document("refunds:0", "You may request a refund within 30 days of delivery."),
        Document("shipping:0", "Shipping is free on orders over $50."),
        # A doc with NO metadata — used to crash Chroma before the doc_id fix.
        Document("warranty:0", "Most products carry a one-year limited warranty."),
    ])
    return r


def test_add_handles_empty_metadata(store):
    # If add() didn't backfill doc_id, the warranty doc would have failed to index.
    assert store.get_chunk("warranty:0") is not None


def test_semantic_search_finds_relevant_chunk(store):
    hits = store.semantic_search("how long do I have to return something", k=3)
    assert any(h.id == "refunds:0" for h in hits)
    assert all(h.score is not None for h in hits)  # distances converted to scores


def test_get_chunk_roundtrip_and_missing(store):
    doc = store.get_chunk("shipping:0")
    assert doc is not None and "free" in doc.text
    assert store.get_chunk("does-not-exist") is None
