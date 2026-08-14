import numpy as np
import pytest

from doc_qa_agent.chunking import Chunk
from doc_qa_agent.vector_store import VectorStore


def make_store():
    chunks = [
        Chunk("a.md#0", "a.md", "alpha text"),
        Chunk("a.md#1", "a.md", "beta text"),
        Chunk("b.md#0", "b.md", "gamma text"),
    ]
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]], dtype=np.float32
    )
    return VectorStore(vectors, chunks)


def test_search_returns_most_similar_first():
    store = make_store()
    results = store.search(np.array([1.0, 0.05, 0.0]), top_k=3)
    assert results[0].chunk.chunk_id == "a.md#0"
    assert results[0].score >= results[1].score >= results[2].score


def test_top_k_is_capped_at_store_size():
    store = make_store()
    assert len(store.search(np.array([1.0, 0.0, 0.0]), top_k=10)) == 3


def test_zero_query_vector_raises():
    store = make_store()
    with pytest.raises(ValueError):
        store.search(np.zeros(3))


def test_save_and_load_roundtrip(tmp_path):
    store = make_store()
    store.save(tmp_path / "idx")
    loaded = VectorStore.load(tmp_path / "idx")
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in store.chunks]
    original = store.search(np.array([0.6, 0.8, 0.0]), top_k=2)
    reloaded = loaded.search(np.array([0.6, 0.8, 0.0]), top_k=2)
    assert [r.chunk.chunk_id for r in original] == [r.chunk.chunk_id for r in reloaded]
