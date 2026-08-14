"""A small local vector store: numpy matrix + JSON metadata on disk.

Keeping the index local (instead of OpenSearch/Kendra/pgvector on RDS) is
what keeps the AWS bill near zero — the only paid calls are Bedrock model
invocations.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]):
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        # normalize so dot product == cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = vectors / norms
        self.chunks = chunks

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            raise ValueError("query vector has zero norm")
        q = q / q_norm
        scores = self.vectors @ q
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        meta = [c.to_dict() for c in self.chunks]
        (index_dir / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        index_dir = Path(index_dir)
        vectors = np.load(index_dir / "vectors.npy")
        meta = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**m) for m in meta]
        return cls(vectors, chunks)
