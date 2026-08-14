"""Split documents into overlapping chunks for embedding.

Chunks are cut at paragraph boundaries where possible so a retrieved chunk
reads as coherent text, and each chunk carries a stable id
("<doc_name>#<n>") that the agent cites in its answers.
"""

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str   # e.g. "billing_faq.md#3"
    doc_name: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_text(text: str, doc_name: str, chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Greedily pack paragraphs into chunks of at most ``chunk_size`` chars.

    A paragraph longer than ``chunk_size`` is split hard. Consecutive chunks
    share roughly ``overlap`` characters so facts near a boundary are not
    lost to either side.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        while len(p) > chunk_size:
            pieces.append(p[:chunk_size])
            p = p[chunk_size - overlap:]
        pieces.append(p)

    chunks: list[Chunk] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(_make_chunk(current, doc_name, len(chunks)))
            # carry a tail of the previous chunk forward as overlap
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}".strip()
            if len(current) > chunk_size:
                current = current[-chunk_size:]
        else:
            current = piece
    if current:
        chunks.append(_make_chunk(current, doc_name, len(chunks)))
    return chunks


def _make_chunk(text: str, doc_name: str, index: int) -> Chunk:
    return Chunk(chunk_id=f"{doc_name}#{index}", doc_name=doc_name, text=text.strip())


def chunk_directory(docs_dir: str | Path, chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Chunk every .md / .txt file in a directory."""
    docs_dir = Path(docs_dir)
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_text(text, path.name, chunk_size=chunk_size, overlap=overlap))
    if not chunks:
        raise ValueError(f"No .md or .txt documents found in {docs_dir}")
    return chunks
