import pytest

from doc_qa_agent.chunking import chunk_directory, chunk_text


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "x" * 200 for i in range(20))
    chunks = chunk_text(text, "doc.md", chunk_size=500, overlap=100)
    assert chunks
    assert all(len(c.text) <= 500 for c in chunks)


def test_chunk_ids_are_stable_and_carry_doc_name():
    chunks = chunk_text("a\n\nb\n\nc", "notes.md", chunk_size=3, overlap=1)
    assert [c.chunk_id for c in chunks] == [f"notes.md#{i}" for i in range(len(chunks))]
    assert all(c.doc_name == "notes.md" for c in chunks)


def test_long_paragraph_is_hard_split():
    chunks = chunk_text("y" * 5000, "big.md", chunk_size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("hello", "d.md", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", "d.md", chunk_size=100, overlap=100)


def test_chunk_directory_reads_sample_docs():
    chunks = chunk_directory("data/sample_docs")
    doc_names = {c.doc_name for c in chunks}
    assert {"aurora_platform.md", "billing_faq.md", "security_policy.md"} <= doc_names
