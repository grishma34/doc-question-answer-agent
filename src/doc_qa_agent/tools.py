"""Agent tools. The only tool is top-k vector search over the document index."""

from langchain_core.tools import tool

from .embeddings import BedrockEmbedder
from .vector_store import VectorStore


def make_search_tool(store: VectorStore, embedder: BedrockEmbedder, top_k: int = 4):
    """Build the search_documents tool bound to a loaded index."""

    @tool
    def search_documents(query: str) -> str:
        """Search the indexed documents and return the most relevant passages.

        Use this before answering any question about the documents. Each
        passage is labeled with a source id like [source: doc.md#3] — cite
        those ids in your answer. Call again with a rephrased query if the
        first results don't contain the answer.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        results = store.search(embedder.embed(query), top_k=top_k)
        blocks = [
            f"[source: {r.chunk.chunk_id}] (score={r.score:.3f})\n{r.chunk.text}"
            for r in results
        ]
        return "\n\n---\n\n".join(blocks)

    return search_documents
