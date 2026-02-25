"""Retrieves relevant document chunks from ChromaDB for a given query."""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from ingestion.embedder import get_embedding_model
from vector_store.store import get_or_create_vector_store
import config


def get_retriever() -> Chroma:
    """Load the existing vector store for querying."""
    embedding_model = get_embedding_model()
    return get_or_create_vector_store(embedding_model)


def retrieve_relevant_chunks(
    query: str,
    top_k: int = config.TOP_K,
) -> List[Tuple[Document, float]]:
    """Similarity search returning top-k (Document, score) tuples."""
    vectorstore = get_retriever()
    results = vectorstore.similarity_search_with_relevance_scores(
        query, k=top_k
    )
    return results


def format_context(results: List[Tuple[Document, float]]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM prompt."""
    if not results:
        return ""

    context_parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        context_parts.append(
            f"[Source {i}: {source} (chunk {chunk_idx}, relevance: {score:.2f})]\n"
            f"{doc.page_content}\n"
        )

    return "\n---\n".join(context_parts)


def has_relevant_results(
    results: List[Tuple[Document, float]],
    threshold: float = 0.05,
) -> bool:
    """Soft relevance check used as a UI confidence indicator."""
    if not results:
        return False
    best_score = max(score for _, score in results)
    return best_score >= threshold

