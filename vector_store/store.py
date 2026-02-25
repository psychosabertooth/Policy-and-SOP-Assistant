"""Manages the ChromaDB vector store."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config


def get_or_create_vector_store(
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """Return a ChromaDB-backed LangChain vector store."""
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embedding_model,
        persist_directory=str(config.CHROMA_DIR),
    )
    return vectorstore


def reset_vector_store() -> None:
    """Delete persisted ChromaDB data so the next ingestion starts fresh."""
    import shutil

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        print(f"🗑️  Deleted existing ChromaDB at {config.CHROMA_DIR}")
    else:
        print("ℹ️  No existing ChromaDB found — nothing to delete.")

