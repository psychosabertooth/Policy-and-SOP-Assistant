"""Local embedding model (sentence-transformers) for document chunks."""

from langchain_huggingface import HuggingFaceEmbeddings

import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a LangChain-compatible local embedding model."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

