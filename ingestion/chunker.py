"""Splits documents into overlapping chunks for embedding and vector search."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def create_splitter() -> RecursiveCharacterTextSplitter:
    """Build a text splitter tuned for policy/SOP documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,       # approximate chars
        chunk_overlap=config.CHUNK_OVERLAP * 4,  # overlap in chars
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split a list of Documents into smaller chunks with chunk_index metadata."""
    splitter = create_splitter()
    chunks: List[Document] = []

    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = idx
        chunks.extend(doc_chunks)

    print(f"✅ Created {len(chunks)} chunks from {len(documents)} document(s)")
    return chunks

