"""Loads raw documents (.txt, .pdf) from the /data directory."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)

import config


def load_text_files(directory: Path = config.DATA_DIR) -> List[Document]:
    """Load all .txt files from the given directory."""
    docs: List[Document] = []
    for txt_path in sorted(directory.glob("*.txt")):
        loader = TextLoader(str(txt_path), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = txt_path.name
        docs.extend(loaded)
    return docs


def load_pdf_files(directory: Path = config.DATA_DIR) -> List[Document]:
    """Load all .pdf files from the given directory (one Document per page)."""
    docs: List[Document] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = pdf_path.name
        docs.extend(loaded)
    return docs


def load_all_documents(directory: Path = config.DATA_DIR) -> List[Document]:
    """
    Master loader — loads every supported file type from /data.
    Returns a flat list of LangChain Document objects.
    """
    docs: List[Document] = []
    docs.extend(load_text_files(directory))
    docs.extend(load_pdf_files(directory))

    if not docs:
        print(f"⚠️  No documents found in {directory}")
    else:
        print(f"✅ Loaded {len(docs)} document(s) from {directory}")

    return docs

