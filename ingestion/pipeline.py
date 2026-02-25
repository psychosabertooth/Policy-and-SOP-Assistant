"""End-to-end ingestion pipeline: load, scrape, chunk, embed, and store."""

import config
from ingestion.loader import load_all_documents
from ingestion.scraper import scrape_att_pages
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model
from vector_store.store import get_or_create_vector_store, reset_vector_store


def run_ingestion(include_web: bool = True) -> int:
    """Execute the full ingestion pipeline. Returns the number of chunks stored."""
    all_documents = []

    print("\n📂 Step 1: Loading local documents …")
    local_docs = load_all_documents()
    all_documents.extend(local_docs)
    print(f"   → {len(local_docs)} local document(s)")

    if include_web and config.ENABLE_WEB_SCRAPING:
        print("\n🌐 Step 2: Scraping AT&T website …")
        try:
            web_docs = scrape_att_pages()
            all_documents.extend(web_docs)
            print(f"   → {len(web_docs)} web page(s) scraped successfully")
        except Exception as e:
            print(f"   ⚠️  Web scraping failed (continuing with local docs): {e}")
    else:
        print("\n⏭️  Step 2: Web scraping skipped")

    if not all_documents:
        print("❌ No documents to process. Add files to /data and retry.")
        return 0

    print(f"\n✂️  Step 3: Chunking {len(all_documents)} document(s) …")
    chunks = chunk_documents(all_documents)

    print("\n💾 Step 4: Embedding and storing in ChromaDB …")
    embedding_model = get_embedding_model()
    vectorstore = get_or_create_vector_store(embedding_model)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vectorstore.add_texts(texts=texts, metadatas=metadatas)

    local_count = sum(1 for c in chunks if c.metadata.get("type") != "web_scrape")
    web_count = sum(1 for c in chunks if c.metadata.get("type") == "web_scrape")
    print(f"\n🎉 Ingestion complete!")
    print(f"   Total chunks stored : {len(chunks)}")
    print(f"   From local files    : {local_count}")
    print(f"   From web scraping   : {web_count}")

    return len(chunks)


def run_fresh_ingestion(include_web: bool = True) -> int:
    """Reset the vector store and re-ingest everything from scratch."""
    print("\n🗑️  Clearing existing ChromaDB data …")
    reset_vector_store()
    return run_ingestion(include_web=include_web)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--no-web", action="store_true",
        help="Skip web scraping (local documents only)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear ChromaDB before ingesting (full rebuild)",
    )
    args = parser.parse_args()

    if args.fresh:
        run_fresh_ingestion(include_web=not args.no_web)
    else:
        run_ingestion(include_web=not args.no_web)
