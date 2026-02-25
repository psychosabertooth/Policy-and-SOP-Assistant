"""Streamlit UI for the Telecom Policy & SOP Assistant."""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Policy & SOP Assistant",
    page_icon="📋",
    layout="wide",
)

import config
from rag.generator import generate_answer
from agent.tools import draft_email, create_checklist, route_to_tool
from ingestion.pipeline import run_ingestion, run_fresh_ingestion
from ingestion.scraper import scrape_custom_url, ATT_URLS
from vector_store.store import reset_vector_store



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None


with st.sidebar:
    st.title("⚙️ Settings")


    provider = st.radio(
        "LLM Provider",
        options=["openai", "azure", "ollama"],
        index=["openai", "azure", "ollama"].index(config.LLM_PROVIDER)
            if config.LLM_PROVIDER in ["openai", "azure", "ollama"] else 0,
        help="OpenAI/Azure require API keys. Ollama runs fully locally.",
    )
    config.LLM_PROVIDER = provider

    if provider == "openai":
        api_key = st.text_input(
            "OpenAI API Key",
            value=config.OPENAI_API_KEY,
            type="password",
            help="Paste your OpenAI key here (or set OPENAI_API_KEY env var).",
        )
        config.OPENAI_API_KEY = api_key

    elif provider == "azure":
        azure_key = st.text_input(
            "Azure OpenAI API Key",
            value=config.AZURE_OPENAI_API_KEY,
            type="password",
        )
        config.AZURE_OPENAI_API_KEY = azure_key
        azure_endpoint = st.text_input(
            "Azure Endpoint",
            value=config.AZURE_OPENAI_ENDPOINT,
            placeholder="https://my-resource.openai.azure.com/",
        )
        config.AZURE_OPENAI_ENDPOINT = azure_endpoint
        azure_deploy = st.text_input(
            "Deployment Name",
            value=config.AZURE_DEPLOYMENT_NAME,
            placeholder="gpt-35-turbo",
        )
        config.AZURE_DEPLOYMENT_NAME = azure_deploy

    st.divider()

    st.subheader("📂 Document Ingestion")

    data_files = list(config.DATA_DIR.glob("*"))
    st.caption(f"Local documents in `/data`: **{len(data_files)}**")
    for f in data_files:
        st.text(f"  📄 {f.name}")

    include_web = st.checkbox(
        "🌐 Include AT&T web scraping",
        value=config.ENABLE_WEB_SCRAPING,
        help=f"Scrape {len(ATT_URLS)} public AT&T policy/support pages",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Full Re-ingest", use_container_width=True):
            with st.spinner("Rebuilding vector store (local docs + web) …"):
                count = run_fresh_ingestion(include_web=include_web)
            st.success(f"Stored {count} chunks in ChromaDB ✓")

    with col2:
        if st.button("🗑️ Clear DB", use_container_width=True):
            reset_vector_store()
            st.info("ChromaDB cleared.")

    st.divider()


    st.subheader("🌐 Scrape Custom URL")
    custom_url = st.text_input(
        "Page URL",
        placeholder="https://www.att.com/support/article/...",
    )
    custom_title = st.text_input(
        "Document title (optional)",
        placeholder="AT&T Support Page",
    )
    scrape_clicked = st.button("➕ Scrape & Add", use_container_width=True)

    if scrape_clicked:
        if custom_url:
            st.session_state.scrape_pipeline = {
                "status": "running",
                "url": custom_url,
                "title": custom_title if custom_title else None,
            }
        else:
            st.warning("Enter a URL first.")
            st.session_state.scrape_pipeline = None

    st.divider()
    st.caption(
        f"Embedding model: `{config.EMBEDDING_MODEL_NAME}`\n\n"
        f"Chunk size: {config.CHUNK_SIZE} tokens | Overlap: {config.CHUNK_OVERLAP}\n\n"
        f"Top-K retrieval: {config.TOP_K}\n\n"
        f"All data is stored and retrieved from **ChromaDB**"
    )


st.title(config.APP_TITLE)
st.caption(config.APP_SUBTITLE)


if "scrape_pipeline" not in st.session_state:
    st.session_state.scrape_pipeline = None

if st.session_state.scrape_pipeline and st.session_state.scrape_pipeline.get("status") == "running":
    from ingestion.chunker import chunk_documents
    from ingestion.embedder import get_embedding_model
    from vector_store.store import get_or_create_vector_store

    scrape_url = st.session_state.scrape_pipeline["url"]
    scrape_title = st.session_state.scrape_pipeline.get("title")

    st.divider()
    st.subheader("🔬 Scrape Pipeline — Step-by-Step Breakdown")


    st.markdown("### Step 1: 🌐 Scraping the URL")
    st.code(scrape_url, language=None)

    with st.spinner("Fetching page content …"):
        doc = scrape_custom_url(scrape_url, scrape_title or None)

    if not doc:
        st.error("❌ **Scraping failed.** The page returned no usable content. "
                 "This can happen if the site blocks scrapers, uses heavy JavaScript rendering, "
                 "or the URL is invalid.")
        st.session_state.scrape_pipeline = None
        st.stop()

    raw_content = doc.page_content
    st.success(f"✅ Scraped **{len(raw_content):,} characters** from the page")

    with st.expander("📄 View raw scraped content", expanded=True):
        st.text_area(
            "Raw text extracted from HTML",
            value=raw_content[:3000] + ("\n\n… [truncated]" if len(raw_content) > 3000 else ""),
            height=250,
            disabled=True,
        )
        st.caption(f"Total length: {len(raw_content):,} characters | "
                   f"Source: `{doc.metadata.get('source', 'N/A')}` | "
                   f"Type: `{doc.metadata.get('type', 'N/A')}`")

    st.divider()


    st.markdown("### Step 2: ✂️ Chunking the Document")
    st.caption(f"Settings — chunk size: **{config.CHUNK_SIZE} tokens** (~{config.CHUNK_SIZE * 4} chars) | "
               f"overlap: **{config.CHUNK_OVERLAP} tokens** (~{config.CHUNK_OVERLAP * 4} chars)")

    with st.spinner("Splitting into chunks …"):
        chunks = chunk_documents([doc])

    st.success(f"✅ Created **{len(chunks)} chunk(s)**")

    with st.expander(f"📦 View all {len(chunks)} chunks", expanded=True):
        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk.page_content)
            approx_tokens = chunk_len // 4
            st.markdown(f"**Chunk {i}** — {chunk_len:,} chars (~{approx_tokens} tokens)")
            st.code(chunk.page_content[:500] + ("\n… [truncated]" if chunk_len > 500 else ""),
                    language=None)
            st.caption(f"Metadata: source=`{chunk.metadata.get('source')}` | "
                       f"chunk_index=`{chunk.metadata.get('chunk_index')}` | "
                       f"type=`{chunk.metadata.get('type', 'local')}`")
            if i < len(chunks) - 1:
                st.markdown("---")

    st.divider()

    st.markdown("### Step 3: 💾 Embedding & Storing in ChromaDB")
    st.caption(f"Embedding model: **{config.EMBEDDING_MODEL_NAME}** | "
               f"Collection: **{config.CHROMA_COLLECTION}** | "
               f"Persist dir: `{config.CHROMA_DIR}`")

    with st.spinner("Generating embeddings and writing to ChromaDB …"):
        emb = get_embedding_model()
        sample_vector = emb.embed_query(chunks[0].page_content[:200])

        vs = get_or_create_vector_store(emb)
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vs.add_texts(texts=texts, metadatas=metadatas)

    st.success(f"✅ **{len(chunks)} chunk(s)** embedded and stored in ChromaDB")

    with st.expander("🔢 View embedding details", expanded=True):
        st.markdown(f"**Vector dimensions:** {len(sample_vector)}")
        st.markdown(f"**Sample vector** (first chunk, first 20 values):")
        st.code(str([round(v, 6) for v in sample_vector[:20]]) + " …", language="python")
        st.caption(f"Each chunk is converted into a {len(sample_vector)}-dimensional vector "
                   f"for similarity search in ChromaDB.")

    st.divider()
    st.markdown("### ✅ Pipeline Complete!")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Chars Scraped", f"{len(raw_content):,}")
    col_b.metric("Chunks Created", len(chunks))
    col_c.metric("Vectors Stored", len(chunks))
    st.info("💡 You can now ask questions about this content in the chat below.")

    st.session_state.scrape_pipeline = {"status": "done"}

    st.divider()


with st.expander("💡 Example questions you can ask"):
    st.markdown(
        """
        - What are the consequences of violating the acceptable use policy?
        - How should a Severity 1 network outage be handled?
        - What training is required in the first week of onboarding?
        - Can I use my personal phone to access company email?
        - How long are billing records retained?
        - Who do I contact to report a data breach?
        """
    )



for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 View Sources & Citations"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']}** (chunk {src['chunk_index']}) "
                        f"— relevance: {src['score']}"
                    )
                    st.code(src["preview"], language=None)


query = st.chat_input("Ask a question about company policies or SOPs …")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating answer …"):
            result = generate_answer(query)

        st.markdown(result["answer"])
        st.session_state.last_answer = result

        if result["sources"]:
            with st.expander("📎 View Sources & Citations"):
                for src in result["sources"]:
                    st.markdown(
                        f"**{src['source']}** (chunk {src['chunk_index']}) "
                        f"— relevance: {src['score']}"
                    )
                    st.code(src["preview"], language=None)

        if not result["has_info"]:
            st.info(
                "ℹ️ The retrieved documents had low relevance scores for this "
                "query. The answer may be less precise — verify with the "
                "original policy documents if needed."
            )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        }
    )


if st.session_state.last_answer and st.session_state.last_answer.get("has_info"):
    st.divider()
    st.subheader("🛠️ Agent Tools")
    st.caption(
        "Transform the last answer into actionable formats. "
        "Use **Auto-Route** to let the LLM decide which tool fits best."
    )

    tool_col1, tool_col2, tool_col3 = st.columns(3)

    with tool_col1:
        if st.button("📧 Draft Email Summary", use_container_width=True):
            with st.spinner("Drafting email …"):
                email = draft_email(st.session_state.last_answer["answer"])
            st.text_area("Draft Email", value=email, height=300)

    with tool_col2:
        if st.button("✅ Create Checklist", use_container_width=True):
            with st.spinner("Creating checklist …"):
                checklist = create_checklist(st.session_state.last_answer["answer"])
            st.text_area("Checklist", value=checklist, height=300)

    with tool_col3:
        if st.button("🤖 Auto-Route (Agent)", use_container_width=True):
            last_query = ""
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_query = msg["content"]
                    break

            with st.spinner("Agent deciding which tool to use …"):
                tool_choice = route_to_tool(last_query)

            if tool_choice == "email":
                st.info("🤖 Agent chose: **Draft Email**")
                with st.spinner("Drafting email …"):
                    output = draft_email(st.session_state.last_answer["answer"])
                st.text_area("Agent Output (Email)", value=output, height=300)
            elif tool_choice == "checklist":
                st.info("🤖 Agent chose: **Create Checklist**")
                with st.spinner("Creating checklist …"):
                    output = create_checklist(st.session_state.last_answer["answer"])
                st.text_area("Agent Output (Checklist)", value=output, height=300)
            else:
                st.info("🤖 Agent decided: **No tool needed** — the answer above is sufficient.")
