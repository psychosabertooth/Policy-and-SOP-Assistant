# 📋 Telecom Policy & SOP Assistant

An AI-powered Retrieval Augmented Generation (RAG) system that helps employees quickly find answers from company policies and standard operating procedures.

Built as a **production-inspired prototype** using open-source tools that run locally.

---

## Architecture

```
SOPasst.building/
│
├── config.py              # Central configuration (models, paths, params)
├── app.py                 # Streamlit UI — the main entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
│
├── data/                  # Raw policy & SOP documents
│   ├── acceptable_use_policy.txt
│   ├── customer_data_privacy_policy.txt
│   ├── employee_onboarding_sop.txt
│   └── network_outage_sop.txt
│
├── ingestion/             # Document processing pipeline
│   ├── loader.py          # Loads .txt and .pdf files
│   ├── chunker.py         # Splits documents into overlapping chunks
│   ├── embedder.py        # Local embedding model (sentence-transformers)
│   └── pipeline.py        # End-to-end: load → chunk → embed → store
│
├── vector_store/          # Vector database
│   ├── store.py           # ChromaDB setup and helpers
│   └── chroma_db/         # Persisted vector data (auto-created)
│
├── rag/                   # Retrieval + Generation
│   ├── retriever.py       # Similarity search against ChromaDB
│   ├── prompts.py         # Prompt templates with citation control
│   └── generator.py       # LLM answer generation (OpenAI / Ollama)
│
└── agent/                 # Optional multi-step tools
    └── tools.py           # Draft email, create checklist from answers
```

## Tech Stack

| Component         | Technology                        |
|-------------------|-----------------------------------|
| Embeddings        | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Vector Store      | ChromaDB (local, persistent)      |
| LLM               | OpenAI GPT-3.5 **or** Ollama (Mistral/Llama 3) |
| Orchestration     | LangChain                         |
| UI                | Streamlit                          |
| Documents         | Plain text / PDF                   |

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure LLM provider

**Option A — OpenAI (easiest for demo):**
```bash
# Copy env template and add your key
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```
Or set it directly:
```bash
set OPENAI_API_KEY=sk-your-key-here    # Windows
export OPENAI_API_KEY=sk-your-key-here  # macOS/Linux
```

**Option B — Ollama (fully local, no API key):**
1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull mistral`
3. Set `LLM_PROVIDER=ollama` in `.env` or `config.py`

### 4. Ingest documents

```bash
python -m ingestion.pipeline
```

This loads the sample documents from `/data`, chunks them, generates embeddings, and stores everything in ChromaDB.

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

```
User Query
    │
    ▼
┌──────────────┐
│  Embedding   │  ← same model used during ingestion
│  (MiniLM)    │
└──────┬───────┘
       │ query vector
       ▼
┌──────────────┐
│  ChromaDB    │  ← similarity search (top-K)
│  Vector DB   │
└──────┬───────┘
       │ relevant chunks + scores
       ▼
┌──────────────┐
│ Relevance    │  ← hallucination guard (score threshold)
│ Check        │
└──────┬───────┘
       │ if relevant
       ▼
┌──────────────┐
│  LLM Prompt  │  ← context + question + citation rules
│  (RAG)       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Answer +    │  ← with [Source N] citations
│  Citations   │
└──────────────┘
```

## Key Design Decisions

### Hallucination Guard
If retrieved chunks score below a relevance threshold (0.25), the system returns a "no information found" message instead of generating a potentially hallucinated response. See `rag/retriever.py` → `has_relevant_results()`.

### Citation Control
The prompt template strictly instructs the LLM to:
- Only use provided context
- Cite sources with `[Source N]` notation
- Refuse to answer if context is insufficient

### Chunking Strategy
- **Size:** ~600 tokens (2400 characters) — within the 500-800 sweet spot
- **Overlap:** 100 tokens — preserves context across chunk boundaries
- **Splitter:** Recursive, preferring paragraph → sentence → word boundaries

### Ingestion/Query Separation
Ingestion (`python -m ingestion.pipeline`) runs separately from the query path (`streamlit run app.py`). This mirrors production systems where document processing is a batch job and querying is real-time.

## Example Questions

- "What are the consequences of violating the acceptable use policy?"
- "How should a Severity 1 network outage be handled?"
- "What training must a new employee complete in their first week?"
- "Can I access company email on my personal phone?"
- "How long are call detail records retained?"
- "Who do I contact to report a suspected data breach?"

## Sample Documents

The `/data` folder includes 4 realistic telecom company documents:

1. **Acceptable Use Policy** — rules for network/device usage, BYOD, violations
2. **Customer Data Privacy Policy** — PII, CPNI, PCI handling, breach notification
3. **Network Outage SOP** — severity levels, troubleshooting steps, communication
4. **Employee Onboarding SOP** — pre-arrival through 90-day review

## Extending the System

**Add new documents:** Drop `.txt` or `.pdf` files into `/data` and re-run ingestion.

**Switch embedding model:** Change `EMBEDDING_MODEL_NAME` in `config.py` (e.g., to `BAAI/bge-small-en-v1.5`). Remember to re-ingest after changing.

**Add web scraping:** Extend `ingestion/loader.py` with a `load_web_pages()` function using `BeautifulSoup`.

**Production upgrades:**
- Swap ChromaDB for Pinecone / Weaviate / Qdrant
- Add authentication
- Containerise with Docker
- Add evaluation metrics (RAGAS)
