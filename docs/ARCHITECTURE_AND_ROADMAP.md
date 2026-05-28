# AI Document Intelligence Platform — Architecture, Evaluation & Roadmap

> Comprehensive analysis of the existing SOP Assistant codebase and a complete plan to transform it into a production-quality AI Document Intelligence Platform.

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Production Readiness Evaluation](#2-production-readiness-evaluation)
3. [Redesigned Production Architecture](#3-redesigned-production-architecture)
4. [Extended Features Specification](#4-extended-features-specification)
5. [Production-Grade Engineering](#5-production-grade-engineering)
6. [Deployment Strategy](#6-deployment-strategy)
7. [Clean Project Structure](#7-clean-project-structure)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [AI Engineer Portfolio Enhancements](#9-ai-engineer-portfolio-enhancements)
10. [Prioritized Feature List & Next Steps](#10-prioritized-feature-list--next-steps)

---

## 1. Current Architecture Analysis

### 1.1 Architecture Overview

The system is a **monolithic Streamlit application** implementing a Retrieval-Augmented Generation (RAG) pipeline for telecom SOP/policy question answering. Everything runs in a single process — UI, ingestion, retrieval, and generation.

**Data Flow:**
```
Documents (.txt/.pdf) + Web Scraping
        ↓
    Loader (loader.py)
        ↓
    Chunker (chunker.py) — RecursiveCharacterTextSplitter
        ↓
    Embedder (embedder.py) — all-MiniLM-L6-v2
        ↓
    Vector Store (store.py) — ChromaDB local
        ↓
    User Query → Retriever (retriever.py) → top-K similarity search
        ↓
    Prompt Assembly (prompts.py) — context + question
        ↓
    LLM Generator (generator.py) — OpenAI / Azure / Ollama
        ↓
    Answer with Citations → Agent Tools (email / checklist)
```

### 1.2 Core Modules & Responsibilities

| Module | File(s) | Responsibility |
|--------|---------|---------------|
| **UI Layer** | `app.py` | Streamlit UI: chat interface, sidebar config, scrape visualization, agent tool buttons |
| **Configuration** | `config.py` | Central settings: paths, model names, chunk params, API keys, provider selection |
| **Document Loading** | `ingestion/loader.py` | Load `.txt` and `.pdf` files from `/data` directory using LangChain loaders |
| **Web Scraping** | `ingestion/scraper.py` | Scrape AT&T policy pages, clean HTML to text, produce LangChain Documents |
| **Chunking** | `ingestion/chunker.py` | Split documents into overlapping chunks using `RecursiveCharacterTextSplitter` |
| **Embedding** | `ingestion/embedder.py` | Generate embeddings via `all-MiniLM-L6-v2` (HuggingFace, runs on CPU) |
| **Pipeline Orchestration** | `ingestion/pipeline.py` | End-to-end: load → scrape → chunk → embed → store in ChromaDB |
| **Vector Store** | `vector_store/store.py` | ChromaDB collection management: create, persist, reset |
| **Retrieval** | `rag/retriever.py` | Similarity search against ChromaDB, format context, relevance scoring |
| **Generation** | `rag/generator.py` | LLM-based answer generation with multi-provider support |
| **Prompts** | `rag/prompts.py` | Prompt templates: RAG answer, email draft, checklist, agent router |
| **Agent Tools** | `agent/tools.py` | Post-answer tools: draft email, create checklist, LLM-based intent router |

### 1.3 ML/AI Components

| Component | Technology | Details |
|-----------|-----------|---------|
| **Embeddings** | `all-MiniLM-L6-v2` via `sentence-transformers` | 384-dim vectors, runs locally on CPU |
| **Vector DB** | ChromaDB (local, persistent) | Single collection `telecom_policies`, file-backed |
| **LLM (primary)** | OpenAI GPT-3.5-turbo | Via `langchain-openai`, temp=0.2 for answers, 0.3 for tools |
| **LLM (azure)** | Azure OpenAI | Same model via Azure endpoint |
| **LLM (local)** | Ollama (Mistral) | Fully local inference on `localhost:11434` |
| **Orchestration** | LangChain | Prompt templates, output parsers, LCEL chains |
| **Text Splitting** | `RecursiveCharacterTextSplitter` | Chunk size ~2400 chars, overlap ~400 chars |
| **Web Scraping** | BeautifulSoup4 + requests | Manual HTML → text extraction |

### 1.4 What Works Well

- **Multi-provider LLM support** — seamless switching between OpenAI, Azure, and Ollama
- **Clean separation** of ingestion, RAG, and agent modules
- **Citation-based answers** with source tracking and relevance scores
- **Interactive scrape pipeline visualization** in the Streamlit UI
- **Configurable parameters** via `config.py` and environment variables
- **Agent tools** for email drafting and checklist generation with LLM-based routing

---

## 2. Production Readiness Evaluation

### 2.1 Critical Gaps

| Area | Current State | Production Requirement | Severity |
|------|--------------|----------------------|----------|
| **API Layer** | None — Streamlit is the only interface | RESTful/GraphQL API for programmatic access | 🔴 Critical |
| **Authentication** | None — anyone can access | JWT/OAuth2, user accounts, RBAC | 🔴 Critical |
| **Error Handling** | Minimal `try/except` in scraper only | Structured error handling, retry logic, circuit breakers | 🔴 Critical |
| **Logging** | `print()` statements only | Structured logging (JSON), log levels, correlation IDs | 🔴 Critical |
| **Security** | API keys in config/env, no input validation | Secret management, input sanitization, rate limiting | 🔴 Critical |
| **Database** | No relational DB — metadata only in ChromaDB | PostgreSQL for users, documents, audit trails | 🟠 High |
| **Background Processing** | Everything synchronous in UI thread | Task queues (Celery/Redis) for ingestion, embedding | 🟠 High |
| **File Storage** | Local filesystem only | Object storage (S3/MinIO) for uploaded documents | 🟠 High |
| **Scalability** | Single-process, single-machine | Horizontal scaling, load balancing, worker pools | 🟠 High |
| **Monitoring** | None | Metrics, health checks, alerting, tracing | 🟠 High |
| **Testing** | No tests | Unit tests, integration tests, RAG evaluation | 🟡 Medium |
| **Caching** | None | Response caching, embedding caching, query caching | 🟡 Medium |
| **Document Support** | `.txt` and `.pdf` only | DOCX, images (OCR), HTML, Markdown, CSV | 🟡 Medium |
| **Multi-tenancy** | Single collection for all data | Per-user/per-org document isolation | 🟡 Medium |
| **Deployment** | Manual `streamlit run` | Docker, CI/CD, environment management | 🟡 Medium |

### 2.2 Code-Level Issues

1. **Duplicated LLM initialization** — `_get_llm()` is copy-pasted in both `generator.py` and `tools.py`
2. **Mutable global config** — `config.py` values are mutated at runtime via Streamlit sidebar (`config.LLM_PROVIDER = provider`)
3. **No input validation** — user queries and URLs are passed directly to LLM and HTTP requests
4. **Blocking I/O in UI** — web scraping and embedding run synchronously, blocking the Streamlit event loop
5. **No document deduplication** — re-ingesting adds duplicate chunks to ChromaDB
6. **Hard-coded scrape targets** — AT&T URLs are baked into source code
7. **No retry logic** — failed LLM calls or scrapes fail silently or crash
8. **`shutil.rmtree` for reset** — destructive filesystem operation with no confirmation or backup

---

## 3. Redesigned Production Architecture

### 3.1 Architecture Overview

The production system is decomposed into **6 layers**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React / Next.js)                      │
│  Document Upload │ Search & Chat │ Collections │ Admin Dashboard    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTPS
┌───────────────────────────▼─────────────────────────────────────────┐
│                    API GATEWAY (Nginx / Traefik)                    │
│            Rate Limiting │ TLS Termination │ Load Balancing         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                     FASTAPI BACKEND                                 │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │Auth Svc  │  │ Document API │  │ Search API │  │  Admin API   │  │
│  │JWT/OAuth │  │ CRUD/Upload  │  │ Query/RAG  │  │  Users/Roles │  │
│  └──────────┘  └──────┬───────┘  └─────┬──────┘  └──────────────┘  │
└─────────────────────────┼──────────────┼────────────────────────────┘
                          │              │
         ┌────────────────▼──┐     ┌─────▼────────────────────┐
         │   TASK QUEUE      │     │   AI / ML LAYER          │
         │  Redis + Celery   │     │  Embedding Service       │
         │  ┌─────────────┐  │     │  LLM Service             │
         │  │Ingest Worker│  │     │  Re-Ranker (optional)    │
         │  │Scrape Worker│  │     │  OCR Service (Tesseract) │
         │  │Embed Worker │  │     └──────────────────────────┘
         │  └─────────────┘  │
         └───────────────────┘
                  │
┌─────────────────▼──────────────────────────────────────────────────┐
│                      STORAGE LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ PostgreSQL   │  │ ChromaDB /   │  │ MinIO / S3               │ │
│  │ Users, Docs, │  │ Weaviate     │  │ Document Files           │ │
│  │ Metadata,    │  │ Vector       │  │ (PDF, DOCX, images)      │ │
│  │ Audit Logs   │  │ Embeddings   │  │                          │ │
│  └──────────────┘  └──────────────┘  └───────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────────────────────────┐
│                    OBSERVABILITY                                    │
│  Structured Logging (JSON) │ Prometheus + Grafana │ OpenTelemetry  │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

#### Frontend Layer (React / Next.js)
- **Document Upload** — drag-and-drop PDF/DOCX/image upload with progress bar
- **Search & Chat** — conversational interface with citation highlights
- **Collections** — manage document groups, tags, and versions
- **Admin Dashboard** — user management, system metrics, ingestion status

#### API Gateway (Nginx or Traefik)
- TLS termination and HTTPS enforcement
- Rate limiting (per-user and global)
- Load balancing across API replicas
- Request/response logging

#### FastAPI Backend
- **Auth Service** — JWT token issuance, OAuth2 flows, RBAC middleware
- **Document API** — upload, CRUD, tagging, versioning, collection management
- **Search API** — Query endpoint, RAG pipeline invocation, citation formatting
- **Admin API** — user management, system health, configuration

#### Async Workers (Celery + Redis)
- **Ingestion Worker** — parse documents, extract text, chunk, generate embeddings
- **Scrape Worker** — web content fetching and processing
- **Batch Worker** — bulk imports, re-indexing operations
- Task status tracking, retries with exponential backoff, dead-letter queues

#### AI/ML Layer
- **Embedding Service** — wraps sentence-transformers, batched inference, GPU support
- **LLM Service** — unified interface to OpenAI / Azure / Ollama with fallback chains
- **Re-Ranker** — cross-encoder re-ranking of retrieved chunks for higher precision
- **OCR Service** — Tesseract / EasyOCR for image-based document extraction

#### Storage Layer
- **PostgreSQL** — users, documents metadata, collections, tags, versions, audit log
- **ChromaDB / Weaviate** — vector embeddings with collection-level isolation
- **MinIO / S3** — original document files, intermediate processing artifacts

#### Observability
- **Structured Logging** — JSON logs via `structlog`, correlation IDs per request
- **Metrics** — Prometheus counters/histograms for latency, throughput, queue depth
- **Tracing** — OpenTelemetry spans across API → worker → LLM calls

---

## 4. Extended Features Specification

### 4.1 Document Ingestion

```python
# Supported formats and their processors
DOCUMENT_PROCESSORS = {
    ".pdf":  "PyPDFLoader + OCR fallback",
    ".docx": "python-docx / Unstructured",
    ".doc":  "python-docx / LibreOffice conversion",
    ".txt":  "TextLoader (UTF-8)",
    ".md":   "MarkdownLoader",
    ".html": "BeautifulSoup extraction",
    ".csv":  "Row-based document creation",
    ".png/.jpg/.tiff": "Tesseract OCR → text extraction",
}
```

**Features:**
- Upload via API with multipart/form-data
- Automatic format detection and routing to appropriate parser
- OCR fallback for scanned PDFs (detect image-only pages)
- Metadata extraction: title, author, creation date, page count
- Document fingerprinting (SHA-256) for deduplication
- Configurable chunking strategies per document type
- Progress tracking via WebSocket or polling

### 4.2 Search & Q/A

- **Semantic search** — vector similarity with configurable top-K
- **Hybrid search** — combine vector similarity + BM25 keyword matching
- **Re-ranking** — cross-encoder re-ranking of candidate chunks
- **RAG pipeline** — retrieve → re-rank → assemble context → generate
- **Citation-based answers** — inline `[Source N]` references with clickable links
- **Conversation memory** — multi-turn context with sliding window
- **Query refinement** — automatic query expansion for better recall
- **Streaming responses** — SSE/WebSocket for real-time LLM token streaming

### 4.3 Knowledge Management

- **Collections** — group documents by project, department, or topic
- **Document tagging** — user-defined tags with tag-based filtering
- **Document versioning** — upload new versions, keep history, diff summaries
- **Access control** — per-collection and per-document permissions
- **Search scoping** — restrict searches to specific collections

### 4.4 User Features

- **User accounts** — registration, login, profile management
- **Roles** — Admin, Editor, Viewer with granular permissions
- **Document ownership** — uploaded documents belong to users/orgs
- **API keys** — per-user API keys for programmatic access
- **Audit trail** — who uploaded, queried, or modified what and when

### 4.5 Performance

- **Response caching** — Redis cache for repeated queries (TTL-based)
- **Embedding caching** — cache computed embeddings to avoid re-computation
- **Async processing** — all ingestion jobs run in background workers
- **Batch ingestion** — upload multiple documents in a single request
- **Connection pooling** — database and HTTP connection pools
- **Streaming** — LLM response streaming for improved perceived latency

---

## 5. Production-Grade Engineering

### 5.1 FastAPI Service Structure

```python
# backend/api/main.py — Application factory pattern
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import auth, documents, search, admin, health
from backend.api.middleware import RateLimitMiddleware, RequestIDMiddleware
from backend.core.config import settings
from backend.core.database import init_db

def create_app() -> FastAPI:
    app = FastAPI(
        title="Document Intelligence API",
        version="2.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # Middleware
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, prefix="/api/health", tags=["health"])
    app.include_router(auth.router,   prefix="/api/auth",   tags=["auth"])
    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    app.include_router(admin.router,  prefix="/api/admin",  tags=["admin"])

    @app.on_event("startup")
    async def startup():
        await init_db()

    return app
```

### 5.2 Background Workers (Celery + Redis)

```python
# backend/workers/tasks.py
from celery import Celery
from backend.core.config import settings

celery_app = Celery("doc_intel", broker=settings.REDIS_URL)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_document(self, document_id: str, file_path: str):
    """Background task: parse, chunk, embed, and store a document."""
    try:
        # 1. Parse document (PDF/DOCX/image → text)
        text = parse_document(file_path)

        # 2. Chunk text
        chunks = chunk_text(text, document_id)

        # 3. Generate embeddings
        embeddings = generate_embeddings(chunks)

        # 4. Store in vector DB
        store_vectors(document_id, chunks, embeddings)

        # 5. Update metadata in PostgreSQL
        update_document_status(document_id, status="ready")

    except Exception as exc:
        update_document_status(document_id, status="failed", error=str(exc))
        raise self.retry(exc=exc)
```

### 5.3 Vector Database

ChromaDB remains suitable for development and small-to-medium deployments. For enterprise scale:

| Scale | Recommendation |
|-------|---------------|
| Dev / Portfolio | ChromaDB (local, zero config) |
| Small production (<100K docs) | ChromaDB server mode |
| Medium (100K–1M docs) | Weaviate or Qdrant (self-hosted) |
| Large (>1M docs) | Pinecone or Weaviate Cloud |

### 5.4 PostgreSQL Metadata Schema

```sql
-- Core tables
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'viewer',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) UNIQUE,          -- SHA-256 for dedup
    file_size BIGINT,
    mime_type VARCHAR(100),
    collection_id UUID REFERENCES collections(id),
    uploaded_by UUID REFERENCES users(id),
    status VARCHAR(50) DEFAULT 'pending',  -- pending/processing/ready/failed
    chunk_count INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE document_tags (
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    tag VARCHAR(100) NOT NULL,
    PRIMARY KEY (document_id, tag)
);

CREATE TABLE query_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    query TEXT NOT NULL,
    answer TEXT,
    sources JSONB,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 5.5 Structured Logging

```python
# backend/core/logging.py
import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )

# Usage in any module:
logger = structlog.get_logger()
logger.info("document_ingested", document_id=doc_id, chunks=42, latency_ms=1230)
# Output: {"event":"document_ingested","document_id":"abc-123","chunks":42,"latency_ms":1230,"level":"info","timestamp":"2026-03-15T..."}
```

### 5.6 API Rate Limiting

```python
# backend/api/middleware.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Per-endpoint limits
@router.post("/search")
@limiter.limit("30/minute")
async def search(request: Request, query: SearchQuery):
    ...

@router.post("/documents/upload")
@limiter.limit("10/minute")
async def upload(request: Request, file: UploadFile):
    ...
```

---

## 6. Deployment Strategy

### 6.1 Docker Containers

```dockerfile
# deployment/docker/Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

# System deps for PDF/OCR processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libmagic1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY backend/ backend/

EXPOSE 8000
CMD ["uvicorn", "backend.api.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
```

```dockerfile
# deployment/docker/Dockerfile.worker
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libmagic1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY backend/ backend/

CMD ["celery", "-A", "backend.workers.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
```

### 6.2 Docker Compose

```yaml
# deployment/docker-compose.yml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.api
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - postgres
      - redis
      - chromadb
    volumes:
      - upload_data:/app/uploads

  worker:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.worker
    env_file: .env
    depends_on:
      - postgres
      - redis
      - chromadb
    volumes:
      - upload_data:/app/uploads

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: doc_intel
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pg_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - api

volumes:
  pg_data:
  chroma_data:
  upload_data:
```

### 6.3 Environment Variable Management

```bash
# .env.example
# ── API ──
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=change-me-in-production
CORS_ORIGINS=["http://localhost:3000"]

# ── Database ──
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/doc_intel
POSTGRES_USER=doc_intel_user
POSTGRES_PASSWORD=strong-password-here

# ── Redis ──
REDIS_URL=redis://redis:6379/0

# ── Vector DB ──
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# ── LLM ──
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo

# ── Embedding ──
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ── Storage ──
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE_MB=50
```

### 6.4 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov=backend --cov-report=xml
      - run: ruff check backend/
      - run: mypy backend/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: deployment/docker/Dockerfile.api
          push: true
          tags: ghcr.io/${{ github.repository }}/api:latest
```

---

## 7. Clean Project Structure

```
doc-intelligence-platform/
│
├── README.md                          # Project overview, setup, and usage
├── pyproject.toml                     # Python project configuration
├── .env.example                       # Environment variable template
├── .gitignore
├── Makefile                           # Common commands (make run, make test, etc.)
│
├── backend/                           # Python backend (FastAPI)
│   ├── __init__.py
│   ├── core/                          # Core infrastructure
│   │   ├── __init__.py
│   │   ├── config.py                  # Pydantic Settings (typed, validated)
│   │   ├── database.py                # SQLAlchemy async engine + session
│   │   ├── security.py                # JWT, password hashing, auth utils
│   │   ├── logging.py                 # Structured logging setup
│   │   └── exceptions.py             # Custom exception classes
│   │
│   ├── models/                        # SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── document.py
│   │   ├── collection.py
│   │   └── query_log.py
│   │
│   ├── schemas/                       # Pydantic request/response schemas
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── document.py
│   │   ├── search.py
│   │   └── collection.py
│   │
│   ├── api/                           # FastAPI routers
│   │   ├── __init__.py
│   │   ├── main.py                    # App factory, middleware, startup
│   │   ├── deps.py                    # Dependency injection (get_db, get_user)
│   │   ├── middleware.py              # Rate limiting, request IDs, timing
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── auth.py                # POST /login, /register, /refresh
│   │       ├── documents.py           # CRUD + upload + tagging
│   │       ├── search.py              # POST /search, /ask
│   │       ├── collections.py         # Collection management
│   │       ├── admin.py               # User management, system config
│   │       └── health.py              # GET /health, /ready
│   │
│   ├── services/                      # Business logic layer
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── document_service.py
│   │   ├── search_service.py
│   │   └── collection_service.py
│   │
│   ├── ingestion/                     # Document processing pipeline
│   │   ├── __init__.py
│   │   ├── parsers/                   # Format-specific parsers
│   │   │   ├── __init__.py
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   ├── text_parser.py
│   │   │   ├── image_parser.py        # OCR via Tesseract
│   │   │   └── web_parser.py          # HTML/URL scraping
│   │   ├── chunker.py                 # Chunking strategies
│   │   ├── embedder.py                # Embedding generation
│   │   └── pipeline.py                # Orchestration: parse → chunk → embed → store
│   │
│   ├── rag/                           # RAG pipeline
│   │   ├── __init__.py
│   │   ├── retriever.py               # Vector search + hybrid search
│   │   ├── reranker.py                # Cross-encoder re-ranking
│   │   ├── generator.py               # LLM answer generation
│   │   ├── prompts.py                 # Prompt templates (Jinja2 or plain)
│   │   └── chains.py                  # LangChain chain definitions
│   │
│   ├── vector_store/                  # Vector DB abstraction
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base class
│   │   ├── chroma_store.py            # ChromaDB implementation
│   │   └── weaviate_store.py          # Weaviate implementation (optional)
│   │
│   ├── workers/                       # Celery background tasks
│   │   ├── __init__.py
│   │   ├── celery_app.py              # Celery configuration
│   │   └── tasks.py                   # Task definitions
│   │
│   └── agent/                         # AI agent tools
│       ├── __init__.py
│       ├── tools.py                   # Email, checklist, summary tools
│       └── router.py                  # Intent classification / routing
│
├── frontend/                          # React / Next.js frontend
│   ├── package.json
│   ├── Dockerfile
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/                  # API client
│   │   └── styles/
│   └── public/
│
├── deployment/                        # DevOps & deployment
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.worker
│   │   └── Dockerfile.frontend
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   ├── nginx/
│   │   └── nginx.conf
│   └── kubernetes/                    # (future) K8s manifests
│       ├── api-deployment.yaml
│       └── worker-deployment.yaml
│
├── migrations/                        # Alembic database migrations
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
│
├── tests/                             # Test suite
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   └── test_generator.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   └── test_search_api.py
│   └── evaluation/                    # RAG evaluation
│       ├── eval_retrieval.py
│       ├── eval_generation.py
│       └── test_datasets/
│
├── scripts/                           # Utility scripts
│   ├── seed_data.py
│   ├── benchmark.py
│   └── migrate.py
│
├── data/                              # Sample documents (for development)
│   ├── acceptable_use_policy.txt
│   ├── customer_data_privacy_policy.txt
│   ├── employee_onboarding_sop.txt
│   └── network_outage_sop.txt
│
└── docs/                              # Documentation
    ├── ARCHITECTURE_AND_ROADMAP.md    # This document
    ├── API_REFERENCE.md
    ├── DEPLOYMENT_GUIDE.md
    └── EVALUATION_RESULTS.md
```

---

## 8. Implementation Roadmap

### Phase 1 — Code Refactoring & Foundation (Week 1–2)

**Goal:** Clean up the codebase, establish proper project structure, and add foundational infrastructure.

| # | Task | Priority |
|---|------|----------|
| 1.1 | Restructure project into `backend/` folder layout | 🔴 |
| 1.2 | Replace `config.py` with Pydantic `BaseSettings` (typed, validated, `.env` support) | 🔴 |
| 1.3 | Extract shared `_get_llm()` into `backend/core/llm_factory.py` | 🟠 |
| 1.4 | Replace all `print()` with `structlog` structured logging | 🟠 |
| 1.5 | Add custom exception hierarchy (`DocumentNotFound`, `IngestionError`, etc.) | 🟠 |
| 1.6 | Add `pytest` with unit tests for chunker, retriever, and embedding | 🟡 |
| 1.7 | Add `ruff` linter and `mypy` type checking to CI | 🟡 |
| 1.8 | Create `Makefile` with common commands | 🟡 |

**Suggested Code Changes for Phase 1:**

```python
# BEFORE (config.py — mutable global):
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# AFTER (backend/core/config.py — Pydantic Settings):
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_collection: str = "telecom_policies"
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4
    database_url: str = "postgresql+asyncpg://..."
    redis_url: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"

settings = Settings()
```

```python
# BEFORE (duplicated in generator.py AND tools.py):
def _get_llm():
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(...)
    elif config.LLM_PROVIDER == "azure":
        ...

# AFTER (backend/core/llm_factory.py — single source of truth):
from backend.core.config import settings

def create_llm(temperature: float = 0.2):
    """Factory function for LLM instances. Single source of truth."""
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
    elif settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=settings.azure_deployment_name,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_api_version,
            temperature=temperature,
        )
    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
```

---

### Phase 2 — RAG Pipeline Improvements (Week 3–4)

**Goal:** Improve answer quality, add hybrid search, and establish evaluation baselines.

| # | Task | Priority |
|---|------|----------|
| 2.1 | Add hybrid search (BM25 + vector similarity fusion) | 🔴 |
| 2.2 | Implement cross-encoder re-ranking (e.g., `ms-marco-MiniLM`) | 🔴 |
| 2.3 | Add conversation memory (sliding window of prior turns) | 🟠 |
| 2.4 | Implement streaming LLM responses (SSE) | 🟠 |
| 2.5 | Build RAG evaluation pipeline (retrieval recall, answer faithfulness) | 🟠 |
| 2.6 | Create golden test dataset (50+ question-answer pairs with ground truth) | 🟡 |
| 2.7 | Add query expansion / HyDE (Hypothetical Document Embeddings) | 🟡 |
| 2.8 | Evaluate and benchmark different embedding models | 🟡 |

**Key Improvement — Hybrid Search + Re-ranking:**

```python
# backend/rag/retriever.py — Enhanced retrieval
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vector_store, documents, alpha=0.5):
        self.vector_store = vector_store
        self.bm25 = BM25Okapi([doc.split() for doc in documents])
        self.alpha = alpha  # weight: 0=BM25-only, 1=vector-only

    def retrieve(self, query: str, top_k: int = 10) -> list:
        # Vector search
        vector_results = self.vector_store.similarity_search_with_scores(query, k=top_k * 2)

        # BM25 keyword search
        bm25_scores = self.bm25.get_scores(query.split())

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(vector_results, bm25_scores, alpha=self.alpha)

        return fused[:top_k]
```

---

### Phase 3 — Scalable Ingestion Pipeline (Week 5–6)

**Goal:** Support more document types, add async processing, and handle scale.

| # | Task | Priority |
|---|------|----------|
| 3.1 | Add DOCX parser (`python-docx` or `unstructured`) | 🔴 |
| 3.2 | Add OCR parser (Tesseract) for images and scanned PDFs | 🔴 |
| 3.3 | Set up Celery + Redis for background ingestion | 🔴 |
| 3.4 | Implement document deduplication (SHA-256 hash check) | 🟠 |
| 3.5 | Add batch ingestion endpoint (multi-file upload) | 🟠 |
| 3.6 | Add ingestion progress tracking (status: pending → processing → ready) | 🟠 |
| 3.7 | Implement document metadata extraction (title, author, page count) | 🟡 |
| 3.8 | Add file size limits and format validation | 🟡 |

---

### Phase 4 — API, Auth & User Features (Week 7–9)

**Goal:** Build the FastAPI service, add authentication, and implement user management.

| # | Task | Priority |
|---|------|----------|
| 4.1 | Create FastAPI app with router structure | 🔴 |
| 4.2 | Set up PostgreSQL with SQLAlchemy async + Alembic migrations | 🔴 |
| 4.3 | Implement JWT authentication (login, register, refresh tokens) | 🔴 |
| 4.4 | Build document CRUD API (upload, list, get, delete) | 🔴 |
| 4.5 | Build search/ask API endpoints | 🔴 |
| 4.6 | Add collection management (create, list, assign documents) | 🟠 |
| 4.7 | Add document tagging and versioning | 🟠 |
| 4.8 | Implement RBAC (admin, editor, viewer roles) | 🟠 |
| 4.9 | Add API rate limiting (`slowapi`) | 🟡 |
| 4.10 | Add comprehensive API documentation (OpenAPI/Swagger) | 🟡 |

---

### Phase 5 — Deployment & Observability (Week 10–12)

**Goal:** Containerize, deploy, and add production monitoring.

| # | Task | Priority |
|---|------|----------|
| 5.1 | Create Dockerfiles (API, worker, frontend) | 🔴 |
| 5.2 | Create docker-compose for full stack | 🔴 |
| 5.3 | Set up CI/CD with GitHub Actions (test → build → deploy) | 🔴 |
| 5.4 | Add health check endpoints (`/health`, `/ready`) | 🟠 |
| 5.5 | Integrate Prometheus metrics (request latency, queue depth) | 🟠 |
| 5.6 | Add OpenTelemetry tracing (API → worker → LLM call spans) | 🟡 |
| 5.7 | Set up Grafana dashboards | 🟡 |
| 5.8 | Write deployment guide and README | 🟡 |
| 5.9 | Build React/Next.js frontend (optional, or keep Streamlit as demo UI) | 🟡 |

---

## 9. AI Engineer Portfolio Enhancements

### 9.1 System Architecture Diagrams
- **Current vs. Production architecture** diagrams (included above)
- **Data flow diagram** showing document lifecycle from upload → answer
- **Sequence diagram** for the RAG query flow
- Include these in README and docs/ — visual artifacts demonstrate systems thinking

### 9.2 Dataset Handling
- Create a **curated test dataset** of 50+ Q&A pairs with ground truth answers
- Include diverse question types: factual, procedural, comparative, multi-hop
- Show data preprocessing decisions with justification

### 9.3 RAG Evaluation Pipeline
This is the single most impressive addition for a portfolio:

```python
# tests/evaluation/eval_rag.py
"""
Evaluation metrics:
- Retrieval: Recall@K, MRR, nDCG
- Generation: BLEU, ROUGE-L, BERTScore, Faithfulness (LLM-as-judge)
- End-to-End: Answer correctness, citation accuracy
"""

class RAGEvaluator:
    def evaluate_retrieval(self, queries, ground_truth_docs):
        """Measure if the right chunks are retrieved."""
        recall_at_k = ...
        mrr = ...
        return {"recall@4": recall_at_k, "mrr": mrr}

    def evaluate_faithfulness(self, query, answer, context):
        """LLM-as-judge: does the answer stay faithful to the context?"""
        # Use GPT-4 to rate faithfulness 1-5
        ...

    def evaluate_citation_accuracy(self, answer, sources):
        """Check if cited sources actually support the claims."""
        ...
```

### 9.4 Benchmarking
- **Latency benchmarks**: end-to-end query time, embedding time, retrieval time
- **Embedding model comparison**: all-MiniLM-L6-v2 vs. bge-small vs. e5-small
- **Chunking strategy comparison**: different chunk sizes, overlap ratios
- Present results as tables and charts in `docs/EVALUATION_RESULTS.md`

### 9.5 Observability & Monitoring
- Show Grafana dashboards with real metrics
- Include LLM cost tracking (tokens used, cost per query)
- Log and visualize retrieval quality over time

### 9.6 Advanced Features That Impress

| Feature | Why It Impresses | Difficulty |
|---------|-----------------|------------|
| **RAG evaluation pipeline** | Shows ML rigor, not just "it works" | Medium |
| **Hybrid search (BM25 + vector)** | Demonstrates search expertise | Medium |
| **Cross-encoder re-ranking** | Shows awareness of retrieval best practices | Medium |
| **Streaming responses** | Production UX pattern | Easy |
| **Document versioning** | Real-world feature recruiters understand | Medium |
| **Celery task queue + status tracking** | Demonstrates distributed systems knowledge | Medium |
| **OpenTelemetry tracing** | Shows production ops mindset | Easy |
| **CI/CD with automated RAG eval** | Shows ML engineering maturity | Medium |
| **Architecture decision records (ADRs)** | Demonstrates senior-level thinking | Easy |
| **Cost analysis** (tokens/query) | Shows business awareness | Easy |

### 9.7 Portfolio README Structure

```markdown
# AI Document Intelligence Platform

## Overview
Production-grade RAG system for enterprise document Q&A.

## Architecture
[Architecture diagram]

## Key Technical Decisions
- Why ChromaDB over Pinecone (cost, local-first, portfolio-friendly)
- Why hybrid search (BM25 + vector) improves recall by X%
- Why cross-encoder re-ranking improves precision by X%

## Evaluation Results
| Metric | Baseline | With Re-ranking | Improvement |
|--------|----------|----------------|-------------|
| Recall@4 | 0.72 | 0.89 | +23.6% |
| MRR | 0.65 | 0.81 | +24.6% |
| Faithfulness | 4.1/5 | 4.3/5 | +4.9% |

## Tech Stack
[Table of technologies]

## How to Run
[Docker compose one-liner]

## API Documentation
[Link to /api/docs]
```

---

## 10. Prioritized Feature List & Next Steps

### Immediate (Start Now)

1. **Refactor config to Pydantic Settings** — eliminates mutable global state
2. **Extract shared LLM factory** — removes code duplication
3. **Replace print() with structlog** — production-grade logging
4. **Add FastAPI skeleton** with health check endpoint
5. **Write first unit tests** for chunker and retriever

### Short-Term (Next 2–4 Weeks)

6. **Implement hybrid search** (BM25 + vector fusion)
7. **Add cross-encoder re-ranking**
8. **Set up Celery + Redis** for background ingestion
9. **Add PostgreSQL** for metadata storage
10. **Build document upload API** with format validation
11. **Add JWT authentication**
12. **Create RAG evaluation pipeline** with golden dataset

### Medium-Term (1–2 Months)

13. **Add DOCX and image/OCR parsers**
14. **Implement document collections and tagging**
15. **Add document versioning**
16. **Build React frontend** (or keep Streamlit as demo)
17. **Containerize with Docker + docker-compose**
18. **Set up CI/CD pipeline**

### Long-Term (Portfolio Polish)

19. **Add OpenTelemetry tracing**
20. **Build Grafana dashboards**
21. **Write benchmarking suite** (embedding models, chunk strategies)
22. **Add LLM cost tracking**
23. **Write Architecture Decision Records (ADRs)**
24. **Create demo video / live deployment**

---

## Summary

The current system is a **well-structured prototype** with clean module separation and a working RAG pipeline. The core data flow (load → chunk → embed → retrieve → generate) is sound. The main gaps are:

1. **No API layer** — everything runs through Streamlit
2. **No authentication or multi-tenancy**
3. **No persistent metadata store** (PostgreSQL)
4. **No async processing** — blocking I/O on every operation
5. **No logging, monitoring, or error handling infrastructure**
6. **No tests or evaluation pipeline**

The roadmap above transforms this into a **production-quality platform** in 5 phases while building a portfolio that demonstrates:
- **Systems architecture** (multi-service, async, storage layers)
- **ML engineering rigor** (evaluation pipelines, benchmarking)
- **Production operations** (Docker, CI/CD, observability)
- **Software engineering quality** (testing, typed config, structured logging)

The most impactful single addition for both production readiness and portfolio value is the **RAG evaluation pipeline** — it proves the system works quantitatively, not just anecdotally.
