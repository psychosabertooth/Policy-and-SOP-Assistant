"""Central configuration for the Policy & SOP Assistant."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "vector_store" / "chroma_db"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "telecom_policies"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")

OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"

ENABLE_WEB_SCRAPING = True
SCRAPE_DELAY = 2

TOP_K = 4

APP_TITLE = "📋 Telecom Policy & SOP Assistant"
APP_SUBTITLE = "Ask questions about company policies, SOPs, and procedures"
