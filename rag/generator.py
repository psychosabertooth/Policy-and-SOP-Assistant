"""RAG answer generation: retrieve context, check relevance, generate via LLM."""

import os
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from rag.prompts import RAG_PROMPT_TEMPLATE, NO_CONTEXT_RESPONSE
from rag.retriever import retrieve_relevant_chunks, format_context, has_relevant_results


def _get_llm():
    """Return the configured LLM instance based on LLM_PROVIDER."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.2,
            openai_api_key=config.OPENAI_API_KEY,
        )
    elif config.LLM_PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=config.AZURE_DEPLOYMENT_NAME,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_API_VERSION,
            temperature=0.2,
        )
    elif config.LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama

        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.2,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{config.LLM_PROVIDER}'. "
            "Set LLM_PROVIDER to 'openai', 'azure', or 'ollama'."
        )


def generate_answer(query: str) -> dict:
    """Full RAG pipeline: retrieve, check relevance, generate. Returns dict with answer, sources, scores, has_info."""
    results: List[Tuple[Document, float]] = retrieve_relevant_chunks(query)

    if not results:
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "sources": [],
            "scores": [],
            "has_info": False,
        }

    confident = has_relevant_results(results)
    context = format_context(results)

    llm = _get_llm()
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    sources = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "score": round(score, 3),
            "preview": doc.page_content[:200].replace("\n", " "),
        }
        for doc, score in results
    ]

    return {
        "answer": answer.strip(),
        "sources": sources,
        "scores": [s["score"] for s in sources],
        "has_info": confident,
    }

