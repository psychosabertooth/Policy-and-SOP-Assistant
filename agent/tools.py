"""Agent tools: email drafting, checklist creation, and smart routing."""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.prompts import DRAFT_EMAIL_PROMPT, CHECKLIST_PROMPT, AGENT_ROUTER_PROMPT


def _get_llm():
    """Shared LLM loader (same logic as generator.py)."""
    import config

    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=config.OPENAI_API_KEY,
        )
    elif config.LLM_PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=config.AZURE_DEPLOYMENT_NAME,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_API_VERSION,
            temperature=0.3,
        )
    else:
        from langchain_community.llms import Ollama

        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.3,
        )


def draft_email(policy_info: str) -> str:
    """Convert policy information into a professional internal email."""
    llm = _get_llm()
    prompt = PromptTemplate(
        template=DRAFT_EMAIL_PROMPT,
        input_variables=["policy_info"],
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"policy_info": policy_info}).strip()


def create_checklist(procedure_info: str) -> str:
    """Convert procedure/SOP information into a numbered checklist."""
    llm = _get_llm()
    prompt = PromptTemplate(
        template=CHECKLIST_PROMPT,
        input_variables=["procedure_info"],
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"procedure_info": procedure_info}).strip()


def route_to_tool(question: str) -> str:
    """LLM-based router that classifies user intent into 'answer', 'email', or 'checklist'."""
    llm = _get_llm()
    prompt = PromptTemplate(
        template=AGENT_ROUTER_PROMPT,
        input_variables=["question"],
    )
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({"question": question}).strip().lower()

    if "email" in decision:
        return "email"
    elif "checklist" in decision:
        return "checklist"
    else:
        return "answer"

