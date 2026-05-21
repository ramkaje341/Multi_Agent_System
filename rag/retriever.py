"""
rag/retriever.py — Retrieval logic: query → context string for agents.
"""
import logging
from typing import List, Dict, Any
from rag.vectorstore import similarity_search
from config.settings import TOP_K_RESULTS

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    n_results: int = TOP_K_RESULTS,
    source_filter: str | None = None,
) -> str:
    """
    Retrieve relevant chunks and format as a context block
    ready to be injected into an agent's prompt.

    Args:
        query: The clinical question or symptom description.
        n_results: Number of chunks to retrieve.
        source_filter: Optional metadata filter (e.g. 'drug_monograph').

    Returns:
        Formatted context string.
    """
    filter_meta = {"source_type": source_filter} if source_filter else None
    docs = similarity_search(query, n_results=n_results, filter_meta=filter_meta)

    if not docs:
        return "No relevant context found in the knowledge base."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        source = meta.get("source", "Unknown")
        section = meta.get("section", "")
        relevance = round(1 - doc["distance"], 3)

        context_parts.append(
            f"[{i}] Source: {source}"
            + (f" | Section: {section}" if section else "")
            + f" | Relevance: {relevance}\n{doc['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def retrieve_raw(
    query: str,
    n_results: int = TOP_K_RESULTS,
) -> List[Dict[str, Any]]:
    """Return raw list of retrieved doc dicts (for agents that need metadata)."""
    return similarity_search(query, n_results=n_results)