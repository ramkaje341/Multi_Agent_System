"""
tools/web_search_tool.py — Tavily-powered internet search for the medical assistant.

"""

import os
import logging
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)



TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# Trusted medical domains — Tavily will prioritise results from these
TRUSTED_MEDICAL_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "nih.gov",
    "nejm.org",
    "bmj.com",
    "thelancet.com",
    "jamanetwork.com",
    "mayoclinic.org",
    "nice.org.uk",
    "who.int",
    "cdc.gov",
    "cochranelibrary.com",
    "medscape.com",
    "emedicine.medscape.com",
    "uptodate.com",
    "aafp.org",
    "acpjournals.org",
    "nature.com",
]

MAX_RESULTS       = 5     # results per search
SEARCH_DEPTH      = "advanced"   # "basic" (faster) or "advanced" (more thorough)
MAX_TOKENS_RESULT = 300   # max chars per result snippet kept for the prompt


# ─── Result schema ────────────────────────────────────────────────────────────

def _make_result(
    title:   str,
    url:     str,
    content: str,
    score:   float = 0.0,
) -> Dict[str, Any]:
    trusted = any(domain in url for domain in TRUSTED_MEDICAL_DOMAINS)
    return {
        "title":   title,
        "url":     url,
        "content": content[:MAX_TOKENS_RESULT],
        "score":   round(score, 3),
        "trusted": trusted,
    }


# ─── Core Tavily search ───────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _call_tavily(
    query: str,
    max_results: int = MAX_RESULTS,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    search_depth: str = SEARCH_DEPTH,
    include_answer: bool = True,
) -> Dict[str, Any]:
    """
    Low-level Tavily API call.
    Returns raw Tavily response dict.
    Raises RuntimeError if API key is not configured.
    """
    if not TAVILY_API_KEY or TAVILY_API_KEY == "your_tavily_api_key_here":
        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            
        )

    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "tavily-python is not installed. Run: pip install tavily-python"
        )

    client = TavilyClient(api_key=TAVILY_API_KEY)

    kwargs: Dict[str, Any] = {
        "query":        query,
        "max_results":  max_results,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "include_raw_content": False,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains
    if exclude_domains:
        kwargs["exclude_domains"] = exclude_domains

    return client.search(**kwargs)


# ─── Public API ───────────────────────────────────────────────────────────────

def tavily_search(
    query: str,
    max_results: int = MAX_RESULTS,
    search_depth: str = SEARCH_DEPTH,
) -> Dict[str, Any]:
    """
    General Tavily web search. Searches the open web.

    Returns:
        {
          "query":       str,
          "answer":      str,   # Tavily's AI-generated direct answer
          "results":     List[{title, url, content, score, trusted}],
          "source":      "tavily",
          "error":       str | None,
        }
    """
    try:
        raw = _call_tavily(query, max_results=max_results, search_depth=search_depth)

        results = [
            _make_result(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in raw.get("results", [])
        ]

        # Sort: trusted medical sources first, then by score
        results.sort(key=lambda r: (not r["trusted"], -r["score"]))

        return {
            "query":   query,
            "answer":  raw.get("answer", ""),
            "results": results,
            "source":  "tavily",
            "error":   None,
        }

    except RuntimeError as e:
        logger.error(f"[TavilySearch] Config error: {e}")
        return {"query": query, "answer": "", "results": [], "source": "tavily", "error": str(e)}
    except Exception as e:
        logger.error(f"[TavilySearch] Search failed: {e}")
        return {"query": query, "answer": "", "results": [], "source": "tavily", "error": str(e)}


def tavily_medical_search(
    query: str,
    max_results: int = MAX_RESULTS,
) -> Dict[str, Any]:
    """
    Medical-domain-restricted Tavily search.
    Results are limited to trusted medical authority websites only.
    Use this when you want high-quality, peer-reviewed sources.

    Args:
        query:       Clinical question or topic.
        max_results: Number of results (default 5).

    Returns: same schema as tavily_search()
    """
    try:
        raw = _call_tavily(
            query,
            max_results=max_results,
            include_domains=TRUSTED_MEDICAL_DOMAINS,
            search_depth="advanced",
        )

        results = [
            _make_result(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in raw.get("results", [])
        ]

        # All results here are from trusted domains — sort by score
        results.sort(key=lambda r: -r["score"])

        return {
            "query":   query,
            "answer":  raw.get("answer", ""),
            "results": results,
            "source":  "tavily_medical",
            "error":   None,
        }

    except RuntimeError as e:
        logger.error(f"[TavilyMedicalSearch] Config error: {e}")
        return {"query": query, "answer": "", "results": [], "source": "tavily_medical", "error": str(e)}
    except Exception as e:
        logger.error(f"[TavilyMedicalSearch] Search failed for '{query}': {e}")
        return {"query": query, "answer": "", "results": [], "source": "tavily_medical", "error": str(e)}


def format_search_results_for_prompt(search_output: Dict[str, Any]) -> str:
    """
    Format Tavily search results as a clean context block
    ready to be injected into an agent's LLM prompt.

    Args:
        search_output: Output from tavily_search() or tavily_medical_search()

    Returns:
        Formatted string with answer + numbered source snippets.
    """
    if search_output.get("error"):
        return f"[Web search unavailable: {search_output['error']}]"

    parts = []

    # Tavily's direct answer (usually very good)
    if search_output.get("answer"):
        parts.append(f"WEB SEARCH DIRECT ANSWER:\n{search_output['answer']}")

    # Individual source results
    results = search_output.get("results", [])
    if results:
        source_lines = []
        for i, r in enumerate(results, 1):
            trusted_tag = " ✓" if r["trusted"] else ""
            source_lines.append(
                f"[{i}]{trusted_tag} {r['title']}\n"
                f"    URL: {r['url']}\n"
                f"    {r['content']}"
            )
        parts.append("WEB SEARCH SOURCES:\n" + "\n\n".join(source_lines))

    if not parts:
        return "[Web search returned no results]"

    return "\n\n" + ("─" * 50 + "\n").join(parts)


def is_tavily_configured() -> bool:
    """Check if Tavily API key is present and non-empty."""
    return bool(TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key_here")