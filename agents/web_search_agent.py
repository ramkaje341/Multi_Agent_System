"""
agents/web_search_agent.py — MCP Web Search Agent (ReAct tool-calling loop).

Dedicated Tavily internet search agent.

Triggered by the orchestrator when:
  TIER 1 — Explicit trigger:
    • Query contains recency keywords (latest, recent, 2024, 2025…)
    • Intent classifier flagged 'web'
  TIER 2 — Confidence-based fallback:
    • Pipeline confidence score < FALLBACK_CONFIDENCE_THR after all specialist agents ran

The LLM autonomously calls:
  tavily_medical_web_search → trusted medical domains first (preferred)
  tavily_web_search         → open web fallback if medical search insufficient
"""
import logging
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from mcp.tool_definitions import WEB_TOOLS
from mcp.tool_executor import run_tool_call_loop, format_tool_trace_for_display
from tools.web_search_tool import is_tavily_configured
from memory.memory_store import log_agent_trace
from config.settings import GROQ_API_KEY, PRIMARY_MODEL, MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical information specialist with internet search tools.

Your goal is to find current, authoritative medical information from the web.

TOOL USAGE STRATEGY:
1. ALWAYS call tavily_medical_web_search first.
   This restricts results to trusted medical authority sites:
   PubMed, NEJM, BMJ, Lancet, JAMA, WHO, CDC, NICE, Cochrane, Mayo Clinic, etc.

2. If tavily_medical_web_search returns "No medical web results" or < 200 characters,
   immediately call tavily_web_search as an open-web fallback.

3. You may refine your search query and call tools again if the first result
   is not specific enough to answer the clinical question.

4. Stop when you have sufficient information from the web.

OUTPUT FORMAT:
### Web-Retrieved Answer
[Main clinical answer based on web sources]

### Key Points
- [Finding — source name + URL]
- [Finding — source name + URL]

### Sources Consulted
- [Title] — [URL] (trusted medical source ✓ / general web)

### Recency & Confidence
[When the information is from, any conflicts between sources, confidence level]

---
⚠️ Internet-sourced information. Verify with current clinical guidelines before clinical application.
"""


def run_web_search_agent(
    query: str,
    session_id: str,
    force_medical_domains: bool = False,
    context_from_other_agents: str = "",
) -> Dict[str, Any]:
    """
    Run the MCP web search agent via ReAct tool-calling loop.

    Args:
        query:                      Clinical question to search for.
        session_id:                 Session ID for memory logging.
        force_medical_domains:      Hint to LLM to prefer medical-domain search.
        context_from_other_agents:  Text summary from prior agents (for synthesis context).

    Returns:
        Dict with web search result text, search type, sources, tool trace.
    """
    logger.info(f"[WebSearchAgent] Starting MCP ReAct loop — session {session_id}")

    if not is_tavily_configured():
        msg = (
            "Tavily API key not configured. "
            "Get your free key at https://app.tavily.com "
            "and add TAVILY_API_KEY to your .env file."
        )
        logger.warning(f"[WebSearchAgent] {msg}")
        return {
            "agent":          "WebSearchAgent",
            "result":         f"⚠️ {msg}",
            "search_type":    "none",
            "sources":        [],
            "tavily_enabled": False,
            "error":          msg,
            "tools_called":   [],
            "tool_trace":     [],
        }

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=PRIMARY_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    llm_with_tools = llm.bind_tools(WEB_TOOLS)

    # Build user content with prior agent context if available
    user_content = f"CLINICAL QUERY: {query}"
    if context_from_other_agents.strip():
        user_content += (
            f"\n\nCONTEXT FROM PRIOR AGENTS "
            f"(use web search to ADD TO or CORRECT this):\n"
            f"{context_from_other_agents[:600]}"
        )
    if force_medical_domains:
        user_content += "\n\nPrefer tavily_medical_web_search (trusted medical domains only)."

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    final_text, tool_trace, _ = run_tool_call_loop(
        llm_with_tools=llm_with_tools,
        messages=messages,
    )

    tools_called = [t["tool"] for t in tool_trace]

    # Determine which search type was actually used
    search_type = (
        "medical"  if "tavily_medical_web_search" in tools_called else
        "general"  if "tavily_web_search"         in tools_called else
        "none"
    )

    log_agent_trace(
        session_id=session_id,
        agent_name="WebSearchAgent",
        tool_calls=tools_called,
        result_summary=f"Type={search_type}. {final_text[:200]}",
    )

    logger.info(
        f"[WebSearchAgent] Complete — type={search_type}, "
        f"{len(tool_trace)} tool calls:\n"
        f"{format_tool_trace_for_display(tool_trace)}"
    )

    return {
        "agent":          "WebSearchAgent",
        "result":         final_text,
        "search_type":    search_type,
        "sources":        [],
        "tavily_enabled": True,
        "error":          None,
        "tools_called":   tools_called,
        "tool_trace":     tool_trace,
    }