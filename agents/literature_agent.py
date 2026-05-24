"""
agents/literature_agent.py — MCP Literature Agent (ReAct tool-calling loop).

The LLM autonomously decides which tools to call:
  1. pubmed_search          → PubMed peer-reviewed evidence (always first)
  2. rag_search             → local knowledge base for guidelines
  3. tavily_medical_web_search → internet fallback for latest publications/guidelines
  4. tavily_web_search      → open web fallback if medical search insufficient
"""
import logging
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from mcp.tool_definitions import LITERATURE_TOOLS
from mcp.tool_executor import run_tool_call_loop, format_tool_trace_for_display
from memory.memory_store import log_agent_trace
from config.settings import GROQ_API_KEY, PRIMARY_MODEL, MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical evidence synthesis specialist with access to literature tools.

Your goal is to find, appraise, and synthesise the best available clinical evidence.

TOOL USAGE STRATEGY:
1. pubmed_search → ALWAYS call first. Optimise the query:
   - Add 'systematic review' or 'meta-analysis' for high-quality evidence
   - Add 'RCT' for interventional evidence
   - Add 'guideline' for clinical guidelines
   - If first search returns 0 articles, retry with a broader/different query
2. rag_search → Call to find relevant local clinical guidelines and protocols.
3. tavily_medical_web_search → Call if:
   - pubmed_search returns 0 or very few articles
   - Query asks for 'latest', 'recent', 'current', '2024', '2025' evidence
   - You need information about very new treatments not yet indexed on PubMed
4. tavily_web_search → Use as last resort if tavily_medical_web_search is also insufficient.

RULES:
- Always call pubmed_search first, no exceptions.
- Try at least 2 different PubMed search queries before giving up.
- Aim for at least 3 pieces of evidence before synthesising.
- Cite PubMed IDs (PMIDs) and URLs for every claim.
- Clearly state the level of evidence (RCT / meta-analysis / cohort / expert opinion).

OUTPUT FORMAT:
## Evidence Summary

### What the Evidence Shows
[Concise overall summary of what the evidence says]

### Key Findings
- [Finding 1] — [Author et al., Year, Journal] PMID: [xxx] | Evidence level: [RCT/MA/etc]
- [Finding 2] — [Source]
- [Finding 3] — [Source]

### Evidence Quality
[Overall assessment: strong/moderate/weak, based on study types found]

### Clinical Implications
[What these findings mean for clinical practice]

### Knowledge Gaps & Limitations
[What remains unanswered, conflicting evidence, outdated studies]

### Full References
- [PMID/URL] [Title] — [Authors] ([Year]) [Journal]

---
⚠️ Evidence synthesis by AI. Clinical application requires professional judgement.
"""


def run_literature_agent(
    query: str,
    session_id: str,
    specific_pubmed_query: str | None = None,
) -> Dict[str, Any]:
    """
    Run the MCP-enabled literature agent via ReAct tool-calling loop.

    Args:
        query:                Clinical question or topic.
        session_id:           Session ID for memory logging.
        specific_pubmed_query: Optional override PubMed query string.

    Returns:
        Dict with evidence text, citations, tool trace, and tools called.
    """
    logger.info(f"[LiteratureAgent] Starting MCP ReAct loop — session {session_id}")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=PRIMARY_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    llm_with_tools = llm.bind_tools(LITERATURE_TOOLS)

    if specific_pubmed_query:
        user_content = (
            f"Suggested PubMed query: {specific_pubmed_query}\n\n"
            f"Clinical question: {query}"
        )
    else:
        user_content = query

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    final_text, tool_trace, _ = run_tool_call_loop(
        llm_with_tools=llm_with_tools,
        messages=messages,
    )

    tools_called = [t["tool"] for t in tool_trace]

    # Extract citations from pubmed_search tool results in the trace
    citations = []
    for tc in tool_trace:
        if tc["tool"] == "pubmed_search":
            for line in tc.get("result", "").split("\n"):
                if line.startswith("PMID"):
                    parts = line.split("|")
                    pmid = parts[0].replace("PMID", "").strip()
                    citations.append({
                        "pmid":    pmid,
                        "title":   parts[1].strip() if len(parts) > 1 else "",
                        "year":    parts[2].strip() if len(parts) > 2 else "",
                        "journal": parts[3].strip() if len(parts) > 3 else "",
                        "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "authors": [],
                    })

    log_agent_trace(
        session_id=session_id,
        agent_name="LiteratureAgent",
        tool_calls=tools_called,
        result_summary=f"{len(citations)} citations. {final_text[:200]}",
    )

    logger.info(
        f"[LiteratureAgent] Complete — {len(tool_trace)} tool calls, "
        f"{len(citations)} citations:\n{format_tool_trace_for_display(tool_trace)}"
    )

    return {
        "agent":        "LiteratureAgent",
        "result":       final_text,
        "citations":    citations,
        "tool_trace":   tool_trace,
        "tools_called": tools_called,
    }