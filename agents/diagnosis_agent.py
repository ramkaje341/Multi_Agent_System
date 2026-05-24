"""
agents/diagnosis_agent.py — MCP Diagnosis Agent .

The LLM autonomously decides which tools to call and in what order:
  1. rag_search            → search local medical knowledge base FIRST
  2. icd10_lookup          → get ICD-10 codes for each condition considered
  3. snomed_lookup         → standardised terminology
  4. pubmed_search         → supporting literature evidence
  5. tavily_medical_web_search → internet fallback if RAG/PubMed insufficient,
                                 or query asks for latest/current guidelines
"""
import logging
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from mcp.tool_definitions import DIAGNOSIS_TOOLS
from mcp.tool_executor import run_tool_call_loop, format_tool_trace_for_display
from memory.memory_store import log_agent_trace
from config.settings import GROQ_API_KEY, PRIMARY_MODEL, MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert clinical diagnostician AI with access to medical tools.

Your goal is to produce a thorough, evidence-based differential diagnosis.

TOOL USAGE STRATEGY (follow this order):
1. rag_search            → ALWAYS call first. Search the local medical knowledge base.
2. icd10_lookup          → Call for each diagnosis you are considering.
3. pubmed_search         → Find supporting evidence. Add 'systematic review' or 'RCT' for quality.
4. snomed_lookup         → If you need standardised clinical terminology.
5. tavily_medical_web_search → Call if:
   - rag_search returns "No relevant information" or < 200 characters
   - pubmed_search returns 0 articles
   - The query mentions "latest", "recent", "current guidelines", "2024", "2025"
   - You need information not found in the local knowledge base

RULES:
- Call rag_search FIRST for every query, no exceptions.
- If rag_search is insufficient, immediately call tavily_medical_web_search.
- Call icd10_lookup for EVERY diagnosis you mention.
- Only stop calling tools when you have enough information for a complete DDx.
- Never fabricate clinical information — only use what the tools return.

OUTPUT FORMAT (after gathering all information via tools):
## Differential Diagnosis

### Primary Diagnosis (Most Likely)
**[Condition Name]** — ICD-10: [code]
- Supporting features from presentation: ...
- Confirmatory investigations: ...
- Urgency: Emergency / Urgent / Routine

### Alternative Differentials (ranked by likelihood)
1. **[Condition]** — ICD-10: [code]
   - For: ... | Against: ... | Tests: ...
2. **[Condition]** — ICD-10: [code]
   - For: ... | Against: ... | Tests: ...

### Red Flags to Watch
- ...

### Recommended Workup
- Bloods: ...
- Imaging: ...
- Other: ...

### Evidence Base
[Brief summary of literature/guidelines found]

---
⚠️ AI-assisted clinical decision support. All findings must be verified by a qualified clinician.
"""


def run_diagnosis_agent(
    patient_query: str,
    session_id: str,
    patient_context: str = "",
) -> Dict[str, Any]:
    """
    Run the MCP-enabled diagnosis agent via ReAct tool-calling loop.

    The LLM autonomously:
      - Decides which tools to call
      - Calls them in the right order
      - Reasons over the results
      - Iterates until it has enough information
      - Produces a structured DDx

    Args:
        patient_query:   Symptom description / clinical question.
        session_id:      Session ID for memory logging.
        patient_context: Optional patient demographics/history.

    Returns:
        Dict with diagnosis text, tool trace, and tools called.
    """
    logger.info(f"[DiagnosisAgent] Starting MCP ReAct loop — session {session_id}")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=PRIMARY_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    # Bind tools — this is the MCP tool-calling registration
    llm_with_tools = llm.bind_tools(DIAGNOSIS_TOOLS)

    # Build initial messages
    user_content = patient_query
    if patient_context.strip():
        user_content = f"PATIENT CONTEXT: {patient_context}\n\nCLINICAL QUERY: {patient_query}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    # Run the ReAct tool-calling loop
    final_text, tool_trace, _ = run_tool_call_loop(
        llm_with_tools=llm_with_tools,
        messages=messages,
    )

    tools_called = [t["tool"] for t in tool_trace]

    log_agent_trace(
        session_id=session_id,
        agent_name="DiagnosisAgent",
        tool_calls=tools_called,
        result_summary=final_text[:300],
    )

    logger.info(
        f"[DiagnosisAgent] Complete — {len(tool_trace)} tool calls:\n"
        f"{format_tool_trace_for_display(tool_trace)}"
    )

    return {
        "agent":        "DiagnosisAgent",
        "diagnosis":    final_text,
        "result":       final_text,
        "tool_trace":   tool_trace,
        "tools_called": tools_called,
    }