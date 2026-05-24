"""
agents/drug_agent.py — MCP Drug Information Agent (ReAct tool-calling loop).

The LLM autonomously decides which tools to call:
  1. drug_label_lookup       → FDA drug label (indications, warnings, dosing)
  2. drug_interaction_check  → interactions between multiple drugs
  3. adverse_events_lookup   → FAERS real-world adverse event data
  4. rxnorm_lookup           → standardised RxNorm drug codes
  5. rag_search              → local knowledge base for guidelines/context
  6. tavily_medical_web_search → internet fallback for latest drug news/approvals
"""
import logging
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from mcp.tool_definitions import DRUG_TOOLS
from mcp.tool_executor import run_tool_call_loop, format_tool_trace_for_display
from memory.memory_store import log_agent_trace
from config.settings import GROQ_API_KEY, PRIMARY_MODEL, MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical pharmacology AI with access to drug information tools.

Your goal is to provide comprehensive, accurate drug information for safe clinical use.

TOOL USAGE STRATEGY (follow this order):
1. drug_label_lookup       → ALWAYS call first for every drug mentioned in the query.
2. drug_interaction_check  → ALWAYS call if 2 or more drugs are mentioned.
3. adverse_events_lookup   → Call for safety-critical questions or when asked about side effects.
4. rxnorm_lookup           → Call to get standardised RxNorm drug codes.
5. rag_search              → Search local knowledge base for clinical guidelines and context.
6. tavily_medical_web_search → Call if:
   - FDA data is insufficient or not found
   - Query mentions "latest", "new", "recently approved", "2024", "2025"
   - You need current prescribing information not in FDA database

RULES:
- Call drug_label_lookup FIRST for each drug — never skip this.
- If 2+ drugs are mentioned ANYWHERE in the query, ALWAYS call drug_interaction_check.
- Call adverse_events_lookup to complement the FDA label with real-world FAERS data.
- Use tavily_medical_web_search as fallback if FDA data is insufficient.
- Never guess drug dosing — rely only on what the tools return.

OUTPUT FORMAT:
## Drug Information Summary

### Drug Overview
**[Drug Name(s)]**
Therapeutic class: ...
Mechanism of action: ...
RxNorm code: ...

### Indications & Dosing
- Standard adult dose: ...
- Renal adjustment: ...
- Hepatic adjustment: ...
- Special populations (elderly, paediatric): ...

### ⚠️ Warnings & Contraindications
- Black box warnings: ...
- Contraindications: ...
- Serious precautions: ...

### Adverse Effects
**Common (>10%):** ...
**Serious:** ...
**Real-world FAERS data:** ...

### Drug Interactions
[From drug_interaction_check results]
- [Drug A] + [Drug B]: ...

### Clinical Pearls
- Monitoring parameters: ...
- Patient counselling: ...
- Key clinical tips: ...

---
⚠️ AI-generated pharmacology information. Prescribing decisions require physician judgement.
"""


def run_drug_agent(
    query: str,
    session_id: str,
    drug_names: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Run the MCP-enabled drug information agent via ReAct tool-calling loop.

    Args:
        query:       Clinical drug question.
        session_id:  Session ID for memory logging.
        drug_names:  Optional explicit drug name list (supplements auto-detection by LLM).

    Returns:
        Dict with drug info text, tool trace, and tools called.
    """
    logger.info(f"[DrugAgent] Starting MCP ReAct loop — session {session_id}")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=PRIMARY_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    llm_with_tools = llm.bind_tools(DRUG_TOOLS)

    user_content = query
    if drug_names:
        user_content = f"Focus on these drugs: {', '.join(drug_names)}\n\nQuery: {query}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    final_text, tool_trace, _ = run_tool_call_loop(
        llm_with_tools=llm_with_tools,
        messages=messages,
    )

    tools_called = [t["tool"] for t in tool_trace]

    log_agent_trace(
        session_id=session_id,
        agent_name="DrugAgent",
        tool_calls=tools_called,
        result_summary=final_text[:300],
    )

    logger.info(
        f"[DrugAgent] Complete — {len(tool_trace)} tool calls:\n"
        f"{format_tool_trace_for_display(tool_trace)}"
    )

    return {
        "agent":        "DrugAgent",
        "result":       final_text,
        "tool_trace":   tool_trace,
        "tools_called": tools_called,
    }