"""
agents/summariser_agent.py — Final synthesis agent.

Does NOT use tool-calling — it receives all specialist agent outputs
(each of which ran their own MCP ReAct loop) and synthesises them into:
  1. A structured clinician-facing summary
  2. A patient-friendly plain-language summary
"""
import logging
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from memory.memory_store import log_agent_trace, save_clinical_note
from config.settings import GROQ_API_KEY, PRIMARY_MODEL, MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

CLINICIAN_PROMPT = """You are a senior physician synthesising inputs from multiple
specialist clinical AI agents. Each agent used MCP tool-calling (ReAct loop)
to autonomously gather information from RAG, PubMed, OpenFDA, and the internet.

Synthesise all agent outputs into ONE coherent, actionable clinical response.

OUTPUT FORMAT:
## Clinical Summary

### Primary Assessment
[Overall clinical picture based on all evidence gathered]

### Differential Diagnosis
[Ranked DDx with ICD-10 codes — from diagnosis agent]

### Drug Information
[Key pharmacology points — from drug agent]

### Evidence Base
[Key literature findings with PMIDs — from literature agent]

### Web-Sourced Updates
[Internet search findings if Tavily was used — from web search agent]

### Image Analysis
[Medical image findings if an image was uploaded — from image agent]

### Recommended Next Steps
1. Immediate actions
2. Investigations to order
3. Treatment considerations
4. Follow-up plan

### ⚠️ Safety Flags
[Any urgent concerns, red flags, or critical findings — prominently listed]

### MCP Tool Audit
[Brief list: which tools each agent called during its ReAct loop]

---
*AI-generated clinical decision support. All recommendations require physician verification.*
"""

PATIENT_PROMPT = """You are a compassionate medical communicator.
Take the clinical summary and rewrite it in simple, plain language for a patient.

Rules:
- No medical jargon (explain any necessary medical terms simply)
- Warm, clear, reassuring tone
- Focus on: what's happening, what happens next, what to watch for
- Keep to 150–200 words maximum
- End with: "Always talk to your doctor before making any health decisions."
"""


def _build_combined_input(original_query: str, agent_outputs: Dict[str, Any]) -> str:
    """
    Combine all agent outputs into a single text block for the summariser LLM.
    Includes MCP tool audit for each agent.
    """
    parts = [f"ORIGINAL QUERY:\n{original_query}"]

    for agent_name, output in agent_outputs.items():
        if not isinstance(output, dict):
            parts.append(f"{agent_name.upper()}:\n{output}")
            continue

        content = (
            output.get("result") or
            output.get("diagnosis") or
            output.get("clinician_summary") or
            ""
        )
        tools_called = output.get("tools_called", [])
        tool_trace   = output.get("tool_trace", [])

        section = f"{agent_name.upper()}:\n{content}"

        if tools_called:
            section += f"\n\n[MCP Tools called by {agent_name}: {', '.join(tools_called)}]"

        if tool_trace:
            steps = [
                f"  Step {i+1}: {t['tool']}({list(t['args'].values())[:1]}) "
                f"→ {t['result_len']} chars"
                for i, t in enumerate(tool_trace[:5])
            ]
            section += "\n[ReAct steps:\n" + "\n".join(steps) + "]"

        parts.append(section)

    return ("\n\n" + "=" * 60 + "\n\n").join(parts)


def run_summariser_agent(
    original_query: str,
    agent_outputs: Dict[str, Any],
    session_id: str,
    generate_patient_summary: bool = True,
) -> Dict[str, Any]:
    """
    Synthesise all specialist agent outputs into a final clinical response.

    Args:
        original_query:           The original user question.
        agent_outputs:            Dict of outputs from all specialist agents.
        session_id:               Session ID for memory logging.
        generate_patient_summary: Whether to also produce a patient-friendly version.

    Returns:
        Dict with clinician_summary, patient_summary, agents_consulted, all_tools_used.
    """
    logger.info(f"[SummariserAgent] Synthesising {len(agent_outputs)} agent outputs")

    combined = _build_combined_input(original_query, agent_outputs)

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=PRIMARY_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    # Clinician summary
    try:
        resp = llm.invoke([
            SystemMessage(content=CLINICIAN_PROMPT),
            HumanMessage(content=combined),
        ])
        clinician_summary = resp.content
    except Exception as e:
        logger.error(f"[SummariserAgent] Clinician summary error: {e}")
        clinician_summary = f"Synthesis failed: {e}"

    # Patient summary
    patient_summary = ""
    if generate_patient_summary:
        try:
            resp2 = llm.invoke([
                SystemMessage(content=PATIENT_PROMPT),
                HumanMessage(content=f"Simplify this for a patient:\n\n{clinician_summary}"),
            ])
            patient_summary = resp2.content
        except Exception as e:
            logger.error(f"[SummariserAgent] Patient summary error: {e}")

    # Collect all MCP tools used across all agents
    all_tools = []
    for out in agent_outputs.values():
        if isinstance(out, dict):
            all_tools.extend(out.get("tools_called", []))
    unique_tools = list(set(all_tools))

    # Save to long-term memory
    save_clinical_note(
        session_id=session_id,
        note_type="synthesis",
        content=clinician_summary,
        metadata={
            "query":       original_query,
            "agents_used": list(agent_outputs.keys()),
            "tools_used":  unique_tools,
        },
    )

    log_agent_trace(
        session_id=session_id,
        agent_name="SummariserAgent",
        tool_calls=["ChatGroq(clinician)", "ChatGroq(patient)"],
        result_summary=clinician_summary[:300],
    )

    return {
        "agent":             "SummariserAgent",
        "clinician_summary": clinician_summary,
        "patient_summary":   patient_summary,
        "agents_consulted":  list(agent_outputs.keys()),
        "all_tools_used":    unique_tools,
    }