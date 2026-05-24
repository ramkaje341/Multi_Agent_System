"""
agents/orchestrator.py 

Every specialist agent uses its own internal MCP ReAct tool-calling loop.

Pipeline:
  classify_intent
       ↓
  diagnosis_node   ← MCP ReAct: rag_search, icd10_lookup, pubmed_search, tavily…
       ↓
  drug_node        ← MCP ReAct: drug_label_lookup, drug_interaction_check, rag_search…
       ↓
  literature_node  ← MCP ReAct: pubmed_search, rag_search, tavily_medical_web_search…
       ↓
  image_node       ← MCP ReAct: analyse_medical_image_tool, rag_search, icd10_lookup…
       ↓
  web_search_node  ← MCP ReAct: tavily_medical_web_search, tavily_web_search
       |               Fires when: explicit trigger OR confidence < threshold
       ↓
  summarise_node   ← Plain LLM (no tools) — synthesises all agent outputs
       ↓
  reflect_node     ← Plain LLM QA check — retries summarise once if FAIL
       ↓
  END
"""
import logging
import uuid
from typing import TypedDict, Dict, Any, List, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from agents.diagnosis_agent  import run_diagnosis_agent
from agents.drug_agent        import run_drug_agent
from agents.literature_agent  import run_literature_agent
from agents.summariser_agent  import run_summariser_agent
from agents.web_search_agent  import run_web_search_agent
from mcp.tool_definitions     import IMAGE_TOOLS
from mcp.tool_executor        import run_tool_call_loop
from tools.web_search_tool    import is_tavily_configured
from memory.memory_store      import add_turn, get_history, log_agent_trace
from config.settings          import (
    GROQ_API_KEY, PRIMARY_MODEL, MAX_TOKENS, TEMPERATURE,
    MAX_REFLECTION_LOOPS, FALLBACK_CONFIDENCE_THR,
)

logger = logging.getLogger(__name__)

# Keywords that always trigger Tavily web search
WEB_TRIGGER_KEYWORDS = [
    "latest", "recent", "current", "2024", "2025",
    "new guideline", "updated", "just approved", "newly approved",
    "fda approved", "news", "emerging", "novel", "breakthrough", "trial results",
]

# Phrases that indicate an agent failed to find useful information
LOW_CONFIDENCE_PHRASES = [
    "no relevant", "not found", "no data", "no results", "unable to find",
    "could not find", "no information", "not available", "error fetching",
    "no label data", "no abstract", "failed", "no relevant literature",
    "no icd", "no context found", "no content extracted",
    "i don't know", "insufficient", "n/a", "none found",
]


# ─── State ────────────────────────────────────────────────────────────────────

class ClinicalState(TypedDict):
    session_id:          str
    original_query:      str
    patient_context:     str
    intent:              str
    active_agents:       List[str]
    needs_web_search:    bool
    agent_outputs:       Dict[str, Any]
    image_path:          str | None
    messages:            Annotated[List[BaseMessage], add_messages]
    final_response:      str
    patient_summary:     str
    reflection_count:    int
    reflection_passed:   bool
    pipeline_confidence: float
    web_search_skipped:  bool


# ─── Confidence scoring ───────────────────────────────────────────────────────

def _score_output(output: dict) -> float:
    """Score a single agent output 0.0 (failed) to 1.0 (rich)."""
    if not isinstance(output, dict):
        return 0.0

    text = (
        output.get("result") or
        output.get("diagnosis") or
        output.get("clinician_summary") or ""
    ).strip()

    # Having a tool trace is a positive signal even if text is short
    has_trace = bool(output.get("tool_trace"))
    has_citations = bool(output.get("citations"))

    if not text:
        return 0.1 if (has_trace or has_citations) else 0.0
    if len(text) < 150:
        return 0.2

    low_hits = sum(1 for p in LOW_CONFIDENCE_PHRASES if p in text.lower())
    if low_hits >= 3: return 0.25
    if low_hits >= 1: return 0.4
    return 1.0 if len(text) >= 500 else 0.7


def _pipeline_confidence(outputs: dict) -> float:
    """Compute overall confidence across all agent outputs (0.0–1.0)."""
    if not outputs:
        return 0.0
    scores = [_score_output(v) for v in outputs.values() if isinstance(v, dict)]
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    if min(scores) == 0.0:
        avg *= 0.6  # Penalise if any agent completely failed
    return round(avg, 3)


# ─── Intent classifier ────────────────────────────────────────────────────────

INTENT_PROMPT = """Classify this clinical query into one or more categories.
Reply with ONLY a comma-separated list — no explanation, no punctuation.

Categories:
- diagnosis:   symptoms, conditions, differential diagnosis, clinical presentation
- drug:        medications, dosing, interactions, adverse effects, pharmacology
- literature:  evidence, studies, clinical guidelines, systematic reviews
- image:       X-ray, scan, MRI, CT, pathology, image, photo, picture
- web:         asks for latest/recent/current/news, or very new topic unlikely in static knowledge base

Examples:
"chest pain with dyspnoea" → diagnosis
"metformin dose in CKD" → drug
"SGLT2 inhibitors in HFrEF evidence" → literature
"latest FDA approved drugs for Alzheimer 2024" → drug,web
"analyse this chest X-ray" → image,diagnosis
"current WHO TB treatment guidelines" → literature,web
"patient on warfarin started amoxicillin" → drug,literature

Query: {query}
"""


def classify_intent(state: ClinicalState) -> ClinicalState:
    logger.info(f"[Orchestrator] Classifying: '{state['original_query'][:60]}'")
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY, model=PRIMARY_MODEL,
            temperature=0, max_tokens=60,
        )
        resp    = llm.invoke([HumanMessage(content=INTENT_PROMPT.format(
            query=state["original_query"]
        ))])
        raw     = resp.content.strip().lower()
        intents = [
            i.strip() for i in raw.split(",")
            if i.strip() in ("diagnosis", "drug", "literature", "image", "web")
        ]
        if not intents:
            intents = ["diagnosis"]
    except Exception as e:
        logger.error(f"[Orchestrator] Classification failed: {e}")
        intents = ["diagnosis"]

    active_agents = []
    if "diagnosis"  in intents: active_agents.append("DiagnosisAgent")
    if "drug"       in intents: active_agents.append("DrugAgent")
    if "literature" in intents: active_agents.append("LiteratureAgent")
    if "image"      in intents and state.get("image_path"):
        active_agents.append("ImageAgent")

    query_lower  = state["original_query"].lower()
    needs_web    = (
        "web" in intents or
        any(kw in query_lower for kw in WEB_TRIGGER_KEYWORDS)
    )

    logger.info(
        f"[Orchestrator] Intents={intents} | "
        f"Agents={active_agents} | WebSearch={needs_web}"
    )

    return {
        **state,
        "intent":              intents[0],
        "active_agents":       active_agents,
        "needs_web_search":    needs_web,
        "agent_outputs":       {},
        "reflection_count":    0,
        "reflection_passed":   False,
        "pipeline_confidence": 0.0,
        "web_search_skipped":  False,
    }


# ─── Specialist agent nodes ───────────────────────────────────────────────────

def diagnosis_node(state: ClinicalState) -> ClinicalState:
    if "DiagnosisAgent" not in state["active_agents"]:
        return state
    result = run_diagnosis_agent(
        patient_query=state["original_query"],
        session_id=state["session_id"],
        patient_context=state.get("patient_context", ""),
    )
    return {**state, "agent_outputs": {**state["agent_outputs"], "DiagnosisAgent": result}}


def drug_node(state: ClinicalState) -> ClinicalState:
    if "DrugAgent" not in state["active_agents"]:
        return state
    result = run_drug_agent(
        query=state["original_query"],
        session_id=state["session_id"],
    )
    return {**state, "agent_outputs": {**state["agent_outputs"], "DrugAgent": result}}


def literature_node(state: ClinicalState) -> ClinicalState:
    if "LiteratureAgent" not in state["active_agents"]:
        return state
    result = run_literature_agent(
        query=state["original_query"],
        session_id=state["session_id"],
    )
    return {**state, "agent_outputs": {**state["agent_outputs"], "LiteratureAgent": result}}


def image_node(state: ClinicalState) -> ClinicalState:
    """Image agent — full MCP ReAct loop using IMAGE_TOOLS."""
    if "ImageAgent" not in state["active_agents"] or not state.get("image_path"):
        return state

    IMAGE_SYSTEM = """You are a medical image analysis specialist with access to tools.

TOOL USAGE STRATEGY:
1. analyse_medical_image_tool → ALWAYS call first with the image_path provided.
2. rag_search                 → Search for clinical context relevant to your findings.
3. icd10_lookup               → Get ICD-10 codes for any conditions you identify.
4. tavily_medical_web_search  → Search for more information about specific findings.

Provide a structured clinical image analysis report including:
- Image type and quality assessment
- Key radiological / pathological findings
- Clinical significance
- Differential diagnosis with ICD-10 codes
- Recommended follow-up investigations
"""

    try:
        llm_with_tools = ChatGroq(
            api_key=GROQ_API_KEY, model=PRIMARY_MODEL,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        ).bind_tools(IMAGE_TOOLS)

        final_text, tool_trace, _ = run_tool_call_loop(
            llm_with_tools=llm_with_tools,
            messages=[
                SystemMessage(content=IMAGE_SYSTEM),
                HumanMessage(content=(
                    f"Image file path: {state['image_path']}\n"
                    f"Patient context: {state.get('patient_context', 'None provided')}\n"
                    f"Clinical query: {state['original_query']}"
                )),
            ],
        )
        result = {
            "agent":        "ImageAgent",
            "result":       final_text,
            "tools_called": [t["tool"] for t in tool_trace],
            "tool_trace":   tool_trace,
        }
    except Exception as e:
        logger.error(f"[ImageNode] Error: {e}")
        result = {
            "agent":        "ImageAgent",
            "result":       f"Image analysis failed: {e}",
            "tools_called": [],
            "tool_trace":   [],
        }

    return {**state, "agent_outputs": {**state["agent_outputs"], "ImageAgent": result}}


# ─── Web search node (Tavily fallback) ───────────────────────────────────────

def web_search_node(state: ClinicalState) -> ClinicalState:
    """
    Tavily web search with two-tier trigger:

    TIER 1 — Explicit trigger (always fire):
      • Intent classifier flagged 'web'
      • Query contains recency keywords (latest, 2024, current guidelines…)

    TIER 2 — Confidence-based fallback (fire because prior agents failed):
      • Pipeline confidence score < FALLBACK_CONFIDENCE_THR (default 0.5)
      • Scored by inspecting actual text quality + failure phrase detection
    """
    outputs    = state.get("agent_outputs", {})
    confidence = _pipeline_confidence(outputs)
    explicit   = state.get("needs_web_search", False)
    fallback   = confidence < FALLBACK_CONFIDENCE_THR

    logger.info(
        f"[WebSearchNode] Confidence={confidence:.2f} | "
        f"Explicit={explicit} | Fallback={fallback}"
    )

    if not (explicit or fallback):
        logger.info(
            f"[WebSearchNode] Skipping — confidence {confidence:.2f} sufficient"
        )
        return {**state, "web_search_skipped": True, "pipeline_confidence": confidence}

    if not is_tavily_configured():
        logger.warning("[WebSearchNode] Tavily not configured — skipping web search fallback")
        return {**state, "web_search_skipped": True, "pipeline_confidence": confidence}

    # Build context summary from prior agents
    context_snippets = []
    for name, out in outputs.items():
        if isinstance(out, dict):
            txt = out.get("result", out.get("diagnosis", ""))
            if txt and len(txt) > 50:
                context_snippets.append(f"{name}: {txt[:300]}")

    trigger_reason = "explicit_trigger" if explicit else f"low_confidence({confidence:.2f})"
    logger.info(f"[WebSearchNode] Triggering Tavily — reason: {trigger_reason}")

    result = run_web_search_agent(
        query=state["original_query"],
        session_id=state["session_id"],
        context_from_other_agents="\n".join(context_snippets),
        force_medical_domains=(fallback and not explicit),
    )
    result["trigger_reason"]             = trigger_reason
    result["pipeline_confidence_before"] = confidence

    return {
        **state,
        "agent_outputs":       {**state["agent_outputs"], "WebSearchAgent": result},
        "active_agents":       state["active_agents"] + ["WebSearchAgent"],
        "pipeline_confidence": confidence,
        "web_search_skipped":  False,
    }


# ─── Summariser node ──────────────────────────────────────────────────────────

def summarise_node(state: ClinicalState) -> ClinicalState:
    result = run_summariser_agent(
        original_query=state["original_query"],
        agent_outputs=state["agent_outputs"],
        session_id=state["session_id"],
    )
    return {
        **state,
        "final_response":  result["clinician_summary"],
        "patient_summary": result["patient_summary"],
    }


# ─── Reflection node ──────────────────────────────────────────────────────────

REFLECTION_PROMPT = """You are a medical QA reviewer evaluating an AI clinical response.

Query: {query}
Response (first 1500 chars): {response}

Check ALL of the following:
1. Does it directly answer the query?
2. Does it include a safety disclaimer / "verify with clinician" statement?
3. Are there any dangerous, incorrect, or unsupported recommendations?
4. Is the response clinically coherent and well-structured?

Reply with ONLY one of:
PASS
FAIL: [one-line reason]
"""


def reflect_node(state: ClinicalState) -> ClinicalState:
    if state["reflection_count"] >= MAX_REFLECTION_LOOPS:
        logger.info("[Orchestrator] Max reflection loops reached — passing through")
        return {**state, "reflection_passed": True}
    try:
        llm  = ChatGroq(
            api_key=GROQ_API_KEY, model=PRIMARY_MODEL,
            temperature=0, max_tokens=60,
        )
        resp = llm.invoke([HumanMessage(content=REFLECTION_PROMPT.format(
            query=state["original_query"],
            response=state["final_response"][:1500],
        ))])
        passed = resp.content.strip().upper().startswith("PASS")
        logger.info(f"[Orchestrator] Reflection: {'PASS' if passed else 'FAIL — ' + resp.content[:60]}")
    except Exception as e:
        logger.error(f"[Orchestrator] Reflection error: {e} — defaulting to PASS")
        passed = True

    return {
        **state,
        "reflection_passed": passed,
        "reflection_count":  state["reflection_count"] + 1,
    }


def should_retry(state: ClinicalState) -> str:
    return "end" if state["reflection_passed"] else "retry"


# ─── Build LangGraph ──────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ClinicalState)

    g.add_node("classify",   classify_intent)
    g.add_node("diagnosis",  diagnosis_node)
    g.add_node("drug",       drug_node)
    g.add_node("literature", literature_node)
    g.add_node("image",      image_node)
    g.add_node("web_search", web_search_node)
    g.add_node("summarise",  summarise_node)
    g.add_node("reflect",    reflect_node)

    g.set_entry_point("classify")
    g.add_edge("classify",   "diagnosis")
    g.add_edge("diagnosis",  "drug")
    g.add_edge("drug",       "literature")
    g.add_edge("literature", "image")
    g.add_edge("image",      "web_search")
    g.add_edge("web_search", "summarise")
    g.add_edge("summarise",  "reflect")
    g.add_conditional_edges(
        "reflect", should_retry,
        {"end": END, "retry": "summarise"},
    )

    return g.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─── Public entry point ───────────────────────────────────────────────────────

def run_pipeline(
    query:           str,
    session_id:      str | None = None,
    patient_context: str = "",
    image_path:      str | None = None,
) -> Dict[str, Any]:
    """
    Main entry point — runs the full MCP multi-agent pipeline.

    Args:
        query:           Clinical question from user/clinician.
        session_id:      Session ID (auto-generated if None).
        patient_context: Optional patient demographics / history.
        image_path:      Optional path to a medical image file.

    Returns:
        Dict with final_response, patient_summary, agents_consulted,
        all_mcp_tool_calls, pipeline_confidence, web_search_used.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    add_turn(session_id, "user", query)

    history  = get_history(session_id, last_n=6)
    messages = [
        HumanMessage(content=f"{h['role']}: {h['content']}")
        for h in history
    ]

    initial: ClinicalState = {
        "session_id":          session_id,
        "original_query":      query,
        "patient_context":     patient_context,
        "intent":              "",
        "active_agents":       [],
        "needs_web_search":    False,
        "agent_outputs":       {},
        "image_path":          image_path,
        "messages":            messages,
        "final_response":      "",
        "patient_summary":     "",
        "reflection_count":    0,
        "reflection_passed":   False,
        "pipeline_confidence": 0.0,
        "web_search_skipped":  False,
    }

    final = get_graph().invoke(initial)

    add_turn(session_id, "assistant", final["final_response"])

    # Collect all MCP tool calls made across all agents
    all_tool_calls = []
    for out in final["agent_outputs"].values():
        if isinstance(out, dict):
            all_tool_calls.extend(out.get("tools_called", []))

    log_agent_trace(
        session_id=session_id,
        agent_name="Orchestrator",
        tool_calls=final["active_agents"],
        result_summary=f"Done. Agents={final['active_agents']} Tools={list(set(all_tool_calls))}",
    )

    return {
        "session_id":          session_id,
        "final_response":      final["final_response"],
        "patient_summary":     final["patient_summary"],
        "agents_consulted":    final["active_agents"],
        "agent_outputs":       final["agent_outputs"],
        "intent":              final["intent"],
        "web_search_used":     "WebSearchAgent" in final["active_agents"],
        "web_search_skipped":  final.get("web_search_skipped", True),
        "pipeline_confidence": final.get("pipeline_confidence", 0.0),
        "all_mcp_tool_calls":  list(set(all_tool_calls)),
    }