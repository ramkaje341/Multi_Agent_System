"""
ui/app.py — Streamlit frontend for the Multi-Agent Medical Clinical Assistant.
Full MCP tool-call architecture with ReAct loop visualisation.

Run: streamlit run ui/app.py
  or: python run.py
"""
import sys, os, uuid, logging, tempfile
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.orchestrator import run_pipeline
from memory.memory_store import get_history, get_traces, clear_history
from rag.vectorstore import count_documents
from tools.multimodal_parser import parse_input

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.agent-badge {
    display:inline-block; padding:2px 10px; border-radius:12px;
    font-size:0.75rem; font-weight:600; margin:2px;
}
.badge-diagnosis  { background:#dbeafe; color:#1d4ed8; }
.badge-drug       { background:#dcfce7; color:#15803d; }
.badge-literature { background:#fef9c3; color:#854d0e; }
.badge-image      { background:#f3e8ff; color:#7e22ce; }
.badge-web        { background:#fee2e2; color:#dc2626; }
.tool-step {
    font-family: monospace; font-size:0.78rem;
    background:#f8fafc; border-left:3px solid #94a3b8;
    padding:4px 10px; margin:2px 0; border-radius:0 4px 4px 0;
}
.conf-high { color:#15803d; font-weight:600; }
.conf-mid  { color:#b45309; font-weight:600; }
.conf-low  { color:#b91c1c; font-weight:600; }
.disclaimer { background:#fff7ed; border-left:4px solid #f97316;
              padding:8px 14px; border-radius:4px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "session_id"    not in st.session_state: st.session_state.session_id    = str(uuid.uuid4())
if "messages"      not in st.session_state: st.session_state.messages      = []
if "pending_query" not in st.session_state: st.session_state.pending_query = None

# ─── Helpers ──────────────────────────────────────────────────────────────────
BADGE_MAP = {
    "DiagnosisAgent":  ("Diagnosis",  "badge-diagnosis"),
    "DrugAgent":       ("Drug",       "badge-drug"),
    "LiteratureAgent": ("Literature", "badge-literature"),
    "ImageAgent":      ("Image",      "badge-image"),
    "WebSearchAgent":  ("🌐 Web",     "badge-web"),
}

TOOL_ICONS = {
    "rag_search":                "📚",
    "pubmed_search":             "🔬",
    "drug_label_lookup":         "💊",
    "drug_interaction_check":    "⚠️",
    "adverse_events_lookup":     "🚨",
    "icd10_lookup":              "🏷️",
    "snomed_lookup":             "🏷️",
    "rxnorm_lookup":             "💊",
    "tavily_web_search":         "🌐",
    "tavily_medical_web_search": "🌐✓",
    "analyse_medical_image_tool":"🩻",
}

def conf_class(c: float) -> str:
    return "conf-high" if c >= 0.7 else "conf-mid" if c >= 0.4 else "conf-low"

def render_tool_trace(agent_name: str, output: dict):
    """Render MCP tool trace for one agent as collapsible steps."""
    trace = output.get("tool_trace", [])
    tools = output.get("tools_called", [])
    if not trace:
        return
    with st.expander(f"🔧 {agent_name} — MCP tool calls ({len(trace)} steps)"):
        for step in trace:
            icon = TOOL_ICONS.get(step["tool"], "🔧")
            args_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in step["args"].items())
            st.markdown(
                f'<div class="tool-step">'
                f'Step {step["iteration"]} &nbsp;{icon}&nbsp; '
                f'<b>{step["tool"]}</b>({args_str}) '
                f'→ {step["result_len"]} chars</div>',
                unsafe_allow_html=True,
            )
            if step.get("result"):
                with st.expander(f"  ↳ Result preview", expanded=False):
                    st.text(step["result"][:400] + ("…" if len(step["result"]) > 400 else ""))

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Clinical Assistant")
    st.caption("MCP Tool-Calling · LangGraph · ReAct")
    st.divider()

    st.subheader("📋 Session")
    st.code(st.session_state.session_id[:20] + "...", language=None)
    try:
        doc_count = count_documents()
        st.metric("Knowledge Base", f"{doc_count} chunks" if doc_count > 0 else "⚠️ Empty — run ingest.py")
    except Exception:
        st.metric("Knowledge Base", "Not initialised")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear", use_container_width=True):
            clear_history(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("🔄 New", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.rerun()

    st.divider()
    st.subheader("👤 Patient Context")
    patient_context = st.text_area(
        "Demographics / background (optional)",
        placeholder="e.g. 65yo male, T2DM, HTN, eGFR 45. On metformin 1g BD, lisinopril 10mg.",
        height=100,
    )

    st.divider()
    st.subheader("📎 Upload Document / Image")
    uploaded_file = st.file_uploader(
        "PDF lab report, discharge summary, or medical image",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
    )
    if uploaded_file:
        if uploaded_file.type.startswith("image/"):
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        else:
            st.success(f"📄 {uploaded_file.name} ready")

    st.divider()
    view_mode = st.radio("👁️ Response view:", ["🩺 Clinician", "👤 Patient-friendly"], index=0)

    st.divider()
    with st.expander("🔍 Agent execution traces"):
        try:
            traces = get_traces(st.session_state.session_id)
            if traces:
                for t in reversed(traces[-5:]):
                    st.markdown(f"**{t['agent']}** — {t['timestamp'][11:19]}")
                    st.caption(f"Tools: {', '.join(t['tools'])}")
                    st.caption(t['summary'][:120] + "…")
                    st.divider()
            else:
                st.caption("No traces yet.")
        except Exception:
            st.caption("Traces unavailable.")

    st.divider()
    st.caption("Groq · Gemini · Tavily · PubMed · OpenFDA · ChromaDB · All Free")

# ─── Main area ─────────────────────────────────────────────────────────────────
st.title("🏥 Multi-Agent Medical Clinical Assistant")
st.caption("MCP Tool-Calling Architecture · LangGraph ReAct Pipeline · All Free APIs")

st.markdown("""<div class="disclaimer">
⚠️ <strong>Clinical decision support only.</strong>
All AI outputs must be reviewed and verified by a qualified medical professional.
Never use for autonomous clinical decision-making.
</div>""", unsafe_allow_html=True)
st.markdown("")

# Example queries
with st.expander("💡 Example queries"):
    examples = [
        ("🫀 Diagnosis",      "Patient has 3-day pleuritic chest pain, fever 38.4°C, HR 112, SpO2 94%. What are the differential diagnoses?"),
        ("💊 Drug",           "Patient on warfarin INR 2.8 started metronidazole 400mg TDS for C. diff. What are the risks and management?"),
        ("📚 Evidence",       "What is the current evidence for SGLT2 inhibitors in heart failure with reduced ejection fraction?"),
        ("🧪 Lab result",     "HbA1c 9.4%, eGFR 36, urine ACR 180 mg/mmol. Patient on metformin 1g BD. What changes are needed?"),
        ("🌐 Latest",         "What are the latest 2024 guidelines for management of resistant hypertension?"),
        ("🧠 Emergency",      "GCS 13, sudden onset worst headache of life, neck stiffness, photophobia. Immediate management?"),
    ]
    cols = st.columns(2)
    for i, (label, qtext) in enumerate(examples):
        with cols[i % 2]:
            if st.button(label, key=f"ex_{i}", use_container_width=True, help=qtext):
                st.session_state.pending_query = qtext

# ─── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role   = msg["role"]
    avatar = "👤" if role == "user" else "🏥"
    with st.chat_message(role, avatar=avatar):
        if role == "user":
            st.write(msg["content"])
        else:
            show_patient = "Patient" in view_mode and msg.get("patient_summary")
            if show_patient:
                st.info("**Patient-friendly summary**")
                st.write(msg["patient_summary"])
                with st.expander("Show full clinician report"):
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

            # Badge strip
            agents = msg.get("agents_consulted", [])
            intent = msg.get("intent", "")
            conf   = msg.get("pipeline_confidence", 0.0)
            if agents:
                badges = " ".join(
                    f'<span class="agent-badge {BADGE_MAP.get(a,(" "," "))[1]}">'
                    f'{BADGE_MAP.get(a,(a,""))[0]}</span>'
                    for a in agents
                )
                web_used = msg.get("web_search_used", False)
                web_reason = msg.get("web_trigger_reason", "")
                web_tag = ""
                if web_used:
                    label = "explicit" if "explicit" in web_reason else f"fallback(conf={conf:.2f})"
                    web_tag = f'&nbsp;|&nbsp;🌐 {label}'
                st.markdown(
                    f'<div style="margin-top:6px;font-size:0.8rem">'
                    f'Intent:<code>{intent}</code>&nbsp;|&nbsp;'
                    f'Agents:{badges}&nbsp;|&nbsp;'
                    f'Confidence:<span class="{conf_class(conf)}">{conf:.2f}</span>'
                    f'{web_tag}</div>',
                    unsafe_allow_html=True,
                )

            # MCP tool traces per agent
            for agent_name, output in msg.get("agent_outputs", {}).items():
                if isinstance(output, dict) and output.get("tool_trace"):
                    render_tool_trace(agent_name, output)

            # All MCP tools used summary
            all_tools = msg.get("all_mcp_tool_calls", [])
            if all_tools:
                with st.expander(f"🛠️ All MCP tools used ({len(all_tools)} unique)"):
                    for t in all_tools:
                        icon = TOOL_ICONS.get(t, "🔧")
                        st.markdown(f"- {icon} `{t}`")

            # PubMed citations
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📖 {len(citations)} PubMed references"):
                    for c in citations:
                        st.markdown(
                            f"- **[{c['title'][:80]}...]({c['url']})** "
                            f"— {', '.join(c.get('authors',[])[:2])} ({c['year']}) *{c['journal']}*"
                        )

            # Web search sources
            if "WebSearchAgent" in msg.get("agent_outputs", {}):
                web_out = msg["agent_outputs"]["WebSearchAgent"]
                sources = web_out.get("sources", [])
                stype   = web_out.get("search_type", "")
                if sources:
                    with st.expander(f"🌐 Web sources — {stype} ({len(sources)})"):
                        for s in sources:
                            tag = " ✓" if s.get("trusted") else ""
                            st.markdown(f"- [{s['title'][:70]}]({s['url']}){tag}")

# ─── Chat input ────────────────────────────────────────────────────────────────
query        = st.chat_input("Ask a clinical question, or upload a PDF/image above…")
active_query = query or st.session_state.pop("pending_query", None)

if active_query:
    tmp_path = None
    try:
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            parsed = parse_input(
                text=active_query if active_query.strip() else None,
                pdf_path=tmp_path if uploaded_file.type == "application/pdf" else None,
                image_path=tmp_path if uploaded_file.type.startswith("image/") else None,
                clinical_context=patient_context,
            )
        else:
            parsed = parse_input(text=active_query, clinical_context=patient_context)
    except Exception as e:
        st.error(f"Input parsing error: {e}")
        parsed = None

    if parsed:
        display_query = active_query or f"[{uploaded_file.name}]"
        st.session_state.messages.append({"role": "user", "content": display_query})

        with st.chat_message("user", avatar="👤"):
            st.write(display_query)
            if uploaded_file:
                st.caption(f"📎 {uploaded_file.name} ({parsed.input_type})")

        with st.chat_message("assistant", avatar="🏥"):
            status   = st.empty()
            progress = st.progress(0)
            try:
                status.info("🧠 Classifying intent → routing to specialist agents…")
                progress.progress(10)

                result = run_pipeline(
                    query=parsed.text,
                    session_id=st.session_state.session_id,
                    patient_context=patient_context,
                    image_path=parsed.image_path,
                )

                progress.progress(95)
                status.empty()
                progress.empty()

                response      = result["final_response"]
                patient_sum   = result["patient_summary"]
                agents        = result["agents_consulted"]
                intent        = result["intent"]
                conf          = result["pipeline_confidence"]
                web_used      = result["web_search_used"]
                all_tools     = result["all_mcp_tool_calls"]
                agent_outputs = result["agent_outputs"]

                # Web trigger reason
                web_reason = ""
                if web_used and "WebSearchAgent" in agent_outputs:
                    web_reason = agent_outputs["WebSearchAgent"].get("trigger_reason", "")

                # Citations
                citations = []
                if "LiteratureAgent" in agent_outputs:
                    citations = agent_outputs["LiteratureAgent"].get("citations", [])

                # Display response
                show_patient = "Patient" in view_mode and patient_sum
                if show_patient:
                    st.info("**Patient-friendly summary**")
                    st.write(patient_sum)
                    with st.expander("Show full clinician report"):
                        st.markdown(response)
                else:
                    st.markdown(response)

                # Badge strip
                badges = " ".join(
                    f'<span class="agent-badge {BADGE_MAP.get(a,("",""))[1]}">'
                    f'{BADGE_MAP.get(a,(a,""))[0]}</span>'
                    for a in agents
                )
                web_tag = ""
                if web_used:
                    label = "explicit" if "explicit" in web_reason else f"fallback(conf={conf:.2f})"
                    web_tag = f'&nbsp;|&nbsp;🌐 {label}'
                st.markdown(
                    f'<div style="margin-top:6px;font-size:0.8rem">'
                    f'Intent:<code>{intent}</code>&nbsp;|&nbsp;'
                    f'Agents:{badges}&nbsp;|&nbsp;'
                    f'Confidence:<span class="{conf_class(conf)}">{conf:.2f}</span>'
                    f'{web_tag}</div>',
                    unsafe_allow_html=True,
                )

                # MCP tool traces per agent
                for agent_name, output in agent_outputs.items():
                    if isinstance(output, dict) and output.get("tool_trace"):
                        render_tool_trace(agent_name, output)

                # All MCP tools summary
                if all_tools:
                    with st.expander(f"🛠️ All MCP tools used ({len(all_tools)} unique)"):
                        for t in all_tools:
                            icon = TOOL_ICONS.get(t, "🔧")
                            st.markdown(f"- {icon} `{t}`")

                # PubMed citations
                if citations:
                    with st.expander(f"📖 {len(citations)} PubMed references"):
                        for c in citations:
                            st.markdown(
                                f"- **[{c['title'][:80]}...]({c['url']})** "
                                f"— {', '.join(c.get('authors',[])[:2])} ({c['year']}) *{c['journal']}*"
                            )

                # Web sources
                if "WebSearchAgent" in agent_outputs:
                    web_out = agent_outputs["WebSearchAgent"]
                    sources = web_out.get("sources", [])
                    stype   = web_out.get("search_type", "")
                    if sources:
                        with st.expander(f"🌐 Web sources — {stype} ({len(sources)})"):
                            for s in sources:
                                tag = " ✓" if s.get("trusted") else ""
                                st.markdown(f"- [{s['title'][:70]}]({s['url']}){tag}")

                # Save to session
                st.session_state.messages.append({
                    "role":               "assistant",
                    "content":            response,
                    "patient_summary":    patient_sum,
                    "agents_consulted":   agents,
                    "intent":             intent,
                    "pipeline_confidence": conf,
                    "web_search_used":    web_used,
                    "web_trigger_reason": web_reason,
                    "all_mcp_tool_calls": all_tools,
                    "agent_outputs":      agent_outputs,
                    "citations":          citations,
                })

            except Exception as e:
                status.empty()
                progress.empty()
                st.error(
                    f"**Pipeline error:** {e}\n\n"
                    "**Quick fixes:**\n"
                    "1. Check `GROQ_API_KEY` is set in `.env`\n"
                    "2. Run `python ingest.py --sample` to populate knowledge base\n"
                    "3. Check internet connection (PubMed, OpenFDA, Tavily need internet)"
                )
                logger.error("Pipeline error", exc_info=True)

    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)