"""
mcp/tool_definitions.py — All clinical tools as LangChain @tool functions.

This is the MCP (tool-calling) layer. The LLM autonomously decides:
  - WHICH tools to call
  - IN WHAT ORDER
  - HOW MANY TIMES
  - WHEN it has enough information to stop and answer

11 tools registered:
  1.  rag_search                — ChromaDB local knowledge base
  2.  pubmed_search             — PubMed / NCBI literature
  3.  drug_label_lookup         — OpenFDA drug labels
  4.  drug_interaction_check    — OpenFDA drug interactions
  5.  adverse_events_lookup     — FDA FAERS adverse events
  6.  icd10_lookup              — ICD-10-CM codes (NLM)
  7.  snomed_lookup             — SNOMED CT concepts (NLM)
  8.  rxnorm_lookup             — RxNorm drug codes (NLM)
  9.  tavily_web_search         — General internet search (Tavily)
  10. tavily_medical_web_search — Medical-domain-restricted search (Tavily)
  11. analyse_medical_image_tool— Gemini Vision medical image analysis
"""

import logging
from langchain.tools import tool

from rag.retriever import retrieve_context
from tools.pubmed_tool import search_and_fetch, format_articles_for_prompt
from tools.openfda_tool import (
    search_drug_label,
    check_drug_interactions,
    search_drug_adverse_events,
)
from tools.icd_tool import lookup_icd10, lookup_snomed, lookup_rxnorm, format_icd_results
from tools.web_search_tool import (
    tavily_search,
    tavily_medical_search,
    format_search_results_for_prompt,
    is_tavily_configured,
)
from tools.vision_tool import analyse_medical_image
from config.settings import TOP_K_RESULTS, PUBMED_MAX_RESULTS

logger = logging.getLogger(__name__)


# ── 1. RAG Search ─────────────────────────────────────────────────────────────
@tool
def rag_search(query: str) -> str:
    """Search the local medical knowledge base (ChromaDB) using semantic similarity.
    Use this FIRST for any clinical question. Searches ingested medical documents,
    guidelines, and textbooks. Returns the most relevant chunks with source info.
    Args:
        query: Clinical question or topic to search for.
    """
    try:
        ctx = retrieve_context(query, n_results=TOP_K_RESULTS)
        if not ctx or "No relevant context" in ctx:
            return "No relevant information found in the local knowledge base for this query."
        return f"KNOWLEDGE BASE RESULTS:\n{ctx}"
    except Exception as e:
        logger.error(f"[rag_search] {e}")
        return f"Knowledge base search failed: {e}"


# ── 2. PubMed Search ──────────────────────────────────────────────────────────
@tool
def pubmed_search(query: str) -> str:
    """Search PubMed/NCBI for peer-reviewed medical literature and abstracts.
    Use for evidence-based questions, clinical trials, systematic reviews,
    meta-analyses. Returns titles, abstracts, authors, journal, year, PubMed URLs.
    Args:
        query: Medical topic or clinical question to search PubMed for.
    """
    try:
        articles = search_and_fetch(query, max_results=PUBMED_MAX_RESULTS)
        if not articles:
            return f"No PubMed articles found for: '{query}'. Try a broader search term."
        return f"PUBMED RESULTS ({len(articles)} articles):\n{format_articles_for_prompt(articles)}"
    except Exception as e:
        logger.error(f"[pubmed_search] {e}")
        return f"PubMed search failed: {e}"


# ── 3. Drug Label Lookup ──────────────────────────────────────────────────────
@tool
def drug_label_lookup(drug_name: str) -> str:
    """Look up FDA-approved drug label information from OpenFDA.
    Returns indications, warnings, dosing, adverse reactions, contraindications.
    Use for questions about a specific drug's clinical profile.
    Args:
        drug_name: Generic or brand name of the drug (e.g. 'metformin', 'Glucophage').
    """
    try:
        data = search_drug_label(drug_name)
        if "error" in data:
            return f"No FDA label data found for '{drug_name}'."
        return (
            f"FDA DRUG LABEL — {drug_name.upper()}\n"
            f"Brand Names: {', '.join(data.get('brand_names', ['N/A']))}\n"
            f"Generic Names: {', '.join(data.get('generic_names', ['N/A']))}\n"
            f"Indications: {data.get('indications', 'N/A')}\n"
            f"Warnings: {data.get('warnings', 'N/A')}\n"
            f"Dosage: {data.get('dosage', 'N/A')}\n"
            f"Adverse Reactions: {data.get('adverse_reactions', 'N/A')}\n"
            f"Contraindications: {data.get('contraindications', 'N/A')}"
        )
    except Exception as e:
        logger.error(f"[drug_label_lookup] {e}")
        return f"Drug lookup failed for '{drug_name}': {e}"


# ── 4. Drug Interaction Check ─────────────────────────────────────────────────
@tool
def drug_interaction_check(drugs: str) -> str:
    """Check for drug-drug interactions between multiple medications using OpenFDA.
    ALWAYS call this when 2 or more drugs are mentioned.
    Args:
        drugs: Comma-separated drug names (e.g. 'warfarin, amoxicillin, aspirin').
    """
    try:
        drug_list = [d.strip() for d in drugs.split(",") if d.strip()]
        if len(drug_list) < 2:
            return "Provide at least 2 drug names separated by commas."
        data = check_drug_interactions(drug_list)
        interactions = data.get("interactions", {})
        parts = [f"DRUG INTERACTION CHECK — {', '.join(drug_list)}"]
        for drug, info in interactions.items():
            parts.append(f"\n{drug.upper()}:\n{info}")
        return "\n".join(parts)
    except Exception as e:
        logger.error(f"[drug_interaction_check] {e}")
        return f"Interaction check failed: {e}"


# ── 5. Adverse Events Lookup ──────────────────────────────────────────────────
@tool
def adverse_events_lookup(drug_name: str) -> str:
    """Look up most commonly reported adverse events for a drug from FDA FAERS database.
    Use to understand real-world safety profile of a drug.
    Args:
        drug_name: Name of the drug to look up adverse events for.
    """
    try:
        data = search_drug_adverse_events(drug_name)
        if "error" in data:
            return f"No adverse event data found for '{drug_name}'."
        events = data.get("top_adverse_events", [])
        if not events:
            return f"No adverse events reported in FAERS for '{drug_name}'."
        return (
            f"FDA FAERS ADVERSE EVENTS — {drug_name.upper()}\n"
            + "\n".join(f"  • {e}" for e in events)
        )
    except Exception as e:
        logger.error(f"[adverse_events_lookup] {e}")
        return f"Adverse events lookup failed: {e}"


# ── 6. ICD-10 Lookup ──────────────────────────────────────────────────────────
@tool
def icd10_lookup(term: str) -> str:
    """Look up ICD-10-CM diagnostic codes for a clinical condition or symptom.
    Use to find correct diagnostic codes for any condition in a differential.
    Args:
        term: Clinical condition or symptom (e.g. 'type 2 diabetes', 'chest pain').
    """
    try:
        results = lookup_icd10(term, max_results=5)
        if not results:
            return f"No ICD-10 codes found for '{term}'."
        return f"ICD-10 CODES for '{term}':\n{format_icd_results(results)}"
    except Exception as e:
        logger.error(f"[icd10_lookup] {e}")
        return f"ICD-10 lookup failed: {e}"


# ── 7. SNOMED Lookup ──────────────────────────────────────────────────────────
@tool
def snomed_lookup(term: str) -> str:
    """Look up SNOMED CT clinical concept IDs for a medical term.
    Use for standardised clinical terminology and concept mapping.
    Args:
        term: Medical term or concept to look up in SNOMED CT.
    """
    try:
        results = lookup_snomed(term, max_results=5)
        if not results:
            return f"No SNOMED CT concepts found for '{term}'."
        lines = [f"  SNOMED {r['snomed_id']} — {r['name']}" for r in results]
        return f"SNOMED CT CONCEPTS for '{term}':\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"[snomed_lookup] {e}")
        return f"SNOMED lookup failed: {e}"


# ── 8. RxNorm Lookup ──────────────────────────────────────────────────────────
@tool
def rxnorm_lookup(drug_name: str) -> str:
    """Look up RxNorm standardised drug codes for a medication.
    Use to get the standardised drug identifier used in clinical systems.
    Args:
        drug_name: Drug name to look up in RxNorm.
    """
    try:
        results = lookup_rxnorm(drug_name, max_results=5)
        if not results:
            return f"No RxNorm codes found for '{drug_name}'."
        lines = [f"  RxNorm {r['rxnorm_id']} — {r['name']}" for r in results]
        return f"RXNORM CODES for '{drug_name}':\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"[rxnorm_lookup] {e}")
        return f"RxNorm lookup failed: {e}"


# ── 9. Tavily Web Search (General) ────────────────────────────────────────────
@tool
def tavily_web_search(query: str) -> str:
    """Search the internet using Tavily for current, up-to-date medical information.
    Use when: query asks for 'latest'/'recent'/'current'/'2024'/'2025', or when
    local knowledge base and PubMed returned insufficient results.
    Returns web results with source URLs, prioritising trusted medical sources.
    Args:
        query: Search query for internet search.
    """
    if not is_tavily_configured():
        return "Tavily web search not configured. Add TAVILY_API_KEY to .env (free: https://app.tavily.com)"
    try:
        results = tavily_search(query)
        formatted = format_search_results_for_prompt(results)
        return f"WEB SEARCH RESULTS:\n{formatted}" if formatted else f"No web results for: '{query}'"
    except Exception as e:
        logger.error(f"[tavily_web_search] {e}")
        return f"Web search failed: {e}"


# ── 10. Tavily Medical Search (Domain-restricted) ─────────────────────────────
@tool
def tavily_medical_web_search(query: str) -> str:
    """Search the internet restricted to trusted medical authority websites only.
    Sources: PubMed, NEJM, BMJ, Lancet, JAMA, WHO, CDC, NICE, Mayo Clinic,
    Cochrane Library. PREFER this over tavily_web_search for clinical questions.
    Falls back to open web automatically if medical domains return nothing.
    Args:
        query: Clinical question to search on trusted medical websites.
    """
    if not is_tavily_configured():
        return "Tavily web search not configured. Add TAVILY_API_KEY to .env (free: https://app.tavily.com)"
    try:
        results = tavily_medical_search(query)
        formatted = format_search_results_for_prompt(results)
        if not formatted or "no results" in formatted.lower():
            results   = tavily_search(query)
            formatted = format_search_results_for_prompt(results)
        return f"MEDICAL WEB SEARCH RESULTS:\n{formatted}" if formatted else f"No medical web results for: '{query}'"
    except Exception as e:
        logger.error(f"[tavily_medical_web_search] {e}")
        return f"Medical web search failed: {e}"


# ── 11. Medical Image Analysis ────────────────────────────────────────────────
@tool
def analyse_medical_image_tool(image_path: str, clinical_context: str = "") -> str:
    """Analyse a medical image (X-ray, CT, MRI, pathology slide, skin lesion,
    lab report photo) using Gemini Vision AI. Returns image type, key findings,
    clinical observations, and differential considerations.
    Args:
        image_path: File path to the medical image.
        clinical_context: Optional patient context to guide analysis.
    """
    try:
        result = analyse_medical_image(image_path, clinical_context=clinical_context)
        return f"MEDICAL IMAGE ANALYSIS:\n{result}" if result else "Image analysis returned no output."
    except Exception as e:
        logger.error(f"[analyse_medical_image_tool] {e}")
        return f"Image analysis failed: {e}"


# ── Tool registries ───────────────────────────────────────────────────────────

ALL_TOOLS = [
    rag_search, pubmed_search,
    drug_label_lookup, drug_interaction_check, adverse_events_lookup,
    icd10_lookup, snomed_lookup, rxnorm_lookup,
    tavily_web_search, tavily_medical_web_search,
    analyse_medical_image_tool,
]

DIAGNOSIS_TOOLS = [
    rag_search, pubmed_search,
    icd10_lookup, snomed_lookup,
    tavily_medical_web_search, tavily_web_search,
]

DRUG_TOOLS = [
    drug_label_lookup, drug_interaction_check, adverse_events_lookup,
    rxnorm_lookup, rag_search,
    tavily_medical_web_search,
]

LITERATURE_TOOLS = [
    pubmed_search, rag_search,
    tavily_medical_web_search, tavily_web_search,
]

IMAGE_TOOLS = [
    analyse_medical_image_tool,
    rag_search, icd10_lookup,
    tavily_medical_web_search,
]

WEB_TOOLS = [tavily_medical_web_search, tavily_web_search]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}