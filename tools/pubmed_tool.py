"""
tools/pubmed_tool.py — PubMed / NCBI E-utilities wrapper.
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""
import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import PUBMED_BASE, NCBI_EMAIL, PUBMED_MAX_RESULTS

logger = logging.getLogger(__name__)

BASE_PARAMS = {"tool": "MedicalClinicalAssistant", "email": NCBI_EMAIL}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def _get(endpoint: str, params: Dict) -> requests.Response:
    resp = requests.get(f"{PUBMED_BASE}/{endpoint}", params={**BASE_PARAMS, **params}, timeout=15)
    resp.raise_for_status()
    return resp


def search_pubmed(query: str, max_results: int = PUBMED_MAX_RESULTS) -> List[str]:
    """Search PubMed and return a list of PMIDs."""
    try:
        resp = _get("esearch.fcgi", {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
        })
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        return ids
    except Exception as e:
        logger.error(f"PubMed search error: {e}")
        return []


def fetch_abstracts(pmids: List[str]) -> List[Dict[str, Any]]:
    """Fetch abstract details for a list of PMIDs."""
    if not pmids:
        return []
    try:
        resp = _get("efetch.fcgi", {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        })
        return _parse_pubmed_xml(resp.text)
    except Exception as e:
        logger.error(f"PubMed fetch error: {e}")
        return []


def _parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    """Parse PubMed XML response into structured article dicts."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            year_el = article.find(".//PubDate/Year")
            journal_el = article.find(".//Journal/Title")

            authors = []
            for author in article.findall(".//Author")[:3]:
                last = author.findtext("LastName", "")
                first = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {first}".strip())

            articles.append({
                "pmid": pmid_el.text if pmid_el is not None else "N/A",
                "title": title_el.text if title_el is not None else "No title",
                "abstract": abstract_el.text if abstract_el is not None else "No abstract available.",
                "year": year_el.text if year_el is not None else "N/A",
                "journal": journal_el.text if journal_el is not None else "N/A",
                "authors": authors,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_el.text}/" if pmid_el is not None else "",
            })
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
    return articles


def search_and_fetch(query: str, max_results: int = PUBMED_MAX_RESULTS) -> List[Dict[str, Any]]:
    """One-shot: search PubMed then fetch abstracts."""
    pmids = search_pubmed(query, max_results)
    return fetch_abstracts(pmids)


def format_articles_for_prompt(articles: List[Dict[str, Any]]) -> str:
    """Format articles as a readable block for LLM context."""
    if not articles:
        return "No relevant literature found."
    parts = []
    for a in articles:
        parts.append(
            f"PMID {a['pmid']} | {a['year']} | {a['journal']}\n"
            f"Title: {a['title']}\n"
            f"Authors: {', '.join(a['authors'])}\n"
            f"Abstract: {a['abstract'][:500]}...\n"
            f"URL: {a['url']}"
        )
    return "\n\n".join(parts)