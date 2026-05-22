"""
tools/openfda_tool.py — OpenFDA API wrapper.
Docs: https://open.fda.gov/apis/
"""
import logging
import requests
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import OPENFDA_BASE, OPENFDA_MAX_RESULTS

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def _get(url: str, params: Dict) -> Dict:
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def search_drug_label(drug_name: str) -> Dict[str, Any]:
    """
    Fetch drug label information from OpenFDA.
    Returns indications, warnings, dosage, adverse reactions.
    """
    try:
        data = _get(
            f"{OPENFDA_BASE}/label.json",
            params={
                "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                "limit": 1,
            },
        )
        if not data.get("results"):
            return {"error": f"No label data found for '{drug_name}'"}

        result = data["results"][0]
        return {
            "drug_name": drug_name,
            "brand_names": result.get("openfda", {}).get("brand_name", []),
            "generic_names": result.get("openfda", {}).get("generic_name", []),
            "indications": result.get("indications_and_usage", ["N/A"])[0][:800],
            "warnings": result.get("warnings", ["N/A"])[0][:800],
            "dosage": result.get("dosage_and_administration", ["N/A"])[0][:600],
            "adverse_reactions": result.get("adverse_reactions", ["N/A"])[0][:600],
            "contraindications": result.get("contraindications", ["N/A"])[0][:500],
        }
    except Exception as e:
        logger.error(f"OpenFDA label error for '{drug_name}': {e}")
        return {"error": str(e)}


def search_drug_adverse_events(drug_name: str) -> Dict[str, Any]:
    """
    Fetch top adverse event reports for a drug from FAERS.
    """
    try:
        data = _get(
            f"{OPENFDA_BASE}event.json",
            params={
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": 10,
            },
        )
        terms = [r["term"] for r in data.get("results", [])]
        return {"drug_name": drug_name, "top_adverse_events": terms}
    except Exception as e:
        logger.error(f"OpenFDA adverse events error: {e}")
        return {"error": str(e)}


def check_drug_interactions(drugs: List[str]) -> Dict[str, Any]:
    """
    Basic interaction check: fetch labels for each drug and
    extract 'drug_interactions' section.
    """
    interactions = {}
    for drug in drugs:
        try:
            data = _get(
                f"{OPENFDA_BASE}/label.json",
                params={
                    "search": f'openfda.generic_name:"{drug}"',
                    "limit": 1,
                },
            )
            if data.get("results"):
                section = data["results"][0].get("drug_interactions", ["N/A"])[0]
                interactions[drug] = section[:800]
            else:
                interactions[drug] = "No interaction data found."
        except Exception as e:
            interactions[drug] = f"Error: {e}"
    return {"drugs": drugs, "interactions": interactions}