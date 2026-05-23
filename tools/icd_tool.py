"""
tools/icd_tool.py — NLM ClinicalTables API for ICD-10 and SNOMED lookups.
Docs: https://clinicaltables.nlm.nih.gov/
"""
import logging
import requests
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import NLM_CLINICAL_TABLES_BASE

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def _get(url: str, params: Dict) -> Any:
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def lookup_icd10(term: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Look up ICD-10-CM codes for a clinical term.
    Returns list of {code, name} dicts.
    """
    try:
        data = _get(
            f"{NLM_CLINICAL_TABLES_BASE}/icd10cm/v3/search",
            params={"terms": term, "maxList": max_results, "sf": "code,name", "df": "code,name"},
        )
        # Response format: [total, codes_array, extra, display_array]
        codes = data[1] if len(data) > 1 else []
        names = data[3] if len(data) > 3 else []
        results = []
        for code, name_pair in zip(codes, names):
            results.append({"code": code, "name": name_pair[1] if name_pair else ""})
        return results
    except Exception as e:
        logger.error(f"ICD-10 lookup error: {e}")
        return []


def lookup_snomed(term: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Look up SNOMED CT concept IDs for a clinical term.
    """
    try:
        data = _get(
            f"{NLM_CLINICAL_TABLES_BASE}/snomed_ct/v3/search",
            params={"terms": term, "maxList": max_results, "df": "id,display"},
        )
        names = data[3] if len(data) > 3 else []
        results = []
        for pair in names:
            if len(pair) >= 2:
                results.append({"snomed_id": pair[0], "name": pair[1]})
        return results
    except Exception as e:
        logger.error(f"SNOMED lookup error: {e}")
        return []


def lookup_rxnorm(drug_name: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Look up RxNorm codes for a drug name.
    """
    try:
        data = _get(
            f"{NLM_CLINICAL_TABLES_BASE}/rxterms/v3/search",
            params={"terms": drug_name, "maxList": max_results},
        )
        names = data[3] if len(data) > 3 else []
        results = []
        for pair in names:
            if len(pair) >= 2:
                results.append({"rxnorm_id": pair[0], "name": pair[1]})
        return results
    except Exception as e:
        logger.error(f"RxNorm lookup error: {e}")
        return []


def format_icd_results(results: List[Dict]) -> str:
    if not results:
        return "No ICD-10 codes found."
    return "\n".join(f"  {r['code']} — {r['name']}" for r in results)