"""
ingest.py — Ingest medical documents into the ChromaDB vector store.

Run once before starting the app:
    python ingest.py

Supports: PDF files, plain text files.
Place documents in data/sample_docs/ or pass a custom directory.
"""
import os
import sys
import uuid
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber

from rag.vectorstore import add_documents, count_documents
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
    return "\n".join(text_parts)


def ingest_file(file_path: str) -> int:
    """
    Ingest a single file into ChromaDB.
    Returns number of chunks added.
    """
    path = Path(file_path)
    logger.info(f"Ingesting: {path.name}")

    # Extract text
    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(str(path))
    elif path.suffix.lower() in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        logger.warning(f"Unsupported file type: {path.suffix}. Skipping.")
        return 0

    if not text.strip():
        logger.warning(f"No text extracted from {path.name}. Skipping.")
        return 0

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        return 0

    # Build metadata and IDs
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append({
            "source": path.name,
            "source_type": "medical_document",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "file_path": str(path),
        })
        ids.append(f"{path.stem}_{i}_{uuid.uuid4().hex[:8]}")

    add_documents(texts, metadatas, ids)
    logger.info(f"  → Added {len(chunks)} chunks from {path.name}")
    return len(chunks)


def ingest_directory(directory: str) -> int:
    """Ingest all supported files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return 0

    supported = [".pdf", ".txt", ".md"]
    files = [f for f in dir_path.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.warning(f"No supported files found in {directory}")
        logger.info("Tip: Add PDF or text files to data/sample_docs/")
        return 0

    total = 0
    for f in files:
        total += ingest_file(str(f))

    return total


def ingest_sample_data():
    """
    Ingest built-in sample medical text data if no documents are found.
    Useful for testing without real medical PDFs.
    """
    sample_docs = [
        {
            "filename": "hypertension_guidelines.txt",
            "content": """Hypertension Management Guidelines

Blood pressure classification:
- Normal: <120/80 mmHg
- Elevated: 120-129/<80 mmHg  
- Stage 1 HTN: 130-139/80-89 mmHg
- Stage 2 HTN: ≥140/90 mmHg
- Hypertensive crisis: >180/120 mmHg

First-line antihypertensive agents:
1. Thiazide diuretics (e.g., hydrochlorothiazide, chlorthalidone)
2. ACE inhibitors (e.g., lisinopril, enalapril) — preferred in diabetes/CKD
3. ARBs (e.g., losartan, valsartan) — alternative to ACE inhibitors
4. Calcium channel blockers (e.g., amlodipine, nifedipine)

Compelling indications for specific agents:
- Heart failure: ACE inhibitor/ARB + beta-blocker + aldosterone antagonist
- Post-MI: Beta-blocker + ACE inhibitor
- Diabetes: ACE inhibitor or ARB
- CKD: ACE inhibitor or ARB
- Recurrent stroke prevention: ACE inhibitor + thiazide diuretic

Lifestyle modifications:
- DASH diet, weight reduction (target BMI <25)
- Sodium restriction <2.3g/day
- Regular aerobic exercise 150min/week
- Alcohol limitation, smoking cessation

Target BP goals:
- General adults: <130/80 mmHg
- Age ≥65: <130/80 mmHg
- CKD with proteinuria: <130/80 mmHg
""",
        },
        {
            "filename": "diabetes_type2_management.txt",
            "content": """Type 2 Diabetes Mellitus — Clinical Management

Diagnostic criteria:
- Fasting plasma glucose ≥126 mg/dL (7.0 mmol/L)
- 2-hour plasma glucose ≥200 mg/dL during OGTT
- HbA1c ≥6.5%
- Random plasma glucose ≥200 mg/dL with symptoms

HbA1c targets:
- General: <7% (53 mmol/mol)
- Older adults with frailty: <8%
- Pregnant: <6%

First-line: Metformin 500mg BD, titrate to 1000mg BD
- Contraindications: eGFR <30, hepatic impairment, iodinated contrast (hold 48h)

Second-line options:
1. SGLT2 inhibitors (empagliflozin, dapagliflozin) — preferred in heart failure, CKD, ASCVD
2. GLP-1 RAs (semaglutide, liraglutide) — preferred in obesity, ASCVD
3. DPP-4 inhibitors (sitagliptin, saxagliptin) — weight neutral, low hypoglycaemia
4. Sulfonylureas (glipizide, glyburide) — low cost, risk of hypoglycaemia
5. Insulin — when HbA1c >10% or symptomatic hyperglycaemia

Monitoring:
- HbA1c: every 3 months until stable, then every 6 months
- eGFR + urine ACR: annually
- Foot exam: annually
- Retinal screening: annually
- BP target: <130/80 mmHg
- Lipids: statin therapy for most adults >40 years

Hypoglycaemia management:
- Mild: 15g fast-acting carbohydrate (glucose tablets, juice)
- Severe: glucagon 1mg IM/SC or IV dextrose 25g
""",
        },
        {
            "filename": "chest_pain_differential.txt",
            "content": """Chest Pain — Differential Diagnosis and Management

Life-threatening causes (must exclude first):
1. Acute Coronary Syndrome (ACS)
   - STEMI: ST elevation, troponin rise, requires immediate reperfusion
   - NSTEMI: No ST elevation, troponin rise
   - Unstable angina: No troponin rise
   - ECG, troponin (at 0h and 3h), aspirin 300mg, nitrates

2. Pulmonary Embolism (PE)
   - Pleuritic chest pain, dyspnoea, tachycardia
   - Risk: Wells score, D-dimer, CTPA
   - Treatment: anticoagulation (LMWH, DOAC)

3. Aortic Dissection
   - Tearing/ripping pain radiating to back, unequal BP in arms
   - CT aortography urgently required
   - NO anticoagulation, surgical emergency

4. Tension Pneumothorax
   - Sudden onset, deviated trachea, absent breath sounds
   - Immediate needle decompression

Common non-cardiac causes:
- Musculoskeletal: reproducible with palpation, positional
- GERD/oesophageal: burning, relieved by antacids
- Costochondritis: Tietze syndrome, tender costochondral junctions
- Anxiety/panic: hyperventilation, situational

Red flags requiring immediate assessment:
- Diaphoresis, radiation to arm/jaw, nausea/vomiting with chest pain
- Haemodynamic instability, oxygen saturation <95%
- Pleuritic chest pain with dyspnoea and risk factors for PE
""",
        },
    ]

    sample_dir = Path("data/sample_docs")
    sample_dir.mkdir(parents=True, exist_ok=True)

    for doc in sample_docs:
        filepath = sample_dir / doc["filename"]
        if not filepath.exists():
            filepath.write_text(doc["content"], encoding="utf-8")
            logger.info(f"Created sample document: {doc['filename']}")


def main():
    parser = argparse.ArgumentParser(description="Ingest medical documents into ChromaDB")
    parser.add_argument(
        "--dir",
        default="data/sample_docs",
        help="Directory containing documents to ingest (default: data/sample_docs)",
    )
    parser.add_argument(
        "--file",
        help="Ingest a single file",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate and ingest sample medical documents",
    )
    args = parser.parse_args()

    os.makedirs("data/chroma_db", exist_ok=True)
    os.makedirs("data/sample_docs", exist_ok=True)

    logger.info(f"Documents already in store: {count_documents()}")

    if args.sample or not Path(args.dir).exists():
        logger.info("Generating sample medical documents...")
        ingest_sample_data()

    if args.file:
        n = ingest_file(args.file)
    else:
        n = ingest_directory(args.dir)

    logger.info(f"\n Ingestion complete. Total new chunks added: {n}")
    logger.info(f"   Total documents in store: {count_documents()}")


if __name__ == "__main__":
    main()