"""
tools/multimodal_parser.py

Unified input parser that normalises ANY input type into a plain-text
string the orchestrator and agents can work with.

Supported input types:
  - Plain text / clinical question  →  returned as-is
  - PDF file (lab report, discharge summary, prescription)  →  extracted text
  - Image file (X-ray, scan, lab result photo)  →  Gemini Vision OCR + description
  - Base64-encoded image string  →  same as image file

"""

import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ParsedInput:
    """
    The normalised output of parse_input().
    Always contains a text field the orchestrator can use directly.
    """
    text: str                        
    input_type: str                  
    image_path: Optional[str] = None 
    raw_extracted: str = ""          
    metadata: dict = field(default_factory=dict)


# ─── PDF extraction ───────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF using pdfplumber.
    Falls back page-by-page and skips empty pages.
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append(f"[Page {i + 1}]\n{page_text.strip()}")
    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}")
        return ""
    return "\n\n".join(pages)


# ─── Image handling ───────────────────────────────────────────────────────────

def _save_image_to_temp(image_data) -> str:
    """
    Save image bytes / PIL Image / path to a temp file.
    Returns the temp file path.
    """
    suffix = ".png"
    if isinstance(image_data, (str, Path)):
        return str(image_data)          # already a path
    if isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        img = image_data
    else:
        raise ValueError(f"Unsupported image type: {type(image_data)}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


def _describe_image_with_gemini(image_path: str, clinical_context: str = "") -> str:
    """
    Use Gemini Vision to extract text AND describe a medical image.
    Combines OCR (for lab values etc.) with visual description.
    Falls back gracefully if Gemini key is missing.
    """
    try:
        from tools.vision_tool import extract_text_from_image, analyse_medical_image
        from config.settings import GEMINI_API_KEY

        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY not set — skipping vision analysis.")
            return "[Image uploaded — Gemini Vision not configured. Please add GEMINI_API_KEY to .env]"

        # First: extract any text visible in the image (lab values, prescriptions, etc.)
        ocr_text = extract_text_from_image(image_path)

        # Second: full clinical image analysis
        analysis = analyse_medical_image(image_path, clinical_context=clinical_context)

        parts = []
        if ocr_text and ocr_text.strip():
            parts.append(f"EXTRACTED TEXT FROM IMAGE:\n{ocr_text.strip()}")
        if analysis and analysis.strip():
            parts.append(f"VISUAL ANALYSIS:\n{analysis.strip()}")

        return "\n\n".join(parts) if parts else "[No content extracted from image]"

    except Exception as e:
        logger.error(f"Gemini image description failed: {e}")
        return f"[Image analysis error: {e}]"


# ─── Main parser ──────────────────────────────────────────────────────────────

def parse_input(
    text: Optional[str] = None,
    pdf_path: Optional[str] = None,
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    clinical_context: str = "",
) -> ParsedInput:
    """
    Parse any input type into a normalised ParsedInput.

    Priority order: image_bytes > image_path > pdf_path > text

    Args:
        text:             Raw clinical question / text.
        pdf_path:         Path to a PDF (lab report, discharge summary, etc.).
        image_path:       Path to a medical image file.
        image_bytes:      Raw bytes of an image.
        clinical_context: Optional patient background to include.

    Returns:
        ParsedInput with .text ready to pass to run_pipeline().
    """

    # ── IMAGE (bytes) ─────────────────────────────────────────────────────
    if image_bytes is not None:
        logger.info("[MultimodalParser] Input type: image (bytes)")
        tmp_path = _save_image_to_temp(image_bytes)
        description = _describe_image_with_gemini(tmp_path, clinical_context)
        combined = _combine_with_context(description, clinical_context)
        return ParsedInput(
            text=combined,
            input_type="image",
            image_path=tmp_path,
            raw_extracted=description,
            metadata={"source": "image_bytes"},
        )

    # ── IMAGE (file path) ─────────────────────────────────────────────────
    if image_path is not None:
        logger.info(f"[MultimodalParser] Input type: image (file: {image_path})")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        description = _describe_image_with_gemini(image_path, clinical_context)
        combined = _combine_with_context(description, clinical_context)
        return ParsedInput(
            text=combined,
            input_type="image",
            image_path=image_path,
            raw_extracted=description,
            metadata={"source": image_path},
        )

    # ── PDF ───────────────────────────────────────────────────────────────
    if pdf_path is not None:
        logger.info(f"[MultimodalParser] Input type: PDF ({pdf_path})")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        raw_text = _extract_pdf_text(pdf_path)
        if not raw_text.strip():
            logger.warning("PDF extraction returned empty text.")
            raw_text = "[PDF text extraction returned no content]"
        combined = _combine_with_context(raw_text, clinical_context)
        return ParsedInput(
            text=combined,
            input_type="pdf",
            raw_extracted=raw_text,
            metadata={"source": pdf_path, "pages_extracted": raw_text.count("[Page ")},
        )

    # ── PLAIN TEXT ────────────────────────────────────────────────────────
    if text is not None:
        logger.info("[MultimodalParser] Input type: text")
        combined = _combine_with_context(text.strip(), clinical_context)
        return ParsedInput(
            text=combined,
            input_type="text",
            raw_extracted=text,
            metadata={"source": "text_input"},
        )

    raise ValueError("parse_input() requires at least one of: text, pdf_path, image_path, image_bytes")


def _combine_with_context(main_text: str, context: str) -> str:
    """Prepend patient context if provided."""
    if context and context.strip():
        return f"PATIENT CONTEXT: {context.strip()}\n\n{main_text}"
    return main_text