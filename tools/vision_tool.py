"""
tools/vision_tool.py — Gemini Vision for medical image analysis.
Free tier: 15 requests/min, 1M tokens/day.
Supports X-rays, skin lesions, lab result images, pathology slides.
"""
import base64
import logging
from pathlib import Path
from typing import Union
import google.generativeai as genai
from PIL import Image
from config.settings import GEMINI_API_KEY, VISION_MODEL

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

MEDICAL_IMAGE_SYSTEM_PROMPT = """You are a medical image analysis assistant with expertise in 
radiology, pathology, and clinical imaging. Analyse the provided image carefully and provide:

1. IMAGE TYPE: Identify the type of medical image (X-ray, MRI, CT, histology, skin lesion, etc.)
2. KEY FINDINGS: Describe visible anatomical structures and any abnormalities.
3. CLINICAL OBSERVATIONS: Note any findings of potential clinical significance.
4. DIFFERENTIAL CONSIDERATIONS: Suggest possible interpretations based on visible findings.
5. LIMITATIONS: Note what cannot be determined from this image alone.

IMPORTANT: This is an AI-assisted analysis for educational/support purposes only. 
All findings must be verified by a qualified medical professional.
"""


def analyse_medical_image(
    image_source: Union[str, bytes, Path],
    clinical_context: str = "",
) -> str:
    """
    Analyse a medical image using Gemini Vision.

    Args:
        image_source: File path (str/Path) or raw bytes of the image.
        clinical_context: Optional patient context to guide analysis.

    Returns:
        Analysis text from Gemini.
    """
    try:
        model = genai.GenerativeModel(VISION_MODEL)

        # Load image
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            import io
            img = Image.open(io.BytesIO(image_source))
        else:
            return "Error: unsupported image source type."

        prompt_parts = [MEDICAL_IMAGE_SYSTEM_PROMPT]
        if clinical_context:
            prompt_parts.append(f"\nClinical context: {clinical_context}")
        prompt_parts.append(img)

        response = model.generate_content(prompt_parts)
        return response.text

    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        return f"Image analysis failed: {str(e)}"


def analyse_image_from_base64(b64_string: str, clinical_context: str = "") -> str:
    """Analyse a base64-encoded image (useful for API endpoints)."""
    image_bytes = base64.b64decode(b64_string)
    return analyse_medical_image(image_bytes, clinical_context)


def extract_text_from_image(image_source: Union[str, bytes, Path]) -> str:
    """
    Extract text from a medical document image (lab reports, prescriptions).
    Uses Gemini for high accuracy on medical terminology.
    """
    try:
        model = genai.GenerativeModel(VISION_MODEL)
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            import io
            img = Image.open(io.BytesIO(image_source))
        else:
            return ""

        response = model.generate_content([
            "Extract all text from this medical document exactly as it appears. "
            "Preserve all values, units, reference ranges, and test names.",
            img,
        ])
        return response.text
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""