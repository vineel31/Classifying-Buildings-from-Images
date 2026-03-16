"""
Vision analyzer for building classification using Groq/Llama 4 Scout.

Analyzes Street View images and extracts OCR text, building features,
and classification predictions. Outputs structured JSON to intermediate_results.jsonl
"""

import base64
import json
import re
import time
from typing import Any, Dict, Optional

from groq import Groq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_RETRIES,
    REQUEST_DELAY,
    RETRY_DELAY,
    VALID_LABELS,
)

_client = Groq(api_key=GROQ_API_KEY)

DEFAULT_CONFIDENCE = 0.5
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
IMAGE_ENCODING = "utf-8"
B64_PREFIX = "data:image/jpeg;base64,"
TEMPERATURE = 0.0
MAX_TOKENS = 1024

ANALYSIS_PROMPT = """Analyze this Google Street View image of a property and return a JSON object.

Your task: classify the building type and extract structured evidence.

Use ONLY one of these labels:
- single_family     : detached house / residential home
- apartment_condo   : apartment, condo, townhouse, or multi-unit housing
- commercial        : primarily non-residential / business use
- mixed_use         : residential AND commercial combined
- empty_land        : vacant lot / no clear building present
- unknown           : insufficient visual information to classify

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{
  "ocr_text": {
    "detected": <true or false>,
    "text_found": [<list of any readable text strings>],
    "signage": [<business signs, storefront labels>],
    "business_names": [<any business names visible>]
  },
  "detected_objects": [<list of objects visible: e.g. "house", "car", "tree", "fence", "parking_lot">],
  "visual_cues": {
    "num_entrances": <integer>,
    "has_storefront": <true or false>,
    "has_porch": <true or false>,
    "has_garage": <true or false>,
    "has_balcony": <true or false>,
    "parking_type": <"driveway", "parking_lot", "street_only", "none">,
    "signage_visible": <true or false>,
    "residential_indicators": [<list of residential cues>],
    "commercial_indicators": [<list of commercial cues>]
  },
  "building_features": {
    "building_present": <true or false>,
    "num_stories": <integer or null if unknown>,
    "architectural_style": <e.g. "ranch", "colonial", "craftsman", "commercial_flat", "unknown">,
    "primary_material": <e.g. "brick", "vinyl_siding", "wood", "concrete", "unknown">,
    "building_condition": <"good", "fair", "poor", "unknown">
  },
  "preliminary_classification": <one of the 6 labels above>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": <one sentence explaining why you chose this classification>
}"""


def analyze_image(image_info: Dict[str, Any], location_id: str) -> Dict[str, Any]:
    """Analyze a single image and return structured classification."""
    image_path = image_info["path"]
    filename = image_info["filename"]
    year = image_info["year"]

    result = {
        "location_id": location_id,
        "image_file": filename,
        "year": year,
        "analysis": None,
        "error": None,
    }

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode(IMAGE_ENCODING)
    except Exception as e:
        result["error"] = f"Failed to open image: {e}"
        result["analysis"] = _fallback_analysis("unknown", 0.0, f"Image load error: {e}")
        return result

    raw_text = _call_groq_with_retry(image_b64, filename, result)
    if raw_text:
        result["analysis"] = _parse_response(raw_text, filename)

    return result


def _call_groq_with_retry(
    image_b64: str, filename: str, result: Dict[str, Any]
) -> Optional[str]:
    """Call Groq API with retry logic on rate limits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{B64_PREFIX}{image_b64}"},
                            },
                            {
                                "type": "text",
                                "text": ANALYSIS_PROMPT,
                            },
                        ],
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_text = response.choices[0].message.content.strip()
            time.sleep(REQUEST_DELAY)
            return raw_text
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = _parse_retry_delay(str(e))
                print(f"  [Retry {attempt}/{MAX_RETRIES}] {filename} — waiting {wait}s...")
                time.sleep(wait)
            else:
                result["error"] = f"API error after {MAX_RETRIES} attempts: {e}"
                result["analysis"] = _fallback_analysis("unknown", 0.0, str(e))

    return None


def _parse_retry_delay(error_str: str) -> int:
    """Extract retry delay from error message, or use default."""
    match = re.search(r"retry.after[:\s]+(\d+)", error_str, re.IGNORECASE)
    return int(match.group(1)) + 2 if match else RETRY_DELAY


def _parse_response(raw_text: str, filename: str) -> Dict[str, Any]:
    """Parse and validate model response JSON."""
    if not raw_text:
        return _fallback_analysis("unknown", 0.0, "Empty response from model")

    cleaned = raw_text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(line for line in lines if not line.startswith("```")).strip()

    try:
        parsed = json.loads(cleaned)

        label = parsed.get("preliminary_classification", "unknown")
        parsed["preliminary_classification"] = (
            label if label in VALID_LABELS else "unknown"
        )

        confidence = float(parsed.get("confidence", DEFAULT_CONFIDENCE))
        parsed["confidence"] = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

        return parsed

    except json.JSONDecodeError as e:
        print(f"  [Warning] JSON parse failed for {filename}: {e}")
        return _fallback_analysis(
            "unknown", 0.0, f"JSON parse error: {e}. Raw: {raw_text[:200]}"
        )


def _fallback_analysis(label: str, confidence: float, reason: str) -> Dict[str, Any]:
    """Generate fallback analysis structure on API failure."""
    return {
        "ocr_text": {
            "detected": False,
            "text_found": [],
            "signage": [],
            "business_names": [],
        },
        "detected_objects": [],
        "visual_cues": {
            "num_entrances": 0,
            "has_storefront": False,
            "has_porch": False,
            "has_garage": False,
            "has_balcony": False,
            "parking_type": "none",
            "signage_visible": False,
            "residential_indicators": [],
            "commercial_indicators": [],
        },
        "building_features": {
            "building_present": False,
            "num_stories": None,
            "architectural_style": "unknown",
            "primary_material": "unknown",
            "building_condition": "unknown",
        },
        "preliminary_classification": label,
        "confidence": confidence,
        "reasoning": reason,
    }
