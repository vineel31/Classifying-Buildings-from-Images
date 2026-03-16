import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"

VALID_LABELS = [
    "single_family",
    "apartment_condo",
    "commercial",
    "mixed_use",
    "empty_land",
    "unknown",
]

CONFIDENCE_THRESHOLD = 0.60

YEAR_WEIGHT = {
    range(2000, 2013): 0.5,
    range(2013, 2018): 0.75,
    range(2018, 2030): 1.0,
}

def get_year_weight(year: int) -> float:
    for year_range, weight in YEAR_WEIGHT.items():
        if year in year_range:
            return weight
    return 1.0

MAX_RETRIES   = 3
RETRY_DELAY   = 20
REQUEST_DELAY = 5
