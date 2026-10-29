"""Target condition definitions and cross-dataset label mapping.

Shared 5 common conditions:
- Common cold
- Allergy
- Migraine
- GERD
- Urinary tract infection
"""

from typing import Dict, List, Optional

# Canonical condition keys used by the disease pattern detector
CANONICAL_COMMON_COLD = "common_cold"
CANONICAL_ALLERGY = "allergy"
CANONICAL_MIGRAINE = "migraine"
CANONICAL_GERD = "gerd"
CANONICAL_UTI = "urinary_tract_infection"

SHARED_5_CLASSES = [
    CANONICAL_ALLERGY,
    CANONICAL_COMMON_COLD,
    CANONICAL_GERD,
    CANONICAL_MIGRAINE,
    CANONICAL_UTI,
]

# Display labels for clinical / user presentation
DISPLAY_LABELS: Dict[str, str] = {
    CANONICAL_ALLERGY: "Allergy",
    CANONICAL_COMMON_COLD: "Common Cold",
    CANONICAL_GERD: "Gastroesophageal Reflux Disease (GERD)",
    CANONICAL_MIGRAINE: "Migraine",
    CANONICAL_UTI: "Urinary Tract Infection (UTI)",
}

# Raw label mappings for Dataset 1: Kaggle (kaushil268/disease-prediction-using-machine-learning)
# Target column: 'prognosis'
KAGGLE_RAW_TARGET_MAP: Dict[str, str] = {
    "Allergy": CANONICAL_ALLERGY,
    "Common Cold": CANONICAL_COMMON_COLD,
    "GERD": CANONICAL_GERD,
    "Migraine": CANONICAL_MIGRAINE,
    "Urinary tract infection": CANONICAL_UTI,
}

# Raw label mappings for Dataset 2: Hugging Face Gretel (gretelai/symptom_to_diagnosis)
# Target column: 'output_text'
GRETEL_RAW_TARGET_MAP: Dict[str, str] = {
    "allergy": CANONICAL_ALLERGY,
    "common cold": CANONICAL_COMMON_COLD,
    "gastroesophageal reflux disease": CANONICAL_GERD,
    "migraine": CANONICAL_MIGRAINE,
    "urinary tract infection": CANONICAL_UTI,
}

def normalize_kaggle_label(raw_label: str) -> Optional[str]:
    """Map raw Kaggle prognosis label after whitespace trimming."""
    if not isinstance(raw_label, str):
        return None
    cleaned = raw_label.strip()
    return KAGGLE_RAW_TARGET_MAP.get(cleaned)

def normalize_gretel_label(raw_label: str) -> Optional[str]:
    """Map raw Gretel output_text label after trimming and lowercase."""
    if not isinstance(raw_label, str):
        return None
    cleaned = raw_label.strip().lower()
    return GRETEL_RAW_TARGET_MAP.get(cleaned)
