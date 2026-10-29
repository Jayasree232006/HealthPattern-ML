"""Symptom vocabulary, synonym lookup, canonical normalization, and negation handling.

Provides reproducible text-to-symptom mapping with explicit negation detection
and confidence weighting.
"""

import re
from typing import Dict, List, Set, Tuple, Optional

# Kaggle dataset 132 features (exact column order in raw Testing.csv)
KAGGLE_SYMPTOM_FEATURES: List[str] = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_ urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising",
    "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes",
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness",
    "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements",
    "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
    "loss_of_smell", "bladder_discomfort", "foul_smell_of urine",
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching",
    "toxic_look_(typhos)", "depression", "irritability", "muscle_pain",
    "altered_sensorium", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic _patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
    "rusty_sputum", "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma",
    "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload.1", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking",
    "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
    "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

# Clinical / colloquial synonyms mapping to canonical symptom keys
SYMPTOM_SYNONYMS: Dict[str, str] = {
    # Headache / Migraine
    "headache": "headache",
    "head pain": "headache",
    "head throbbing": "headache",
    "throbbing head": "headache",
    "migraine": "headache",
    "pain behind eyes": "pain_behind_the_eyes",
    "pain behind the eyes": "pain_behind_the_eyes",
    "eye pain": "pain_behind_the_eyes",
    "vision disturbance": "visual_disturbances",
    "visual disturbances": "visual_disturbances",
    "blurred vision": "blurred_and_distorted_vision",
    "distorted vision": "blurred_and_distorted_vision",
    "aura": "visual_disturbances",

    # Common cold / Respiratory / Allergy
    "continuous sneezing": "continuous_sneezing",
    "sneezing": "continuous_sneezing",
    "sneeze": "continuous_sneezing",
    "shivering": "shivering",
    "chills": "chills",
    "feeling cold": "chills",
    "cough": "cough",
    "coughing": "cough",
    "persistent cough": "cough",
    "phlegm": "phlegm",
    "mucus": "phlegm",
    "sputum": "phlegm",
    "throat irritation": "throat_irritation",
    "sore throat": "throat_irritation",
    "scratchy throat": "throat_irritation",
    "runny nose": "runny_nose",
    "rhinorrhea": "runny_nose",
    "nasal congestion": "congestion",
    "congestion": "congestion",
    "stuffy nose": "congestion",
    "blocked nose": "congestion",
    "sinus pressure": "sinus_pressure",
    "sinus pain": "sinus_pressure",
    "loss of smell": "loss_of_smell",
    "watering from eyes": "watering_from_eyes",
    "watery eyes": "watering_from_eyes",
    "tearing eyes": "watering_from_eyes",
    "redness of eyes": "redness_of_eyes",
    "red eyes": "redness_of_eyes",

    # Allergy / Skin
    "itching": "itching",
    "itchy": "itching",
    "pruritus": "itching",
    "skin rash": "skin_rash",
    "rash": "skin_rash",
    "eruptions": "nodal_skin_eruptions",

    # GERD / Gastrointestinal
    "acidity": "acidity",
    "acid reflux": "acidity",
    "heartburn": "acidity",
    "chest burning": "acidity",
    "stomach pain": "stomach_pain",
    "abdominal pain": "abdominal_pain",
    "belly pain": "belly_pain",
    "cramps": "cramps",
    "stomach cramps": "cramps",
    "indigestion": "indigestion",
    "dyspepsia": "indigestion",
    "ulcers on tongue": "ulcers_on_tongue",
    "tongue ulcers": "ulcers_on_tongue",
    "mouth sores": "ulcers_on_tongue",
    "vomiting": "vomiting",
    "threw up": "vomiting",
    "nausea": "nausea",
    "feeling sick": "nausea",
    "queasy": "nausea",
    "chest pain": "chest_pain",

    # UTI / Genitourinary
    "burning micturition": "burning_micturition",
    "burning urination": "burning_micturition",
    "burning when urinating": "burning_micturition",
    "painful urination": "burning_micturition",
    "dysuria": "burning_micturition",
    "bladder discomfort": "bladder_discomfort",
    "bladder pain": "bladder_discomfort",
    "foul smell of urine": "foul_smell_of urine",
    "smelly urine": "foul_smell_of urine",
    "continuous feel of urine": "continuous_feel_of_urine",
    "urinary urgency": "continuous_feel_of_urine",
    "frequent urination": "polyuria",
    "polyuria": "polyuria",
    "spotting urination": "spotting_ urination",
    "dark urine": "dark_urine",

    # Systemic / Constitutional
    "fatigue": "fatigue",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "mild fever": "mild_fever",
    "low grade fever": "mild_fever",
    "high fever": "high_fever",
    "fever": "high_fever",
    "malaise": "malaise",
    "dizziness": "dizziness",
    "lightheadedness": "dizziness",
    "vertigo": "spinning_movements",
    "spinning sensations": "spinning_movements",
    "joint pain": "joint_pain",
    "muscle pain": "muscle_pain",
    "stiff neck": "stiff_neck",
    "neck pain": "neck_pain",
}

# Negation prefixes and trigger phrases
NEGATION_TRIGGERS: List[str] = [
    "no", "without", "deny", "denies", "denied", "ruled out",
    "never had", "no sign of", "free of", "negative for",
    "haven't had", "hasn't had", "didn't have", "did not have",
    "don't have", "doesn't have", "do not have", "does not have",
    "not have", "not experiencing", "not feel", "cannot feel",
    "neither", "nor"
]

HISTORICAL_CUES: List[str] = [
    "history of", "historical", "previously", "in the past", "last year",
    "last month", "last week", "years ago", "months ago", "weeks ago",
    "used to", "no longer", "resolved"
]

UNCERTAINTY_CUES: List[str] = [
    "might", "may", "maybe", "possibly", "possible", "perhaps", "could",
    "unsure", "uncertain", "not sure", "not sure if", "not sure whether"
]

AFFIRMATION_RESETS: List[str] = [
    "have", "has", "am experiencing", "are experiencing", "experience",
    "experiencing", "report", "reports", "feel", "feels", "notice", "notices"
]

POST_NEGATION_CUES: List[str] = [
    "ruled out", "not present", "absent", "denied"
]

CLAUSE_DELIMITERS: re.Pattern = re.compile(
    r"[;:.!?]|\b(?:but|however|except|although|whereas|though)\b",
    re.IGNORECASE,
)


def normalize_text_token(token: str) -> str:
    """Lowercase and remove non-alphanumeric punctuation except underscore and hyphen."""
    token = token.lower().strip()
    token = re.sub(r"[^\w\s-]", " ", token)
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def _phrase_tokens(phrases: List[str]) -> List[List[str]]:
    return [normalize_text_token(phrase).split() for phrase in phrases]


def _contains_phrase(words: List[str], phrase: List[str]) -> bool:
    width = len(phrase)
    return any(words[index:index + width] == phrase for index in range(len(words) - width + 1))


def _match_symptoms(words: List[str], allowed_features: Set[str]) -> List[Tuple[int, int, str]]:
    candidates = sorted(
        (
            (phrase.split(), canonical)
            for phrase, canonical in SYMPTOM_SYNONYMS.items()
            if canonical in allowed_features
        ),
        key=lambda item: (len(item[0]), len(" ".join(item[0]))),
        reverse=True,
    )
    matches: List[Tuple[int, int, str]] = []
    index = 0
    while index < len(words):
        match = next(
            (
                (index, index + len(phrase), canonical)
                for phrase, canonical in candidates
                if words[index:index + len(phrase)] == phrase
            ),
            None,
        )
        if match is None:
            index += 1
            continue
        matches.append(match)
        index = match[1]
    return matches


def _has_negation(words: List[str], start: int, end: int) -> bool:
    negation_phrases = _phrase_tokens(NEGATION_TRIGGERS)
    reset_phrases = _phrase_tokens(AFFIRMATION_RESETS)
    preceding = words[:start]
    cue_positions = [
        index + len(phrase)
        for phrase in negation_phrases
        for index in range(len(preceding) - len(phrase) + 1)
        if preceding[index:index + len(phrase)] == phrase
    ]
    reset_positions = [
        index + len(phrase)
        for phrase in reset_phrases
        for index in range(len(preceding) - len(phrase) + 1)
        if preceding[index:index + len(phrase)] == phrase
    ]
    if cue_positions and max(cue_positions) > (max(reset_positions) if reset_positions else -1):
        return True

    following = words[end:min(len(words), end + 4)]
    return any(_contains_phrase(following, phrase) for phrase in _phrase_tokens(POST_NEGATION_CUES))


def _has_context_cue(words: List[str], start: int, end: int, cues: List[str]) -> bool:
    context = words[max(0, start - 6):min(len(words), end + 6)]
    return any(_contains_phrase(context, phrase) for phrase in _phrase_tokens(cues))


def _has_historical_cue(
    words: List[str],
    matches: List[Tuple[int, int, str]],
    match_index: int,
) -> bool:
    start, end, _ = matches[match_index]
    previous_end = matches[match_index - 1][1] if match_index > 0 else max(0, start - 6)
    next_start = matches[match_index + 1][0] if match_index + 1 < len(matches) else min(len(words), end + 6)
    context = words[previous_end:next_start]
    return any(_contains_phrase(context, phrase) for phrase in _phrase_tokens(HISTORICAL_CUES))


def extract_symptoms_with_context(
    text: str,
    feature_set: Optional[Set[str]] = None
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """Return current affirmed, denied, historical, and uncertain symptoms.

    Returns:
        (affirmed, negated, historical, uncertain)

    Commas and conjunctions are kept within a clause so a leading negation
    applies to a coordinated list. Contrast terms and sentence boundaries end
    its scope; an explicit affirmation or a new negation cue updates polarity.
    """
    if not text or not isinstance(text, str):
        return set(), set(), set(), set()

    allowed_features = feature_set if feature_set is not None else set(KAGGLE_SYMPTOM_FEATURES)
    affirmed: Set[str] = set()
    negated: Set[str] = set()
    historical: Set[str] = set()
    uncertain: Set[str] = set()

    clauses = CLAUSE_DELIMITERS.split(text)
    for clause in clauses:
        clause_clean = normalize_text_token(clause)
        if not clause_clean:
            continue
        words = clause_clean.split()
        matches = _match_symptoms(words, allowed_features)
        for match_index, (start, end, canonical) in enumerate(matches):
            is_historical = _has_historical_cue(words, matches, match_index)
            is_uncertain = _has_context_cue(words, start, end, UNCERTAINTY_CUES)
            if is_historical:
                historical.add(canonical)
            if is_uncertain:
                uncertain.add(canonical)
            if is_historical or is_uncertain:
                continue
            if _has_negation(words, start, end):
                negated.add(canonical)
            else:
                affirmed.add(canonical)

    return affirmed, negated, historical, uncertain


def extract_symptoms_with_negation(
    text: str,
    feature_set: Optional[Set[str]] = None
) -> Tuple[Set[str], Set[str]]:
    """Extract current affirmed and explicitly denied symptoms from free text."""
    affirmed, negated, _, _ = extract_symptoms_with_context(text, feature_set)
    return affirmed - negated, negated


def vector_from_symptoms(
    affirmed_symptoms: Set[str],
    feature_list: Optional[List[str]] = None
) -> List[int]:
    """Build binary feature vector matching specified feature order."""
    features = feature_list or KAGGLE_SYMPTOM_FEATURES
    return [1 if feat in affirmed_symptoms else 0 for feat in features]
