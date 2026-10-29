"""Approved medical symptom vocabulary, synonym maps, and canonical normalization.

Derived from the clinical feature set and synonyms in ml_pipeline/src/nlp_rules.py,
providing backend-side validation without hardcoding assumptions or calling unverified sources.
"""

from typing import Dict, List, Set, Optional

# Canonical symptom vocabulary (exact feature tokens accepted by ML models)
APPROVED_SYMPTOM_KEYS: List[str] = [
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

APPROVED_SYMPTOM_SET: Set[str] = set(APPROVED_SYMPTOM_KEYS)

# Canonical human-readable names for UI presentation
SYMPTOM_DISPLAY_NAMES: Dict[str, str] = {
    key: key.replace("_", " ").title()
    for key in APPROVED_SYMPTOM_KEYS
}
# Specific display overrides where underscores or grammar need special formatting
SYMPTOM_DISPLAY_NAMES.update({
    "spotting_ urination": "Spotting Urination",
    "foul_smell_of urine": "Foul Smelling Urine",
    "dischromic _patches": "Dischromic Patches",
    "fluid_overload.1": "Fluid Overload",
    "toxic_look_(typhos)": "Toxic Appearance",
    "cold_hands_and_feets": "Cold Hands and Feet",
    "swollen_extremeties": "Swollen Extremities",
    "belly_pain": "Abdominal Pain",
    "spinning_movements": "Vertigo / Spinning Sensation",
    "continuous_feel_of_urine": "Urinary Urgency",
})

# Known clinical & colloquial aliases mapping to canonical keys
APPROVED_SYNONYM_MAP: Dict[str, str] = {
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
    "itching": "itching",
    "itchy": "itching",
    "pruritus": "itching",
    "skin rash": "skin_rash",
    "rash": "skin_rash",
    "eruptions": "nodal_skin_eruptions",
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
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "shortness of breath": "breathlessness",
    "breathlessness": "breathlessness",
    "difficulty breathing": "breathlessness",
    "sweating": "sweating",
    "night sweats": "sweating",
    "muscle weakness": "muscle_weakness",
    "swollen legs": "swollen_legs",
    "palpitations": "palpitations",
    "heart racing": "fast_heart_rate",
    "fast heart rate": "fast_heart_rate",
}


def normalize_symptom_key(term: str) -> Optional[str]:
    """Check if a term maps to an approved canonical symptom key."""
    if not term:
        return None
    cleaned = term.strip().lower()
    if cleaned in APPROVED_SYMPTOM_SET:
        return cleaned
    cleaned_spaces = cleaned.replace("_", " ")
    if cleaned_spaces in APPROVED_SYNONYM_MAP:
        return APPROVED_SYNONYM_MAP[cleaned_spaces]
    if cleaned in APPROVED_SYNONYM_MAP:
        return APPROVED_SYNONYM_MAP[cleaned]
    return None
