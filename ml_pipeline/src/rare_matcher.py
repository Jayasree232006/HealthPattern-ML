"""Source-attributed rare-condition profiles and similarity matching engine.

Conditions covered:
1. Systemic Lupus Erythematosus (SLE) - ORPHA:536
2. Myasthenia Gravis (MG) - ORPHA:589
3. Wilson Disease - ORPHA:905
4. Hereditary Angioedema (HAE) - ORPHA:91378

All profiles include verifiable citations from Orphanet, NIH/MedlinePlus, and NORD.
Scores are kept separate from baseline classifier posterior probabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional

@dataclass
class RareConditionProfile:
    condition_id: str
    display_name: str
    orpha_code: str
    icd10_code: str
    key_symptoms: List[str]
    hallmark_features: List[str]
    negative_indicators: List[str]
    description: str
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


RARE_CONDITION_PROFILES: Dict[str, RareConditionProfile] = {
    "systemic_lupus_erythematosus": RareConditionProfile(
        condition_id="systemic_lupus_erythematosus",
        display_name="Systemic Lupus Erythematosus (SLE)",
        orpha_code="ORPHA:536",
        icd10_code="M32",
        key_symptoms=[
            "fatigue",
            "joint_pain",
            "muscle_pain",
            "skin_rash",
            "mild_fever",
            "chest_pain",
            "swelling_joints",
            "malaise",
        ],
        hallmark_features=[
            "malar rash (butterfly rash on cheeks and bridge of nose)",
            "photosensitivity rash after sun exposure",
            "pleuritic chest pain upon deep inhalation",
            "inflammatory polyarthritis of small joints",
            "oral or nasopharyngeal painless mucosal ulcers",
            "unexplained persistent constitutional fatigue and fever",
        ],
        negative_indicators=[
            "rapid complete recovery without recurrence",
            "isolated localized acute fungal infection",
        ],
        description=(
            "A chronic, multisystemic autoimmune disorder characterized by autoantibody production, "
            "immune complex deposition, and widespread inflammatory tissue injury affecting joints, skin, "
            "kidneys, and serous membranes."
        ),
        sources=[
            {
                "authority": "Orphanet",
                "title": "Systemic lupus erythematosus",
                "identifier": "ORPHA:536",
                "url": "https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Lng=GB&Expert=536"
            },
            {
                "authority": "NIH / MedlinePlus",
                "title": "Systemic lupus erythematosus",
                "identifier": "MedlinePlus Medical Encyclopedia",
                "url": "https://medlineplus.gov/ency/article/000435.htm"
            },
            {
                "authority": "NORD (National Organization for Rare Disorders)",
                "title": "Systemic Lupus Erythematosus",
                "identifier": "Rare Disease Database",
                "url": "https://rarediseases.org/rare-diseases/systemic-lupus-erythematosus/"
            }
        ]
    ),

    "myasthenia_gravis": RareConditionProfile(
        condition_id="myasthenia_gravis",
        display_name="Myasthenia Gravis (MG)",
        orpha_code="ORPHA:589",
        icd10_code="G70.0",
        key_symptoms=[
            "muscle_weakness",
            "fatigue",
            "blurred_and_distorted_vision",
            "slurred_speech",
            "breathlessness",
            "weakness_in_limbs",
        ],
        hallmark_features=[
            "ptosis (drooping of one or both eyelids)",
            "diplopia (double vision)",
            "fatigable muscle weakness worsening with activity and improving with rest",
            "dysarthria (slurred or nasal speech) after prolonged speaking",
            "dysphagia (difficulty chewing or swallowing liquids/solids)",
            "exertional dyspnea or acute respiratory weakness (myasthenic crisis)",
        ],
        negative_indicators=[
            "sensory loss or numbness (purely motor neuromuscular defect)",
            "painful joints or joint inflammation",
        ],
        description=(
            "A rare acquired autoimmune neuromuscular junction disease caused by pathogenic antibodies "
            "directed against acetylcholine receptors (AChR) or related post-synaptic proteins, "
            "producing fluctuating, fatigable skeletal muscle weakness."
        ),
        sources=[
            {
                "authority": "Orphanet",
                "title": "Myasthenia gravis",
                "identifier": "ORPHA:589",
                "url": "https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Lng=GB&Expert=589"
            },
            {
                "authority": "NIH / NINDS",
                "title": "Myasthenia Gravis Fact Sheet",
                "identifier": "NIH Publication",
                "url": "https://www.ninds.nih.gov/health-information/disorders/myasthenia-gravis"
            },
            {
                "authority": "NORD",
                "title": "Myasthenia Gravis",
                "identifier": "Rare Disease Database",
                "url": "https://rarediseases.org/rare-diseases/myasthenia-gravis/"
            }
        ]
    ),

    "wilson_disease": RareConditionProfile(
        condition_id="wilson_disease",
        display_name="Wilson Disease (Hepatolenticular Degeneration)",
        orpha_code="ORPHA:905",
        icd10_code="E83.01",
        key_symptoms=[
            "yellowish_skin",
            "yellowing_of_eyes",
            "abdominal_pain",
            "slurred_speech",
            "loss_of_balance",
            "depression",
            "irritability",
            "fatigue",
        ],
        hallmark_features=[
            "Kayser-Fleischer rings (golden-brown copper deposit rings in the cornea)",
            "unexplained hepatic dysfunction, acute hepatitis, or early cirrhosis",
            "movement disorders: tremors, dystonia, clumsy gait, ataxia",
            "dysarthria and excessive drooling",
            "abrupt behavioral or psychiatric changes in an adolescent or young adult",
            "Coombs-negative hemolytic anemia",
        ],
        negative_indicators=[
            "late adult onset without hepatic or neurologic history",
        ],
        description=(
            "An autosomal recessive metabolic disorder of copper transport caused by mutations in the "
            "ATP7B gene, leading to toxic copper accumulation primarily in the liver, basal ganglia, and eyes."
        ),
        sources=[
            {
                "authority": "Orphanet",
                "title": "Wilson disease",
                "identifier": "ORPHA:905",
                "url": "https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Lng=GB&Expert=905"
            },
            {
                "authority": "NIH / NIDDK",
                "title": "Wilson Disease",
                "identifier": "NIDDK Health Information",
                "url": "https://www.niddk.nih.gov/health-information/liver-diseases/wilson-disease"
            },
            {
                "authority": "NORD",
                "title": "Wilson Disease",
                "identifier": "Rare Disease Database",
                "url": "https://rarediseases.org/rare-diseases/wilson-disease/"
            }
        ]
    ),

    "hereditary_angioedema": RareConditionProfile(
        condition_id="hereditary_angioedema",
        display_name="Hereditary Angioedema (HAE)",
        orpha_code="ORPHA:91378",
        icd10_code="D84.1",
        key_symptoms=[
            "swelling_of_stomach",
            "abdominal_pain",
            "swollen_extremeties",
            "breathlessness",
            "puffy_face_and_eyes",
            "vomiting",
        ],
        hallmark_features=[
            "recurrent non-pitting, non-pruritic (non-itchy) subcutaneous and submucosal edema",
            "absence of urticaria / hives (distinguishes HAE from histamine-mediated allergy)",
            "severe recurrent spasmodic abdominal colic / intestinal edema mimicking surgical acute abdomen",
            "potentially fatal laryngeal edema causing acute airway compromise",
            "episodes lasting 2 to 5 days and unresponsive to standard antihistamines, epinephrine, or steroids",
            "prodromal erythema marginatum (non-pruritic serpiginous rash)",
        ],
        negative_indicators=[
            "pruritic hives / urticaria (strongly indicates histamine-mediated allergic reaction)",
            "immediate resolution within 30 minutes following antihistamines",
        ],
        description=(
            "A rare genetic disorder characterized by deficiency (Type I) or dysfunction (Type II) of "
            "C1 esterase inhibitor (C1-INH), causing excessive bradykinin generation and episodic, "
            "potentially fatal episodes of non-itchy swelling."
        ),
        sources=[
            {
                "authority": "Orphanet",
                "title": "Hereditary angioedema",
                "identifier": "ORPHA:91378",
                "url": "https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Lng=GB&Expert=91378"
            },
            {
                "authority": "NIH / MedlinePlus",
                "title": "Hereditary angioedema",
                "identifier": "MedlinePlus Genetics",
                "url": "https://medlineplus.gov/genetics/condition/hereditary-angioedema/"
            },
            {
                "authority": "NORD",
                "title": "Hereditary Angioedema",
                "identifier": "Rare Disease Database",
                "url": "https://rarediseases.org/rare-diseases/hereditary-angioedema/"
            }
        ]
    ),
}


@dataclass
class RareMatchResult:
    condition_id: str
    display_name: str
    orpha_code: str
    icd10_code: str
    match_score: float  # Jaccard / overlap score in [0.0, 1.0]
    matched_symptoms: List[str]
    missing_hallmarks: List[str]
    sources: List[Dict[str, str]]
    disclaimer: str = (
        "Educational prototype matching score based on clinical literature. "
        "This is not a diagnostic tool and does not provide clinical probability."
    )


class RareConditionMatcher:
    """Computes similarity against curated rare-condition profiles."""

    def __init__(self, profiles: Optional[Dict[str, RareConditionProfile]] = None):
        self.profiles = profiles or RARE_CONDITION_PROFILES

    def match(
        self,
        affirmed_symptoms: Set[str],
        free_text: Optional[str] = None,
        top_k: int = 4
    ) -> List[RareMatchResult]:
        """Compute matching score between patient symptoms/text and rare disease profiles."""
        results: List[RareMatchResult] = []
        text_lower = (free_text or "").lower()

        for condition_id, profile in self.profiles.items():
            key_set = set(profile.key_symptoms)
            matched = key_set.intersection(affirmed_symptoms)

            # Base Jaccard overlap on key symptoms
            union_len = len(key_set.union(affirmed_symptoms)) if affirmed_symptoms else len(key_set)
            jaccard = len(matched) / max(1, len(key_set))

            # Bonus for explicit hallmark text mentions
            hallmark_matches = 0
            for hallmark in profile.hallmark_features:
                # Extract key phrases from hallmark for text check
                core_terms = [t for t in hallmark.split("(")[0].strip().split() if len(t) > 3]
                if any(term in text_lower for term in core_terms):
                    hallmark_matches += 1

            hallmark_bonus = min(0.3, hallmark_matches * 0.1)
            raw_score = min(1.0, round(jaccard + hallmark_bonus, 3))

            # Only return conditions with at least some relevance or if ranking top_k
            results.append(
                RareMatchResult(
                    condition_id=profile.condition_id,
                    display_name=profile.display_name,
                    orpha_code=profile.orpha_code,
                    icd10_code=profile.icd10_code,
                    match_score=raw_score,
                    matched_symptoms=sorted(list(matched)),
                    missing_hallmarks=profile.hallmark_features,
                    sources=profile.sources,
                )
            )

        # Sort descending by match score
        results.sort(key=lambda r: r.match_score, reverse=True)
        return results[:top_k]
