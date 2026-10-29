"""Automated unit and integration tests for the ML pipeline.

Tests:
1. Label normalization and mapping for 5 shared conditions across both datasets.
2. Gretel train/test split loading, deduplication, and leakage prevention.
3. Kaggle deduplication, empty column dropping, and train/test feature isolation.
4. NLP symptom extraction, synonym matching, and negation detection.
5. Rare-condition profile integrity and literature provenance.
6. Model training and artifact serialization.
7. End-to-end inference adherence to backend contract.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.constants import (
    SHARED_5_CLASSES,
    DISPLAY_LABELS,
    CANONICAL_ALLERGY,
    CANONICAL_COMMON_COLD,
    CANONICAL_GERD,
    CANONICAL_MIGRAINE,
    CANONICAL_UTI,
    normalize_kaggle_label,
    normalize_gretel_label,
)
from src.nlp_rules import (
    extract_symptoms_with_context,
    extract_symptoms_with_negation,
    vector_from_symptoms,
    KAGGLE_SYMPTOM_FEATURES,
)
from src.rare_matcher import RARE_CONDITION_PROFILES, RareConditionMatcher
from src.data_loader import load_gretel_splits, load_kaggle_deduplicated_splits
from src.train import build_and_save_artifacts
from src.inference import CareAtlasPredictor, SymptomInput, predict_condition


def test_label_normalization():
    """Verify raw labels from both datasets map strictly to the 5 canonical classes."""
    # Kaggle mappings
    assert normalize_kaggle_label("Allergy") == CANONICAL_ALLERGY
    assert normalize_kaggle_label("Common Cold") == CANONICAL_COMMON_COLD
    assert normalize_kaggle_label("GERD") == CANONICAL_GERD
    assert normalize_kaggle_label("Migraine") == CANONICAL_MIGRAINE
    assert normalize_kaggle_label("Urinary tract infection") == CANONICAL_UTI
    assert normalize_kaggle_label("Diabetes ") is None  # Not in shared 5

    # Gretel mappings
    assert normalize_gretel_label("allergy") == CANONICAL_ALLERGY
    assert normalize_gretel_label("common cold") == CANONICAL_COMMON_COLD
    assert normalize_gretel_label("gastroesophageal reflux disease") == CANONICAL_GERD
    assert normalize_gretel_label("migraine") == CANONICAL_MIGRAINE
    assert normalize_gretel_label("urinary tract infection") == CANONICAL_UTI
    assert normalize_gretel_label("pneumonia") is None  # Not in shared 5


def test_gretel_data_loader():
    """Verify Gretel loading preserves train/test split and filters correctly."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw", "gretelai_symptom_to_diagnosis")

    train_df, test_df, meta = load_gretel_splits(raw_dir)
    assert len(train_df) == 189
    assert len(test_df) == 49
    assert set(train_df["target"].unique()) == set(SHARED_5_CLASSES)
    assert set(test_df["target"].unique()) == set(SHARED_5_CLASSES)


def test_kaggle_deduplication_and_no_leakage():
    """Verify Kaggle deduplication drops duplicates and prevents data leakage between splits."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw", "kaushil268_disease_prediction")

    train_df, test_df, feature_cols, meta = load_kaggle_deduplicated_splits(raw_dir, test_size=0.25, random_state=42)

    # Empty column must be removed
    assert "Unnamed: 133" not in feature_cols
    assert "Unnamed: 133" not in train_df.columns
    assert "Unnamed: 133" not in test_df.columns

    # Verify deduplication reduced hundreds of rows to exactly 36 unique patterns
    assert meta["unique_patterns_total"] == 36
    assert len(train_df) == 27
    assert len(test_df) == 9

    # Strict check: NO identical symptom vectors in train and test splits (zero data leakage)
    train_patterns = set(tuple(row) for row in train_df[feature_cols].values)
    test_patterns = set(tuple(row) for row in test_df[feature_cols].values)
    overlap = train_patterns.intersection(test_patterns)
    assert len(overlap) == 0, f"Detected {len(overlap)} leaking patterns across train/test splits!"


def test_nlp_extraction_and_negation():
    """Verify phrase matching, canonical mapping, and negation parsing."""
    text1 = "Patient has persistent cough and severe sore throat."
    affirmed1, negated1 = extract_symptoms_with_negation(text1)
    assert "cough" in affirmed1
    assert "throat_irritation" in affirmed1
    assert len(negated1) == 0

    text2 = "I have a terrible headache and nausea, but no fever and denied burning urination."
    affirmed2, negated2 = extract_symptoms_with_negation(text2)
    assert "headache" in affirmed2
    assert "nausea" in affirmed2
    assert "high_fever" in negated2
    assert "burning_micturition" in negated2
    assert "high_fever" not in affirmed2
    assert "burning_micturition" not in affirmed2

    vec = vector_from_symptoms(affirmed2, KAGGLE_SYMPTOM_FEATURES)
    assert len(vec) == 132
    headache_idx = KAGGLE_SYMPTOM_FEATURES.index("headache")
    nausea_idx = KAGGLE_SYMPTOM_FEATURES.index("nausea")
    assert vec[headache_idx] == 1
    assert vec[nausea_idx] == 1


def test_nlp_multiword_negation_triggers():
    cases = [
        ("I never had a headache", "headache"),
        ("Headache was ruled out", "headache"),
        ("Negative for cough", "cough"),
        ("I am free of nausea", "nausea"),
    ]

    for text, symptom in cases:
        affirmed, negated = extract_symptoms_with_negation(text)
        assert symptom in negated, text
        assert symptom not in affirmed, text


def test_nlp_negation_scope_tracks_coordinated_symptoms():
    affirmed, negated = extract_symptoms_with_negation("No headache and nausea")
    assert affirmed == set()
    assert {"headache", "nausea"}.issubset(negated)

    affirmed, negated = extract_symptoms_with_negation("Headache and no nausea")
    assert "headache" in affirmed
    assert "nausea" in negated
    assert "headache" not in negated

    affirmed, negated = extract_symptoms_with_negation("No headache, but nausea")
    assert "headache" in negated
    assert "nausea" in affirmed

    affirmed, negated = extract_symptoms_with_negation(
        "No headache, nausea, cough, fatigue, and dizziness"
    )
    assert affirmed == set()
    assert {"headache", "nausea", "cough", "fatigue", "dizziness"}.issubset(negated)

    affirmed, negated = extract_symptoms_with_negation("No headache, I have nausea")
    assert "headache" in negated
    assert "nausea" in affirmed


def test_nlp_longest_synonym_match_prevents_overlapping_fever_features():
    affirmed, negated = extract_symptoms_with_negation("Mild fever and low grade fever")
    assert {"mild_fever"}.issubset(affirmed)
    assert "high_fever" not in affirmed
    assert not negated

    affirmed, negated = extract_symptoms_with_negation("No mild fever")
    assert "mild_fever" in negated
    assert "high_fever" not in negated
    assert "mild_fever" not in affirmed


def test_nlp_historical_and_uncertain_mentions_are_not_current_affirmations():
    historical_cases = ("I had a headache last year", "I used to have a headache")
    uncertain_cases = ("I might have a headache", "Maybe headache")

    for text in historical_cases:
        affirmed, negated, historical, uncertain = extract_symptoms_with_context(text)
        assert "headache" not in affirmed, text
        assert "headache" not in negated, text
        assert "headache" in historical, text
        assert "headache" not in uncertain, text

    for text in uncertain_cases:
        affirmed, negated, historical, uncertain = extract_symptoms_with_context(text)
        assert "headache" not in affirmed, text
        assert "headache" not in negated, text
        assert "headache" not in historical, text
        assert "headache" in uncertain, text

    affirmed, negated, historical, uncertain = extract_symptoms_with_context(
        "No fever, and a cough last month"
    )
    assert "high_fever" in negated
    assert "high_fever" not in historical
    assert "cough" in historical
    assert not uncertain


def test_nlp_conflicting_current_mentions_are_visible_but_denial_wins_legacy_result():
    affirmed, negated, historical, uncertain = extract_symptoms_with_context(
        "No cough, but I have a cough"
    )
    assert "cough" in affirmed
    assert "cough" in negated
    assert not historical
    assert not uncertain

    affirmed, negated = extract_symptoms_with_negation("No cough, but I have a cough")
    assert "cough" not in affirmed
    assert "cough" in negated


def test_rare_condition_profiles_and_matcher():
    """Verify rare condition profiles have verifiable sources and score appropriately."""
    for cond_id in ["systemic_lupus_erythematosus", "myasthenia_gravis", "wilson_disease", "hereditary_angioedema"]:
        assert cond_id in RARE_CONDITION_PROFILES
        profile = RARE_CONDITION_PROFILES[cond_id]
        assert profile.orpha_code.startswith("ORPHA:")
        assert len(profile.key_symptoms) > 0
        assert len(profile.hallmark_features) > 0
        assert len(profile.sources) >= 2
        for src in profile.sources:
            assert "url" in src and src["url"].startswith("http")

    matcher = RareConditionMatcher()
    results = matcher.match(
        affirmed_symptoms={"fatigue", "joint_pain", "skin_rash"},
        free_text="Patient presents with butterfly rash and severe fatigue and joint pain",
        top_k=4
    )
    assert len(results) == 4
    # SLE should be top ranked
    assert results[0].condition_id == "systemic_lupus_erythematosus"
    assert results[0].match_score > 0.5
    assert "fatigue" in results[0].matched_symptoms
    assert "joint_pain" in results[0].matched_symptoms


def test_end_to_end_inference():
    """Verify inference pipeline produces compliant schema and predictions."""
    res = predict_condition(
        free_text_symptoms="Burning sensation when urinating and lower abdominal discomfort.",
        structured_symptoms=["burning_micturition", "bladder_discomfort"]
    )
    assert "model_version" in res
    assert "predictions" in res
    assert "metadata" in res
    assert "rare_condition_matches" in res
    assert len(res["predictions"]) == 5

    # UTI should be top predicted
    top_pred = res["predictions"][0]
    assert top_pred["condition"] == DISPLAY_LABELS[CANONICAL_UTI]
    assert top_pred["confidence"] > 0.3
    assert len(res["rare_condition_matches"]) == 4
