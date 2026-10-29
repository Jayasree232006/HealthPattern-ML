"""Data loading, cleaning, deduplication, and schema validation.

Handles:
- Loading raw Gretel symptom_to_diagnosis JSONL
- Filtering and mapping the 5 shared target classes
- Dropping 'Unnamed: 133' from Kaggle training CSV
- Kaggle row deduplication and split isolation to prevent data leakage
"""

import os
from typing import Dict, Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .constants import (
    SHARED_5_CLASSES,
    KAGGLE_RAW_TARGET_MAP,
    GRETEL_RAW_TARGET_MAP,
    normalize_kaggle_label,
    normalize_gretel_label,
)


def load_gretel_splits(
    raw_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Load and prepare Gretel Symptom-to-Diagnosis dataset for the 5 shared classes.

    Returns:
        (train_df, test_df, metadata)
    """
    train_file = os.path.join(raw_dir, "train.jsonl")
    test_file = os.path.join(raw_dir, "test.jsonl")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Gretel jsonl files not found in {raw_dir}")

    train_raw = pd.read_json(train_file, lines=True)
    test_raw = pd.read_json(test_file, lines=True)

    # Map labels to canonical names
    train_raw["target"] = train_raw["output_text"].apply(normalize_gretel_label)
    test_raw["target"] = test_raw["output_text"].apply(normalize_gretel_label)

    # Filter to the 5 shared classes
    train_clean = train_raw[train_raw["target"].isin(SHARED_5_CLASSES)].copy()
    test_clean = test_raw[test_raw["target"].isin(SHARED_5_CLASSES)].copy()

    # Reset indices
    train_clean = train_clean.reset_index(drop=True)
    test_clean = test_clean.reset_index(drop=True)

    metadata = {
        "raw_train_total": len(train_raw),
        "raw_test_total": len(test_raw),
        "filtered_train_total": len(train_clean),
        "filtered_test_total": len(test_clean),
        "train_class_counts": train_clean["target"].value_counts().to_dict(),
        "test_class_counts": test_clean["target"].value_counts().to_dict(),
    }

    return train_clean, test_clean, metadata


def load_kaggle_deduplicated_splits(
    raw_dir: str,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, any]]:
    """Load Kaggle dataset, drop empty columns, filter 5 classes, deduplicate identical patterns,
    and perform stratified train/test split on unique symptom vectors to guarantee NO LEAKAGE.

    Returns:
        (train_df, test_df, feature_names, metadata)
    """
    train_file = os.path.join(raw_dir, "Training.csv")
    test_file = os.path.join(raw_dir, "Testing.csv")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Kaggle Training.csv not found in {raw_dir}")

    raw_train = pd.read_csv(train_file)
    raw_test = pd.read_csv(test_file) if os.path.exists(test_file) else None

    # Drop Unnamed: 133 if present
    if "Unnamed: 133" in raw_train.columns:
        raw_train = raw_train.drop(columns=["Unnamed: 133"])

    raw_train = raw_train.copy()
    raw_train["target"] = raw_train["prognosis"].apply(normalize_kaggle_label)
    k_train_5 = raw_train[raw_train["target"].isin(SHARED_5_CLASSES)].copy()

    if raw_test is not None:
        raw_test = raw_test.copy()
        if "Unnamed: 133" in raw_test.columns:
            raw_test = raw_test.drop(columns=["Unnamed: 133"])
        raw_test["target"] = raw_test["prognosis"].apply(normalize_kaggle_label)
        k_test_5 = raw_test[raw_test["target"].isin(SHARED_5_CLASSES)].copy()
        combined = pd.concat([k_train_5, k_test_5], ignore_index=True)
    else:
        combined = k_train_5

    feature_cols = [c for c in combined.columns if c not in ["prognosis", "target"]]

    # Deduplicate: exact symptom pattern + target
    unique_patterns = combined.drop_duplicates(subset=feature_cols + ["target"]).copy()
    unique_patterns = unique_patterns.reset_index(drop=True)

    # Perform stratified split on the unique symptom patterns
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(unique_patterns[feature_cols], unique_patterns["target"]))

    train_df = unique_patterns.iloc[train_idx].copy().reset_index(drop=True)
    test_df = unique_patterns.iloc[test_idx].copy().reset_index(drop=True)

    metadata = {
        "raw_total_rows": len(combined),
        "duplicate_rows_dropped": len(combined) - len(unique_patterns),
        "unique_patterns_total": len(unique_patterns),
        "unique_patterns_per_class": unique_patterns["target"].value_counts().to_dict(),
        "train_unique_patterns": len(train_df),
        "test_unique_patterns": len(test_df),
        "train_class_counts": train_df["target"].value_counts().to_dict(),
        "test_class_counts": test_df["target"].value_counts().to_dict(),
    }

    return train_df, test_df, feature_cols, metadata
