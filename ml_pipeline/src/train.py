"""Model training, evaluation, and serialized artifact generation.

Trains:
1. TF-IDF + Multinomial Logistic Regression for free-text (Gretel)
2. Multinomial Naive Bayes / Logistic Regression for structured symptoms (Kaggle deduplicated)
Generates full classification reports, confusion matrices, and metrics json.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from .constants import SHARED_5_CLASSES, DISPLAY_LABELS
from .data_loader import load_gretel_splits, load_kaggle_deduplicated_splits


def train_text_classifier(
    gretel_raw_dir: str,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train TF-IDF + LogisticRegression baseline on Gretel text training split only."""
    train_df, test_df, split_meta = load_gretel_splits(gretel_raw_dir)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=1500,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=random_state,
            class_weight="balanced"
        ))
    ])

    # Fit ONLY on training data
    pipeline.fit(train_df["input_text"], train_df["target"])

    # Predict on test data
    y_true = test_df["target"].values
    y_pred = pipeline.predict(test_df["input_text"])
    y_prob = pipeline.predict_proba(test_df["input_text"])

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    labels_in_test = sorted(list(set(y_true)))
    report_dict = classification_report(
        y_true, y_pred, labels=labels_in_test, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels_in_test).tolist()

    metrics = {
        "model_name": "gretel_tfidf_logistic_regression",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "target_classes": SHARED_5_CLASSES,
        "split_metadata": split_meta,
        "evaluation": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "labels": labels_in_test,
            "confusion_matrix": conf_mat,
            "classification_report": report_dict,
        }
    }

    return pipeline, metrics


def train_symptom_classifier(
    kaggle_raw_dir: str,
    random_state: int = 42
) -> Tuple[Pipeline, List[str], Dict[str, Any]]:
    """Train structured symptom classifier on Kaggle deduplicated dataset."""
    train_df, test_df, feature_cols, split_meta = load_kaggle_deduplicated_splits(
        kaggle_raw_dir, test_size=0.25, random_state=random_state
    )

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    # MultinomialNB baseline for binary count features
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    labels_in_test = sorted(list(set(y_test)))
    report_dict = classification_report(
        y_test, y_pred, labels=labels_in_test, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(y_test, y_pred, labels=labels_in_test).tolist()

    metrics = {
        "model_name": "kaggle_symptom_multinomial_nb",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "feature_count": len(feature_cols),
        "target_classes": SHARED_5_CLASSES,
        "split_metadata": split_meta,
        "evaluation": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "labels": labels_in_test,
            "confusion_matrix": conf_mat,
            "classification_report": report_dict,
        }
    }

    return clf, feature_cols, metrics


def build_and_save_artifacts(
    data_dir: Optional[str] = None,
    artifacts_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train both models and save artifacts with metadata and metrics."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "data", "raw")
    if artifacts_dir is None:
        artifacts_dir = os.path.join(base_dir, "artifacts")

    os.makedirs(artifacts_dir, exist_ok=True)

    gretel_dir = os.path.join(data_dir, "gretelai_symptom_to_diagnosis")
    kaggle_dir = os.path.join(data_dir, "kaushil268_disease_prediction")

    # 1. Train text classifier
    text_model, text_metrics = train_text_classifier(gretel_dir)
    text_artifact_path = os.path.join(artifacts_dir, "text_classifier.joblib")
    joblib.dump(text_model, text_artifact_path)

    # 2. Train symptom classifier
    symptom_model, feature_cols, symptom_metrics = train_symptom_classifier(kaggle_dir)
    symptom_artifact_path = os.path.join(artifacts_dir, "symptom_classifier.joblib")
    joblib.dump(
        {"model": symptom_model, "feature_names": feature_cols},
        symptom_artifact_path
    )

    # 3. Save combined metadata & metrics
    combined_metadata = {
        "pipeline_version": "0.1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "target_classes": SHARED_5_CLASSES,
        "display_labels": DISPLAY_LABELS,
        "text_classifier": text_metrics,
        "symptom_classifier": symptom_metrics,
        "disclaimer": (
            "Educational and research prototype only. Models are trained on small, "
            "synthetic, and augmented datasets and must not be used as clinical diagnostic tools."
        )
    }

    meta_path = os.path.join(artifacts_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, indent=2)

    return combined_metadata
