"""
BhasaQuest-1: Baseline Training, Leakage-Free CV, and Submission Pipeline
========================================================================
This module implements:
- 5-Fold Stratified Cross-Validation with strict anti-leakage isolation
  (vectorizers fitted strictly inside training folds).
- Hybrid Word (1-2) + Character (3-5 char_wb) TF-IDF feature extraction.
- Balanced Logistic Regression classifier.
- Out-Of-Fold (OOF) prediction generation and probability matrix caching.
- Macro F1, precision, recall, and confusion matrix calculation.
- Final model retraining on 100% labeled training data.
- Test prediction generation and automated 10-point submission format audit.
- Generates:
    Submission/submission_tfidf_baseline.csv
    Submission/submission_final.csv
    Submission/sample_submission_template.csv
"""

import os
import re
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and pandas data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def setup_logger(log_file: Path) -> logging.Logger:
    """Set up logging to console and file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("SubmissionPipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def normalize_text(text: str) -> str:
    """Phonetically stabilize Romanized Bangla text while preserving hyphens."""
    s = str(text) if text is not None else ""
    s = s.lower()
    # Collapse runaway repeated characters (>2)
    s = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_vectorizers(
    word_ngram_range: Tuple[int, int] = (1, 2),
    word_max_features: int = 15000,
    char_ngram_range: Tuple[int, int] = (3, 5),
    char_max_features: int = 35000
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Create word and char_wb TF-IDF vectorizers."""
    word_vec = TfidfVectorizer(
        ngram_range=word_ngram_range,
        max_features=word_max_features,
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b"
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=char_max_features,
        min_df=3,
        sublinear_tf=True
    )
    return word_vec, char_vec


def audit_submission_file(
    sub_path: Path,
    expected_test_path: Path,
    expected_labels: List[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Execute rigorous 10-point compliance audit on submission CSV."""
    logger.info(f"Auditing submission compliance: {sub_path.name}")
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file missing: {sub_path.resolve()}")

    sub_df = pd.read_csv(sub_path)
    test_df = pd.read_csv(expected_test_path)

    checks = {}

    # 1. Exact columns
    checks["columns_match"] = list(sub_df.columns) == ["id", "label"]

    # 2. Row count exact
    checks["row_count_exact"] = len(sub_df) == len(test_df) == 2048

    # 3. No nulls or NaNs
    checks["no_null_values"] = sub_df.isnull().sum().sum() == 0

    # 4. No empty strings or whitespaces
    checks["no_empty_strings"] = (sub_df["id"].astype(str).str.strip() != "").all() and \
                                 (sub_df["label"].astype(str).str.strip() != "").all()

    # 5. ID uniqueness
    checks["ids_unique"] = sub_df["id"].nunique() == 2048

    # 6. ID identical match to test in exact order
    checks["ids_match_test_exact_sequence"] = (sub_df["id"] == test_df["id"]).all()

    # 7. Valid labels only
    sub_label_set = set(sub_df["label"].unique())
    expected_set = set(expected_labels)
    checks["valid_labels_only"] = sub_label_set.issubset(expected_set)
    checks["unexpected_labels"] = list(sub_label_set - expected_set)

    # 8. All classes predicted (guard against mode collapse)
    checks["all_classes_predicted"] = len(sub_label_set) == len(expected_set)

    # 9. Class distribution
    class_dist = sub_df["label"].value_counts().to_dict()
    checks["class_distribution"] = class_dist

    # 10. Audit result
    all_passed = (
        checks["columns_match"] and
        checks["row_count_exact"] and
        checks["no_null_values"] and
        checks["no_empty_strings"] and
        checks["ids_unique"] and
        checks["ids_match_test_exact_sequence"] and
        checks["valid_labels_only"] and
        checks["all_classes_predicted"]
    )
    checks["audit_passed"] = bool(all_passed)

    if not all_passed:
        logger.error(f"SUBMISSION AUDIT FAILED: {checks}")
        raise ValueError(f"Submission validation failed for {sub_path.name}")

    logger.info(f"Submission audit PASSED: 2048 rows, correct columns, exact test sequence, valid labels.")
    return checks


def run_cross_validation_and_submission(
    data_dir: Path,
    submission_dir: Path,
    n_splits: int = 5,
    c_param: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    """Execute complete CV, model training, and submission generation."""
    models_dir = submission_dir / "models"
    reports_dir = submission_dir / "reports"
    logs_dir = submission_dir / "logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "submission_pipeline.log")
    logger.info("=" * 60)
    logger.info("STARTING BHASAQUEST-1 BASELINE CV & SUBMISSION PIPELINE")
    logger.info("=" * 60)

    # 1. Copy sample submission template
    sample_sub_src = data_dir / "sample_submission.csv"
    sample_sub_dst = submission_dir / "sample_submission_template.csv"
    if sample_sub_src.exists():
        shutil.copyfile(sample_sub_src, sample_sub_dst)
        logger.info(f"Created reference template: {sample_sub_dst.name}")

    # Load datasets
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Map labels to integer indices
    classes = sorted(train_df["label"].unique().tolist())
    label2idx = {lbl: i for i, lbl in enumerate(classes)}
    idx2label = {i: lbl for i, lbl in enumerate(classes)}

    y = np.array([label2idx[l] for l in train_df["label"]])
    train_texts = [normalize_text(t) for t in train_df["text"]]
    test_texts = [normalize_text(t) for t in test_df["text"]]

    # 2. Stratified 5-Fold Cross Validation (Strict Anti-Leakage)
    logger.info(f"Running {n_splits}-Fold Stratified Cross-Validation (Seed={seed})...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_probs = np.zeros((len(train_df), len(classes)), dtype=np.float32)
    fold_macro_f1s = []
    fold_details = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, y), 1):
        fold_train_texts = [train_texts[i] for i in train_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit vectorizers strictly on training fold
        word_vec, char_vec = build_vectorizers()
        X_tr_w = word_vec.fit_transform(fold_train_texts)
        X_tr_c = char_vec.fit_transform(fold_train_texts)
        X_train_fold = sparse.hstack([X_tr_w, X_tr_c], format="csr")

        # Transform validation fold strictly with fold-fitted vectorizers
        X_val_w = word_vec.transform(fold_val_texts)
        X_val_c = char_vec.transform(fold_val_texts)
        X_val_fold = sparse.hstack([X_val_w, X_val_c], format="csr")

        # Train model
        clf = LogisticRegression(
            C=c_param,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=seed + fold
        )
        clf.fit(X_train_fold, y_train)

        # Predict probabilities
        val_probs = clf.predict_proba(X_val_fold)
        oof_probs[val_idx] = val_probs

        val_preds = np.argmax(val_probs, axis=1)
        fold_f1 = f1_score(y_val, val_preds, average="macro")
        fold_macro_f1s.append(fold_f1)

        logger.info(f"Fold {fold}/{n_splits} Macro F1: {fold_f1:.4f}")
        fold_details.append({
            "fold": fold,
            "val_samples": len(val_idx),
            "macro_f1": round(float(fold_f1), 4)
        })

    # Overall OOF Evaluation
    oof_preds = np.argmax(oof_probs, axis=1)
    overall_macro_f1 = f1_score(y, oof_preds, average="macro")
    overall_acc = (oof_preds == y).mean()

    # Per-class metrics
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y, oof_preds, labels=range(len(classes)))
    per_class_metrics = {}
    for i, c in enumerate(classes):
        per_class_metrics[c] = {
            "precision": round(float(precisions[i]), 4),
            "recall": round(float(recalls[i]), 4),
            "f1_score": round(float(f1s[i]), 4),
            "support": int(supports[i])
        }

    conf_mat = confusion_matrix(y, oof_preds)

    logger.info("=" * 40)
    logger.info(f"OVERALL OUT-OF-FOLD MACRO F1: {overall_macro_f1:.4f}")
    logger.info(f"OVERALL OUT-OF-FOLD ACCURACY: {overall_acc:.4f}")
    logger.info("=" * 40)

    # Save OOF probability matrix
    np.save(models_dir / "oof_baseline_probs.npy", oof_probs)
    logger.info(f"Saved OOF probability matrix: {models_dir / 'oof_baseline_probs.npy'}")

    # Generate Confusion Matrix Visualization
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax
    )
    ax.set_title(f"Baseline 5-Fold OOF Confusion Matrix\n(Macro F1 = {overall_macro_f1:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    plt.tight_layout()
    conf_mat_path = reports_dir / "confusion_matrix.png"
    plt.savefig(conf_mat_path, bbox_inches="tight")
    plt.close()

    # 3. Final Model Training on 100% Labeled Training Data
    logger.info("Fitting final production model on 100% labeled training dataset...")
    final_word_vec, final_char_vec = build_vectorizers()
    X_train_full_w = final_word_vec.fit_transform(train_texts)
    X_train_full_c = final_char_vec.fit_transform(train_texts)
    X_train_full = sparse.hstack([X_train_full_w, X_train_full_c], format="csr")

    final_clf = LogisticRegression(
        C=c_param,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed
    )
    final_clf.fit(X_train_full, y)

    # Transform test set
    X_test_w = final_word_vec.transform(test_texts)
    X_test_c = final_char_vec.transform(test_texts)
    X_test_full = sparse.hstack([X_test_w, X_test_c], format="csr")

    # Predict test probabilities
    test_probs = final_clf.predict_proba(X_test_full)
    np.save(models_dir / "test_baseline_probs.npy", test_probs)
    joblib.dump(final_clf, models_dir / "final_logistic_regression.joblib")
    joblib.dump(final_word_vec, models_dir / "final_word_vectorizer.joblib")
    joblib.dump(final_char_vec, models_dir / "final_char_vectorizer.joblib")

    # Generate predictions
    test_preds_idx = np.argmax(test_probs, axis=1)
    test_preds_labels = [idx2label[i] for i in test_preds_idx]

    # Create submission dataframe
    sub_df = pd.DataFrame({
        "id": test_df["id"],
        "label": test_preds_labels
    })

    # Save both named submissions
    baseline_sub_path = submission_dir / "submission_tfidf_baseline.csv"
    final_sub_path = submission_dir / "submission_final.csv"

    sub_df.to_csv(baseline_sub_path, index=False)
    sub_df.to_csv(final_sub_path, index=False)
    logger.info(f"Saved submission files: {baseline_sub_path.name}, {final_sub_path.name}")

    # 4. Strict Submission Validation
    audit_results = audit_submission_file(final_sub_path, data_dir / "test.csv", classes, logger)

    # 5. Save CV Evaluation Report
    cv_report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "Hybrid TF-IDF (Word 1-2 + Char 3-5) + Balanced LogisticRegression",
        "hyperparameters": {
            "C": c_param,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "n_splits": n_splits,
            "seed": seed
        },
        "overall_oof_macro_f1": round(float(overall_macro_f1), 4),
        "overall_oof_accuracy": round(float(overall_acc), 4),
        "mean_fold_macro_f1": round(float(np.mean(fold_macro_f1s)), 4),
        "std_fold_macro_f1": round(float(np.std(fold_macro_f1s)), 4),
        "folds": fold_details,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_mat.tolist(),
        "classes": classes,
        "submission_audit": audit_results
    }

    report_path = reports_dir / "cv_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(cv_report, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Saved evaluation report to: {report_path.resolve()}")
    logger.info("Submission pipeline completed successfully.")
    return cv_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Training and Submission Generation")
    parser.add_argument("--data_dir", type=str, default="Datasets", help="Directory containing train.csv and test.csv")
    parser.add_argument("--submission_dir", type=str, default="Submission", help="Directory where models, reports, and submissions will be saved")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--c_param", type=float, default=1.0, help="Inverse regularization strength C")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    d_path = Path(args.data_dir)
    if not d_path.is_absolute():
        d_path = (project_root / d_path).resolve()

    s_path = Path(args.submission_dir)
    if not s_path.is_absolute():
        s_path = (project_root / s_path).resolve()

    run_cross_validation_and_submission(
        d_path, s_path, n_splits=args.n_splits, c_param=args.c_param, seed=args.seed
    )
