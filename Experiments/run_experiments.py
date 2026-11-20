"""
BhasaQuest-1: Phase 2 — Controlled NLP Model Experiments Runner
=============================================================
This module conducts controlled, reproducible, leak-free model experiments
comparing Logistic Regression and LinearSVC across multiple feature representations
(Word TF-IDF, Character TF-IDF, and Combined TF-IDF), class weighting regimes,
and regularization strengths C.

Evaluated on exact 5-Fold StratifiedKFold (random_state=42).
Produces:
- Experiments/experiment_results.csv (ranked performance table)
- Experiments/experiment_report.json (detailed metric breakdown)
- Experiments/reports/error_analysis.json (focused error analysis on 19E1A1 vs 72A2C8)
- Experiments/figures/model_comparison_macro_f1.png
- Experiments/figures/best_model_confusion_matrix.png
- Experiments/figures/confusion_19E1A1_vs_72A2C8.png
- Refits the best candidate on 100% training data and generates:
  Submission/submission_experiment_best.csv (audited 10-point compliance)
"""

import os
import re
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and pandas types."""
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
    """Configure console and file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Phase2Experiments")
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
    """Standardize Romanized Bangla text while preserving hyphens and structural casing."""
    s = str(text) if text is not None else ""
    s = s.lower()
    # Normalize expressive character repetitions (threshold=2)
    s = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def audit_submission_file(
    sub_path: Path,
    expected_test_path: Path,
    expected_labels: List[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Execute 10-point compliance audit on submission CSV."""
    logger.info(f"Auditing submission compliance: {sub_path.name}")
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file missing: {sub_path.resolve()}")

    sub_df = pd.read_csv(sub_path)
    test_df = pd.read_csv(expected_test_path)

    checks = {
        "columns_match": list(sub_df.columns) == ["id", "label"],
        "row_count_exact": len(sub_df) == len(test_df) == 2048,
        "no_null_values": sub_df.isnull().sum().sum() == 0,
        "no_empty_strings": (sub_df["id"].astype(str).str.strip() != "").all() and
                             (sub_df["label"].astype(str).str.strip() != "").all(),
        "ids_unique": sub_df["id"].nunique() == 2048,
        "ids_match_test_exact_sequence": (sub_df["id"] == test_df["id"]).all(),
        "valid_labels_only": set(sub_df["label"].unique()).issubset(set(expected_labels)),
        "all_classes_predicted": len(set(sub_df["label"].unique())) == len(expected_labels),
        "class_distribution": sub_df["label"].value_counts().to_dict(),
    }
    checks["audit_passed"] = bool(
        checks["columns_match"] and
        checks["row_count_exact"] and
        checks["no_null_values"] and
        checks["no_empty_strings"] and
        checks["ids_unique"] and
        checks["ids_match_test_exact_sequence"] and
        checks["valid_labels_only"] and
        checks["all_classes_predicted"]
    )
    if not checks["audit_passed"]:
        logger.error(f"AUDIT FAILED: {checks}")
        raise ValueError(f"Submission audit failed for {sub_path.name}")
    logger.info(f"Submission audit PASSED: 2048 rows, correct format, exact ID sequence, valid labels.")
    return checks


def precompute_fold_features(
    texts: List[str],
    y: np.ndarray,
    skf: StratifiedKFold,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Precompute and cache leakage-free sparse feature matrices for each fold.
    Guarantees 100% strict anti-leakage: vectorizers are fitted exclusively
    on each fold's training text, then transform validation text.
    """
    logger.info("Precomputing leakage-free fold vectorizers & feature matrices...")
    fold_cache = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y), 1):
        tr_txt = [texts[i] for i in train_idx]
        val_txt = [texts[i] for i in val_idx]

        # Word TF-IDF Vectorizer
        wv = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b"
        )
        # Character TF-IDF Vectorizer
        cv = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=35000,
            min_df=3,
            sublinear_tf=True
        )

        # Fit strictly on fold training data
        X_tr_w = wv.fit_transform(tr_txt)
        X_val_w = wv.transform(val_txt)

        X_tr_c = cv.fit_transform(tr_txt)
        X_val_c = cv.transform(val_txt)

        # Concatenate for combined
        X_tr_comb = sparse.hstack([X_tr_w, X_tr_c], format="csr")
        X_val_comb = sparse.hstack([X_val_w, X_val_c], format="csr")

        fold_cache.append({
            "fold": fold,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "y_tr": y[train_idx],
            "y_val": y[val_idx],
            "val_texts": val_txt,
            "word_tr": X_tr_w, "word_val": X_val_w,
            "char_tr": X_tr_c, "char_val": X_val_c,
            "comb_tr": X_tr_comb, "comb_val": X_val_comb
        })
        logger.info(f"Fold {fold}/5 matrices ready: Word {X_tr_w.shape[1]}, Char {X_tr_c.shape[1]}, Comb {X_tr_comb.shape[1]}")

    logger.info(f"All 5 folds precomputed in {time.time() - t0:.2f}s")
    return fold_cache


def run_experiment_suite(
    data_dir: Path,
    exp_dir: Path,
    sub_dir: Path,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete controlled experiment suite."""
    models_dir = exp_dir / "models"
    reports_dir = exp_dir / "reports"
    figures_dir = exp_dir / "figures"
    logs_dir = exp_dir / "logs"

    for d in [models_dir, reports_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "experiments.log")
    logger.info("=" * 65)
    logger.info("STARTING BHASAQUEST-1 CONTROLLED NLP MODEL EXPERIMENTS")
    logger.info("=" * 65)

    # Load labeled training dataset
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    classes = sorted(train_df["label"].unique().tolist())
    label2idx = {lbl: i for i, lbl in enumerate(classes)}
    idx2label = {i: lbl for i, lbl in enumerate(classes)}

    y = np.array([label2idx[l] for l in train_df["label"]])
    texts = [normalize_text(t) for t in train_df["text"]]

    # Stratified 5-Fold Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_cache = precompute_fold_features(texts, y, skf, logger)

    # Define Experiment Configurations Matrix
    configs = [
        # --- Baseline Group: Logistic Regression Feature Ablation ---
        {"id": "EXP01_LogReg_Word_C1.0_Balanced", "model": "LogisticRegression", "features": "word", "C": 1.0, "class_weight": "balanced"},
        {"id": "EXP02_LogReg_Char_C1.0_Balanced", "model": "LogisticRegression", "features": "char", "C": 1.0, "class_weight": "balanced"},
        {"id": "EXP03_LogReg_Comb_C1.0_Balanced", "model": "LogisticRegression", "features": "comb", "C": 1.0, "class_weight": "balanced"},  # Existing Baseline
        {"id": "EXP04_LogReg_Comb_C1.0_None",     "model": "LogisticRegression", "features": "comb", "C": 1.0, "class_weight": None},
        {"id": "EXP05_LogReg_Comb_C0.5_Balanced", "model": "LogisticRegression", "features": "comb", "C": 0.5, "class_weight": "balanced"},
        {"id": "EXP06_LogReg_Comb_C2.0_Balanced", "model": "LogisticRegression", "features": "comb", "C": 2.0, "class_weight": "balanced"},

        # --- LinearSVC Feature Ablation ---
        {"id": "EXP07_LinearSVC_Word_C1.0_Balanced", "model": "LinearSVC", "features": "word", "C": 1.0, "class_weight": "balanced"},
        {"id": "EXP08_LinearSVC_Char_C1.0_Balanced", "model": "LinearSVC", "features": "char", "C": 1.0, "class_weight": "balanced"},
        {"id": "EXP09_LinearSVC_Comb_C1.0_Balanced", "model": "LinearSVC", "features": "comb", "C": 1.0, "class_weight": "balanced"},
        {"id": "EXP10_LinearSVC_Comb_C1.0_None",     "model": "LinearSVC", "features": "comb", "C": 1.0, "class_weight": None},

        # --- LinearSVC Regularization Strength Search ---
        {"id": "EXP11_LinearSVC_Comb_C0.25_Balanced", "model": "LinearSVC", "features": "comb", "C": 0.25, "class_weight": "balanced"},
        {"id": "EXP12_LinearSVC_Comb_C0.5_Balanced",  "model": "LinearSVC", "features": "comb", "C": 0.5,  "class_weight": "balanced"},
        {"id": "EXP13_LinearSVC_Comb_C0.75_Balanced", "model": "LinearSVC", "features": "comb", "C": 0.75, "class_weight": "balanced"},
        {"id": "EXP14_LinearSVC_Comb_C0.5_None",      "model": "LinearSVC", "features": "comb", "C": 0.5,  "class_weight": None},
        {"id": "EXP15_LinearSVC_Comb_C1.5_Balanced",  "model": "LinearSVC", "features": "comb", "C": 1.5,  "class_weight": "balanced"},
    ]

    results_table = []
    detailed_reports = {}
    oof_predictions_dict = {}

    logger.info(f"Executing {len(configs)} controlled model experiments...")

    for exp in configs:
        exp_id = exp["id"]
        model_type = exp["model"]
        feat_name = exp["features"]
        C_val = exp["C"]
        cw_val = exp["class_weight"]

        logger.info(f"--- Running {exp_id} ---")
        t_start = time.time()

        oof_preds = np.zeros(len(texts), dtype=int)
        fold_scores = []

        for f in fold_cache:
            X_tr = f[f"{feat_name}_tr"]
            X_val = f[f"{feat_name}_val"]
            y_tr = f["y_tr"]
            y_val = f["y_val"]
            val_idx = f["val_idx"]

            if model_type == "LogisticRegression":
                clf = LogisticRegression(
                    C=C_val,
                    class_weight=cw_val,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=seed + f["fold"]
                )
            elif model_type == "LinearSVC":
                clf = LinearSVC(
                    C=C_val,
                    class_weight=cw_val,
                    max_iter=2500,
                    random_state=seed + f["fold"]
                )
            else:
                raise ValueError(f"Unknown model: {model_type}")

            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            oof_preds[val_idx] = preds

            fold_f1 = f1_score(y_val, preds, average="macro")
            fold_scores.append(fold_f1)

        dur = time.time() - t_start
        overall_macro_f1 = f1_score(y, oof_preds, average="macro")
        overall_acc = accuracy_score(y, oof_preds)
        mean_fold_f1 = float(np.mean(fold_scores))
        std_fold_f1 = float(np.std(fold_scores))

        # Per-class metrics
        prec, rec, f1_cls, supp = precision_recall_fscore_support(y, oof_preds, labels=range(len(classes)))
        per_class = {
            classes[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1_score": round(float(f1_cls[i]), 4),
                "support": int(supp[i])
            }
            for i in range(len(classes))
        }

        conf_mat = confusion_matrix(y, oof_preds).tolist()

        logger.info(f"Finished {exp_id}: OOF Macro F1 = {overall_macro_f1:.4f} (Mean Folds = {mean_fold_f1:.4f} ± {std_fold_f1:.4f}), Duration = {dur:.2f}s")

        oof_predictions_dict[exp_id] = oof_preds

        results_table.append({
            "experiment_id": exp_id,
            "model": model_type,
            "feature_set": feat_name,
            "C": C_val,
            "class_weight": str(cw_val),
            "oof_macro_f1": round(float(overall_macro_f1), 4),
            "oof_accuracy": round(float(overall_acc), 4),
            "fold_f1_mean": round(mean_fold_f1, 4),
            "fold_f1_std": round(std_fold_f1, 4),
            "f1_14AC78": per_class["14AC78"]["f1_score"],
            "f1_19E1A1": per_class["19E1A1"]["f1_score"],
            "f1_6C0197": per_class["6C0197"]["f1_score"],
            "f1_72A2C8": per_class["72A2C8"]["f1_score"],
            "duration_sec": round(dur, 2)
        })

        detailed_reports[exp_id] = {
            "config": exp,
            "overall_oof_macro_f1": round(float(overall_macro_f1), 4),
            "overall_oof_accuracy": round(float(overall_acc), 4),
            "fold_scores": [round(float(s), 4) for s in fold_scores],
            "fold_f1_mean": round(mean_fold_f1, 4),
            "fold_f1_std": round(std_fold_f1, 4),
            "per_class_metrics": per_class,
            "confusion_matrix": conf_mat,
            "duration_seconds": round(dur, 2)
        }

    # Rank results descending by OOF Macro F1
    df_results = pd.DataFrame(results_table).sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)
    csv_path = exp_dir / "experiment_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Ranked experiment results saved to: {csv_path.resolve()}")

    best_exp_row = df_results.iloc[0]
    best_exp_id = best_exp_row["experiment_id"]
    logger.info("=" * 60)
    logger.info(f"TOP MODEL IDENTIFIED: {best_exp_id}")
    logger.info(f"OOF Macro F1 = {best_exp_row['oof_macro_f1']:.4f} (Accuracy = {best_exp_row['oof_accuracy']:.4f})")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # In-Depth Error Analysis (Focusing on 19E1A1 vs 72A2C8)
    # ---------------------------------------------------------
    logger.info("Executing focused error analysis on best model and baseline...")
    best_oof_preds = oof_predictions_dict[best_exp_id]
    baseline_oof_preds = oof_predictions_dict["EXP03_LogReg_Comb_C1.0_Balanced"]

    idx_19E1A1 = label2idx["19E1A1"]
    idx_72A2C8 = label2idx["72A2C8"]

    # Samples where true is 19E1A1 but predicted is 72A2C8
    confused_19_as_72 = np.where((y == idx_19E1A1) & (best_oof_preds == idx_72A2C8))[0]
    # Samples where true is 72A2C8 but predicted is 19E1A1
    confused_72_as_19 = np.where((y == idx_72A2C8) & (best_oof_preds == idx_19E1A1))[0]

    logger.info(f"Best model confusion: True 19E1A1 -> Pred 72A2C8: {len(confused_19_as_72)} cases")
    logger.info(f"Best model confusion: True 72A2C8 -> Pred 19E1A1: {len(confused_72_as_19)} cases")

    def inspect_samples(indices: np.ndarray, max_n: int = 5) -> List[Dict[str, Any]]:
        extracted = []
        for i in indices[:max_n]:
            extracted.append({
                "id": train_df.loc[i, "id"],
                "text": train_df.loc[i, "text"],
                "char_length": len(train_df.loc[i, "text"]),
                "word_count": len(train_df.loc[i, "text"].split())
            })
        return extracted

    # Extract top overlapping tokens driving confusion
    token_pattern = re.compile(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b")
    tokens_19_as_72 = Counter([t.lower() for i in confused_19_as_72 for t in token_pattern.findall(train_df.loc[i, "text"])])
    tokens_72_as_19 = Counter([t.lower() for i in confused_72_as_19 for t in token_pattern.findall(train_df.loc[i, "text"])])

    common_shared_confusion_tokens = [
        {"token": tok, "freq_in_19_as_72": tokens_19_as_72[tok], "freq_in_72_as_19": tokens_72_as_19[tok]}
        for tok in set(tokens_19_as_72.keys()).intersection(set(tokens_72_as_19.keys()))
        if tokens_19_as_72[tok] >= 5 and tokens_72_as_19[tok] >= 5
    ]
    common_shared_confusion_tokens.sort(key=lambda x: x["freq_in_19_as_72"] + x["freq_in_72_as_19"], reverse=True)

    error_analysis_report = {
        "best_model_id": best_exp_id,
        "baseline_model_id": "EXP03_LogReg_Comb_C1.0_Balanced",
        "confusion_counts": {
            "best_model": {
                "true_19E1A1_pred_72A2C8": len(confused_19_as_72),
                "true_72A2C8_pred_19E1A1": len(confused_72_as_19),
                "total_19_72_confusion": len(confused_19_as_72) + len(confused_72_as_19)
            },
            "baseline_model": {
                "true_19E1A1_pred_72A2C8": int(detailed_reports["EXP03_LogReg_Comb_C1.0_Balanced"]["confusion_matrix"][1][3]),
                "true_72A2C8_pred_19E1A1": int(detailed_reports["EXP03_LogReg_Comb_C1.0_Balanced"]["confusion_matrix"][3][1]),
                "total_19_72_confusion": int(detailed_reports["EXP03_LogReg_Comb_C1.0_Balanced"]["confusion_matrix"][1][3]) +
                                         int(detailed_reports["EXP03_LogReg_Comb_C1.0_Balanced"]["confusion_matrix"][3][1])
            }
        },
        "shared_confused_vocabulary": common_shared_confusion_tokens[:20],
        "representative_cases": {
            "true_19E1A1_predicted_as_72A2C8": inspect_samples(confused_19_as_72, 5),
            "true_72A2C8_predicted_as_19E1A1": inspect_samples(confused_72_as_19, 5)
        }
    }

    with open(reports_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis_report, f, indent=2, cls=NumpyEncoder)

    # ---------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------
    # 1. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    top_plot_df = df_results.sort_values("oof_macro_f1", ascending=True)
    colors = ["#2b5c8f" if exp_id != best_exp_id and exp_id != "EXP03_LogReg_Comb_C1.0_Balanced"
              else ("#d9534f" if exp_id == "EXP03_LogReg_Comb_C1.0_Balanced" else "#5cb85c")
              for exp_id in top_plot_df["experiment_id"]]
    bars = ax.barh(top_plot_df["experiment_id"], top_plot_df["oof_macro_f1"], color=colors, height=0.65, edgecolor="#222222")
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.4f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontweight='bold', fontsize=9)
    ax.set_xlim(0.60, max(top_plot_df["oof_macro_f1"]) + 0.03)
    ax.set_title("BhasaQuest-1 Phase 2: OOF Macro F1 Across Model Configurations", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Out-Of-Fold Macro F1 Score (5-Fold Stratified CV)", fontweight='bold')
    plt.tight_layout()
    p_comp = figures_dir / "model_comparison_macro_f1.png"
    plt.savefig(p_comp, bbox_inches="tight")
    plt.close()

    # 2. Best Model Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    best_cm = np.array(detailed_reports[best_exp_id]["confusion_matrix"])
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, cbar=True, ax=ax)
    ax.set_title(f"Best Model Confusion Matrix ({best_exp_id})\nMacro F1 = {best_exp_row['oof_macro_f1']:.4f}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Class", fontweight='bold')
    ax.set_ylabel("True Class", fontweight='bold')
    plt.tight_layout()
    p_cm = figures_dir / "best_model_confusion_matrix.png"
    plt.savefig(p_cm, bbox_inches="tight")
    plt.close()

    # 3. Focused 19E1A1 vs 72A2C8 Confusion Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    models_compare = ["Baseline (LogReg C=1.0)", f"Best Model ({best_exp_id})"]
    true19_pred72 = [
        error_analysis_report["confusion_counts"]["baseline_model"]["true_19E1A1_pred_72A2C8"],
        error_analysis_report["confusion_counts"]["best_model"]["true_19E1A1_pred_72A2C8"]
    ]
    true72_pred19 = [
        error_analysis_report["confusion_counts"]["baseline_model"]["true_72A2C8_pred_19E1A1"],
        error_analysis_report["confusion_counts"]["best_model"]["true_72A2C8_pred_19E1A1"]
    ]
    x_pos = np.arange(len(models_compare))
    width = 0.35
    ax.bar(x_pos - width/2, true19_pred72, width, label="True 19E1A1 -> Pred 72A2C8", color="#e74c3c", edgecolor="#222222")
    ax.bar(x_pos + width/2, true72_pred19, width, label="True 72A2C8 -> Pred 19E1A1", color="#e67e22", edgecolor="#222222")
    for i in x_pos:
        ax.annotate(f"{true19_pred72[i]}", xy=(i - width/2, true19_pred72[i]/2), ha='center', va='center', color='white', fontweight='bold')
        ax.annotate(f"{true72_pred19[i]}", xy=(i + width/2, true72_pred19[i]/2), ha='center', va='center', color='white', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_compare, fontweight='bold')
    ax.set_ylabel("Number of Misclassified Samples", fontweight='bold')
    ax.set_title("Cross-Confusion Between 19E1A1 (Harm) & 72A2C8 (Injection)", fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    p_conf19_72 = figures_dir / "confusion_19E1A1_vs_72A2C8.png"
    plt.savefig(p_conf19_72, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # Refit Selected Best Model on 100% Labeled Training Data
    # ---------------------------------------------------------
    logger.info("Refitting selected best model on 100% of labeled training data...")
    best_config = detailed_reports[best_exp_id]["config"]
    best_feat = best_config["features"]
    best_model_name = best_config["model"]
    best_C = best_config["C"]
    best_cw = best_config["class_weight"]

    # Full training text and test text
    test_texts = [normalize_text(t) for t in test_df["text"]]

    # Build and fit full vectorizers
    wv_full = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b"
    )
    cv_full = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=35000,
        min_df=3,
        sublinear_tf=True
    )

    if best_feat == "word":
        X_tr_full = wv_full.fit_transform(texts)
        X_te_full = wv_full.transform(test_texts)
        joblib.dump(wv_full, models_dir / "best_word_vectorizer.joblib")
    elif best_feat == "char":
        X_tr_full = cv_full.fit_transform(texts)
        X_te_full = cv_full.transform(test_texts)
        joblib.dump(cv_full, models_dir / "best_char_vectorizer.joblib")
    elif best_feat == "comb":
        X_tr_w = wv_full.fit_transform(texts)
        X_te_w = wv_full.transform(test_texts)
        X_tr_c = cv_full.fit_transform(texts)
        X_te_c = cv_full.transform(test_texts)
        X_tr_full = sparse.hstack([X_tr_w, X_tr_c], format="csr")
        X_te_full = sparse.hstack([X_te_w, X_te_c], format="csr")
        joblib.dump(wv_full, models_dir / "best_word_vectorizer.joblib")
        joblib.dump(cv_full, models_dir / "best_char_vectorizer.joblib")

    # Fit final classifier
    if best_model_name == "LogisticRegression":
        final_best_clf = LogisticRegression(
            C=best_C,
            class_weight=best_cw,
            max_iter=1500,
            solver="lbfgs",
            random_state=seed
        )
    else:
        final_best_clf = LinearSVC(
            C=best_C,
            class_weight=best_cw,
            max_iter=3000,
            random_state=seed
        )

    final_best_clf.fit(X_tr_full, y)
    joblib.dump(final_best_clf, models_dir / "best_model_checkpoint.joblib")

    # Predict test samples
    test_preds_idx = final_best_clf.predict(X_te_full)
    test_preds_labels = [idx2label[i] for i in test_preds_idx]

    # Save to Submission/submission_experiment_best.csv
    best_sub_df = pd.DataFrame({
        "id": test_df["id"],
        "label": test_preds_labels
    })
    best_sub_file = sub_dir / "submission_experiment_best.csv"
    best_sub_df.to_csv(best_sub_file, index=False)
    logger.info(f"Saved best experiment submission to: {best_sub_file.resolve()}")

    # Run full submission validation checks
    audit_results = audit_submission_file(best_sub_file, data_dir / "test.csv", classes, logger)

    # Master Experiment Report JSON
    master_report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_model_id": "EXP03_LogReg_Comb_C1.0_Balanced",
        "baseline_macro_f1": float(df_results.loc[df_results["experiment_id"] == "EXP03_LogReg_Comb_C1.0_Balanced", "oof_macro_f1"].iloc[0]),
        "best_model_id": best_exp_id,
        "best_macro_f1": float(best_exp_row["oof_macro_f1"]),
        "macro_f1_improvement": round(float(best_exp_row["oof_macro_f1"]) - float(df_results.loc[df_results["experiment_id"] == "EXP03_LogReg_Comb_C1.0_Balanced", "oof_macro_f1"].iloc[0]), 4),
        "total_experiments_evaluated": len(configs),
        "rankings": df_results.to_dict(orient="records"),
        "detailed_experiments": detailed_reports,
        "error_analysis": error_analysis_report,
        "submission_audit": audit_results
    }

    with open(exp_dir / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2, cls=NumpyEncoder)

    logger.info("Controlled experiment suite completed successfully.")
    return df_results, master_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Controlled NLP Model Experiments")
    parser.add_argument("--data_dir", type=str, default="Datasets")
    parser.add_argument("--exp_dir", type=str, default="Experiments")
    parser.add_argument("--sub_dir", type=str, default="Submission")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    d_path = Path(args.data_dir)
    if not d_path.is_absolute():
        d_path = (project_root / d_path).resolve()

    e_path = Path(args.exp_dir)
    if not e_path.is_absolute():
        e_path = (project_root / e_path).resolve()

    s_path = Path(args.sub_dir)
    if not s_path.is_absolute():
        s_path = (project_root / s_path).resolve()

    run_experiment_suite(d_path, e_path, s_path, seed=args.seed)
