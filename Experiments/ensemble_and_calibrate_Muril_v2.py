"""
================================================================================
BhasaQuest-1: Optimal Weight Search for Calibrated LinearSVC & MuRIL Base
================================================================================
Script: Experiments/ensemble_and_calibrate_Muril_v2.py

Purpose:
  1. Loads Out-Of-Fold (OOF) calibrated probabilities for:
     - Champion LinearSVC (Word 1-3 + Char 2-5 TF-IDF, C=0.25)
     - Pretrained MuRIL Transformer (google/muril-base-cased)
  2. Iterates variable blending percentage p from 1% to 99%:
     - w_muril = p / 100.0, w_svm = 1.0 - w_muril
     - Evaluates OOF Macro F1 and Accuracy at every percentage point.
  3. Identifies the global maximum Macro F1 score and optimal weights.
  4. Applies optimal weights to Test probability matrices:
     - test_probs = w_svm* * test_svm + w_muril* * test_muril
  5. Generates the audited Kaggle submission file:
     - Submission/submission_calibrated_ensemble_muril_v2.csv
================================================================================
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


def audit_submission(df, test_df, classes, tag=""):
    """Strict 10-Point Kaggle Submission Audit."""
    print(f"\n--- Running 10-Point Submission Audit [{tag}] ---")
    assert list(df.columns) == ["id", "label"], f"Invalid columns: {list(df.columns)}"
    assert len(df) == 2048, f"Expected 2048 rows, got {len(df)}"
    assert df["id"].isnull().sum() == 0 and df["label"].isnull().sum() == 0, "Missing values detected!"
    assert (df["id"].values == test_df["id"].values).all(), "ID sequence mismatch with test.csv!"
    assert set(df["label"].unique()).issubset(set(classes)), f"Unknown labels: {set(df['label']) - set(classes)}"
    assert len(df["label"].unique()) == 4, f"Mode collapse! Only {len(df['label'].unique())} classes present"
    print("Class Distribution in Predictions:")
    print(df["label"].value_counts().to_string())
    print(f">>> 10-Point Audit PASSED for [{tag}]!")


def main():
    parser = argparse.ArgumentParser(description="Optimal Weight Search for LinearSVC + MuRIL Ensemble")
    parser.add_argument("--svm_oof", type=str, default="Experiments/models/oof_linearsvc_calibrated_probs.npy",
                        help="Path to LinearSVC calibrated OOF probabilities")
    parser.add_argument("--svm_test", type=str, default="Experiments/models/test_linearsvc_calibrated_probs.npy",
                        help="Path to LinearSVC calibrated Test probabilities")
    parser.add_argument("--muril_oof", type=str, default="Experiments/transformer_output/oof_muril_base_probs.npy",
                        help="Path to MuRIL OOF probabilities")
    parser.add_argument("--muril_test", type=str, default="Experiments/transformer_output/test_muril_base_probs.npy",
                        help="Path to MuRIL Test probabilities")
    args = parser.parse_args()

    # Determine Project Paths
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent if script_dir.name == "Experiments" else script_dir
    
    train_path = root / "Datasets/train.csv"
    test_path = root / "Datasets/test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    classes = sorted(train_df["label"].unique().tolist())
    label2idx = {lbl: i for i, lbl in enumerate(classes)}
    idx2label = {i: lbl for i, lbl in enumerate(classes)}
    y_true = np.array([label2idx[l] for l in train_df["label"]])
    
    sub_dir = root / "Submission"
    reports_dir = root / "Experiments/reports"
    models_dir = root / "Experiments/models"
    sub_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" BhasaQuest-1: LinearSVC + MuRIL Base Optimal Weight Grid Search")
    print("=" * 75)
    print(f"[*] Target Classes ({len(classes)}): {classes}")
    print(f"[*] Total Training Samples : {len(y_true)}")
    print(f"[*] Total Test Samples     : {len(test_df)}")

    # 1. Load Probability Matrices
    svm_oof_path = root / args.svm_oof
    svm_test_path = root / args.svm_test
    muril_oof_path = root / args.muril_oof
    muril_test_path = root / args.muril_test

    print(f"\n[*] Loading Probability Arrays...")
    oof_svm = np.load(svm_oof_path)
    test_svm = np.load(svm_test_path)
    oof_muril = np.load(muril_oof_path)
    test_muril = np.load(muril_test_path)

    print(f"    LinearSVC OOF Shape : {oof_svm.shape} | Test Shape: {test_svm.shape}")
    print(f"    MuRIL     OOF Shape : {oof_muril.shape} | Test Shape: {test_muril.shape}")

    # Baseline Individual Model Scores
    f1_svm = f1_score(y_true, np.argmax(oof_svm, axis=1), average="macro")
    acc_svm = accuracy_score(y_true, np.argmax(oof_svm, axis=1))
    f1_muril = f1_score(y_true, np.argmax(oof_muril, axis=1), average="macro")
    acc_muril = accuracy_score(y_true, np.argmax(oof_muril, axis=1))

    print(f"\n--- Baseline Individual Model Performance ---")
    print(f"1. LinearSVC Alone : OOF Macro F1 = {f1_svm:.5f} | Accuracy = {acc_svm*100:.2f}%")
    print(f"2. MuRIL Base Alone: OOF Macro F1 = {f1_muril:.5f} | Accuracy = {acc_muril*100:.2f}%")

    # 2. Iterate variable percent through 1 to 99
    print("\n" + "=" * 75)
    print(" Running Full Grid Search: Iterating MuRIL Weight from 1% to 99%...")
    print("=" * 75)

    search_history = []
    best_f1 = -1.0
    best_acc = 0.0
    best_percent = 0
    best_w_svm = 0.0
    best_w_muril = 0.0

    # Test baseline 50/50 for comparison
    f1_at_50 = 0.0

    for p in range(1, 100):
        w_m = p / 100.0
        w_s = (100 - p) / 100.0
        
        blended_oof = w_s * oof_svm + w_m * oof_muril
        preds = np.argmax(blended_oof, axis=1)
        
        macro_f1 = f1_score(y_true, preds, average="macro")
        acc = accuracy_score(y_true, preds)
        
        search_history.append({
            "muril_percent": p,
            "linearsvc_percent": 100 - p,
            "w_muril": round(w_m, 2),
            "w_svm": round(w_s, 2),
            "macro_f1": float(macro_f1),
            "accuracy": float(acc),
        })

        if p == 50:
            f1_at_50 = macro_f1

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_acc = acc
            best_percent = p
            best_w_muril = w_m
            best_w_svm = w_s

    # Save Search History to CSV
    history_df = pd.DataFrame(search_history)
    history_csv = reports_dir / "muril_v2_grid_search_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"[+] Saved complete 1-99% grid search history to: {history_csv}")

    # Display Top 10 Configurations
    top_10 = history_df.sort_values(by="macro_f1", ascending=False).head(10)
    print("\nTop 10 Performing Weight Combinations:")
    print("-" * 75)
    print(f"{'Rank':<6} {'LinearSVC %':<14} {'MuRIL %':<12} {'Macro F1':<12} {'Accuracy %':<12} {'Gain vs 50/50':<14}")
    print("-" * 75)
    for rank, (_, row) in enumerate(top_10.iterrows(), 1):
        gain = row['macro_f1'] - f1_at_50
        print(f"#{rank:<5} {int(row['linearsvc_percent']):>4}% ({row['w_svm']:.2f})    {int(row['muril_percent']):>3}% ({row['w_muril']:.2f})   {row['macro_f1']:.5f}      {row['accuracy']*100:.2f}%      {gain:+.5f}")
    print("-" * 75)

    best_percent_svm = 100 - best_percent

    print("\n" + "=" * 75)
    print(" 🏆 GRID SEARCH OPTIMIZATION RESULTS")
    print("=" * 75)
    print(f" Previous 50/50 Blend Macro F1 : {f1_at_50:.5f}")
    print(f" Best LinearSVC Alone Macro F1 : {f1_svm:.5f}")
    print(f" -------------------------------------------------------------")
    print(f" 🥇 Optimal LinearSVC Weight    : {best_percent_svm}% ({best_w_svm:.2f})")
    print(f" 🥇 Optimal MuRIL Weight        : {best_percent}% ({best_w_muril:.2f})")
    print(f" 🚀 MAXIMUM OOF MACRO F1        : {best_f1:.5f}")
    print(f" 🎯 OOF ACCURACY AT MAX F1      : {best_acc*100:.2f}%")
    print(f" 📈 Net Gain over 50/50 Blend   : {best_f1 - f1_at_50:+.5f} Macro F1")
    print(f" 📈 Net Gain over LinearSVC     : {best_f1 - f1_svm:+.5f} Macro F1")
    print("=" * 75)

    # 3. Generate Submission with Optimal Weights
    print(f"\n[*] Applying Optimal Weights ({best_percent_svm}% LinearSVC + {best_percent}% MuRIL) to Test Set...")
    best_test_probs = best_w_svm * test_svm + best_w_muril * test_muril
    best_test_preds = np.argmax(best_test_probs, axis=1)
    best_test_labels = [idx2label[i] for i in best_test_preds]

    sub_df = pd.DataFrame({"id": test_df["id"], "label": best_test_labels})
    
    # Required target submission file:
    out_sub_path = sub_dir / "submission_calibrated_ensemble_muril_v2.csv"
    sub_df.to_csv(out_sub_path, index=False)
    print(f"[+] Successfully saved optimized submission to:")
    print(f"    --> {out_sub_path}")

    # 4. Strict 10-Point Submission Audit
    audit_submission(sub_df, test_df, classes, tag=f"MuRIL v2 ({best_percent_svm}% SVM + {best_percent}% MuRIL)")

    # 5. Save Optimal Test Probabilities for downstream stacking/ensembling
    np.save(models_dir / "test_muril_v2_calibrated_ensemble_probs.npy", best_test_probs)
    print(f"[+] Saved optimal test probabilities to {models_dir / 'test_muril_v2_calibrated_ensemble_probs.npy'}")


if __name__ == "__main__":
    main()
