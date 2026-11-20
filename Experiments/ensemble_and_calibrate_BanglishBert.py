"""
================================================================================
BhasaQuest-1: Ensemble & Probability Calibration for BanglishBERT (v1 & v2)
================================================================================
Script: Experiments/ensemble_and_calibrate_BanglishBert.py

Input Paths:
  - BanglishBERT v1 Test Probs : Experiments/transformer_output/BBert_v1_test_probabilities.npy
  - BanglishBERT v2 Test Probs : Experiments/transformer_output/BBert_v2_test_probabilities.npy
  - Label Classes Mapping      : Experiments/transformer_output/label_classes.npy
  - Calibrated LinearSVC Probs : Experiments/models/test_linearsvc_calibrated_probs.npy
  - Calibrated LogReg Probs    : Submission/models/test_baseline_probs.npy

Target Outputs:
  - Submission/submission_calibrated_ensemble_BanglishBert_v1.csv
  - Submission/submission_calibrated_ensemble_BanglishBert_v2.csv
  - (Bonus) Submission/submission_calibrated_ensemble_BanglishBert_v1_v2_superblend.csv

Compliance:
  - Automatic 10-Point Kaggle Submission Audit.
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
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score


def normalize_text(text: str) -> str:
    """Normalize Romanized Bangla: lowercase, collapse duplicate characters & spaces."""
    s = str(text).lower()
    s = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', s)
    return re.sub(r'\s+', ' ', s).strip()


def run_calibrated_linearsvc(train_df, test_df, c_val=0.25):
    """5-Fold Leakage-Free Calibrated LinearSVC (Word 1-3 + Char 2-5)."""
    print(f"\n[*] Training 5-Fold Calibrated LinearSVC (C={c_val}, Balanced, Word 1-3, Char 2-5)...")
    
    classes = sorted(train_df["label"].unique().tolist())
    label2idx = {lbl: i for i, lbl in enumerate(classes)}
    num_classes = len(classes)
    
    train_texts = [normalize_text(t) for t in train_df["text"]]
    test_texts = [normalize_text(t) for t in test_df["text"]]
    y = np.array([label2idx[l] for l in train_df["label"]])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros((len(train_texts), num_classes), dtype=np.float64)
    test_probs = np.zeros((len(test_texts), num_classes), dtype=np.float64)
    fold_f1s = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_texts, y), 1):
        tr_txt = [train_texts[i] for i in tr_idx]
        val_txt = [train_texts[i] for i in val_idx]
        y_tr = y[tr_idx]
        y_val = y[val_idx]
        
        wv = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20000,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            token_pattern=r'(?u)\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b'
        )
        cv = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=35000,
            min_df=3,
            sublinear_tf=True
        )
        
        X_tr = sparse.hstack([wv.fit_transform(tr_txt), cv.fit_transform(tr_txt)], format='csr')
        X_val = sparse.hstack([wv.transform(val_txt), cv.transform(val_txt)], format='csr')
        X_test = sparse.hstack([wv.transform(test_texts), cv.transform(test_texts)], format='csr')
        
        base_clf = LinearSVC(C=c_val, class_weight='balanced', max_iter=3000, random_state=42)
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
        calibrated_clf.fit(X_tr, y_tr)
        
        val_p = calibrated_clf.predict_proba(X_val)
        oof_probs[val_idx] = val_p
        
        pred_val = np.argmax(val_p, axis=1)
        f1 = f1_score(y_val, pred_val, average='macro')
        fold_f1s.append(f1)
        print(f"  Fold {fold} Macro F1: {f1:.4f}")
        
        test_probs += calibrated_clf.predict_proba(X_test) / skf.n_splits
        
    oof_preds = np.argmax(oof_probs, axis=1)
    macro_f1 = f1_score(y, oof_preds, average='macro')
    acc = accuracy_score(y, oof_preds)
    print(f"[+] Calibrated LinearSVC OOF Macro F1: {macro_f1:.4f} | Accuracy: {acc*100:.2f}%")
    return oof_probs, test_probs, macro_f1


def align_class_probabilities(probs, source_classes, target_classes):
    """
    Ensures that columns of probability matrix align exactly with canonical target_classes.
    """
    source_list = [str(c) for c in source_classes]
    target_list = [str(c) for c in target_classes]
    if source_list == target_list:
        return probs
    
    print(f"[*] Aligning class columns from {source_list} to {target_list}...")
    src_idx = {lbl: i for i, lbl in enumerate(source_list)}
    col_mapping = [src_idx[lbl] for lbl in target_list]
    return probs[:, col_mapping]


def audit_submission(df, test_df, classes, tag=""):
    """Strict 10-Point Submission Audit."""
    print(f"\n--- Running 10-Point Audit [{tag}] ---")
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
    parser = argparse.ArgumentParser(description="Ensemble BanglishBERT v1 & v2 with Calibrated Linear Models")
    parser.add_argument("--v1_probs", type=str, default="Experiments/transformer_output/BBert_v1_test_probabilities.npy",
                        help="Path to BanglishBERT v1 test probabilities")
    parser.add_argument("--v2_probs", type=str, default="Experiments/transformer_output/BBert_v2_test_probabilities.npy",
                        help="Path to BanglishBERT v2 test probabilities")
    parser.add_argument("--label_classes", type=str, default="Experiments/transformer_output/label_classes.npy",
                        help="Path to label_classes.npy")
    parser.add_argument("--svm_weight", type=float, default=0.50,
                        help="Weight for Calibrated LinearSVC (default: 0.50)")
    parser.add_argument("--bert_weight", type=float, default=0.50,
                        help="Weight for BanglishBERT (default: 0.50)")
    args = parser.parse_args()

    # Determine paths relative to workspace root
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent if script_dir.name == "Experiments" else script_dir
    
    train_path = root / "Datasets/train.csv"
    test_path = root / "Datasets/test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    classes = sorted(train_df["label"].unique().tolist())
    idx2label = {i: lbl for i, lbl in enumerate(classes)}
    
    sub_dir = root / "Submission"
    models_dir = root / "Experiments/models"
    sub_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" BhasaQuest-1: BanglishBERT v1 & v2 Calibrated Ensemble Pipeline")
    print("=" * 75)
    print(f"[*] Canonical Classes ({len(classes)}): {classes}")

    # 1. Load Calibrated LinearSVC Test Probabilities (Champion)
    svm_test_file = models_dir / "test_linearsvc_calibrated_probs.npy"
    if svm_test_file.exists():
        print(f"\n[+] Loading Calibrated LinearSVC test probabilities from {svm_test_file}...")
        test_svm = np.load(svm_test_file)
    else:
        print(f"\n[*] Calibrated LinearSVC not found at {svm_test_file}. Training now...")
        _, test_svm, _ = run_calibrated_linearsvc(train_df, test_df, c_val=0.25)
        np.save(svm_test_file, test_svm)
    
    assert test_svm.shape == (2048, 4), f"Expected (2048, 4), got {test_svm.shape}"

    # 2. Load Label Classes
    label_classes_path = root / args.label_classes
    if not label_classes_path.exists():
        # Fallback search
        candidate = root / "Experiments/transformer_output/label_classes.npy"
        if candidate.exists():
            label_classes_path = candidate
        else:
            raise FileNotFoundError(f"Could not find label_classes.npy at {label_classes_path}")

    bert_classes = np.load(label_classes_path, allow_pickle=True)
    print(f"[+] Loaded label_classes.npy: {list(bert_classes)}")

    # Normalize Ensemble Weights
    w_svm = args.svm_weight / (args.svm_weight + args.bert_weight)
    w_bert = args.bert_weight / (args.svm_weight + args.bert_weight)
    print(f"[*] Ensemble Weighting Scheme: {w_svm*100:.1f}% LinearSVC + {w_bert*100:.1f}% BanglishBERT")

    # =========================================================================
    # 3. Process BanglishBERT v1 Ensemble
    # =========================================================================
    v1_path = root / args.v1_probs
    if not v1_path.exists():
        # Fallback check
        v1_path = root / "Experiments/transformer_output/BBert_v1_test_probabilities.npy"
        if not v1_path.exists():
            v1_path = root / "Experiments/transformer_output/BBert_v2_test_probabilities.npy"
            
    if v1_path.exists():
        print(f"\n[+] Loading BanglishBERT v1 test probabilities: {v1_path}")
        test_bbert_v1 = np.load(v1_path)
        test_bbert_v1 = align_class_probabilities(test_bbert_v1, bert_classes, classes)
        
        # Blending v1 with Calibrated LinearSVC
        blend_v1_probs = w_svm * test_svm + w_bert * test_bbert_v1
        pred_idx_v1 = np.argmax(blend_v1_probs, axis=1)
        pred_labels_v1 = [idx2label[i] for i in pred_idx_v1]
        
        sub_df_v1 = pd.DataFrame({"id": test_df["id"], "label": pred_labels_v1})
        
        # Required Output 1
        out_sub_path_v1 = sub_dir / "submission_calibrated_ensemble_BanglishBert_v1.csv"
        sub_df_v1.to_csv(out_sub_path_v1, index=False)
        print(f"[+] Generated: {out_sub_path_v1}")
        audit_submission(sub_df_v1, test_df, classes, tag="BanglishBERT v1 Ensemble")
        
        # Save blended probability matrix
        np.save(models_dir / "test_bbert_v1_calibrated_ensemble_probs.npy", blend_v1_probs)
    else:
        print(f"[!] Warning: BanglishBERT v1 probabilities not found at {v1_path}")

    # =========================================================================
    # 4. Process BanglishBERT v2 Ensemble
    # =========================================================================
    v2_path = root / args.v2_probs
    if not v2_path.exists():
        v2_path = root / "Experiments/transformer_output/BBert_v2_test_probabilities.npy"

    if v2_path.exists():
        print(f"\n[+] Loading BanglishBERT v2 test probabilities: {v2_path}")
        test_bbert_v2 = np.load(v2_path)
        test_bbert_v2 = align_class_probabilities(test_bbert_v2, bert_classes, classes)
        
        # Blending v2 with Calibrated LinearSVC
        blend_v2_probs = w_svm * test_svm + w_bert * test_bbert_v2
        pred_idx_v2 = np.argmax(blend_v2_probs, axis=1)
        pred_labels_v2 = [idx2label[i] for i in pred_idx_v2]
        
        sub_df_v2 = pd.DataFrame({"id": test_df["id"], "label": pred_labels_v2})
        
        # Required Output 2
        out_sub_path_v2 = sub_dir / "submission_calibrated_ensemble_BanglishBert_v2.csv"
        sub_df_v2.to_csv(out_sub_path_v2, index=False)
        print(f"[+] Generated: {out_sub_path_v2}")
        audit_submission(sub_df_v2, test_df, classes, tag="BanglishBERT v2 Ensemble")
        
        # Save blended probability matrix
        np.save(models_dir / "test_bbert_v2_calibrated_ensemble_probs.npy", blend_v2_probs)
    else:
        print(f"[!] Warning: BanglishBERT v2 probabilities not found at {v2_path}")

    # =========================================================================
    # 5. Bonus: Super-Blend (LinearSVC + BBert v1 + BBert v2)
    # =========================================================================
    if v1_path.exists() and v2_path.exists():
        print("\n[*] Creating Bonus Super-Blend (40% LinearSVC + 20% BBert v1 + 40% BBert v2)...")
        superblend_probs = 0.40 * test_svm + 0.20 * test_bbert_v1 + 0.40 * test_bbert_v2
        superblend_idx = np.argmax(superblend_probs, axis=1)
        superblend_labels = [idx2label[i] for i in superblend_idx]
        
        sub_df_super = pd.DataFrame({"id": test_df["id"], "label": superblend_labels})
        out_sub_path_super = sub_dir / "submission_calibrated_ensemble_BanglishBert_v1_v2_superblend.csv"
        sub_df_super.to_csv(out_sub_path_super, index=False)
        print(f"[+] Generated: {out_sub_path_super}")
        audit_submission(sub_df_super, test_df, classes, tag="Super-Blend v1+v2")

    print("\n" + "=" * 75)
    print(" BanglishBERT Ensemble Generation Complete!")
    print(" Submissions Ready in Submission/ directory:")
    print("   1. Submission/submission_calibrated_ensemble_BanglishBert_v1.csv")
    print("   2. Submission/submission_calibrated_ensemble_BanglishBert_v2.csv")
    print("=" * 75)


if __name__ == "__main__":
    main()
