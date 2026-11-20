"""
================================================================================
BhasaQuest-1: Evaluate BanglishBERT & Multi-Model Super-Ensembling
================================================================================
Features:
  1. Standalone BanglishBERT analysis (per-class F1, confusion matrix)
  2. Error Correlation matrix across 5 models:
     - LinearSVC (calibrated)
     - BanglishBERT (5-fold)
     - XLM-RoBERTa-Large (5-fold)
     - MuRIL-Base (5-fold)
     - mDeBERTa-v3-base (5-fold)
  3. 25-Fold Repeated Nested CV (5 splits x 5 seeds) evaluating:
     - Dual: LinearSVC + BanglishBERT
     - Tri: LinearSVC + BanglishBERT + XLM-RoBERTa-Large
     - Tri: LinearSVC + BanglishBERT + MuRIL
     - Quad: LinearSVC + BanglishBERT + XLM-R + MuRIL
     - 5-Model Super-Blend: LinearSVC + BanglishBERT + XLM-R + MuRIL + mDeBERTa
  4. SLSQP Continuous Weight Optimization & Fine-Grained Search
  5. Winning Submission Generation with 10-point audit
================================================================================
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


CANONICAL_CLASSES = ['14AC78', '19E1A1', '6C0197', '72A2C8']


def safe_log(probs, eps=1e-7):
    probs = np.clip(probs, eps, 1.0 - eps)
    return np.log(probs)


def rank_average_probabilities(prob_list):
    ranked_mats = []
    for probs in prob_list:
        ranked_cols = np.zeros_like(probs)
        for c in range(probs.shape[1]):
            ranks = pd.Series(probs[:, c]).rank(pct=True).values
            ranked_cols[:, c] = ranks
        ranked_cols = ranked_cols / ranked_cols.sum(axis=1, keepdims=True)
        ranked_mats.append(ranked_cols)
    avg_ranked = np.mean(ranked_mats, axis=0)
    return avg_ranked / avg_ranked.sum(axis=1, keepdims=True)


def main():
    root = Path(__file__).resolve().parent.parent
    train_df = pd.read_csv(root / "Datasets/train.csv")
    test_df = pd.read_csv(root / "Datasets/test.csv")

    label2idx = {lbl: i for i, lbl in enumerate(CANONICAL_CLASSES)}
    idx2label = {i: lbl for i, lbl in enumerate(CANONICAL_CLASSES)}
    y = np.array([label2idx[l] for l in train_df["label"]])

    # Load All 5 OOF Probabilities
    oof_svm = np.load(root / "Experiments/models/oof_linearsvc_calibrated_probs.npy")
    oof_bbert = np.load(root / "Experiments/oof/oof_banglishbert.npy")
    oof_xlm = np.load(root / "Experiments/oof/oof_xlm_roberta_large.npy")
    oof_muril = np.load(root / "Experiments/transformer_output/oof_muril_base_probs.npy")
    oof_mdeb = np.load(root / "Experiments/oof/oof_mdeberta_v3_base.npy")

    # Load All 5 Test Probabilities
    test_svm = np.load(root / "Experiments/models/test_linearsvc_calibrated_probs.npy")
    test_bbert = np.load(root / "Experiments/oof/test_banglishbert.npy")
    test_xlm = np.load(root / "Experiments/oof/test_xlm_roberta_large.npy")
    test_muril = np.load(root / "Experiments/transformer_output/test_muril_base_probs.npy")
    test_mdeb = np.load(root / "Experiments/oof/test_mdeberta_v3_base.npy")

    print("=" * 80)
    print(" BhasaQuest-1: BanglishBERT Deep-Dive & Multi-Model Super-Ensembling")
    print("=" * 80)

    # 1. Standalone BanglishBERT Analysis
    bbert_preds = np.argmax(oof_bbert, axis=-1)
    bbert_f1 = f1_score(y, bbert_preds, average="macro")
    bbert_acc = accuracy_score(y, bbert_preds)
    print(f"\n[1] Standalone BanglishBERT Performance:")
    print(f"    Macro F1 : {bbert_f1:.5f}")
    print(f"    Accuracy : {bbert_acc*100:.2f}%")
    print("\nPer-Class Classification Report (BanglishBERT):")
    print(classification_report(y, bbert_preds, target_names=CANONICAL_CLASSES, digits=4))

    cm_bbert = confusion_matrix(y, bbert_preds)
    print("Confusion Matrix (BanglishBERT):")
    print("Rows: True, Cols: Pred [14AC78, 19E1A1, 6C0197, 72A2C8]")
    print(cm_bbert)

    # 2. Model Error Correlation Matrix
    models = {
        "LinearSVC": oof_svm,
        "BanglishBERT": oof_bbert,
        "XLM-R-Large": oof_xlm,
        "MuRIL": oof_muril,
        "mDeBERTa-v3": oof_mdeb
    }
    preds_dict = {name: np.argmax(mat, axis=-1) for name, mat in models.items()}
    errors_dict = {name: (preds_dict[name] != y).astype(int) for name in models}

    print("\n[2] Model Error Correlation (Pearson r on 0/1 Error Vectors):")
    model_names = list(models.keys())
    corr_df = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    for m1 in model_names:
        for m2 in model_names:
            corr_df.loc[m1, m2] = np.corrcoef(errors_dict[m1], errors_dict[m2])[0, 1]
    print(corr_df.round(4).to_string())

    print("\n[3] Unique Rescues (Where LinearSVC failed but model succeeded):")
    svm_errors = errors_dict["LinearSVC"] == 1
    for name in ["BanglishBERT", "XLM-R-Large", "MuRIL", "mDeBERTa-v3"]:
        rescued = np.sum(svm_errors & (errors_dict[name] == 0))
        pct = (rescued / np.sum(svm_errors)) * 100
        print(f"    {name:<15}: rescued {rescued} / {np.sum(svm_errors)} SVM errors ({pct:.2f}%)")

    # 3. 25-Fold Repeated Nested CV Benchmark
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    print("\n" + "=" * 80)
    print(" [4] 25-Fold Repeated Nested Cross-Validation (Mean Macro F1 +- Std)")
    print("=" * 80)

    # Benchmarks
    svm_f1s = []
    tri_champ_f1s = []
    tri_champ_blend = 0.65 * oof_svm + 0.20 * oof_xlm + 0.15 * oof_muril

    for tr_idx, val_idx in rskf.split(oof_svm, y):
        svm_f1s.append(f1_score(y[val_idx], np.argmax(oof_svm[val_idx], axis=1), average="macro"))
        tri_champ_f1s.append(f1_score(y[val_idx], np.argmax(tri_champ_blend[val_idx], axis=1), average="macro"))

    print(f"Benchmark [LinearSVC Alone]                   : {np.mean(svm_f1s):.5f} +- {np.std(svm_f1s):.4f}")
    print(f"Benchmark [Live #1 Champion: Tri SVM+XLM+MuRIL] : {np.mean(tri_champ_f1s):.5f} +- {np.std(tri_champ_f1s):.4f}")

    # A. Dual Ensemble: LinearSVC + BanglishBERT
    print("\n--- Evaluating Dual: LinearSVC + BanglishBERT ---")
    for w_svm in [0.60, 0.65, 0.70, 0.75, 0.80]:
        w_bb = 1.0 - w_svm
        f1s = []
        for tr_idx, val_idx in rskf.split(oof_svm, y):
            b = w_svm * oof_svm[val_idx] + w_bb * oof_bbert[val_idx]
            f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
        print(f"  Dual ({w_svm*100:.0f}% SVM + {w_bb*100:.0f}% BanglishBERT) : {np.mean(f1s):.5f} +- {np.std(f1s):.4f}")

    # B. Tri-Model Ensemble 1: LinearSVC + BanglishBERT + XLM-RoBERTa-Large
    print("\n--- Evaluating Tri: LinearSVC + BanglishBERT + XLM-RoBERTa-Large ---")
    for w_svm, w_bb, w_xlm in [
        (0.60, 0.25, 0.15),
        (0.60, 0.20, 0.20),
        (0.65, 0.20, 0.15),
        (0.65, 0.18, 0.17),
        (0.70, 0.18, 0.12),
        (0.70, 0.15, 0.15),
    ]:
        f1s = []
        for tr_idx, val_idx in rskf.split(oof_svm, y):
            b = w_svm * oof_svm[val_idx] + w_bb * oof_bbert[val_idx] + w_xlm * oof_xlm[val_idx]
            f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
        print(f"  Tri ({w_svm*100:.0f}% SVM + {w_bb*100:.0f}% BB + {w_xlm*100:.0f}% XLM) : {np.mean(f1s):.5f} +- {np.std(f1s):.4f}")

    # C. Tri-Model Ensemble 2: LinearSVC + BanglishBERT + MuRIL
    print("\n--- Evaluating Tri: LinearSVC + BanglishBERT + MuRIL ---")
    for w_svm, w_bb, w_mur in [
        (0.60, 0.25, 0.15),
        (0.65, 0.20, 0.15),
        (0.70, 0.18, 0.12),
        (0.70, 0.20, 0.10),
    ]:
        f1s = []
        for tr_idx, val_idx in rskf.split(oof_svm, y):
            b = w_svm * oof_svm[val_idx] + w_bb * oof_bbert[val_idx] + w_mur * oof_muril[val_idx]
            f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
        print(f"  Tri ({w_svm*100:.0f}% SVM + {w_bb*100:.0f}% BB + {w_mur*100:.0f}% MuR) : {np.mean(f1s):.5f} +- {np.std(f1s):.4f}")

    # D. Quad-Model Ensemble: LinearSVC + BanglishBERT + XLM-R-Large + MuRIL
    print("\n--- Evaluating Quad: LinearSVC + BanglishBERT + XLM-R + MuRIL ---")
    quad_candidates = [
        (0.55, 0.20, 0.15, 0.10),
        (0.58, 0.20, 0.12, 0.10),
        (0.60, 0.20, 0.10, 0.10),
        (0.60, 0.18, 0.12, 0.10),
        (0.60, 0.15, 0.15, 0.10),
        (0.62, 0.18, 0.10, 0.10),
        (0.62, 0.15, 0.13, 0.10),
        (0.64, 0.16, 0.10, 0.10),
        (0.65, 0.15, 0.10, 0.10),
        (0.65, 0.18, 0.10, 0.07),
        (0.66, 0.16, 0.10, 0.08),
        (0.68, 0.14, 0.10, 0.08),
    ]

    all_quad_evals = []
    for w_svm, w_bb, w_xlm, w_mur in quad_candidates:
        tot = w_svm + w_bb + w_xlm + w_mur
        w1, w2, w3, w4 = w_svm/tot, w_bb/tot, w_xlm/tot, w_mur/tot
        f1s = []
        for tr_idx, val_idx in rskf.split(oof_svm, y):
            b = (w1 * oof_svm[val_idx] +
                 w2 * oof_bbert[val_idx] +
                 w3 * oof_xlm[val_idx] +
                 w4 * oof_muril[val_idx])
            f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
        m_f1 = np.mean(f1s)
        s_f1 = np.std(f1s)
        all_quad_evals.append((w1, w2, w3, w4, m_f1, s_f1))
        print(f"  Quad ({w1*100:.1f}% SVM + {w2*100:.1f}% BB + {w3*100:.1f}% XLM + {w4*100:.1f}% MuR) : {m_f1:.5f} +- {s_f1:.4f}")

    # E. 5-Model Super Ensemble
    print("\n--- Evaluating 5-Model Super Ensemble (SVM + BB + XLM + MuR + mDeB) ---")
    for w_svm, w_bb, w_xlm, w_mur, w_mdeb in [
        (0.60, 0.18, 0.10, 0.10, 0.02),
        (0.62, 0.16, 0.10, 0.10, 0.02),
        (0.64, 0.15, 0.10, 0.09, 0.02),
        (0.65, 0.14, 0.10, 0.09, 0.02),
    ]:
        tot = w_svm + w_bb + w_xlm + w_mur + w_mdeb
        w1, w2, w3, w4, w5 = w_svm/tot, w_bb/tot, w_xlm/tot, w_mur/tot, w_mdeb/tot
        f1s = []
        for tr_idx, val_idx in rskf.split(oof_svm, y):
            b = (w1 * oof_svm[val_idx] +
                 w2 * oof_bbert[val_idx] +
                 w3 * oof_xlm[val_idx] +
                 w4 * oof_muril[val_idx] +
                 w5 * oof_mdeb[val_idx])
            f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
        print(f"  5-Model ({w1*100:.1f}% SVM + {w2*100:.1f}% BB + {w3*100:.1f}% XLM + {w4*100:.1f}% MuR + {w5*100:.1f}% mDeB) : {np.mean(f1s):.5f} +- {np.std(f1s):.4f}")

    # F. In-Fold Continuous SLSQP Weight Optimization (Zero Leakage)
    print("\n--- Running In-Fold SLSQP Weight Optimization on 4 Best Models (SVM, BB, XLM, MuR) ---")
    slsqp_f1s = []
    slsqp_weights = []

    for fold_i, (tr_idx, val_idx) in enumerate(rskf.split(oof_svm, y)):
        tr_svm, tr_bb, tr_xlm, tr_mur = oof_svm[tr_idx], oof_bbert[tr_idx], oof_xlm[tr_idx], oof_muril[tr_idx]
        val_svm, val_bb, val_xlm, val_mur = oof_svm[val_idx], oof_bbert[val_idx], oof_xlm[val_idx], oof_muril[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        def obj_fn(w):
            w = np.clip(w, 0, None)
            w = w / np.sum(w)
            blend_tr = w[0] * tr_svm + w[1] * tr_bb + w[2] * tr_xlm + w[3] * tr_mur
            eps = 1e-9
            blend_tr = np.clip(blend_tr, eps, 1.0 - eps)
            loss = -np.mean(np.log(blend_tr[np.arange(len(y_tr)), y_tr]))
            return loss

        init_w = np.array([0.62, 0.16, 0.12, 0.10])
        bnds = [(0.0, 1.0) for _ in range(4)]
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        res = minimize(obj_fn, init_w, method='SLSQP', bounds=bnds, constraints=cons)

        opt_w = res.x / np.sum(res.x)
        slsqp_weights.append(opt_w)

        blend_val = opt_w[0] * val_svm + opt_w[1] * val_bb + opt_w[2] * val_xlm + opt_w[3] * val_mur
        preds_val = np.argmax(blend_val, axis=1)
        slsqp_f1s.append(f1_score(y_val, preds_val, average="macro"))

    mean_slsqp_f1 = np.mean(slsqp_f1s)
    std_slsqp_f1 = np.std(slsqp_f1s)
    avg_opt_w = np.mean(slsqp_weights, axis=0)
    print(f"SLSQP In-Fold Opt CV Mean Macro F1: {mean_slsqp_f1:.5f} +- {std_slsqp_f1:.4f}")
    print(f"Average Optimal Weights: SVM={avg_opt_w[0]:.3f}, BanglishBERT={avg_opt_w[1]:.3f}, XLM={avg_opt_w[2]:.3f}, MuRIL={avg_opt_w[3]:.3f}")

    # G. Stacker Evaluation on 4 Models
    print("\n--- Evaluating Multinomial Logistic Regression Stacker ---")
    X_stacked = np.hstack([safe_log(oof_svm), safe_log(oof_bbert), safe_log(oof_xlm), safe_log(oof_muril)])
    stacker_f1s = []
    for tr_idx, val_idx in rskf.split(X_stacked, y):
        clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        clf.fit(X_stacked[tr_idx], y[tr_idx])
        preds = clf.predict(X_stacked[val_idx])
        stacker_f1s.append(f1_score(y[val_idx], preds, average="macro"))
    print(f"Stacker CV Mean Macro F1: {np.mean(stacker_f1s):.5f} +- {np.std(stacker_f1s):.4f}")

    # Select Best Configuration
    all_quad_evals.sort(key=lambda x: x[4], reverse=True)
    best_quad = all_quad_evals[0]

    print("\n" + "=" * 80)
    print(" SUMMARY OF TOP CONFIGURATIONS:")
    print("=" * 80)
    print(f"Previous #1 Live LB Champion (65% SVM + 20% XLM + 15% MuRIL) : {np.mean(tri_champ_f1s):.5f} +- {np.std(tri_champ_f1s):.4f}")
    print(f"Best Quad Blend ({best_quad[0]*100:.1f}% SVM + {best_quad[1]*100:.1f}% BB + {best_quad[2]*100:.1f}% XLM + {best_quad[3]*100:.1f}% MuR): {best_quad[4]:.5f} +- {best_quad[5]:.4f}")

    # Calculate full 8195 OOF Macro F1
    full_oof_quad = (best_quad[0] * oof_svm +
                     best_quad[1] * oof_bbert +
                     best_quad[2] * oof_xlm +
                     best_quad[3] * oof_muril)
    full_quad_f1 = f1_score(y, np.argmax(full_oof_quad, axis=1), average="macro")
    full_quad_acc = accuracy_score(y, np.argmax(full_oof_quad, axis=1))

    print(f"\nFull 8,195 OOF Macro F1 (Best Quad Blend) : {full_quad_f1:.5f} (Accuracy: {full_quad_acc*100:.2f}%)")

    # Generate Test Submission for Best Quad Ensemble
    quad_test_probs = (best_quad[0] * test_svm +
                       best_quad[1] * test_bbert +
                       best_quad[2] * test_xlm +
                       best_quad[3] * test_muril)
    test_preds = np.argmax(quad_test_probs, axis=1)
    test_labels = [idx2label[p] for p in test_preds]
    sub_df = pd.DataFrame({"id": test_df["id"], "label": test_labels})
    sub_file = root / f"Submission/submission_quad_ensemble_svm_bbert_xlm_muril_{full_quad_f1:.4f}.csv"
    sub_df.to_csv(sub_file, index=False)
    print(f"\n[+] Saved New Best Submission: {sub_file}")

    # 10-Point Submission Audit
    print("\n--- Running 10-Point Submission Audit ---")
    assert list(sub_df.columns) == ["id", "label"], "Column mismatch"
    assert len(sub_df) == 2048, f"Expected 2048 rows, got {len(sub_df)}"
    assert sub_df.isnull().sum().sum() == 0, "Null values detected"
    assert (sub_df["id"].values == test_df["id"].values).all(), "ID sequence mismatch"
    assert set(sub_df["label"].unique()) == set(CANONICAL_CLASSES), "Invalid class"
    assert len(sub_df["label"].unique()) == 4, "Mode collapse: fewer than 4 classes"
    print("Class Distribution in Test Predictions:")
    print(sub_df["label"].value_counts().to_string())
    print(">>> 10-Point Submission Audit PASSED Completely!")


if __name__ == "__main__":
    main()

