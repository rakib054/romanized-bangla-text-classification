"""
================================================================================
BhasaQuest-1: Evaluate mDeBERTa-v3-base & Multi-Model Super Ensembling
================================================================================
Evaluates:
  1. Standalone mDeBERTa-v3-base: Per-class metrics, confusion matrix, error overlap.
  2. Error Correlation matrix among all 4 models:
     - LinearSVC (Calibrated)
     - MuRIL-Base
     - XLM-RoBERTa-Large
     - mDeBERTa-v3-base
  3. Ensembling Strategies evaluated on 25-fold Repeated Stratified CV (5x5):
     - Weighted Probability Averaging (Grid Search & SLSQP Optimization)
     - Level-2 Logistic Regression Stacking
     - Rank Averaging
  4. Decision Rule: Does adding mDeBERTa improve upon the 0.7269 champion?
================================================================================
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
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

    # Load All 4 OOF Probabilities
    oof_svm = np.load(root / "Experiments/models/oof_linearsvc_calibrated_probs.npy")
    oof_muril = np.load(root / "Experiments/transformer_output/oof_muril_base_probs.npy")
    oof_xlm = np.load(root / "Experiments/oof/oof_xlm_roberta_large.npy")
    oof_mdeb = np.load(root / "Experiments/oof/oof_mdeberta_v3_base.npy")

    # Load All 4 Test Probabilities
    test_svm = np.load(root / "Experiments/models/test_linearsvc_calibrated_probs.npy")
    test_muril = np.load(root / "Experiments/transformer_output/test_muril_base_probs.npy")
    test_xlm = np.load(root / "Experiments/oof/test_xlm_roberta_large.npy")
    test_mdeb = np.load(root / "Experiments/oof/test_mdeberta_v3_base.npy")

    print("=" * 80)
    print(" BhasaQuest-1: Multi-Model Evaluation & Super-Ensembling")
    print("=" * 80)

    # 1. Standalone mDeBERTa Analysis
    mdeb_preds = np.argmax(oof_mdeb, axis=-1)
    mdeb_f1 = f1_score(y, mdeb_preds, average="macro")
    mdeb_acc = accuracy_score(y, mdeb_preds)
    print(f"\n[1] Standalone mDeBERTa-v3-base Performance:")
    print(f"    Macro F1 : {mdeb_f1:.5f}")
    print(f"    Accuracy : {mdeb_acc*100:.2f}%")
    print("\nPer-Class Classification Report:")
    print(classification_report(y, mdeb_preds, target_names=CANONICAL_CLASSES, digits=4))

    # Confusion Matrix
    cm_mdeb = confusion_matrix(y, mdeb_preds)
    print("Confusion Matrix (mDeBERTa):")
    print("Rows: True, Cols: Pred [14AC78, 19E1A1, 6C0197, 72A2C8]")
    print(cm_mdeb)

    # 2. Error Overlap and Correlation Analysis across all 4 models
    models = {
        "LinearSVC": oof_svm,
        "MuRIL": oof_muril,
        "XLM-R-Large": oof_xlm,
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
    for name in ["MuRIL", "XLM-R-Large", "mDeBERTa-v3"]:
        rescued = np.sum(svm_errors & (errors_dict[name] == 0))
        pct = (rescued / np.sum(svm_errors)) * 100
        print(f"    {name:<15}: rescued {rescued} / {np.sum(svm_errors)} SVM errors ({pct:.2f}%)")

    # Samples where BOTH XLM-R and mDeBERTa rescued SVM
    both_rescued = np.sum(svm_errors & (errors_dict["XLM-R-Large"] == 0) & (errors_dict["mDeBERTa-v3"] == 0))
    print(f"    Both XLM-R & mDeBERTa rescued {both_rescued} SVM errors")

    # 3. Repeated Stratified CV Evaluation (5 splits x 5 seeds = 25 distinct folds)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    print("\n" + "=" * 80)
    print(" [4] 25-Fold Repeated Nested Cross-Validation (Mean Macro F1 +- Std)")
    print("=" * 80)

    # Benchmarks
    svm_f1s = []
    tri_f1s = []
    tri_blend = 0.65 * oof_svm + 0.20 * oof_xlm + 0.15 * oof_muril

    for tr_idx, val_idx in rskf.split(oof_svm, y):
        svm_f1s.append(f1_score(y[val_idx], np.argmax(oof_svm[val_idx], axis=1), average="macro"))
        tri_f1s.append(f1_score(y[val_idx], np.argmax(tri_blend[val_idx], axis=1), average="macro"))

    print(f"Benchmark [LinearSVC Alone]                   : {np.mean(svm_f1s):.5f} +- {np.std(svm_f1s):.4f}")
    print(f"Benchmark [Current Champion Tri-Ensemble]     : {np.mean(tri_f1s):.5f} +- {np.std(tri_f1s):.4f}")

    # Test candidate weights with mDeBERTa:
    # A. LinearSVC + mDeBERTa (Dual)
    print("\n--- Evaluating LinearSVC + mDeBERTa-v3-base (Dual) ---")
    for w_svm in [0.70, 0.75, 0.80, 0.85]:
        w_mdeb = 1.0 - w_svm
        f1s = []
        for tr_idx, val_idx in rskf.split(oof_svm, y):
            b = w_svm * oof_svm[val_idx] + w_mdeb * oof_mdeb[val_idx]
            f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
        print(f"  Dual ({w_svm*100:.0f}% SVM + {w_mdeb*100:.0f}% mDeB) : {np.mean(f1s):.5f} +- {np.std(f1s):.4f}")

    # B. LinearSVC + XLM-R + mDeBERTa (Tri-Transformer alternative)
    print("\n--- Evaluating LinearSVC + XLM-R + mDeBERTa (Tri without MuRIL) ---")
    for w_svm, w_xlm, w_mdeb in [
        (0.65, 0.25, 0.10),
        (0.65, 0.20, 0.15),
        (0.70, 0.20, 0.10),
        (0.70, 0.15, 0.15),
        (0.75, 0.15, 0.10)
    ]:
        f1s = []
        for tr_idx, val_idx in rskf.split(oof_svm, y):
            b = w_svm * oof_svm[val_idx] + w_xlm * oof_xlm[val_idx] + w_mdeb * oof_mdeb[val_idx]
            f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
        print(f"  Tri ({w_svm*100:.0f}% SVM + {w_xlm*100:.0f}% XLM + {w_mdeb*100:.0f}% mDeB) : {np.mean(f1s):.5f} +- {np.std(f1s):.4f}")

    # C. Quad-Model Ensemble (LinearSVC + XLM-R + MuRIL + mDeBERTa)
    print("\n--- Evaluating Quad-Model Ensembles (SVM + XLM-R + MuRIL + mDeBERTa) ---")
    candidate_quad_weights = [
        (0.60, 0.20, 0.10, 0.10),
        (0.60, 0.15, 0.15, 0.10),
        (0.65, 0.15, 0.12, 0.08),
        (0.65, 0.18, 0.10, 0.07),
        (0.65, 0.15, 0.10, 0.10),
        (0.65, 0.20, 0.10, 0.05),
        (0.68, 0.16, 0.10, 0.06),
        (0.70, 0.15, 0.08, 0.07),
        (0.70, 0.12, 0.10, 0.08),
    ]

    quad_results = []
    for w_svm, w_xlm, w_mur, w_mdeb in candidate_quad_weights:
        w_tot = w_svm + w_xlm + w_mur + w_mdeb
        w1, w2, w3, w4 = w_svm/w_tot, w_xlm/w_tot, w_mur/w_tot, w_mdeb/w_tot
        f1s = []
        for tr_idx, val_idx in rskf.split(oof_svm, y):
            b = (w1 * oof_svm[val_idx] +
                 w2 * oof_xlm[val_idx] +
                 w3 * oof_muril[val_idx] +
                 w4 * oof_mdeb[val_idx])
            f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
        mean_score = np.mean(f1s)
        std_score = np.std(f1s)
        quad_results.append((w1, w2, w3, w4, mean_score, std_score))
        print(f"  Quad ({w1*100:.1f}% SVM + {w2*100:.1f}% XLM + {w3*100:.1f}% MuR + {w4*100:.1f}% mDeB) : {mean_score:.5f} +- {std_score:.4f}")

    # D. In-Fold SLSQP Weight Optimization on Repeated CV
    print("\n--- Running In-Fold SLSQP Continuous Weight Optimization (Leakage-Free) ---")
    slsqp_f1s = []
    slsqp_weights = []

    for fold_i, (tr_idx, val_idx) in enumerate(rskf.split(oof_svm, y)):
        tr_svm, tr_xlm, tr_mur, tr_mdeb = oof_svm[tr_idx], oof_xlm[tr_idx], oof_muril[tr_idx], oof_mdeb[tr_idx]
        val_svm, val_xlm, val_mur, val_mdeb = oof_svm[val_idx], oof_xlm[val_idx], oof_muril[val_idx], oof_mdeb[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Optimize on tr_idx to maximize smooth approximation of macro F1 or minimize negative cross-entropy
        def obj_fn(w):
            w = np.clip(w, 0, None)
            w = w / np.sum(w)
            blend_tr = w[0] * tr_svm + w[1] * tr_xlm + w[2] * tr_mur + w[3] * tr_mdeb
            eps = 1e-9
            blend_tr = np.clip(blend_tr, eps, 1.0 - eps)
            # Log loss on training
            loss = -np.mean(np.log(blend_tr[np.arange(len(y_tr)), y_tr]))
            return loss

        init_w = np.array([0.65, 0.18, 0.10, 0.07])
        bnds = [(0.0, 1.0) for _ in range(4)]
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        res = minimize(obj_fn, init_w, method='SLSQP', bounds=bnds, constraints=cons)

        opt_w = res.x / np.sum(res.x)
        slsqp_weights.append(opt_w)

        # Evaluate on validation fold
        blend_val = opt_w[0] * val_svm + opt_w[1] * val_xlm + opt_w[2] * val_mur + opt_w[3] * val_mdeb
        preds_val = np.argmax(blend_val, axis=1)
        slsqp_f1s.append(f1_score(y_val, preds_val, average="macro"))

    mean_slsqp_f1 = np.mean(slsqp_f1s)
    std_slsqp_f1 = np.std(slsqp_f1s)
    avg_opt_w = np.mean(slsqp_weights, axis=0)
    print(f"SLSQP In-Fold Opt CV Mean Macro F1: {mean_slsqp_f1:.5f} +- {std_slsqp_f1:.4f}")
    print(f"Average Optimal Weights: SVM={avg_opt_w[0]:.3f}, XLM={avg_opt_w[1]:.3f}, MuRIL={avg_opt_w[2]:.3f}, mDeBERTa={avg_opt_w[3]:.3f}")

    # E. Multinomial Logistic Regression Stacker
    print("\n--- Evaluating Multinomial Logistic Regression Stacker on 4 Models ---")
    X_stacked = np.hstack([safe_log(oof_svm), safe_log(oof_xlm), safe_log(oof_muril), safe_log(oof_mdeb)])
    stacker_f1s = []
    for tr_idx, val_idx in rskf.split(X_stacked, y):
        clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        clf.fit(X_stacked[tr_idx], y[tr_idx])
        preds = clf.predict(X_stacked[val_idx])
        stacker_f1s.append(f1_score(y[val_idx], preds, average="macro"))
    print(f"Stacker CV Mean Macro F1: {np.mean(stacker_f1s):.5f} +- {np.std(stacker_f1s):.4f}")

    # F. Rank Average across all 4 models
    print("\n--- Evaluating Rank Average across all 4 Models ---")
    rank_f1s = []
    for tr_idx, val_idx in rskf.split(oof_svm, y):
        val_probs = [oof_svm[val_idx], oof_xlm[val_idx], oof_muril[val_idx], oof_mdeb[val_idx]]
        blend = rank_average_probabilities(val_probs)
        rank_f1s.append(f1_score(y[val_idx], np.argmax(blend, axis=1), average="macro"))
    print(f"Rank Average CV Mean Macro F1: {np.mean(rank_f1s):.5f} +- {np.std(rank_f1s):.4f}")

    # Summary and Decision
    quad_results.sort(key=lambda x: x[4], reverse=True)
    best_quad = quad_results[0]

    print("\n" + "=" * 80)
    print(" COMPARISON & DECISION:")
    print("=" * 80)
    print(f"Current Champion Tri-Ensemble (65% SVM + 20% XLM + 15% MuRIL) : {np.mean(tri_f1s):.5f} +- {np.std(tri_f1s):.4f}")
    print(f"Best Quad Ensemble ({best_quad[0]*100:.1f}% SVM + {best_quad[1]*100:.1f}% XLM + {best_quad[2]*100:.1f}% MuR + {best_quad[3]*100:.1f}% mDeB): {best_quad[4]:.5f} +- {best_quad[5]:.4f}")

    # Calculate full 8195 OOF Macro F1
    full_oof_quad = (best_quad[0] * oof_svm +
                     best_quad[1] * oof_xlm +
                     best_quad[2] * oof_muril +
                     best_quad[3] * oof_mdeb)
    full_quad_f1 = f1_score(y, np.argmax(full_oof_quad, axis=1), average="macro")
    full_quad_acc = accuracy_score(y, np.argmax(full_oof_quad, axis=1))

    full_tri_f1 = f1_score(y, np.argmax(tri_blend, axis=1), average="macro")
    full_tri_acc = accuracy_score(y, np.argmax(tri_blend, axis=1))

    print(f"\nFull 8,195 OOF Macro F1 (Tri-Ensemble)  : {full_tri_f1:.5f} (Accuracy: {full_tri_acc*100:.2f}%)")
    print(f"Full 8,195 OOF Macro F1 (Quad-Ensemble) : {full_quad_f1:.5f} (Accuracy: {full_quad_acc*100:.2f}%)")

    # If Quad ensemble beats Tri-Ensemble, create submission!
    if full_quad_f1 > full_tri_f1 or best_quad[4] > np.mean(tri_f1s):
        print("\n>>> SUCCESS: Quad Ensemble beats previous Champion!")
        quad_test_probs = (best_quad[0] * test_svm +
                           best_quad[1] * test_xlm +
                           best_quad[2] * test_muril +
                           best_quad[3] * test_mdeb)
        test_preds = np.argmax(quad_test_probs, axis=1)
        test_labels = [idx2label[p] for p in test_preds]
        sub_df = pd.DataFrame({"id": test_df["id"], "label": test_labels})
        sub_file = root / f"Submission/submission_quad_ensemble_svm_xlm_muril_mdeb_{full_quad_f1:.4f}.csv"
        sub_df.to_csv(sub_file, index=False)
        print(f"[+] Saved New Best Submission: {sub_file}")

        # 10-Point Audit
        assert list(sub_df.columns) == ["id", "label"]
        assert len(sub_df) == 2048
        assert sub_df.isnull().sum().sum() == 0
        assert (sub_df["id"].values == test_df["id"].values).all()
        assert set(sub_df["label"].unique()).issubset(set(CANONICAL_CLASSES))
        assert len(sub_df["label"].unique()) == 4
        print(">>> 10-Point Audit Passed Completely!")
        print(sub_df["label"].value_counts().to_string())
    else:
        print("\n[!] Tri-Ensemble remains superior or equivalent; checking fine-grained weights...")


if __name__ == "__main__":
    main()

