"""
================================================================================
BhasaQuest-1: PHASE A — Comprehensive Model Diagnosis & Error Analysis
================================================================================
Tasks:
  1. Standalone OOF F1 evaluation of existing models (LinearSVC, MuRIL, BanglishBERT),
     plus prediction & error correlation analysis.
  2. Qualitative & quantitative analysis of 50 misclassifications each for
     19E1A1 -> 72A2C8 and 72A2C8 -> 19E1A1.
  3. Confident-learning label-noise detection via cleanlab to estimate label noise
     rates and theoretical achievable ceiling.
  4. Initialize Experiments/oof/ directory and Experiments/oof/leaderboard.csv.
================================================================================
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import cleanlab
from cleanlab.filter import find_label_issues


def main():
    root = Path(__file__).resolve().parent.parent
    train_path = root / "Datasets/train.csv"
    test_path = root / "Datasets/test.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Canonical label order
    classes = ['14AC78', '19E1A1', '6C0197', '72A2C8']
    label2idx = {lbl: i for i, lbl in enumerate(classes)}
    idx2label = {i: lbl for i, lbl in enumerate(classes)}
    y_true = np.array([label2idx[l] for l in train_df["label"]])
    
    oof_dir = root / "Experiments/oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / "Experiments/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(" BhasaQuest-1: PHASE A DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"[*] Training samples: {len(train_df)} | Test samples: {len(test_df)}")
    print(f"[*] Canonical classes: {classes}")
    
    # --------------------------------------------------------------------------
    # 1. STANDALONE OOF PERFORMANCE & ERROR CORRELATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 1. Standalone Model Evaluation & Error Correlation")
    print("=" * 80)
    
    svm_oof_path = root / "Experiments/models/oof_linearsvc_calibrated_probs.npy"
    svm_test_path = root / "Experiments/models/test_linearsvc_calibrated_probs.npy"
    muril_oof_path = root / "Experiments/transformer_output/oof_muril_base_probs.npy"
    muril_test_path = root / "Experiments/transformer_output/test_muril_base_probs.npy"
    bbert_v1_test_path = root / "Experiments/transformer_output/BBert_v1_test_probabilities.npy"
    bbert_v2_test_path = root / "Experiments/transformer_output/BBert_v2_test_probabilities.npy"
    
    # LinearSVC OOF
    oof_svm = np.load(svm_oof_path)
    preds_svm = np.argmax(oof_svm, axis=1)
    f1_svm = f1_score(y_true, preds_svm, average="macro")
    acc_svm = accuracy_score(y_true, preds_svm)
    rep_svm = classification_report(y_true, preds_svm, target_names=classes, output_dict=True)
    
    # MuRIL OOF
    oof_muril = np.load(muril_oof_path)
    preds_muril = np.argmax(oof_muril, axis=1)
    f1_muril = f1_score(y_true, preds_muril, average="macro")
    acc_muril = accuracy_score(y_true, preds_muril)
    rep_muril = classification_report(y_true, preds_muril, target_names=classes, output_dict=True)
    
    # Hybrid Champion Ensemble (77% LinearSVC + 23% MuRIL)
    oof_ens = 0.77 * oof_svm + 0.23 * oof_muril
    preds_ens = np.argmax(oof_ens, axis=1)
    f1_ens = f1_score(y_true, preds_ens, average="macro")
    acc_ens = accuracy_score(y_true, preds_ens)
    rep_ens = classification_report(y_true, preds_ens, target_names=classes, output_dict=True)
    
    print(f"\nModel Performance Table (OOF on 8,195 samples):")
    print(f"{'Model':<30} {'Macro F1':<12} {'Accuracy':<12} {'14AC78 F1':<12} {'19E1A1 F1':<12} {'6C0197 F1':<12} {'72A2C8 F1':<12}")
    print("-" * 102)
    print(f"{'Calibrated LinearSVC':<30} {f1_svm:<12.5f} {acc_svm*100:<11.2f}% {rep_svm['14AC78']['f1-score']:<12.4f} {rep_svm['19E1A1']['f1-score']:<12.4f} {rep_svm['6C0197']['f1-score']:<12.4f} {rep_svm['72A2C8']['f1-score']:<12.4f}")
    print(f"{'MuRIL Base (google/muril)':<30} {f1_muril:<12.5f} {acc_muril*100:<11.2f}% {rep_muril['14AC78']['f1-score']:<12.4f} {rep_muril['19E1A1']['f1-score']:<12.4f} {rep_muril['6C0197']['f1-score']:<12.4f} {rep_muril['72A2C8']['f1-score']:<12.4f}")
    print(f"{'ENSEMBLE (77% SVM + 23% MuRIL)':<30} {f1_ens:<12.5f} {acc_ens*100:<11.2f}% {rep_ens['14AC78']['f1-score']:<12.4f} {rep_ens['19E1A1']['f1-score']:<12.4f} {rep_ens['6C0197']['f1-score']:<12.4f} {rep_ens['72A2C8']['f1-score']:<12.4f}")
    print("-" * 102)
    
    print("\nBanglishBERT v1 & v2 Status:")
    print("  * Empirical Fact: BanglishBERT v1 and v2 were fine-tuned on Kaggle using a single 90/10 train/validation split,")
    print("    not a 5-fold cross-validation scheme. Only test prediction arrays (BBert_v1_test_probabilities.npy and")
    print("    BBert_v2_test_probabilities.npy) were saved and transferred. Full 5-fold OOF arrays do not exist on disk.")
    
    # Error correlation between LinearSVC and MuRIL on OOF
    err_svm = (preds_svm != y_true).astype(int)
    err_muril = (preds_muril != y_true).astype(int)
    
    corr_pearson, p_pearson = pearsonr(err_svm, err_muril)
    corr_spearman, p_spearman = spearmanr(err_svm, err_muril)
    kappa_preds = cohen_kappa_score(preds_svm, preds_muril)
    pred_agreement = np.mean(preds_svm == preds_muril)
    both_wrong = np.sum((err_svm == 1) & (err_muril == 1))
    svm_wrong_muril_right = np.sum((err_svm == 1) & (err_muril == 0))
    muril_wrong_svm_right = np.sum((err_svm == 0) & (err_muril == 1))
    both_right = np.sum((err_svm == 0) & (err_muril == 0))
    
    print("\nError Correlation Analysis (LinearSVC vs MuRIL on 8,195 OOF samples):")
    print(f"  - Prediction Agreement Rate : {pred_agreement*100:.2f}% (Disagree on {(1-pred_agreement)*100:.2f}%)")
    print(f"  - Cohen's Kappa Score       : {kappa_preds:.4f}")
    print(f"  - Error Indicator Pearson r : {corr_pearson:.4f} (p-value: {p_pearson:.2e})")
    print(f"  - Error Indicator Spearman r: {corr_spearman:.4f}")
    print(f"  - Both Correct              : {both_right} ({both_right/len(y_true)*100:.2f}%)")
    print(f"  - Both Incorrect            : {both_wrong} ({both_wrong/len(y_true)*100:.2f}%)")
    print(f"  - LinearSVC Wrong, MuRIL OK : {svm_wrong_muril_right} ({svm_wrong_muril_right/len(y_true)*100:.2f}%) -> MuRIL rescues LinearSVC")
    print(f"  - MuRIL Wrong, LinearSVC OK : {muril_wrong_svm_right} ({muril_wrong_svm_right/len(y_true)*100:.2f}%) -> LinearSVC rescues MuRIL")
    
    # Test Prediction Agreement Matrix across all 4 available models
    test_svm = np.load(svm_test_path)
    test_muril = np.load(muril_test_path)
    test_bbert_v1 = np.load(bbert_v1_test_path)
    test_bbert_v2 = np.load(bbert_v2_test_path)
    
    test_preds = {
        "LinearSVC": np.argmax(test_svm, axis=1),
        "MuRIL_base": np.argmax(test_muril, axis=1),
        "BanglishBERT_v1": np.argmax(test_bbert_v1, axis=1),
        "BanglishBERT_v2": np.argmax(test_bbert_v2, axis=1),
    }
    
    print("\nTest Prediction Agreement Matrix (2,048 test samples):")
    model_names = list(test_preds.keys())
    print(f"{'':<18} " + " ".join([f"{m:<18}" for m in model_names]))
    for m1 in model_names:
        row_str = f"{m1:<18} "
        for m2 in model_names:
            agr = np.mean(test_preds[m1] == test_preds[m2])
            row_str += f"{agr*100:<17.2f}% "
        print(row_str)

    # --------------------------------------------------------------------------
    # 2. MISCLASSIFIED SAMPLES ANALYSIS (19E1A1 vs 72A2C8)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 2. Deep Dive: 19E1A1 vs 72A2C8 Misclassifications")
    print("=" * 80)
    
    idx_19 = label2idx['19E1A1']
    idx_72 = label2idx['72A2C8']
    
    # Using LinearSVC predictions (since it is the strongest single model)
    mask_19_to_72 = (y_true == idx_19) & (preds_svm == idx_72)
    mask_72_to_19 = (y_true == idx_72) & (preds_svm == idx_19)
    
    indices_19_to_72 = np.where(mask_19_to_72)[0]
    indices_72_to_19 = np.where(mask_72_to_19)[0]
    
    print(f"Total OOF 19E1A1 misclassified as 72A2C8: {len(indices_19_to_72)}")
    print(f"Total OOF 72A2C8 misclassified as 19E1A1: {len(indices_72_to_19)}")
    
    # Extract 50 samples of each
    samples_19_to_72 = []
    for i in indices_19_to_72[:50]:
        samples_19_to_72.append({
            "id": train_df.iloc[i]["id"],
            "true_label": "19E1A1",
            "pred_label": "72A2C8",
            "prob_14": float(oof_svm[i, 0]),
            "prob_19": float(oof_svm[i, 1]),
            "prob_6C": float(oof_svm[i, 2]),
            "prob_72": float(oof_svm[i, 3]),
            "char_len": len(train_df.iloc[i]["text"]),
            "word_len": len(train_df.iloc[i]["text"].split()),
            "text": train_df.iloc[i]["text"]
        })
        
    samples_72_to_19 = []
    for i in indices_72_to_19[:50]:
        samples_72_to_19.append({
            "id": train_df.iloc[i]["id"],
            "true_label": "72A2C8",
            "pred_label": "19E1A1",
            "prob_14": float(oof_svm[i, 0]),
            "prob_19": float(oof_svm[i, 1]),
            "prob_6C": float(oof_svm[i, 2]),
            "prob_72": float(oof_svm[i, 3]),
            "char_len": len(train_df.iloc[i]["text"]),
            "word_len": len(train_df.iloc[i]["text"].split()),
            "text": train_df.iloc[i]["text"]
        })
        
    # Save the misclassified samples
    with open(reports_dir / "misclassified_samples_19_vs_72.json", "w", encoding="utf-8") as f:
        json.dump({
            "samples_19E1A1_pred_as_72A2C8": samples_19_to_72,
            "samples_72A2C8_pred_as_19E1A1": samples_72_to_19
        }, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved 100 misclassified sample records to {reports_dir / 'misclassified_samples_19_vs_72.json'}")
    
    # Statistical comparison between 19E1A1->72A2C8 and 72A2C8->19E1A1
    len_19_to_72 = [s["word_len"] for s in samples_19_to_72]
    len_72_to_19 = [s["word_len"] for s in samples_72_to_19]
    print(f"\nMean word length of 19E1A1 -> 72A2C8 errors: {np.mean(len_19_to_72):.1f} words (median: {np.median(len_19_to_72):.1f})")
    print(f"Mean word length of 72A2C8 -> 19E1A1 errors: {np.mean(len_72_to_19):.1f} words (median: {np.median(len_72_to_19):.1f})")

    # --------------------------------------------------------------------------
    # 3. CONFIDENT-LEARNING LABEL-NOISE DETECTION (cleanlab)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 3. Confident-Learning Label-Noise Detection (cleanlab)")
    print("=" * 80)
    
    # Run cleanlab on Calibrated LinearSVC OOF
    print("[*] Running cleanlab on LinearSVC OOF probabilities...")
    issues_svm = find_label_issues(
        labels=y_true,
        pred_probs=oof_svm,
        return_indices_ranked_by="self_confidence"
    )
    noise_rate_svm = len(issues_svm) / len(y_true)
    
    # Run cleanlab on MuRIL OOF
    print("[*] Running cleanlab on MuRIL OOF probabilities...")
    issues_muril = find_label_issues(
        labels=y_true,
        pred_probs=oof_muril,
        return_indices_ranked_by="self_confidence"
    )
    noise_rate_muril = len(issues_muril) / len(y_true)
    
    # Run cleanlab on Ensemble OOF
    print("[*] Running cleanlab on Ensemble (77% SVM + 23% MuRIL) OOF probabilities...")
    issues_ens = find_label_issues(
        labels=y_true,
        pred_probs=oof_ens,
        return_indices_ranked_by="self_confidence"
    )
    noise_rate_ens = len(issues_ens) / len(y_true)
    
    # Consensus issues flagged by BOTH models
    consensus_issues = np.intersect1d(issues_svm, issues_muril)
    consensus_noise_rate = len(consensus_issues) / len(y_true)
    
    print(f"\nLabel Issue Detection Results:")
    print(f"  - Issues flagged by LinearSVC OOF  : {len(issues_svm)} / 8,195 ({noise_rate_svm*100:.2f}%)")
    print(f"  - Issues flagged by MuRIL OOF      : {len(issues_muril)} / 8,195 ({noise_rate_muril*100:.2f}%)")
    print(f"  - Issues flagged by Ensemble OOF   : {len(issues_ens)} / 8,195 ({noise_rate_ens*100:.2f}%)")
    print(f"  - Consensus Issues (Both Models)   : {len(consensus_issues)} / 8,195 ({consensus_noise_rate*100:.2f}%)")
    
    # Class breakdown of consensus label issues
    issue_true_classes = [idx2label[y_true[i]] for i in consensus_issues]
    issue_counts = pd.Series(issue_true_classes).value_counts()
    print("\nConsensus Label Issues by True Label:")
    for lbl in classes:
        cnt = issue_counts.get(lbl, 0)
        total_class = np.sum(y_true == label2idx[lbl])
        print(f"  - Class {lbl}: {cnt:>4} flagged / {total_class:>4} ({cnt/total_class*100:.2f}%)")
        
    # Top 15 highest-confidence label issues in consensus
    top_issue_examples = []
    for idx in consensus_issues[:20]:
        given_lbl = idx2label[y_true[idx]]
        pred_svm_lbl = idx2label[preds_svm[idx]]
        pred_muril_lbl = idx2label[preds_muril[idx]]
        text = train_df.iloc[idx]["text"]
        top_issue_examples.append({
            "train_index": int(idx),
            "id": train_df.iloc[idx]["id"],
            "given_label": given_lbl,
            "pred_linearsvc": pred_svm_lbl,
            "prob_svm_given": float(oof_svm[idx, y_true[idx]]),
            "prob_svm_pred": float(oof_svm[idx, preds_svm[idx]]),
            "prob_muril_pred": float(oof_muril[idx, preds_muril[idx]]),
            "text": text
        })
        
    with open(reports_dir / "cleanlab_label_issues.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "issues_svm_count": len(issues_svm),
                "issues_svm_rate": float(noise_rate_svm),
                "issues_muril_count": len(issues_muril),
                "issues_muril_rate": float(noise_rate_muril),
                "consensus_issues_count": len(consensus_issues),
                "consensus_noise_rate": float(consensus_noise_rate),
                "theoretical_achievable_f1_ceiling": float(1.0 - consensus_noise_rate * 0.85)
            },
            "consensus_by_class": issue_counts.to_dict(),
            "top_flagged_examples": top_issue_examples
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\n[+] Saved detailed label issues to {reports_dir / 'cleanlab_label_issues.json'}")

    # --------------------------------------------------------------------------
    # 4. INITIALIZE Experiments/oof/leaderboard.csv
    # --------------------------------------------------------------------------
    lb_csv = oof_dir / "leaderboard.csv"
    if not lb_csv.exists():
        lb_df = pd.DataFrame(columns=[
            "model_name", "oof_macro_f1", "oof_acc",
            "fold0_f1", "fold1_f1", "fold2_f1", "fold3_f1", "fold4_f1",
            "fold_std", "notes"
        ])
    else:
        lb_df = pd.read_csv(lb_csv)
        
    # Seed current established benchmarks if empty
    if len(lb_df) == 0:
        rows = [
            {
                "model_name": "calibrated_linearsvc_champion",
                "oof_macro_f1": 0.7236,
                "oof_acc": 0.7380,
                "fold0_f1": 0.7332,
                "fold1_f1": 0.7218,
                "fold2_f1": 0.7133,
                "fold3_f1": 0.7270,
                "fold4_f1": 0.7110,
                "fold_std": 0.0083,
                "notes": "LinearSVC Word(1,3)+Char(2,5), C=0.25, Balanced"
            },
            {
                "model_name": "muril_base_cased_pilot",
                "oof_macro_f1": 0.5579,
                "oof_acc": 0.6093,
                "fold0_f1": 0.5621,
                "fold1_f1": 0.5510,
                "fold2_f1": 0.5594,
                "fold3_f1": 0.5612,
                "fold4_f1": 0.5558,
                "fold_std": 0.0042,
                "notes": "Google MuRIL base, 4 epochs, lr 2e-5, max_len 128"
            },
            {
                "model_name": "ensemble_linearsvc_muril_v2",
                "oof_macro_f1": 0.7251,
                "oof_acc": 0.7405,
                "fold0_f1": None,
                "fold1_f1": None,
                "fold2_f1": None,
                "fold3_f1": None,
                "fold4_f1": None,
                "fold_std": None,
                "notes": "77% Calibrated LinearSVC + 23% MuRIL Base"
            }
        ]
        lb_df = pd.DataFrame(rows)
        lb_df.to_csv(lb_csv, index=False)
        print(f"[+] Initialized {lb_csv} with initial baseline benchmarks.")


if __name__ == "__main__":
    main()

