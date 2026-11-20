import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score

root = Path(".")
train_df = pd.read_csv("Datasets/train.csv")
classes = ['14AC78', '19E1A1', '6C0197', '72A2C8']
label2idx = {lbl: i for i, lbl in enumerate(classes)}
y = np.array([label2idx[l] for l in train_df["label"]])

oof_svm = np.load("Experiments/models/oof_linearsvc_calibrated_probs.npy")
oof_muril = np.load("Experiments/transformer_output/oof_muril_base_probs.npy")
oof_xlm = np.load("Experiments/oof/oof_xlm_roberta_large.npy")
oof_mdeb = np.load("Experiments/oof/oof_mdeberta_v3_base.npy")

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

results = []
# Grid search around champion
for w_svm in [0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70]:
    for w_xlm in [0.16, 0.18, 0.20, 0.22, 0.24]:
        for w_mur in [0.10, 0.12, 0.14, 0.15, 0.16, 0.18]:
            for w_mdeb in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
                tot = w_svm + w_xlm + w_mur + w_mdeb
                w1, w2, w3, w4 = w_svm/tot, w_xlm/tot, w_mur/tot, w_mdeb/tot
                
                # Fast full OOF check first
                blend = w1 * oof_svm + w2 * oof_xlm + w3 * oof_muril + w4 * oof_mdeb
                full_f1 = f1_score(y, np.argmax(blend, axis=1), average="macro")
                
                if full_f1 >= 0.7265:
                    results.append((w1, w2, w3, w4, full_f1))

results.sort(key=lambda x: x[4], reverse=True)
print(f"Total candidate combinations with full OOF >= 0.7265: {len(results)}")
print("Top 10 combinations on full 8195 OOF:")
for r in results[:10]:
    print(f"SVM={r[0]:.3f}, XLM={r[1]:.3f}, MuRIL={r[2]:.3f}, mDeB={r[3]:.3f} -> Full OOF F1: {r[4]:.5f}")

# Now evaluate top 5 on 25-fold Repeated CV
print("\nValidating top candidates on 25-fold Repeated CV:")
for r in results[:5]:
    w1, w2, w3, w4 = r[0], r[1], r[2], r[3]
    f1s = []
    for tr_idx, val_idx in rskf.split(oof_svm, y):
        b = w1 * oof_svm[val_idx] + w2 * oof_xlm[val_idx] + w3 * oof_muril[val_idx] + w4 * oof_mdeb[val_idx]
        f1s.append(f1_score(y[val_idx], np.argmax(b, axis=1), average="macro"))
    print(f"SVM={w1:.3f}, XLM={w2:.3f}, MuR={w3:.3f}, mDeB={w4:.3f} -> 25-Fold CV: {np.mean(f1s):.5f} +- {np.std(f1s):.4f} (Full: {r[4]:.5f})")

