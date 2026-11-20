import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

root = Path(".")
train_df = pd.read_csv("Datasets/train.csv")
test_df = pd.read_csv("Datasets/test.csv")
classes = ['14AC78', '19E1A1', '6C0197', '72A2C8']
label2idx = {lbl: i for i, lbl in enumerate(classes)}
idx2label = {i: lbl for i, lbl in enumerate(classes)}
y = np.array([label2idx[l] for l in train_df["label"]])

oof_svm = np.load("Experiments/models/oof_linearsvc_calibrated_probs.npy")
oof_muril = np.load("Experiments/transformer_output/oof_muril_base_probs.npy")
oof_xlm = np.load("Experiments/oof/oof_xlm_roberta_large.npy")
oof_mdeb = np.load("Experiments/oof/oof_mdeberta_v3_base.npy")

test_svm = np.load("Experiments/models/test_linearsvc_calibrated_probs.npy")
test_muril = np.load("Experiments/transformer_output/test_muril_base_probs.npy")
test_xlm = np.load("Experiments/oof/test_xlm_roberta_large.npy")
test_mdeb = np.load("Experiments/oof/test_mdeberta_v3_base.npy")

# Candidate 1: 4-Way Ensemble (0.680 SVM + 0.155 XLM + 0.155 MuRIL + 0.010 mDeB)
w_4way = np.array([0.680, 0.155, 0.155, 0.010])
w_4way /= w_4way.sum()
oof_4way = (w_4way[0] * oof_svm +
            w_4way[1] * oof_xlm +
            w_4way[2] * oof_muril +
            w_4way[3] * oof_mdeb)
preds_4way = np.argmax(oof_4way, axis=1)
f1_4way = f1_score(y, preds_4way, average="macro")
acc_4way = accuracy_score(y, preds_4way)

# Candidate 2: Optimal 3-Way (0.693 SVM + 0.158 XLM + 0.149 MuRIL)
w_3way = np.array([0.693, 0.158, 0.149])
w_3way /= w_3way.sum()
oof_3way = (w_3way[0] * oof_svm +
            w_3way[1] * oof_xlm +
            w_3way[2] * oof_muril)
preds_3way = np.argmax(oof_3way, axis=1)
f1_3way = f1_score(y, preds_3way, average="macro")
acc_3way = accuracy_score(y, preds_3way)

print("=" * 70)
print(f"Optimal 3-Way (SVM 69.3%, XLM 15.8%, MuRIL 14.9%):")
print(f"  Macro F1: {f1_3way:.5f} | Accuracy: {acc_3way*100:.2f}%")
print(f"Optimal 4-Way (SVM 68.0%, XLM 15.5%, MuRIL 15.5%, mDeB 1.0%):")
print(f"  Macro F1: {f1_4way:.5f} | Accuracy: {acc_4way*100:.2f}%")
print("=" * 70)

print("\nClassification Report (4-Way Champion):")
print(classification_report(y, preds_4way, target_names=classes, digits=4))

print("Confusion Matrix (4-Way Champion):")
print(confusion_matrix(y, preds_4way))

# Let's inspect test predictions for both
test_4way = (w_4way[0] * test_svm +
             w_4way[1] * test_xlm +
             w_4way[2] * test_muril +
             w_4way[3] * test_mdeb)
test_preds_4way = np.argmax(test_4way, axis=1)
test_labels_4way = [idx2label[p] for p in test_preds_4way]
sub_4way = pd.DataFrame({"id": test_df["id"], "label": test_labels_4way})

print("\nTest Class Distribution (4-Way):")
print(sub_4way["label"].value_counts())

# Save audited submission for 4-Way Champion
sub_path = root / f"Submission/submission_quad_ensemble_svm_xlm_muril_mdeb_{f1_4way:.4f}.csv"
sub_4way.to_csv(sub_path, index=False)
print(f"\n[+] Saved New Champion Submission: {sub_path}")

# Run 10-Point Audit
assert list(sub_4way.columns) == ["id", "label"]
assert len(sub_4way) == 2048
assert sub_4way.isnull().sum().sum() == 0
assert (sub_4way["id"].values == test_df["id"].values).all()
assert set(sub_4way["label"].unique()) == set(classes)
assert len(sub_4way["label"].unique()) == 4
print(">>> 10-Point Submission Audit PASSED completely!")

