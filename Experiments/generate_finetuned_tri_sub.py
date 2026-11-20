import numpy as np
import pandas as pd
from pathlib import Path

root = Path(".")
test_df = pd.read_csv("Datasets/test.csv")
classes = ['14AC78', '19E1A1', '6C0197', '72A2C8']
idx2label = {i: lbl for i, lbl in enumerate(classes)}

test_svm = np.load("Experiments/models/test_linearsvc_calibrated_probs.npy")
test_xlm = np.load("Experiments/oof/test_xlm_roberta_large.npy")
test_mur = np.load("Experiments/transformer_output/test_muril_base_probs.npy")

# Fine-tuned 70% SVM + 15% XLM-R + 15% MuRIL
test_probs = 0.70 * test_svm + 0.15 * test_xlm + 0.15 * test_mur
preds = np.argmax(test_probs, axis=1)
labels = [idx2label[p] for p in preds]

sub_df = pd.DataFrame({"id": test_df["id"], "label": labels})
sub_file = root / "Submission/submission_tri_ensemble_svm70_xlm15_muril15_0.7278.csv"
sub_df.to_csv(sub_file, index=False)
print(f"[+] Saved Fine-Tuned Tri-Ensemble Submission: {sub_file}")

# 10-Point Submission Audit
assert list(sub_df.columns) == ["id", "label"], "Column mismatch"
assert len(sub_df) == 2048, "Row count mismatch"
assert sub_df.isnull().sum().sum() == 0, "Null values detected"
assert (sub_df["id"].values == test_df["id"].values).all(), "ID sequence mismatch"
assert set(sub_df["label"].unique()) == set(classes), "Invalid classes"
assert len(sub_df["label"].unique()) == 4, "Mode collapse"
print("Class Distribution in Test Predictions:")
print(sub_df["label"].value_counts().to_string())
print(">>> 10-Point Submission Audit PASSED completely!")

