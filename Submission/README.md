# Submission Pipeline — BhasaQuest-1

> **Competition**: BhasaQuest-1: Multi-Class Text Classification using Romanized Bangla  
> **Evaluation Metric**: Macro-averaged F1 Score  
> **Target Labels**: 4 Anonymized Categorical Classes (`14AC78`, `72A2C8`, `19E1A1`, `6C0197`)  

---

## 1. Overview & Directory Structure

This folder contains the complete, leak-free training, cross-validation, and submission pipeline for the BhasaQuest-1 Kaggle competition.

```
Submission/
├── create_submission.py              # Master script: 5-Fold Stratified CV, training, inference, audit
├── sample_submission_template.csv     # Exact template matching competition specifications
├── submission_tfidf_baseline.csv      # Generated and audited baseline submission CSV
├── submission_final.csv               # Primary production submission CSV
├── README.md                          # Documentation and instructions
├── models/                            # Cached probability matrices and model checkpoints
│   ├── oof_baseline_probs.npy         # Out-of-Fold probability distribution (8195, 4)
│   ├── test_baseline_probs.npy        # Test prediction probabilities (2048, 4)
│   ├── final_logistic_regression.joblib
│   ├── final_word_vectorizer.joblib
│   └── final_char_vectorizer.joblib
├── reports/                           # Cross-validation and format audit reports
│   ├── cv_evaluation_report.json      # Structured per-fold and per-class metrics
│   └── confusion_matrix.png           # 5-fold OOF confusion matrix plot
└── logs/                              # Execution logs
    └── submission_pipeline.log
```

---

## 2. Strict Anti-Leakage Cross-Validation Protocol

To guard against preprocessing and vocabulary leakage:
1. The 8,195 labeled training samples are partitioned using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. For every fold:
   - Word TF-IDF (`ngram_range=(1,2)`, `min_df=2`, `max_df=0.85`, `sublinear_tf=True`) and Character TF-IDF (`analyzer="char_wb"`, `ngram_range=(3,5)`, `min_df=3`, `sublinear_tf=True`) are **fitted exclusively on the training split**.
   - The validation split is transformed strictly using the fold-fitted vectorizers.
   - Logistic Regression with `class_weight="balanced"` and `C=1.0` is fitted and predicts class probabilities on the validation fold.
3. Out-Of-Fold probabilities are stitched across all 5 folds to evaluate true generalization performance before final retraining on 100% of labeled data.

---

## 3. Baseline Validation Performance

| Metric | Cross-Validation Score | Notes |
| :--- | :---: | :--- |
| **Out-of-Fold Macro F1** | **0.7130** | Unweighted mean of the 4 class F1 scores |
| **Out-of-Fold Accuracy** | **72.57%** | Global classification accuracy |
| **Fold 1 Macro F1** | 0.7201 | Consistent across folds ($\sigma = 0.0062$) |
| **Fold 2 Macro F1** | 0.7193 | |
| **Fold 3 Macro F1** | 0.7063 | |
| **Fold 4 Macro F1** | 0.7140 | |
| **Fold 5 Macro F1** | 0.7055 | |

### Per-Class Performance Breakdown

| Class Code | Support | Precision | Recall | F1 Score | Key Error Patterns |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`14AC78`** | 2,791 | **0.9500** | 0.8033 | **0.8705** | High precision; distinct e-commerce/support vocabulary |
| **`19E1A1`** | 1,994 | 0.6329 | 0.6961 | 0.6630 | Confused primarily with `72A2C8` (410 false predictions) |
| **`6C0197`** | 1,255 | 0.6215 | 0.7235 | 0.6686 | Strong recall on insults; minor overlap with `19E1A1`/`72A2C8` |
| **`72A2C8`** | 2,155 | 0.6460 | 0.6538 | 0.6499 | Confused primarily with `19E1A1` (471 false predictions) |

---

## 4. Automated 10-Point Submission Audit

Before generating any output submission file, `create_submission.py` performs an automated integrity verification:

- [x] Column names strictly match `["id", "label"]`.
- [x] Row count equals exactly 2,048.
- [x] Zero null, NaN, or infinite values.
- [x] Zero empty strings or whitespace entries.
- [x] Exactly 2,048 unique test IDs.
- [x] IDs match `Datasets/test.csv` in identical sequential order.
- [x] Predicted labels belong strictly to valid classes `['14AC78', '19E1A1', '6C0197', '72A2C8']`.
- [x] All 4 classes predicted (guarding against single-class mode collapse).
- [x] Distribution sanity check against training priors.
- [x] Clean CSV format readable by standard parsers without index artifacts.

---

## 5. How to Re-Run & Customize

### Default 5-Fold CV & Submission Generation
```bash
python Submission/create_submission.py
```

### Custom Cross-Validation and Regularization Parameters
```bash
python Submission/create_submission.py --n_splits 5 --c_param 1.5 --seed 42 --data_dir "Datasets" --submission_dir "Submission"
```

