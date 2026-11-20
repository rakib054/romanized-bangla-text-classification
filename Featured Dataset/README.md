# Featured Dataset — Reusable Feature Engineering Toolkit

> **Competition**: BhasaQuest-1: Multi-Class Text Classification using Romanized Bangla  
> **Evaluation Metric**: Macro F1 Score  
> **Target**: 4 Categorical Classes (`14AC78`, `72A2C8`, `19E1A1`, `6C0197`)  

---

## 1. Overview & Architecture

The `Featured Dataset` directory houses the modular, non-destructive feature engineering pipeline for Romanized Bangla NLP modeling. Rather than forcing all engineered features into a monolithic table, transformations are isolated into independent, cleanly versioned feature families.

```
Featured Dataset/
├── clean_dataset.py           # Master execution script for preprocessing & feature extraction
├── config.json                # User-configurable parameters (n-gram ranges, dims, cleaners)
├── README.md                  # Operational documentation
├── artifacts/                 # Serialized fitted vectorizers (.joblib)
│   ├── word_vectorizer.joblib # Full-train fitted word TF-IDF vectorizer
│   └── char_vectorizer.joblib # Full-train fitted char_wb TF-IDF vectorizer
├── data/
│   ├── processed/             # Parallel non-destructive text columns (.csv)
│   │   ├── train_processed_text.csv
│   │   └── test_processed_text.csv
│   ├── features/              # Sparse (.npz) and dense (.csv) feature matrices
│   │   ├── train_word_tfidf.npz (8195, 15000)
│   │   ├── test_word_tfidf.npz  (2048, 15000)
│   │   ├── train_char_tfidf.npz (8195, 35000)
│   │   ├── test_char_tfidf.npz  (2048, 35000)
│   │   ├── train_combined_tfidf.npz (8195, 50000)
│   │   ├── test_combined_tfidf.npz  (2048, 50000)
│   │   ├── train_numerical_features.csv (17 meta-features)
│   │   └── test_numerical_features.csv  (17 meta-features)
│   └── manifests/             # Machine-readable metadata & documentation
│       ├── dataset_manifest.json
│       └── feature_registry.json
├── reports/                   # Performance reports and schema logs
└── logs/                      # Execution log files
```

---

## 2. Text Normalization Representations

Four distinct text versions are generated without ever mutating the original raw text:

1. **`text_raw`**: Exact original text directly from `Datasets/train.csv` and `Datasets/test.csv`. Preserves original capitalization, symbols, and spelling.
2. **`text_lower`**: Lowercased raw text.
3. **`text_conservative_clean`**: Stripped whitespace without altering punctuation, casing, or code-switching markers.
4. **`text_normalized`**: Phonetically stabilized text normalizing repeated expressive characters (`threshold=2`, e.g., `khuuuub` -> `khuub`), NFKD unicode normalization, and collapsed whitespace while strictly preserving hyphens (e.g., `cloud-er`, `time-release`).

---

## 3. Feature Families & Registry Summary

See [`data/manifests/feature_registry.json`](data/manifests/feature_registry.json) for full metadata.

| Feature Family | Format | Dimension | Intended Use Case |
| :--- | :---: | :---: | :--- |
| **`numerical_meta_features`** | Dense CSV | 17 features | Gradient Boosted Trees (LightGBM, XGBoost) & stacking |
| **`word_tfidf`** | Sparse CSR (`.npz`) | 15,000 | Classical linear baselines, Naive Bayes, Ridge |
| **`char_tfidf` (`char_wb`)** | Sparse CSR (`.npz`) | 35,000 | Primary mitigation against the **34.76% word OOV rate** |
| **`combined_tfidf`** | Sparse CSR (`.npz`) | 50,000 | Best-performing classical NLP representation |

---

## 4. How to Run & Configure

### Basic Execution (Default Settings)
```bash
python "Featured Dataset/clean_dataset.py"
```

### Custom Dataset Path or Configuration
```bash
python "Featured Dataset/clean_dataset.py" --config "Featured Dataset/config.json" --data_dir "Datasets" --output_dir "Featured Dataset"
```

### Strict Anti-Leakage Protocol Note
When evaluating models via K-Fold Cross-Validation, **never** evaluate models on the global `train_combined_tfidf.npz`. Instead, use `Submission/create_submission.py`, which strictly instantiates and fits vectorizers on training folds only. Global matrices are intended solely for final full-dataset model fitting and fast exploratory inference.

