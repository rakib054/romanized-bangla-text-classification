# Analysis Folder — BhasaQuest-1 Automated NLP Dataset Analysis

> **Competition**: BhasaQuest-1: Multi-Class Text Classification using Romanized Bangla  
> **Evaluation Metric**: Macro F1 Score  
> **Target Labels**: 4 Anonymized Categorical Classes (`14AC78`, `72A2C8`, `19E1A1`, `6C0197`)  

---

## 1. Overview

The `Analysis/` folder contains the automated Exploratory Data Analysis (EDA) suite, statistical profiling tools, generated figures, structured machine-readable reports, and the comprehensive human-readable report [`Analysis.md`](Analysis.md).

```
Analysis/
├── analysis.py               # Master automated EDA execution script
├── Analysis.md               # 17-section comprehensive analytical report with embedded figures
├── README.md                 # Documentation and execution guide
├── reports/                  # Machine-readable JSON summary reports
│   ├── dataset_verification.json
│   ├── text_structural_summary.json
│   ├── linguistic_patterns.json
│   ├── class_analysis.json
│   ├── duplication_leakage_report.json
│   ├── train_test_distribution.json
│   ├── feature_correlations.json
│   └── master_eda_summary.json
├── figures/                  # Publication-grade analytical figures
│   ├── distributions/
│   │   ├── class_distribution.png
│   │   ├── length_distribution_train_vs_test.png
│   │   └── length_by_class.png
│   ├── ngrams/
│   │   ├── top_words_by_class.png
│   │   └── char_ngrams_top.png
│   ├── linguistics/
│   │   └── spelling_variations_and_suffixes.png
│   └── comparisons/
│       ├── train_test_vocab_overlap.png
│       ├── feature_correlation_heatmap.png
│       ├── punctuation_and_casing_by_class.png
│       └── baseline_confusion_matrix.png
└── logs/                     # Execution logs
    └── analysis.log
```

---

## 2. Key Analysis Modules

`analysis.py` performs the following automated assessments:

1. **Schema & Integrity Verification**: File dimensions, null audits, ID uniqueness, label validity, test/sample submission alignment.
2. **Structural & Text Quality Profiling**: Character count, word count, unique tokens, TTR, casing ratios, digit/punctuation frequency, repeated expressive characters.
3. **Romanized Bangla Linguistics**: Morphology, Bengali inflections (`-er`, `-te`, `-guli`, `-der`), English code-switching loanwords, character n-grams (`char_wb`), and candidate spelling variations via subword Jaccard similarity.
4. **Target Class Profiling**: Class imbalance ratio (2.22x), class-specific text length statistics, distinctive n-grams via TF-IDF log-odds ratio, vocabulary overlap matrix.
5. **Duplication & Leakage Detection**: Exact duplicates, normalized duplicates, near-duplicates, cross-split test overlap, and common prefix templates.
6. **Train-Test Distribution Shifts**: Kolmogorov-Smirnov tests on lengths, vocabulary overlap, Out-Of-Vocabulary (OOV) rates (34.76% word OOV, 24.06% char_wb OOV).
7. **Meta-Feature Association**: Spearman rank correlations, One-Way ANOVA, and Kruskal-Wallis H tests across classes.

---

## 3. How to Run

### Default Execution
```bash
python Analysis/analysis.py
```

### Custom Directories
```bash
python Analysis/analysis.py --data_dir "Datasets" --output_dir "Analysis" --seed 42
```
All reports and figures will be regenerated automatically.

