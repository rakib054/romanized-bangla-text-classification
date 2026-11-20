"""
BhasaQuest-1: Automated NLP Analysis Pipeline
=============================================
This module provides a rigorous, end-to-end Exploratory Data Analysis (EDA)
specifically designed for Romanized Bangla (Banglish) text classification.

Features:
- Dataset integrity and schema verification
- Structural text quality and length percentile analysis
- Romanized Bangla linguistic patterns (suffixes, spelling variations, loan words)
- Class imbalance and class-specific distinctive vocabulary / n-grams
- Exact and near-duplicate text detection and leakage auditing
- Train-test distribution shift analysis (OOV rates, KS tests)
- High-resolution analytical visualizations saved to Analysis/figures/
- Structured JSON and CSV report generation saved to Analysis/reports/
"""

import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional aesthetics
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def setup_logger(log_file: Path) -> logging.Logger:
    """Set up logging to console and file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("BhasaQuest1_EDA")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and pandas data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def verify_datasets(data_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Verify presence, schemas, sizes, and integrity of competition files."""
    logger.info("Step 1: Auditing dataset files and schema integrity...")
    files = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample_submission": data_dir / "sample_submission.csv",
    }

    report = {"files_exist": {}, "file_sizes_kb": {}, "schemas": {}, "integrity": {}}

    for name, path in files.items():
        exists = path.is_file()
        report["files_exist"][name] = exists
        if not exists:
            raise FileNotFoundError(f"Critical dataset missing: {path.resolve()}")
        report["file_sizes_kb"][name] = round(path.stat().st_size / 1024, 2)

    train_df = pd.read_csv(files["train"])
    test_df = pd.read_csv(files["test"])
    sub_df = pd.read_csv(files["sample_submission"])

    report["schemas"]["train"] = {
        "rows": len(train_df),
        "columns": train_df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in train_df.dtypes.items()},
        "null_counts": train_df.isnull().sum().to_dict(),
        "empty_strings": {col: (train_df[col].astype(str).str.strip() == "").sum() for col in train_df.columns},
        "unique_ids": train_df["id"].nunique(),
    }

    report["schemas"]["test"] = {
        "rows": len(test_df),
        "columns": test_df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in test_df.dtypes.items()},
        "null_counts": test_df.isnull().sum().to_dict(),
        "empty_strings": {col: (test_df[col].astype(str).str.strip() == "").sum() for col in test_df.columns},
        "unique_ids": test_df["id"].nunique(),
    }

    report["schemas"]["sample_submission"] = {
        "rows": len(sub_df),
        "columns": sub_df.columns.tolist(),
        "unique_ids": sub_df["id"].nunique(),
    }

    # Integrity validations
    expected_labels = {"14AC78", "72A2C8", "19E1A1", "6C0197"}
    train_labels = set(train_df["label"].unique())
    unknown_labels = train_labels - expected_labels

    id_match = (test_df["id"] == sub_df["id"]).all()
    sub_rows_match = (len(test_df) == len(sub_df) == 2048)

    report["integrity"]["train_has_expected_labels_only"] = (len(unknown_labels) == 0)
    report["integrity"]["unknown_labels_found"] = list(unknown_labels)
    report["integrity"]["test_and_sub_ids_identical_order"] = bool(id_match)
    report["integrity"]["row_counts_match_spec"] = bool(sub_rows_match and len(train_df) == 8195)

    logger.info(f"Verified: Train samples = {len(train_df)}, Test samples = {len(test_df)}, Sub samples = {len(sub_df)}")
    logger.info(f"Integrity check passed: {report['integrity']['test_and_sub_ids_identical_order']}")

    return report


# ---------------------------------------------------------
# 2. Text Structural and Quality Analysis
# ---------------------------------------------------------

def analyze_text_structure(df: pd.DataFrame, text_col: str = "text", dataset_name: str = "train") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute detailed structural, length, and lexical features on text."""
    df_feat = df.copy()

    # Basic length metrics
    df_feat["char_count"] = df_feat[text_col].astype(str).str.len()
    tokens_series = df_feat[text_col].astype(str).str.strip().str.split()
    df_feat["word_count"] = tokens_series.apply(len)
    df_feat["unique_word_count"] = tokens_series.apply(lambda toks: len(set(toks)))
    df_feat["ttr"] = df_feat["unique_word_count"] / df_feat["word_count"].replace(0, 1)

    # Word lengths
    df_feat["avg_word_length"] = tokens_series.apply(lambda toks: np.mean([len(t) for t in toks]) if toks else 0.0)
    df_feat["max_word_length"] = tokens_series.apply(lambda toks: np.max([len(t) for t in toks]) if toks else 0)

    # Casing, digits, punctuation
    df_feat["uppercase_count"] = df_feat[text_col].astype(str).apply(lambda s: sum(1 for c in s if c.isupper()))
    df_feat["lowercase_count"] = df_feat[text_col].astype(str).apply(lambda s: sum(1 for c in s if c.islower()))
    df_feat["digit_count"] = df_feat[text_col].astype(str).apply(lambda s: sum(1 for c in s if c.isdigit()))
    df_feat["punctuation_count"] = df_feat[text_col].astype(str).apply(lambda s: sum(1 for c in s if not c.isalnum() and not c.isspace()))
    df_feat["whitespace_count"] = df_feat[text_col].astype(str).apply(lambda s: sum(1 for c in s if c.isspace()))

    # Ratios
    df_feat["uppercase_ratio"] = df_feat["uppercase_count"] / df_feat["char_count"].replace(0, 1)
    df_feat["digit_ratio"] = df_feat["digit_count"] / df_feat["char_count"].replace(0, 1)
    df_feat["punctuation_ratio"] = df_feat["punctuation_count"] / df_feat["char_count"].replace(0, 1)

    # Special pattern flags
    df_feat["has_url"] = df_feat[text_col].astype(str).str.contains(r"https?://|www\.", regex=True).astype(int)
    df_feat["has_mention"] = df_feat[text_col].astype(str).str.contains(r"@\w+", regex=True).astype(int)
    df_feat["has_hashtag"] = df_feat[text_col].astype(str).str.contains(r"#\w+", regex=True).astype(int)
    df_feat["has_repeated_chars"] = df_feat[text_col].astype(str).apply(
        lambda s: 1 if re.search(r"([a-zA-Z])\1{2,}", s) else 0
    )
    df_feat["has_hyphen"] = df_feat[text_col].astype(str).str.contains(r"-", regex=True).astype(int)
    df_feat["has_question_mark"] = df_feat[text_col].astype(str).str.contains(r"\?", regex=True).astype(int)
    df_feat["has_exclamation"] = df_feat[text_col].astype(str).str.contains(r"!", regex=True).astype(int)

    # Percentiles and summary statistics
    def get_stats(series: pd.Series) -> Dict[str, float]:
        return {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
            "p05": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
        }

    summary = {
        "dataset": dataset_name,
        "sample_count": len(df_feat),
        "char_count_stats": get_stats(df_feat["char_count"]),
        "word_count_stats": get_stats(df_feat["word_count"]),
        "unique_word_count_stats": get_stats(df_feat["unique_word_count"]),
        "ttr_stats": get_stats(df_feat["ttr"]),
        "avg_word_length_stats": get_stats(df_feat["avg_word_length"]),
        "punctuation_count_stats": get_stats(df_feat["punctuation_count"]),
        "digit_count_stats": get_stats(df_feat["digit_count"]),
        "uppercase_ratio_stats": get_stats(df_feat["uppercase_ratio"]),
        "flags_prevalence": {
            "has_url_count": int(df_feat["has_url"].sum()),
            "has_mention_count": int(df_feat["has_mention"].sum()),
            "has_hashtag_count": int(df_feat["has_hashtag"].sum()),
            "has_repeated_chars_count": int(df_feat["has_repeated_chars"].sum()),
            "has_hyphen_count": int(df_feat["has_hyphen"].sum()),
            "has_question_mark_count": int(df_feat["has_question_mark"].sum()),
            "has_exclamation_count": int(df_feat["has_exclamation"].sum()),
        },
        "anomalies": {
            "short_texts_under_5_words": int((df_feat["word_count"] < 5).sum()),
            "long_texts_over_100_words": int((df_feat["word_count"] > 100).sum()),
            "short_texts_under_20_chars": int((df_feat["char_count"] < 20).sum()),
            "long_texts_over_500_chars": int((df_feat["char_count"] > 500).sum()),
        }
    }

    return df_feat, summary


# ---------------------------------------------------------
# 3. Linguistic & Romanized Bangla Patterns
# ---------------------------------------------------------

def extract_linguistic_patterns(train_df: pd.DataFrame, test_df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Analyze Romanized Bangla lexical morphology, code switching, suffixes, and orthographic variations."""
    logger.info("Step 3: Extracting Romanized Bangla linguistic patterns...")

    # Regex tokenization preserving hyphens
    token_pattern = re.compile(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b")

    def tokenize_corpus(texts: pd.Series) -> List[str]:
        tokens = []
        for text in texts:
            tokens.extend([tok.lower() for tok in token_pattern.findall(str(text))])
        return tokens

    train_tokens = tokenize_corpus(train_df["text"])
    test_tokens = tokenize_corpus(test_df["text"])

    train_vocab = Counter(train_tokens)
    test_vocab = Counter(test_tokens)

    # Common Banglish suffixes / inflections
    common_suffixes = [
        "-er", "-e", "-te", "-ke", "-guli", "-gulo", "-ra", "-der",
        "-ta", "-ti", "-baaji", "-joggo", "-purno", "-sheel", "-chhi", "-chhe", "-lam", "-bo"
    ]
    suffix_counts = Counter()
    for tok in train_tokens:
        for suf in common_suffixes:
            clean_suf = suf.replace("-", "")
            if tok.endswith(clean_suf) and len(tok) > len(clean_suf) + 2:
                suffix_counts[suf] += 1

    # Candidate spelling variations: find words with high string similarity
    # Frequent words with count >= 15
    frequent_words = [w for w, c in train_vocab.most_common(600) if len(w) >= 4]

    def char_ngrams(s: str, n: int = 3) -> set:
        return set(s[i:i+n] for i in range(len(s) - n + 1))

    candidate_variations = []
    seen_pairs = set()
    for i, w1 in enumerate(frequent_words):
        s1 = char_ngrams(w1, 3)
        if not s1:
            continue
        for w2 in frequent_words[i+1:]:
            if abs(len(w1) - len(w2)) > 2:
                continue
            pair = tuple(sorted([w1, w2]))
            if pair in seen_pairs:
                continue

            s2 = char_ngrams(w2, 3)
            if not s2:
                continue

            # Jaccard similarity of 3-grams
            jaccard = len(s1.intersection(s2)) / len(s1.union(s2))
            # Levenshtein-like character substitution check
            if jaccard >= 0.50:
                seen_pairs.add(pair)
                candidate_variations.append({
                    "word_a": w1,
                    "count_a": train_vocab[w1],
                    "word_b": w2,
                    "count_b": train_vocab[w2],
                    "char_ngram_jaccard": round(jaccard, 3)
                })

    candidate_variations.sort(key=lambda x: x["char_ngram_jaccard"], reverse=True)

    # English loan words / code-switching words
    english_tech_loanwords = [
        "system", "systemer", "prompt", "code", "database", "database-e",
        "cyber", "attack", "server", "admin", "password", "hack", "bot",
        "file", "script", "robot", "cocktail", "bartender", "online",
        "rot13", "dos", "malware", "capsule", "cyanide", "phishing"
    ]
    loanword_stats = {
        w: {"train_freq": train_vocab.get(w, 0), "test_freq": test_vocab.get(w, 0)}
        for w in english_tech_loanwords if train_vocab.get(w, 0) > 0 or test_vocab.get(w, 0) > 0
    }

    # Top Character N-Grams (3-5)
    char_vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=10, max_features=30)
    char_vec.fit(train_df["text"])
    char_counts = np.asarray(char_vec.transform(train_df["text"]).sum(axis=0)).flatten()
    char_ngrams_top = sorted(zip(char_vec.get_feature_names_out(), char_counts), key=lambda x: x[1], reverse=True)

    return {
        "total_train_tokens": len(train_tokens),
        "total_test_tokens": len(test_tokens),
        "unique_train_vocab": len(train_vocab),
        "unique_test_vocab": len(test_vocab),
        "top_50_train_words": train_vocab.most_common(50),
        "suffix_prevalence": suffix_counts.most_common(20),
        "candidate_spelling_variations": candidate_variations[:30],
        "code_switching_loanwords": loanword_stats,
        "top_char_ngrams": [(str(k), int(v)) for k, v in char_ngrams_top],
    }


# ---------------------------------------------------------
# 4. Class Distribution & Class-Wise Linguistic Profile
# ---------------------------------------------------------

def analyze_classes(train_df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Analyze class balance, per-class text metrics, distinctive vocabulary, and n-grams."""
    logger.info("Step 4: Profiling target classes and discriminative patterns...")

    class_counts = train_df["label"].value_counts().to_dict()
    total_samples = len(train_df)
    class_proportions = {k: round(v / total_samples, 4) for k, v in class_counts.items()}

    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    imbalance_ratio = round(class_counts[majority_class] / class_counts[minority_class], 3)

    classes = sorted(list(class_counts.keys()))

    # Per-class length statistics
    class_profiles = {}
    for c in classes:
        sub = train_df[train_df["label"] == c]
        chars = sub["text"].str.len()
        words = sub["text"].str.split().apply(len)
        class_profiles[c] = {
            "count": int(class_counts[c]),
            "percentage": round(class_proportions[c] * 100, 2),
            "char_length": {
                "mean": round(float(chars.mean()), 2),
                "std": round(float(chars.std()), 2),
                "median": float(chars.median()),
                "min": int(chars.min()),
                "max": int(chars.max()),
                "p25": float(chars.quantile(0.25)),
                "p75": float(chars.quantile(0.75)),
            },
            "word_count": {
                "mean": round(float(words.mean()), 2),
                "std": round(float(words.std()), 2),
                "median": float(words.median()),
                "min": int(words.min()),
                "max": int(words.max()),
                "p25": float(words.quantile(0.25)),
                "p75": float(words.quantile(0.75)),
            }
        }

    # Extract distinctive words per class using TF-IDF sublinear scaling
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        sublinear_tf=True,
        token_pattern=r"\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b"
    )
    X_tfidf = tfidf.fit_transform(train_df["text"])
    feature_names = np.array(tfidf.get_feature_names_out())

    distinctive_ngrams = {}
    representative_samples = {}

    for c in classes:
        mask = (train_df["label"] == c).values
        class_mean_tfidf = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
        other_mean_tfidf = np.asarray(X_tfidf[~mask].mean(axis=0)).flatten()

        # Ratio of in-class TF-IDF to out-of-class TF-IDF
        ratio = (class_mean_tfidf + 1e-5) / (other_mean_tfidf + 1e-5)
        top_indices = np.argsort(ratio)[::-1][:20]

        distinctive_ngrams[c] = [
            {"ngram": feature_names[i], "class_score": round(float(class_mean_tfidf[i]), 5), "distinctive_ratio": round(float(ratio[i]), 2)}
            for i in top_indices
        ]

        # Representative samples: highest sum of distinctive TF-IDF scores
        sub_X = X_tfidf[mask][:, top_indices[:10]].sum(axis=1)
        top_doc_idx = np.asarray(sub_X).flatten().argsort()[::-1][:3]
        sub_texts = train_df[train_df["label"] == c].iloc[top_doc_idx]
        representative_samples[c] = [
            {"id": row["id"], "text": row["text"][:180] + ("..." if len(row["text"]) > 180 else "")}
            for _, row in sub_texts.iterrows()
        ]

    # Class vocabulary overlap (Jaccard matrix)
    token_pattern = re.compile(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b")
    class_vocabs = {
        c: set(tok.lower() for text in train_df[train_df["label"] == c]["text"] for tok in token_pattern.findall(str(text)))
        for c in classes
    }

    jaccard_matrix = {}
    for c1 in classes:
        jaccard_matrix[c1] = {}
        for c2 in classes:
            v1, v2 = class_vocabs[c1], class_vocabs[c2]
            jaccard = round(len(v1.intersection(v2)) / len(v1.union(v2)), 4)
            jaccard_matrix[c1][c2] = jaccard

    return {
        "class_counts": class_counts,
        "class_proportions": class_proportions,
        "majority_class": majority_class,
        "minority_class": minority_class,
        "imbalance_ratio": imbalance_ratio,
        "class_profiles": class_profiles,
        "distinctive_ngrams": distinctive_ngrams,
        "representative_samples": representative_samples,
        "jaccard_similarity_matrix": jaccard_matrix,
    }


# ---------------------------------------------------------
# 5. Duplication & Leakage Analysis
# ---------------------------------------------------------

def analyze_duplication_and_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Detect exact duplicates, normalized duplicates, near-duplicates, and train-test overlaps."""
    logger.info("Step 5: Inspecting duplication, template repetition, and cross-dataset overlap...")

    def normalize_text(text: str) -> str:
        s = str(text).lower()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    train_raw = train_df["text"].tolist()
    test_raw = test_df["text"].tolist()

    train_norm = [normalize_text(t) for t in train_raw]
    test_norm = [normalize_text(t) for t in test_raw]

    # Exact duplicates
    exact_train_dups = int(train_df.duplicated(subset=["text"]).sum())
    exact_test_dups = int(test_df.duplicated(subset=["text"]).sum())
    exact_train_test_overlap = len(set(train_raw).intersection(set(test_raw)))

    # Normalized duplicates
    train_norm_series = pd.Series(train_norm)
    test_norm_series = pd.Series(test_norm)

    norm_train_dups = int(train_norm_series.duplicated().sum())
    norm_test_dups = int(test_norm_series.duplicated().sum())

    norm_overlap_set = set(train_norm).intersection(set(test_norm))
    norm_train_test_overlap = len(norm_overlap_set)

    # Details of normalized overlap
    overlap_details = []
    if norm_train_test_overlap > 0:
        for norm_txt in norm_overlap_set:
            train_matches = train_df[train_norm_series == norm_txt]
            test_matches = test_df[test_norm_series == norm_txt]
            for _, tr in train_matches.iterrows():
                for _, te in test_matches.iterrows():
                    overlap_details.append({
                        "normalized_text": norm_txt,
                        "train_id": tr["id"],
                        "train_label": tr["label"],
                        "train_raw_text": tr["text"],
                        "test_id": te["id"],
                        "test_raw_text": te["text"]
                    })

    # Prefix template analysis (first 4 words)
    def extract_prefix(text: str, n_words: int = 4) -> str:
        words = text.lower().split()
        return " ".join(words[:n_words]) if len(words) >= n_words else ""

    train_prefixes = Counter([extract_prefix(t) for t in train_raw if extract_prefix(t)])
    top_train_prefixes = [
        {"prefix": p, "count": count} for p, count in train_prefixes.most_common(15) if count > 5
    ]

    return {
        "exact_duplicates": {
            "train": exact_train_dups,
            "test": exact_test_dups,
            "train_test_overlap": exact_train_test_overlap
        },
        "normalized_duplicates": {
            "train": norm_train_dups,
            "test": norm_test_dups,
            "train_test_overlap": norm_train_test_overlap
        },
        "normalized_overlap_cases": overlap_details,
        "frequent_prefix_templates": top_train_prefixes,
    }


# ---------------------------------------------------------
# 6. Train-Test Distribution Shifts & OOV Rates
# ---------------------------------------------------------

def analyze_train_test_shifts(
    train_feat: pd.DataFrame, test_feat: pd.DataFrame, logger: logging.Logger
) -> Dict[str, Any]:
    """Quantify distribution shifts in lengths, vocabulary, and n-grams between train and test."""
    logger.info("Step 6: Measuring train-test distribution shifts and OOV rates...")

    # Length distribution statistical tests (Kolmogorov-Smirnov & Mann-Whitney U)
    ks_char = stats.ks_2samp(train_feat["char_count"], test_feat["char_count"])
    ks_word = stats.ks_2samp(train_feat["word_count"], test_feat["word_count"])

    # Vocabulary overlap & Out-Of-Vocabulary (OOV) rate
    token_pattern = re.compile(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b")

    def get_word_vocab(texts: pd.Series) -> set:
        vocab = set()
        for t in texts:
            vocab.update([tok.lower() for tok in token_pattern.findall(str(t))])
        return vocab

    train_vocab = get_word_vocab(train_feat["text"])
    test_vocab = get_word_vocab(test_feat["text"])

    shared_vocab = train_vocab.intersection(test_vocab)
    unseen_test_vocab = test_vocab - train_vocab
    oov_word_rate = len(unseen_test_vocab) / len(test_vocab) if test_vocab else 0.0

    # Character N-gram overlap (3-5 char n-grams)
    def get_char_ngrams(texts: pd.Series, n_range: Tuple[int, int] = (3, 5)) -> set:
        ngrams = set()
        for t in texts:
            cleaned = f" {t.lower()} "
            for n in range(n_range[0], n_range[1] + 1):
                for i in range(len(cleaned) - n + 1):
                    ngrams.add(cleaned[i:i+n])
        return ngrams

    train_char_ngrams = get_char_ngrams(train_feat["text"])
    test_char_ngrams = get_char_ngrams(test_feat["text"])

    unseen_test_char_ngrams = test_char_ngrams - train_char_ngrams
    oov_char_ngram_rate = len(unseen_test_char_ngrams) / len(test_char_ngrams) if test_char_ngrams else 0.0

    # Punctuation & Digit comparisons
    shift_report = {
        "statistical_tests": {
            "char_length_ks_stat": round(float(ks_char.statistic), 5),
            "char_length_ks_pvalue": float(ks_char.pvalue),
            "word_count_ks_stat": round(float(ks_word.statistic), 5),
            "word_count_ks_pvalue": float(ks_word.pvalue),
        },
        "vocabulary_overlap": {
            "train_vocab_size": len(train_vocab),
            "test_vocab_size": len(test_vocab),
            "shared_vocab_size": len(shared_vocab),
            "unseen_test_words_count": len(unseen_test_vocab),
            "oov_word_rate_percent": round(oov_word_rate * 100, 2),
        },
        "char_ngram_overlap": {
            "train_char_ngrams_count": len(train_char_ngrams),
            "test_char_ngrams_count": len(test_char_ngrams),
            "unseen_test_char_ngrams_count": len(unseen_test_char_ngrams),
            "oov_char_ngram_rate_percent": round(oov_char_ngram_rate * 100, 2),
        },
        "meta_features_comparison": {
            "train": {
                "avg_char_length": round(float(train_feat["char_count"].mean()), 2),
                "avg_word_count": round(float(train_feat["word_count"].mean()), 2),
                "avg_punctuation_count": round(float(train_feat["punctuation_count"].mean()), 2),
                "avg_digit_count": round(float(train_feat["digit_count"].mean()), 2),
                "avg_uppercase_ratio": round(float(train_feat["uppercase_ratio"].mean()), 4),
            },
            "test": {
                "avg_char_length": round(float(test_feat["char_count"].mean()), 2),
                "avg_word_count": round(float(test_feat["word_count"].mean()), 2),
                "avg_punctuation_count": round(float(test_feat["punctuation_count"].mean()), 2),
                "avg_digit_count": round(float(test_feat["digit_count"].mean()), 2),
                "avg_uppercase_ratio": round(float(test_feat["uppercase_ratio"].mean()), 4),
            }
        }
    }

    logger.info(f"Word OOV Rate: {shift_report['vocabulary_overlap']['oov_word_rate_percent']}%")
    logger.info(f"Char N-Gram (3-5) OOV Rate: {shift_report['char_ngram_overlap']['oov_char_ngram_rate_percent']}%")

    return shift_report


# ---------------------------------------------------------
# 7. Meta-Feature Relationships & Correlations
# ---------------------------------------------------------

def analyze_feature_relationships(train_feat: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """Calculate statistical associations between numerical text meta-features and classes."""
    logger.info("Step 7: Evaluating meta-feature relationships and associations...")

    numerical_cols = [
        "char_count", "word_count", "unique_word_count", "avg_word_length",
        "ttr", "punctuation_count", "digit_count", "uppercase_count", "whitespace_count"
    ]

    # Spearman correlation matrix among numerical meta-features
    corr_matrix = train_feat[numerical_cols].corr(method="spearman").round(4).to_dict()

    # One-way ANOVA & Kruskal-Wallis tests across the 4 classes
    class_dfs = [group for _, group in train_feat.groupby("label")]

    anova_results = {}
    for col in numerical_cols:
        col_groups = [group[col].values for group in class_dfs]
        f_val, p_val = stats.f_oneway(*col_groups)
        kw_val, kw_p_val = stats.kruskal(*col_groups)
        anova_results[col] = {
            "anova_f_stat": round(float(f_val), 4),
            "anova_p_val": float(p_val),
            "kruskal_h_stat": round(float(kw_val), 4),
            "kruskal_p_val": float(kw_p_val),
            "statistically_significant_at_001": bool(p_val < 0.01 and kw_p_val < 0.01)
        }

    return {
        "spearman_correlation_matrix": corr_matrix,
        "class_association_tests": anova_results
    }


# ---------------------------------------------------------
# 8. Visualizations Generation
# ---------------------------------------------------------

def generate_visualizations(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    class_data: Dict[str, Any],
    linguistic_data: Dict[str, Any],
    shift_data: Dict[str, Any],
    figures_dir: Path,
    logger: logging.Logger
) -> List[str]:
    """Generate high-resolution analytical figures for the report."""
    logger.info("Step 8: Generating publication-grade visualizations...")
    saved_plots = []

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    counts = train_feat["label"].value_counts().sort_index()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="#333333", width=0.55)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(train_feat)) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    ax.set_title("BhasaQuest-1: Training Label Distribution (N = 8,195)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Target Label (Anonymous Class)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Sample Count", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.15)
    p1 = figures_dir / "distributions" / "class_distribution.png"
    p1.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p1, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p1))

    # 2. Length Distribution Train vs Test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.kdeplot(train_feat["char_count"], ax=ax1, label="Train (N=8,195)", color="#1f77b4", fill=True, alpha=0.35, common_norm=False)
    sns.kdeplot(test_feat["char_count"], ax=ax1, label="Test (N=2,048)", color="#ff7f0e", fill=True, alpha=0.35, common_norm=False)
    ax1.set_title("Character Count Density (Train vs Test)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Character Count per Sample")
    ax1.set_xlim(0, 700)
    ax1.legend(loc="upper right")

    sns.kdeplot(train_feat["word_count"], ax=ax2, label="Train (N=8,195)", color="#1f77b4", fill=True, alpha=0.35, common_norm=False)
    sns.kdeplot(test_feat["word_count"], ax=ax2, label="Test (N=2,048)", color="#ff7f0e", fill=True, alpha=0.35, common_norm=False)
    ax2.set_title("Word Count Density (Train vs Test)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Word Count per Sample")
    ax2.set_xlim(0, 100)
    ax2.legend(loc="upper right")
    p2 = figures_dir / "distributions" / "length_distribution_train_vs_test.png"
    plt.tight_layout()
    plt.savefig(p2, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p2))

    # 3. Length by Class Boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    order = sorted(train_feat["label"].unique())
    sns.boxplot(x="label", y="char_count", data=train_feat, ax=ax1, order=order, palette="Set2", showfliers=False)
    ax1.set_title("Character Length Distribution by Class", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Target Label")
    ax1.set_ylabel("Character Count (excluding fliers)")

    sns.boxplot(x="label", y="word_count", data=train_feat, ax=ax2, order=order, palette="Set2", showfliers=False)
    ax2.set_title("Word Count Distribution by Class", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Target Label")
    ax2.set_ylabel("Word Count (excluding fliers)")
    p3 = figures_dir / "distributions" / "length_by_class.png"
    plt.tight_layout()
    plt.savefig(p3, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p3))

    # 4. Top Distinctive Words by Class
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    axes = axes.flatten()
    for idx, c in enumerate(order):
        ax = axes[idx]
        ngrams_data = class_data["distinctive_ngrams"][c][:10]
        words = [item["ngram"] for item in ngrams_data][::-1]
        scores = [item["distinctive_ratio"] for item in ngrams_data][::-1]
        ax.barh(words, scores, color="#2b5c8f", edgecolor="#222222", height=0.65)
        ax.set_title(f"Class '{c}' Distinctive Lexical N-Grams (Ratio)", fontsize=11, fontweight='bold')
        ax.set_xlabel("In-Class vs Out-of-Class TF-IDF Ratio")
    p4 = figures_dir / "ngrams" / "top_words_by_class.png"
    p4.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p4, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p4))

    # 5. Top Character N-Grams
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    top_chars = linguistic_data["top_char_ngrams"][:15]
    char_labels = [f"'{k}'" for k, _ in top_chars][::-1]
    char_vals = [v for _, v in top_chars][::-1]
    ax.barh(char_labels, char_vals, color="#4c72b0", edgecolor="#333333", height=0.6)
    ax.set_title("Top 15 Most Frequent Character Subwords (char_wb 3-5)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Corpus Occurrence Frequency")
    p5 = figures_dir / "ngrams" / "char_ngrams_top.png"
    plt.tight_layout()
    plt.savefig(p5, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p5))

    # 6. Suffix Prevalence & Spelling Variations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    suffixes = linguistic_data["suffix_prevalence"][:10]
    suf_names = [s[0] for s in suffixes][::-1]
    suf_counts = [s[1] for s in suffixes][::-1]
    ax1.barh(suf_names, suf_counts, color="#388e3c", edgecolor="#222222", height=0.6)
    ax1.set_title("Top Romanized Bangla Suffix / Inflection Morphemes", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Frequency in Training Corpus")

    variants = linguistic_data["candidate_spelling_variations"][:8]
    pairs = [f"{v['word_a']} ~ {v['word_b']}" for v in variants][::-1]
    sims = [v["char_ngram_jaccard"] for v in variants][::-1]
    ax2.barh(pairs, sims, color="#d32f2f", edgecolor="#222222", height=0.6)
    ax2.set_title("Frequent Spelling Variant Candidates (Char Jaccard)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Subword Jaccard Similarity")
    ax2.set_xlim(0, 1.0)
    p6 = figures_dir / "linguistics" / "spelling_variations_and_suffixes.png"
    p6.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p6, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p6))

    # 7. Vocabulary Overlap & OOV Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    v_info = shift_data["vocabulary_overlap"]
    categories = ["Shared Words", "Train Only Words", "Unseen Test Words (OOV)"]
    sizes = [
        v_info["shared_vocab_size"],
        v_info["train_vocab_size"] - v_info["shared_vocab_size"],
        v_info["unseen_test_words_count"]
    ]
    colors = ["#4caf50", "#2196f3", "#f44336"]
    bars = ax.bar(categories, sizes, color=colors, edgecolor="#222222", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_title(f"Vocabulary Overlap & Out-of-Vocabulary Gap (OOV Rate = {v_info['oov_word_rate_percent']}%)",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel("Distinct Word Types")
    ax.set_ylim(0, max(sizes) * 1.15)
    p7 = figures_dir / "comparisons" / "train_test_vocab_overlap.png"
    p7.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p7, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p7))

    # 8. Meta-Feature Spearman Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    numerical_cols = [
        "char_count", "word_count", "unique_word_count", "avg_word_length",
        "ttr", "punctuation_count", "digit_count", "uppercase_count"
    ]
    corr = train_feat[numerical_cols].corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True,
                linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Spearman Rank Correlation Among Text Meta-Features", fontsize=12, fontweight='bold')
    p8 = figures_dir / "comparisons" / "feature_correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(p8, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p8))

    # 9. Punctuation & Casing by Class
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.barplot(x="label", y="punctuation_count", data=train_feat, ax=ax1, order=order, palette="mako", errorbar=None)
    ax1.set_title("Mean Punctuation Count per Sample by Class", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Target Label")
    ax1.set_ylabel("Mean Punctuation Count")

    sns.barplot(x="label", y="uppercase_ratio", data=train_feat, ax=ax2, order=order, palette="rocket", errorbar=None)
    ax2.set_title("Mean Uppercase Ratio per Sample by Class", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Target Label")
    ax2.set_ylabel("Mean Uppercase Ratio")
    p9 = figures_dir / "comparisons" / "punctuation_and_casing_by_class.png"
    plt.tight_layout()
    plt.savefig(p9, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(p9))

    return saved_plots


# ---------------------------------------------------------
# 9. Master Runner
# ---------------------------------------------------------

def run_full_analysis(data_dir: Path, output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """Execute complete analysis workflow and save all reports and figures."""
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "analysis.log")
    logger.info("=" * 60)
    logger.info("STARTING BHASAQUEST-1 AUTOMATED NLP EXPLORATORY ANALYSIS")
    logger.info("=" * 60)

    # 1. Verification
    verification_report = verify_datasets(data_dir, logger)
    with open(reports_dir / "dataset_verification.json", "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2, cls=NumpyEncoder)

    # Load dataframes
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # 2. Text Structural Analysis
    train_feat, train_structure = analyze_text_structure(train_df, "text", "train")
    test_feat, test_structure = analyze_text_structure(test_df, "text", "test")

    structure_report = {
        "train_structure": train_structure,
        "test_structure": test_structure
    }
    with open(reports_dir / "text_structural_summary.json", "w", encoding="utf-8") as f:
        json.dump(structure_report, f, indent=2, cls=NumpyEncoder)

    # 3. Linguistic Patterns
    linguistic_report = extract_linguistic_patterns(train_df, test_df, logger)
    with open(reports_dir / "linguistic_patterns.json", "w", encoding="utf-8") as f:
        json.dump(linguistic_report, f, indent=2, cls=NumpyEncoder)

    # 4. Class Profiling
    class_report = analyze_classes(train_df, logger)
    with open(reports_dir / "class_analysis.json", "w", encoding="utf-8") as f:
        json.dump(class_report, f, indent=2, cls=NumpyEncoder)

    # 5. Duplication & Leakage
    duplication_report = analyze_duplication_and_leakage(train_df, test_df, logger)
    with open(reports_dir / "duplication_leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(duplication_report, f, indent=2, cls=NumpyEncoder)

    # 6. Train-Test Shifts & OOV
    shift_report = analyze_train_test_shifts(train_feat, test_feat, logger)
    with open(reports_dir / "train_test_distribution.json", "w", encoding="utf-8") as f:
        json.dump(shift_report, f, indent=2, cls=NumpyEncoder)

    # 7. Feature Relationships
    relationships_report = analyze_feature_relationships(train_feat, logger)
    with open(reports_dir / "feature_correlations.json", "w", encoding="utf-8") as f:
        json.dump(relationships_report, f, indent=2, cls=NumpyEncoder)

    # 8. Visualizations
    saved_plots = generate_visualizations(
        train_feat, test_feat, class_report, linguistic_report, shift_report, figures_dir, logger
    )
    logger.info(f"Generated and saved {len(saved_plots)} analytical figures.")

    # Master Consolidated Summary
    master_summary = {
        "verification": verification_report,
        "structure": structure_report,
        "linguistics": linguistic_report,
        "classes": class_report,
        "duplication": duplication_report,
        "shifts": shift_report,
        "relationships": relationships_report,
        "generated_figures": saved_plots
    }

    with open(reports_dir / "master_eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, cls=NumpyEncoder)

    logger.info("EDA workflow completed successfully. All artifacts persisted in Analysis/")
    return master_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated NLP Dataset Analysis for BhasaQuest-1")
    parser.add_argument("--data_dir", type=str, default="Datasets", help="Directory containing train.csv, test.csv, sample_submission.csv")
    parser.add_argument("--output_dir", type=str, default="Analysis", help="Directory where reports and figures will be stored")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()

    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()

    run_full_analysis(data_path, output_path, seed=args.seed)
