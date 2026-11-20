"""
BhasaQuest-1: Configurable Feature Engineering Pipeline
======================================================
This script provides modular text transformation and feature extraction for
Romanized Bangla multi-class classification.

Key capabilities:
- Non-destructive text preservation (text_raw, text_lower, text_normalized, text_conservative_clean)
- Numerical text meta-feature extraction (length, ratios, character categories)
- Configurable Word-level TF-IDF sparse matrix generation
- Configurable Word-boundary Character-level TF-IDF sparse matrix generation
- Sparse feature concatenation and independent family selection
- Exporting artifacts (vectorizers, feature manifests, and sparse .npz matrices)
"""

import os
import re
import sys
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def setup_logger(log_file: Path) -> logging.Logger:
    """Set up logging to console and file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("CleanDataset")
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


# ---------------------------------------------------------
# 1. Text Normalization Functions
# ---------------------------------------------------------

def normalize_unicode(text: str, form: str = "NFKD") -> str:
    """Normalize unicode representations."""
    return unicodedata.normalize(form, str(text))


def collapse_repeated_characters(text: str, threshold: int = 2) -> str:
    """Collapse elongated character sequences (e.g., 'khuuuub' -> 'khuub')."""
    pattern = re.compile(r"([a-zA-Z])\1{" + str(threshold) + r",}")
    return pattern.sub(r"\1" * threshold, text)


def conservative_clean(text: str) -> str:
    """Strip extraneous whitespace while strictly preserving all punctuation and casing."""
    s = str(text) if text is not None else ""
    return re.sub(r"\s+", " ", s).strip()


def normalize_romanized_bangla(
    text: str,
    lowercase: bool = True,
    normalize_repeated: bool = True,
    threshold: int = 2,
    unicode_form: str = "NFKD"
) -> str:
    """
    Standardize Romanized Bangla text while preserving hyphens and meaningful structure.
    Does NOT remove punctuation or digits destructively.
    """
    s = str(text) if text is not None else ""
    s = normalize_unicode(s, unicode_form)
    if lowercase:
        s = s.lower()
    if normalize_repeated:
        s = collapse_repeated_characters(s, threshold)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_text_representations(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Create parallel normalized representations preserving original text."""
    res = pd.DataFrame(index=df.index)
    res["id"] = df["id"]
    if "label" in df.columns:
        res["label"] = df["label"]

    res["text_raw"] = df[text_col].fillna("").astype(str)
    res["text_lower"] = res["text_raw"].str.lower()
    res["text_conservative_clean"] = res["text_raw"].apply(conservative_clean)
    res["text_normalized"] = res["text_raw"].apply(
        lambda t: normalize_romanized_bangla(t, lowercase=True, normalize_repeated=True, threshold=2)
    )
    return res


# ---------------------------------------------------------
# 2. Basic Numerical Text Features
# ---------------------------------------------------------

def extract_numerical_features(texts: pd.Series) -> pd.DataFrame:
    """Compute dense structural and lexical metrics."""
    feats = pd.DataFrame(index=texts.index)
    str_series = texts.fillna("").astype(str)

    feats["num_char_count"] = str_series.str.len()
    words_series = str_series.apply(lambda s: s.strip().split())
    feats["num_word_count"] = words_series.apply(len)
    feats["num_unique_words"] = words_series.apply(lambda toks: len(set(toks)))
    feats["num_ttr"] = (feats["num_unique_words"] / feats["num_word_count"].replace(0, 1)).astype(np.float32)

    feats["num_avg_word_len"] = words_series.apply(
        lambda toks: np.mean([len(t) for t in toks]) if toks else 0.0
    ).astype(np.float32)
    feats["num_max_word_len"] = words_series.apply(
        lambda toks: np.max([len(t) for t in toks]) if toks else 0
    ).astype(np.int32)

    feats["num_uppercase_count"] = str_series.apply(lambda s: sum(1 for c in s if c.isupper())).astype(np.int32)
    feats["num_digit_count"] = str_series.apply(lambda s: sum(1 for c in s if c.isdigit())).astype(np.int32)
    feats["num_punctuation_count"] = str_series.apply(
        lambda s: sum(1 for c in s if not c.isalnum() and not c.isspace())
    ).astype(np.int32)
    feats["num_whitespace_count"] = str_series.apply(lambda s: sum(1 for c in s if c.isspace())).astype(np.int32)

    feats["num_uppercase_ratio"] = (feats["num_uppercase_count"] / feats["num_char_count"].replace(0, 1)).astype(np.float32)
    feats["num_digit_ratio"] = (feats["num_digit_count"] / feats["num_char_count"].replace(0, 1)).astype(np.float32)
    feats["num_punctuation_ratio"] = (feats["num_punctuation_count"] / feats["num_char_count"].replace(0, 1)).astype(np.float32)

    feats["flag_has_repeated_chars"] = str_series.apply(
        lambda s: 1 if re.search(r"([a-zA-Z])\1{2,}", s) else 0
    ).astype(np.int8)
    feats["flag_has_hyphen"] = str_series.str.contains(r"-", regex=True).astype(np.int8)
    feats["flag_has_question"] = str_series.str.contains(r"\?", regex=True).astype(np.int8)
    feats["flag_has_exclamation"] = str_series.str.contains(r"!", regex=True).astype(np.int8)

    return feats


# ---------------------------------------------------------
# 3. TF-IDF Extractors
# ---------------------------------------------------------

def fit_word_tfidf(
    train_texts: pd.Series,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 15000,
    min_df: int = 2,
    max_df: float = 0.85,
    sublinear_tf: bool = True,
    token_pattern: str = r"(?u)\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b"
) -> TfidfVectorizer:
    """Fit word-level TF-IDF vectorizer on training texts."""
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        token_pattern=token_pattern
    )
    vec.fit(train_texts)
    return vec


def fit_char_tfidf(
    train_texts: pd.Series,
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: int = 35000,
    min_df: int = 3,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """Fit character-level (char_wb) TF-IDF vectorizer on training texts."""
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf
    )
    vec.fit(train_texts)
    return vec


# ---------------------------------------------------------
# 4. Pipeline Execution
# ---------------------------------------------------------

def run_feature_pipeline(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    feature_group: str = "all",
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Execute complete feature extraction pipeline."""
    if logger is None:
        logger = setup_logger(output_dir / "logs" / "clean_dataset.log")

    logger.info("=" * 60)
    logger.info("STARTING BHASAQUEST-1 FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Directories
    processed_dir = output_dir / "data" / "processed"
    features_dir = output_dir / "data" / "features"
    manifests_dir = output_dir / "data" / "manifests"
    artifacts_dir = output_dir / "artifacts"
    reports_dir = output_dir / "reports"

    for d in [processed_dir, features_dir, manifests_dir, artifacts_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load raw data
    train_path = data_dir / cfg["dataset"]["train_file"]
    test_path = data_dir / cfg["dataset"]["test_file"]

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Raw data not found in {data_dir.resolve()}")

    logger.info(f"Loading raw datasets from: {data_dir.resolve()}")
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    # 1. Text Normalization
    logger.info("Building parallel text representations...")
    train_processed = build_text_representations(train_raw, cfg["dataset"]["text_col"])
    test_processed = build_text_representations(test_raw, cfg["dataset"]["text_col"])

    train_proc_file = processed_dir / "train_processed_text.csv"
    test_proc_file = processed_dir / "test_processed_text.csv"
    train_processed.to_csv(train_proc_file, index=False)
    test_processed.to_csv(test_proc_file, index=False)
    logger.info(f"Saved processed text: {train_proc_file.name}, {test_proc_file.name}")

    manifest_outputs = {
        "train_rows": len(train_processed),
        "test_rows": len(test_processed),
        "representations": ["text_raw", "text_lower", "text_conservative_clean", "text_normalized"]
    }

    # 2. Numerical Meta-Features
    if cfg["numerical_features"]["enabled"]:
        logger.info("Extracting numerical meta-features...")
        train_num = extract_numerical_features(train_raw[cfg["dataset"]["text_col"]])
        test_num = extract_numerical_features(test_raw[cfg["dataset"]["text_col"]])

        train_num.insert(0, "id", train_raw["id"])
        test_num.insert(0, "id", test_raw["id"])

        train_num_file = features_dir / "train_numerical_features.csv"
        test_num_file = features_dir / "test_numerical_features.csv"
        train_num.to_csv(train_num_file, index=False)
        test_num.to_csv(test_num_file, index=False)
        logger.info(f"Saved numerical features: {train_num.shape[1]-1} columns each.")
        manifest_outputs["numerical_features_count"] = train_num.shape[1] - 1

    # 3. TF-IDF Fit and Extraction (Full dataset for inference & standalone experiments)
    # NOTE: In CV validation (Submission/create_submission.py), vectorizers are fit strictly inside folds!
    if cfg["word_tfidf"]["enabled"]:
        logger.info("Fitting word TF-IDF vectorizer (1-2 ngrams)...")
        w_cfg = cfg["word_tfidf"]
        word_vec = fit_word_tfidf(
            train_processed["text_normalized"],
            ngram_range=tuple(w_cfg["ngram_range"]),
            max_features=w_cfg["max_features"],
            min_df=w_cfg["min_df"],
            max_df=w_cfg["max_df"],
            sublinear_tf=w_cfg["sublinear_tf"],
            token_pattern=w_cfg["token_pattern"]
        )
        joblib.dump(word_vec, artifacts_dir / "word_vectorizer.joblib")

        X_train_word = word_vec.transform(train_processed["text_normalized"])
        X_test_word = word_vec.transform(test_processed["text_normalized"])

        sparse.save_npz(features_dir / "train_word_tfidf.npz", X_train_word)
        sparse.save_npz(features_dir / "test_word_tfidf.npz", X_test_word)
        logger.info(f"Word TF-IDF shape: Train {X_train_word.shape}, Test {X_test_word.shape}")
        manifest_outputs["word_tfidf_vocab_size"] = len(word_vec.vocabulary_)

    if cfg["char_tfidf"]["enabled"]:
        logger.info("Fitting char_wb TF-IDF vectorizer (3-5 ngrams)...")
        c_cfg = cfg["char_tfidf"]
        char_vec = fit_char_tfidf(
            train_processed["text_normalized"],
            analyzer=c_cfg["analyzer"],
            ngram_range=tuple(c_cfg["ngram_range"]),
            max_features=c_cfg["max_features"],
            min_df=c_cfg["min_df"],
            sublinear_tf=c_cfg["sublinear_tf"]
        )
        joblib.dump(char_vec, artifacts_dir / "char_vectorizer.joblib")

        X_train_char = char_vec.transform(train_processed["text_normalized"])
        X_test_char = char_vec.transform(test_processed["text_normalized"])

        sparse.save_npz(features_dir / "train_char_tfidf.npz", X_train_char)
        sparse.save_npz(features_dir / "test_char_tfidf.npz", X_test_char)
        logger.info(f"Char TF-IDF shape: Train {X_train_char.shape}, Test {X_test_char.shape}")
        manifest_outputs["char_tfidf_vocab_size"] = len(char_vec.vocabulary_)

    # 4. Combined Features
    if cfg["word_tfidf"]["enabled"] and cfg["char_tfidf"]["enabled"]:
        logger.info("Assembling combined sparse word+char TF-IDF matrix...")
        X_train_comb = sparse.hstack([X_train_word, X_train_char], format="csr")
        X_test_comb = sparse.hstack([X_test_word, X_test_char], format="csr")

        sparse.save_npz(features_dir / "train_combined_tfidf.npz", X_train_comb)
        sparse.save_npz(features_dir / "test_combined_tfidf.npz", X_test_comb)
        logger.info(f"Combined TF-IDF shape: Train {X_train_comb.shape}, Test {X_test_comb.shape}")
        manifest_outputs["combined_feature_dim"] = X_train_comb.shape[1]

    # 5. Dataset Manifest
    manifest = {
        "project": "BhasaQuest-1",
        "timestamp": datetime.now().isoformat(),
        "input_directory": str(data_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "summary": manifest_outputs,
        "config_applied": cfg
    }

    manifest_file = manifests_dir / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Pipeline finished successfully. Manifest saved to: {manifest_file.resolve()}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline for BhasaQuest-1")
    parser.add_argument("--config", type=str, default="Featured Dataset/config.json", help="Path to config.json")
    parser.add_argument("--data_dir", type=str, default="Datasets", help="Directory containing raw CSVs")
    parser.add_argument("--output_dir", type=str, default="Featured Dataset", help="Output directory")
    parser.add_argument("--feature_group", type=str, default="all", help="Feature group to generate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()

    d_path = Path(args.data_dir)
    if not d_path.is_absolute():
        d_path = (project_root / d_path).resolve()

    out_path = Path(args.output_dir)
    if not out_path.is_absolute():
        out_path = (project_root / out_path).resolve()

    run_feature_pipeline(cfg_path, d_path, out_path, feature_group=args.feature_group)

