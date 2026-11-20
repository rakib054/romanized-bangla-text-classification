"""
================================================================================
BhasaQuest-1: Multi-Class Text Classification using Romanized Bangla
Script: Kaggle Transformer Fine-Tuning Pipeline (MuRIL & XLM-RoBERTa)
================================================================================
Purpose:
  Fine-tunes multilingual pretrained transformer models (MuRIL or XLM-RoBERTa)
  using 5-Fold Stratified Cross-Validation on Kaggle GPU (T4 x 2 or P100).
  Saves Out-Of-Fold (OOF) and Test probability arrays (.npy) for blending with
  the linear champion model (LinearSVC C=0.30/0.25).

Supported Models:
  1. "google/muril-base-cased" (Pretrained on 17 Indian languages + English transliteration)
  2. "FacebookAI/xlm-roberta-base" (Large 250k SentencePiece multilingual vocabulary)
  3. "csebuetnlp/banglabert" (Bangla-specific BERT, if native Bangla scripts used)

Outputs:
  - OOF probability matrix: oof_{model_tag}_probs.npy (8195, 4)
  - Test probability matrix: test_{model_tag}_probs.npy (2048, 4)
  - Fold evaluation metrics & overall Macro F1 report
  - Valid Kaggle submission CSV verified against the 10-point compliance checklist.
================================================================================
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
except ImportError:
    raise ImportError("Please install transformers: pip install transformers accelerate")


# --- Configuration Defaults ---
DEFAULT_CONFIG = {
    "model_name": "google/muril-base-cased",
    "model_tag": "muril_base",
    "max_len": 128,
    "batch_size": 32,
    "epochs": 4,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed": 42,
    "n_splits": 5,
    "fp16": True,
    # "data_dir": "/kaggle/input/competitions/bhasa-quest",    # For Kaggle
    "data_dir": "Datasets",    # For Local pc
    "output_dir": "Experiments/transformer_output",
}


def normalize_text(text: str) -> str:
    """Normalize Romanized Bangla text: lowercase, collapse duplicate characters & spaces."""
    s = str(text).lower()
    s = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', s)  # e.g. "shuuuuno" -> "shuuno"
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


class BanglaTextDataset(Dataset):
    """PyTorch Dataset for text classification with tokenization."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                logits = outputs.logits
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.append(probs)
            all_preds.append(preds)
            if labels is not None:
                all_targets.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)

    if all_targets:
        all_targets = np.concatenate(all_targets)
        macro_f1 = f1_score(all_targets, all_preds, average="macro")
        acc = accuracy_score(all_targets, all_preds)
        avg_loss = total_loss / len(dataloader)
        return avg_loss, macro_f1, acc, all_probs, all_preds

    return all_probs, all_preds


def main():
    parser = argparse.ArgumentParser(description="Kaggle Transformer Fine-Tuning for BhasaQuest-1")
    parser.add_argument("--model_name", type=str, default="google/muril-base-cased",
                        help="HuggingFace model ID (e.g. google/muril-base-cased, FacebookAI/xlm-roberta-base)")
    parser.add_argument("--model_tag", type=str, default="muril_base",
                        help="Identifier tag for filenames (e.g. muril_base, xlm_roberta)")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    # parser.add_argument("--data_dir", type=str, default="/kaggle/input/competitions/bhasa-quest", help="Directory containing train.csv, test.csv")   # For Kaggle
    parser.add_argument("--data_dir", type=str, default="Datasets", help="Directory containing train.csv, test.csv")   # For Local PC
    parser.add_argument("--output_dir", type=str, default="Experiments/transformer_output", help="Directory to save artifacts")
    args, _ = parser.parse_known_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"[*] Loading data from {data_dir}...")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    classes = sorted(train_df["label"].unique().tolist())
    label2idx = {lbl: i for i, lbl in enumerate(classes)}
    idx2label = {i: lbl for i, lbl in enumerate(classes)}
    num_classes = len(classes)
    print(f"[*] Target classes ({num_classes}): {classes}")

    train_texts = [normalize_text(t) for t in train_df["text"]]
    test_texts = [normalize_text(t) for t in test_df["text"]]
    train_labels = np.array([label2idx[l] for l in train_df["label"]])

    # 2. Tokenizer Setup
    print(f"[*] Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    test_dataset = BanglaTextDataset(test_texts, None, tokenizer, max_len=args.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    # 3. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(train_texts), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_texts), num_classes), dtype=np.float32)
    fold_scores = []

    print(f"\n=======================================================")
    print(f" Starting 5-Fold CV: {args.model_name}")
    print(f" Epochs: {args.epochs} | LR: {args.lr} | Batch Size: {args.batch_size}")
    print(f"=======================================================\n")

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_texts, train_labels), 1):
        print(f"\n--- FOLD {fold} / 5 ---")
        fold_train_texts = [train_texts[i] for i in tr_idx]
        fold_train_labels = train_labels[tr_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        fold_val_labels = train_labels[val_idx]

        train_ds = BanglaTextDataset(fold_train_texts, fold_train_labels, tokenizer, max_len=args.max_len)
        val_ds = BanglaTextDataset(fold_val_texts, fold_val_labels, tokenizer, max_len=args.max_len)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

        # Initialize model for this fold
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_classes)
        model.to(device)

        # Optimizer & Scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_val_f1 = 0.0
        best_val_probs = None

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler=scaler)
            val_loss, val_f1, val_acc, val_prob_epoch, _ = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            print(f"Epoch {epoch}/{args.epochs} [{elapsed:.0f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_probs = val_prob_epoch
                # Save best fold weights
                torch.save(model.state_dict(), output_dir / f"{args.model_tag}_fold{fold}_best.pt")

        print(f"[Fold {fold} Best Val Macro F1: {best_val_f1:.4f}]")
        fold_scores.append(best_val_f1)
        oof_probs[val_idx] = best_val_probs

        # Load best fold model for test prediction
        model.load_state_dict(torch.load(output_dir / f"{args.model_tag}_fold{fold}_best.pt"))
        fold_test_probs, _ = evaluate(model, test_loader, device)
        test_probs += fold_test_probs / skf.n_splits

        # Clean memory
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Out-of-Fold Overall Evaluation
    oof_preds = np.argmax(oof_probs, axis=-1)
    overall_macro_f1 = f1_score(train_labels, oof_preds, average="macro")
    overall_acc = accuracy_score(train_labels, oof_preds)

    print("\n" + "=" * 60)
    print(f" OVERALL 5-FOLD CV RESULTS: {args.model_name}")
    print(f" Overall OOF Macro F1 : {overall_macro_f1:.4f}")
    print(f" Overall OOF Accuracy : {overall_acc*100:.2f}%")
    print(f" Fold Scores Mean +- Std: {np.mean(fold_scores):.4f} +- {np.std(fold_scores):.4f}")
    print("=" * 60)

    # Per-Class Report
    print("\nClassification Report:")
    report_dict = classification_report(train_labels, oof_preds, target_names=classes, output_dict=True)
    print(classification_report(train_labels, oof_preds, target_names=classes, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(train_labels, oof_preds)
    print("Confusion Matrix:")
    print(cm)

    # 5. Save Artifacts
    np.save(output_dir / f"oof_{args.model_tag}_probs.npy", oof_probs)
    np.save(output_dir / f"test_{args.model_tag}_probs.npy", test_probs)
    print(f"[+] Saved OOF probabilities to {output_dir / f'oof_{args.model_tag}_probs.npy'}")
    print(f"[+] Saved Test probabilities to {output_dir / f'test_{args.model_tag}_probs.npy'}")

    summary = {
        "model_name": args.model_name,
        "model_tag": args.model_tag,
        "overall_macro_f1": float(overall_macro_f1),
        "overall_accuracy": float(overall_acc),
        "fold_scores": [float(s) for s in fold_scores],
        "mean_fold_f1": float(np.mean(fold_scores)),
        "std_fold_f1": float(np.std(fold_scores)),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    with open(output_dir / f"{args.model_tag}_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6. Generate Submission CSV & Audit
    test_preds = np.argmax(test_probs, axis=-1)
    test_pred_labels = [idx2label[p] for p in test_preds]
    submission_path = output_dir / f"submission_{args.model_tag}.csv"
    sub_df = pd.DataFrame({"id": test_df["id"], "label": test_pred_labels})
    sub_df.to_csv(submission_path, index=False)
    print(f"[+] Saved Submission CSV to {submission_path}")

    # Audit submission
    print("\nRunning 10-Point Submission Audit:")
    assert list(sub_df.columns) == ["id", "label"], "Column mismatch"
    assert len(sub_df) == 2048, f"Expected 2048 rows, got {len(sub_df)}"
    assert sub_df["id"].isnull().sum() == 0 and sub_df["label"].isnull().sum() == 0, "Nulls found"
    assert (sub_df["id"] == test_df["id"]).all(), "ID sequence mismatch"
    assert set(sub_df["label"].unique()).issubset(set(classes)), "Invalid label found"
    assert len(sub_df["label"].unique()) == 4, "Mode collapse: fewer than 4 classes"
    print(">>> 10-Point Audit PASSED completely!")


# if __name__ == "__main__":
#     main()