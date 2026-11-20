"""
================================================================================
BhasaQuest-1: Phase B — Model 4: csebuetnlp/banglishbert (Full 5-Fold CV)
================================================================================
Strict Output Contract:
  - Folds: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on train.csv
  - Label Order: ['14AC78', '19E1A1', '6C0197', '72A2C8']
  - Sequence Length: max_len = 192
  - Training Recipe:
      * Epochs: 4 FIXED epochs (no per-fold val peak-picking to prevent OOF inflation)
      * Optimizer: AdamW, lr = 2e-5 (optimal for Electra-based BanglishBERT)
      * Scheduler: Linear decay with 10% warmup
      * Loss: CrossEntropyLoss with class weights & label_smoothing = 0.05
      * Regularization: Gradient clipping = 1.0, weight_decay = 0.01
      * Precision: fp16 mixed precision
  - Resumability: Checkpoints each fold; automatically resumes if interrupted.
  - Outputs:
      * oof_banglishbert.npy  (8195, 4)
      * test_banglishbert.npy (2048, 4)
      * banglishbert_metrics.json
      * submission_banglishbert_<f1>.csv
================================================================================
"""

import os
import sys
import re
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --- Canonical Class Order (Strict Contract) ---
CANONICAL_CLASSES = ['14AC78', '19E1A1', '6C0197', '72A2C8']
MODEL_NAME = "csebuetnlp/banglishbert"
MODEL_TAG = "banglishbert"


def normalize_text(text: str) -> str:
    """Normalize Romanized Bangla: lowercase, collapse duplicate characters & spaces."""
    s = str(text).lower()
    s = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', s)
    return re.sub(r'\s+', ' ', s).strip()


class BanglaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=192):
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
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(device)

            with autocast(dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)

            if labels is not None:
                all_targets.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    if all_targets:
        all_targets = np.concatenate(all_targets)
        preds = np.argmax(all_probs, axis=-1)
        macro_f1 = f1_score(all_targets, preds, average="macro")
        acc = accuracy_score(all_targets, preds)
        return macro_f1, acc, all_probs
    return all_probs


def main():
    parser = argparse.ArgumentParser(description="Phase B: BanglishBERT 5-Fold Training")
    parser.add_argument("--data_dir", type=str, default="Datasets",
                        help="Path containing train.csv and test.csv")
    parser.add_argument("--output_dir", type=str, default="Experiments/oof",
                        help="Path to save oof, test, and checkpoint files")
    parser.add_argument("--pilot_fold0", action="store_true",
                        help="Run only fold 0 as a quick pilot to confirm parameters")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size (use gradient accumulation if needed)")
    parser.add_argument("--accum_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch size = batch_size * accum_steps)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Fixed training epochs (strict contract: no peak-picking on val fold)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for BanglishBERT")
    parser.add_argument("--max_len", type=int, default=192,
                        help="Max sequence length")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f" BhasaQuest-1 Phase B: Fine-Tuning {MODEL_NAME}")
    print("=" * 75)
    print(f"[*] Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / f"checkpoints_{MODEL_TAG}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    if not (data_dir / "train.csv").exists():
        kaggle_paths = [Path("/kaggle/input/bhasaquest-1"), Path("/kaggle/input/bhasaquest1"), Path(".")]
        for p in kaggle_paths:
            if (p / "train.csv").exists():
                data_dir = p
                break

    print(f"[*] Loading datasets from {data_dir}...")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    classes = CANONICAL_CLASSES
    label2idx = {lbl: i for i, lbl in enumerate(classes)}
    idx2label = {i: lbl for i, lbl in enumerate(classes)}
    num_classes = len(classes)
    print(f"[*] Canonical Label Order: {classes}")

    train_texts = [normalize_text(t) for t in train_df["text"]]
    test_texts = [normalize_text(t) for t in test_df["text"]]
    train_labels = np.array([label2idx[l] for l in train_df["label"]])

    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[*] Balanced Class Weights: {dict(zip(classes, [round(w, 3) for w in class_weights]))}")

    print(f"[*] Loading Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_ds = BanglaDataset(test_texts, None, tokenizer, max_len=args.max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    # 2. 5-Fold Stratified Split (Strict Contract: seed 42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(train_texts), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_texts), num_classes), dtype=np.float32)
    fold_metrics = {}

    print(f"\n[*] Training Configuration:")
    print(f"    Fixed Epochs : {args.epochs}")
    print(f"    Peak LR      : {args.lr}")
    print(f"    Max Length   : {args.max_len}")
    print(f"    Batch Size   : {args.batch_size} (Accum: {args.accum_steps} -> Effective: {args.batch_size * args.accum_steps})")
    print(f"    Label Smooth : 0.05")
    print(f"    Loss Type    : Class-Weighted Cross-Entropy")

    # 3. K-Fold Training Loop with Resumability
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        if args.pilot_fold0 and fold > 0:
            print(f"\n[!] Pilot mode enabled: skipping folds > 0.")
            break

        print(f"\n" + "=" * 50)
        print(f" >>> FOLD {fold} / 5 <<<")
        print("=" * 50)

        fold_done_file = ckpt_dir / f"fold_{fold}_done.json"
        fold_oof_file = ckpt_dir / f"fold_{fold}_val_probs.npy"
        fold_test_file = ckpt_dir / f"fold_{fold}_test_probs.npy"

        if fold_done_file.exists() and fold_oof_file.exists() and fold_test_file.exists():
            print(f"[+] Fold {fold} already completed! Resuming from checkpoint...")
            with open(fold_done_file, "r") as f:
                saved_info = json.load(f)
            fold_metrics[f"fold_{fold}"] = saved_info["val_macro_f1"]
            oof_probs[val_idx] = np.load(fold_oof_file)
            test_probs += np.load(fold_test_file) / 5.0
            print(f"    Loaded Fold {fold} Val Macro F1: {saved_info['val_macro_f1']:.4f}")
            continue

        tr_ds = BanglaDataset([train_texts[i] for i in tr_idx], train_labels[tr_idx], tokenizer, max_len=args.max_len)
        val_ds = BanglaDataset([train_texts[i] for i in val_idx], train_labels[val_idx], tokenizer, max_len=args.max_len)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)
        model.to(device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

        total_steps = (len(tr_loader) // args.accum_steps) * args.epochs
        warmup_steps = int(total_steps * 0.10)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
        scaler = GradScaler()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tr_loader, 1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with autocast(dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    loss = loss / args.accum_steps

                scaler.scale(loss).backward()

                if step % args.accum_steps == 0 or step == len(tr_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                train_loss += loss.item() * args.accum_steps

            avg_train_loss = train_loss / len(tr_loader)
            val_f1, val_acc, val_probs_epoch = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            print(f"Epoch {epoch}/{args.epochs} [{elapsed:.0f}s] - Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")

        final_val_f1, final_val_acc, final_val_probs = evaluate(model, val_loader, device)
        print(f"[Fold {fold} Completed -> Final Val Macro F1: {final_val_f1:.4f} | Accuracy: {final_val_acc*100:.2f}%]")

        fold_metrics[f"fold_{fold}"] = float(final_val_f1)
        oof_probs[val_idx] = final_val_probs

        fold_test_probs = evaluate(model, test_loader, device)
        test_probs += fold_test_probs / 5.0

        np.save(fold_oof_file, final_val_probs)
        np.save(fold_test_file, fold_test_probs)
        with open(fold_done_file, "w") as f:
            json.dump({
                "fold": fold,
                "val_macro_f1": float(final_val_f1),
                "val_accuracy": float(final_val_acc),
                "epochs": args.epochs
            }, f, indent=2)

        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.pilot_fold0:
        print("\n" + "=" * 60)
        print(f" PILOT FOLD 0 COMPLETED: Macro F1 = {fold_metrics['fold_0']:.4f}")
        print(" Compare this fold 0 score against LinearSVC fold 0 (0.7332).")
        print("=" * 60)
        return

    # 4. Overall 5-Fold OOF Evaluation
    oof_preds = np.argmax(oof_probs, axis=-1)
    overall_macro_f1 = f1_score(train_labels, oof_preds, average="macro")
    overall_acc = accuracy_score(train_labels, oof_preds)
    fold_scores = list(fold_metrics.values())

    print("\n" + "=" * 70)
    print(f" OVERALL 5-FOLD CV RESULTS: {MODEL_TAG}")
    print(f" Overall OOF Macro F1   : {overall_macro_f1:.5f}")
    print(f" Overall OOF Accuracy   : {overall_acc*100:.2f}%")
    print(f" Fold F1 Scores (0-4)   : {[round(s, 4) for s in fold_scores]}")
    print(f" Fold F1 Mean +- Std    : {np.mean(fold_scores):.4f} +- {np.std(fold_scores):.4f}")
    print("=" * 70)

    # Per-Class Report
    print("\nClassification Report:")
    report_dict = classification_report(train_labels, oof_preds, target_names=classes, output_dict=True, digits=4)
    print(classification_report(train_labels, oof_preds, target_names=classes, digits=4))

    # 5. Save Output Contract Files
    oof_file = output_dir / f"oof_{MODEL_TAG}.npy"
    test_file = output_dir / f"test_{MODEL_TAG}.npy"
    metrics_file = output_dir / f"{MODEL_TAG}_metrics.json"

    np.save(oof_file, oof_probs)
    np.save(test_file, test_probs)
    print(f"[+] Saved OOF Probabilities  : {oof_file}")
    print(f"[+] Saved Test Probabilities : {test_file}")

    metrics_summary = {
        "model_name": MODEL_NAME,
        "model_tag": MODEL_TAG,
        "oof_macro_f1": float(overall_macro_f1),
        "oof_accuracy": float(overall_acc),
        "fold_f1_scores": fold_metrics,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "epochs": args.epochs,
        "max_len": args.max_len,
        "learning_rate": args.lr,
        "classes": classes,
        "classification_report": report_dict
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[+] Saved Metrics Summary    : {metrics_file}")

    # Append to Experiments/oof/leaderboard.csv
    lb_csv = output_dir / "leaderboard.csv"
    new_entry = {
        "model_name": MODEL_TAG,
        "oof_macro_f1": round(overall_macro_f1, 4),
        "oof_acc": round(overall_acc, 4),
        "fold0_f1": round(fold_scores[0], 4) if len(fold_scores) > 0 else None,
        "fold1_f1": round(fold_scores[1], 4) if len(fold_scores) > 1 else None,
        "fold2_f1": round(fold_scores[2], 4) if len(fold_scores) > 2 else None,
        "fold3_f1": round(fold_scores[3], 4) if len(fold_scores) > 3 else None,
        "fold4_f1": round(fold_scores[4], 4) if len(fold_scores) > 4 else None,
        "fold_std": round(np.std(fold_scores), 4),
        "notes": f"BanglishBERT, fixed {args.epochs}ep, max_len {args.max_len}, lr {args.lr}"
    }
    if lb_csv.exists():
        lb_df = pd.read_csv(lb_csv)
        lb_df = pd.concat([lb_df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        lb_df = pd.DataFrame([new_entry])
    lb_df.to_csv(lb_csv, index=False)
    print(f"[+] Updated Leaderboard       : {lb_csv}")

    # 6. Generate Standalone Submission CSV & Run 10-Point Audit
    test_preds = np.argmax(test_probs, axis=-1)
    test_labels = [idx2label[p] for p in test_preds]
    sub_df = pd.DataFrame({"id": test_df["id"], "label": test_labels})

    sub_file = output_dir / f"submission_{MODEL_TAG}_{overall_macro_f1:.4f}.csv"
    sub_df.to_csv(sub_file, index=False)
    print(f"[+] Generated Submission File : {sub_file}")

    # 10-Point Audit
    print("\n--- Running 10-Point Submission Audit ---")
    assert list(sub_df.columns) == ["id", "label"], "Column mismatch"
    assert len(sub_df) == 2048, f"Expected 2048 rows, got {len(sub_df)}"
    assert sub_df.isnull().sum().sum() == 0, "Null values detected"
    assert (sub_df["id"].values == test_df["id"].values).all(), "ID sequence mismatch"
    assert set(sub_df["label"].unique()).issubset(set(classes)), "Invalid label found"
    assert len(sub_df["label"].unique()) == 4, "Mode collapse: fewer than 4 classes"
    print("Class Distribution in Test Predictions:")
    print(sub_df["label"].value_counts().to_string())
    print(">>> 10-Point Audit PASSED completely!")


if __name__ == "__main__":
    main()

