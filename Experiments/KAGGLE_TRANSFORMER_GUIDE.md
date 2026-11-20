# Kaggle GPU Execution Guide: MuRIL & XLM-RoBERTa Fine-Tuning

> **Project**: BhasaQuest-1: Multi-Class Text Classification using Romanized Bangla  
> **Target Metric**: **Macro-Averaged F1 Score**  
> **Goal**: Train contextual multilingual transformers to break the $\sim 0.721$ baseline ceiling and resolve lexical ambiguity between classes `19E1A1` and `72A2C8`.

---

## 1. Why Fine-Tune on Kaggle?
- **Local PC Hardware**: Running CPU-only PyTorch (`2.10.0+cpu`). 5-fold transformer fine-tuning would take $>8$ hours locally on CPU.
- **Kaggle Hardware**: Free GPU T4 x 2 or P100 completes full 5-fold training in **~15–20 minutes** using mixed precision (FP16).

---

## 2. Recommended Pretrained Transformer Architectures

| Model Name | HuggingFace Identifier | Strengths for Romanized Bangla |
| :--- | :--- | :--- |
| **Google MuRIL** (Recommended First) | `google/muril-base-cased` | Pretrained specifically on 17 Indian languages and **English-transliterated text**, making its subword embeddings uniquely aligned with Romanized Bangla syntax and morphology. |
| **XLM-RoBERTa** | `FacebookAI/xlm-roberta-base` | Giant 250k SentencePiece multilingual vocabulary offering robust subword tokenization for rare/slang words. |
| **BanglaBERT** | `csebuetnlp/banglabert` | Best for native Bengali script, but can be tested with transliterated text. |

---

## 3. Step-by-Step Instructions to Run on Kaggle

### Running via Script (`train_transformer_kaggle.py`)
1. In a fresh Kaggle notebook, add your competition dataset as input.
2. Run in a cell:
   ```bash
   !python train_transformer_kaggle.py --model_name google/muril-base-cased --model_tag muril_base --epochs 4 --batch_size 32 --lr 2e-5 --data_dir /kaggle/input/bhasaquest-1
   ```

---

## 4. Expected Output Files & Transfer to Local Workspace

Once Kaggle training completes, download these generated files from the `/kaggle/working/transformer_output/` directory into your local `Experiments/transformer_output/` folder:

```text
Experiments/transformer_output/
├── oof_muril_base_probs.npy            # (8195, 4) Out-of-fold probability matrix
├── test_muril_base_probs.npy           # (2048, 4) Test set probability matrix
├── muril_base_evaluation_report.json   # Precision, Recall, F1 per class
├── muril_base_confusion_matrix.png     # High-resolution error breakdown
└── submission_muril_base.csv           # Audited stand-alone submission
```

---

## 5. Local Blending & Ensembling

Once the transformer probability files are placed in `Experiments/transformer_output/`:
1. Run the local ensembling pipeline:
   ```bash
   python Experiments/ensemble_and_calibrate.py
   ```
2. The script uses SLSQP to automatically solve for the optimal weight vector:
   $$P_{\text{ensemble}} = w_1 \cdot P_{\text{LinearSVC}} + w_2 \cdot P_{\text{LogReg}} + w_3 \cdot P_{\text{MuRIL}}$$
3. It will generate `Submission/submission_calibrated_ensemble.csv` and execute the 10-point audit checklist.

