# `src/vibe_scripts/lab3_v2/` — Reusable Hugging Face Core Engine

## Purpose

Cung cấp toàn bộ mã nguồn lõi tái sử dụng (SRP, tách biệt khỏi thin CLIs và notebooks) cho **Lab 3 v2** (`0007_lab3_v2`): giải phẫu WordPiece tokenizer & chạy `pipeline` sentiment analysis (Ex1), kiểm toán/chia tập chống leakage trên IMDB, fine-tuning `distilbert-base-uncased` bằng Hugging Face `Trainer` chuẩn `transformers v5`, đánh giá Val/Test tách biệt, trích xuất lỗi và trực quan hóa biểu đồ.

## Responsibilities

| Module | Trách nhiệm chính |
| :--- | :--- |
| `ex1_sentiment.py` | Giải phẫu tokenizer (`tokens`, `input_ids`, `attention_mask`, `decode`, batch padding), nạp `pipeline` sentiment & xuất `ex1_results.json` |
| `data.py` | Tải IMDB (`datasets`), chia stratified subset cố định (`split_indices.json`, `seed=248`), 6.1–6.3 Data Audit & EDA (gồm thống kê truncation `256` tokens), và tokenization |
| `ex2_finetune.py` | Tensor Contract `(B, L) → (B, 2)`, `compute_metrics` (`evaluate` + fallback `sklearn`), `TrainingArguments` & `Trainer` chuẩn `transformers v5` (`eval_strategy`, `processing_class`), Val Baseline vs Post-Fine-Tune, Final Test 1 lần & offline inference |
| `analysis.py` | Vẽ biểu đồ headless (`Agg` guard cho Windows): `loss_curve.png`, `val_comparison.png`, `confusion_matrix.png` |

## Dependencies

- `torch`, `transformers >= 5.0`, `datasets`, `evaluate`, `accelerate`, `scikit-learn`, `matplotlib`, `numpy`.

## Usage

Gọi trực tiếp qua các thin CLI scripts tại [`labs/lab3_v2/README.md`](../../../labs/lab3_v2/README.md) hoặc import từ `src.vibe_scripts.lab3_v2`.
