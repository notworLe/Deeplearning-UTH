# Experiments Artifacts — `lab3_v2` (Hugging Face Sentiment & DistilBERT Fine-Tuning)

## Purpose

Lưu trữ toàn bộ kết quả Ex1, split cố định chống leakage, báo cáo/biểu đồ EDA và nhật ký huấn luyện/đánh giá Ex2 theo đúng bố cục **`MODEL_PIPELINE.md v1.4.0`**.

## Directory Structure

```text
experiments/lab3_v2/
├── README.md
├── ex1/
│   └── ex1_results.json               # Tokenizer anatomy, batch padding & sentiment predictions (Ex1)
├── splits/
│   └── split_indices.json             # Split cố định seed=248 (Train 4,000 / Val 1,000 / Test 1,000)
├── eda/
│   ├── collection_report.json         # [6.1] Nguồn & quy mô IMDB subset
│   ├── cleaning_report.json           # [6.2] Kiểm toán rỗng, thẻ <br />, trùng lặp nội bộ & chéo split
│   ├── eda_summary.json               # [6.3] Cân bằng nhãn & độ dài WordPiece token (>256)
│   ├── eda_class_distribution.png
│   └── eda_token_length.png
└── runs/
    └── distilbert_imdb/
        ├── monitor/                   # [7.3] trainer_log_history.json, train_summary.json, loss_curve.png, tb_logs/
        ├── val_eval/                  # [8.1] baseline_before_finetune.json, after_finetune_val.json, val_comparison.png
        ├── test_eval/                 # [8.2–9] final_test_metrics.json, confusion_matrix.png, error_examples.json (1 lần cuối)
        └── weights/                   # Checkpoint HF best_model/ (gitignored — không commit *.safetensors/*.bin)
```

## Related Modules

- Cấu hình, CLI & Notebooks: [`labs/lab3_v2/README.md`](../../labs/lab3_v2/README.md)
- Mã nguồn lõi tái sử dụng: [`src/vibe_scripts/lab3_v2/README.md`](../../src/vibe_scripts/lab3_v2/README.md)
- Tài liệu & 3 Báo cáo SRP: [`labs/lab3_v2/docs/README.md`](../../labs/lab3_v2/docs/README.md)
