# Experiments Artifacts — `lab2_v2` (Oxford-IIIT Pets: Pre-trained CNNs)

## Purpose

Lưu trữ toàn bộ kết quả Data Engineering, split cố định, nhật ký huấn luyện & TensorBoard logs, bảng so sánh 9 runs, đánh giá Test cuối, Error Analysis 4 bước và biểu đồ của dự án **`lab2_v2`** (cô lập hoàn toàn khỏi `experiments/lab2/` v1).

## Directory Structure

```text
experiments/lab2_v2/
├── README.md
├── splits/
│   └── split_indices.json          # Split cố định seed=248 (Train=trainval 3,680 | Val=35% test 1,284 | Test=65% test 2,385)
├── eda/                            # [6.1–6.3 Data Engineering] sinh bởi prepare_data.py
│   ├── collection_report.json
│   ├── cleaning_report.json
│   ├── eda_summary.json
│   └── eda_*.png                   # 5 biểu đồ EDA
├── tb_logs/<run_id>/               # [7.3 Training Monitoring] TensorBoard event logs
├── runs/<run_id>/                  # [7–8.2 Training & Per-Run Val Eval]
│   ├── config.json
│   ├── best.pt / last.pt           # Trọng số checkpoint (gitignored)
│   ├── historys/history.{json,csv}
│   ├── eval/                       # Đánh giá trên Val (1,284 mẫu) ngay sau khi train
│   └── test_eval/                  # CHỈ có ở 2 finalist (Baseline Scratch & Best Pre-trained) — Test 1 lần cuối
├── ablation/
│   ├── sanity_checks.json          # [Layer 4] Initial loss & overfit-1-batch verification
│   ├── ablation_summary.json       # Bảng xếp hạng theo (val_acc, val_macro_f1, -val_loss)
│   └── ablation_table.csv
├── evaluation/
│   ├── test_comparison.{json,csv}
│   ├── per_class_metrics.json
│   ├── species_analysis.json       # Phân tích Cat (12 giống) vs Dog (25 giống)
│   ├── inference_benchmark.json    # [Layer 10] So sánh latency BS=1/BS=32 & checkpoint size đa kiến trúc
│   ├── error_analysis.json
│   ├── error_audit_60_candidates.json
│   ├── error_taxonomy_60.{csv,json}
│   ├── slice_analysis.json
│   └── audit_sheets/               # 3 bảng ảnh gốc kiểm toán 60 lỗi
└── figures/                        # Bộ biểu đồ tổng hợp cho báo cáo & thuyết trình
```

## Related Modules

- CLI Entrypoints & Config: [`labs/lab2_v2/README.md`](../../labs/lab2_v2/README.md)
- Reusable Core Engine: [`src/vibe_scripts/lab2_v2/README.md`](../../src/vibe_scripts/lab2_v2/README.md)
- Tài liệu kỹ thuật & Báo cáo: [`labs/lab2_v2/docs/README.md`](../../labs/lab2_v2/docs/README.md)
