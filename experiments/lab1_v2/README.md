# Experiments Artifacts — `lab1_v2` (FashionMNIST: Logistic Regression vs MLP)

## Purpose

Lưu trữ toàn bộ kết quả thực nghiệm, nhật ký huấn luyện/giám sát, bảng đối chiếu ablation, báo cáo đánh giá và biểu đồ của dự án **`lab1_v2`** (tách biệt hoàn toàn khỏi mã nguồn thực thi tại [`labs/lab1_v2/`](../../labs/lab1_v2/README.md) và [`src/vibe_scripts/lab1_v2/`](../../src/vibe_scripts/lab1_v2/README.md)).

## Responsibilities

- **Chứa:** kết quả Data Engineering (`eda/`), nhật ký & checkpoints từng run (`runs/`), tổng hợp chuỗi ablation (`ablation/`), đánh giá Test cuối & phân tích lỗi (`evaluation/`), biểu đồ báo cáo (`figures/`).
- **Không chứa:** mã nguồn production (đặt tại `src/vibe_scripts/lab1_v2/`) và file cấu hình gốc (đặt tại `labs/lab1_v2/config.py`). Các file trọng số `*.pt` được `.gitignore` loại trừ khỏi Git.

## Directory Structure

```text
experiments/lab1_v2/
├── README.md
├── eda/                            # [6.1–6.3 Data Engineering] sinh bởi labs/lab1_v2/prepare_data.py
│   ├── collection_report.json      # Kiểm toán nguồn dữ liệu & 8/8 SHA-256 manifest
│   ├── cleaning_report.json        # Kiểm toán ảnh lỗi, nhãn lỗi, exact duplicates
│   ├── eda_summary.json            # Phân bố lớp, pixel stats, outliers, prototype cosine similarity
│   └── eda_*.png                   # 5 biểu đồ EDA
│
├── runs/<run_id>/                  # [7. Training, Monitoring & Per-Run Eval] nhật ký riêng từng run
│   ├── config.json                 # Snapshot cấu hình & norm_stats của run
│   ├── best.pt                     # Checkpoint tốt nhất theo val_loss thấp nhất (gitignored)
│   ├── last.pt                     # Checkpoint epoch cuối (gitignored)
│   ├── historys/                   # Thống nhất toàn bộ log theo từng epoch (JSON + CSV)
│   │   ├── history.json            # Cấu trúc đầy đủ + best_epoch + tổng thời gian train
│   │   └── history.csv             # Dạng bảng mở nhanh (train/val loss, acc, lr, throughput, VRAM)
│   ├── eval/                       # Đánh giá trên tập VAL (5k) ngay sau khi run train xong
│   │   ├── eval_metrics.json       # Val Loss, Acc, Macro-F1, Per-class, CM 10×10, Save/Load parity
│   │   ├── loss_curve.png          # Train vs Val Loss theo epoch
│   │   ├── accuracy_curve.png      # Train vs Val Accuracy theo epoch
│   │   ├── confusion_matrix.png    # Confusion Matrix trên Val (5k)
│   │   └── predicted_vs_actual.png # Lưới ảnh dự đoán trên Val (5k)
│   └── test_eval/                  # CHỈ có ở 2 model chung kết (Baseline & Winning MLP) — TEST (10k) 1 lần cuối
│       ├── eval_metrics.json       # Test Loss, Acc, Macro-F1, Per-class, CM 10×10, Save/Load parity
│       ├── confusion_matrix.png    # Confusion Matrix trên Test (10k)
│       └── predicted_vs_actual.png # (KHÔNG có loss/acc curve vì Test chỉ chấm 1 lần trên best.pt)
│
├── ablation/                       # [8. Model Selection] tổng hợp chuỗi chạy (cập nhật sau mỗi run)
│   ├── ablation_summary.json       # Trạng thái dừng sớm, stop_reason, baseline_run_id, winning_mlp_run_id
│   └── ablation_table.csv          # Bảng so sánh toàn bộ runs đã chạy theo Val metrics
│
├── evaluation/                     # [8–9. Final Evaluation & Error Analysis] so sánh Baseline vs Winning MLP trên Test (10k)
│   ├── test_comparison.json        # Test Loss/Acc/Macro-F1 & gap (MLP − Baseline)
│   ├── test_comparison.csv         # Bảng đối chiếu CSV
│   ├── per_class_metrics.json      # Precision/Recall/F1 từng lớp (10 lớp) & delta F1
│   ├── error_analysis.json         # Top confusion pairs (kèm % hàng) & 3 lớp khó nhất
│   ├── error_taxonomy_60.{csv,json}# Gán nhãn nguyên nhân định lượng trên 60 mẫu lỗi Top-3 cặp
│   ├── slice_analysis.json         # Tỷ lệ lỗi theo độ sáng, kích thước vật thể & cường độ biên
│   └── audit_sheets/               # 3 bảng ảnh gốc (20 ảnh/cặp × Top-3 cặp = 60 mẫu kiểm toán)
│
└── figures/                        # Bộ biểu đồ tổng hợp dùng trực tiếp cho báo cáo & thuyết trình
    ├── ablation_comparison.png
    ├── loss_curves.png
    ├── accuracy_curves.png
    ├── confusion_matrix_baseline.png
    ├── confusion_matrix_mlp.png
    ├── confusion_matrix_mlp_normalized.png
    ├── predicted_vs_actual.png
    ├── error_gallery.png
    ├── error_gallery_top3_high_conf.png
    ├── error_gallery_top3_low_conf.png
    ├── error_taxonomy_breakdown.png
    └── error_slice_analysis.png
```

## Anti-Leakage Contract (`historys/` vs `eval/` vs `test_eval/`)

| Thành phần | Thời điểm ghi | Tập dữ liệu | Mục đích |
| :--- | :--- | :---: | :--- |
| `runs/<id>/historys/history.{json,csv}` | Sau **mỗi epoch** khi train | **Train (55k)** & **Val (5k)** | Theo dõi hội tụ (`loss`, `acc`, `lr`, `sps`, `vram`) & chọn `best.pt` |
| `runs/<id>/eval/` | **1 lần** ngay sau khi run train xong | **Val (5k)** | So sánh giữa các runs ablation & quyết định dừng sớm chuỗi |
| `runs/<id>/test_eval/` | **1 lần duy nhất** sau khi chốt model cuối | **Test (10k)** | Đo khả năng tổng quát thật cho đúng 2 model chung kết (`baseline_logistic` & `winning_mlp`) |

## Related Modules

- CLI Entrypoints & Config: [`labs/lab1_v2/README.md`](../../labs/lab1_v2/README.md)
- Reusable Core Engine: [`src/vibe_scripts/lab1_v2/README.md`](../../src/vibe_scripts/lab1_v2/README.md)
- Tài liệu kỹ thuật & Báo cáo: [`labs/lab1_v2/docs/README.md`](../../labs/lab1_v2/docs/README.md)
