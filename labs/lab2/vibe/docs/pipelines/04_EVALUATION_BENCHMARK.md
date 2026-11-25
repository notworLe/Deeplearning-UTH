---
title: "Pipe 04: Evaluation & Error Analysis — Lab 2 (Oxford-IIIT Pets)"
doc_type: pipeline-analysis
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-21
updated: 2026-11-21
last_updated_by: Antigravity
version: 0.1.0
status: active
related_docs:
  - ../REPORT.md
  - ./README.md
  - ../../config.py
tags: [lab2, pipe-04, evaluation, error-analysis, cat-vs-dog]
---

# Pipe 04: Evaluation & Error Analysis

Triển khai tại [`src/vibe_scripts/lab2/evaluate.py`](../../../../../src/vibe_scripts/lab2/evaluate.py).

## 1. Hợp Đồng I/O
- **Input**: Các thư mục run `experiments/lab2/<exp_id>/{best.pt, history.json}` và `test_loader` (`3.302` ảnh, `shuffle=False`).
- **Output (`experiments/lab2/evaluation/` & `predictions/`)**:
  - Số liệu: `benchmark_summary.json`, `benchmark_table.csv`, `per_class_metrics.json`, `species_analysis.json` (Cat 12 giống vs Dog 25 giống).
  - Biểu đồ: `learning_curves_comparison.png`, `learning_curve_<exp_id>.png`, `confusion_matrix_<exp_id>.png` (37×37), `cat_vs_dog_error_comparison.png`, `predictions_gallery_<exp_id>.png` (khử chuẩn hóa ImageNet).

## 2. Cơ Chế Kỹ Thuật Chính
1. **Single-shot Test**: chỉ đánh giá tập `Test` (`3.302` ảnh) đúng 1 lần trên `best.pt` dưới `model.eval()` + `torch.no_grad()`.
2. **Kiểm chứng giả thuyết Cat vs Dog**: tách riêng Test Error Rate, Accuracy và Macro-F1 cho 12 giống mèo so với 25 giống chó, kèm kiểm định tỷ lệ sai (two-proportion z-test) để báo cáo trung thực.
3. **Save/Load Integrity**: đối chiếu `max |logits_before - logits_after| < 1e-6`.
4. **Guard Matplotlib Backend**: bảo vệ `%matplotlib inline` khi import trong Jupyter Notebook.

## 3. Tiêu Chí Kiểm Chứng (`test/test_lab2_evaluate.py`)
- **Smoke**: chạy `evaluate_checkpoint()` với dummy checkpoint trên subset nhỏ không crash.
- **Normal**: bất biến Confusion Matrix `sum(CM) == N`, `acc == trace(CM)/N`, kiểm tra tách đúng 12 mèo / 25 chó, Save/Load `< 1e-6`, sinh đủ JSON/CSV/PNG.
