---
title: "Pipe 04: Evaluation & Benchmark — Phân Tích Đặc Trưng (Lab 1)"
doc_type: pipeline-analysis
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-18
updated: 2026-11-20
last_updated_by: Antigravity
version: 0.4.0
status: approved
reviewers: []
related_docs:
  - ../REPORT.md
  - ./README.md
  - ../../config.py
  - ../../../../../.agents/rules/AI_FIRST_LIFECYCLE.md
  - ../../../../../docs/references/DL_PIPELINE.md
  - ../../../../../src/vibe_scripts/lab1/evaluate.py
tags: [lab1, pipe-04, evaluation, benchmark, error-analysis]
---

# Pipe 04: Evaluation & Benchmark (Đánh Giá Định Lượng & Phân Tích Lỗi)

Ánh xạ **Giai đoạn 6 (Evaluation & Bench)** trong [`AI_FIRST_LIFECYCLE.md`](../../../../../.agents/rules/AI_FIRST_LIFECYCLE.md) và mục `Evaluate` trong [`DL_PIPELINE.md`](../../../../../docs/references/DL_PIPELINE.md), triển khai tại [`src/vibe_scripts/lab1/evaluate.py`](../../../../../src/vibe_scripts/lab1/evaluate.py).

---

## 1. Hợp Đồng Giao Diện (Input / Output)

- **Input**: Các thư mục kết quả `experiments/lab1/<exp_id>/{best.pt, history.json}` (ma trận 3×2) và `test_loader` (`10.000` mẫu, `shuffle=False`).
- **Output (`experiments/lab1/evaluation/` & `predictions/`)**:
  - Bảng định lượng: `benchmark_summary.json`, `benchmark_table.csv`, `per_class_metrics.json`.
  - Đồ thị chẩn đoán: `learning_curves_comparison.png`, 6× `learning_curve_<exp_id>.png` (2 panel Loss/Acc đánh dấu Best Epoch & Overfit Gap), `per_class_f1_comparison.png`, `confusion_matrix_<exp_id>.png`, `predictions_gallery_<exp_id>.png`.

---

## 2. Phân Tích Đặc Trưng & Cơ Chế Kỹ Thuật

1. **Nguyên tắc đánh giá một lần (Single-Shot Test Invariant)**:
   - Toàn bộ quá trình tinh chỉnh mô hình dựa trên Validation; tập Test `10.000` mẫu chỉ được đánh giá **một lần duy nhất** trên `best.pt` dưới `model.eval()` và `torch.no_grad()`.
2. **Đánh giá đa chiều (Hiệu năng + Tài nguyên + Độ tin cậy)**:
   - Không chỉ nhìn tổng `Accuracy`, Pipe 04 tách riêng `Precision / Recall / F1` cho từng lớp trong 10 lớp, giúp kiểm chứng giả thuyết hình học từ EDA (lớp `Shirt`, `T-shirt/top`, `Pullover`, `Coat` là điểm nghẽn chính).
   - Đo kèm số tham số, `best_epoch`, dung lượng file `best.pt` (KB) và thời gian suy luận trung bình mỗi batch (ms).
3. **Kiểm chứng toàn vẹn Save/Load**:
   - Đọc lại `best.pt` vào hai đối tượng mô hình khởi tạo mới và đối chiếu tối đa chênh lệch logits `< 1e-6`, bảo đảm trọng số lưu ra đĩa tái tạo chính xác 100% kết quả đánh giá.
4. **Phân tích lỗi trực quan (Qualitative Error Gallery)**:
   - Lọc các mẫu dự đoán sai có độ tự tin Softmax cao nhất (`Top Confident Errors`) để tìm ra điểm mù đặc trưng của từng kiến trúc.

---

## 3. Mục Tiêu

- Tạo bộ số liệu định lượng 3×2 và bằng chứng trực quan đầy đủ để điền vào [`REPORT.md`](../REPORT.md) và kiểm chứng 3 giả thuyết `H1–H3`.

---

## 4. Hạn Chế & Rủi Ro Đặc Thù

1. **Hiển thị sai màu ảnh sau Normalize**: Vẽ trực tiếp tensor Z-Score (có giá trị âm) khiến Matplotlib cắt ngầm dải màu làm ảnh bị bệt tương phản.
2. **Đo sai độ trễ GPU**: Chạy GPU bất đồng bộ nếu bấm giờ mà thiếu warm-up và `torch.cuda.synchronize()` sẽ cho số đo thời gian thấp giả tạo.

---

## 5. Giải Pháp Kỹ Thuật

- Khử chuẩn hóa tường minh (`img * std + mean`, clip `[0.0, 1.0]`) trước khi vẽ `predictions_gallery_*.png`.
- Thực hiện `warmup_batches` và gọi `torch.cuda.synchronize()` trước/sau vòng đo thời gian.
- Luôn dùng `sharey=False` và `bbox_inches="tight"` để biểu đồ không lặp lại lỗi khoảng trắng dư (`Error 0001`).
- Bảo vệ backend Matplotlib (`if "inline" not in matplotlib.get_backend().lower(): matplotlib.use("Agg")`) để tránh ghi đè `%matplotlib inline` khi import `evaluate.py` từ Jupyter Notebook.

---

## 6. Tiêu Chí Kiểm Chứng (`test/test_lab1_evaluate.py`)

- **Smoke Test**: Chạy `evaluate_checkpoint()` với dummy checkpoint trên 40 mẫu không crash.
- **Normal Test**: Tổng phần tử Confusion Matrix bằng đúng tổng mẫu Test, `Accuracy == trace(CM) / sum(CM)`, P/R/F1 thuộc `[0, 1]`, `verify_save_load_integrity() < 1e-6`, `run_benchmark()` sinh đủ JSON, CSV và PNG.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.4.0 | 2026-11-20 | notworLe | Antigravity | Bổ sung 6 đồ thị `learning_curve_<exp_id>.png` (`plot_individual_learning_curves`) và cơ chế bảo vệ Matplotlib inline backend |
| 0.3.0 | 2026-11-19 | notworLe | Antigravity | Viết lại theo cấu trúc phân tích đặc trưng `AI_FIRST_LIFECYCLE.md`, loại bỏ khung prompt mẫu |
| 0.2.0 | 2026-11-18 | notworLe | Antigravity | Đồng bộ `labs/lab1/vibe/config.py` |
