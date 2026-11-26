---
title: Knowledge Extraction — Lab 1 v2 (FashionMNIST Classification Pipeline)
doc_type: Knowledge
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-26
updated: 2026-11-26
version: 1.0.0
status: completed
---

# Knowledge Extraction: `0005_lab1v2`

> **Source Log:** [`0005_lab1v2_log.md`](../logs/0005_lab1v2_log.md) · **Standard:** [`KNOWLEDGE_EXTRACTION.md`](../rules/dl-rule/KNOWLEDGE_EXTRACTION.md)

---

## 1. Matplotlib Headless Backend Guard on Windows (`"agg" in "tkagg"` Trap)

- **Category:** Debugging
- **Description:** Trên Windows headless/CLI, kiểm tra `if "agg" not in backend.lower()` sẽ bỏ qua chuyển backend vì chuỗi con `"agg"` có mặt bên trong `"tkagg"`, dẫn đến lỗi `TclError: Can't find a usable init.tcl` khi gọi `plt.subplots()`.
- **Evidence:** Lỗi xảy ra thực tế khi chạy `prepare_data.py` & `evaluator.py` trong môi trường venv Windows; được sửa dứt điểm bằng `if "inline" not in backend and backend != "agg": matplotlib.use("Agg", force=True)` và vượt qua `13/13` pytest.
- **Recommendation:** Mọi module vẽ biểu đồ trong `src/vibe_scripts/` cần dùng phép so sánh chính xác `backend != "agg"` (và giữ nguyên `inline` cho Jupyter Notebook) kèm `plt.close(fig)` sau khi lưu ảnh.

---

## 2. Two-Tier Model Selection Rule (`min val_loss` Intra-Run vs `(Macro-F1, Acc, -Loss)` Inter-Run)

- **Category:** Model Engineering
- **Description:** Cần tách bạch tiêu chí chọn checkpoint **trong cùng 1 run** và tiêu chí chọn mô hình **giữa các runs**:
  - Trong 1 run (`best.pt`): chọn theo `min(val_loss)` vì hàm loss liên tục, phát hiện sớm *loss-accuracy divergence* (overconfidence/overfitting) trước khi accuracy tụt (minh chứng ở `mlp_ep30`: epoch 15 → 30 `val_acc` tăng nhưng `val_loss` vọt từ `0.3043` lên `0.3477`).
  - Giữa các runs (`winning_mlp_run_id`): xếp hạng theo `(val_macro_f1, val_acc, -val_loss)` vì loss thấp hơn chỉ đảm bảo xác suất hiệu chuẩn mượt hơn chứ chưa chắc đoán đúng nhiều mẫu hơn (minh chứng: `mlp_opt_sgd_ep40` có `val_loss = 0.2862 < 0.2955` nhưng đoán sai nhiều hơn `mlp_default` `20` mẫu trên Val).
- **Evidence:** Xác thực qua chuỗi `12+1` runs ablation và `mlp_opt_sgd_ep40` trong [`evaluation_report.md`](../../labs/lab1_v2/docs/evaluation_report.md).
- **Recommendation:** Luôn khai báo rõ quy tắc chọn mô hình 2 tầng ngay từ đầu trong Layer 8.1 của `docs/pipelines/modeling.md` và mã hóa nhất quán trong `trainer.py`.

---

## 3. Anti-Leakage Artifact Separation (`historys/` vs `eval/` vs `test_eval/`)

- **Category:** Data Engineering
- **Description:** Phân tách thư mục artifact theo đúng giai đoạn & tập dữ liệu giúp chặn triệt để *test-set leakage* và tránh sinh biểu đồ trùng lặp:
  - `runs/<id>/historys/history.{json,csv}`: log theo từng epoch trên Train & Val.
  - `runs/<id>/eval/`: đánh giá 1 lần sau khi train trên **Val (5k)** (gồm learning curves + CM + metrics) để so sánh ablation & quyết định dừng sớm.
  - `runs/<finalist>/test_eval/`: chỉ tồn tại ở 2 model chung kết, đánh giá duy nhất 1 lần trên **Test (10k)** và **không** vẽ lại learning curves.
- **Evidence:** Kiểm chứng bằng test hồi quy trong [`test_lab1_v2_pipeline.py`](../../test/test_lab1_v2_pipeline.py) và chạy sạch `12+1` runs.
- **Recommendation:** Tái sử dụng cấu trúc `historys/` + `eval/` + `test_eval/` làm chuẩn cho `lab2_v2` và các dự án phân loại sau.

---

## 4. 4-Step Quantitative Error Analysis & Simpson's Paradox in Slice Analysis

- **Category:** Best Practice
- **Description:** Thay vì chỉ xem Top-16 ảnh lỗi cảm tính, quy trình 4 bước định lượng (`Row-Normalized CM %` → `Top-K Pair High/Low Conf Galleries` → `Taxonomy Labeling trên N ≥ 50 mẫu` → `Slice Analysis`) giúp định lượng rõ nút thắt cần bỏ công (`48.3%` Model Limitation vs `38.3%` Ambiguity vs `13.3%` Label Noise / Class Overlap). Đặc biệt trong Slice Analysis, cần tách lát riêng cho **nhóm lớp khó** bên cạnh toàn tập để tránh **Simpson's Paradox** (trên toàn tập `Q1 Small` chỉ sai `3.8%` do chứa giày/quần dễ, nhưng riêng cụm 4 lớp áo lại sai vọt lên `38.0%`).
- **Evidence:** Kết quả từ `evaluate.py --error-analysis` trên 10,000 mẫu Test ([`slice_analysis.json`](../../experiments/lab1_v2/evaluation/slice_analysis.json), [`error_taxonomy_60.json`](../../experiments/lab1_v2/evaluation/error_taxonomy_60.json)).
- **Recommendation:** Tái sử dụng mẫu `run_deep_error_analysis()` và `plot_error_taxonomy_breakdown()` trong `evaluator.py` cho mọi bài toán phân loại ảnh; khi nhúng bảng ảnh audit trong Markdown Preview của VS Code, đặt ảnh trong thẻ `<details>` với cú pháp `![](...)` thay vì link `[text](*.png)` để tránh bị sandbox webview chặn click.

---

## 5. Strict SRP Documentation Split Mapped 1-to-1 to Pipeline Layers

- **Category:** Project Convention
- **Description:** Tách bộ báo cáo thành 3 tài liệu đơn nhiệm ánh xạ 1-1 theo `MODEL_PIPELINE.md` giúp tránh phình file và lẫn lộn trách nhiệm:
  - `eda_report.md` → **Layer 6 (6.1–6.3 Data Engineering)**
  - `experiment_report.md` → **Layer 4 & Layer 7 (Sanity Checks, Experiment Matrix, Training Diagnostics)**
  - `evaluation_report.md` → **Layer 8, 9 & 10 (Model Selection trên Val, Final Test 10k, 4-Step Error Analysis & Inference)**
- **Evidence:** Đã áp dụng thành công cho `lab1_v2` với `100%` liên kết chéo đồng bộ giữa 3 báo cáo, `modeling.md`, `docs/README.md` và `experiment.ipynb`.
- **Recommendation:** Khởi tạo ngay khung 3 file báo cáo riêng từ Step 1 ở các bài lab mới (`0006_lab2_v2` trở đi) thay vì gộp chung rồi tách sau.

---

## 6. Safe Checkpoint Resume Must Persist `optimizer_state_dict` & Write to Separate `output_run_id`

- **Category:** Model Engineering
- **Description:** Khi resume huấn luyện từ checkpoint cũ để kiểm chứng giả thuyết (như chạy tiếp `mlp_opt_sgd` từ epoch `20 → 40`), cần: (1) lưu kèm `optimizer_state_dict` trong `last.pt` để không làm mất momentum/Adam moments, (2) xuất sang thư mục `--output-run-id` riêng để bảo toàn run gốc trong bảng ablation, và (3) nối liên tục `history.json` từ epoch `1 → N` để vẽ đường học xuyên suốt.
- **Evidence:** Đã triển khai trong [`trainer.py`](../../src/vibe_scripts/lab1_v2/trainer.py), kiểm chứng bằng unit test `test_resume_training_continues_epochs_and_preserves_history` và chạy thực tế `mlp_opt_sgd_ep40`.
- **Recommendation:** Mọi training loop mới phải luôn lưu `optimizer_state_dict` ngay từ đầu và hỗ trợ cặp cờ `--resume` + `--output-run-id`.

---

## Suggested Documentation Updates (Chờ Kỹ sư duyệt — không tự ý sửa)

1. **`.gitignore`:** Xem xét thu hẹp dòng `runs/` thành `/runs/` (chỉ chặn TensorBoard ở root) thay vì chặn mọi `experiments/*/runs/`, vì `*.pt` đã được chặn riêng ở dòng 228 — giúp không cần `git add -f` khi commit log/PNG của từng run.
2. **`.agents/rules/dl-rule/MODEL_PIPELINE.md`:** Có thể bổ sung gợi ý khung báo cáo 3 file SRP (`eda_report.md`, `experiment_report.md`, `evaluation_report.md`) và quy tắc 4 bước Error Analysis định lượng vào Layer 8–9 cho các lab sau (`0006_lab2_v2`).
