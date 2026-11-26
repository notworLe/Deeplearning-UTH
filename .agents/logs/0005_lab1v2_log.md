---
title: Execution Log — Lab 1 v2 (FashionMNIST Classification Pipeline)
doc_type: Execution Log
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-26
updated: 2026-11-26
version: 1.0.0
status: completed
---

# Execution Log: `0005_lab1v2` — FashionMNIST Classification Mini-Project

> **Standard:** [`LOGGING.md`](../rules/dl-rule/LOGGING.md) · **Test Report:** [`0005_lab1v2_test_report.md`](../tests/0005_lab1v2_test_report.md)

---

## Task

- **Task ID:** `0005_lab1v2` — [`task.md`](../tasks/0005_lab1v2/task.md)
- **Approved Plan:** [`0005_lab1v2_plan.md`](../plans/0005_lab1v2_plan.md) (`v1.7.0`, `status: completed`)

---

## Objective

Xây dựng mini-project Deep Learning hoàn chỉnh phân loại 10 lớp ảnh xám `28×28` **FashionMNIST** (`Logistic Regression` Baseline vs `MLPModel`), tuân thủ trọn bộ 10 tầng [`MODEL_PIPELINE.md`](../rules/dl-rule/MODEL_PIPELINE.md), cô lập hoàn toàn khỏi `labs/lab1/vibe/` và `src/` gốc, chống data/test leakage tuyệt đối và tách sạch vai trò CLI scripts vs executable report notebook.

---

## Summary

- **What was changed:**
  - Triển khai 9 module lõi tái sử dụng tại `src/vibe_scripts/lab1_v2/` (Collection/Cleaning/EDA, Stratified split & train-only normalization, `LogisticRegressionModel`, `MLPModel`, Trainer có `tqdm` monitoring + `--resume` + `optimizer_state_dict`, Evaluator có per-run Val eval + Final Test 10k + 4-step Error Analysis, và `FashionPredictor` inference).
  - Triển khai cấu hình kế thừa `VibeConfig` (`labs/lab1_v2/config.py`), 4 thin CLI scripts (`prepare_data.py`, `train.py`, `evaluate.py`, `inference.py`) và executable report notebook (`experiment.ipynb`).
  - Thực thi chuỗi 12 runs có kiểm soát (`--no-stop-on-target`, Kỹ sư chạy trên RTX 4060) + 1 thực nghiệm bổ sung resume `mlp_opt_sgd_ep40` (`20 → 40` epochs theo chỉ đạo trực tiếp của Kỹ sư) + Error Analysis 4 bước định lượng (CM chuẩn hóa %, Gallery Top-3 High/Low Conf, gán nhãn `N = 60` ảnh thật, Slice Analysis).
  - Chuẩn hóa bộ tài liệu theo nguyên tắc SRP 1-1 với các tầng pipeline: `modeling.md` (`v2.2.0`), `eda_report.md` (`v1.0.0`), `experiment_report.md` (`v2.1.0` — Layer 4 & 7) và `evaluation_report.md` (`v1.2.0` — Layer 8–10).
- **Why it was changed:** Mô phỏng quy trình kỹ thuật ML/DL chuẩn công nghiệp (config-first, evidence-based, chống leakage, tách rõ Experiment Diagnostics vs Evaluation & Error Analysis) để làm khuôn mẫu tái sử dụng cho các bài lab tiếp theo.
- **Final Outcome:**
  - Chọn `mlp_default` trên Validation theo quy tắc `(val_macro_f1, val_acc, -val_loss)`: `Val Macro-F1 = 89.62%`, `Val Acc = 89.74%`, `Val Loss = 0.2955` (vượt `mlp_opt_sgd_ep40` về cả Macro-F1 `89.34%` và Acc `89.34%`).
  - Trên tập Test độc lập (`10,000` mẫu, chấm 1 lần): **`mlp_default` đạt `88.79%` Accuracy / `88.68%` Macro-F1 / `0.3362` Loss** vs **`baseline_logistic` `84.35%` / `84.21%` / `0.4473`** — gap cải thiện **`+4.44 pp` Acc, `+4.47 pp` Macro-F1, `-0.1110` Loss**, dương trên `10/10` lớp.
  - Error Analysis định lượng (`N = 60` ảnh lỗi Top-3 cặp) cho thấy **`48.3%` Model Limitation**, **`38.3%` Ambiguity (`28×28`)**, **`13.3%` Label Noise / Class Overlap**; Slice Analysis chỉ ra lỗi tăng gấp đôi ở áo thân hẹp/nhỏ (`33.8%–38.0%` vs `18.1%–19.5%`).

---

## Modified Files

- **Reusable Core (`src/vibe_scripts/lab1_v2/`):** `__init__.py`, `README.md`, `transforms.py`, `data.py`, `logistic.py`, `mlp.py`, `trainer.py`, `evaluator.py`, `inference.py`
- **Config, CLIs & Notebook (`labs/lab1_v2/`):** `README.md`, `config.py`, `prepare_data.py`, `train.py`, `evaluate.py`, `inference.py`, `experiment.ipynb`
- **Documentation (`labs/lab1_v2/docs/`):** `README.md`, `pipelines/modeling.md`, `models/logistic_regression.md`, `models/mlp.md`, `eda_report.md`, `experiment_report.md`, `evaluation_report.md`
- **Unit Tests (`test/`):** `test_lab1_v2_data.py`, `test_lab1_v2_models.py`, `test_lab1_v2_pipeline.py`
- **Artifacts (`experiments/lab1_v2/`):** `README.md`, `eda/*`, `ablation/*`, `evaluation/*` (gồm `audit_sheets/*`, `error_taxonomy_60.{csv,json}`, `slice_analysis.json`), `figures/*`, `runs/<13 runs>/{config.json, historys/*, eval/*, test_eval/*}` *(không commit `*.pt`)*
- **Workflow Records (`.agents/`):** `.agents/tasks/0005_lab1v2/task.md`, `.agents/plans/0005_lab1v2_plan.md`, `.agents/tests/0005_lab1v2_test_report.md`, `.agents/logs/0005_lab1v2_log.md`, `.agents/knowledge_extraction/0005_lab1v2_knowledge.md`
- **Untouched Scope (Verified):** Không sửa `src/*.py` gốc, `labs/lab1/vibe/`, `experiments/lab1/`, `data/FashionMNIST/raw/`, `.gitignore`, `README.md` gốc.

---

## Deliverables

- **Commit nền (Steps 1–7):** `7fb2472` (`feat(lab1_v2): implement FashionMNIST pipeline, 12+1 runs, reports, and tests`, `172` files, `0` file `*.pt`).
- **Báo cáo kỹ thuật chính:** [`modeling.md`](../../labs/lab1_v2/docs/pipelines/modeling.md) · [`eda_report.md`](../../labs/lab1_v2/docs/eda_report.md) · [`experiment_report.md`](../../labs/lab1_v2/docs/experiment_report.md) · [`evaluation_report.md`](../../labs/lab1_v2/docs/evaluation_report.md) · [`experiment.ipynb`](../../labs/lab1_v2/experiment.ipynb).

---

## Test Result

- **Stage 1 — Smoke Test:** **PASS** (`4 passed, 9 deselected in 9.23s`).
- **Stage 2 — Normal Test:** **PASS** (`13 passed in 21.04s`).
- **Verification khác:** `0` broken relative links across docs; `nbformat.validate` PASS; Save/Load parity `0.0 < 1e-6` trên toàn bộ checkpoints.
- Chi tiết bằng chứng xem tại [`0005_lab1v2_test_report.md`](../tests/0005_lab1v2_test_report.md).

---

## Remaining Issues

1. **Giới hạn kiến trúc MLP phẳng:** Phép `Flatten(784)` đánh mất đặc trưng không gian 2D (hàng khuy dọc, khóa kéo, cổ bẻ, đường chần bông), chiếm `48.3%` lỗi Top-3 cặp và giữ MLP ở trần `~89.5%` Val Acc → bước cải tiến tiếp theo là nâng cấp lên **CNN** kèm augmentation nhẹ (dịch chuyển/scale nhỏ, chỉnh sáng nhẹ) cho nhóm áo hẹp & áo tối.
2. **Quy tắc `.gitignore` (`runs/`):** `.gitignore` gốc đang bỏ qua mọi thư mục `runs/` (nên cần `git add -f` chọn lọc file `.json/.csv/.png` trong `experiments/lab1_v2/runs/` mà vẫn loại trừ `*.pt`).
3. **Gán nhãn `error_taxonomy_60.csv`:** 60 nhãn nguyên nhân được gắn trực quan bằng mắt/vision-assisted trên 3 Audit Sheets (`#1–#60`), có sai số chủ quan tự nhiên `±5–10%` ở mẫu ranh giới — Kỹ sư có thể trực tiếp rà soát/hiệu chỉnh trong [`error_taxonomy_60.csv`](../../experiments/lab1_v2/evaluation/error_taxonomy_60.csv) và vẽ lại biểu đồ bằng `plot_error_taxonomy_breakdown()`.
