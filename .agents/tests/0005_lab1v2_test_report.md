---
title: Test Report — Lab 1 v2 (FashionMNIST: Logistic Regression vs MLP)
doc_type: Test Report
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-26
updated: 2026-11-26
version: 1.0.0
status: completed
---

# Testing Report: `0005_lab1v2` — FashionMNIST Classification Pipeline

> **Source Task:** [`0005_lab1v2/task.md`](../tasks/0005_lab1v2/task.md) · **Approved Plan:** [`0005_lab1v2_plan.md`](../plans/0005_lab1v2_plan.md) · **Standard:** [`TESTING.md`](../rules/dl-rule/TESTING.md)

---

## Summary

- **Overall Result:** **PASS** (100% Smoke Tests, Normal Tests, Experimental Acceptance Criteria, Save/Load Parity, Notebook Schema và Documentation Link Integrity đều đạt).
- **Environment:** Windows, Python `3.13.14`, PyTorch `2.15.0+cu132`, CUDA GPU `NVIDIA GeForce RTX 4060 Laptop GPU`, `pytest 9.1.1`.

---

## Executed Tests

### Stage 1 — Smoke Test (`pytest -m smoke`)

- **Purpose:** Xác minh import sạch, config `VibeConfig`, forward pass Logistic & MLP đúng tensor contract `(B, 1, 28, 28) → (B, 10)`, chạy 1 epoch trên subset nhỏ trong `tmp_path` tạo đủ `runs/<id>/eval/*`, và inference không lỗi.
- **Command:** `pytest -m smoke test/test_lab1_v2_data.py test/test_lab1_v2_models.py test/test_lab1_v2_pipeline.py -v`
- **Result:** **PASS (`4 passed, 9 deselected in 9.23s`)**

| Test Case | Module | Result |
| :--- | :--- | :---: |
| `test_config_and_synthetic_dataloader_smoke` | [`test_lab1_v2_data.py`](../../test/test_lab1_v2_data.py) | **PASS** |
| `test_models_forward_smoke[logistic-hidden_dims0]` | [`test_lab1_v2_models.py`](../../test/test_lab1_v2_models.py) | **PASS** |
| `test_models_forward_smoke[mlp-hidden_dims1]` | [`test_lab1_v2_models.py`](../../test/test_lab1_v2_models.py) | **PASS** |
| `test_single_run_per_run_eval_and_inference_smoke` | [`test_lab1_v2_pipeline.py`](../../test/test_lab1_v2_pipeline.py) | **PASS** |

### Stage 2 — Normal Test (Unit, Integration, Functional & Experimental Acceptance)

#### 2.1 Full Pytest Suite (`13/13` PASS)
- **Command:** `pytest test/test_lab1_v2_data.py test/test_lab1_v2_models.py test/test_lab1_v2_pipeline.py -v`
- **Result:** **PASS (`13 passed in 21.04s`)**
- **Verified Invariants:**
  - Stratified split `55,000 / 5,000 / 10,000` không giao nhau (`seed = 248`), chuẩn hóa train-only (`mean = 0.2863`, `std = 0.3532`), Data Collection/Cleaning/EDA pipeline sạch (`8/8` SHA-256 pass, `0` corrupt/duplicate).
  - Exact parameter counts (`7,850` Logistic; `235,146` MLP `[256, 128]`), raw logits output, Fail-Fast `ValueError` khi sai input shape.
  - Sanity checks: initial loss `≈ ln(10) ≈ 2.3026`, overfit 1 batch 32 mẫu `≥ 95%` (thực tế `100%`).
  - Controlled suite early-stop, Save/Load parity (`max_abs_diff < 1e-6`), không vẽ lặp learning curves vào `test_eval/`, `--resume` nối liên tục lịch sử epochs và khôi phục `optimizer_state_dict`.
  - Schema `labs/lab1_v2/experiment.ipynb` hợp lệ (`nbformat.validate`).

#### 2.2 Experimental Acceptance Criteria & End-to-End Pipeline Validation

| Hạng mục kiểm chứng | Tiêu chí đạt | Kết quả thực tế | Result |
| :--- | :--- | :--- | :---: |
| **6.1–6.3 Data Audit (`prepare_data.py`)** | `8/8` SHA-256 pass, `0` corrupt/duplicate, 5 biểu đồ EDA | `70,000` mẫu sạch, `0` mismatches/corrupt/duplicates, 5/5 PNGs + [`eda_report.md`](../../labs/lab1_v2/docs/eda_report.md) | **PASS** |
| **Controlled Sweep (`train.py --run-all --no-stop-on-target`)** | Hoàn tất `12/12` runs + `1` resume `mlp_opt_sgd_ep40`, per-run Val eval | `12/12` runs + `mlp_opt_sgd_ep40` (`1 → 40 ep`) đều hoàn tất, Save/Load parity `0.0` toàn bộ | **PASS** |
| **Final Held-Out Test 10k (`evaluate.py`)** | Best MLP tạo gap cải thiện dương rõ rệt so với Baseline | `mlp_default` (`88.79%` Acc / `88.68%` Macro-F1 / `0.3362` Loss) vs `baseline_logistic` (`84.35%` / `84.21%` / `0.4473`) → **`+4.44 pp` Acc, `+4.47 pp` Macro-F1, `-0.1110` Loss**, cải thiện `10/10` lớp | **PASS** |
| **4-Step Deep Error Analysis (`evaluate.py --error-analysis`)** | CM chuẩn hóa %, Gallery Top-3 High/Low Conf, Audit 60 mẫu, Slice Analysis | Sinh đủ 5 biểu đồ mới, 3 audit sheets (`60` mẫu), [`error_taxonomy_60.csv`](../../experiments/lab1_v2/evaluation/error_taxonomy_60.csv), [`slice_analysis.json`](../../experiments/lab1_v2/evaluation/slice_analysis.json) | **PASS** |
| **Offline Inference & Latency (`inference.py`)** | Top-3 Softmax, `requires_grad=False`, Save/Load `< 1e-6`, đo latency | `Test[0]` đoán đúng `Ankle boot` (`99.96%`), diff `0.0`, `0.396 ms/ảnh` (`BS=1`), `0.012 ms/ảnh` (`BS=32`), ckpt `922.2 KB` | **PASS** |
| **Executable Report Notebook (`experiment.ipynb`)** | Chạy sạch 100% cells ở chế độ đọc artifact + kiểm chứng nhẹ | `7/7` code cells + `2` markdown cells (`9` cells) thực thi sạch & pass `nbformat.validate` | **PASS** |
| **Documentation Links Integrity** | `0` broken relative links trên toàn bộ tài liệu `lab1_v2` | `100%` liên kết hợp lệ (`0` broken) | **PASS** |

---

## Failed Tests

- **0 failed tests** tại thời điểm chốt nghiệm thu.
- *(Các lỗi trung gian phát sinh trong quá trình phát triển — lỗi backend Matplotlib `"agg" in "tkagg"` trên Windows và lỗi vẽ lặp curve vào `test_eval/` — đã được sửa tận gốc và bổ sung test hồi quy trong [`test_lab1_v2_pipeline.py`](../../test/test_lab1_v2_pipeline.py).)*

---

## Evidence

- Test suites: [`test_lab1_v2_data.py`](../../test/test_lab1_v2_data.py) · [`test_lab1_v2_models.py`](../../test/test_lab1_v2_models.py) · [`test_lab1_v2_pipeline.py`](../../test/test_lab1_v2_pipeline.py)
- Reports: [`eda_report.md`](../../labs/lab1_v2/docs/eda_report.md) · [`experiment_report.md`](../../labs/lab1_v2/docs/experiment_report.md) · [`evaluation_report.md`](../../labs/lab1_v2/docs/evaluation_report.md) · [`experiment.ipynb`](../../labs/lab1_v2/experiment.ipynb)
- Verified JSON artifacts: [`ablation_summary.json`](../../experiments/lab1_v2/ablation/ablation_summary.json) · [`test_comparison.json`](../../experiments/lab1_v2/evaluation/test_comparison.json) · [`error_taxonomy_60.json`](../../experiments/lab1_v2/evaluation/error_taxonomy_60.json) · [`slice_analysis.json`](../../experiments/lab1_v2/evaluation/slice_analysis.json)

---

## Recommendation

**Ready for Logging** (toàn bộ Smoke Test, Normal Test và Acceptance Criteria đã đạt, không còn lỗi chặn).
