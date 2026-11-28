---
title: Test Report — Lab 3 v2 (Get Started with Hugging Face: Sentiment Pipeline & DistilBERT IMDB)
doc_type: Test Report
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-28
updated: 2026-11-28
version: 1.0.0
status: completed
---

# Testing Report: `0007_lab3_v2` — Hugging Face Sentiment & DistilBERT IMDB Fine-Tuning

> **Source Task:** [`0007_lab3_v2/task.md`](../tasks/0007_lab3_v2/task.md) · **Approved Plan:** [`0007_lab3_v2/plan.md`](../tasks/0007_lab3_v2/plan.md) · **Standard:** [`TESTING.md`](../rules/dl-rule/TESTING.md)

---

## Summary

- **Overall Result:** **PASS** (100% Smoke Tests, Normal Tests, Experimental Acceptance Criteria trên Ex1 & Ex2, Held-Out Test 1 lần, Live Inference, 2 Executable Notebooks và Documentation Links Integrity đều đạt với `0` warnings).
- **Environment:** Windows, Python `3.13.14`, PyTorch `2.15.0+cu132`, `transformers 5.23.0`, `datasets 5.2.0`, `evaluate 0.4.6`, `accelerate 1.16.0`, `tensorboard 2.21.0`, CUDA GPU `NVIDIA GeForce RTX 4060 Laptop GPU`, `pytest 9.1.1`.

---

## Executed Tests

### Stage 1 — Smoke Test (`pytest -m smoke`)

- **Purpose:** Xác minh cấu hình `VibeConfig`, giải phẫu WordPiece tokenizer + batch padding + `pipeline("text-classification")` offline, và 1-epoch mini HF `Trainer` (4 batches, tiny `DistilBertConfig`) xuất đủ `monitor/`, `val_eval/`, `weights/best_model/` mà không tạo `test_eval/`.
- **Command:** `pytest -m smoke test/test_lab3_v2_ex1.py test/test_lab3_v2_ex2.py -v`
- **Result:** **PASS (`2 passed, 4 deselected in 12.37s`)**

| Test Case | Module | Result |
| :--- | :--- | :---: |
| `test_config_and_ex1_tokenizer_and_pipeline_smoke` | [`test_lab3_v2_ex1.py`](../../test/test_lab3_v2_ex1.py) | **PASS** |
| `test_ex2_tiny_trainer_smoke` | [`test_lab3_v2_ex2.py`](../../test/test_lab3_v2_ex2.py) | **PASS** |

### Stage 2 — Normal Test (Unit, Integration, Functional & Experimental Acceptance)

#### 2.1 Full Pytest Suite (`6/6` PASS, `0` warnings — Post-Execution Re-run)
- **Command:** `pytest test/test_lab3_v2_ex1.py test/test_lab3_v2_ex2.py -v`
- **Result:** **PASS (`6 passed in 22.44s`, 0 warnings)**
- **Verified Invariants:**
  - Stratified split 50/50 không giao nhau (`Train ∩ Val = ∅`) & tái lập tuyệt đối từ `split_indices.json`.
  - Data Audit & EDA bắt đúng `<br />` HTML tags, độ dài vượt `max_length` và xuất 3 JSON + 2 PNG sạch.
  - Fail-Fast Tensor Contract `(B, L)` `torch.int64` (`validate_classifier_batch` ném `ValueError` khi sai shape/dtype/max_length).
  - End-to-end Train → Final Test 1 lần → Error Analysis (`truncation_slice`) → Live Inference (`requires_grad = False`).
  - Schema `nbformat v4` hợp lệ cho cả `ex1_sentiment.ipynb` và `ex2_finetune.ipynb` sau khi thực thi.

#### 2.2 Experimental Acceptance Criteria & End-to-End Pipeline Validation

| Hạng mục kiểm chứng | Tiêu chí đạt (`task.md`) | Kết quả thực tế | Result |
| :--- | :--- | :--- | :---: |
| **Ex1 Tokenizer & Pipeline (`run_ex1.py`)** | Giải thích rõ `input_ids`, `attention_mask`, `decode` & chạy các câu dễ/khó/mỉa mai | `14` tokens + batch padding `(2, 11)` + `8` câu mẫu (`5/5` câu rõ & phủ định OK, `2/2` mỉa mai & `1` câu pha trộn phân tích rõ nguyên nhân) tại [`ex1_results.json`](../../experiments/lab3_v2/ex1/ex1_results.json) | **PASS** |
| **Ex2 Data Audit & EDA (`stanfordnlp/imdb`)** | Lấy tập con `4,000 / 1,000 / 1,000`, tách Val cố định `seed = 248` | `0` review rỗng, `0` trùng `Train ∩ Test`, đo `45.07%` Train / `40.80%` Test vượt `256` tokens, xuất 2 biểu đồ tại `experiments/lab3_v2/eda/` | **PASS** |
| **Ex2 Fine-Tuning (`train_ex2.py`)** | Chạy hết `3 epochs` không lỗi, loss giảm, **Val Acc sau > trước fine-tune** | Trước: `49.70%` Acc (`Loss 0.6957`) → Sau: **`87.80%` Acc / `88.11%` F1 (`Loss 0.2764`, `+38.10 pp`)** trong `91.41s`, TensorBoard active | **PASS** |
| **Ex2 Held-Out Test 1 lần (`eval_ex2.py`)** | Test chỉ chấm 1 lần cuối sau khi Val đạt; phân tích ca dự đoán sai | **`89.10%` Test Acc / `89.26%` Test F1 / `0.2518` Test Loss**; Truncation slice: `14.46%` lỗi (`>256`) vs `8.45%` (`≤256`) tại [`error_examples.json`](../../experiments/lab3_v2/runs/distilbert_imdb/test_eval/error_examples.json) | **PASS** |
| **Live Inference (`eval_ex2.py --predict`)** | Suy luận sạch trên câu review mới dưới `@torch.inference_mode()` | `3/3` câu chính xác (`96.19%` Pos, `96.73%` Neg, `87.51%` Neg câu tương phản), `requires_grad = False` | **PASS** |
| **2 Executable Notebooks & Docs Integrity** | Chạy sạch `ex1_sentiment.ipynb` & `ex2_finetune.ipynb`, `0` broken links, SRP sạch | `4/4` & `5/5` code cells OK (`nbformat.validate` PASS), toàn bộ liên kết Markdown hợp lệ (`0` broken) | **PASS** |

---

## Failed Tests

- **0 failed tests** tại thời điểm chốt nghiệm thu.
- *(3 lỗi trung gian phát hiện sớm trong quá trình chạy thực tế — xung đột key `MAX_LENGTH` trong `VibeConfig`, đổi API offline `BertTokenizer(vocab=...)` ở `transformers 5.23.0`, yêu cầu namespace `"stanfordnlp/imdb"` ở `datasets 5.2.0`, và lọc điểm eval hậu-train trong `plot_loss_curve` — đã được sửa triệt để và xác minh lại.)*

---

## Evidence

- Test suites: [`test_lab3_v2_ex1.py`](../../test/test_lab3_v2_ex1.py) · [`test_lab3_v2_ex2.py`](../../test/test_lab3_v2_ex2.py)
- Reports & Notebooks: [`REPORT.md`](../../labs/lab3_v2/REPORT.md) · [`eda_report.md`](../../labs/lab3_v2/docs/eda_report.md) · [`experiment_report.md`](../../labs/lab3_v2/docs/experiment_report.md) · [`evaluation_report.md`](../../labs/lab3_v2/docs/evaluation_report.md) · [`ex1_sentiment.ipynb`](../../labs/lab3_v2/ex1_sentiment.ipynb) · [`ex2_finetune.ipynb`](../../labs/lab3_v2/ex2_finetune.ipynb)
- Verified JSON artifacts: [`ex1_results.json`](../../experiments/lab3_v2/ex1/ex1_results.json) · [`eda_summary.json`](../../experiments/lab3_v2/eda/eda_summary.json) · [`train_summary.json`](../../experiments/lab3_v2/runs/distilbert_imdb/monitor/train_summary.json) · [`final_test_metrics.json`](../../experiments/lab3_v2/runs/distilbert_imdb/test_eval/final_test_metrics.json) · [`error_examples.json`](../../experiments/lab3_v2/runs/distilbert_imdb/test_eval/error_examples.json)

---

## Recommendation

**Ready for Logging** (toàn bộ Smoke Test, Normal Test, Experimental Acceptance Criteria và Documentation Audits đã đạt 100%).
