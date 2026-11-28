---
title: Execution Log — Lab 3 v2 (Hugging Face Sentiment Pipeline & DistilBERT IMDB Fine-Tuning)
doc_type: Execution Log
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-28
updated: 2026-11-28
version: 1.0.0
status: completed
---

# Execution Log: `0007_lab3_v2` — Get Started with Hugging Face (`Ex1` & `Ex2`)

> **Standard:** [`LOGGING.md`](../rules/dl-rule/LOGGING.md) · **Test Report:** [`0007_lab3_v2_test_report.md`](../tests/0007_lab3_v2_test_report.md)

---

## Task

- **Task ID:** `0007_lab3_v2` — [`task.md`](../tasks/0007_lab3_v2/task.md)
- **Approved Plan:** [`plan.md`](../tasks/0007_lab3_v2/plan.md) (`v1.3.0`, `status: completed`)

---

## Objective

Làm chủ trọn vẹn luồng hệ sinh thái **Hugging Face (`transformers`, `datasets`, `evaluate`)** qua 2 bài thực hành: **Ex1** giải phẫu WordPiece tokenizer & chạy `pipeline` sentiment trên `distilbert-base-uncased-finetuned-sst-2-english`, và **Ex2** fine-tune `distilbert-base-uncased` trên tập con cân bằng **IMDB (`4,000 / 1,000 / 1,000`, `seed = 248`)**, đo định lượng Trước vs Sau fine-tune trên Validation, chấm Held-Out Test đúng 1 lần cuối và phân tích nguyên nhân lỗi theo chuẩn `MODEL_PIPELINE.md v1.4.0`.

---

## Summary

- **What was changed:**
  - Xây dựng trọn bộ thiết kế & tài liệu SRP tại `labs/lab3_v2/docs/` (`README.md`, `pipelines/modeling.md`, `models/distilbert.md`, `eda_report.md`, `experiment_report.md`, `evaluation_report.md`) cùng báo cáo điều hướng `labs/lab3_v2/REPORT.md`.
  - Triển khai 4 module lõi tái sử dụng tại `src/vibe_scripts/lab3_v2/` (`ex1_sentiment.py`, `data.py`, `ex2_finetune.py`, `analysis.py`), cấu hình `labs/lab3_v2/config.py`, 3 thin CLIs (`run_ex1.py`, `train_ex2.py`, `eval_ex2.py`), 2 executable notebooks (`ex1_sentiment.ipynb`, `ex2_finetune.ipynb`) và bộ test offline 2 tầng (`test/test_lab3_v2_{ex1,ex2}.py`).
  - Xử lý trọn vẹn các thay đổi API của **`transformers 5.23.0`** (`text-classification`, `eval_strategy`, `processing_class`, bỏ `logging_dir`) và **`datasets 5.2.0`** (dùng namespace chính thức `"stanfordnlp/imdb"`).
  - Sau khi hoàn tất Steps 1–3 và bàn giao Step 4, theo ủy quyền trực tiếp của Kỹ sư (*"phần còn lại fine tune hay training gì của lab3 bạn thực hiện đi"*), Agent thực thi trọn vẹn `run_ex1.py`, `train_ex2.py` (`91.41s` trên RTX 4060 `fp16`), `eval_ex2.py` (1 lần duy nhất) + `--predict`, hoàn thiện 100% số liệu thật trong 3 báo cáo SRP + `REPORT.md` và thực thi sạch cả 2 notebooks.
- **Why it was changed:** Cung cấp bài học trực quan, có thể tái lập 100%, chống Test leakage và tương thích phiên bản Hugging Face mới nhất (`v5`), đồng thời cô lập hoàn toàn khỏi `labs/lab3/` cũ.
- **Final Outcome:**
  - **Ex1 (`distilbert-sst2`):** Minh họa rõ `14` tokens (`input_ids`, `attention_mask`, `decode`), batch padding `(2, 11)` và `5/8` khớp kỳ vọng trên 8 câu mẫu (`5/5` câu rõ & phủ định OK `99.92%–99.99%`; `2/2` câu mỉa mai & `1` câu tương phản lộ giới hạn thực tế).
  - **Ex2 Validation (`1,000` mẫu):** **Trước fine-tune `49.70%` Acc (`Val F1 0.00%`, `Val Loss 0.6957`) → Sau fine-tune `87.80%` Acc (`Val F1 88.11%`, `Val Loss 0.2764`, tăng `+38.10 pp`)** với `checkpoint-250` (Epoch 1) tự chọn bởi `min eval_loss`.
  - **Ex2 Held-Out Test (`1,000` mẫu, chấm 1 lần):** **`89.10%` Test Acc / `89.26%` Binary F1 / `89.10%` Macro-F1 / `0.2518` Test Loss** (`Negative F1 = 88.93%`, `Positive F1 = 89.26%`).
  - **Error Analysis (`109` lỗi Test):** Chứng minh định lượng review bị cắt `> 256` tokens có tỷ lệ lỗi **`14.46%` (`59/408`) — gấp `1.71×`** review `≤ 256` tokens (`8.45%`, `50/592`); `6/10` lỗi tự tin cao nhất bị mất đoạn kết luận cuối review do cắt cụt.

---

## Modified Files

- **Reusable Core (`src/vibe_scripts/lab3_v2/`):** `__init__.py`, `README.md`, `ex1_sentiment.py`, `data.py`, `ex2_finetune.py`, `analysis.py`
- **Config, CLIs, Notebooks & Reports (`labs/lab3_v2/`):** `README.md`, `config.py`, `run_ex1.py`, `train_ex2.py`, `eval_ex2.py`, `ex1_sentiment.ipynb`, `ex2_finetune.ipynb`, `REPORT.md`, `docs/{README.md, pipelines/modeling.md, models/distilbert.md, eda_report.md, experiment_report.md, evaluation_report.md}`
- **Unit Tests (`test/`):** `test_lab3_v2_ex1.py`, `test_lab3_v2_ex2.py`
- **Artifacts (`experiments/lab3_v2/`):** `README.md`, `ex1/ex1_results.json`, `splits/split_indices.json`, `eda/*`, `runs/distilbert_imdb/{monitor/*, val_eval/*, test_eval/*}` *(loại trừ tuyệt đối thư mục `weights/` chứa `*.safetensors`)*
- **Workflow Records (`.agents/`):** `.agents/tasks/0007_lab3_v2/plan.md`, `.agents/tests/0007_lab3_v2_test_report.md`, `.agents/logs/0007_lab3_v2_log.md`, `.agents/knowledge_extraction/0007_lab3_v2_knowledge.md`
- **Untouched Scope (Verified):** Không sửa `labs/lab3/` (v1), `src/*.py` gốc, `labs/lab1*`, `labs/lab2*`, `src/vibe_scripts/{lab1*,lab2*}`, `experiments/{lab1*,lab2*}`, `.gitignore`.

---

## Deliverables

- **2 Executable Notebooks:** [`ex1_sentiment.ipynb`](../../labs/lab3_v2/ex1_sentiment.ipynb) · [`ex2_finetune.ipynb`](../../labs/lab3_v2/ex2_finetune.ipynb)
- **Báo cáo SRP & Executive Report:** [`REPORT.md`](../../labs/lab3_v2/REPORT.md) · [`eda_report.md`](../../labs/lab3_v2/docs/eda_report.md) · [`experiment_report.md`](../../labs/lab3_v2/docs/experiment_report.md) · [`evaluation_report.md`](../../labs/lab3_v2/docs/evaluation_report.md)

---

## Test Result

- **Stage 1 — Smoke Test:** **PASS** (`2 passed, 4 deselected in 12.37s`).
- **Stage 2 — Normal Test:** **PASS** (`6 passed in 22.44s`, `0` warnings).
- Chi tiết bằng chứng xem tại [`0007_lab3_v2_test_report.md`](../tests/0007_lab3_v2_test_report.md).

---

## Remaining Issues & Improvement Opportunities

1. **Giới hạn `max_length = 256` tokens:** Gây tăng tỷ lệ lỗi `1.71×` trên nhóm review IMDB dài (`40.8%` tập Test). Nâng lên `512` tokens hoặc dùng **Head + Tail truncation (`128 + 384`)** sẽ khắc phục trực tiếp nút thắt lớn nhất này.
2. **Overconfidence từ Epoch 2–3:** `eval_loss` tăng từ `0.2764 (ep 1)` lên `0.3442 (ep 3)` khi fine-tune `66.96M` tham số trên `4,000` mẫu; có thể cải thiện bằng `label_smoothing_factor = 0.05` + `warmup_ratio = 0.1`.
3. **Chưa commit vào Git:** Đang chờ xác nhận của Kỹ sư trước khi thực hiện commit (khi commit cần loại trừ tuyệt đối `experiments/lab3_v2/runs/distilbert_imdb/weights/`).
