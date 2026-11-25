---
title: "Lab 2 Pipelines — Chỉ Mục 5 Pipeline Độc Lập"
doc_type: pipeline-index
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
  - ../../config.py
  - ../../../../prompts/0003_lab2/plans/README.md
tags: [lab2, pipelines, index]
---

# Chỉ Mục 5 Pipeline Độc Lập — Lab 2 (Oxford-IIIT Pets)

Toàn bộ mã nguồn đặt tại `src/vibe_scripts/lab2/`, dùng chung cấu hình override tại [`labs/lab2/vibe/config.py`](../../config.py).

| Pipeline | Module mã nguồn | Tài liệu đặc tả | Unit Test | Trách nhiệm chính |
| :---: | :--- | :--- | :--- | :--- |
| **01** | [`data.py`](../../../../src/vibe_scripts/lab2/data.py), [`eda.py`](../../../../src/vibe_scripts/lab2/eda.py) | [`01_DATA_PREPARATION.md`](./01_DATA_PREPARATION.md) | [`test_lab2_data.py`](../../../../test/test_lab2_data.py) | Cleaning (corrupt/RGB/dup), EDA 37 giống, stratified split `3.680 / 367 / 3.302`, transforms & DataLoader |
| **02** | [`models.py`](../../../../src/vibe_scripts/lab2/models.py) | [`02_MODEL_ARCHITECTURE.md`](./02_MODEL_ARCHITECTURE.md) | [`test_lab2_models.py`](../../../../test/test_lab2_models.py) | 3 chế độ ResNet-18 (`baseline`, `transfer`, `finetune`), thứ tự thay `fc` trước nạp ckpt, khóa BN `eval()` |
| **03** | [`train.py`](../../../../src/vibe_scripts/lab2/train.py) | [`03_TRAINING.md`](./03_TRAINING.md) | [`test_lab2_train.py`](../../../../test/test_lab2_train.py) | Vòng lặp 5 bước, AMP, differential LR (`1e-4 / 1e-3`), khóa BN phần đóng băng mỗi epoch, atomic save |
| **04** | [`evaluate.py`](../../../../src/vibe_scripts/lab2/evaluate.py) | [`04_EVALUATION_BENCHMARK.md`](./04_EVALUATION_BENCHMARK.md) | [`test_lab2_evaluate.py`](../../../../test/test_lab2_evaluate.py) | Đánh giá Test 1 lần (`3.302` ảnh), Save/Load `< 1e-6`, Cat vs Dog error rate, CM 37×37, Error Gallery |
| **05** | [`infer.py`](../../../../src/vibe_scripts/lab2/infer.py) | [`05_INFERENCE_SERVING.md`](./05_INFERENCE_SERVING.md) | [`test_lab2_infer.py`](../../../../test/test_lab2_infer.py) | Nạp `best.pt`, suy luận đơn ảnh/batch dưới `@torch.inference_mode()`, Top-3 Softmax & latency |
