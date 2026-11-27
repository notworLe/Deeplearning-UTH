---
title: Execution Log — Lab 2 v2 (Oxford-IIIT Pets Pre-trained CNNs Pipeline)
doc_type: Execution Log
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-28
updated: 2026-11-28
version: 1.0.0
status: completed
---

# Execution Log: `0006_lab2_v2` — Oxford-IIIT Pets Transfer Learning & Fine-Tuning Mini-Project

> **Standard:** [`LOGGING.md`](../rules/dl-rule/LOGGING.md) · **Test Report:** [`0006_lab2_v2_test_report.md`](../tests/0006_lab2_v2_test_report.md)

---

## Task

- **Task ID:** `0006_lab2_v2` — [`task.md`](../tasks/0006_lab2_v2/task.md)
- **Approved Plan:** [`plan.md`](../tasks/0006_lab2_v2/plan.md) (`v1.4.0`, `status: completed`)

---

## Objective

Xây dựng mini-project Deep Learning hoàn chỉnh phân loại **37 giống chó & mèo** trên tập ảnh tự nhiên **Oxford-IIIT Pets** (`(3, 224, 224)` RGB, ImageNet-normalized) sử dụng các kiến trúc pre-trained từ `torchvision.models` (`ResNet18`, `DenseNet121`, `VGG16`), tuân thủ trọn bộ 10 tầng [`MODEL_PIPELINE.md`](../rules/dl-rule/MODEL_PIPELINE.md), cô lập hoàn toàn khỏi bản v1 (`labs/lab2/vibe/`, `src/vibe_scripts/lab2/`, `experiments/lab2/`) và áp dụng toàn bộ 6 bài học từ `0005_lab1v2`.

---

## Summary

- **What was changed:**
  - Xây dựng bộ tài liệu thiết kế chuẩn hóa tại `labs/lab2_v2/docs/`: `README.md`, `pipelines/modeling.md` (`v1.0.0`, 10 tầng + quy tắc chọn mô hình 2 tầng §8.1), 3 đặc tả kiến trúc (`models/{resnet18,densenet121,vgg16}.md`) và 3 báo cáo đơn nhiệm SRP (`eda_report.md` Layer 6, `experiment_report.md` Layer 4 & 7, `evaluation_report.md` Layer 8–10).
  - Triển khai 6 module lõi tái sử dụng tại `src/vibe_scripts/lab2_v2/`: `transforms.py` (ImageNet norm + Fail-Fast Tensor Contract + `.convert("RGB")`), `data.py` (6.1 Collection SHA-256 → 6.2 Cleaning quét `7,349` ảnh → 6.3 EDA 5 biểu đồ → 6.4 Stratified split khóa tại `split_indices.json`), `models.py` (`PetClassifier` với freeze/unfreeze `scratch/transfer/finetune`, differential LR groups, frozen `BatchNorm2d` `.eval()` guard), `trainer.py` (`AMP FP16`, `tqdm`, CSV/JSON + TensorBoard `SummaryWriter`, `--resume` + `optimizer_state_dict`, Sanity Checks), `evaluator.py` (per-run Val eval, Final Test 1 lần, 4-Step Error Analysis gồm phân tích loài `Cat vs Dog` + 60-sample Audit Sheets + Slice Analysis) và `inference.py` (`PetPredictor` Top-5 Softmax + latency benchmark).
  - Triển khai cấu hình `VibeConfig` (`labs/lab2_v2/config.py`), 4 thin CLI scripts (`prepare_data.py`, `train.py`, `evaluate.py`, `inference.py`), executable report notebook (`labs/lab2_v2/experiment.ipynb`) và bộ test offline 2 tầng (`test/test_lab2_v2_{data,models,pipeline}.py`).
  - Kỹ sư trực tiếp thực thi trọn bộ **`9/9` training runs** bằng CLI với `--amp` và TensorBoard active (`experiments/lab2_v2/tb_logs/`); sau đó tổng hợp Validation, đánh giá Final Held-Out Test (1 lần duy nhất), kiểm toán trực quan `60` ảnh lỗi thật, đo benchmark suy luận 5 checkpoints, điền 100% số liệu thật vào 2 báo cáo SRP (kèm phân tích đối chiếu nhận xét Kỹ sư trên `resnet18_transfer_lr5e3` và hiện tượng tách đường Train–Val) và thực thi sạch `experiment.ipynb`.
- **Why it was changed:** Định lượng trực tiếp hiệu quả của Transfer Learning & Fine-Tuning so với train từ đầu trên tập ảnh tự nhiên cỡ vừa (`3,680` Train), so sánh đa chiều `ResNet18` vs `DenseNet121` vs `VGG16`, và cung cấp bằng chứng Error Analysis định lượng cho vòng cải tiến tiếp theo.
- **Final Outcome:**
  - **Val Model Selection (`1,284` mẫu, quy tắc `max(val_acc, val_macro_f1, -val_loss)`):** **`densenet121_transfer`** đứng #1 tuyệt đối (`93.22%` Val Acc, `93.18%` Val Macro-F1, `0.2409` Val Loss, `99.61%` Species Acc, chỉ `37.9K` trainable / `6.99M` tổng, `200 MB` VRAM, `423.6s`).
  - **Final Held-Out Test (`2,385` mẫu, chấm 1 lần):** **`densenet121_transfer` đạt `91.91%` Acc / `91.76%` Macro-F1 / `0.2678` Loss / `99.71%` Species Acc** vs **`resnet18_scratch` `47.09%` / `46.26%` / `1.7643` / `88.34%`** — gap cải thiện **`+44.82 pp` Acc, `+45.49 pp` Macro-F1, `-1.4965` Loss**, cải thiện dương trên **`37/37` giống** và giảm lỗi khác loài (`Cat ↔ Dog`) tới **`39.7×`** (`278 → 7` lỗi).
  - **4-Step Error Analysis (`193` lỗi Test, kiểm toán `N = 60` ảnh thật):** **`50.0%` Model Limitation**, **`41.7%` Ambiguity / Framing**, **`8.3%` Label Noise / Overlap**; Slice Analysis phát hiện điểm yếu lớn nhất ở ảnh độ phân giải thấp `Q1 LowRes` (`13.45%` vs `6.33%` HiRes; riêng **Mèo `Q1 LowRes` vọt lên `20.41%`**).
  - **Multi-Arch Inference (`RTX 4060`):** `densenet121_transfer` có checkpoint nhẹ nhất (`27.55 MB`) và nhanh gấp `2×` `VGG16` ở `BS = 32` (`1.88 ms/ảnh` vs `3.77 ms/ảnh`), trong khi `resnet18_transfer` (`42.93 MB`) nhanh nhất ở `BS = 1` (`3.17 ms` vs `20.13 ms`).

---

## Modified Files

- **Reusable Core (`src/vibe_scripts/lab2_v2/`):** `__init__.py`, `README.md`, `transforms.py`, `data.py`, `models.py`, `trainer.py`, `evaluator.py`, `inference.py`
- **Config, CLIs & Notebook (`labs/lab2_v2/`):** `README.md`, `config.py`, `prepare_data.py`, `train.py`, `evaluate.py`, `inference.py`, `experiment.ipynb`
- **Documentation (`labs/lab2_v2/docs/`):** `README.md`, `pipelines/modeling.md`, `models/{resnet18,densenet121,vgg16}.md`, `eda_report.md`, `experiment_report.md` (`v1.1.0`), `evaluation_report.md` (`v1.0.0`)
- **Unit Tests (`test/`):** `test_lab2_v2_data.py`, `test_lab2_v2_models.py`, `test_lab2_v2_pipeline.py`
- **Artifacts (`experiments/lab2_v2/`):** `README.md`, `splits/split_indices.json`, `eda/*`, `tb_logs/*` (`41.2 KB`), `ablation/*`, `evaluation/*` (gồm `audit_sheets/*`, `error_taxonomy_60.{csv,json}`, `slice_analysis.json`, `inference_benchmark.json`), `figures/*`, `runs/<9 runs>/{config.json, historys/*, eval/*, test_eval/*}` *(loại trừ toàn bộ `*.pt`)*
- **Workflow Records (`.agents/`):** `.agents/tasks/0006_lab2_v2/{task.md, plan.md}`, `.agents/tests/0006_lab2_v2_test_report.md`, `.agents/logs/0006_lab2_v2_log.md`, `.agents/knowledge_extraction/0006_lab2_v2_knowledge.md`
- **Untouched Scope (Verified):** Không sửa `src/*.py` gốc, `src/vibe_scripts/lab1*`, `src/vibe_scripts/lab2/` (v1), `labs/lab1*`, `labs/lab2/{main,vibe,submit}/`, `experiments/lab1*`, `experiments/lab2/`, `data/raw/oxford-iiit-pet/`, `.gitignore`, `README.md` gốc.

---

## Deliverables

- **Báo cáo kỹ thuật SRP:** [`modeling.md`](../../labs/lab2_v2/docs/pipelines/modeling.md) · [`eda_report.md`](../../labs/lab2_v2/docs/eda_report.md) · [`experiment_report.md`](../../labs/lab2_v2/docs/experiment_report.md) · [`evaluation_report.md`](../../labs/lab2_v2/docs/evaluation_report.md) · [`experiment.ipynb`](../../labs/lab2_v2/experiment.ipynb).
- **TensorBoard Monitoring Logs:** `experiments/lab2_v2/tb_logs/` (`9/9` runs, mở trực tiếp bằng `tensorboard --logdir experiments/lab2_v2/tb_logs`).

---

## Test Result

- **Stage 1 — Smoke Test:** **PASS** (`7 passed, 7 deselected in 23.44s`).
- **Stage 2 — Normal Test:** **PASS** (`14 passed in 48.76s`).
- **Verification khác:** `193/193` relative Markdown links PASS (`0` broken); `nbformat.validate` PASS sau khi chạy `5/5` code cells; Save/Load Parity `0.0 < 1e-6` trên `9/9` checkpoints; `0` SRP leaks.
- Chi tiết bằng chứng xem tại [`0006_lab2_v2_test_report.md`](../tests/0006_lab2_v2_test_report.md).

---

## Remaining Issues & Optional Follow-up

1. **Thu hẹp Train–Val gap (`+5.6 pp → +10.5 pp`) & bứt phá trên `94%` Test Acc (tùy chọn vòng cải tiến sau commit này):** Bổ sung regularization (`Dropout(0.3)` trước head, `label_smoothing = 0.1`, `weight_decay = 1e-2`, `ColorJitter` + `RandomErasing`), mở khóa chọn lọc `denseblock4 + norm5` của `DenseNet121` với differential LR (`5e-5 / 1e-3`) và nâng độ phân giải lên `288–384 px` để đánh thẳng vào `50.0%` Model Limitation (cụm chó Bully & mèo màu điểm) và `20.41%` lỗi Mèo `Q1 LowRes`.
2. **Quy tắc `.gitignore` (`runs/`):** Tương tự `lab1_v2`, `.gitignore` gốc chặn mọi thư mục `runs/`, nên bước commit sử dụng `git add -f` chọn lọc chỉ các file nhẹ `.json/.csv/.png` trong `experiments/lab2_v2/runs/` và tuyệt đối không commit `*.pt`.
3. **Gán nhãn `error_taxonomy_60.csv`:** 60 nhãn nguyên nhân được kiểm toán trực quan trên 3 Audit Sheets (`#1–#60`); Kỹ sư có thể rà soát/hiệu chỉnh trực tiếp trong [`error_taxonomy_60.csv`](../../experiments/lab2_v2/evaluation/error_taxonomy_60.csv) và vẽ lại biểu đồ bằng `plot_error_taxonomy_breakdown()`.
