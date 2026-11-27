---
title: Test Report — Lab 2 v2 (Oxford-IIIT Pets: Pre-trained CNN Transfer Learning & Fine-Tuning)
doc_type: Test Report
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-28
updated: 2026-11-28
version: 1.0.0
status: completed
---

# Testing Report: `0006_lab2_v2` — Oxford-IIIT Pets Pre-trained CNNs Pipeline

> **Source Task:** [`0006_lab2_v2/task.md`](../tasks/0006_lab2_v2/task.md) · **Approved Plan:** [`0006_lab2_v2/plan.md`](../tasks/0006_lab2_v2/plan.md) · **Standard:** [`TESTING.md`](../rules/dl-rule/TESTING.md)

---

## Summary

- **Overall Result:** **PASS** (100% Smoke Tests, Normal Tests, Layer 4 Sanity Checks, Experimental Acceptance Criteria trên `9/9` runs, Final Held-Out Test, 4-Step Error Analysis, Multi-Arch Inference Benchmark, Notebook Schema và Documentation Link Integrity đều đạt).
- **Environment:** Windows, Python `3.13.14`, PyTorch `2.15.0+cu132`, Torchvision `0.29.2+cu132`, CUDA GPU `NVIDIA GeForce RTX 4060 Laptop GPU`, `tensorboard` installed, `pytest 9.1.1`.

---

## Executed Tests

### Stage 1 — Smoke Test (`pytest -m smoke`)

- **Purpose:** Xác minh import sạch, config `VibeConfig`, dataset tổng hợp 37 lớp, forward pass cả 3 kiến trúc (`ResNet18` scratch/transfer/finetune, `DenseNet121` transfer, `VGG16` transfer) đúng tensor contract `(B, 3, 224, 224) → (B, 37)` hoàn toàn offline (`weights=None`), chạy 1 epoch trên tập nhỏ trong `tmp_path` + per-run Val eval + `PetPredictor` inference.
- **Command:** `pytest -m smoke test/test_lab2_v2_data.py test/test_lab2_v2_models.py test/test_lab2_v2_pipeline.py -v`
- **Result:** **PASS (`7 passed, 7 deselected in 23.44s`)**

| Test Case | Module | Result |
| :--- | :--- | :---: |
| `test_config_and_synthetic_pet_dataset_smoke` | [`test_lab2_v2_data.py`](../../test/test_lab2_v2_data.py) | **PASS** |
| `test_models_forward_smoke[resnet18-scratch]` | [`test_lab2_v2_models.py`](../../test/test_lab2_v2_models.py) | **PASS** |
| `test_models_forward_smoke[resnet18-transfer]` | [`test_lab2_v2_models.py`](../../test/test_lab2_v2_models.py) | **PASS** |
| `test_models_forward_smoke[resnet18-finetune]` | [`test_lab2_v2_models.py`](../../test/test_lab2_v2_models.py) | **PASS** |
| `test_models_forward_smoke[densenet121-transfer]` | [`test_lab2_v2_models.py`](../../test/test_lab2_v2_models.py) | **PASS** |
| `test_models_forward_smoke[vgg16-transfer]` | [`test_lab2_v2_models.py`](../../test/test_lab2_v2_models.py) | **PASS** |
| `test_single_run_per_run_eval_and_inference_smoke` | [`test_lab2_v2_pipeline.py`](../../test/test_lab2_v2_pipeline.py) | **PASS** |

### Stage 2 — Normal Test (Unit, Integration, Functional & Experimental Acceptance)

#### 2.1 Full Pytest Suite (`14/14` PASS — Post-Step 7 Final Re-run)
- **Command:** `pytest test/test_lab2_v2_data.py test/test_lab2_v2_models.py test/test_lab2_v2_pipeline.py -v`
- **Result:** **PASS (`14 passed in 48.76s`)**
- **Verified Invariants:**
  - Stratified split `3,680 / 1,284 / 2,385` không giao nhau trên tập thật `Oxford-IIIT Pets` (`seed = 248`, `37` lớp = `12` mèo + `25` chó), ảnh ép chuẩn `RGB` `(3, 224, 224)` với ImageNet normalization tất định.
  - Exact parameter counts & freeze/unfreeze policies cho cả 3 kiến trúc (`ResNet18`: `11,195,493` / `18,981` / `8,412,709`; `DenseNet121`: `6,991,781` / `37,925`; `VGG16`: `134,412,133` / `151,589`), differential LR groups (`layer4` vs `fc`), frozen `BatchNorm2d` `.eval()` guard và Fail-Fast `ValueError` khi sai tensor shape.
  - Layer 4 Sanity checks (`initial_loss` trong dải quanh `ln 37 ≈ 3.6109`, overfit 1 batch `≥ 95%`), `--resume` nối liên tục lịch sử + khôi phục `optimizer_state_dict`, `test_eval/` không vẽ lặp learning curves, 4-Step Error Analysis pipeline end-to-end, Save/Load Parity `< 1e-6`, và schema `labs/lab2_v2/experiment.ipynb` hợp lệ sau khi thực thi (`nbformat.validate`).

#### 2.2 Experimental Acceptance Criteria & End-to-End Pipeline Validation

| Hạng mục kiểm chứng | Tiêu chí đạt | Kết quả thực tế | Result |
| :--- | :--- | :--- | :---: |
| **6.1–6.3 Data Audit (`prepare_data.py`)** | Xác thực `manifest.json`, quét `7,349` ảnh, 5 biểu đồ EDA + `split_indices.json` | `1/1` SHA-256 pass, `0` corrupt/duplicate, phát hiện & chuẩn hóa `3` ảnh `RGBA → RGB`, 5/5 PNGs + [`eda_report.md`](../../labs/lab2_v2/docs/eda_report.md) | **PASS** |
| **Layer 4 Sanity Checks** | Initial loss quanh `ln 37 ≈ 3.6109`, Overfit 1 batch `≥ 95%` | `actual_init_loss = 4.6785`, Overfit 1 batch (`16` mẫu) đạt `100.0%` sau `5` bước ([`sanity_checks.json`](../../experiments/lab2_v2/ablation/sanity_checks.json)) | **PASS** |
| **Controlled Suite (`9/9` runs + TensorBoard)** | Hoàn tất đủ `9/9` runs (Scratch, Transfer, Fine-Tune, 3 Archs, LR/BS ablation) + TensorBoard | `9/9` runs hoàn tất (`tb=True` trên `9/9`), Save/Load Parity `0.0` toàn bộ, chọn **`densenet121_transfer`** (`93.22%` Val Acc / `93.18%` Macro-F1 / `0.2409` Loss) | **PASS** |
| **Final Held-Out Test (`evaluate.py`, `2,385` mẫu)** | Best Pre-trained vượt trội rõ rệt Baseline Scratch (chấm 1 lần duy nhất) | `densenet121_transfer` (`91.91%` Acc / `91.76%` Macro-F1 / `0.2678` Loss / `99.71%` Species Acc) vs `resnet18_scratch` (`47.09%` / `46.26%` / `1.7643` / `88.34%`) → **`+44.82 pp` Acc, `+45.49 pp` F1**, dương trên **`37/37` giống** | **PASS** |
| **4-Step Deep Error Analysis (`--error-analysis`)** | CM chuẩn hóa %, Gallery High/Low Conf, Audit `N = 60` ảnh thật, Slice Analysis | Sinh đủ 5 biểu đồ + 3 Audit Sheets (`60` mẫu), [`error_taxonomy_60.csv`](../../experiments/lab2_v2/evaluation/error_taxonomy_60.csv) (`50.0%` Model Limitation / `41.7%` Ambiguity / `8.3%` Label Noise), [`slice_analysis.json`](../../experiments/lab2_v2/evaluation/slice_analysis.json) (`Q1 LowRes` Cat `20.41%`) | **PASS** |
| **Offline Inference & Multi-Arch Benchmark** | Top-5 Softmax, `requires_grad=False`, Save/Load `< 1e-6`, đo latency `BS=1 / 32` | `Test[0]` (`Abyssinian_202.jpg`) đoán đúng `Abyssinian` (`97.66%`), diff `0.0`; đo đủ 5 checkpoints tại [`inference_benchmark.json`](../../experiments/lab2_v2/evaluation/inference_benchmark.json) | **PASS** |
| **Executable Notebook & Docs Integrity** | Chạy sạch `experiment.ipynb` (không train lại), `0` broken links, SRP sạch | `5/5` code cells OK (`nbformat.validate` PASS), **`193/193` relative links PASS**, `0` SRP leaks trong `experiment_report.md` | **PASS** |

---

## Failed Tests

- **0 failed tests** tại thời điểm chốt nghiệm thu.

---

## Evidence

- Test suites: [`test_lab2_v2_data.py`](../../test/test_lab2_v2_data.py) · [`test_lab2_v2_models.py`](../../test/test_lab2_v2_models.py) · [`test_lab2_v2_pipeline.py`](../../test/test_lab2_v2_pipeline.py)
- Reports & Notebook: [`eda_report.md`](../../labs/lab2_v2/docs/eda_report.md) · [`experiment_report.md`](../../labs/lab2_v2/docs/experiment_report.md) · [`evaluation_report.md`](../../labs/lab2_v2/docs/evaluation_report.md) · [`experiment.ipynb`](../../labs/lab2_v2/experiment.ipynb)
- Verified JSON artifacts: [`sanity_checks.json`](../../experiments/lab2_v2/ablation/sanity_checks.json) · [`ablation_summary.json`](../../experiments/lab2_v2/ablation/ablation_summary.json) · [`test_comparison.json`](../../experiments/lab2_v2/evaluation/test_comparison.json) · [`species_analysis.json`](../../experiments/lab2_v2/evaluation/species_analysis.json) · [`error_taxonomy_60.json`](../../experiments/lab2_v2/evaluation/error_taxonomy_60.json) · [`slice_analysis.json`](../../experiments/lab2_v2/evaluation/slice_analysis.json) · [`inference_benchmark.json`](../../experiments/lab2_v2/evaluation/inference_benchmark.json)

---

## Recommendation

**Ready for Logging** (toàn bộ Smoke Test, Normal Test, Experimental Acceptance Criteria và Documentation Audits đã đạt 100%, không còn lỗi chặn).
