---
title: Nhật Ký Script AI & Cải Tiến Chức Năng (`src/vibe_scripts/`)
doc_type: log
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-18
updated: 2026-11-18
last_updated_by: Antigravity
version: 0.1.0
status: active
reviewers: []
related_docs:
  - ../../.agents/rules/logs/LOG_VIBE_SCRIPTS.md
  - ../../src/README.md
tags: [log, vibe-scripts, enhancements]
---

# Nhật Ký `src/vibe_scripts/` & Các Chức Năng Cải Tiến

Bảng đăng ký tập trung các module do AI tạo tại `src/vibe_scripts/` và các chức năng cải thiện theo yêu cầu Kỹ sư, tuân thủ [`LOG_VIBE_SCRIPTS.md`](../../.agents/rules/logs/LOG_VIBE_SCRIPTS.md).

---

## Danh Mục Script & Cải Tiến Hiện Có

| Script / Module | Prompt nguồn | Chức năng & Cải thiện chính | Phụ thuộc `src/` | Test kiểm chứng | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :---: |
| [`eda_helpers.py`](../../src/vibe_scripts/eda_helpers.py) | Lab 1 EDA (trước `0001`) | Kiểm toán 6 chiều, stratified split (`seed=248`), thống kê pixel, xác thực `manifest.json` SHA-256 và vẽ EDA 3 Thấu kính (thay thế `src/visualize.py` bị lỗi `tensorboard`) | Độc lập (`src/` giữ nguyên) | Đã chạy qua `lab1.ipynb` | `Active` |
| [`generate_manifest.py`](../../src/vibe_scripts/generate_manifest.py) | Lab 1 Manifest §3.2 | Tự động sinh `manifest.json` và `<file>.manifest.json` kèm SHA-256 cho `data/FashionMNIST/raw/` | — | Chạy trực tiếp | `Active` |
| [`execute_lab1_notebook.py`](../../src/vibe_scripts/execute_lab1_notebook.py) | Lab 1 EDA Execution | Thực thi tuần tự các cell của `labs/lab1/vibe/lab1.ipynb` ngoài ZeroMQ sandbox và nhúng output Rich/PNG sạch | `src/config.py`, `src/consoles.py` | Chạy trực tiếp | `Active` |
| [`config_override.py`](../../src/vibe_scripts/config_override.py) | [`0002_config_and_vibe_logs`](../../prompts/0002_config_and_vibe_logs/prompt.md) | Object `VibeConfig` tự động kế thừa hằng số viết HOA từ `src/config.py`, kiểm tra Fail-Fast đúng tên biến khi override và tách tham số mở rộng `extras` | `src/config.py` | [`test_config_override.py`](../../test/test_config_override.py) | `Active` |
| [`lab1/data.py`](../../src/vibe_scripts/lab1/data.py) | [`0001_lab1`](../../prompts/0001_lab1/prompt.md) | Pipeline 01 Lab 1: xác thực `manifest.json` SHA-256, chia tầng `55k/5k/10k`, fit `mean/std` trên Train, `_FastFashionDataset` trong RAM + `_VectorizedAugCollate` (`F.affine_grid`) nhanh ~10–20× | `src/config.py` (qua `lab1/vibe/config.py`) | [`test_lab1_data.py`](../../test/test_lab1_data.py) | `Active` |
| [`lab1/models.py`](../../src/vibe_scripts/lab1/models.py) | [`0001_lab1`](../../prompts/0001_lab1/prompt.md) | Pipeline 02 Lab 1: 3 mô hình độc lập `LogisticClassifier` (7.850), `MLPClassifier` (109.386), `CNNClassifier` (105.866) khớp `REPORT.md` | `labs/lab1/vibe/config.py` | [`test_lab1_models.py`](../../test/test_lab1_models.py) | `Active` |
| [`lab1/train.py`](../../src/vibe_scripts/lab1/train.py) | [`0001_lab1`](../../prompts/0001_lab1/prompt.md) | Pipeline 03 Lab 1: vòng lặp 5 bước `CrossEntropyLoss` + `Adam`, Baseline sanity check, `tqdm` progress bar từng epoch, Early Stopping tùy chọn, atomic checkpoint `best.pt`/`last.pt` & `history.json` | `labs/lab1/vibe/config.py` | [`test_lab1_train.py`](../../test/test_lab1_train.py) | `Active` |
| [`lab1/evaluate.py`](../../src/vibe_scripts/lab1/evaluate.py) | [`0001_lab1`](../../prompts/0001_lab1/prompt.md) | Pipeline 04 Lab 1: đánh giá Test 1 lần, benchmark 3×2, kiểm chứng Save/Load `< 1e-6`, vẽ learning curves, confusion matrix & error gallery | `labs/lab1/vibe/config.py` | [`test_lab1_evaluate.py`](../../test/test_lab1_evaluate.py) | `Active` |
| [`lab1/infer.py`](../../src/vibe_scripts/lab1/infer.py) | [`0001_lab1`](../../prompts/0001_lab1/prompt.md) | Pipeline 05 Lab 1: nạp `best.pt` tự khôi phục kiến trúc + `norm_stats`, suy luận đơn ảnh/batch dưới `@torch.inference_mode()`, xuất Top-3 Softmax và đo latency | `labs/lab1/vibe/config.py` | [`test_lab1_infer.py`](../../test/test_lab1_infer.py) | `Active` |

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo bảng log cho `src/vibe_scripts/` và đăng ký 4 module hiện có |
