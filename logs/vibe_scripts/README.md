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

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo bảng log cho `src/vibe_scripts/` và đăng ký 4 module hiện có |
