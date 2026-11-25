---
title: "Lab 2 (Vibe) — Oxford-IIIT Pets: Baseline vs Transfer vs Fine-tune"
doc_type: readme
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-21
updated: 2026-11-21
last_updated_by: Antigravity
version: 0.1.0
status: active
related_docs:
  - ./config.py
  - ./lab2.ipynb
  - ./docs/REPORT.md
  - ./docs/pipelines/README.md
  - ../QUESTION.md
tags: [lab2, oxford-iiit-pet, resnet18, transfer-learning, fine-tuning]
---

# Lab 2 (Vibe) — Đánh Giá Fine-tune Trên Oxford-IIIT Pets

Không gian thực nghiệm chuẩn hóa cho Lab 2 (ResNet-18 trên 37 giống chó/mèo Oxford-IIIT Pets), đối sánh công bằng 3 cấu hình: `resnet18_baseline` (từ đầu) → `resnet18_transfer` (freeze backbone, train `fc`) → `resnet18_finetune` (nạp checkpoint transfer, unfreeze `layer4 + fc`).

## Cấu trúc thư mục

| Thành phần | Vai trò | Liên kết |
| :--- | :--- | :--- |
| Cấu hình override | Kế thừa `src/config.py` qua `VibeConfig` (`LR=1e-3`, `LR_BACKBONE=1e-4`, `EPOCH=30`, `SEED=248`) | [`config.py`](./config.py) |
| Notebook thực thi | Đồng bộ flow & phân tích với `REPORT.md`, gọi 5 pipeline từ `src/vibe_scripts/lab2/` | [`lab2.ipynb`](./lab2.ipynb) |
| Báo cáo chính | Báo cáo định lượng ngắn gọn, xúc tích kèm biểu đồ và kết luận | [`docs/REPORT.md`](./docs/REPORT.md) |
| Đặc tả 5 Pipeline | Chi tiết kỹ thuật ngắn gọn tách riêng khỏi báo cáo | [`docs/pipelines/README.md`](./docs/pipelines/README.md) |
| Đề bài gốc | Yêu cầu thực hành Pretrained & Transfer Learning | [`../QUESTION.md`](../QUESTION.md) |
