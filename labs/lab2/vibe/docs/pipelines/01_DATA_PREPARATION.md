---
title: "Pipe 01: Data Preparation & EDA — Lab 2 (Oxford-IIIT Pets)"
doc_type: pipeline-analysis
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
  - ./README.md
  - ../../config.py
tags: [lab2, pipe-01, data, eda, oxford-iiit-pet]
---

# Pipe 01: Data Preparation & EDA

Triển khai tại [`src/vibe_scripts/lab2/data.py`](../../../../../src/vibe_scripts/lab2/data.py) và [`src/vibe_scripts/lab2/eda.py`](../../../../../src/vibe_scripts/lab2/eda.py).

## 1. Hợp Đồng I/O
- **Input**: `OxfordIIITPet` (`split="trainval"` 3.680 ảnh, `split="test"` 3.669 ảnh; `target_types="category"`, 37 lớp).
- **Output**: `(train_loader, val_loader, test_loader, split_meta)` với batch tensor `[B, 3, 224, 224]` (`float32`), `targets [B]` (`int64`, `{0..36}`).

## 2. Cơ Chế Kỹ Thuật Chính
1. **Cleaning**: quét toàn bộ ảnh xác nhận 0 ảnh lỗi/corrupt, chuẩn hóa chế độ màu về `.convert("RGB")` (xử lý ảnh RGBA/Grayscale nếu có), kiểm tra trùng lặp bằng SHA-256 trên byte nội dung file ảnh (gắn cờ trùng trong cùng tập hoặc chéo `trainval`/`test`).
2. **Stratified Split chống rò rỉ**:
   - `Train`: 100% `trainval` (`3.680` ảnh) gắn `train_transform`.
   - Từ `test` gốc (`3.669` ảnh), tách phân tầng theo 37 lớp với `seed = 248`: **10% làm `Val`** (`367` ảnh) và **90% làm `Test`** (`3.302` ảnh). Hai tập rời nhau hoàn toàn và cùng dùng `eval_transform` (khắc phục lỗi truyền nhầm `train_transform` vào `test`/`val` của bản `main/baseline.py` cũ).
3. **Transforms**:
   - `train_transform`: `RGB → RandomResizedCrop(224, scale=(0.7, 1.0)) → RandomHorizontalFlip(0.5) → ToTensor → IMAGE_NET_NORMALIZE`.
   - `eval_transform`: `RGB → Resize(256) → CenterCrop(224) → ToTensor → IMAGE_NET_NORMALIZE`.
4. **Nhận diện loài (Cat vs Dog)**: kiểm chứng quy ước đặt tên file gốc Oxford-IIIT Pets (tên file bắt đầu bằng chữ **HOA** = Mèo, 12 giống; chữ **thường** = Chó, 25 giống).

## 3. Tiêu Chí Kiểm Chứng (`test/test_lab2_data.py`)
- **Smoke**: dựng DataLoader với subset nhỏ không crash, shape `(B, 3, 224, 224)` `float32`.
- **Normal**: `Val ∩ Test = ∅`, phân tầng đủ 37 lớp, `eval_transform` tất định (không dính random crop/flip), nhận diện đúng 12 giống mèo / 25 giống chó, tái lập theo `seed`.
