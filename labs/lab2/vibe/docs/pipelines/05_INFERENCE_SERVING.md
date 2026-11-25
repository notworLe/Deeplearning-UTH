---
title: "Pipe 05: Inference & Serving — Lab 2 (Oxford-IIIT Pets)"
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
tags: [lab2, pipe-05, inference, serving]
---

# Pipe 05: Inference & Serving

Triển khai tại [`src/vibe_scripts/lab2/infer.py`](../../../../../src/vibe_scripts/lab2/infer.py).

## 1. Hợp Đồng I/O
- **Input**: Đường dẫn `best.pt` và ảnh đơn (PIL / `Path` / tensor `[3, 224, 224]`) hoặc batch `[B, 3, 224, 224]`.
- **Output**: Dict Top-3 `(class_id, label, species, prob)`, `latency_ms`, `checkpoint_size_kb`.

## 2. Cơ Chế Kỹ Thuật
- Tự khôi phục ResNet-18 (37 lớp) từ checkpoint dưới `@torch.inference_mode()` + `model.eval()`.
- Tiền xử lý tự động ảnh ngoài (`RGB → Resize(256) → CenterCrop(224) → ToTensor → IMAGE_NET_NORMALIZE`).
- Benchmark latency với warm-up + `torch.cuda.synchronize()` cho `batch_size = 1, 32`.

## 3. Tiêu Chí Kiểm Chứng (`test/test_lab2_infer.py`)
- **Smoke**: nạp dummy checkpoint, suy luận 1 ảnh PIL RGB không crash.
- **Normal**: tổng Softmax `= 1.0 ± 1e-5`, Top-3 giảm dần, `requires_grad == False`, xử lý được ảnh grayscale/kích thước bất kỳ.
