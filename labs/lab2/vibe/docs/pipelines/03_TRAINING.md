---
title: "Pipe 03: Training Loop — Lab 2 (Oxford-IIIT Pets)"
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
tags: [lab2, pipe-03, training, differential-lr]
---

# Pipe 03: Training Loop

Triển khai tại [`src/vibe_scripts/lab2/train.py`](../../../../../src/vibe_scripts/lab2/train.py).

## 1. Hợp Đồng I/O
- **Input**: `mode ∈ {"baseline", "transfer", "finetune"}`, `train_loader`, `val_loader`, `epochs = 30`, `lr_head = 1e-3`, `lr_backbone = 1e-4`, `seed = 248`, `init_checkpoint` (bắt buộc với `finetune`).
- **Output (`experiments/lab2/<exp_id>/`)**: `best.pt` (theo `val_loss` thấp nhất), `last.pt`, `history.json` (log từng epoch thay cho TensorBoard bị thiếu — xem `logs/errors/0001.md`).

## 2. Cơ Chế Kỹ Thuật Chính
1. **Khóa ngẫu nhiên**: `set_global_seed(seed)` khóa `random`, `numpy`, `torch`, `cuda`.
2. **Bảo toàn trạng thái BatchNorm**: gọi `set_mode_for_training(model, mode)` ở đầu **mỗi epoch** huấn luyện thay vì `model.train()` trần.
3. **Vòng lặp 5 bước + AMP**: `forward (autocast trên CUDA) → CrossEntropyLoss → zero_grad(set_to_none=True) → backward → step`, tách `.detach().item()` chống rò rỉ VRAM, phát hiện `NaN/Inf` fail-fast.
4. **Atomic checkpointing**: ghi `.tmp → replace` đính kèm `model_state_dict`, `mode`, `seed`, `best_epoch`, `best_val_loss`, `trainable_params`, `hyperparams`.

## 3. Tiêu Chí Kiểm Chứng (`test/test_lab2_train.py`)
- **Smoke**: chạy 1 epoch trên subset giả lập, xác nhận sinh đủ `best.pt`, `last.pt`, `history.json`.
- **Normal**: xác minh **bất biến BatchNorm** (`bn1.running_mean` không đổi 1 bit sau bước train ở `transfer` và `finetune`; chỉ `layer4` BN thay đổi ở `finetune`), xác minh weight tầng đóng băng bất biến, bắt `NaN` loss.
