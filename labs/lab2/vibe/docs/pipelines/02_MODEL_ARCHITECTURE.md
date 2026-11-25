---
title: "Pipe 02: Model Architecture — Lab 2 (ResNet-18 3 Modes)"
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
tags: [lab2, pipe-02, models, resnet18, transfer, finetune]
---

# Pipe 02: Model Architecture (ResNet-18 — Baseline / Transfer / Fine-tune)

Triển khai tại [`src/vibe_scripts/lab2/models.py`](../../../../../src/vibe_scripts/lab2/models.py).

## 1. Hợp Đồng I/O
- **Input**: Tensor `[B, 3, 224, 224]` (`float32`), kiểm tra fail-fast chiều đầu vào.
- **Output**: Raw logits `[B, 37]` (`float32`, không gắn Softmax trong `forward`).

## 2. Cấu Hình 3 Chế Độ & Số Tham Số

| Chế độ (`mode`) | Khởi tạo trọng số | Tầng mở khóa | Trainable Params | Tỷ lệ | Trạng thái BatchNorm khi train |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `baseline` | `weights=None` | Toàn bộ mạng | `11.195.493` | `100.0%` | Tất cả BN ở `train()` |
| `transfer` | `IMAGENET1K_V1` | Chỉ `fc` (`512→37`) | `18.981` | `0.17%` | Toàn bộ backbone ở `eval()` (khóa running stats) |
| `finetune` | Thay `fc` trước → nạp `resnet18_transfer/best.pt` sau | `layer4` + `fc` | `8.412.709` | `75.14%` | `conv1/bn1/layer1..3` ở `eval()`; `layer4` ở `train()` |

## 3. Ràng Buộc Kỹ Thuật Bắt Buộc
- **Thứ tự nạp checkpoint ở `finetune`**: luôn thay `fc = nn.Linear(512, 37)` **TRƯỚC**, sau đó mới `load_state_dict` từ checkpoint `resnet18_transfer` (đã có `fc` 37 lớp).
- **Khóa BatchNorm**: gọi `set_mode_for_training(model, mode)` đầu mỗi epoch để đảm bảo `model.train()` không vô tình bật lại cập nhật `running_mean/running_var` trên các tầng bị đóng băng.
- **Parameter groups**: `build_optimizer_param_groups(model, mode, lr_head, lr_backbone)` chỉ đưa tham số có `requires_grad=True` vào optimizer; ở `finetune`, `layer4` nhận `lr_backbone` (`1e-4`), `fc` nhận `lr_head` (`1e-3`).

## 4. Tiêu Chí Kiểm Chứng (`test/test_lab2_models.py`)
- **Smoke**: khởi tạo & forward dummy `(2, 3, 224, 224)` cho cả 3 chế độ.
- **Normal**: khớp chính xác số tham số trainable (`11.195.493` / `18.981` / `8.412.709`), kiểm tra thứ tự thay `fc` → load checkpoint `transfer`, kiểm tra `bn.training == False` ở các khối khóa, fail-fast shape sai.
