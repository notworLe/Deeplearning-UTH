---
title: "Pipe 03: Training Loop — Phân Tích Đặc Trưng (Lab 1)"
doc_type: pipeline-analysis
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-18
updated: 2026-11-19
last_updated_by: Antigravity
version: 0.3.0
status: draft
reviewers: []
related_docs:
  - ../REPORT.md
  - ./README.md
  - ../../config.py
  - ../../../../../.agents/rules/AI_FIRST_LIFECYCLE.md
  - ../../../../../docs/references/DL_PIPELINE.md
  - ../../../../../src/vibe_scripts/lab1/train.py
  - ../../../../../logs/errors/0001.md
tags: [lab1, pipe-03, training-loop, baseline, feature-analysis]
---

# Pipe 03: Training Loop (Tối Ưu Hóa, Baseline Sanity Check & Checkpointing)

Ánh xạ **Giai đoạn 5 (Training Loop)** trong [`AI_FIRST_LIFECYCLE.md`](../../../../../.agents/rules/AI_FIRST_LIFECYCLE.md) và mục `Training / Baseline` trong [`DL_PIPELINE.md`](../../../../../docs/references/DL_PIPELINE.md), triển khai tại [`src/vibe_scripts/lab1/train.py`](../../../../../src/vibe_scripts/lab1/train.py).

---

## 1. Hợp Đồng Giao Diện (Input / Output)

- **Input**: `model_name ∈ {"logistic", "mlp", "cnn"}`, `aug_enabled: bool`, `(train_loader, val_loader, norm_stats)` từ Pipe 01, siêu tham số từ [`config.py`](../../config.py) (`LR = 1e-3`, `EPOCH = 20`, `SEED = 248`, `BEST_METRIC = "val_loss"`).
- **Output (`experiments/lab1/<model>_<no_aug|aug>/`)**:
  - `best.pt` và `last.pt` chứa đầy đủ: `model_name`, `aug_enabled`, `epoch`, `best_metric_name`, `best_metric_value`, `model_state_dict`, `optimizer_state_dict`, `norm_stats`, `seed`, `hyperparams`.
  - `history.json` ghi chuỗi số liệu theo từng epoch (`train_loss`, `train_acc`, `val_loss`, `val_acc`, `epoch_time_sec`).

---

## 2. Phân Tích Đặc Trưng & Cơ Chế Kỹ Thuật

1. **Vòng lặp 5 bước & Gradient Hygiene**:
   - Thực thi chuẩn `logits = model(x)` → `loss = CrossEntropyLoss(logits, y)` → `optimizer.zero_grad(set_to_none=True)` → `loss.backward()` → `optimizer.step()`.
   - Tích lũy loss có trọng số theo `batch_size` thực tế với `.detach().item()` để không giữ computational graph gây phình VRAM.
2. **Baseline Sanity Check (Mốc tham chiếu trước khi chạy full)**:
   - Kiểm tra initial loss trên `Logistic` gần giá trị lý thuyết đồng phân phối 10 lớp $\ln(10) \approx 2.3026$ và kiểm chứng khả năng overfit 1 batch đơn lẻ trước khi huấn luyện toàn tập 6 cấu hình.
3. **Cân bằng công bằng giữa 3 mô hình**:
   - Cả 3 kiến trúc dùng chung `Adam(lr=1e-3)`, `batch_size = 32`, `epochs = 20` và khóa `seed = 248` trên `random`, `numpy`, `torch`, `cuda`.
   - Mức `lr = 1e-3` override giá trị `1e-5` ở `src/config.py` vì `1e-5` hội tụ quá chậm cho huấn luyện từ đầu trong 20 epoch.
4. **Cách ly tuyệt đối Test Set & Checkpoint tự mô tả**:
   - Không tham chiếu `test_loader` trong `train.py`; `best.pt` đính kèm sẵn `model_name` và `norm_stats` để Pipe 04/05 nạp lại chính xác.

---

## 3. Mục Tiêu

- Tối ưu hóa công bằng cả 6 thí nghiệm (`3 mô hình × 2 cấu hình`), phát hiện sớm lỗi số học (`NaN/Inf`) và lưu giữ an toàn mô hình có `val_loss` tốt nhất.

---

## 4. Hạn Chế & Rủi Ro Đặc Thù

1. **Gián đoạn khi ghi file checkpoint**: Mất điện hoặc ngắt tiến trình giữa lệnh `torch.save` làm hỏng toàn bộ `best.pt`.
2. **Phụ thuộc `tensorboard`**: Môi trường hiện tại thiếu gói `tensorboard` (đã ghi tại [`logs/errors/0001.md`](../../../../../logs/errors/0001.md)); import cứng sẽ làm sập toàn bộ pipeline.
3. **Thất bại ngầm với `NaN/Inf` loss**: Nếu không chặn ngay sẽ tiếp tục chạy đủ 20 epoch vô nghĩa.

---

## 5. Giải Pháp Kỹ Thuật

- Ghi nguyên tử `save_checkpoint_atomic()`: lưu vào file `.tmp` rồi gọi `.replace()` để đổi tên tức thời.
- Ghi nhật ký trung lập bằng JSON (`history.json`) + Rich Console, tách biệt hoàn toàn khỏi `tensorboard`.
- Kiểm tra `torch.isfinite(loss)` sau mỗi bước forward để fail-fast ngay lập tức khi có bất thường số học.

---

## 6. Tiêu Chí Kiểm Chứng (`test/test_lab1_train.py`)

- **Smoke Test**: Chạy 1 epoch trên tập con 64 mẫu sinh đủ `best.pt`, `last.pt`, `history.json`, không sót `.tmp`.
- **Normal Test**: Sanity check `Logistic` có initial loss gần `ln(10)` và overfit 1 batch `acc > 95%`; checkpoint đủ khóa metadata; Early Stopping hoạt động đúng; `NaN` loss ném `FloatingPointError`; không chứa `test_loader`.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.3.0 | 2026-11-19 | notworLe | Antigravity | Viết lại theo cấu trúc phân tích đặc trưng `AI_FIRST_LIFECYCLE.md`, loại bỏ khung prompt mẫu |
| 0.2.0 | 2026-11-18 | notworLe | Antigravity | Đồng bộ `labs/lab1/vibe/config.py` |
