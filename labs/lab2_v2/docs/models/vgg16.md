---
title: Model Specification — VGG-16 (Transfer Learning)
doc_type: Model
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-27
updated: 2026-11-27
version: 1.0.0
status: approved
---

# Model Specification: `VGG-16` (`torchvision.models.vgg16`)

## 1. Vai trò trong Lab 2 v2

`VGG-16` đại diện cho họ kiến trúc CNN cổ điển (tuần tự phẳng không có skip connection, dùng 3 tầng Fully-Connected lớn ở cuối) với tổng số tham số khổng lồ **`134.41M`** — lớn gấp **`12×`** `ResNet-18` và **`19×`** `DenseNet-121`. Đưa `VGG-16` vào đối chiếu ở chế độ **`transfer`** (chỉ mở khóa tầng phân loại cuối `classifier[6]`) giúp kiểm chứng trực tiếp rằng **nhiều tham số hơn không đồng nghĩa tốt hơn hay hiệu quả hơn**.

---

## 2. Kiến trúc & Luồng Biến Đổi Tensor Shape

| Giai đoạn | Khối `torchvision` | Output Shape | Ghi chú |
| :--- | :--- | :---: | :--- |
| Input | ImageNet-normalized RGB | `(B, 3, 224, 224)` | — |
| Block 1 | `2 × Conv3×3(64) → MaxPool` | `(B, 64, 112, 112)` | Frozen (`features`) |
| Block 2 | `2 × Conv3×3(128) → MaxPool` | `(B, 128, 56, 56)` | Frozen (`features`) |
| Block 3 | `3 × Conv3×3(256) → MaxPool` | `(B, 256, 28, 28)` | Frozen (`features`) |
| Block 4 | `3 × Conv3×3(512) → MaxPool` | `(B, 512, 14, 14)` | Frozen (`features`) |
| Block 5 | `3 × Conv3×3(512) → MaxPool` | `(B, 512, 7, 7)` | Frozen (`features`) |
| AvgPool + Flatten | `AdaptiveAvgPool2d((7, 7)) → Flatten(1)` | `(B, 25088)` | Không có tham số học |
| FC Hidden (`classifier[0..5]`) | `Linear(25088→4096) → ReLU → Dropout(0.5) → Linear(4096→4096) → ReLU → Dropout(0.5)` | `(B, 4096)` | **Frozen** (~`119.5M` params nằm ở 2 lớp này!) |
| **Head (`classifier[6]`)** | **`Linear(4096 → 37)`** *(thay từ `Linear(4096, 1000)`)* | **`(B, 37)`** | **Trainable (`151,589` params)** |

---

## 3. Thống Kê Tham Số & Đánh Đổi Kỹ Thuật

- **Total Params:** `134,412,133` (`~134.41M`, kích thước checkpoint `~513 MB` FP32 — phần lớn tập trung ở lớp `Linear(25088 → 4096)` chiếm `102.76M` tham số!).
- **Trainable Params (`transfer`):** `151,589` (`0.11%` tổng tham số = `4096 × 37 + 37`).
- **Ưu điểm:** Kiến trúc đơn giản, đồng nhất `Conv 3×3`, dễ quan sát khái niệm `features` vs `classifier` khi làm quen Transfer Learning.
- **Hạn chế:**
  - Tiêu tốn VRAM, dung lượng đĩa (checkpoint `> 500 MB`) và FLOPs/độ trễ suy luận cao nhất trong 3 kiến trúc.
  - Không có `BatchNorm` và `Skip Connection`, nên việc train từ đầu hoặc fine-tune sâu khó và kém ổn định hơn `ResNet` / `DenseNet`.
  - Số liệu đối chiếu thực tế về độ chính xác, VRAM, thời gian train và latency suy luận sẽ được ghi nhận tại [`experiment_report.md`](../experiment_report.md) và [`evaluation_report.md`](../evaluation_report.md).
