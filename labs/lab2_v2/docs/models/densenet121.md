---
title: Model Specification — DenseNet-121 (Transfer Learning)
doc_type: Model
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-27
updated: 2026-11-27
version: 1.0.0
status: approved
---

# Model Specification: `DenseNet-121` (`torchvision.models.densenet121`)

## 1. Vai trò trong Lab 2 v2

`DenseNet-121` đại diện cho họ kiến trúc **tái sử dụng đặc trưng dày đặc (Dense Feature Reuse)** — tổng số tham số nhỏ hơn cả `ResNet-18` (`6.99M` vs `11.20M`) nhưng độ sâu lớn (`121` lớp) và biểu diễn đặc trưng cuối rộng gấp đôi (`1,024` chiều vs `512` chiều). Trong Lab 2 v2, `DenseNet-121` được đánh giá ở chế độ **`transfer`** (đóng băng toàn bộ `features`, chỉ train `classifier`).

---

## 2. Kiến trúc & Luồng Biến Đổi Tensor Shape

Trong mỗi **Dense Block**, tầng thứ $\ell$ nhận phép nối kênh (*channel concatenation*) của toàn bộ feature maps từ các tầng trước đó: $x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}])$ (growth rate $k = 32$), giúp bảo toàn cả đặc trưng kết cấu lông/màu cấp thấp lẫn hình dạng cấp cao tới tận lớp phân loại.

| Giai đoạn | Khối `torchvision` | Output Shape | Ghi chú |
| :--- | :--- | :---: | :--- |
| Input | ImageNet-normalized RGB | `(B, 3, 224, 224)` | — |
| Stem | `Conv2d(3→64, k=7, s=2) → BN → ReLU → MaxPool(3, s=2)` | `(B, 64, 56, 56)` | Frozen |
| DenseBlock 1 + Trans 1 | `6` dense layers → compression `0.5` | `(B, 128, 28, 28)` | Frozen |
| DenseBlock 2 + Trans 2 | `12` dense layers → compression `0.5` | `(B, 256, 14, 14)` | Frozen |
| DenseBlock 3 + Trans 3 | `24` dense layers → compression `0.5` | `(B, 512, 7, 7)` | Frozen |
| DenseBlock 4 + Norm5 | `16` dense layers → `BatchNorm2d(1024)` | `(B, 1024, 7, 7)` | Frozen |
| Pool | `ReLU → AdaptiveAvgPool2d((1, 1)) → Flatten(1)` | `(B, 1024)` | Không có tham số học |
| **Head** | **`classifier: Linear(1024 → 37)`** *(thay từ `Linear(1024, 1000)`)* | **`(B, 37)`** | **Trainable (`37,925` params)** |

---

## 3. Thống Kê Tham Số & Đánh Đổi Kỹ Thuật

- **Total Params:** `6,991,781` (`~6.99M` — **nhỏ nhất** trong 3 kiến trúc).
- **Trainable Params (`transfer`):** `37,925` (`0.54%` tổng tham số = `1024 × 37 + 37`).
- **Ưu điểm:**
  - Hiệu quả tham số rất cao nhờ tái sử dụng feature maps qua phép concat thay vì cộng (`+`).
  - Biểu diễn cuối `1,024` chiều giữ tốt chi tiết fine-grained (hoa văn lông, hình dạng tai/mõm) nên thường rất mạnh ở chế độ Feature Extraction (`transfer`).
- **Hạn chế:**
  - Chi phí bộ nhớ trung gian (activation memory) và thời gian forward cao hơn `ResNet-18` do nhiều phép nối tensor dọc mạng (giảm đáng kể khi bật `--amp` FP16).
  - Kiểm chứng thực nghiệm đối chiếu với `ResNet-18` và `VGG-16` sẽ được ghi nhận tại [`experiment_report.md`](../experiment_report.md) và [`evaluation_report.md`](../evaluation_report.md).
