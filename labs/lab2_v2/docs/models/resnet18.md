---
title: Model Specification — ResNet-18 (Scratch / Transfer / Fine-Tune)
doc_type: Model
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-27
updated: 2026-11-27
version: 1.0.0
status: approved
---

# Model Specification: `ResNet-18` (`torchvision.models.resnet18`)

## 1. Vai trò trong Lab 2 v2

`ResNet-18` đóng vai trò **mô hình xương sống (anchor)** để kiểm chứng trọn vẹn 3 chế độ học trên **Oxford-IIIT Pets (37 lớp)**:
1. **`resnet18_scratch` (Baseline):** khởi tạo trọng số ngẫu nhiên (`weights=None`), train toàn bộ `11.20M` tham số từ đầu để làm mốc đối chứng thực nghiệm.
2. **`resnet18_transfer` (Feature Extraction):** nạp trọng số ImageNet (`weights="DEFAULT"`), **đóng băng toàn bộ backbone**, chỉ train lớp phân loại cuối `fc` (`18,981` tham số = `0.17%`).
3. **`resnet18_finetune` (Partial Fine-Tuning):** nạp trọng số ImageNet, **mở khóa `layer4` + `fc`** (`8.41M` tham số = `75.14%`) với **differential learning rate** (`layer4: 1e-4`, `fc: 1e-3`) để tinh chỉnh đặc trưng cấp cao phù hợp 37 giống thú cưng mà không làm vỡ bộ lọc cấp thấp.

---

## 2. Kiến trúc & Luồng Biến Đổi Tensor Shape

`ResNet-18` sử dụng kết nối tắt (**Residual Skip Connection**): $y = \mathcal{F}(x, \{W_i\}) + x$, giải quyết suy biến gradient khi mạng sâu 18 tầng trọng số.

| Giai đoạn | Khối `torchvision` | Output Shape | Ghi chú Freeze/Unfreeze |
| :--- | :--- | :---: | :--- |
| Input | ImageNet-normalized RGB | `(B, 3, 224, 224)` | — |
| Stem | `Conv2d(3→64, k=7, s=2) → BN → ReLU → MaxPool(3, s=2)` | `(B, 64, 56, 56)` | Freeze ở `transfer` & `finetune` |
| Stage 1 | `layer1` (`2 × BasicBlock(64)`) | `(B, 64, 56, 56)` | Freeze ở `transfer` & `finetune` |
| Stage 2 | `layer2` (`2 × BasicBlock(128, s=2)`) | `(B, 128, 28, 28)` | Freeze ở `transfer` & `finetune` |
| Stage 3 | `layer3` (`2 × BasicBlock(256, s=2)`) | `(B, 256, 14, 14)` | Freeze ở `transfer` & `finetune` |
| Stage 4 | `layer4` (`2 × BasicBlock(512, s=2)`) | `(B, 512, 7, 7)` | Freeze ở `transfer`; **Unfreeze (`lr=1e-4`) ở `finetune`** |
| Pool | `AdaptiveAvgPool2d((1, 1)) → Flatten(1)` | `(B, 512)` | Không có tham số học |
| **Head** | **`fc: Linear(512 → 37)`** *(thay từ `Linear(512, 1000)`)* | **`(B, 37)`** | **Luôn Trainable (`lr=1e-3`)** |

---

## 3. Thống Kê Tham Số & Cấu Hình Tối Ưu

| Chế độ | Total Params | Trainable Params | Tỷ lệ Trainable | Optimizer & LR |
| :--- | :---: | :---: | :---: | :--- |
| `scratch` | `11,195,493` | `11,195,493` | `100.00%` | `AdamW(lr=1e-3, wd=1e-4)` + `CosineAnnealingLR` |
| `transfer` | `11,195,493` | `18,981` | `0.17%` | `AdamW(lr=1e-3, wd=1e-4)` + `CosineAnnealingLR` |
| `finetune` | `11,195,493` | `8,412,709` | `75.14%` | `AdamW([layer4: 1e-4, fc: 1e-3], wd=1e-4)` + `CosineAnnealingLR` |

- **Lưu ý BatchNorm khi Freeze:** Ở chế độ `transfer`, các tầng `BatchNorm2d` trong backbone bị đóng băng được giữ ở trạng thái `eval()` để không làm trôi `running_mean / running_var` của ImageNet theo batch nhỏ (`32` ảnh).

---

## 4. Ưu Điểm, Hạn Chế & Giả Thuyết Cần Kiểm Chứng

- **Ưu điểm:** Nhẹ (`~43 MB` FP32), tốc độ train & suy luận nhanh, kết nối residual ổn định gradient khi fine-tune.
- **Hạn chế:** Ở chế độ `scratch` trên tập train nhỏ (`3,680` ảnh / `37` lớp ≈ `100` ảnh/lớp), dung lượng `11.2M` tham số dễ overfit hoặc hội tụ chậm; ở chế độ `transfer`, đặc trưng cố định của ImageNet có thể chưa tối ưu hoàn toàn cho các cặp giống rất giống nhau (fine-grained breeds).
- **Giả thuyết thực nghiệm:** Kỳ vọng `resnet18_transfer` và `resnet18_finetune` đều tạo **gap cải thiện dương rõ rệt** về Validation/Test Accuracy và tốc độ hội tụ so với `resnet18_scratch` (số liệu thực tế sẽ ghi nhận sau khi Kỹ sư chạy Step 6 tại [`experiment_report.md`](../experiment_report.md) và [`evaluation_report.md`](../evaluation_report.md)).
