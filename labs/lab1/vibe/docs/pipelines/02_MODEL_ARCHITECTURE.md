---
title: "Pipe 02: Model Architecture — Phân Tích Đặc Trưng (Lab 1)"
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
  - ../../../../../src/vibe_scripts/lab1/models.py
tags: [lab1, pipe-02, model-architecture, feature-analysis]
---

# Pipe 02: Model Architecture (Kiến Trúc & Trích Xuất Đặc Trưng)

Ánh xạ **Giai đoạn 4 (Model Architecture)** trong [`AI_FIRST_LIFECYCLE.md`](../../../../../.agents/rules/AI_FIRST_LIFECYCLE.md), triển khai tại [`src/vibe_scripts/lab1/models.py`](../../../../../src/vibe_scripts/lab1/models.py).

---

## 1. Hợp Đồng Giao Diện (Input / Output)

- **Input**: Tensor `x` kích thước `[B, 1, 28, 28]` (`float32`).
- **Output**: Raw logits kích thước `[B, 10]` (`float32`), **không áp dụng Softmax** bên trong `forward()`.

| Kiến trúc | Luồng biến đổi Tensor | Tham số | Vai trò trong thí nghiệm |
| :--- | :--- | :---: | :--- |
| **`LogisticClassifier`** | `[B,1,28,28] → Flatten [B,784] → Linear(784→10)` | `7.850` | Mốc sàn tuyến tính (Baseline) |
| **`MLPClassifier`** | `[B,1,28,28] → Flatten [B,784] → Linear(784→128) → ReLU → Linear(128→64) → ReLU → Linear(64→10)` | `109.386` | Đối chứng phi tuyến không có cấu trúc 2D |
| **`CNNClassifier`** | `[B,1,28,28] → Conv(1→16,3×3,p=1) → ReLU → Pool(2) [B,16,14,14] → Conv(16→32,3×3,p=1) → ReLU → Pool(2) [B,32,7,7] → Flatten [B,1568] → Linear(1568→64) → ReLU → Linear(64→10)` | `105.866` | Mô hình trọng tâm khai thác tương quan không gian 2D |

---

## 2. Phân Tích Đặc Trưng & Cơ Chế Kỹ Thuật

1. **Khả năng biểu diễn không gian (Spatial Inductive Bias)**:
   - `Logistic` chỉ học 10 siêu phẳng phân tách tuyến tính trên 784 pixel độc lập; dễ bế tắc khi các lớp thân trên (`Shirt`, `T-shirt/top`, `Pullover`, `Coat`) có độ tương đồng cấu trúc cosine $> 0.85$.
   - `MLP` bổ sung biến đổi phi tuyến qua 2 tầng ẩn ReLU, học được tổ hợp điểm ảnh phức tạp hơn, nhưng bước `Flatten` ban đầu phá hủy cấu trúc lân cận 2D và làm mô hình nhạy với phép dịch chuyển nhỏ.
   - `CNN` giữ nguyên cấu trúc lưới 2D qua tích chập $3 \times 3$ (chia sẻ trọng số cục bộ) kết hợp `MaxPool2d(2)` giảm chiều $28 \times 28 \to 14 \times 14 \to 7 \times 7$, tạo bất biến dịch chuyển cục bộ và tách lọc đặc trưng vi mô (cổ áo, tay áo, đế giày).
2. **Thiết kế công bằng dung lượng tham số**:
   - `CNN` (`105.866`) và `MLP` (`109.386`) có tổng tham số cùng bậc (chênh lệch ~3,2%), giúp chứng minh phần hiệu năng vượt trội đến từ cấu trúc tích chập chứ không phải do phình kích thước mạng.
3. **Tách rời hoàn toàn khỏi Training Loop**:
   - Lớp mô hình chỉ chứa tầng biến đổi tensor; hàm mất mát `CrossEntropyLoss` và `Adam` nằm hoàn toàn bên ngoài tại Pipe 03.

---

## 3. Mục Tiêu

- Chuẩn hóa cùng một chữ ký `forward(x) -> logits` và cơ chế khởi tạo `build_model(model_name)` để Pipe 03, 04, 05 thay thế chéo 3 mô hình không phải sửa code huấn luyện/đánh giá.

---

## 4. Hạn Chế & Rủi Ro Đặc Thù

1. **Mất khớp chiều giữa mã cũ và `REPORT.md`**: `src/model.py` (`FashionCNN` 32→64→128 + BatchNorm + Dropout) khác hẳn kiến trúc trong `REPORT.md`; nếu import nhầm sẽ phá vỡ ràng buộc cùng bậc tham số với `MLP`.
2. **Mất đồng bộ số liệu Loss**: Thêm `Softmax` cuối `forward()` sẽ làm `CrossEntropyLoss` tính log-softmax hai lần, bóp nghẹt gradient.

---

## 5. Giải Pháp Kỹ Thuật

- Cài đặt sạch 3 lớp tại `src/vibe_scripts/lab1/models.py` theo đúng `REPORT.md`.
- Kiểm tra Fail-Fast shape đầu vào `(B, 1, 28, 28)` ở hàm `_validate_input_shape(x)` ngay đầu `forward()`.
- Trả thẳng raw logits `[B, 10]`.

---

## 6. Tiêu Chí Kiểm Chứng (`test/test_lab1_models.py`)

- **Smoke Test**: Khởi tạo và forward `torch.randn(2, 1, 28, 28)` không crash.
- **Normal Test**: Đếm tham số khớp đúng `7.850` / `109.386` / `105.866`, hỗ trợ `B = 1`, không chứa `Softmax` hay `torch.optim`, backward đủ gradient hữu hạn, ném `ValueError` khi sai shape.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.3.0 | 2026-11-19 | notworLe | Antigravity | Viết lại theo cấu trúc phân tích đặc trưng `AI_FIRST_LIFECYCLE.md`, loại bỏ khung prompt mẫu |
| 0.2.0 | 2026-11-18 | notworLe | Antigravity | Cập nhật theo `REPORT.md` |
