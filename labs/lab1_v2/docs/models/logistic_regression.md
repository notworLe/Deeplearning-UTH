---
title: Model Specification — Logistic Regression Baseline
doc_type: Model
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-25
updated: 2026-11-25
version: 1.0.0
status: approved
---

# Model Specification: Logistic Regression (Baseline)

> **Implementation:** `src/vibe_scripts/lab1_v2/logistic.py` (`LogisticRegressionModel`) · **Pipeline:** [`modeling.md`](../pipelines/modeling.md)

---

## 1. Mục Tiêu Vai Trò Baseline

Trong tư duy ML Engineering, trước khi huấn luyện mạng nơ-ron nhiều tầng cần một **baseline đơn giản, nhanh, ít siêu tham số** để:
- Xác nhận toàn bộ data pipeline, loss, optimizer và evaluation chạy đúng.
- Thiết lập mốc hiệu năng tối thiểu cho mô hình tuyến tính trên không gian pixel phẳng `784` chiều.
- Đo chính xác mức tăng hiệu năng (Accuracy / Macro-F1) mà các tầng ẩn phi tuyến của [`MLPModel`](mlp.md) mang lại so với chi phí tham số tăng thêm.

---

## 2. Kiến Trúc & Công Thức Toán Học

Mô hình thực hiện phép chiếu tuyến tính trực tiếp từ vector ảnh phẳng sang 10 raw logits:

$$\mathbf{x} = \text{Flatten}(\mathbf{I}) \in \mathbb{R}^{784}, \qquad \mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b} \in \mathbb{R}^{10}$$

Trong đó $\mathbf{W} \in \mathbb{R}^{10 \times 784}$ và $\mathbf{b} \in \mathbb{R}^{10}$. Hàm mất mát `nn.CrossEntropyLoss` áp dụng `LogSoftmax` ổn định số học trực tiếp lên raw logits $\mathbf{z}$:

$$\mathcal{L}(\mathbf{z}, y) = -\log \frac{\exp(z_y)}{\sum_{c=0}^{9} \exp(z_c)}$$

---

## 3. Biến Đổi Tensor Shape & Số Tham Số

| Layer | Cấu hình | Input Shape | Output Shape | Số tham số |
| :--- | :--- | :---: | :---: | :---: |
| Input Contract | Normalized Grayscale | `(B, 1, 28, 28)` | `(B, 1, 28, 28)` | `0` |
| `nn.Flatten(1)` | Gộp `(C, H, W)` | `(B, 1, 28, 28)` | `(B, 784)` | `0` |
| `nn.Linear(784, 10)` | Trọng số $\mathbf{W}$ + bias $\mathbf{b}$ | `(B, 784)` | `(B, 10)` *(raw logits)* | `784 × 10 + 10 = 7,850` |
| **Tổng cộng** | | | | **`7,850` params** |

- **Fail-Fast Contract:** `forward(x)` kiểm tra `x.ndim == 4` và `tuple(x.shape[1:]) == (1, 28, 28)` (nếu sai lập tức ném `ValueError`). Không gọi `Softmax` trong `forward()` để tránh double-softmax với `CrossEntropyLoss`.

---

## 4. Góc Nhìn Engineering: Ưu Điểm & Giới Hạn

- **Ưu điểm:**
  - Cực nhẹ (`7,850` tham số, checkpoint ~`35 KB`), suy luận cực nhanh, hầu như không overfit nặng.
  - Hàm mất mát Cross-Entropy trên mô hình tuyến tính là **hàm lồi (convex)** theo $(\mathbf{W}, \mathbf{b})$ → hội tụ ổn định, ít nhạy với khởi tạo trọng số.
  - Mỗi hàng $\mathbf{W}_c \in \mathbb{R}^{784}$ có thể reshape lại thành ảnh `28×28` như một "template" đại diện của lớp $c$, dễ diễn giải.
- **Giới hạn cốt lõi trên FashionMNIST:**
  - Chỉ học được **1 siêu phẳng tuyến tính / lớp** (tương ứng 1 khuôn mẫu trung bình cho mỗi loại trang phục).
  - Không biểu diễn được tương tác phi tuyến giữa các vùng ảnh (ví dụ phân biệt cổ áo, chiều dài tay áo, họa tiết giữa `Shirt`, `T-shirt/top`, `Coat`, `Pullover`) → dễ chạm trần **underfitting**; kết quả đo được thực tế của mô hình này sẽ làm **mốc chuẩn (baseline anchor)** để tính gap cải thiện của MLP.
  - Mất hoàn toàn cấu trúc lân cận 2D sau bước `Flatten`.
