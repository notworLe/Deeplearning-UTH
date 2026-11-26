---
title: Model Specification — Multi-Layer Perceptron (MLP)
doc_type: Model
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-25
updated: 2026-11-25
version: 1.0.0
status: approved
---

# Model Specification: Multi-Layer Perceptron (MLP)

> **Implementation:** `src/vibe_scripts/lab1_v2/mlp.py` (`MLPModel`) · **Pipeline:** [`modeling.md`](../pipelines/modeling.md) · **Baseline:** [`logistic_regression.md`](logistic_regression.md)

---

## 1. Động Cơ Thiết Kế

[`LogisticRegressionModel`](logistic_regression.md) bị giới hạn bởi ranh giới quyết định tuyến tính trực tiếp trên 784 pixel thô. **MLP** bổ sung các tầng ẩn nối đầy đủ kèm hàm kích hoạt phi tuyến **`ReLU`** và điều chuẩn **`Dropout`** nhằm:
- Học các tổ hợp đặc trưng phi tuyến đa mức (ví dụ kết hợp đường viền vai + chiều dài thân + phần tay để tách `Shirt`, `Coat`, `Pullover`).
- Giảm mạnh **high bias (underfitting)** so với Logistic Regression.
- Kiểm soát **variance (overfitting)** khi tăng số tham số thông qua `Dropout` và chọn checkpoint theo `val_loss`.

---

## 2. Kiến Trúc Tổng Quát

Với danh sách chiều ẩn cấu hình động `hidden_dims = [H_1, H_2, ..., H_L]` và `dropout = p`:

$$\mathbf{h}_0 = \text{Flatten}(\mathbf{I}) \in \mathbb{R}^{784}$$

$$\mathbf{h}_\ell = \text{Dropout}_p\big(\text{ReLU}(\mathbf{W}_\ell \mathbf{h}_{\ell-1} + \mathbf{b}_\ell)\big), \qquad \ell = 1, \dots, L$$

$$\mathbf{z} = \mathbf{W}_{\text{out}} \mathbf{h}_L + \mathbf{b}_{\text{out}} \in \mathbb{R}^{10} \quad (\text{raw logits})$$

- **Cấu hình Anchor mặc định (`mlp_default`):** `hidden_dims = [256, 128]`, `ReLU`, `dropout = 0.2`.

---

## 3. Biến Đổi Tensor Shape & Quy Mô Tham Số (3 Cấu Hình Ablation)

### Luồng Tensor cho cấu hình Anchor `hidden_dims = [256, 128]`

| Layer | Cấu hình | Input Shape | Output Shape | Số tham số |
| :--- | :--- | :---: | :---: | :---: |
| Input Contract | Normalized Grayscale | `(B, 1, 28, 28)` | `(B, 1, 28, 28)` | `0` |
| `nn.Flatten(1)` | Gộp `(C, H, W)` | `(B, 1, 28, 28)` | `(B, 784)` | `0` |
| Hidden Block 1 | `Linear(784, 256) → ReLU → Dropout(0.2)` | `(B, 784)` | `(B, 256)` | `784 × 256 + 256 = 200,960` |
| Hidden Block 2 | `Linear(256, 128) → ReLU → Dropout(0.2)` | `(B, 256)` | `(B, 128)` | `256 × 128 + 128 = 32,896` |
| Output Head | `Linear(128, 10)` | `(B, 128)` | `(B, 10)` *(raw logits)* | `128 × 10 + 10 = 1,290` |
| **Tổng cộng** | | | | **`235,146` params** |

### So sánh 3 cấu hình `hidden_dims` trong chuỗi Ablation

| Run ID | `hidden_dims` | Độ sâu (Hidden Layers) | Tổng số tham số | Tỉ lệ so với Baseline (`7,850`) |
| :--- | :--- | :---: | :---: | :---: |
| `mlp_h128` | `[128]` | 1 | `101,770` | `13.0×` |
| `mlp_default` | `[256, 128]` | 2 | `235,146` | `30.0×` |
| `mlp_h512_256_128` | `[512, 256, 128]` | 3 | `567,434` | `72.3×` |

- **Fail-Fast Contract:** `forward(x)` kiểm tra `x.ndim == 4` và `tuple(x.shape[1:]) == (1, 28, 28)`; constructor kiểm tra `len(hidden_dims) >= 1`, mọi `h > 0`, `0.0 <= dropout < 1.0`. Không áp `Softmax` bên trong `forward()`.

---

## 4. Góc Nhìn Engineering: Trade-offs & Giới Hạn

- **Điểm mạnh so với Logistic Regression:**
  - Có khả năng xấp xỉ ranh giới quyết định phi tuyến phức tạp → kỳ vọng tạo **gap cải thiện dương rõ rệt so với Baseline Logistic Regression** đo được thực tế trên cùng split ($\Delta\text{Accuracy} > 0$, $\Delta\text{Macro-F1} > 0$ và giảm `Loss`).
  - Kiến trúc linh hoạt theo config (`hidden_dims`, `dropout`), dễ chạy ablation có kiểm soát.
- **Rủi ro & Trade-off cần theo dõi trong thí nghiệm:**
  - **Overfitting khi tăng chiều sâu/epoch:** Số tham số gấp `13×–72×` Baseline nên ở epoch muộn (`epoch = 30`) `train_loss` tiếp tục giảm mạnh trong khi `val_loss` có thể chững hoặc tăng ngược → cần lưu `best.pt` theo `val_loss`.
  - **Nhạy với Learning Rate & Optimizer:** Bài toán tối ưu không lồi (non-convex); `lr` quá lớn (`1e-2`) gây dao động mạnh, `lr` quá nhỏ (`1e-4`) hội tụ chậm trong 20 epochs; `Adam`/`AdamW` thích nghi bước học theo từng tham số nhanh hơn `SGD` cùng `lr`.
- **Giới hạn kiến trúc (so với CNN):**
  - Bước `Flatten(28×28 → 784)` phá vỡ quan hệ lân cận 2D và không có tính bất biến tịnh tiến (translation equivariance) hay chia sẻ trọng số (weight sharing). Nếu ảnh bị dịch vài pixel, độ chính xác của MLP sẽ giảm nhanh hơn CNN.
