# Sanity Check trong Deep Learning

## What

**Sanity Check** là các bài kiểm tra đơn giản nhằm xác nhận **pipeline huấn luyện đang hoạt động hợp lý** trước khi chạy toàn bộ thí nghiệm.

Mục tiêu không phải đánh giá chất lượng model, mà để phát hiện các lỗi logic hoặc bug nghiêm trọng từ sớm.

---

## Why

Trong Deep Learning, nhiều lỗi không làm chương trình crash.

Ví dụ:

- Quên `optimizer.step()`
- Label bị lệch
- Normalize sai
- Learning rate quá nhỏ
- Tensor shape sai nhưng vẫn chạy

Model vẫn train, nhưng sẽ không học được gì.

Sanity Check giúp phát hiện các lỗi này trước khi tốn hàng giờ huấn luyện.

---

## Các Sanity Check phổ biến

### 1. Initial Loss

Với bài toán phân loại `C` lớp:

```
Initial Loss ≈ ln(C)
```

Ví dụ Fashion-MNIST (10 lớp):

```
ln(10) ≈ 2.3026
```

Nếu loss ban đầu quá khác giá trị này, cần kiểm tra lại pipeline.

---

### 2. Overfit 1 Batch

Lấy duy nhất một batch (ví dụ 32 ảnh) và train nhiều epoch.

Kỳ vọng:

- Train Loss → gần 0
- Train Accuracy → gần 100%

Nếu không làm được, khả năng cao pipeline có lỗi.

---

### 3. Tensor Shape

Kiểm tra kích thước tensor sau từng layer.

Ví dụ:

```
Input
[32,1,28,28]

↓

Conv2d

↓

[32,16,28,28]

↓

MaxPool

↓

[32,16,14,14]
```

---

### 4. Gradient & Weight Update

Kiểm tra:

- Gradient có khác 0 không.
- Weight có thay đổi sau `optimizer.step()` không.

Nếu không, model sẽ không học.

---

## Sanity Check ≠ Evaluation

| Sanity Check                    | Evaluation                            |
| ------------------------------- | ------------------------------------- |
| Kiểm tra pipeline có đúng       | Đánh giá chất lượng model             |
| Chạy trước khi train chính thức | Chạy sau khi train                    |
| Phát hiện bug                   | Đo hiệu năng (Accuracy, Loss, F1,...) |

---

## Liên hệ với Software Engineering

Sanity Check khá giống **Smoke Test**.

- **Smoke Test**: Hệ thống có chạy được không?
- **Sanity Check**: Pipeline Deep Learning có hoạt động hợp lý không?

Khác biệt là Sanity Check tập trung vào các lỗi logic trong quá trình huấn luyện thay vì lỗi làm chương trình dừng.
