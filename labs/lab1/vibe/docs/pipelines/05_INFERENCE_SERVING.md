---
title: "Pipe 05: Inference & Serving — Phân Tích Đặc Trưng (Lab 1)"
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
  - ../../../../../src/vibe_scripts/lab1/infer.py
tags: [lab1, pipe-05, inference, serving, feature-analysis]
---

# Pipe 05: Inference & Serving (Suy Luận Thực Tế & Đo Độ Trễ)

Ánh xạ **Giai đoạn 7 (Full Integration / Serving)** trong [`AI_FIRST_LIFECYCLE.md`](../../../../../.agents/rules/AI_FIRST_LIFECYCLE.md), triển khai tại [`src/vibe_scripts/lab1/infer.py`](../../../../../src/vibe_scripts/lab1/infer.py).

---

## 1. Hợp Đồng Giao Diện (Input / Output)

- **Input**: Đường dẫn checkpoint `best.pt` và ảnh đầu vào (index tập Test, file ảnh ngoài PNG/JPG, PIL/NumPy, hoặc batch tensor `[B, 1, 28, 28]`).
- **Output**: Dictionary kết quả `model_name`, `checkpoint_size_kb`, `predicted_class_id`, `predicted_label`, `confidence`, `top3`, `latency_ms` và bảng thống kê latency theo `batch_size ∈ {1, 32}`.

---

## 2. Phân Tích Đặc Trưng & Cơ Chế Kỹ Thuật

1. **Tự phục hồi cấu hình chống lệch (Zero Training-Serving Skew)**:
   - `load_predictor()` đọc `model_name` và `norm_stats` trực tiếp từ bên trong `best.pt`; người gọi không cần nhớ lại tham số `mean`/`std` lúc huấn luyện hay truyền thủ công tên kiến trúc.
2. **Chuẩn hóa đa định dạng đầu vào**:
   - `preprocess_image()` quy đổi đồng nhất mọi nguồn (ảnh RGB 3 kênh, kích thước bất kỳ, ndarray, Tensor 2D/3D/4D) về tensor xám `[1, 1, 28, 28]` chuẩn Z-Score.
3. **Tối ưu thực thi suy luận**:
   - Đóng băng mô hình với `model.eval()` và `@torch.inference_mode()` (loại bỏ hoàn toàn chi phí theo dõi autograd và view-versioning so với `no_grad()`).
   - Chuyển đổi raw logits `[B, 10]` sang xác suất qua Softmax để xuất xếp hạng **Top-3**, hữu ích khi chẩn đoán các cặp trang phục gần nhau (`Shirt` vs `T-shirt/top`).
4. **Đo lường chi phí suy luận theo chế độ sử dụng**:
   - Tách riêng độ trễ **`batch_size = 1`** (đại diện tương tác thời gian thực từng ảnh) và **`batch_size = 32`** (đại diện thông lượng xử lý hàng loạt `samples/sec`), có bước warm-up và `torch.cuda.synchronize()`.

---

## 3. Mục Tiêu

- Tách rời hoàn toàn quy trình dự đoán thực tế khỏi mã huấn luyện; sẵn sàng tích hợp vào CLI hoặc notebook để demo kết quả đối chiếu giữa `Logistic`, `MLP` và `CNN`.

---

## 4. Hạn Chế & Rủi Ro Đặc Thù

1. **Sai lệch miền (Domain Shift) với ảnh chụp ngoài thực tế**: Fashion-MNIST là ảnh xám $28 \times 28$ căn giữa trên nền đen tuyệt đối; ảnh chụp thật nền sáng hoặc chưa tách nền sẽ làm độ chính xác giảm mạnh nếu chỉ Resize đơn thuần.
2. **Đọc cứng tham số ở code cũ**: [`labs/lab1/vibe/inference.py`](../../inference.py) cũ hardcode `MEAN`, `STD` và gắn cứng kiến trúc `FashionCNN` cũ.

---

## 5. Giải Pháp Kỹ Thuật

- Đóng gói `norm_stats` và `model_name` đi cùng `best.pt`, đọc động qua `load_predictor()` trong `src/vibe_scripts/lab1/infer.py`.
- Bọc toàn bộ hàm suy luận và benchmark trong `@torch.inference_mode()` để đảm bảo `requires_grad == False` và tốc độ tối ưu.

---

## 6. Tiêu Chí Kiểm Chứng (`test/test_lab1_infer.py`)

- **Smoke Test**: Nạp dummy checkpoint, suy luận 1 tensor `(1, 28, 28)` không crash.
- **Normal Test**: Tổng 10 xác suất Softmax bằng `1.0 ± 1e-5`, Top-3 giảm dần nghiêm ngặt, `requires_grad == False`, xử lý được ảnh RGB kích thước khác `(28, 28)`, `benchmark_latency()` đo đủ cho `batch_size = 1` và `32`.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.3.0 | 2026-11-19 | notworLe | Antigravity | Viết lại theo cấu trúc phân tích đặc trưng `AI_FIRST_LIFECYCLE.md`, loại bỏ khung prompt mẫu |
| 0.2.0 | 2026-11-18 | notworLe | Antigravity | Đồng bộ `labs/lab1/vibe/config.py` |
