---
title: "Pipeline 05: Inference & Serving — Lab 1 (Fashion-MNIST)"
doc_type: pipeline-spec
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-18
updated: 2026-11-18
last_updated_by: Antigravity
version: 0.1.0
status: draft
reviewers: []
related_docs:
  - ../README.md
  - ./README.md
  - ./02_MODEL_ARCHITECTURE.md
  - ./04_EVALUATION_BENCHMARK.md
  - ../REPORT.md
  - ../QUESTION.md
  - ../../../../../.agents/temp/06_PIPELINE_SEPARATION.md
  - ../../../../../.agents/rules/BEHAVIORAL_LAYER.md
tags: [lab1, pipeline-05, inference, serving, latency]
---

# Pipeline 05: Inference & Serving (Suy Luận Đơn Ảnh, Batch & Đo Độ Trễ)

Tài liệu đặc tả độc lập cho **Pipeline 5** trong hệ thống phân tách 5 pipeline của Lab 1, đáp ứng yêu cầu lưu/nạp mô hình, hiển thị dự đoán so với nhãn thực và đo thời gian suy luận tại [`labs/lab1/vibe/docs/REPORT.md`](../REPORT.md) và [`labs/lab1/vibe/docs/QUESTION.md`](../QUESTION.md).

---

## 1. Tài Liệu Đặc Tả Kỹ Thuật

### 1.1. Phạm vi & Ranh giới trách nhiệm
- **Trách nhiệm triển khai**:
  - Nạp bất kỳ checkpoint `best.pt` nào trong 3 kiến trúc (`logistic`, `mlp`, `cnn`), tự động phục hồi đúng kiến trúc và tham số chuẩn hóa `norm_stats` đi kèm trong checkpoint.
  - Cung cấp giao diện suy luận nhẹ cho **1 ảnh đơn lẻ** (từ chỉ số tập Test hoặc đường dẫn file ảnh PNG/JPG bên ngoài được tự động chuyển ảnh xám và resize về $28 \times 28$) và **suy luận theo lô (batch)**.
  - Trả về nhãn dự đoán Top-1, phân phối xác suất Top-3 qua Softmax và đo độ trễ suy luận (`ms/sample`, thông lượng `samples/sec`) cùng kích thước file checkpoint (KB).
- **Ngoài phạm vi (Out-of-Scope)**:
  - Không huấn luyện, không tính gradient, không cần triển khai web server hay đóng gói Docker vì vượt phạm vi bài Lab 1.

### 1.2. Hợp đồng Đầu vào / Đầu ra (Input / Output Contracts)
- **Input**:
  - `checkpoint_path: Path` — đường dẫn đến `best.pt` từ Pipeline 03.
  - Đầu vào ảnh (chọn 1 trong 3):
    1. `sample_idx: int` từ tập Test Fashion-MNIST (`0..9999`).
    2. `image_path: Path` — ảnh bất kỳ bên ngoài (tự động chuyển Grayscale → Resize `(28, 28)` → `ToTensor` → `Normalize(mean, std)`).
    3. `image_tensor: torch.Tensor` shape `[1, 28, 28]` hoặc `[B, 1, 28, 28]`.
- **Output**:
  - Dictionary kết quả chuẩn hóa:
    ```python
    {
        "model_name": str,
        "checkpoint_size_kb": float,
        "predicted_class_id": int,
        "predicted_label": str,
        "confidence": float,          # xác suất Top-1 trong [0.0, 1.0]
        "top3": [
            {"class_id": int, "label": str, "prob": float},
            ...
        ],
        "latency_ms": float,
    }
    ```

### 1.3. Quy Trình Tiền Xử Lý, Suy Luận & Đo Độ Trễ
1. **Tiền xử lý nhất quán (No Training-Serving Skew)**:
   - Bắt buộc đọc `mean` và `std` từ `ckpt["norm_stats"]` thay vì hardcode, đảm bảo phân bố đầu vào khi suy luận trùng khớp 100% với lúc huấn luyện.
   - Với ảnh thực tế từ file ngoài: chuyển ảnh xám 1 kênh (`Image.convert("L")`) → `Resize((28, 28))` → `ToTensor()` → `Normalize((mean,), (std,))` → thêm chiều batch `[1, 1, 28, 28]`.
2. **Thực thi suy luận tối ưu**:
   - Đưa mô hình về `model.eval()` và bọc toàn bộ hàm suy luận trong `@torch.inference_mode()` (nhanh và tiết kiệm bộ nhớ hơn `torch.no_grad()`).
   - Chuyển logits `[B, 10]` sang xác suất bằng `torch.softmax(logits, dim=1)` và lấy `topk(3, dim=1)`.
3. **Đo lường độ trễ chuẩn xác**:
   - Chạy `warmup_runs = 20` lần forward trước khi đo để loại bỏ chi phí cấp phát CUDA/cache ban đầu.
   - Chạy lặp `benchmark_runs = 200` lần với `time.perf_counter()` (gọi `torch.cuda.synchronize()` trước và sau khi bấm giờ nếu chạy CUDA) để báo cáo `mean_latency_ms`, `p95_latency_ms` và `throughput_samples_per_sec` cho cả `batch_size = 1` và `batch_size = 32`.

### 1.4. Cấu Hình Chuẩn Hóa (YAML Spec)

```yaml
inference:
  checkpoint_path: experiments/lab1/cnn_aug/best.pt
  device: auto
  top_k: 3
  benchmark:
    warmup_runs: 20
    measure_runs: 200
    batch_sizes: [1, 32]
```

### 1.5. Ánh Xạ Thư Mục
- **Mã nguồn dự kiến (`SRC.md`)**: `src/vibe_scripts/lab1/infer.py` (tách biệt với [`labs/lab1/vibe/inference.py`](../../inference.py) cũ vốn hardcode lớp `FashionCNN` cũ không khớp `REPORT.md`).
- **Kiểm thử dự kiến**: `test/test_lab1_infer.py`.

---

## 2. Prompt Độc Lập Cho Pipeline 05 (Behavioral Schema)

```markdown
Name:
Lab 1 Inference & Serving Pipeline

Description:
Xây dựng module và CLI suy luận gọn nhẹ cho 3 kiến trúc Lab 1 (`logistic`, `mlp`, `cnn`) từ checkpoint `best.pt`, hỗ trợ dự đoán 1 ảnh hoặc batch dưới `@torch.inference_mode()`, xuất Top-3 xác suất Softmax và đo độ trễ suy luận.

Purpose:
Kiểm chứng khả năng lưu/nạp và phục vụ thực tế của mô hình sau huấn luyện, đồng thời đo chính xác tốc độ suy luận và dung lượng mô hình để đối sánh giữa Logistic, MLP và CNN.

Input:
- File checkpoint `best.pt` chứa `model_name`, `model_state_dict` và `norm_stats`
- Ảnh đầu vào: `--sample-idx <int>` trên tập Test, hoặc `--image-path <path>`, hoặc chế độ `--benchmark`

Output:
- Module & CLI script `src/vibe_scripts/lab1/infer.py`
- Kết quả dự đoán có cấu trúc (Top-1, Top-3 Softmax, latency ms, checkpoint size KB) in qua Rich Table
- Bộ unit test `test/test_lab1_infer.py`

How to do:
1. Viết hàm `load_predictor(checkpoint_path, device)` nạp `best.pt`, khởi tạo đúng kiến trúc từ `src/vibe_scripts/lab1/models.py`, chuyển sang `model.eval()` và đọc `norm_stats`.
2. Viết hàm tiền xử lý thống nhất `preprocess_image(image_or_path, norm_stats)` chuyển ảnh về tensor `[1, 1, 28, 28]` (`float32`).
3. Viết hàm `predict_single()` và `predict_batch()` bọc trong `@torch.inference_mode()`, tính `torch.softmax(logits, dim=1)` và trích Top-3 lớp có xác suất cao nhất.
4. Viết hàm `benchmark_latency(model, device, batch_sizes=(1, 32))` có warm-up và `torch.cuda.synchronize()` (khi chạy GPU), trả về `mean_ms`, `p95_ms`, `samples_per_sec` và kích thước file (KB).
5. Cung cấp giao diện CLI rõ ràng và viết Smoke Test + Normal Test tại `test/test_lab1_infer.py`.

Examples:
python -m src.vibe_scripts.lab1.infer --checkpoint experiments/lab1/cnn_aug/best.pt --sample-idx 18
python -m src.vibe_scripts.lab1.infer --checkpoint experiments/lab1/cnn_aug/best.pt --benchmark
pytest test/test_lab1_infer.py -v
```

---

## 3. Checklist Kiểm Thử Độc Lập (Definition of Done)

- [ ] **Độc lập Kiến trúc**: `load_predictor()` nạp thành công cả 3 loại checkpoint `logistic`, `mlp`, `cnn` mà không cần truyền tay tên mô hình.
- [ ] **Bất biến Xác suất**: Toàn bộ phân phối Softmax nằm trong `[0.0, 1.0]`, tổng xác suất 10 lớp bằng `1.0` (sai số `< 1e-5`), Top-3 sắp xếp giảm dần nghiêm ngặt.
- [ ] **Không tính Gradient**: Chạy dưới `@torch.inference_mode()`; tensor đầu ra có `requires_grad == False`.
- [ ] **Xử lý Ảnh đầu vào đa dạng**: Tiền xử lý đúng cho cả ảnh xám 1 kênh, ảnh màu RGB 3 kênh và ảnh kích thước khác `28×28` từ file ngoài.
- [ ] **Báo cáo Latency & Kích thước**: Đo được `mean_ms > 0`, `p95_ms >= mean_ms` (hoặc phân vị hợp lệ), `samples_per_sec > 0` và `checkpoint_size_kb > 0` cho cả `batch_size = 1` và `batch_size = 32`.
- [ ] **Smoke Test `< 5s`**: Toàn bộ test trên CPU với dummy checkpoint pass 100%.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo đặc tả độc lập Pipeline 05 (Inference & Serving) bám sát `REPORT.md`, `QUESTION.md` và `PIPELINE_SEPARATION.md` |
