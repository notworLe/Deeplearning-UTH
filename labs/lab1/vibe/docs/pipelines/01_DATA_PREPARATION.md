---
title: "Pipe 01: Data Pipeline — Phân Tích Đặc Trưng (Lab 1)"
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
  - ../../lab1.ipynb
  - ../../../../../.agents/rules/AI_FIRST_LIFECYCLE.md
  - ../../../../../src/vibe_scripts/lab1/data.py
tags: [lab1, pipe-01, data-pipeline, feature-analysis]
---

# Pipe 01: Data Pipeline (Thu Nạp, Chia Tầng, Chuẩn Hóa & DataLoader)

Ánh xạ **Giai đoạn 3 (Data Pipeline)** trong [`AI_FIRST_LIFECYCLE.md`](../../../../../.agents/rules/AI_FIRST_LIFECYCLE.md), triển khai tại [`src/vibe_scripts/lab1/data.py`](../../../../../src/vibe_scripts/lab1/data.py).

---

## 1. Hợp Đồng Giao Diện (Input / Output)

- **Input**: `data/FashionMNIST/raw/` + `manifest.json`, cờ `aug_enabled: bool`, tham số từ [`config.py`](../../config.py) (`BATCH_SIZE = 32`, `SEED = 248`, `WOKER = 8`).
- **Output**: `(train_loader, val_loader, test_loader, norm_stats)`
  - Tensor ảnh `images`: `[B, 1, 28, 28]` (`float32`), đã chuẩn hóa Z-Score.
  - Tensor nhãn `labels`: `[B]` (`int64`, giá trị `{0..9}`).
  - `norm_stats`: `{"mean": float, "std": float, "train_size": 55000, "val_size": 5000, "test_size": 10000, "seed": 248}`.

---

## 2. Phân Tích Đặc Trưng & Cơ Chế Kỹ Thuật

1. **Tính toàn vẹn nguồn gốc (Data Provenance)**: Kiểm tra SHA-256 trên 8 tệp nhị phân qua `manifest.json` trước khi tạo tensor; phát hiện ngay sai lệch dữ liệu ngầm giữa các môi trường.
2. **Phân tầng bảo toàn phân bố (Stratified Split)**: Chia 60.000 mẫu Train gốc thành `55.000` Train và `5.000` Validation theo từng lớp (`5.500` và `500` ảnh/lớp), giữ đúng tỷ lệ đồng đều tuyệt đối trên 10 lớp đã xác nhận ở EDA.
3. **Chuẩn hóa Z-Score không rò rỉ dữ liệu**:
   $$\tilde{x} = \frac{x / 255 - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   Từ kết quả EDA (`lab1.ipynb`), pixel nền bằng `0` chiếm `50.21%` và phân bố lệch phải (`skewness = 0.695`). Chuẩn hóa chỉ dùng `mean` và `std` tính trên `55.000` ảnh Train giúp đưa tâm phân bố về `0`, tránh thiên lệch gradient cho cả 3 mô hình.
4. **Đặc trưng Augmentation theo hình thái Fashion-MNIST**:
   - Thứ tự biến đổi: `Augmentation → ToTensor → Normalize`.
   - Chỉ áp dụng **lật ngang** (`RandomHorizontalFlip(p=0.5)`) và **dịch/xoay nhẹ** (`RandomAffine(degrees=10, translate=(0.08, 0.08))`) vì vật thể thời trang được căn giữa và có trục đối xứng trái–phải nhưng mang hướng đứng cố định theo trọng lực.

---

## 3. Mục Tiêu

- Cung cấp luồng batch `[B, 1, 28, 28]` chuẩn hóa đồng nhất, thứ tự ngẫu nhiên tái lập theo `seed = 248` để so sánh công bằng `Logistic`, `MLP` và `CNN`.

---

## 4. Hạn Chế & Rủi Ro Đặc Thù

1. **Rò rỉ Augmentation sang Validation**: Nếu dùng `random_split` trên một `FashionMNIST` đã gắn `train_transform`, tập Validation sẽ vô tình bị biến đổi ngẫu nhiên theo.
2. **Lạm dụng biến đổi làm méo ngữ nghĩa**: Lật dọc hoặc xoay góc lớn khiến áo/giày lộn ngược; tăng biên độ dịch chuyển có thể làm mất chi tiết cổ áo, gót giày ở rìa khung $28 \times 28$.
3. **Nghẽn multiprocessing trên Windows**: `num_workers = 8` khởi tạo nhiều tiến trình `spawn` có thể làm chậm notebook khi dataset rất nhỏ (~47 MB).

---

## 5. Giải Pháp Kỹ Thuật

- Khởi tạo **2 bản thể `FashionMNIST(train=True)` tách rời**: một gắn `train_transform` cho `Subset(train_idx)` và một gắn `eval_transform` cho `Subset(val_idx)`.
- Truyền `worker_init_fn` và `torch.Generator().manual_seed(248)` cho `train_loader`.
- Hỗ trợ chỉnh `num_workers = 0` khi chạy trong notebook để loại chi phí spawn process mà vẫn giữ nguyên toán học chuẩn hóa và thứ tự batch.

---

## 6. Tiêu Chí Kiểm Chứng (`test/test_lab1_data.py`)

- **Smoke Test**: `build_dataloaders()` trả đúng `(32, 1, 28, 28)` `float32`, không `NaN/Inf`.
- **Normal Test**: `train_idx ∩ val_idx = ∅`, cân bằng `5.500 / 500 / 1.000` mỗi lớp, Train `no_aug` sau chuẩn hóa có `mean ~ 0.0` và `std ~ 1.0`, Val/Test bất biến khi bật `aug`, Fail-Fast khi sai SHA-256.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.3.0 | 2026-11-19 | notworLe | Antigravity | Viết lại theo cấu trúc phân tích đặc trưng `AI_FIRST_LIFECYCLE.md`, loại bỏ khung prompt mẫu |
| 0.2.0 | 2026-11-18 | notworLe | Antigravity | Đồng bộ `labs/lab1/vibe/config.py` |
