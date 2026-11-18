---
title: Chỉ Mục 5 Pipeline Độc Lập — Lab 1 (Fashion-MNIST)
doc_type: readme
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
  - ../../../../../.agents/temp/06_PIPELINE_SEPARATION.md
  - ../../../../../.agents/rules/WRITE_README.md
  - ../REPORT.md
tags: [lab1, pipelines, readme, index]
---

# Thư Mục Đặc Tả 5 Pipeline Độc Lập — Lab 1

Thư mục này lưu trữ đặc tả tách rời cho từng pipeline của bài thực hành Fashion-MNIST, tuân thủ nguyên tắc **không gộp toàn bộ logic vào một prompt hoặc một file duy nhất** theo [`.agents/rules/PIPELINE_SEPARATION.md`](../../../../../.agents/temp/06_PIPELINE_SEPARATION.md).

---

## Danh Sách Pipeline

| File | Pipeline | Phạm vi trách nhiệm chính | Phụ thuộc đầu vào |
| :--- | :--- | :--- | :--- |
| [`01_DATA_PREPARATION.md`](./01_DATA_PREPARATION.md) | **Data Preparation** | Xác thực `manifest.json`, chia tầng 55k/5k/10k, tính `mean`/`std` trên Train, tạo 2 nhánh `no_aug`/`aug`, dựng `DataLoader` | `data/FashionMNIST/raw/` |
| [`02_MODEL_ARCHITECTURE.md`](./02_MODEL_ARCHITECTURE.md) | **Model Architecture** | Định nghĩa độc lập 3 lớp mô hình `Logistic` (7.850), `MLP` (109.386), `CNN` (105.866) trả raw logits `[B, 10]` | Không phụ thuộc dữ liệu |
| [`03_TRAINING.md`](./03_TRAINING.md) | **Training Loop** | Điều phối vòng lặp 5 bước với `CrossEntropyLoss` và `Adam`, sanity check Baseline, đánh giá Val mỗi epoch, lưu `best.pt`/`last.pt` | Pipeline 01 + Pipeline 02 |
| [`04_EVALUATION_BENCHMARK.md`](./04_EVALUATION_BENCHMARK.md) | **Evaluation & Benchmark** | Đánh giá 1 lần duy nhất trên Test 10k, lập bảng đối sánh 3×2, đường cong học, Confusion Matrix, phân tích lỗi và kiểm tra Save/Load | Pipeline 01 + Pipeline 02 + Pipeline 03 |
| [`05_INFERENCE_SERVING.md`](./05_INFERENCE_SERVING.md) | **Inference & Serving** | Nạp `best.pt`, suy luận đơn ảnh/batch dưới `@torch.inference_mode()`, xuất Top-3 Softmax, đo độ trễ và kích thước file | Pipeline 02 + Checkpoint từ Pipeline 03 |

---

## Quy Tắc Cổng Kiểm Soát Tích Hợp (Integration Gate)

1. Mỗi lần giao việc cho AI Agent chỉ thực thi **một pipeline duy nhất** bằng cách sử dụng trực tiếp khối **Prompt riêng (Behavioral Schema 7 trường)** nằm trong tài liệu của pipeline đó.
2. Không chuyển sang pipeline kế tiếp khi **Checklist độc lập** (Smoke Test `< 5s` + Normal Test) của pipeline hiện tại chưa đạt 100%.
3. Không sửa trực tiếp các file gốc của người dùng trong `src/`; mã nguồn mới do AI tạo đặt tại `src/vibe_scripts/lab1/`.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo README chỉ mục cho thư mục `docs/lab1/pipelines/` |
