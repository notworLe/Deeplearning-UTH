---
title: Kế Hoạch Đặc Tả & Phân Tách 5 Pipeline — Lab 1 (Fashion-MNIST)
doc_type: plan
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
  - ../../../.agents/temp/06_PIPELINE_SEPARATION.md
  - ../../../.agents/rules/PLANNING.md
  - ../../../.agents/rules/BEHAVIORAL_LAYER.md
  - ../../../labs/lab1/vibe/docs/REPORT.md
  - ../../../labs/lab1/vibe/docs/README.md
tags: [plan, lab1, fashion-mnist, pipeline-separation]
---

# Kế Hoạch Đặc Tả & Triển Khai Phân Tách Pipeline — Lab 1 (Fashion-MNIST)

Tài liệu kế hoạch theo quy định tại [`.agents/rules/PLANS.md`](../../../.agents/rules/PLANNING.md) và [`.agents/rules/PIPELINE_SEPARATION.md`](../../../.agents/temp/06_PIPELINE_SEPARATION.md), lấy [`labs/lab1/vibe/docs/REPORT.md`](../../../labs/lab1/vibe/docs/REPORT.md) làm kim chỉ nam.

---

## 1. Mục Tiêu & Phạm Vi

- **Mục tiêu**: Xây dựng bộ tài liệu đặc tả hoàn chỉnh 5 pipeline độc lập cho Lab 1 tại `docs/lab1/` để làm hợp đồng kỹ thuật trước khi triển khai mã nguồn.
- **In-Scope (Giai đoạn hiện tại)**:
  - Khởi tạo tài liệu tổng quan kiến trúc và ma trận hợp đồng I/O tại `docs/lab1/README.md`.
  - Đặc tả độc lập 5 pipeline tại `docs/lab1/pipelines/` (mỗi pipeline gồm đủ 3 yếu tố: Tài liệu đặc tả I/O & cấu hình, Prompt riêng theo Behavioral Schema 7 trường, và Checklist kiểm thử độc lập).
  - Nêu rõ các điểm mở kỹ thuật (Learning Rate, số Epoch, tham số Augmentation cụ thể, tiêu chí lưu `best.pt`) để kỹ sư phê duyệt trước khi viết mã nguồn.
- **Out-of-Scope (Giai đoạn hiện tại)**:
  - Chưa viết mã nguồn triển khai Data Processing, Model, Training, Evaluation hay Inference.
  - Không chỉnh sửa `labs/lab1/vibe/lab1.ipynb` (đã hoàn thành phạm vi Define → EDA).
  - Không sửa mã nguồn gốc của người dùng trong `src/`.

---

## 2. Phân Rã Giai Đoạn Thực Hiện (Phased Roadmap)

### Giai đoạn 0: Define → EDA (Đã hoàn thành)
- **Đầu ra**: [`labs/lab1/vibe/lab1.ipynb`](../../../labs/lab1/vibe/lab1.ipynb), [`data/FashionMNIST/raw/manifest.json`](../../../data/FashionMNIST/raw/manifest.json), [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py).
- **Trạng thái**: `Done`.

### Giai đoạn 1: Đặc Tả Phân Tách 5 Pipeline (Đang thực hiện)
- **Đầu ra**:
  - [`docs/lab1/README.md`](../../../labs/lab1/vibe/docs/README.md): Tổng quan bài toán, sơ đồ 5 pipeline, ma trận 3 mô hình × 2 cấu hình, bảng điều hướng.
  - [`docs/lab1/pipelines/README.md`](../../../labs/lab1/vibe/docs/pipelines/README.md): Chỉ mục 5 pipeline và quy tắc kiểm soát cổng tích hợp.
  - [`docs/lab1/pipelines/01_DATA_PREPARATION.md`](../../../labs/lab1/vibe/docs/pipelines/01_DATA_PREPARATION.md)
  - [`docs/lab1/pipelines/02_MODEL_ARCHITECTURE.md`](../../../labs/lab1/vibe/docs/pipelines/02_MODEL_ARCHITECTURE.md)
  - [`docs/lab1/pipelines/03_TRAINING.md`](../../../labs/lab1/vibe/docs/pipelines/03_TRAINING.md)
  - [`docs/lab1/pipelines/04_EVALUATION_BENCHMARK.md`](../../../labs/lab1/vibe/docs/pipelines/04_EVALUATION_BENCHMARK.md)
  - [`docs/lab1/pipelines/05_INFERENCE_SERVING.md`](../../../labs/lab1/vibe/docs/pipelines/05_INFERENCE_SERVING.md)
- **Trạng thái**: `In Progress` → Chờ kỹ sư duyệt (`agree / reject / adjust`).

### Giai đoạn 2–6: Triển Khai Độc Lập Từng Pipeline (Sau khi kỹ sư phê duyệt)
Mỗi pipeline chỉ được bắt đầu sau khi pipeline trước vượt qua 100% Checklist độc lập (Smoke Test + Normal Test):
- **Giai đoạn 2**: Implement & Test Pipeline 1 (`Data Preparation`) tại `src/vibe_scripts/lab1/data.py`.
- **Giai đoạn 3**: Implement & Test Pipeline 2 (`Model Architecture`: Logistic, MLP, CNN) tại `src/vibe_scripts/lab1/models.py`.
- **Giai đoạn 4**: Implement & Test Pipeline 3 (`Training Loop` + Baseline Logistic sanity run) tại `src/vibe_scripts/lab1/train.py`.
- **Giai đoạn 5**: Implement & Test Pipeline 4 (`Evaluation & Error Analysis` cho ma trận 3×2) tại `src/vibe_scripts/lab1/evaluate.py`.
- **Giai đoạn 6**: Implement & Test Pipeline 5 (`Inference & Save/Load Verification`) tại `src/vibe_scripts/lab1/infer.py`, sau đó tổng hợp số liệu thật vào `labs/lab1/vibe/docs/REPORT.md`.

---

## 3. Các Điểm Mở Cần Kỹ Sư Chốt Trước Khi Sang Giai Đoạn Code

| Hạng mục | Hiện trạng trong repo | Đề xuất cho Lab 1 | Cần kỹ sư chốt |
| :--- | :--- | :--- | :--- |
| **Learning Rate (Adam)** | `src/config.py` để `LR = 1e-5`; `REPORT.md` chưa khóa số cụ thể | Khảo sát lưới nhỏ `{1e-3, 5e-4, 1e-4}` trên Validation hoặc chốt `1e-3` dùng chung cho cả 3 mô hình | Chốt 1 giá trị `lr` dùng chung |
| **Số Epoch** | `src/config.py` để `EPOCH = 30`; `REPORT.md` chưa khóa số | Cố định `20` hoặc `30` epoch giống nhau cho cả 3 mô hình; ghi nhận `best_epoch` hội tụ | Chốt `epochs` và có bật Early Stopping hay không |
| **Tham số Augmentation** | `REPORT.md` nêu: flip ngang, crop/dịch nhẹ, xoay nhỏ | `RandomHorizontalFlip(p=0.5)` + `RandomAffine(degrees=10, translate=(0.08, 0.08))` hoặc `RandomCrop(28, padding=2)` | Chốt bộ phép biến đổi cụ thể |
| **Tiêu chí lưu `best.pt`** | `REPORT.md` nêu: `val loss/accuracy tốt nhất` | Chọn 1 tiêu chí duy nhất: `val_loss` thấp nhất (ưu tiên xác suất hiệu chuẩn tốt) hoặc `val_acc` cao nhất | Chốt 1 tiêu chí lưu checkpoint |

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo kế hoạch phân tách và đặc tả 5 pipeline cho Lab 1 bám sát `REPORT.md` |
