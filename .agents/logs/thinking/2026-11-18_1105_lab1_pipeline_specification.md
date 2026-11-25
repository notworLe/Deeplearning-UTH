---
title: "Đặc Tả Phân Tách 5 Pipeline Độc Lập — Lab 1 (Fashion-MNIST)"
doc_type: thinking-log
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-18
updated: 2026-11-18
last_updated_by: Antigravity
version: 0.1.0
status: completed
reviewers: []
related_docs:
  - ../../docs/lab1/README.md
  - ../../docs/plans/lab1_pipeline_spec/README.md
  - ../../labs/lab1/vibe/docs/REPORT.md
  - ../../.agents/rules/PIPELINE_SEPARATION.md
tags: [thinking-log, lab1, pipeline-separation, specification]
---

# Đặc Tả Phân Tách 5 Pipeline Độc Lập — Lab 1 (Fashion-MNIST)

- **Date/Time:** 2026-11-18 11:05
- **Agent/Role:** Principal CV/MLOps Architect & Pair Programmer
- **Status:** Completed
- **Related Files:**
  - `docs/lab1/README.md`
  - `docs/lab1/pipelines/README.md`
  - `docs/lab1/pipelines/01_DATA_PREPARATION.md`
  - `docs/lab1/pipelines/02_MODEL_ARCHITECTURE.md`
  - `docs/lab1/pipelines/03_TRAINING.md`
  - `docs/lab1/pipelines/04_EVALUATION_BENCHMARK.md`
  - `docs/lab1/pipelines/05_INFERENCE_SERVING.md`
  - `docs/plans/lab1_pipeline_spec/README.md`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)

- **Yêu cầu ban đầu:** Người dùng nhận thấy `docs/lab1/` đang trống, xác nhận vai trò của `.agents/rules/PIPELINE_SEPARATION.md`, và yêu cầu viết bản đặc tả pipeline cho Lab 1 lấy `labs/lab1/vibe/docs/REPORT.md` làm kim chỉ nam.
- **Vấn đề cốt lõi cần giải quyết:** Trước khi bước vào giai đoạn triển khai Data Processing, Model, Training và Evaluation, dự án cần bộ hợp đồng kỹ thuật tách rời rõ 5 pipeline độc lập, khớp chính xác 3 kiến trúc trong `REPORT.md` (tránh nhầm lẫn với kiến trúc cũ trong `src/model.py` và `labs/lab1/vibe/train.py`).

---

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)

- **Các phương án đã cân nhắc:**
  - *Phương án A:* Viết một file dài duy nhất `docs/lab1/PIPELINE_EXPLANATION.md` theo mẫu cũ của `docs/lab2/` và `docs/lab3/`.
    - *Nhược điểm:* Trái tinh thần mới của `.agents/rules/PIPELINE_SEPARATION.md` vốn yêu cầu mỗi pipeline phải có tài liệu riêng, prompt riêng và checklist độc lập; đồng thời người dùng đã chủ động xóa bộ `PIPELINE_EXPLANATION.md` cũ trong `labs/lab1/vibe/docs/`.
  - *Phương án B:* Tạo `docs/lab1/README.md` làm bản đồ kiến trúc tổng thể + thư mục `docs/lab1/pipelines/` chứa 5 file đặc tả độc lập (`01_DATA_PREPARATION.md` đến `05_INFERENCE_SERVING.md`), mỗi file có đủ 3 yếu tố bắt buộc (Đặc tả I/O & YAML Config + Prompt Behavioral Schema 7 trường + Checklist DoD độc lập).
    - *Ưu điểm:* Tuân thủ 100% `PIPELINE_SEPARATION.md`, `BEHAVIORAL_LAYER.md`, `METADATA_DOC.md`, `WRITE_README.md` và sẵn sàng cho người kỹ sư giao từng pipeline một cho AI triển khai theo chu trình vi mô.
- **Quyết định cuối cùng:** Chọn **Phương án B**, đồng thời lập kế hoạch tại `docs/plans/lab1_pipeline_spec/README.md` (tuân thủ vị trí `docs/plans/` mà người dùng đã chuyển từ `logs/plans/` và không tự tạo thư mục gốc lạ theo `AI_REFERENCE.md` §2).
- **Giả định & Kiểm chứng toán học:**
  - Đã đối chiếu lại phép tính số tham số của 3 kiến trúc trong `REPORT.md`: Logistic = `7.850`, MLP = `109.386`, CNN = `105.866` — khớp chính xác 100%.
  - Không bịa số liệu Test Accuracy/Loss chưa chạy (dùng `[TBD]`).
  - Xác định 4 điểm mở kỹ thuật cần kỹ sư chốt trước khi sang giai đoạn code: `lr` dùng chung cho Adam (vì `src/config.py` để `1e-5` quá nhỏ), `epochs`, tham số Augmentation cụ thể (`degrees`, `translate`, `padding`), và tiêu chí duy nhất lưu `best.pt` (`val_loss` vs `val_acc`).

---

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)

- Tạo bản đồ tổng thể tại `docs/lab1/README.md` gồm sơ đồ Mermaid 5 pipeline, bảng hợp đồng I/O, ma trận 6 thí nghiệm (3 mô hình × 2 cấu hình), giả thuyết H1–H3 và mục References 4 nhóm chuẩn.
- Tạo 5 tài liệu đặc tả độc lập tại `docs/lab1/pipelines/01_DATA_PREPARATION.md` → `05_INFERENCE_SERVING.md` cùng `docs/lab1/pipelines/README.md`.
- Khóa bất biến chống rò rỉ dữ liệu ở Pipeline 01: tách 2 đối tượng `FashionMNIST` độc lập cho Train/Val trước khi gắn transform, tính `mean`/`std` duy nhất trên 55.000 mẫu Train.
- Đính kèm `norm_stats` và `model_name` ngay bên trong schema checkpoint của Pipeline 03 để tránh lệch tiền xử lý ở Pipeline 04 và 05.
- Toàn bộ tài liệu mới đều gắn YAML frontmatter và bảng Lịch sử thay đổi theo `METADATA_DOC.md`, tên file viết hoa 100%, đường dẫn tương đối 100%.

---

## 4. Khó khăn & Cách khắc phục (Challenges & Solutions)

- **Xung đột giữa kiến trúc cũ trong repo và `REPORT.md` mới:** `src/model.py` (`FashionCNN` 32→64→128 + BatchNorm + Dropout) và `labs/lab1/vibe/train.py` (`BaselineMLP` 256→128) khác hoàn toàn `REPORT.md`. Đã ghi cảnh báo tường minh trong Pipeline 02 và chỉ định thư mục sạch `src/vibe_scripts/lab1/` theo `SRC.md`.
- **Thiếu `tensorboard` trong môi trường (`logs/errors/0001.md`):** Đặc tả Pipeline 03 dùng `history.json` + Rich làm logger mặc định để không phụ thuộc cứng vào `tensorboard`.

---

## 5. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)

- **Trạng thái hiện tại:** Bộ đặc tả 5 pipeline đang ở trạng thái `draft`, chờ kỹ sư duyệt và chốt 4 tham số mở (`lr`, `epochs`, tham số augmentation, tiêu chí `best.pt`).
- **Cảnh báo / Lưu ý:** Không sửa `labs/lab1/vibe/lab1.ipynb` (đang khóa phạm vi Define → EDA) và không sửa trực tiếp `src/`. Khi bắt đầu code, chỉ triển khai **từng pipeline một** theo đúng thứ tự 01 → 05 và phải pass 100% Checklist độc lập trước khi sang bước sau.
- **Đề xuất bước tiếp theo:** Sau khi kỹ sư chốt 4 điểm mở, cập nhật status tài liệu sang `approved` và thực thi Pipeline 01 (`src/vibe_scripts/lab1/data.py` + `test/test_lab1_data.py`).

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Ghi log luồng tư duy hoàn tất đặc tả 5 pipeline độc lập cho Lab 1 |
