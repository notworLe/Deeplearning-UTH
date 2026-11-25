---
title: Kế Hoạch Triển Khai Toàn Diện — Lab 1 (Fashion-MNIST PyTorch)
doc_type: plan
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-18
updated: 2026-11-20
last_updated_by: Antigravity
version: 0.4.0
status: completed
reviewers: []
related_docs:
  - ../../../.agents/rules/PLANNING.md
  - ../../../.agents/rules/PROGRESS_TRACKING.md
  - ../../../.agents/rules/TESTING_STRATEGY.md
  - ../prompt.md
  - ../../../labs/lab1/vibe/docs/REPORT.md
  - ../../../labs/lab1/vibe/config.py
tags: [plan, lab1, fashion-mnist, pytorch]
---

# Kế Hoạch Triển Khai: Lab 1 — So Sánh Logistic, MLP, CNN Trên Fashion-MNIST

Tài liệu kế hoạch thực thi theo [`PLANNING.md`](../../../.agents/rules/PLANNING.md) và [`PROGRESS_TRACKING.md`](../../../.agents/rules/PROGRESS_TRACKING.md), triển khai yêu cầu tại [`prompts/0001_lab1/prompt.md`](../prompt.md).

---

## 1. Mục Tiêu & Phạm Vi (Plan)

- **Goal**:
  1. Làm mới [`labs/lab1/vibe/docs/REPORT.md`](../../../labs/lab1/vibe/docs/REPORT.md) và đồng bộ đặc tả 5 pipeline tại [`labs/lab1/vibe/docs/pipelines/`](../../../labs/lab1/vibe/docs/pipelines/README.md) (trình bày rõ từng giai đoạn, vấn đề kỹ thuật, chừa placeholder `[TBD]` cho số liệu thực nghiệm).
  2. Hoàn thiện override cấu hình tại [`labs/lab1/vibe/config.py`](../../../labs/lab1/vibe/config.py) (kế thừa [`src/config.py`](../../../src/config.py)).
  3. Code & kiểm thử độc lập từng pipeline (`Data → Model → Train → Evaluate → Infer`), chạy hoàn chỉnh qua notebook, rồi **điền số liệu thực nghiệm thật** vào `REPORT.md`.
- **In-Scope**:
  - Cấu hình riêng Lab 1 tại `labs/lab1/vibe/config.py`.
  - Mã nguồn sạch 5 pipeline tại `src/vibe_scripts/lab1/` (`data.py`, `models.py`, `train.py`, `evaluate.py`, `infer.py`).
  - Unit test 2 chế độ (Smoke Test `< 5s` + Normal Test) tại `test/test_lab1_*.py`.
  - Chạy thực nghiệm ma trận **3 mô hình × 2 cấu hình** (`Logistic`, `MLP`, `CNN` × `no_aug`, `aug`) qua notebook trong `labs/lab1/vibe/`.
  - Cập nhật số liệu thật, biểu đồ, phân tích lỗi và kết luận vào `labs/lab1/vibe/docs/REPORT.md`.
- **Out-of-Scope**:
  - Không sửa trực tiếp các file thủ công trong `src/` (`model.py`, `datasets.py`, `visualize.py`, `utility.py`, `objects.py`, `config.py`).
  - Không thêm Docker/Web API serving ngoài phạm vi Lab 1.

---

## 2. Cấu Hình Đề Xuất Override Tại `labs/lab1/vibe/config.py`

Kế thừa toàn bộ hằng số nền từ [`src/config.py`](../../../src/config.py) (`ROOT_PATH`, `DATA_PATH`, `DEVICE`, `SEED = 248`, `BATCH_SIZE = 32`, `WOKER = 8`) và **override** các tham số thực nghiệm Lab 1 để kỹ sư duyệt trước khi code:

| Tham số          | Giá trị gốc (`src/config.py`) |                  Đề xuất override (`lab1/vibe/config.py`)                  | Lý do kỹ thuật                                                                      |
| :--------------- | :---------------------------: | :------------------------------------------------------------------------: | :---------------------------------------------------------------------------------- |
| `LR`             |            `1e-5`             |                                   `1e-3`                                   | `1e-5` quá nhỏ để Adam hội tụ từ đầu trong 20–30 epoch trên Fashion-MNIST           |
| `EPOCH`          |             `30`              |                                    `20`                                    | Đủ hội tụ cho cả 3 kiến trúc 7.8k–109k tham số với chi phí chạy vừa phải            |
| `BEST_METRIC`    |          _(chưa có)_          |                                `"val_loss"`                                | Ưu tiên mô hình tổng quát tốt, tránh chọn checkpoint overfit nhẹ                    |
| `AUG_CONFIG`     |          _(chưa có)_          | `HorizontalFlip(0.5)` + `RandomAffine(degrees=10, translate=(0.08, 0.08))` | Khớp mô tả `REPORT.md` (_flip ngang, dịch nhẹ, xoay nhỏ_), giữ đúng hướng trọng lực |
| `EXPERIMENT_DIR` |          _(chưa có)_          |                    `ROOT_PATH / "experiments" / "lab1"`                    | Tách artifact (`best.pt`, `last.pt`, `history.json`, biểu đồ) theo từng thí nghiệm  |
| `MEAN`, `STD`    |      `(0.5,)`, `(0.5,)`       |                      Tính động trên 55.000 mẫu Train                       | Tránh hardcode và loại bỏ data leakage từ Validation/Test                           |

---

## 3. Bảng Theo Dõi Tiến Độ (Kanban Board)

| Task ID  | Công việc                                                                         | Phụ trách | Trạng thái | Phụ thuộc / Ghi chú                                                                    |
| :------: | :-------------------------------------------------------------------------------- | :-------- | :--------: | :------------------------------------------------------------------------------------- |
| **T-01** | Làm mới `REPORT.md` (đủ khung pipeline, chừa `[TBD]`) & đồng bộ `docs/pipelines/` | AI Agent  |   `Done`   | Đã hoàn tất bộ `REPORT.md`, `README.md` và 5 file `pipelines/*.md`                     |
| **T-02** | Triển khai `labs/lab1/vibe/config.py` (kế thừa & override `src/config.py`)        | AI Agent  |   `Done`   | Config OK (`LR=1e-3`, `EPOCH=20`, `BEST_METRIC="val_loss"`), đã đăng ký `smoke` marker |
| **T-03** | Pipeline 01: `src/vibe_scripts/lab1/data.py` + `test/test_lab1_data.py`           | AI Agent  |   `Done`   | Smoke `1 passed`, Normal `6 passed`                                                    |
| **T-04** | Pipeline 02: `src/vibe_scripts/lab1/models.py` + `test/test_lab1_models.py`       | AI Agent  |   `Done`   | Smoke `3 passed`, Normal `13 passed` (7.850 / 109.386 / 105.866)                       |
| **T-05** | Pipeline 03: `src/vibe_scripts/lab1/train.py` + `test/test_lab1_train.py`         | AI Agent  |   `Done`   | Smoke `1 passed`, Normal `5 passed`                                                    |
| **T-06** | Pipeline 04: `src/vibe_scripts/lab1/evaluate.py` + `test/test_lab1_evaluate.py`   | AI Agent  |   `Done`   | Smoke `1 passed`, Normal `4 passed`                                                    |
| **T-07** | Pipeline 05: `src/vibe_scripts/lab1/infer.py` + `test/test_lab1_infer.py`         | AI Agent  |   `Done`   | Smoke `1 passed`, Normal `4 passed`                                                    |
| **T-08** | Thực nghiệm qua Notebook & điền số liệu thật vào `REPORT.md`                      | AI Agent  |   `Done`   | Hoàn tất 6 chạy thực nghiệm, `REPORT.md` v0.13.0 (19 ảnh EDA/Eval/Error, H1–H3)       |
| **T-09** | Audit toàn bộ, chạy full `pytest`, ghi Thinking Log & cập nhật `INDEX.md`         | AI Agent  |   `Done`   | Smoke `9 passed`, Full `55 passed`, Audit 5 Gates sạch, đã ghi Thinking Log & registry |

---

## 4. Detailed Execution Plan (Theo Từng Task)

### T-01: Làm mới `REPORT.md` & Đồng bộ `docs/pipelines/`

- Giữ nguyên khung chuẩn của `labs/lab1/vibe/docs/REPORT.md`: `Abstract → Define → Dataset → Model selection → Data processing → Training → Baseline → Evaluate → Error analysis`.
- Viết rõ luồng, thiết kế và vấn đề kỹ thuật từng mục (dùng lại thông số EDA thật đã chạy ở `lab1.ipynb`: `mean = 0.2860`, `std = 0.3530`, `sparsity = 50.21%`, `skewness = 0.695`).
- Toàn bộ bảng kết quả Training / Evaluation / Error Analysis để rõ `[TBD]` kèm ghi chú sẽ điền sau bước chạy notebook.
- Cập nhật `docs/pipelines/*.md` trỏ cấu hình về `labs/lab1/vibe/config.py`.

### T-02: Cấu hình `labs/lab1/vibe/config.py` & Marker `pytest.ini`

- Import `src.config` làm nền; định nghĩa các hằng số override đã chốt ở §2; không hardcode đường dẫn tuyệt đối.
- Bổ sung `markers = smoke: ...` vào `pytest.ini` để chạy tách riêng 2 chế độ kiểm thử không cảnh báo.

### T-03: Pipeline 01 — Data Preparation (`data.py`)

- Step 1: Xác thực `data/FashionMNIST/raw/manifest.json` (SHA-256) qua `eda_helpers.load_and_verify_manifest()`.
- Step 2: Chia tầng `55.000` Train / `5.000` Val từ 60k gốc bằng `seed = 248`; giữ nguyên `10.000` Test.
- Step 3: Tính `mean`, `std` vector hóa chỉ trên 55.000 mẫu Train.
- Step 4: Dựng 2 nhánh transform tách rời (`no_aug` vs `aug`, đúng thứ tự `Augmentation → ToTensor → Normalize`), khởi tạo 2 `FashionMNIST` riêng cho Train và Val để chống leakage.
- Step 5 (**Dual Testing** — `test/test_lab1_data.py`):
  - **Smoke Test (`@pytest.mark.smoke`, `< 5s`)**: Import module, kiểm tra `build_dataloaders()` khởi tạo không crash và đọc thử 1 batch `(B, 1, 28, 28)` (`float32`).
  - **Normal Test**: Không giao nhau Train/Val (`55k/5k/10k`), cân bằng phân tầng 10 lớp, `mean/std` chuẩn hóa trên Train `~0.0/1.0`, Val/Test không dính Augmentation, tái lập thứ tự batch với `seed = 248`, edge case lỗi `manifest.json`.

### T-04: Pipeline 02 — Model Architecture (`models.py`)

- Step 1: Cài đặt `LogisticClassifier` (`7.850`), `MLPClassifier` (`109.386`), `CNNClassifier` (`105.866`) trả raw logits `[B, 10]`.
- Step 2: Fail-Fast kiểm tra shape `(B, 1, 28, 28)` trong `forward`, không chứa loss/optimizer/softmax bên trong model.
- Step 3: Viết `build_model()`, `count_parameters()`.
- Step 4 (**Dual Testing** — `test/test_lab1_models.py`):
  - **Smoke Test (`@pytest.mark.smoke`, `< 5s`)**: Khởi tạo cả 3 mô hình qua `build_model()`, chạy dummy forward `torch.randn(2, 1, 28, 28)` không crash.
  - **Normal Test**: Khớp chính xác số tham số (`7.850` / `109.386` / `105.866`), output là raw logits (không bị ép Softmax sum=1), backward nhận đủ gradient hữu hạn, tái lập trọng số với `seed = 248`, edge case shape sai `(4, 3, 28, 28)` ném `ValueError` và batch `B = 1`.

### T-05: Pipeline 03 — Training Loop (`train.py`)

- Step 1: `set_global_seed(248)` khóa `random`, `numpy`, `torch`, `cuda`, `cudnn`.
- Step 2: Sanity check trên `LogisticClassifier` (initial loss $\approx \ln 10$, overfit 1 batch).
- Step 3: Vòng lặp 5 bước (`CrossEntropyLoss`, `Adam`) + Validation mỗi epoch (`model.eval()`, `torch.no_grad()`), tích lũy loss bằng `.detach().item() * batch_size`, bắt `NaN/Inf`.
- Step 4: Lưu nguyên tử `best.pt`, `last.pt` (kèm `norm_stats`, `model_name`, `hyperparams`) và `history.json`.
- Step 5 (**Dual Testing** — `test/test_lab1_train.py`):
  - **Smoke Test (`@pytest.mark.smoke`, `< 5s`)**: Chạy 1 epoch trên dummy/subset nhỏ (64 mẫu), xác nhận sinh đủ `best.pt`, `last.pt`, `history.json` và không để lại `.tmp`.
  - **Normal Test**: Kiểm tra initial loss `Logistic` trong `[2.1, 2.5]`, overfit 1 batch (`acc > 95%`), schema checkpoint đủ `norm_stats`/`model_name`, Early Stopping dừng đúng `patience`, bất biến không rò rỉ VRAM/NaN.

### T-06: Pipeline 04 — Evaluation & Benchmark (`evaluate.py`)

- Step 1: Nạp `best.pt` từ 6 thí nghiệm, chạy Test 10.000 mẫu **1 lần duy nhất** dưới `model.eval()` + `torch.no_grad()`.
- Step 2: Tính Test Loss, Accuracy, Precision/Recall/F1 từng lớp & Macro, Confusion Matrix 10×10, kích thước file (KB), độ trễ (warm-up + CUDA sync).
- Step 3: Kiểm chứng Save/Load (sai số logits `< 1e-6`).
- Step 4: Xuất `benchmark_summary.json`, `benchmark_table.csv` và các đồ thị (`learning_curves`, `confusion_matrix`, `per_class_f1`, `predictions_gallery` đã khử `Normalize`) vào `experiments/lab1/`.
- Step 5 (**Dual Testing** — `test/test_lab1_evaluate.py`):
  - **Smoke Test (`@pytest.mark.smoke`, `< 5s`)**: Chạy `evaluate_checkpoint()` với dummy checkpoint trên 32 mẫu không crash.
  - **Normal Test**: Tổng phần tử Confusion Matrix khớp tổng mẫu Test, `Accuracy == diag(CM).sum() / CM.sum()`, mọi chỉ số nằm trong `[0.0, 1.0]`, Save/Load sai số logits `< 1e-6`, sinh đủ file `.json`/`.csv`/`.png`.

### T-07: Pipeline 05 — Inference & Serving (`infer.py`)

- Step 1: `load_predictor(best.pt)` tự khôi phục kiến trúc và `norm_stats`.
- Step 2: Suy luận ảnh đơn / file ảnh ngoài / batch dưới `@torch.inference_mode()`, xuất Top-3 Softmax và đo latency (`batch_size = 1` & `32`).
- Step 3 (**Dual Testing** — `test/test_lab1_infer.py`):
  - **Smoke Test (`@pytest.mark.smoke`, `< 5s`)**: Nạp dummy checkpoint, dự đoán 1 tensor `(1, 28, 28)` trả về Top-3 hợp lệ không crash.
  - **Normal Test**: Tổng 10 xác suất Softmax bằng `1.0 ± 1e-5`, Top-3 giảm dần nghiêm ngặt, `requires_grad == False`, xử lý được ảnh RGB/kích thước khác `(28, 28)`, benchmark trả `mean_ms > 0` và `checkpoint_size_kb > 0`.

### T-08 & T-09: Thực Nghiệm Notebook, Điền Số Liệu `REPORT.md` & Nghiệm Thu

- Xây dựng/chạy notebook thực nghiệm cho phần Processing → Train → Evaluate → Infer (giữ nguyên `lab1.ipynb` hoặc tách notebook tiếp nối theo quy chuẩn Cell 0 + Cell 1 Roadmap, không chứa sơ đồ/references học thuật).
- Thu số liệu thật của cả 6 cấu hình, điền đầy đủ vào `REPORT.md`, đối chiếu 3 giả thuyết `H1–H3`.
- Chạy kiểm thử 2 chế độ theo [`TESTING_STRATEGY.md`](../../../.agents/rules/TESTING_STRATEGY.md):
  - **Smoke Test toàn cục**: `pytest -m smoke -v` (`< 5s`).
  - **Normal Test toàn cục**: `pytest test/ -v` (100% pass).
- Kiểm tra toàn bộ link tương đối và ghi log vào `logs/thinking/`.

---

## 5. Definition of Done (DoD)

- [x] `labs/lab1/vibe/config.py` override sạch từ `src/config.py`, không hardcode đường dẫn tuyệt đối.
- [x] Tất cả 5 pipeline có đủ Smoke Test (`< 5s`) và Normal Test pass 100% qua `pytest`.
- [x] Số tham số khớp chính xác `7.850` / `109.386` / `105.866`; cả 6 thí nghiệm chạy công bằng với `seed = 248`.
- [x] `labs/lab1/vibe/docs/REPORT.md` có đủ số liệu thực nghiệm thật, phân tích theo lớp, phân tích lỗi và kết luận đối sánh 3 mô hình.
- [x] Tuân thủ 5 Quality Gates trong `WORKFLOW_OVERVIEW.md` và cập nhật `logs/thinking/INDEX.md`.

---

## Lịch sử thay đổi

| Version | Ngày       | Người sửa | Agent       | Mô tả thay đổi                                                                                      |
| :------ | :--------- | :-------- | :---------- | :-------------------------------------------------------------------------------------------------- |
| 0.4.0   | 2026-11-20 | notworLe  | Antigravity | Hoàn tất T-08 & T-09, nghiệm thu 5/5 DoD, cập nhật Thinking Log, vibe_scripts registry và audit sạch |
| 0.3.0   | 2026-11-18 | notworLe  | Antigravity | Cập nhật tiến độ Kanban sau khi hoàn thành T-01 → T-07 và bắt đầu T-08                              |
| 0.2.0   | 2026-11-18 | notworLe  | Antigravity | Bổ sung kiểm thử 2 chế độ (Smoke Test `< 5s` + Normal Test) cho T-02 → T-09                         |
| 0.1.0   | 2026-11-18 | notworLe  | Antigravity | Khởi tạo Plan chi tiết thực hiện Lab 1 (T-01 đến T-09) chờ kỹ sư duyệt                              |
