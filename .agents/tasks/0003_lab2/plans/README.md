---
title: Kế Hoạch Triển Khai Toàn Diện — Lab 2 (Oxford-IIIT Pets: Baseline / Transfer / Fine-tune)
doc_type: plan
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-21
updated: 2026-11-21
last_updated_by: Antigravity
version: 0.1.0
status: approved
reviewers: []
related_docs:
  - ../../../.agents/rules/PLANNING.md
  - ../../../.agents/rules/PROGRESS_TRACKING.md
  - ../../../.agents/rules/TESTING_STRATEGY.md
  - ../PROMPT.md
  - ../../../labs/lab2/vibe/docs/REPORT.md
  - ../../../labs/lab2/vibe/config.py
tags: [plan, lab2, oxford-iiit-pet, resnet18, transfer-learning, fine-tuning]
---

# Kế Hoạch Triển Khai: Lab 2 — Đánh Giá Fine-tune Trên Oxford-IIIT Pets (ResNet-18)

Tài liệu kế hoạch thực thi theo [`PLANNING.md`](../../../.agents/rules/PLANNING.md) và [`PROGRESS_TRACKING.md`](../../../.agents/rules/PROGRESS_TRACKING.md), triển khai yêu cầu tại [`prompts/0003_lab2/PROMPT.md`](../PROMPT.md).

---

## 1. Mục Tiêu & Phạm Vi (Plan)

- **Goal**:
  1. So sánh công bằng 3 run ResNet-18 trên **Oxford-IIIT Pets** (37 giống: 25 chó, 12 mèo): `resnet18_baseline` (from scratch) → `resnet18_transfer` (`IMAGENET1K_V1`, freeze backbone + BN `eval()`, train `fc` 18.981 params) → `resnet18_finetune` (thay `fc` trước, nạp `resnet18_transfer/best.pt`, unfreeze `layer4 + fc`, differential LR, BN phần khóa `eval()`).
  2. Trả lời câu hỏi chính **"Fine-tune có hợp lý cho bài này không?"** dựa trên tiêu chí định lượng đặt trước (chênh lệch Test Accuracy & Macro-F1 đối chiếu chi phí; nếu `< 1` điểm % trên 3.302 ảnh test thì chạy thêm 2–3 seeds).
  3. Kiểm chứng giả thuyết: lỗi của `transfer` dồn vào giống mèo nhiều hơn giống chó (vì ImageNet chỉ có 5 lớp mèo).
  4. Hoàn thiện báo cáo ngắn gọn, xúc tích tại [`labs/lab2/vibe/docs/REPORT.md`](../../../labs/lab2/vibe/docs/REPORT.md) đồng bộ 1-1 với [`labs/lab2/vibe/lab2.ipynb`](../../../labs/lab2/vibe/lab2.ipynb), đặc tả kỹ thuật đặt tách riêng tại [`labs/lab2/vibe/docs/pipelines/`](../../../labs/lab2/vibe/docs/pipelines/README.md).
- **In-Scope**:
  - Cấu hình override riêng Lab 2 tại `labs/lab2/vibe/config.py` (tái sử dụng `src/config.py` qua `VibeConfig`).
  - Mã nguồn sạch 5 pipeline tại `src/vibe_scripts/lab2/` (`data.py`, `models.py`, `train.py`, `evaluate.py`, `infer.py`) + helper EDA tại `src/vibe_scripts/lab2/eda.py`.
  - Unit test 2 chế độ (Smoke `< 5s` + Normal) tại `test/test_lab2_*.py`.
  - Dọn file CIFAR-10 cũ trong `labs/lab2/vibe/`, thay bằng cấu trúc chuẩn Lab 1.
  - Chạy đầy đủ 3 run thực nghiệm (và đa seed nếu rơi vùng nhiễu `< 1` điểm %), xuất đồ thị vào `experiments/lab2/` và hoàn thiện `REPORT.md` + `lab2.ipynb`.
- **Out-of-Scope**:
  - Không sửa trực tiếp các file thủ công trong `src/` (`config.py`, `datasets.py`, `model.py`, `visualize.py`, `utility.py`, `objects.py`, `class_name.py`).
  - Không thêm kiến trúc khác ngoài ResNet-18 trong phạm vi thí nghiệm chính.

---

## 2. Cấu Hình Override Tại `labs/lab2/vibe/config.py`

Kế thừa `src/config.py` qua `VibeConfig` (`ROOT_PATH`, `DATA_PATH`, `DEVICE`, `SEED = 248`, `BATCH_SIZE = 32`, `EPOCH = 30`, `IMAGE_NET_NORMALIZE`, `WOKER = 8`) và override các tham số Lab 2:

| Tham số | Giá trị gốc (`src/config.py`) | Override (`lab2/vibe/config.py`) | Lý do kỹ thuật |
| :--- | :---: | :---: | :--- |
| `LR` | `1e-5` | `1e-3` | Dùng chung cho `baseline` và `head` của `transfer` / `finetune`; `1e-5` quá nhỏ để Adam hội tụ trong 30 epoch |
| `EPOCH` | `30` | `30` | Giữ nguyên 30 epochs theo `PROMPT.md` |
| `LR_BACKBONE` *(extra)* | *(chưa có)* | `1e-4` | Differential LR cho `layer4` trong `finetune` (= `1/10` của `LR_HEAD`) |
| `VAL_RATIO` *(extra)* | *(chưa có)* | `0.1` | Tách 10% từ tập `test` gốc (`3.669 → 367 val + 3.302 test`), stratified theo 37 giống, `seed = 248` |
| `IMAGE_SIZE` *(extra)* | *(chưa có)* | `224` | `Resize(256) + CenterCrop(224)` cho Val/Test; `RandomResizedCrop(224, scale=(0.7, 1.0)) + HFlip(0.5)` cho Train |
| `BEST_METRIC` *(extra)* | *(chưa có)* | `"val_loss"` | Chọn `best.pt` mượt và ổn định hơn `val_acc` trên tập Val nhỏ 367 ảnh |
| `EXPERIMENT_DIR` *(extra)* | *(chưa có)* | `ROOT_PATH / "experiments" / "lab2"` | Tách riêng `eda/`, `evaluation/`, `predictions/` và từng run (`resnet18_baseline`, `resnet18_transfer`, `resnet18_finetune`) |

---

## 3. Bảng Theo Dõi Tiến Độ (Kanban Board)

| Task ID | Công việc | Phụ trách | Trạng thái | Ghi chú |
| :---: | :--- | :--- | :---: | :--- |
| **T-01** | Sửa nhất quán `PROMPT.md`, khởi tạo Plan, `labs/lab2/vibe/config.py`, khung `docs/` & `pipelines/` | AI Agent | `In Progress` | Đang triển khai |
| **T-02** | Tải `OxfordIIITPet` + `IMAGENET1K_V1` & sinh `manifest.json` SHA-256 | AI Agent | `To Do` | Cần chạy ngoài sandbox để tải từ mạng |
| **T-03** | Pipeline 01: `src/vibe_scripts/lab2/data.py` + `eda.py` + `test/test_lab2_data.py` | AI Agent | `To Do` | Cleaning (corrupt/RGB/dup), EDA, stratified split `3.680 / 367 / 3.302` |
| **T-04** | Pipeline 02: `src/vibe_scripts/lab2/models.py` + `test/test_lab2_models.py` | AI Agent | `To Do` | 3 chế độ `baseline` / `transfer` (18.981 params, BN eval) / `finetune` (layer4+fc, load transfer ckpt) |
| **T-05** | Pipeline 03: `src/vibe_scripts/lab2/train.py` + `test/test_lab2_train.py` | AI Agent | `To Do` | Differential LR, giữ BN phần khóa `eval()` mỗi epoch, atomic save `best.pt`/`last.pt`/`history.json` |
| **T-06** | Pipeline 04 & 05: `evaluate.py` + `infer.py` + `test/test_lab2_{evaluate,infer}.py` | AI Agent | `To Do` | Test 1 lần trên 3.302 ảnh, Cat vs Dog error rate, CM 37×37, top confusion pairs, error gallery |
| **T-07** | Chạy EDA + 3 run thực nghiệm (`30 epochs`, GPU RTX 4060) & kiểm tra quy tắc đa seed (`< 1` điểm %) | AI Agent | `To Do` | Kiểm chứng trên số liệu thật |
| **T-08** | Hoàn thiện `REPORT.md` (ngắn gọn, xúc tích) & đồng bộ `labs/lab2/vibe/lab2.ipynb` | AI Agent | `To Do` | 10 mục chuẩn + ảnh nhúng tương đối |
| **T-09** | Audit 5 Quality Gates, Dual Testing (`pytest -m smoke` + `pytest test/`), cập nhật registry & Thinking Log | AI Agent | `To Do` | Nghiệm thu cuối |

---

## 4. Detailed Execution Plan

- **T-03 (Data & EDA)**:
  - Cleaning: quét 100% ảnh `trainval + test` kiểm tra ảnh lỗi/corrupt, chế độ màu (RGB vs RGBA/L/P — ép `.convert("RGB")`), và ảnh trùng lặp (MD5/SHA-256 trên nội dung ảnh hoặc pHash).
  - EDA: phân bố ảnh theo 37 giống (trainval vs test), phân bố kích thước `(W, H)` và tỉ lệ cạnh `W/H`, lưới ảnh mẫu, tỉ lệ chó/mèo bằng kiểm tra quy ước tên file Oxford-IIIT Pets (mèo viết hoa chữ đầu, chó viết thường) đối chiếu `class_name.OXFORD_III_PET`.
  - Split: Train = toàn bộ `trainval` (`3.680`), tách stratified `10%` từ `test` gốc (`3.669`, `seed = 248`) thành `Val` (`367`, `eval_transform`) và `Test` (`3.302`, `eval_transform`).
- **T-04 (Models — kiểm soát chặt thứ tự và BatchNorm)**:
  - `baseline`: `resnet18(weights=None)`, thay `fc = nn.Linear(512, 37)` (`11.195.493` trainable params).
  - `transfer`: `resnet18(weights=IMAGENET1K_V1)`, thay `fc = nn.Linear(512, 37)`, `requires_grad=False` toàn bộ backbone (`18.981` trainable params); override/helper `set_train_mode()` giữ toàn bộ backbone ở `eval()` để `running_mean/running_var` của BatchNorm **không đổi** suốt quá trình train!
  - `finetune`: khởi tạo `resnet18(weights=None)`, **thay `fc` 512→37 TRƯỚC**, **nạp `resnet18_transfer/best.pt` SAU**, đóng băng `conv1/bn1/layer1/layer2/layer3` (cả `requires_grad=False` lẫn `eval()` để khóa BN), mở khóa `layer4 + fc` (`8.412.709` trainable params); `param_groups` tách `layer4` (`lr=1e-4`) và `fc` (`lr=1e-3`).
- **T-05 (Training)**:
  - `CrossEntropyLoss`, `Adam`, `batch_size = 32`, `30 epochs`, `seed = 248`, AMP `torch.amp.autocast('cuda')` (tăng tốc trên RTX 4060, an toàn fallback CPU), lưu `best.pt` theo `val_loss` + `last.pt` + `history.json`.
  - Test bất biến: kiểm chứng `bn1.running_mean` và `layer1..3` BN stats **bất biến 100%** sau 1 bước train ở cả `transfer` và `finetune`!
- **T-06 (Evaluation & Error Analysis)**:
  - Đánh giá **1 lần duy nhất** trên `Test` (`3.302` ảnh) cho mỗi run: Accuracy, Macro-F1, Test Loss, Best Epoch, Train Time, Trainable Params, Save/Load max diff `< 1e-6`.
  - Tách riêng **Error Rate / Accuracy / Macro-F1 nhóm Mèo (12 giống) vs nhóm Chó (25 giống)** để kiểm chứng giả thuyết lỗi dồn vào giống mèo.
  - Xuất biểu đồ: `learning_curves_comparison.png`, `learning_curve_<exp_id>.png`, `confusion_matrix_<exp_id>.png` (37×37), `cat_vs_dog_error_comparison.png`, `predictions_gallery_<exp_id>.png` (top confident errors), `top_confusions` JSON/CSV.
- **T-07 (Đa seed nếu cần)**:
  - Nếu `|acc(finetune) - acc(transfer)| < 1.0` điểm % hoặc `|f1(finetune) - f1(transfer)| < 1.0` điểm % → chạy thêm 2 seeds (`seed = 42, 2026`) cho cặp `transfer` & `finetune` để báo cáo `mean ± std` trước khi kết luận!

---

## 5. Definition of Done (DoD)

- [ ] `labs/lab2/vibe/config.py` override sạch từ `src/config.py`, không sửa file thủ công trong `src/`.
- [ ] Tất cả 5 pipeline Lab 2 có đủ Smoke Test (`@pytest.mark.smoke`) và Normal Test pass 100% qua `pytest` (kể cả test bất biến khóa BatchNorm).
- [ ] Cả 3 run (`baseline`, `transfer`, `finetune`) chạy công bằng cùng split (`3.680 / 367 / 3.302`), cùng seed, cùng augmentation; chạy đa seed nếu chênh lệch `< 1` điểm %.
- [ ] `labs/lab2/vibe/docs/REPORT.md` ngắn gọn, xúc tích, đủ 10 mục + giả thuyết/tiêu chí đặt trước + số liệu/ảnh thật; đồng bộ với `labs/lab2/vibe/lab2.ipynb`.
- [ ] Đạt đủ 5 Quality Gates, cập nhật `logs/vibe_scripts/README.md`, `test/README.md` và `logs/thinking/INDEX.md`.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-21 | notworLe | Antigravity | Khởi tạo Plan Lab 2 (T-01 → T-09) sau khi thống nhất split `3.680 / 367 / 3.302` và cấu hình override |
