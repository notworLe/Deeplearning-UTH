---
title: Plan — Lab 2 v2 Oxford-IIIT Pets Pre-trained Architectures (Transfer Learning & Fine-Tuning)
doc_type: Plan
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-27
updated: 2026-11-28
version: 1.4.0
status: completed
---

# Plan: Lab 2 v2 — Pre-trained CNN Architectures on Oxford-IIIT Pets (`0006_lab2_v2`)

> **Source Task:** [`.agents/tasks/0006_lab2_v2/task.md`](task.md) · **Exercise:** [`labs/lab2/QUESTION.md`](../../../labs/lab2/QUESTION.md) · **Workflow:** `DL_RULE v1.3.1` (`MODEL_PIPELINE.md`)

---

## Objective

Xây dựng một **mini-project Deep Learning hoàn chỉnh** cho bài toán phân loại **37 giống chó & mèo trên Oxford-IIIT Pets** (ảnh RGB chuẩn hóa ImageNet `(3, 224, 224)`) sử dụng các kiến trúc pre-trained từ `torchvision.models` (`ResNet18`, `DenseNet121`, `VGG16`), tuân thủ trọn bộ 10 tầng [`MODEL_PIPELINE.md`](../../rules/dl-rule/MODEL_PIPELINE.md) và áp dụng toàn bộ 6 bài học đã trích xuất từ [`0005_lab1v2_knowledge.md`](../../knowledge_extraction/0005_lab1v2_knowledge.md):

1. Đo lường định lượng xem **Transfer Learning** (freeze backbone) và **Fine-Tuning** (unfreeze `layer4`) có vượt trội đáng kể so với **train từ đầu (`ResNet18` scratch baseline)** hay không.
2. So sánh công bằng **3 kiến trúc pre-trained** (`ResNet18`, `DenseNet121`, `VGG16`) về độ chính xác, số tham số trainable, tốc độ hội tụ, VRAM, thời gian huấn luyện và độ trễ inference.
3. Thực hiện khảo sát siêu tham số có kiểm soát (`lr`, `batch_size`), giám sát song song bằng **`tqdm` + CSV/JSON + TensorBoard**, chống tuyệt đối data/test leakage, và phân tích lỗi 4 bước định lượng (gồm phân tích loài `Cat vs Dog`).

---

## Understanding & Assumptions (Cần Kỹ sư Xác Nhận)

### 1. Hiện trạng môi trường & dữ liệu đã xác minh
- **Dataset có sẵn (bất biến):** `data/raw/oxford-iiit-pet/` đã chứa `oxford-iiit-pet.tar.gz` (`811.7 MB`, khớp `manifest.json`), `images/`, `annotations/` (`7,349` ảnh: `3,680` `trainval` + `3,669` `test`, `37` lớp: `12` giống mèo + `25` giống chó). Không cần tải lại (`download=False`).
- **Môi trường GPU:** Python `3.13.14`, PyTorch `2.15.0+cu132`, Torchvision `0.29.2+cu132`, **CUDA `True` (RTX 4060 Laptop GPU)**; `PIL`, `sklearn`, `tqdm`, `matplotlib`, `nbformat`, `pytest` đều đã có sẵn.
- **Lưu ý `tensorboard`:** Hiện **`tensorboard` chưa được cài** trong venv (`find_spec("tensorboard") == False`). Trong `task.md` mục *Constraints* cho phép dùng `tensorboard` → Code sẽ hỗ trợ `SummaryWriter` tự động (ghi vào `experiments/lab2_v2/tb_logs/<run_id>/`) và bật cơ chế graceful fallback nếu chưa cài; Kỹ sư có thể cài `pip install tensorboard` trước Step 6 nếu muốn xem giao diện TensorBoard.
- **Cô lập hoàn toàn khỏi bản v1:** `labs/lab2/vibe/`, `src/vibe_scripts/lab2/` và `experiments/lab2/` đã tồn tại từ bản v1 → bản v2 cô lập 100% tại **`src/vibe_scripts/lab2_v2/`** (sửa typo `vibe_scrtips`), **`labs/lab2_v2/`**, **`experiments/lab2_v2/`** và **`test/test_lab2_v2_*.py`**.

### 2. Các điểm mâu thuẫn / cần xác nhận trong `task.md`

| # | Điểm phát hiện trong `task.md` | Phương án đề xuất | Cần Kỹ sư xác nhận |
| :-: | :--- | :--- | :--- |
| **A1** | **Nội dung sót copy-paste từ Lab 1:** dòng 131, 143, 237, 240, 252 vẫn ghi `"MLP model"`, `"hidden dimensions"`, `"why MLP performs better than baseline"`. | Loại bỏ hoàn toàn phần MLP/hidden_dims; tập trung đúng mục tiêu Lab 2 là **ResNet18 (scratch / transfer / fine-tune), DenseNet121 (transfer), VGG16 (transfer)** và ablation **`lr`, `batch_size`**. | Xác nhận tự động áp dụng |
| **A2** | **Mâu thuẫn cách chia tập dữ liệu:**<br>• Dòng 84–86 (*Problem Definition*): `Train = trainval (3,680)`, `Val = 35% of test (~1,284)`, `Test = 65% of test (~2,385)` (stratified).<br>• Dòng 186 (*T2*): `"create 15% val split"` (tách 85% Train / 15% Val từ `trainval` = `3,128 / 552`, giữ nguyên 100% tập `test` gốc = `3,669`). | **Khuyến nghị (Option 1 — chuẩn benchmark):** tách `85% Train (3,128) / 15% Val (552)` từ `trainval` (stratified `seed=248`), giữ **100% tập `test` gốc (`3,669` ảnh)** hoàn toàn nguyên vẹn để so sánh chuẩn với các benchmark Oxford-IIIT Pets.<br>*(Option 2: giữ đúng dòng 84–86 là `Train = trainval 3,680`, `Val = 35% test`, `Test = 65% test` — vẫn không leak vì Val & Test tách rời và có nhiều dữ liệu train hơn +17.6%.)* | **Chọn Option 1 hay Option 2?** |
| **A3** | **Cấu trúc tài liệu (SRP 1-1 theo Lab 1 v2 vs danh sách ở `task.md`):** `task.md` liệt kê cả `docs/pipelines/{pipeline_overall,experiment,evaluation}.md` lẫn `experiment_report.md`. | **Khuyến nghị (đồng bộ chuẩn `lab1_v2` đã kiểm chứng):** `docs/README.md`, `docs/pipelines/modeling.md` (10 tầng), `docs/models/{resnet18,densenet121,vgg16}.md`, và **3 báo cáo đơn nhiệm SRP**: `docs/eda_report.md` (Layer 6), `docs/experiment_report.md` (Layer 4 & 7), `docs/evaluation_report.md` (Layer 8–10). | **Đồng ý giữ chuẩn 3 báo cáo SRP như `lab1_v2`?** |
| **A4** | **Tiêu chí chọn mô hình thắng cuộc (Layer 8.1):** `task.md` dòng 199 & 266 ghi chọn theo **`val_acc`**; `lab1_v2` dùng `(val_macro_f1, val_acc, -val_loss)`. | Áp dụng quy tắc 2 tầng: trong 1 run chọn `best.pt` theo **`min(val_loss)`**; giữa các runs xếp hạng theo **`(val_acc, val_macro_f1, -val_loss)`** (đúng ưu tiên `val_acc` của `task.md`, dùng `val_macro_f1` & `-val_loss` phá hòa). | **Xác nhận ưu tiên `val_acc` hay `val_macro_f1`?** |
| **A5** | **Giám sát TensorBoard (T9):** `tensorboard` hiện chưa có trong venv. | Tích hợp `SummaryWriter` (có cờ `--no-tensorboard`, tự fallback nếu chưa cài) + hướng dẫn Kỹ sư cài `pip install tensorboard` (hoặc Agent cài với quyền ngoài sandbox nếu Kỹ sư yêu cầu). | **Cài `tensorboard` hay chỉ tích hợp optional?** |
| **A6** | **Default hyperparameters & Mixed Precision (`AMP`):** đề xuất `AdamW(wd=1e-4)` + `CosineAnnealingLR`, `20 ep` cho Scratch, `15 ep` cho Transfer/Fine-tune (`layer4` cho ResNet18 với differential LR `1e-4 / 1e-3`; DenseNet121 & VGG16 chỉ chạy Transfer), bật tùy chọn `--amp` (FP16 trên RTX 4060 giúp nhanh gấp ~2× và tiết kiệm VRAM cho VGG16). | Đề xuất cụ thể trong bảng Controlled Experiment Suite bên dưới. | **Xác nhận cấu hình mặc định & 9 runs?** |

---

## Scope

- **In Scope:**
  - Bộ tài liệu thiết kế 10 tầng pipeline, 3 đặc tả kiến trúc (`ResNet18`, `DenseNet121`, `VGG16`) và 3 báo cáo SRP (`eda_report.md`, `experiment_report.md`, `evaluation_report.md`).
  - Mã nguồn lõi tái sử dụng tại `src/vibe_scripts/lab2_v2/`: `transforms.py`, `data.py` (6.1 Collection, 6.2 Cleaning, 6.3 EDA gồm phân bố 37 giống, tỷ lệ Cat vs Dog, phân bố kích thước/aspect-ratio ảnh gốc, 6.4 Stratified split lưu `split_indices.json`), `models.py` (`build_model`, `freeze_backbone`, `unfreeze_last_block`, `get_param_groups`, `count_parameters`, `trace_shapes`), `trainer.py` (`tqdm` bar, `historys/history.{json,csv}`, TensorBoard `SummaryWriter`, `--resume` + `optimizer_state_dict`, AMP fp16, Sanity Checks), `evaluator.py` (per-run Val eval, Final Test 1 lần, 4-Step Error Analysis gồm Cat-vs-Dog species analysis + 60-image audit sheets + Slice Analysis), `inference.py` (`PetPredictor` Top-5 Softmax + latency benchmark).
  - Cấu hình kế thừa `VibeConfig` (`labs/lab2_v2/config.py`), 4 thin CLI scripts (`prepare_data.py`, `train.py`, `evaluate.py`, `inference.py`), `labs/lab2_v2/experiment.ipynb` (executable report không train lại).
  - Bộ test 2 tầng offline (`weights=None`, synthetic/tiny subset trong `tmp_path`) tại `test/test_lab2_v2_{data,models,pipeline}.py`.
  - Hồ sơ nghiệm thu Phase 6–8 tại `.agents/{tests,logs,knowledge_extraction}/0006_lab2_v2_*`.
- **Out of Scope:**
  - **Hard Constraint:** Agent **không tự chạy full training thật** (Kỹ sư trực tiếp chạy CLI ở Step 6 với `tqdm` bar log + TensorBoard).
  - Không fine-tune DenseNet121 / VGG16 (chỉ chạy Transfer theo Assumption `task.md` dòng 265), không triển khai ONNX/Docker/FastAPI serving.
  - Không chỉnh sửa `src/*.py` gốc, `src/vibe_scripts/lab1*`, `src/vibe_scripts/lab2/` (v1), `labs/lab1*`, `labs/lab2/vibe/`, `experiments/lab1*`, `experiments/lab2/`, `data/raw/oxford-iiit-pet/`, `.gitignore`.

---

## Controlled Experiment Suite (9 Runs — Per-Run Val Eval & TensorBoard)

Cố định `seed = 248`, `CrossEntropyLoss`, ImageNet Normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), `AdamW(weight_decay=1e-4)` + `CosineAnnealingLR`, lưu `best.pt` theo `min(val_loss)` kèm `optimizer_state_dict`.

| # | Nhóm / Task | Run ID | Backbone & Mode | Trainable / Total Params | LR | BS | Epochs |
| :-: | :--- | :--- | :--- | :---: | :---: | :-: | :-: |
| 1 | **Baseline (T5)** | `resnet18_scratch` | `ResNet18` (`weights=None`, train all) | `11.20M / 11.20M (100%)` | `1e-3` | 32 | 20 |
| 2 | **Transfer Anchor (T6)** | `resnet18_transfer` | `ResNet18` (`DEFAULT`, freeze backbone, train `fc`) | `19.0K / 11.20M (0.17%)` | `1e-3` | 32 | 15 |
| 3 | **Fine-Tune (T6)** | `resnet18_finetune` | `ResNet18` (`DEFAULT`, unfreeze `layer4 + fc`) | `8.41M / 11.20M (75.1%)` | `1e-4` (L4) / `1e-3` (fc) | 32 | 15 |
| 4 | **Architecture (T7)** | `densenet121_transfer` | `DenseNet121` (`DEFAULT`, freeze backbone, train `classifier`) | `37.9K / 6.99M (0.54%)` | `1e-3` | 32 | 15 |
| 5 | **Architecture (T7)** | `vgg16_transfer` | `VGG16` (`DEFAULT`, freeze `features + cls[0..5]`, train `cls[6]`) | `151.6K / 134.41M (0.11%)` | `1e-3` | 32 | 15 |
| 6 | **Ablation LR (T8)** | `resnet18_transfer_lr1e4` | `ResNet18` transfer (như #2, đổi `lr`) | `19.0K / 11.20M (0.17%)` | `1e-4` | 32 | 15 |
| 7 | **Ablation LR (T8)** | `resnet18_transfer_lr5e3` | `ResNet18` transfer (như #2, đổi `lr`) | `19.0K / 11.20M (0.17%)` | `5e-3` | 32 | 15 |
| 8 | **Ablation BS (T8)** | `resnet18_transfer_bs16` | `ResNet18` transfer (như #2, đổi `batch_size`) | `19.0K / 11.20M (0.17%)` | `1e-3` | 16 | 15 |
| 9 | **Ablation BS (T8)** | `resnet18_transfer_bs64` | `ResNet18` transfer (như #2, đổi `batch_size`) | `19.0K / 11.20M (0.17%)` | `1e-3` | 64 | 15 |

*(Ghi chú thời gian dự kiến trên RTX 4060 với `--amp`: 9 runs mất khoảng ~12–18 phút tổng cộng; Kỹ sư có thể chạy trọn gói `--run-all` hoặc chạy từng nhóm `--run-id` độc lập.)*

---

## Work Breakdown

Thực thi tuần tự từng bước, mỗi bước đều qua **Self Audit** (`EXECUTION.md`):

- [x] **Step 1 — Tài liệu thiết kế Pipeline & 3 Kiến trúc Pre-trained (T1):**
  - Viết `labs/lab2_v2/docs/README.md`, `labs/lab2_v2/docs/pipelines/modeling.md` (đủ 10 tầng + Tensor Shape Contract `(3, H, W) → (B, 3, 224, 224) → Feature Map → (B, 37)` + quy tắc chọn model §8.1).
  - Viết 3 đặc tả mô hình: `labs/lab2_v2/docs/models/{resnet18,densenet121,vgg16}.md` (cấu trúc chi tiết, vị trí classification head, tỷ lệ trainable params khi freeze/unfreeze, ưu/nhược điểm; không dự đoán trước con số accuracy).
- [x] **Step 2 — Config, Transforms & Data Engineering CLI (`prepare_data.py`, T2–T3):**
  - Tạo `labs/lab2_v2/config.py` (kế thừa `src/config.py` qua `VibeConfig`), `src/vibe_scripts/lab2_v2/{__init__.py, README.md, transforms.py, data.py}` và thin CLI `labs/lab2_v2/prepare_data.py`.
  - Chạy **6.1 Collection Audit** (xác thực `manifest.json` & số ảnh) → **6.2 Cleaning Audit** (quét 100% `7,349` ảnh: corrupt/unreadable, mode RGB/RGBA/L, kích thước bất thường, exact duplicate SHA-256 nội bộ & chéo split) → **6.3 EDA** (phân bố 37 lớp, tỷ lệ Cat 12 vs Dog 25, phân bố kích thước/aspect ratio gốc, lưới ảnh mẫu, thống kê RGB) → lưu `split_indices.json` cố định + `labs/lab2_v2/docs/eda_report.md`.
- [x] **Step 3 — Model Factory & Freeze/Unfreeze Engine (T4):**
  - Tạo `src/vibe_scripts/lab2_v2/models.py`: hỗ trợ `resnet18`, `densenet121`, `vgg16` với chế độ `scratch` / `transfer` / `finetune`, thay đúng head (`fc` / `classifier` / `classifier[6]`), tạo differential parameter groups (`layer4` vs `fc`), `count_parameters(total, trainable)`, `trace_shapes()`, Fail-Fast shape validation `(B, 3, 224, 224) → (B, 37)`.
- [x] **Step 4 — Training, TensorBoard Monitoring, Evaluator (4-Step Error Analysis), Inference & CLI Entrypoints (T5–T10):**
  - Tạo `src/vibe_scripts/lab2_v2/trainer.py`: global seeding, Sanity Checks (`initial_loss ≈ ln 37 ≈ 3.6109`, overfit 1 batch `≥ 95%`), vòng lặp train/val hỗ trợ `--amp`, `tqdm` monitoring (`tr_loss`, `val_loss`, `tr_acc`, `val_acc`, `lr`, `sps`, `vram`), lưu `historys/history.{json,csv}` + TensorBoard `tb_logs/<run_id>/`, `--resume` + `optimizer_state_dict`, tự gọi per-run Val eval vào `runs/<id>/eval/`.
  - Tạo `src/vibe_scripts/lab2_v2/evaluator.py`: `evaluate_run_dir()` (Val & Test tách biệt không vẽ lặp curves), `evaluate_final_test()` (1 lần duy nhất cho Baseline vs Best Pre-trained), `run_deep_error_analysis()` (4 bước định lượng: CM chuẩn hóa % + loài Cat vs Dog, Gallery Top-3 cặp High/Low Conf, 3 Audit Sheets `60` mẫu, Slice Analysis theo độ sáng / aspect ratio / diện tích gốc / loài).
  - Tạo `src/vibe_scripts/lab2_v2/inference.py` (`PetPredictor` Top-5 Softmax, benchmark latency BS=1 & BS=32) và thin CLIs `labs/lab2_v2/{train.py, evaluate.py, inference.py, README.md, experiment.ipynb}` + khung báo cáo SRP (`experiment_report.md`, `evaluation_report.md`).
- [x] **Step 5 — Dual Unit Tests Offline (Smoke + Normal, KHÔNG chạy training thật):**
  - Viết `test/test_lab2_v2_{data,models,pipeline}.py` chạy 100% offline (`weights=None`, synthetic/tiny subset trong `tmp_path`): kiểm chứng split không giao nhau, transforms shape/dtype/ImageNet norm, freeze/unfreeze exact trainable param counts cho cả 3 kiến trúc, differential LR groups, sanity checks, 1-epoch train + `--resume` + per-run eval + Save/Load parity `< 1e-6`, `test_eval/` không lặp curves, TensorBoard writer fallback, và `nbformat.validate`.
  - Chạy `pytest -m smoke` và full `pytest test/test_lab2_v2_*.py` pass 100%.
- [x] **Step 6 — [ENGINEER EXECUTION GATE] Bàn giao CLI cho Kỹ sư tự chạy Training & TensorBoard:**
  - **Agent STOP — không tự chạy full training.**
  - Kỹ sư đã cài `tensorboard` và hoàn tất trọn bộ **`9/9` runs** bằng CLI (`--amp`, `tb=True`, Save/Load Parity `0.0` PASS trên `9/9` checkpoints).
- [x] **Step 7 — Hoàn thiện 2 Báo Cáo SRP (`experiment_report.md`, `evaluation_report.md`) & Chạy Executable Notebook (T11):**
  - Tổng hợp `ablation_summary.json` / `ablation_table.csv` (`9/9` runs → chọn **`densenet121_transfer`**: `93.22%` Val Acc / `93.18%` Val Macro-F1 / `0.2409` Val Loss); chạy Final Test 1 lần (`91.91%` Test Acc / `91.76%` Macro-F1 / `99.71%` Species Acc, `+44.82 pp` Acc & `+45.49 pp` F1 so với `resnet18_scratch`), 4-Step Error Analysis (`N = 60` ảnh kiểm toán trực quan) và Multi-Arch Inference Benchmark.
  - Hoàn thiện 100% số liệu thật trong `experiment_report.md` (`v1.0.0`, Layer 4 & 7, kèm đối chiếu `resnet18_transfer_lr5e3`) và `evaluation_report.md` (`v1.0.0`, Layer 8–10, kèm `<details>` 3 Audit Sheets).
  - Chạy sạch `labs/lab2_v2/experiment.ipynb` (chế độ đọc artifacts & live inference) và xác minh `0` broken links.
- [x] **Step 8 — Testing Report, Execution Logging & Knowledge Extraction (Phase 6–8):**
  - Đã lưu hoàn chỉnh [`.agents/tests/0006_lab2_v2_test_report.md`](../../tests/0006_lab2_v2_test_report.md), [`.agents/logs/0006_lab2_v2_log.md`](../../logs/0006_lab2_v2_log.md), [`.agents/knowledge_extraction/0006_lab2_v2_knowledge.md`](../../knowledge_extraction/0006_lab2_v2_knowledge.md) và cập nhật `experiment_report.md` (`v1.1.0`) ghi nhận phân tích Train–Val gap từ Kỹ sư.

---

## Risks & Mitigations

| Rủi ro | Tác động | Biện pháp kiểm soát |
| :--- | :--- | :--- |
| **Ảnh RGBA / Grayscale / Corrupt trong Oxford-IIIT Pets** | Lỗi channel count `(4, H, W)` hoặc `(1, H, W)` khi chuẩn hóa 3 kênh RGB | Quét toàn bộ `7,349` ảnh ở Layer 6.2 Cleaning và ép `.convert("RGB")` nhất quán trong loader & `preprocess_image()` |
| **VRAM OOM trên RTX 4060 (`8 GB`) với `VGG16` / `DenseNet121`** | Tràn VRAM khi batch size lớn | Freeze backbone ở chế độ Transfer (`VGG16` chỉ giữ gradient ở `classifier[6]`), cung cấp cờ `--amp` (FP16) và theo dõi đỉnh VRAM trên `tqdm` & `history.csv` |
| **Tải weight ImageNet bị chặn trong sandbox test** | `pytest` bị lỗi mạng khi chạy trong sandbox | Unit tests dùng `weights=None` 100% offline; chỉ nạp `weights="DEFAULT"` khi Kỹ sư chạy CLI thật ở Step 6 |
| **Data Leakage giữa Train / Val / Test** | Kết quả benchmark không đáng tin | Lưu cố định `split_indices.json` (`seed=248`), kiểm chứng giao tập rỗng trong unit test, Val eval riêng ở `runs/<id>/eval/`, Test chấm đúng 1 lần cuối ở `test_eval/` |
| **Matplotlib `TkAgg` crash trên Windows** | Treo CLI khi lưu biểu đồ | Áp dụng guard `backend != "agg"` + `plt.close(fig)` theo Knowledge Item #1 |

---

## Validation

1. **Smoke Test (`pytest -m smoke test/test_lab2_v2_*.py -v`):** import sạch, config load OK, forward pass cả 3 kiến trúc (`resnet18`, `densenet121`, `vgg16`) đúng `(B, 37)`, 1-epoch train + eval + inference smoke pass trong `tmp_path`.
2. **Normal Test (`pytest test/test_lab2_v2_*.py -v`):**
   - Split stratified `37` lớp không giao nhau, `RGB` 3 kênh `(3, 224, 224)`, ImageNet normalization đúng.
   - Đếm chính xác `total_params` và `trainable_params` cho từng mode (`scratch` / `transfer` / `finetune`) trên `ResNet18`, `DenseNet121`, `VGG16`.
   - Sanity checks (`initial_loss ≈ ln 37`, overfit 1 batch `≥ 95%`), Save/Load parity `< 1e-6`, `--resume` nối liên tục lịch sử + khôi phục `optimizer_state_dict`, `test_eval/` không vẽ lặp learning curves, `nbformat.validate` pass.
3. **Acceptance Criteria Thực Nghiệm (sau Step 6):**
   - `resnet18_scratch` hội tụ làm mốc Baseline thực nghiệm.
   - Các mô hình Transfer / Fine-tune tạo **gap cải thiện dương rõ rệt** so với `resnet18_scratch` trên tập Test độc lập, đồng thời giảm mạnh số epoch cần để hội tụ.
   - So sánh định lượng đủ 3 kiến trúc (`ResNet18`, `DenseNet121`, `VGG16`), ablation `lr` & `batch_size`, có biểu đồ TensorBoard/curves, Error Analysis 4 bước (gồm 60 ảnh audit) và CLI inference chạy sạch.

---

## Deliverables

- **Documentation (`labs/lab2_v2/docs/`):** `README.md`, `pipelines/modeling.md`, `models/{resnet18,densenet121,vgg16}.md`, `eda_report.md`, `experiment_report.md`, `evaluation_report.md`
- **Reusable Core (`src/vibe_scripts/lab2_v2/`):** `README.md`, `__init__.py`, `transforms.py`, `data.py`, `models.py`, `trainer.py`, `evaluator.py`, `inference.py`
- **Config, CLIs & Notebook (`labs/lab2_v2/`):** `README.md`, `config.py`, `prepare_data.py`, `train.py`, `evaluate.py`, `inference.py`, `experiment.ipynb`
- **Tests (`test/`):** `test_lab2_v2_data.py`, `test_lab2_v2_models.py`, `test_lab2_v2_pipeline.py`
- **Artifacts (`experiments/lab2_v2/`):** `README.md`, `eda/*`, `splits/split_indices.json`, `tb_logs/*`, `runs/<run_id>/*`, `ablation/*`, `evaluation/*`, `figures/*`
- **Workflow Records (`.agents/`):** `.agents/tasks/0006_lab2_v2/plan.md`, `.agents/tests/0006_lab2_v2_test_report.md`, `.agents/logs/0006_lab2_v2_log.md`, `.agents/knowledge_extraction/0006_lab2_v2_knowledge.md`

---

## Estimated Impact

- **Tạo mới:** `src/vibe_scripts/lab2_v2/*`, `labs/lab2_v2/*`, `test/test_lab2_v2_*.py`, `experiments/lab2_v2/*`, `.agents/tasks/0006_lab2_v2/plan.md`, `.agents/{tests,logs,knowledge_extraction}/0006_lab2_v2_*`.
- **Không đụng tới:** Toàn bộ `src/*.py` gốc, `src/vibe_scripts/lab1*`, `src/vibe_scripts/lab2/*` (v1), `labs/lab1*`, `labs/lab2/{main,vibe,submit}/*`, `experiments/lab1*`, `experiments/lab2/*`, `data/raw/oxford-iiit-pet/*`, `.gitignore`, `README.md` gốc.

---

## Approval Gate (Bắt buộc)

- **Status:** `approved` (Kỹ sư đã xác nhận 4 quyết định: **A2** `Train = trainval (3,680)`, `Val = 35% test (~1,284)`, `Test = 65% test (~2,385)` stratified `seed=248`; **A3** chuẩn 3 báo cáo SRP như `lab1_v2`; **A4** ưu tiên chọn model `(val_acc, val_macro_f1, -val_loss)` + `best.pt` theo `min(val_loss)`; **A5** tích hợp `SummaryWriter` optional fallback để Kỹ sư tự cài `tensorboard` trước Step 6. Agent triển khai Step 1 → Step 5 và **STOP tại Step 6** để Kỹ sư chạy training).

---

## Lịch Sử Thay Đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| `1.4.0` | 2026-11-28 | notworLe | Antigravity | Hoàn tất **Step 8** (Phase 6–8: `0006_lab2_v2_{test_report,log,knowledge}.md`, cập nhật `experiment_report.md` `v1.1.0` bổ sung chẩn đoán Train–Val gap từ Kỹ sư, re-run `14/14` pytest PASS) và chuyển `status: completed` sẵn sàng commit |
| `1.3.0` | 2026-11-27 | notworLe | Antigravity | Hoàn tất **Step 6** (`9/9` runs do Kỹ sư chạy, TensorBoard active `9/9`) và **Step 7** (`ablation_summary` 9 runs → `densenet121_transfer`, Final Test `91.91%` Acc / `+44.82 pp` vs scratch, 4-Step Error Analysis `N=60`, Multi-Arch Inference Benchmark, hoàn thiện 2 báo cáo SRP `v1.0.0` & thực thi sạch `experiment.ipynb`) |
| `1.2.0` | 2026-11-27 | notworLe | Antigravity | Hoàn tất **Steps 1–5** (`prepare_data.py` kiểm toán 7,349 ảnh & xuất 5 biểu đồ EDA + `eda_report.md`; xây đủ `src/vibe_scripts/lab2_v2/`, `labs/lab2_v2/`, `experiment.ipynb`; Smoke `7/7` PASS, Full Normal `14/14` PASS, `75/75` doc links PASS) — **STOP tại Step 6** bàn giao CLI cho Kỹ sư tự chạy training |
| `1.1.0` | 2026-11-27 | notworLe | Antigravity | Chốt 4 quyết định **A2–A5** từ Kỹ sư (split `trainval` + `35%/65%` của `test`, chuẩn 3 báo cáo SRP, tiêu chí chọn `(val_acc, val_macro_f1, -val_loss)`, TensorBoard optional fallback) và chuyển `status: approved` để bắt đầu Step 1 → Step 5 |
| `1.0.0` | 2026-11-27 | notworLe | Antigravity | Khởi tạo Plan cho `0006_lab2_v2` tại `.agents/tasks/0006_lab2_v2/plan.md` (theo chuẩn `PLAN.md` cập nhật 2026-11-26), áp dụng 6 bài học từ `lab1_v2` và nêu rõ các giả định cần xác nhận |
