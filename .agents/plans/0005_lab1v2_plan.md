---
title: Plan — Lab 1 v2 FashionMNIST Classification Pipeline
doc_type: Plan
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-25
updated: 2026-11-26
version: 1.7.0
status: completed
---

# Plan: Lab 1 v2 — FashionMNIST Classification Mini-Project

> **Source Task:** [`.agents/tasks/0005_lab1v2/task.md`](../tasks/0005_lab1v2/task.md) · **Workflow:** `DL_RULE v1.3.1` (`MODEL_PIPELINE.md`)

---

## Objective

Xây dựng một **mini-project Deep Learning hoàn chỉnh** cho bài toán phân loại ảnh **FashionMNIST** (10 lớp, ảnh xám `28×28`, shape `(1, 28, 28)`), đi trọn vẹn các lớp của [`MODEL_PIPELINE.md`](../rules/dl-rule/MODEL_PIPELINE.md):
1. Tài liệu hóa kiến trúc pipeline, tensor shapes và thiết kế mô hình (`Logistic Regression` vs `MLP`).
2. Triển khai mã nguồn tái sử dụng, modular, config-first tại `src/vibe_scripts/lab1_v2/` cùng các thin CLI script tại `labs/lab1_v2/` (có `tqdm` progress bar log).
3. Thiết kế vòng lặp thí nghiệm **Train xong 1 run → Evaluate ngay và lưu kết quả eval ngay trong thư mục của run đó → Tự động kết thúc chuỗi thí nghiệm ngay khi tìm thấy MLP đạt ngưỡng evaluation tốt**.
4. **Bàn giao script đã kiểm thử sạch để Kỹ sư trực tiếp chạy training**, sau đó Agent hoàn thiện báo cáo (`experiment_report.md`), executable report notebook (`experiment.ipynb`) và nghiệm thu Phase 6–8.

---

## Understanding & Confirmed Assumptions

Các giả định và ràng buộc đã được Kỹ sư xác nhận / bổ sung:

1. **Cô lập hoàn toàn khỏi bản v1 và `src/` gốc:**
   - Code lõi tái sử dụng đặt tại `src/vibe_scripts/lab1_v2/` (khắc phục typo `vibe_scrtips` trong `task.md`, không đè `src/vibe_scripts/lab1/`).
   - Config, thin CLI entrypoints (`train.py`, `evaluate.py`, `inference.py`), executable report notebook (`experiment.ipynb`), và docs (`docs/pipelines/modeling.md`, `docs/models/{logistic_regression,mlp}.md`, `docs/experiment_report.md`) đặt tại `labs/lab1_v2/`.
   - Artifacts lưu tại `experiments/lab1_v2/`.
   - File nháp 5 dòng [`labs/lab1_v2/docs/PIPELINE_LAYERS.md`](../../labs/lab1_v2/docs/PIPELINE_LAYERS.md) sẽ được gộp nội dung vào `labs/lab1_v2/docs/pipelines/modeling.md` và gỡ bỏ để tránh trùng lặp.
2. **Thiết kế Split & Thí nghiệm chống Data Leakage + Per-Run Eval + Early Stop Sweep:**
   - Dữ liệu gốc `data/FashionMNIST/raw/` (`60,000 train / 10,000 test`) giữ nguyên bất biến; xác thực `manifest.json` (SHA-256) khi nạp.
   - Tách Stratified **`55,000 Train / 5,000 Val / 10,000 Test`** (`seed = 248`); tính `mean/std` **chỉ trên 55k Train** rồi áp dụng cho Val/Test.
   - **Cơ chế chạy từng run (Train → Eval ngay tại chỗ):** mỗi run `<run_id>` khi train xong lập tức gọi hàm evaluate trên tập Validation (`5k Val`) và lưu toàn bộ kết quả eval (`eval_metrics.json`, `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `predicted_vs_actual.png`) **ngay bên trong `experiments/lab1_v2/runs/<run_id>/eval/`**, đồng thời cập nhật ngay bảng `experiments/lab1_v2/ablation/ablation_summary.json` & `.csv`.
   - **Điều kiện dừng sớm chuỗi thí nghiệm (Early-Stop Experiment Suite):**
     - Luôn chạy xong `baseline_logistic` trước để có mốc đối chứng.
     - Từ các run MLP trở đi: ngay sau khi eval xong 1 run, nếu đạt ngưỡng tốt mặc định trong `config.py` (`TARGET_MLP_VAL_ACC = 0.89` và `TARGET_MLP_VAL_F1 = 0.89` — có thể override qua CLI `--target-acc`, `--target-f1`, hoặc tắt dừng sớm bằng `--no-stop-on-target`), hệ thống **kết thúc chuỗi thí nghiệm ngay lập tức** và ghi nhận `stop_reason` + mô hình thắng cuộc.
   - **Đánh giá cuối trên tập Test (10k):** chạy `evaluate.py` đúng **1 lần duy nhất** trên tập Test cho `baseline_logistic` vs `best_mlp` đã chọn (đảm bảo không leak tập Test trong quá trình chọn model).
3. **HARD CONSTRAINT — Agent KHÔNG được tự chạy training thật:**
   - Agent tuyệt đối **không** chạy `labs/lab1_v2/train.py` hay huấn luyện thật vào `experiments/lab1_v2/`.
   - Toàn bộ quá trình training do **Kỹ sư tự chạy** qua CLI với **`tqdm` progress bar logging** (outer bar tiến độ các runs + inner epoch/batch bar hiển thị live `train_loss`, `val_loss`, `val_acc`, `lr`, thời gian, cùng bảng tóm tắt eval ngay sau mỗi run).
   - Agent chỉ được chạy unit test nhanh (`pytest`) trên **dummy/subset siêu nhỏ (`≤ 64` mẫu, 1–2 epochs trong `tmp_path`)** để kiểm chứng logic code, tensor shape, sanity check, save/load, per-run eval và cơ chế dừng sớm trước khi bàn giao script cho Kỹ sư.
4. **Vai trò Script vs Notebook:**
   - Mọi thực nghiệm (EDA export, training, per-run evaluation, final test benchmark, inference) chạy bằng **CLI scripts**.
   - `labs/lab1_v2/experiment.ipynb` là **executable report**: nạp kết quả/biểu đồ từ `experiments/lab1_v2/` do Kỹ sư đã chạy và thực thi các cell kiểm chứng nhẹ (EDA, kiểm tra tensor shape, live inference demo) — **không train lại** trong notebook.
5. **Constraints kỹ thuật khác:**
   - Kế thừa `src/config.py` qua `VibeConfig` (`labs/lab1_v2/config.py`), không hardcode path/hyperparameter trong `src/vibe_scripts/lab1_v2/`.
   - Không sửa `src/` gốc, không sửa `labs/lab1/vibe/` hay `experiments/lab1/`, không cài thêm package ngoài môi trường hiện có (`tqdm 4.68.4` đã có sẵn).

---

## Scope

- **In Scope:**
  - Tài liệu 10 lớp pipeline (`modeling.md`), đặc tả 2 mô hình (`logistic_regression.md`, `mlp.md`), và khung báo cáo thực nghiệm (`experiment_report.md` — điền số liệu thật sau khi Kỹ sư chạy xong).
  - Các module lõi tại `src/vibe_scripts/lab1_v2/`: `transforms.py`, `data.py`, `logistic.py`, `mlp.py`, `trainer.py`, `evaluator.py`, `inference.py`.
  - Cấu hình `labs/lab1_v2/config.py`, thin CLI scripts (`train.py`, `evaluate.py`, `inference.py`), `labs/lab1_v2/README.md`, và `experiment.ipynb`.
  - Cấu hình chuỗi 12 runs có kiểm soát (1 Baseline + 1 Anchor MLP + 10 Ablation variants trên 5 trục) với cơ chế **Train → Eval ngay trong folder run → Dừng chuỗi khi đạt ngưỡng tốt**.
  - Bộ test 2 tầng (Smoke + Normal) trên dữ liệu nhỏ trong `tmp_path` tại `test/test_lab1_v2_*.py`, kèm hồ sơ Phase 6–8 tại `.agents/{tests,logs,knowledge_extraction}/`.
- **Out of Scope:**
  - Agent không tự chạy training thật (Kỹ sư chạy ở Step 6).
  - Không thêm CNN / ViT hay triển khai FastAPI / Docker / ONNX serving.
  - Không chỉnh sửa `src/` gốc, `labs/lab1/vibe/`, `experiments/lab1/`, hay cài thêm package mới.

---

## Controlled Experiment Suite (12 Runs — Per-Run Eval & Early Stop)

Cố định `seed = 248`, `CrossEntropyLoss`, `Normalize(train_mean, train_std)`, lưu `best.pt` theo `val_loss` thấp nhất. Thứ tự chạy tuần tự — mỗi run hoàn tất sẽ **evaluate ngay vào `experiments/lab1_v2/runs/<run_id>/eval/`**:

| Thứ tự | Nhóm | Run ID | Kiến trúc / Thay đổi (1 yếu tố / trục) | Params | Optimizer | LR | BS | Epochs |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| 1 | **Baseline** *(luôn chạy)* | `baseline_logistic` | `Flatten(784) → Linear(784, 10)` | 7,850 | Adam | `1e-3` | 64 | 20 |
| 2 | **MLP Anchor** | `mlp_default` | `hidden=[256, 128], ReLU, dropout=0.2` | 235,146 | Adam | `1e-3` | 64 | 20 |
| 3 | **Trục 1: Hidden** | `mlp_h128` | `hidden=[128]` | 101,770 | Adam | `1e-3` | 64 | 20 |
| 4 | **Trục 1: Hidden** | `mlp_h512_256_128` | `hidden=[512, 256, 128]` | 567,434 | Adam | `1e-3` | 64 | 20 |
| 5 | **Trục 2: LR** | `mlp_lr1e4` | `lr=1e-4` | 235,146 | Adam | `1e-4` | 64 | 20 |
| 6 | **Trục 2: LR** | `mlp_lr1e2` | `lr=1e-2` | 235,146 | Adam | `1e-2` | 64 | 20 |
| 7 | **Trục 3: Optimizer** | `mlp_opt_sgd` | `SGD(momentum=0.9)` | 235,146 | SGD | `1e-3` | 64 | 20 |
| 8 | **Trục 3: Optimizer** | `mlp_opt_adamw` | `AdamW(weight_decay=1e-2)` | 235,146 | AdamW | `1e-3` | 64 | 20 |
| 9 | **Trục 4: Batch Size** | `mlp_bs32` | `batch_size=32` | 235,146 | Adam | `1e-3` | 32 | 20 |
| 10 | **Trục 4: Batch Size** | `mlp_bs256` | `batch_size=256` | 235,146 | Adam | `1e-3` | 256 | 20 |
| 11 | **Trục 5: Epoch** | `mlp_ep10` | `epochs=10` | 235,146 | Adam | `1e-3` | 64 | 10 |
| 12 | **Trục 5: Epoch** | `mlp_ep30` | `epochs=30` | 235,146 | Adam | `1e-3` | 64 | 30 |

- **Quy tắc dừng sớm chuỗi thí nghiệm:** Sau run 1 (`baseline_logistic`), hễ một run MLP nào đạt `val_acc >= TARGET_MLP_VAL_ACC` (`0.89`) **và** `val_f1 >= TARGET_MLP_VAL_F1` (`0.89`), chuỗi thí nghiệm **dừng ngay lập tức** (Kỹ sư có thể chỉnh ngưỡng qua `--target-acc` / `--target-f1`, hoặc dùng `--no-stop-on-target` nếu muốn chạy trọn cả 12 runs để đủ bảng ablation 5 trục).

---

## Work Breakdown

Thực thi tuần tự từng bước, mỗi bước kết thúc bằng **Self Audit** (`EXECUTION.md`):

- [x] **Step 1 — Tài liệu thiết kế Pipeline & Models (T1):**
  - Viết `labs/lab1_v2/docs/pipelines/modeling.md` bao phủ đủ 10 lớp của `MODEL_PIPELINE.md`, bảng biến đổi Tensor Shape xuyên suốt pipeline (`(1,28,28) → (B,1,28,28) → (B,784) → (B,H) → (B,10)`), gộp và gỡ bỏ stub `labs/lab1_v2/docs/PIPELINE_LAYERS.md`.
  - Viết `labs/lab1_v2/docs/models/logistic_regression.md` và `labs/lab1_v2/docs/models/mlp.md`.
- [x] **Step 2 — Config, Transforms & Data/EDA Module (T2, T3):**
  - Tạo `labs/lab1_v2/config.py` (kế thừa `src/config.py` qua `VibeConfig`, chứa đường dẫn `experiments/lab1_v2/`, ngưỡng dừng sớm `TARGET_MLP_VAL_ACC` / `TARGET_MLP_VAL_F1`, và danh sách 12 runs có thứ tự).
  - Tạo `src/vibe_scripts/lab1_v2/__init__.py`, `transforms.py` (train-fitted normalize, tensor contract checks) và `data.py` (nạp FashionMNIST, xác thực `manifest.json`, split Stratified `55k/5k/10k` `seed=248`, hàm xuất thống kê + biểu đồ EDA).
- [x] **Step 2B — Data Engineering Pipeline Thực Thi: Collection → Cleaning → EDA (6.1–6.3, Agent tự chạy):**
  - Bổ sung vào `src/vibe_scripts/lab1_v2/data.py` & tạo thin CLI `labs/lab1_v2/prepare_data.py`:
    - **6.1 Data Collection:** kiểm toán file raw/extracted + xác thực SHA-256 `manifest.json` → `experiments/lab1_v2/eda/collection_report.json`.
    - **6.2 Data Cleaning:** quét toàn bộ 70,000 ảnh (shape/dtype, pixel `[0,255]`, nhãn `{0..9}`, ảnh trống/hằng số, exact byte duplicates nội bộ Train/Val/Test, trùng chéo Train↔Val / Train↔Test và trùng ảnh nhưng khác nhãn) → `experiments/lab1_v2/eda/cleaning_report.json`.
    - **6.3 Exploratory Data Analysis (EDA):** phân bố lớp & imbalance ratio, thống kê pixel từng lớp, phát hiện outliers (IQR/Z-score), class prototypes 10 lớp + ma trận cosine similarity 10×10 → `experiments/lab1_v2/eda/{eda_summary.json, eda_class_distribution.png, eda_sample_grid.png, eda_pixel_distribution.png, eda_class_prototypes_similarity.png, eda_outliers_and_class_stats.png}`.
  - Agent chạy `python labs/lab1_v2/prepare_data.py` (KHÔNG chạy training), viết `labs/lab1_v2/docs/eda_report.md` và điền số liệu thật vào §2 của `experiment_report.md`.
- [x] **Step 3 — Model Modules: Logistic Regression & MLP (T4, T7):**
  - Tạo `src/vibe_scripts/lab1_v2/logistic.py` (`LogisticRegressionModel`, `7,850` params, raw logits, Fail-Fast tensor shape validation).
  - Tạo `src/vibe_scripts/lab1_v2/mlp.py` (`MLPModel` hỗ trợ `hidden_dims` tùy biến + `ReLU` + `Dropout`, `build_model()`, `count_parameters()`, Fail-Fast tensor shape validation).
- [x] **Step 4 — Training (có `tqdm` bar log + Per-Run Eval + Early-Stop Sweep), Evaluation, Inference Core & CLI Scripts (T5, T6, T8):**
  - Tạo `src/vibe_scripts/lab1_v2/evaluator.py`:
    - `evaluate_run_dir(run_dir, ...)` — chạy ngay sau khi 1 run train xong, lưu kết quả eval (Loss, Accuracy, Macro-F1, Per-class metrics, Confusion Matrix 10×10, Save/Load parity `< 1e-6`, `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `predicted_vs_actual.png`) trực tiếp vào `experiments/lab1_v2/runs/<run_id>/eval/`.
    - `evaluate_final_test(...)` — chạy đánh giá cuối đúng 1 lần trên tập Test (10k) so sánh `baseline_logistic` vs `best_mlp`, phân tích lỗi & xuất biểu đồ tổng hợp vào `experiments/lab1_v2/evaluation/` & `figures/`.
  - Tạo `src/vibe_scripts/lab1_v2/trainer.py`:
    - Global seeding, sanity checks (initial loss ≈ `ln 10`, overfit 1 batch), vòng lặp train/val có **`tqdm` progress bar logging** (bar ngoài cho runs + bar trong cho epochs/batches hiển thị live loss/acc/lr), atomic save `best.pt`/`last.pt`/`historys/history.{json,csv}`/`config.json`.
    - Vòng điều phối 12 runs: **Train run `i` → gọi ngay `evaluate_run_dir()` lưu vào `runs/<run_id>/eval/` → cập nhật `ablation_summary.json/.csv` → kiểm tra điều kiện evaluation tốt để kết thúc sớm chuỗi thí nghiệm**.
  - Tạo `src/vibe_scripts/lab1_v2/inference.py` (nạp checkpoint tự tái dựng model + `norm_stats`, suy luận đơn ảnh/batch dưới `@torch.inference_mode()`, Top-3 Softmax, đo latency).
  - Tạo thin CLI entrypoints `labs/lab1_v2/train.py` (hỗ trợ `--run-all`, `--run-id`, `--target-acc`, `--target-f1`, `--no-stop-on-target`, `--eda-only`), `evaluate.py`, `inference.py`, `labs/lab1_v2/README.md`, khung `labs/lab1_v2/docs/experiment_report.md` (chừa `[TBD]` cho số liệu thật) và `labs/lab1_v2/experiment.ipynb` (chế độ executable report đọc artifacts).
- [x] **Step 5 — Dual Unit Tests trên dữ liệu giả/nhỏ (KHÔNG chạy training thật):**
  - Viết `test/test_lab1_v2_data.py`, `test/test_lab1_v2_models.py`, `test/test_lab1_v2_pipeline.py` (đủ `@pytest.mark.smoke` và Normal Tests trong `tmp_path` với `≤ 64` mẫu: tensor shapes, split 55k/5k/10k, sanity check initial loss + overfit 1 batch, per-run eval xuất đúng vào `runs/<run_id>/eval/`, Save/Load `< 1e-6`, cơ chế dừng sớm khi đạt target eval, `tqdm` bar không lỗi).
  - Chạy `pytest -m smoke test/test_lab1_v2_*.py` và `pytest test/test_lab1_v2_*.py` pass 100%.
- [x] **Step 6 — [ENGINEER EXECUTION GATE] Bàn giao CLI cho Kỹ sư tự chạy Training & Evaluation:**
  - **Agent STOP — không chạy `train.py` thật.**
  - Cung cấp lệnh CLI chính xác để Kỹ sư trực tiếp chạy:
    1. `python labs/lab1_v2/train.py --run-all` *(có `tqdm` bar log; tự động eval ngay vào `runs/<run_id>/eval/` sau mỗi run và dừng sớm khi MLP đạt ngưỡng tốt)*
    2. `python labs/lab1_v2/evaluate.py` *(đánh giá Test 10k 1 lần cuối cho Baseline vs Best MLP + sinh biểu đồ so sánh tổng hợp)*
    3. `python labs/lab1_v2/inference.py --checkpoint <best_mlp_ckpt>` *(kiểm chứng CLI suy luận)*
- [x] **Step 7 — Hoàn thiện Báo cáo & Chạy Executable Report Notebook (sau khi Kỹ sư chạy xong Step 6):**
  - Đọc toàn bộ số liệu & biểu đồ thật trong `experiments/lab1_v2/` do Kỹ sư tạo ra.
  - Điền 100% số liệu thật vào `labs/lab1_v2/docs/experiment_report.md` (convergence, stability, overfitting, error analysis, so sánh Baseline vs MLP).
  - Thực thi `labs/lab1_v2/experiment.ipynb` ở chế độ đọc artifacts & live inference demo (không chạy train lại).
- [x] **Step 8 — Testing Report, Execution Logging & Knowledge Extraction (Phase 6–8):**
  - Lưu bằng chứng Dual Testing tại `.agents/tests/0005_lab1v2_test_report.md`.
  - Ghi Execution Log tại `.agents/logs/0005_lab1v2_log.md`.
  - Trích xuất tri thức tái sử dụng tại `.agents/knowledge_extraction/0005_lab1v2_knowledge.md`.

---

## Risks & Mitigations

| Rủi ro | Tác động | Biện pháp kiểm soát |
| :--- | :--- | :--- |
| **Data / Test Leakage** khi eval ngay từng run | Nếu eval trực tiếp trên Test 10k để quyết định dừng sớm sẽ gây rò rỉ Test | Per-run eval trong chuỗi 12 runs chạy trên **Val (5k)**; chỉ mở **Test (10k) đúng 1 lần cuối** ở `evaluate.py` cho Baseline vs Best MLP |
| **Dừng quá sớm** trước khi đủ dữ liệu phân tích 5 trục | Nếu `mlp_default` (run 2) đạt ngay ngưỡng target sẽ dừng sau 2 runs | Ngưỡng mặc định đặt hợp lý (`0.89`), đồng thời cung cấp cờ CLI `--target-acc` / `--no-stop-on-target` để Kỹ sư toàn quyền điều chỉnh |
| **Windows multiprocessing DataLoader / tqdm vỡ dòng** | Treo worker trên Windows hoặc thanh `tqdm` in rối | Dùng tensor dataset trong RAM (`num_workers=0`) an toàn trên Windows và cấu hình `tqdm(..., dynamic_ncols=True, leave=...)` |
| **Matplotlib GUI backend** treo khi CLI lưu ảnh | Treo tiến trình sau khi train xong run 1 | Ép `matplotlib.use("Agg")` trong CLI và `plt.close(fig)` sau mỗi biểu đồ |

---

## Validation

1. **Smoke Test (`pytest -m smoke test/test_lab1_v2_*.py -v`):** import sạch, config load OK, forward pass Logistic & MLP đúng shape `(B, 10)`, chạy 1 epoch trên subset nhỏ trong `tmp_path` tạo đủ `runs/<run_id>/eval/*`, inference không exception.
2. **Normal Test (`pytest test/test_lab1_v2_*.py -v`):**
   - Split 55k/5k/10k không giao nhau, phân tầng đều 10 lớp, chuẩn hóa Train ~`mean=0, std=1`.
   - Đếm tham số chính xác (`7,850` Logistic; công thức đúng cho mọi `hidden_dims` của MLP), sai shape ném `ValueError`.
   - Sanity checks: initial loss `∈ [2.1, 2.5]` (`≈ ln 10`), overfit 1 batch ≥ `95%`, Save/Load max logits diff `< 1e-6`, kiểm chứng cơ chế **per-run eval** & **early-stop sweep** khi đạt target eval.
3. **Acceptance Criteria Thực Nghiệm (sau khi Kỹ sư chạy Step 6):**
   - Baseline Logistic Regression hội tụ ổn định làm mốc tham chiếu thực nghiệm.
   - Best MLP tạo **gap cải thiện dương rõ rệt so với Baseline** cả Accuracy ($\Delta\text{Acc} > 0$) lẫn Macro-F1 ($\Delta\text{Macro-F1} > 0$) và giảm Test Loss.
   - Đủ 4 CLI scripts (`prepare_data.py`, `train.py`, `evaluate.py`, `inference.py`) chạy sạch, `experiment.ipynb` chạy sạch 100% cells (không train lại), báo cáo & biểu đồ đầy đủ.

---

## Deliverables

- **Documentation (`labs/lab1_v2/docs/`):** `README.md`, `pipelines/modeling.md`, `models/logistic_regression.md`, `models/mlp.md`, `eda_report.md`, `experiment_report.md`, `evaluation_report.md`
- **Reusable Core Modules (`src/vibe_scripts/lab1_v2/`):** `README.md`, `__init__.py`, `transforms.py`, `data.py`, `logistic.py`, `mlp.py`, `trainer.py`, `evaluator.py`, `inference.py`
- **Config, Thin CLIs, Notebook & Module README (`labs/lab1_v2/`):** `README.md`, `config.py`, `prepare_data.py`, `train.py`, `evaluate.py`, `inference.py`, `experiment.ipynb`
- **Tests (`test/`):** `test/test_lab1_v2_data.py`, `test/test_lab1_v2_models.py`, `test/test_lab1_v2_pipeline.py`
- **Artifacts (`experiments/lab1_v2/`):**
  - `README.md`, `eda/*` *(sinh bởi `prepare_data.py`)*
  - `runs/<run_id>/{best.pt, last.pt, historys/history.{json,csv}, config.json, eval/*}` *(và `test_eval/*` chỉ cho 2 model chung kết)*
  - `ablation/{ablation_summary.json, ablation_table.csv}`
  - `evaluation/*`, `figures/*`
- **Workflow Records (`.agents/`):** `.agents/tests/0005_lab1v2_test_report.md`, `.agents/logs/0005_lab1v2_log.md`, `.agents/knowledge_extraction/0005_lab1v2_knowledge.md`

---

## Estimated Impact

- **Tạo mới / sửa đổi:**
  - `src/vibe_scripts/lab1_v2/*` (tạo mới)
  - `labs/lab1_v2/*` (tạo mới; gộp và xóa file nháp 5 dòng `PIPELINE_LAYERS.md`)
  - `test/test_lab1_v2_*.py` (tạo mới)
  - `.agents/{plans,tests,logs,knowledge_extraction}/0005_lab1v2_*` (tạo mới)
  - `experiments/lab1_v2/*` (`eda/` từ `prepare_data.py`; `runs/`, `ablation/`, `evaluation/`, `figures/` từ Kỹ sư chạy Step 6)
- **Không đụng tới:**
  - Toàn bộ `src/*.py` gốc, `src/vibe_scripts/lab1/*`, `src/vibe_scripts/lab2/*`
  - `labs/lab1/main/*`, `labs/lab1/vibe/*`, `labs/lab1/submit/*`, `labs/lab2/*`, `labs/lab3/*`
  - `experiments/lab1/*`, `experiments/lab2/*`, `data/FashionMNIST/raw/*`, `requirements.txt`, `.gitignore`, `README.md` gốc

---

## Approval Gate (Bắt buộc)

- **Status:** `approved` (Phương án A: per-run Val eval + dừng sớm trên Val → Final Test 10k duy nhất 1 lần cho Baseline & Best MLP; Agent thực thi Step 1 → Step 5 và **STOP tại Step 6** để Kỹ sư tự chạy training).

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| `1.7.0` | 2026-11-26 | notworLe | Antigravity | Hoàn tất **Step 8 (Phase 6–8)**: tạo `.agents/tests/0005_lab1v2_test_report.md`, `.agents/logs/0005_lab1v2_log.md`, `.agents/knowledge_extraction/0005_lab1v2_knowledge.md`; cập nhật diễn đạt `Label Noise / Class Overlap` & nhúng `<details>` 3 Audit Sheets trong `evaluation_report.md`; chuyển trạng thái Plan sang `completed` |
| `1.6.0` | 2026-11-26 | notworLe | Antigravity | Tách triệt để SRP theo Layer (`experiment_report.md` `v2.1.0` chỉ giữ Layer 4 & 7; `evaluation_report.md` `v1.2.0` gom trọn Layer 8–10); nâng cấp Error Analysis 4 bước định lượng (`evaluate.py --error-analysis`: CM chuẩn hóa %, Error Gallery Top-3 High/Low Conf, gán nhãn `N=60` ảnh thật `48.3%/38.3%/13.3%`, Slice Analysis); đồng bộ `modeling.md` (`v2.2.0`) và tất cả tham chiếu (`199/199` links pass, `13/13` pytest pass) |
| `1.5.0` | 2026-11-26 | notworLe | Antigravity | Bổ sung hỗ trợ `--resume` + `optimizer_state_dict` và chạy bổ sung `mlp_opt_sgd_ep40` (ep `21 → 40`); bổ sung §8.1 *Tiêu Chí Đánh Giá & Chọn Mô Hình* + Output tường minh các tầng vào `modeling.md` (`v2.1.0`); tách `experiment_report.md` (`v2.0.0`, ablation & training diagnostics trên Val) khỏi `evaluation_report.md` (`v1.0.0`, Final Test 10k, Error Analysis & Inference) và đồng bộ toàn bộ tham chiếu |
| `1.4.0` | 2026-11-25 | notworLe | Antigravity | Đồng bộ tên dự án `labs/lab1_v2/`, thống nhất log về `runs/<id>/historys/history.{json,csv}`, bỏ vẽ lặp curve vào `test_eval/`, bổ sung đủ 4 module `README.md`, đánh dấu hoàn tất Step 6–7 (`12/12` runs, `experiment_report.md` v1.1.0 tách Error Analysis & Training Diagnostics, chạy sạch `experiment.ipynb`) |
| `1.3.0` | 2026-11-25 | notworLe | Antigravity | Bổ sung **Step 2B** tách rõ **6.1 Collection → 6.2 Cleaning → 6.3 EDA** (`prepare_data.py`, `eda_report.md`), bỏ mốc Accuracy suy đoán trước thực nghiệm sang đo **gap tương đối so với Baseline**, gỡ logic EDA khỏi `train.py` (SRP) và bổ sung đầy đủ Training Monitoring (`loss/acc/lr/sps/vram`) |
| `1.2.0` | 2026-11-25 | notworLe | Antigravity | Chốt **Phương án A** chống Test Leakage (per-run eval & early-stop trên Val 5k → Final Test 10k duy nhất 1 lần cho Baseline & Best MLP) và chuyển `status: approved` để bắt đầu thực thi Step 1 → Step 5 |
| `1.1.0` | 2026-11-25 | notworLe | Antigravity | Bổ sung yêu cầu Kỹ sư: mỗi run train xong eval ngay lưu tại `runs/<id>/eval/`, dừng sớm chuỗi khi MLP đạt evaluation tốt, **Hard Constraint** Agent không tự chạy training thật (Kỹ sư chạy với `tqdm` bar log) |
| `1.0.0` | 2026-11-25 | notworLe | Antigravity | Khởi tạo Plan ban đầu cho `0005_lab1v2` (cô lập khỏi v1, split Stratified `55k/5k/10k`, ablation 5 trục, phân tách CLI scripts vs executable report notebook) |
