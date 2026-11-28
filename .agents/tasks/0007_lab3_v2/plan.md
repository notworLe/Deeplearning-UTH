---
title: Plan — Lab 3 v2 Get Started with Hugging Face (Sentiment Pipeline & DistilBERT Fine-Tuning on IMDB)
doc_type: Plan
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-28
updated: 2026-11-28
version: 1.3.0
status: completed
---

# Plan: Lab 3 v2 — Get Started with Hugging Face (`0007_lab3_v2`)

> **Source Task:** [`.agents/tasks/0007_lab3_v2/task.md`](task.md) · **Exercise:** [`labs/lab3/QUESTIOn.md`](../../../labs/lab3/QUESTIOn.md) · **Workflow:** `DL_RULE v1.4.0` (`MODEL_PIPELINE.md`, bỏ qua Layer 3 Project Reference theo `task.md`)

---

## Objective

Xây dựng mini-project Deep Learning gọn nhẹ, dễ hiểu và chuẩn kỹ thuật để làm chủ hệ sinh thái **Hugging Face (`transformers`, `datasets`, `evaluate`)** qua 2 bài thực hành:

1. **Exercise 1 (`ex1_sentiment.ipynb` + CLI):** Hiểu rõ luồng **`text → tokenizer (input_ids, attention_mask, decode) → model → logits / Softmax → nhãn cảm xúc`** bằng `pipeline` với `distilbert-base-uncased-finetuned-sst-2-english` trên bộ câu thử nghiệm đa dạng (tích cực, tiêu cực, phủ định kép & câu mỉa mai/khó).
2. **Exercise 2 (`ex2_finetune.ipynb` + CLI):** Fine-tune **`distilbert-base-uncased`** (`num_labels=2`) trên tập con cân bằng của **IMDB** bằng `TrainingArguments` + `Trainer` + `DataCollatorWithPadding`, đo định lượng **Baseline trước fine-tune vs Sau fine-tune** trên Validation, sau đó đánh giá **Test độc lập đúng 1 lần cuối**, trực quan hóa loss curve & phân tích các ca dự đoán sai vào **`REPORT.md`**.

---

## Understanding & Assumptions (Cần Kỹ sư Xác Nhận)

### 1. Hiện trạng môi trường & repo đã xác minh
- **Thư viện đã cài sẵn đầy đủ:** `torch 2.15.0+cu132` (**CUDA `True` — NVIDIA GeForce RTX 4060 Laptop GPU**), `transformers 5.23.0`, `datasets 5.2.0`, `evaluate 0.4.6`, `accelerate 1.16.0`, `tensorboard 2.21.0`, `sklearn 1.9.0`, `matplotlib 3.11.0`, `pandas 3.0.3`, `nbformat 5.10.4`, `pytest 9.1.1`. Không cần cài thêm thư viện nào.
- **Lưu ý API mới của `transformers 5.23.0` (đã xác minh bằng `inspect.signature`):**
  - `TrainingArguments` sử dụng **`eval_strategy="epoch"`** (`evaluation_strategy` đã bị loại bỏ).
  - `Trainer` sử dụng **`processing_class=tokenizer`** (`tokenizer=` đã bị loại bỏ).
- **Cache Hugging Face:** `~/.cache/huggingface/hub/` hiện chưa có `distilbert-*` và `imdb` → Kỹ sư chạy tải Hub & full fine-tune ở Step 4; toàn bộ unit test của Agent sẽ chạy **100% offline** bằng tiny `DistilBertConfig` + synthetic dataset/tokenizer trong `tmp_path`.
- **Cô lập hoàn toàn khỏi bản v1:** `labs/lab3/{main,docs,QUESTIOn.md,README.md}` đã tồn tại (bản cũ chạy trên SST-2) → bản v2 cô lập 100% tại **`src/vibe_scripts/lab3_v2/`** (sửa typo `vibe_scrtips` ở dòng 123 `task.md`), **`labs/lab3_v2/`**, **`experiments/lab3_v2/`** và **`test/test_lab3_v2_*.py`**.

### 2. Các quyết định đã được Kỹ sư chốt

| # | Hạng mục | Quyết định đã chốt |
| :-: | :--- | :--- |
| **A1** | **Quy mô tập con IMDB & chiến lược chia tập (`seed = 248`)** | Stratified subset 50/50 neg/pos, khóa cố định tại `experiments/lab3_v2/splits/split_indices.json`: **Train = `4,000`** (`2,000` neg / `2,000` pos từ IMDB `train`), **Val = `1,000`** (`500` neg / `500` pos tách rời từ IMDB `train`, giao Train `= ∅`), **Test = `1,000`** (`500` neg / `500` pos từ IMDB `test`, **chấm đúng 1 lần cuối**). |
| **A2** | **Siêu tham số Ex2 (`distilbert-base-uncased`)** | `max_length = 256` (truncation + dynamic padding qua `DataCollatorWithPadding`), `epochs = 3`, `learning_rate = 2e-5`, `per_device_train_batch_size = 16`, `per_device_eval_batch_size = 32`, `weight_decay = 0.01`, `eval_strategy = "epoch"`, `save_strategy = "epoch"`, `load_best_model_at_end = True`, `metric_for_best_model = "eval_loss"`, `fp16 = True` (trên CUDA), `report_to = ["tensorboard"]`, `seed = 248`. |
| **A3** | **Cấu trúc tài liệu báo cáo (SRP 3 báo cáo như Lab 2 v2)** | Tách đầy đủ **3 báo cáo đơn nhiệm SRP** tại `labs/lab3_v2/docs/`: `eda_report.md` (Layer 6), `experiment_report.md` (Layer 4 & 7), `evaluation_report.md` (Layer 8–10) kèm `docs/README.md`, `docs/pipelines/modeling.md`, `docs/models/distilbert.md` và một file tóm tắt điều hướng ngắn **`labs/lab3_v2/REPORT.md`** dẫn trực tiếp về 3 báo cáo SRP. |
| **A4** | **Baseline so sánh Ex2 (Đơn giản, trực diện, dễ hiểu)** | So sánh thẳng **2 mốc cốt lõi** trên cùng tập Validation (`1,000` mẫu): **Trước fine-tune** (`distilbert-base-uncased` khởi tạo mới `num_labels = 2`, random head) vs **Sau fine-tune** (`3 epochs` với HF `Trainer`), sau đó chốt best checkpoint (`min eval_loss`) chấm tập Test (`1,000` mẫu) đúng **1 lần duy nhất**. |

---

## Scope

- **In Scope:**
  - **Reusable Core (`src/vibe_scripts/lab3_v2/`):**
    - `__init__.py`, `README.md`
    - `ex1_sentiment.py`: wrapper sạch cho HF `pipeline("sentiment-analysis")`, hàm bóc tách & giải thích tokenizer (`tokens`, `input_ids`, `attention_mask`, `special_tokens`, `decode`), chạy batch phân tích cảm xúc các câu mẫu (tích cực, tiêu cực, phủ định, mỉa mai) và lưu JSON kết quả vào `experiments/lab3_v2/ex1/`.
    - `data.py`: tải IMDB qua `datasets.load_dataset("imdb")`, lấy stratified subset & tách Validation cố định `seed = 248` lưu `split_indices.json` (chống data/test leakage), kiểm tra thống kê cơ bản (phân bố lớp, độ dài review, tỷ lệ bị cắt tại `max_length = 256`), và tokenize với `truncation=True` + dynamic padding (`DataCollatorWithPadding`).
    - `ex2_finetune.py`: xây dựng `AutoModelForSequenceClassification` (`num_labels = 2`), hàm `compute_metrics` dùng `evaluate` (`accuracy`, `f1`, có fallback `sklearn` khi offline), đánh giá Baseline trước fine-tune trên Val, cấu hình `TrainingArguments` (`eval_strategy`, `fp16`, TensorBoard) + `Trainer(processing_class=tokenizer)` chuẩn `transformers 5.23.0`, lưu kết quả theo đúng cấu trúc `MODEL_PIPELINE.md v1.4.0` (`experiments/lab3_v2/runs/distilbert_imdb/{monitor,val_eval,test_eval,weights}`).
    - `analysis.py`: vẽ loss curve (`train_loss` vs `eval_loss`), biểu đồ cột so sánh Accuracy/F1 trước vs sau fine-tune, Confusion Matrix 2×2 trên Test (sau khi chấm Test 1 lần), trích xuất top ca dự đoán sai kèm review snippet & xác suất dự đoán.
  - **Config, Thin CLIs, Notebooks & Docs (`labs/lab3_v2/`):**
    - `config.py` (kế thừa `src/config.py` qua `VibeConfig`), `README.md`
    - Thin CLIs: `run_ex1.py`, `train_ex2.py` (chạy data split → baseline Val → `Trainer.train()` → Val eval sau fine-tune), `eval_ex2.py` (chấm Test **1 lần duy nhất**, vẽ biểu đồ & xuất error examples).
    - 2 Deliverables Notebooks: **`ex1_sentiment.ipynb`** và **`ex2_finetune.ipynb`** (executable notebooks đọc/hiển thị rõ luồng **text → tokenizer → model → kết quả** & kết quả từ `experiments/lab3_v2/`, hỗ trợ chạy trực tiếp từng bước Ex1, không tự retrain lại Ex2 khi đã có artifacts).
    - Báo cáo **`labs/lab3_v2/REPORT.md`** (khung chờ số liệu thật ở Step 2, điền 100% số liệu thật sau Step 4).
  - **Offline Dual Tests (`test/`):** `test_lab3_v2_ex1.py`, `test_lab3_v2_ex2.py` (Smoke + Normal, 100% offline, không cần internet).
  - **Workflow Closure (`.agents/`):** `.agents/{tests,logs,knowledge_extraction}/0007_lab3_v2_*`.
- **Out of Scope:**
  - **Hard Constraint:** Agent **không tự chạy full fine-tune Ex2** (Kỹ sư trực tiếp chạy CLI ở Step 4); Agent chỉ chạy smoke test vài batch trên tiny model/synthetic data.
  - Không tune hyperparameter theo tập Test; không bịa số metric trước khi chạy thật.
  - Không thêm thư viện ngoài bộ chuẩn đã cài (`transformers`, `datasets`, `evaluate`, `torch`, cùng `sklearn`/`matplotlib`/`pytest`/`nbformat` có sẵn trong repo).
  - Không chỉnh sửa `labs/lab3/` (bản v1), `labs/lab1*`, `labs/lab2*`, `src/*.py` gốc, `src/vibe_scripts/{lab1*,lab2*}`, `experiments/{lab1*,lab2*}`, `.gitignore`.

---

## Work Breakdown

Thực thi tuần tự từng bước kèm **Self Audit** (`EXECUTION.md`):

- [x] **Step 1 — Core Modules, Config & Thin CLIs (`T1–T7` code):**
  - Tạo `labs/lab3_v2/config.py` (`VibeConfig`), `src/vibe_scripts/lab3_v2/{__init__.py, README.md, ex1_sentiment.py, data.py, ex2_finetune.py, analysis.py}` và 3 thin CLIs `labs/lab3_v2/{run_ex1.py, train_ex2.py, eval_ex2.py}`.
  - Tuân thủ tuyệt đối API `transformers 5.23.0` (`text-classification`, `eval_strategy`, `processing_class`, không dùng `logging_dir`), cấu trúc run `MODEL_PIPELINE.md v1.4.0` (`monitor/`, `val_eval/`, `test_eval/`, `weights/`), guard Matplotlib `Agg` trên Windows và Tensor Contract `(B, L) → (B, 2)`.
- [x] **Step 2 — Notebooks (`ex1_sentiment.ipynb`, `ex2_finetune.ipynb`) & Khung 3 Báo Cáo SRP + `REPORT.md`:**
  - Tạo 2 notebook trực quan, giàu giải thích cơ chế tokenizer (`[CLS]`, `[SEP]`, subword `##`, `attention_mask`, `decode`) và vòng lặp `Trainer` (`TrainingArguments`, `DataCollatorWithPadding`, `compute_metrics`), kèm bộ thiết kế `labs/lab3_v2/docs/{README.md, pipelines/modeling.md, models/distilbert.md}`, khung 3 báo cáo SRP (`eda_report.md`, `experiment_report.md`, `evaluation_report.md`), `labs/lab3_v2/REPORT.md` và `labs/lab3_v2/README.md`, `experiments/lab3_v2/README.md`.
- [x] **Step 3 — Dual Offline Tests (`pytest -m smoke` & Full `pytest`, KHÔNG chạy full fine-tune):**
  - Viết `test/test_lab3_v2_ex1.py` và `test/test_lab3_v2_ex2.py` chạy 100% offline bằng tiny `DistilBertConfig` + `BertTokenizer(vocab=...)` + synthetic binary text dataset trong `tmp_path`.
  - Xác minh: Smoke `2/2` PASS (`12.37s`), Full Normal `6/6` PASS (`14.74s`), `3/3` CLI `--help` PASS, `53/53` doc links PASS (`0` broken).
- [x] **Step 4 — [ENGINEER EXECUTION GATE] Thực thi Ex1 & Ex2 Fine-Tune + Held-Out Test:**
  - Theo chỉ đạo ủy quyền trực tiếp của Kỹ sư, đã chạy trọn vẹn `run_ex1.py` (`5/8` khớp kỳ vọng, `14` tokens, batch padding `(2, 11)`), `train_ex2.py` (`91.41s` trên RTX 4060 `fp16`, Val Acc `49.70% → 87.80%`, `Val Loss 0.6957 → 0.2764`, TensorBoard active) và `eval_ex2.py` 1 lần duy nhất (`89.10%` Test Acc / `89.26%` Test F1 / `0.2518` Test Loss, truncation slice `14.46%` vs `8.45%`) + live `--predict` (`3/3` đúng).
- [x] **Step 5 — Hoàn thiện 3 Báo Cáo SRP, `REPORT.md` & Chạy sạch 2 Notebooks (`T8`):**
  - Đã điền 100% số liệu thật vào `labs/lab3_v2/docs/{eda_report,experiment_report,evaluation_report}.md` (`v1.0.0`) và `labs/lab3_v2/REPORT.md` (`v1.0.0`), cập nhật `docs/README.md`.
  - Đã thực thi sạch `ex1_sentiment.ipynb` (`4/4` code cells) và `ex2_finetune.ipynb` (`5/5` code cells, live inference OK), pass `nbformat.validate` và `6/6` pytest (0 warnings).
- [x] **Step 6 — Testing Report, Execution Log & Knowledge Extraction (Phase 6–8):**
  - Đã lưu hoàn chỉnh [`.agents/tests/0007_lab3_v2_test_report.md`](../../tests/0007_lab3_v2_test_report.md), [`.agents/logs/0007_lab3_v2_log.md`](../../logs/0007_lab3_v2_log.md) và [`.agents/knowledge_extraction/0007_lab3_v2_knowledge.md`](../../knowledge_extraction/0007_lab3_v2_knowledge.md).

---

## Risks & Mitigations

| Rủi ro | Tác động | Biện pháp kiểm soát |
| :--- | :--- | :--- |
| **Breaking API changes của `transformers 5.23.0`** | `TypeError` nếu dùng `evaluation_strategy` hoặc `Trainer(tokenizer=...)` | Đã kiểm chứng trực tiếp bằng `inspect.signature`; dùng chuẩn `eval_strategy` và `processing_class=tokenizer` |
| **Unit tests bị lỗi mạng trong sandbox** | `pytest` fail khi gọi `from_pretrained` / `load_dataset` / `evaluate.load` | Toàn bộ unit test chạy 100% offline bằng tiny `DistilBertConfig` + local WordPiece vocab + synthetic dataset trong `tmp_path` + fallback `sklearn` trong `compute_metrics` |
| **Checkpoint HF (`model.safetensors` ~268 MB) lọt vào Git** | Repo phình nặng, không thể push GitHub | Lưu weights trong `experiments/lab3_v2/runs/<run>/weights/` (bị chặn bởi `runs/` trong `.gitignore`), tuyệt đối không `git add -f` thư mục `weights/` hay file `*.safetensors`/`*.bin`/`*.pt` |
| **Data/Test Leakage hoặc Tune theo Test** | Kết quả không trung thực | Khóa cố định `split_indices.json` (`seed=248`), kiểm chứng giao tập rỗng bằng unit test; `train_ex2.py` tuyệt đối không chạm tập Test; `eval_ex2.py` chỉ chạy 1 lần cuối sau khi Val đạt |
| **Review IMDB dài hơn `max_length = 256` tokens** | Mất phần kết luận cuối review dẫn tới dự đoán sai | Đo tỷ lệ review vượt `256` tokens ở bước EDA/data audit và đối chiếu trực tiếp trong phần Error Analysis của `REPORT.md` |

---

## Validation

1. **Smoke Test (`pytest -m smoke test/test_lab3_v2_ex1.py test/test_lab3_v2_ex2.py -v`):** config load, tokenizer demo, forward pass `(B, L) → (B, 2)` và 1-epoch mini `Trainer` (vài batch) chạy sạch hoàn toàn offline.
2. **Normal Test (`pytest test/test_lab3_v2_ex1.py test/test_lab3_v2_ex2.py -v`):** stratified split disjointness, dynamic padding, baseline vs post-finetune Val eval, Test eval 1 lần, error extraction + chart generation, `nbformat.validate` cho 2 notebooks.
3. **Acceptance Criteria Thực Nghiệm (sau Step 4):**
   - Ex1 chạy đúng, hiển thị rõ `tokens`, `input_ids`, `attention_mask`, `decode` và kết quả sentiment trên các câu dễ lẫn câu khó/mỉa mai.
   - Ex2 fine-tune chạy hết không lỗi, `train_loss` & `eval_loss` giảm rõ rệt.
   - **Val Accuracy sau fine-tune > Val Accuracy trước fine-tune** một cách rõ rệt.
   - Test chỉ chấm đúng **1 lần cuối** sau khi xong Validation.
   - `REPORT.md` giải thích rõ luồng tokenizer, kết quả trước/sau fine-tune và phân tích cụ thể các ca dự đoán sai.

---

## Deliverables

- **Core Code (`src/vibe_scripts/lab3_v2/`):** `README.md`, `__init__.py`, `ex1_sentiment.py`, `data.py`, `ex2_finetune.py`, `analysis.py`
- **Config, CLIs, Notebooks & Report (`labs/lab3_v2/`):** `README.md`, `config.py`, `run_ex1.py`, `train_ex2.py`, `eval_ex2.py`, `ex1_sentiment.ipynb`, `ex2_finetune.ipynb`, `REPORT.md`
- **Tests (`test/`):** `test_lab3_v2_ex1.py`, `test_lab3_v2_ex2.py`
- **Artifacts (`experiments/lab3_v2/`):** `README.md`, `ex1/*`, `splits/split_indices.json`, `runs/distilbert_imdb/{monitor,val_eval,test_eval,weights}`
- **Workflow Records (`.agents/`):** `.agents/tasks/0007_lab3_v2/plan.md`, `.agents/{tests,logs,knowledge_extraction}/0007_lab3_v2_*`

---

## Estimated Impact

- **Tạo mới:** `src/vibe_scripts/lab3_v2/*`, `labs/lab3_v2/*`, `test/test_lab3_v2_*.py`, `experiments/lab3_v2/*`, `.agents/tasks/0007_lab3_v2/plan.md`, `.agents/{tests,logs,knowledge_extraction}/0007_lab3_v2_*`.
- **Không đụng tới:** Toàn bộ `labs/lab3/{main,docs,QUESTIOn.md,README.md}` (v1), `src/*.py` gốc, `labs/lab1*`, `labs/lab2*`, `src/vibe_scripts/{lab1*,lab2*}`, `experiments/{lab1*,lab2*}`, `data/`, `.gitignore`, `README.md` gốc.

---

## Approval Gate (Bắt buộc)

- **Status:** `approved` (Kỹ sư đã xác nhận: **A1** `Train 4,000 / Val 1,000 / Test 1,000` stratified `seed = 248`; **A2** `distilbert-base-uncased`, `max_length = 256`, `3 ep`, `lr = 2e-5`, `BS = 16`, `fp16`; **A3** tách đầy đủ 3 báo cáo SRP + `REPORT.md` điều hướng như Lab 2 v2; **A4** so sánh trực diện Trước vs Sau fine-tune đơn giản dễ hiểu. Agent triển khai Step 1 → Step 3 và **STOP tại Step 4** bàn giao CLI cho Kỹ sư tự chạy Ex1 & Ex2 full fine-tune).

---

## Lịch Sử Thay Đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| `1.3.0` | 2026-11-28 | notworLe | Antigravity | Hoàn tất **Steps 4–6** (chạy Ex1 + Ex2 fine-tune `91.4s` + Held-Out Test `89.10%` theo ủy quyền Kỹ sư, hoàn thiện 3 báo cáo SRP + `REPORT.md` + 2 notebooks, lập đủ hồ sơ Phase 6–8 `0007_lab3_v2_{test_report,log,knowledge}.md`, `6/6` pytest PASS 0 warnings) và chuyển `status: completed` |
| `1.2.0` | 2026-11-28 | notworLe | Antigravity | Hoàn tất **Steps 1–3** (core modules chuẩn `transformers 5.23.0`, 3 thin CLIs, 2 executable notebooks, bộ thiết kế + khung 3 báo cáo SRP & `REPORT.md`; Smoke `2/2` PASS, Full Normal `6/6` PASS, `53/53` links PASS) — **STOP tại Step 4** bàn giao CLI cho Kỹ sư tự chạy Ex1 & Ex2 full fine-tune |
| `1.1.0` | 2026-11-28 | notworLe | Antigravity | Chốt 4 quyết định A1–A4 từ Kỹ sư (`4,000 / 1,000 / 1,000`, chuẩn 3 báo cáo SRP + `REPORT.md`, so sánh trực diện Trước vs Sau fine-tune) và chuyển `status: approved` |
| `1.0.0` | 2026-11-28 | notworLe | Antigravity | Khởi tạo Plan cho `0007_lab3_v2` tại `.agents/tasks/0007_lab3_v2/plan.md`, kiểm chứng API `transformers 5.23.0` (`eval_strategy`, `processing_class`) |
