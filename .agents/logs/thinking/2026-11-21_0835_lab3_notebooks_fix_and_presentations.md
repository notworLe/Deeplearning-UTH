# AI Thinking Log — Lab 03 Notebooks Execution & Presentation Docs (`labs/lab3/docs/`)

## 1. Metadata
- **Date / Time**: 2026-11-21 08:35
- **Task Slug**: `lab3_notebooks_fix_and_presentations`
- **Agent Role**: Principal Deep-Learning Engineer & Presentation Architect
- **Status**: `Completed`
- **Plan Reference**: [`prompts/0004_lab3/plans/README.md`](../../prompts/0004_lab3/plans/README.md)
- **Error Log Reference**: [`logs/errors/ERR-002_lab3_missing_hf_dependencies.md`](../errors/ERR-002_lab3_missing_hf_dependencies.md)

---

## 2. Context & Reasoning Process (Luồng Tư Duy Kỹ Thuật)

### 2.1. Chẩn đoán Nguyên nhân Gốc rễ (Root Cause Analysis — `ERR-002`)
1. **Thiếu gói phụ thuộc Hugging Face trong venv `anything`**:
   - Venv `anything` (`Python 3.13.14`, `torch 2.15.0+cu132`, `transformers 5.23.0`) thiếu 3 gói bắt buộc để thực thi Exercise 2: `datasets`, `evaluate`, và `accelerate>=1.1.0`.
   - Đã ghi nhận sự cố `ERR-002` tại [`logs/errors/ERR-002_lab3_missing_hf_dependencies.md`](../errors/ERR-002_lab3_missing_hf_dependencies.md) và cài đặt `datasets-5.2.0`, `evaluate-0.4.6`, `accelerate-1.16.0` sau khi Kỹ sư xác nhận.
2. **Xung đột `dataloader_num_workers > 0` trên Jupyter Windows**:
   - Trong `02_ex2_finetune_sentiment.ipynb` (`dataloader_num_workers=2`) và `fine-tune.py` (`dataloader_num_workers=8`), tiến trình con trên Windows sử dụng cơ chế `spawn` gây treo/lỗi khi chạy trong interactive kernel. Đã chuyển về `dataloader_num_workers=0` để chạy ổn định 100% trên Windows.
3. **Chuẩn hóa Cấu trúc Cell 0 & Cell 1 và Bảo mật Đường dẫn (`AI_REFERENCE.md` §5)**:
   - Tách rõ Cell 0 (Header & Mục tiêu) và Cell 1 (Bảng Roadmap 4 cột `| Step | Description | What it does | Import path |` + `Related Documentation References`).
   - Thay thế toàn bộ liên kết chết (`PYTORCH_FRAMEWORK_RULES.md`, `NOTEBOOK_HEADER_CONVENTION.md`, `RESULTS_REPORTING.md`, `docs/lab3/PIPELINE_EXPLANATION.md`) bằng các liên kết tương đối hợp lệ 100% tới [`labs/lab3/docs/PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md`](../../labs/lab3/docs/PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md), [`labs/lab3/docs/PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md`](../../labs/lab3/docs/PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md), [`labs/lab3/docs/README.md`](../../labs/lab3/docs/README.md), [`labs/lab3/QUESTIOn.md`](../../labs/lab3/QUESTIOn.md), và [`.agents/rules/AI_REFERENCE.md`](../../.agents/rules/AI_REFERENCE.md).
   - Bổ sung `os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"` và `os.environ["TOKENIZERS_PARALLELISM"] = "false"` ở Step 1 của cả 2 bài để ngăn cảnh báo symlink của Hugging Face Hub làm lộ đường dẫn người dùng `C:\Users\...` trong cell outputs.

### 2.2. Thực thi Thực tế & Số liệu Nghiệm thu
- **Exercise 1 ([`01_ex1_sentiment_analysis.ipynb`](../../labs/lab3/main/01_ex1_sentiment_analysis.ipynb) & [`main.ipynb`](../../labs/lab3/main/main.ipynb))**:
  - Chạy thực thi toàn bộ 6 code cells trên GPU `NVIDIA GeForce RTX 4060 Laptop GPU` với mô hình `distilbert-base-uncased-finetuned-sst-2-english`, nhúng đầy đủ output của high-level pipeline, WordPiece tokenization + dynamic padding tensor `[4, 15]`, manual PyTorch forward pass + Softmax + biểu đồ thanh ngang phân cực cảm xúc, và stress test cấu trúc phủ định.
- **Exercise 2 ([`02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb) & [`fine-tune.ipynb`](../../labs/lab3/main/fine-tune.ipynb))**:
  - Chạy thực thi trọn vẹn 8 code cells (bao gồm huấn luyện 3 epochs `distilbert-base-uncased` trên `67,349` mẫu train của GLUE SST-2 và đánh giá trên `872` mẫu validation):
    - **Epoch 1.0 (Best Checkpoint, Step 2105)**: `Val Loss = 0.2622`, `Val Accuracy = 90.94%`, `Val F1 = 90.99%`, `Macro-F1 = 90.94%`.
    - **Epoch 2.0**: `Val Loss = 0.3441`, `Val Accuracy = 89.11%`, `Val F1 = 89.82%`.
    - **Epoch 3.0**: `Val Loss = 0.3624`, `Val Accuracy = 90.02%`, `Val F1 = 90.47%`.
    - **Error Diagnostics**: `79 / 872` mẫu sai (`9.06%`), gồm `34` False Positives và `45` False Negatives.
    - Nhúng đầy đủ 3 biểu đồ trực quan hóa (EDA Label & Length Histogram, Confusion Matrix, Error Diagnostics by Length & Confidence).

---

## 3. Deliverables & Handover (Tệp Bàn Giao)
1. **Notebooks Đã Sửa & Chạy Thực Thi (`labs/lab3/main/`)**:
   - [`labs/lab3/main/01_ex1_sentiment_analysis.ipynb`](../../labs/lab3/main/01_ex1_sentiment_analysis.ipynb)
   - [`labs/lab3/main/main.ipynb`](../../labs/lab3/main/main.ipynb)
   - [`labs/lab3/main/02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb)
   - [`labs/lab3/main/fine-tune.ipynb`](../../labs/lab3/main/fine-tune.ipynb)
   - [`labs/lab3/main/fine-tune.py`](../../labs/lab3/main/fine-tune.py)
2. **Tài Liệu Thuyết Trình & Biểu Diễn Từng Notebook (`labs/lab3/docs/`)**:
   - [`labs/lab3/docs/README.md`](../../labs/lab3/docs/README.md)
   - [`labs/lab3/docs/PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md`](../../labs/lab3/docs/PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md)
   - [`labs/lab3/docs/PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md`](../../labs/lab3/docs/PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md)
