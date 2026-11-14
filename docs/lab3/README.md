# Lab 3: Deep Learning NLP Pipeline & Transformer Fine-Tuning

- **Motivation/Background**: Natural Language Processing (NLP) has undergone a fundamental paradigm shift from static word embeddings (Word2Vec, GloVe) and recurrent networks (LSTM, GRU) to self-attention Transformer architectures. Understanding the end-to-end engineering pipeline—from raw text ingestion, subword tokenization, dynamic batch collation, to transfer learning fine-tuning and error diagnostics—is essential for reproducible deep learning development.
- **Purpose**: Serve as the authoritative index and pedagogical architectural documentation for Lab 3 (Hugging Face Transformers, Pre-trained Sentiment Analysis, and DistilBERT Fine-tuning on SST-2).
- **Overview Pipeline**: Synthesizes the end-to-end NLP Deep Learning pipeline following [docs/reference/ML_PIPELINE.md](../reference/ML_PIPELINE.md) and [agents/rules/EXPLAIN_PIPELINE.md](../../.agents/rules/EXPLAIN_PIPELINE.md), decomposing each stage into Input, Output, Đặc trưng, Mục tiêu, Hạn chế, and Giải pháp.
- **Detailed Plan**: §1 Overview & Research Problem; §2 Pipeline Architecture & File Mapping; §3 Artifacts & Checkpoint Index; §4 Exercise Navigation (Part 1 & Part 2); §5 Empirical Results Summary (5W1H).
- **References**: [Hugging Face Transformers](https://huggingface.co/docs/transformers), [nyu-mll/glue (SST-2)](https://huggingface.co/datasets/nyu-mll/glue), [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased), [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), [agents/rules/AGENT_AI.md](../../.agents/rules/AGENT_AI.md), [agents/rules/NOTEBOOK_HEADER_CONVENTION.md](../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md), [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md), [agents/rules/PYTORCH_FRAMEWORK_RULES.md](../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md).
- **Created**: 2026-11-14T08:35:00+07:00
- **Last Updated**: 2026-11-14T08:43:00+07:00

---

## Table of Contents

- [1. Overview &amp; Research Problem](#1-overview--research-problem)
- [2. Pipeline Architecture &amp; File Mapping](#2-pipeline-architecture--file-mapping)
- [3. Deep Learning Pipeline Documentation Index](#3-deep-learning-pipeline-documentation-index)
- [4. Kịch Bản Thuyết Trình &amp; Bảo Vệ Đồ Án (Presentation Script)](#4-kịch-bản-thuyết-trình--bảo-vệ-đồ-án-presentation-script)
- [5. Notebooks &amp; Exercise Navigation](#5-notebooks--exercise-navigation)
- [6. Empirical Benchmark Summary (5W1H)](#6-empirical-benchmark-summary-5w1h)
- [7. Hardware &amp; VRAM Governance](#7-hardware--vram-governance)

---

## 1. Overview & Research Problem

Lab 3 focuses on **Transfer Learning in Natural Language Processing (NLP)** using the Hugging Face ecosystem and PyTorch. The coursework addresses binary sentiment classification on the Stanford Sentiment Treebank (**SST-2**) dataset from the General Language Understanding Evaluation (**GLUE**) benchmark.

* **Task Formulation**: Given a natural language sentence $x = (w_1, w_2, \dots, w_T)$, predict its sentiment label $y \in \{0, 1\}$, where $0$ corresponds to `NEGATIVE` and $1$ corresponds to `POSITIVE`.
* **Core Model**: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) — a 6-layer distilled Transformer model with 66M parameters retaining 97% of BERT-base's language comprehension while running 60% faster with 40% fewer parameters.
* **Dataset**: [`nyu-mll/glue` (sst2)](https://huggingface.co/datasets/nyu-mll/glue), containing 67,349 training sentences and 872 validation sentences.

---

## 2. Pipeline Architecture & File Mapping

The repository structure for Lab 3 strictly conforms to [agents/rules/FOLDER_STRUCTURE.md](../../.agents/rules/FOLDER_STRUCTURE.md):

```text
Deeplearning-UTH/
├── docs/lab3/
│   ├── README.md                      # Index này (7-field header, 5W1H, navigation)
│   ├── PIPELINE_EXPLANATION.md        # Giải thích chi tiết 10 module (Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp)
│   └── PRESENTATION.md                # Kịch bản thuyết trình chi tiết, vị trí file và bộ câu hỏi phản biện Q&A
├── labs/lab3/
│   ├── README.md                      # Hướng dẫn tổng quan bài nộp
│   ├── QUESTIOn.md                    # Đề bài gốc 2 bài tập (Exercise 1 & Exercise 2)
│   └── main/
│       ├── 01_ex1_sentiment_analysis.ipynb  # Exercise 1: Zero-shot & low-level PyTorch inference
│       ├── 02_ex2_finetune_sentiment.ipynb  # Exercise 2: End-to-end DistilBERT Fine-tuning trên SST-2
│       ├── main.ipynb                 # Alias / Notebook gốc Exercise 1
│       ├── fine-tune.ipynb            # Alias / Notebook gốc Exercise 2
│       ├── main.py                    # Script CLI suy luận nhanh qua pipeline
│       ├── fine-tune.py               # Script CLI huấn luyện chính thức (HF Trainer)
│       └── sst2-distilbert/           # Trọng số checkpoint tốt nhất (checkpoint-2105)
└── notebooks/lab3/
    ├── 01_ex1_sentiment_analysis.ipynb # Bản colocation chuẩn theo FOLDER_STRUCTURE.md
    └── 02_ex2_finetune_sentiment.ipynb # Bản colocation chuẩn theo FOLDER_STRUCTURE.md
```

---

## 3. Deep Learning Pipeline Documentation Index

Theo chỉ dẫn tại [agents/rules/EXPLAIN_PIPELINE.md](../../.agents/rules/EXPLAIN_PIPELINE.md) và [docs/reference/ML_PIPELINE.md](../reference/ML_PIPELINE.md), tài liệu phân tích kỹ thuật chi tiết được lưu trữ tại:

👉 **[docs/lab3/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md)**

Tài liệu này phân tích 10 module độc lập của pipeline Transformer với cấu trúc chuẩn:
1. **Module 1: Problem Formulation & Task Definition** (Binary Sentiment Classification trên SST-2)
2. **Module 2: Data Ingestion & Partitioning Lineage** (Tải dữ liệu `nyu-mll/glue`, bảo toàn split, che nhãn test)
3. **Module 3: Exploratory Data Analysis & Text Geometry** (Phân phối nhãn, thống kê độ dài token, phát hiện cụm từ con)
4. **Module 4: Subword Tokenization & Vocabulary Mapping** (Cơ chế WordPiece, tokens `[CLS]`, `[SEP]`, `[PAD]`, truncation `max_length=64`)
5. **Module 5: Dynamic Batch Collation & Memory Alignment** (`DataCollatorWithPadding`, tối ưu padding theo batch)
6. **Module 6: Transformer Model Architecture & Inductive Bias** (DistilBERT Encoder, Self-Attention, feed-forward sublayers, `[CLS]` pooled classification head)
7. **Module 7: Optimization Engine & Fine-Tuning Strategy** (AdamW, learning rate $2\times 10^{-5}$, weight decay 0.01, FP16 AMP)
8. **Module 8: Evaluation Framework & Metric Validation** (Accuracy, Macro-F1, Confusion Matrix, Classification Report)
9. **Module 9: Error Diagnostics & Linguistic Residual Analysis** (False Positives, False Negatives, kiểm thử câu phủ định *"not bad at all"*, mỉa mai, phân tích theo độ dài)
10. **Module 10: Production Inference & Serving Pipeline** (Hugging Face `pipeline` vs native PyTorch `torch.no_grad()` forward pass)

---

## 4. Kịch Bản Thuyết Trình & Bảo Vệ Đồ Án (Presentation Script)

Tài liệu hướng dẫn thuyết trình toàn diện, chỉ rõ vị trí từng file cần mở, kịch bản lời nói chi tiết và bộ câu hỏi phản biện:

👉 **[docs/lab3/PRESENTATION.md](PRESENTATION.md)**

* **Bản đồ phân bổ file**: Chỉ rõ từng file mã nguồn, notebook và checkpoint nằm ở đâu, đóng vai trò gì.
* **Kịch bản thuyết trình (10-15 phút)**: Từng lời từng chữ từ mở đầu, phân tích dữ liệu, mổ xẻ tokenizer, kiến trúc DistilBERT, báo cáo kết quả đến phân tích lỗi.
* **Kịch bản Live Demo**: Hướng dẫn thao tác chạy CLI và chạy interactive notebook trực tiếp trước Giảng viên.
* **Bộ câu hỏi phản biện (Q&A Defense)**: 8 câu hỏi hóc búa thường gặp kèm câu trả lời đắt giá.

---

## 5. Notebooks & Exercise Navigation

Cả hai bài tập trong [labs/lab3/QUESTIOn.md](../../labs/lab3/QUESTIOn.md) đã được hoàn thiện độc lập, tuân thủ nghiêm ngặt [agents/rules/NOTEBOOK_HEADER_CONVENTION.md](../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md):

| Bài tập | File Notebook | Mục tiêu chính | Đầu ra chủ yếu |
| :--- | :--- | :--- | :--- |
| **Exercise 1** | [`01_ex1_sentiment_analysis.ipynb`](../../labs/lab3/main/01_ex1_sentiment_analysis.ipynb) | Làm quen với Hugging Face `transformers`: sử dụng pre-trained pipeline, mổ xẻ cơ chế tokenizer (token, ID, special tokens), thực hiện suy luận native bằng PyTorch tensors, giải mã logits và Softmax probabilities. | Logits tensor, xác suất cảm xúc, kết quả suy luận câu mẫu. |
| **Exercise 2** | [`02_ex2_finetune_sentiment.ipynb`](../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb) | Toàn bộ quy trình Fine-tuning Transformer: nạp dataset SST-2, EDA hình học văn bản, tokenization + dynamic padding, huấn luyện mô hình DistilBERT với Hugging Face `Trainer`, đánh giá toàn diện, phân tích lỗi định lượng và kiểm thử phủ định. | Checkpoint mô hình, Ma trận nhầm lẫn (Confusion Matrix), Báo cáo phân loại (Classification Report), Phân tích lỗi tự tin cao. |

---

## 5. Empirical Benchmark Summary (5W1H)

Tuân thủ nghiêm ngặt quy tắc [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md), kết quả thực nghiệm từ quá trình fine-tune DistilBERT trên SST-2 được công bố với ngữ cảnh 5W1H đầy đủ:

> **5W1H — DistilBERT SST-2 Fine-Tuning Benchmark**
> - **What**: Macro-F1 = **90.41%**, Validation Accuracy = **90.37%**, Validation Loss = **0.2658** trên tập GLUE SST-2 validation split (872 câu).
> - **Why**: Chứng minh năng lực của Transfer Learning: mô hình ngôn ngữ tiền huấn luyện chỉ cần 1 epoch fine-tuning nhẹ đã vượt ngưỡng 90% độ chính xác trong phân loại cảm xúc nhị phân.
> - **When**: Đo lường tại Checkpoint `checkpoint-2105` (Epoch 1.0, Global Step 2105, Seed 42).
> - **Where**: Trọng số lưu tại [`labs/lab3/main/sst2-distilbert/checkpoint-2105/`](../../labs/lab3/main/sst2-distilbert/checkpoint-2105/); chạy trên GPU (NVIDIA RTX, CUDA 12.x, PyTorch 2.x).
> - **Who**: Nhóm nghiên cứu Deep Learning UTH — dành cho Giảng viên và Hội đồng chấm đồ án.
> - **How**: Tối ưu bằng thuật toán AdamW ($\text{lr}=2\times 10^{-5}$, $\text{weight\_decay}=0.01$), `batch_size=32`, `fp16=True`, `max_length=64`, dynamic padding bằng `DataCollatorWithPadding`, metric tiêu chí chọn mô hình là `f1`.

### Bảng Chỉ Số Hiệu Năng Chi Tiết

| Lớp (Class) | Precision | Recall | F1-Score | Số lượng mẫu (Support) | Ý nghĩa kỹ thuật 5W1H |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NEGATIVE (0)** | **0.908** | **0.904** | **0.906** | 428 | Khả năng phát hiện chính xác các câu mang cảm xúc tiêu cực, hạn chế báo động nhầm. |
| **POSITIVE (1)** | **0.907** | **0.912** | **0.910** | 444 | Khả năng thu hồi đúng các nhận xét khen ngợi và tích cực của khán giả. |
| **Macro Average** | **0.908** | **0.908** | **0.908** | 872 | Điểm trung bình không trọng số giữa 2 lớp, phản ánh độ công bằng của mô hình. |
| **Weighted Average**| **0.908** | **0.908** | **0.908** | 872 | Điểm trung bình có trọng số theo kích thước từng lớp trong tập validation. |
| **Accuracy** | — | — | **0.904 (90.37%)** | 872 | Tỷ lệ tổng thể số câu dự đoán chính xác trên toàn bộ tập validation. |

---

## 6. Hardware & VRAM Governance

Theo quy chuẩn [agents/rules/PYTORCH_FRAMEWORK_RULES.md](../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md), toàn bộ cấu hình huấn luyện của Lab 3 được thiết kế để không bao giờ vượt quá ngưỡng trần **4.0GB VRAM** (mục tiêu $\le 3.5\text{GB}$):

* **Kỹ thuật FP16 Mixed Precision**: Kích hoạt `fp16=True` trong `TrainingArguments`, lưu trữ trọng số và tensor kích hoạt ở định dạng half-precision (16-bit), cắt giảm ~50% VRAM so với FP32.
* **Dynamic Padding (`DataCollatorWithPadding`)**: Chỉ đệm chiều dài câu đến độ dài tối đa trong từng batch thay vì cố định 512 hoặc 64 tokens cho toàn tập. Với độ dài trung bình của SST-2 là 9 từ, dynamic padding giúp tiết kiệm tới ~70% dung lượng bộ nhớ tensor kích hoạt của các tầng Self-Attention ($O(N^2)$).
* **Batch Size Điều Độ**: Thiết lập `per_device_train_batch_size=32` và `per_device_eval_batch_size=64`, đảm bảo mức tiêu thụ VRAM đỉnh điểm khi chạy trên card đồ họa chỉ dao động trong khoảng **1.8GB – 2.4GB VRAM**.

