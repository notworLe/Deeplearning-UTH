# Lab 03 (Final): DistilBERT Fine-Tuning for Sentiment Analysis

Bản mã nguồn chính thức (Final Production Code) của nhóm dùng để nghiệm thu và nộp bài Lab 03. Dự án ứng dụng kiến trúc Transformer với mô hình `DistilBERT` để giải quyết bài toán Phân tích Cảm xúc Văn bản (Sentiment Analysis) trên tập dữ liệu SST-2 thuộc bộ chuẩn GLUE Benchmark.

---

## 📌 1. Tổng Quan Bài Toán & Tập Dữ Liệu

- **Bài toán**: Phân loại cảm xúc nhị phân của các câu đánh giá phim (Movie Reviews) thành hai nhóm: `NEGATIVE` (0) hoặc `POSITIVE` (1).
- **Tập dữ liệu**: **Stanford Sentiment Treebank (SST-2)** tải trực tiếp từ Hugging Face Datasets (`nyu-mll/glue`, `sst2`).
  - Phân chia: Sử dụng tập `train` để huấn luyện và tập `validation` để đánh giá định kỳ sau mỗi epoch (do tập test của GLUE ẩn nhãn thật).
- **Mô hình Nền tảng**: **DistilBERT** (`distilbert-base-uncased`) - phiên bản thu gọn nhẹ hơn 40% và nhanh hơn 60% so với BERT-base nhưng vẫn giữ lại 97% năng lực hiểu ngôn ngữ tự nhiên (NLU).

---

## 🏗️ 2. Quy Trình Kỹ Thuật (NLP Pipeline)

```text
Văn bản đầu vào (Raw Sentences)
  │
  ├── [Tokenization] AutoTokenizer (WordPiece, max_length=64, truncation=True)
  │
  ├── [Dynamic Padding] DataCollatorWithPadding (Đệm padding linh hoạt theo độ dài batch)
  │
  ├── [Pre-trained Model] AutoModelForSequenceClassification (DistilBERT + Classifier Head)
  │
  ├── [Trainer Engine] Hugging Face Trainer (fp16=True, AdamW with Weight Decay 0.01, lr=2e-5)
  │
  └── [Evaluation] Accuracy, F1-Score & Confusion Matrix
```

---

## 📂 3. Cấu Trúc Thư Mục `main/`

```text
labs/lab3/main/
├── README.md               # Báo cáo kỹ thuật chi tiết này
├── 01_ex1_sentiment_analysis.ipynb # Notebook thực hành Exercise 1 (Hugging Face pipeline & PyTorch low-level)
├── 02_ex2_finetune_sentiment.ipynb # Notebook thực hành Exercise 2 (Fine-tuning DistilBERT trên SST-2)
├── fine-tune.py            # Script huấn luyện fine-tune toàn diện
├── fine-tune.ipynb         # Jupyter Notebook thực hành fine-tune trực quan (đồng bộ 02_ex2)
├── main.py                 # Script suy luận nhanh với pre-trained pipeline
├── main.ipynb              # Notebook suy luận tương tác (đồng bộ 01_ex1)
└── sst2-distilbert/        # Thư mục lưu checkpoint mô hình tốt nhất (weights, config, tokenizer)
```

---

## 📚 Phân Tích Chuyên Sâu ML/DL Pipeline

Chi tiết phân tích toán học và cơ chế chuyển đổi tensor theo 6 trục chuẩn mực (*Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp*) được lưu trữ tại:
👉 **[`docs/lab3/PIPELINE_EXPLANATION.md`](../../../docs/lab3/PIPELINE_EXPLANATION.md)**.

---

## ⚙️ 4. Thiết Kế Huấn Luyện & Siêu Tham Số (Hyperparameters)

Được cấu hình thông qua `TrainingArguments`:

| Siêu Tham Số | Giá Trị | Mục Đích Kỹ Thuật |
| :--- | :--- | :--- |
| **Learning Rate** | `2e-5` | Tốc độ học nhỏ đặc trưng khi fine-tune Transformer để tránh phá vỡ các trọng số đã học. |
| **Batch Size (Train/Eval)**| `32 / 64` | Tối ưu thông lượng nạp dữ liệu và dung lượng bộ nhớ VRAM. |
| **Epochs** | `3` | Đảm bảo mô hình hội tụ tốt mà không bị overfit trên tập dữ liệu SST-2. |
| **Weight Decay** | `0.01` | Điều chuẩn $L_2$ hạn chế độ phức tạp của trọng số. |
| **fp16 (Mixed Precision)** | `True` | Tăng tốc độ huấn luyện x2 và giảm 50% dung lượng VRAM trên card GPU RTX. |
| **Metric for Best Model** | `f1` | Tiêu chí chọn checkpoint tốt nhất dựa trên điểm số cân bằng Macro-F1. |
| **Data Collator** | `DataCollatorWithPadding`| Padding động theo độ dài câu dài nhất trong từng batch thay vì cố định cả tập, tiết kiệm tính toán. |

---

## 🚀 5. Hướng Dẫn Thực Thi

### 5.1. Chạy Quá Trình Fine-tuning Hoàn Chỉnh

Chạy script chính để huấn luyện và tự động lưu checkpoint:

```bash
python fine-tune.py
```

### 5.2. Chạy Suy Luận Nhanh (Inference)

Sử dụng `main.py` để kiểm tra khả năng dự đoán cảm xúc trên các câu mẫu tùy biến:

```bash
python main.py
```

Đoạn mã mẫu suy luận:

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)

results = classifier([
    "Just have trying, don't end up like me",
    "I love this movie, the plot is brilliant and moving!"
])

for r in results:
    print(r)
# Output:
# {'label': 'NEGATIVE', 'score': 0.998...}
# {'label': 'POSITIVE', 'score': 0.999...}
```

---

## 📊 6. Đánh Giá & Đo Lường Kết Quả

Quá trình đánh giá được thực hiện tự động sau khi kết thúc 3 epochs bằng thư viện `scikit-learn`:
- **Báo cáo phân loại (Classification Report)**: Đầy đủ chỉ số Precision, Recall, F1-Score cho cả hai phân lớp `NEGATIVE` và `POSITIVE`.
- **Confusion Matrix Display**: Trực quan hóa ma trận dự đoán đúng/sai qua biểu đồ trực quan `ConfusionMatrixDisplay` với bảng màu `Blues`.

---

## 🔗 7. Liên Kết Tài Liệu Tham Khảo

- Hugging Face Transformers Documentation: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- GLUE Benchmark Leaderboard: [https://gluebenchmark.com/](https://gluebenchmark.com/)
- Bộ quy chuẩn tham chiếu AI: [`.agents/AI_REFERENCE.md`](../../../.agents/AI_REFERENCE.md)
