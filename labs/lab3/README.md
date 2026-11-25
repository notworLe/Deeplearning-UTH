# Lab 03: Transformer Fine-Tuning for Sentiment Analysis (DistilBERT)

Dự án thực hành Deep Learning: Tinh chỉnh (Fine-tuning) mô hình Transformer tiền huấn luyện (`DistilBERT`) trên tập dữ liệu SST-2 (Stanford Sentiment Treebank) thuộc GLUE Benchmark sử dụng thư viện Hugging Face Transformers và PyTorch.

---

## 🗂️ Cấu trúc Thư mục Lab 03

```text
labs/lab3/
├── README.md               # Tài liệu định hướng và tổng quan bài Lab 03
├── QUESTIOn.md             # Đề bài gốc 2 bài tập (Exercise 1 & Exercise 2)
├── main/                   # NƠI CHỨA CODE FINAL BẢN CUỐI CÙNG ĐỂ NỘP
│   ├── README.md           # Tài liệu kỹ thuật chi tiết của bản nộp chính thức
│   ├── 01_ex1_sentiment_analysis.ipynb # Exercise 1: Pipeline & PyTorch Tensors
│   ├── 02_ex2_finetune_sentiment.ipynb # Exercise 2: Fine-tuning DistilBERT trên SST-2
│   ├── fine-tune.py        # Script huấn luyện fine-tune hoàn chỉnh với Hugging Face Trainer
│   ├── fine-tune.ipynb     # Jupyter Notebook huấn luyện và trực quan hóa kết quả (alias ex2)
│   ├── main.py             # Script kiểm tra suy luận nhanh qua pipeline sentiment-analysis
│   ├── main.ipynb          # Notebook thực hành suy luận tương tác (alias ex1)
│   └── sst2-distilbert/    # Trọng số và cấu hình checkpoint mô hình sau fine-tune
└── submit/                 # NƠI LƯU BẢN CODE ĐÓNG GÓP CỦA CÁC THÀNH VIÊN TRONG NHÓM
    └── (Dành cho các thành viên nộp bài làm độc lập và các thực nghiệm cá nhân)
```

---

## 📚 Tài Liệu Thuyết Trình & Bóc Tách Kỹ Thuật Pipeline (`labs/lab3/docs/`)

Toàn bộ sơ đồ luồng biểu diễn dữ liệu/tensor, kịch bản thuyết trình từng cell và bộ câu hỏi phản biện cho từng notebook được lưu trữ tại:
👉 **[`docs/README.md`](docs/README.md)** (Mục lục tài liệu thuyết trình Lab 03).
👉 **[`docs/PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md`](docs/PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md)** (Thuyết trình & biểu diễn tensor Notebook 1 — Exercise 1).
👉 **[`docs/PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md`](docs/PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md)** (Thuyết trình & biểu diễn pipeline Notebook 2 — Exercise 2).

---

## 📌 Phân Định Chức Năng Giữa `main/` và `submit/`

| Thư mục | Định nghĩa & Trách nhiệm | Đối tượng sử dụng |
| :--- | :--- | :--- |
| **`main/`** | **Bản Code Final Hoàn Chỉnh Nhất**: Triển khai toàn bộ quy trình từ tải dataset GLUE SST-2, tokenization, dynamic padding với `DataCollatorWithPadding`, fine-tuning DistilBERT với Hugging Face Trainer (fp16, evaluation strategy), xuất báo cáo phân loại (Classification Report) và ma trận nhầm lẫn. Dùng để nộp bài và chấm điểm chính thức. | Giảng viên, người chấm bài, nghiệm thu kỹ thuật |
| **`submit/`** | **Bản Đóng Góp Của Thành Viên**: Nơi chứa code thử nghiệm riêng của các thành viên trong nhóm khi khảo sát các tokenizer hoặc siêu tham số khác nhau. | Lưu trữ và đối chiếu đóng góp cá nhân |

---

## 🚀 Hướng Dẫn Bắt Đầu Nhanh Với Bản Final (`main/`)

Di chuyển vào thư mục `labs/lab3/main`:

```bash
# 1. Di chuyển vào thư mục code chính thức
cd labs/lab3/main

# 2. Chạy quá trình Fine-tuning mô hình DistilBERT
python fine-tune.py

# 3. Chạy thử nghiệm suy luận nhanh với pipeline
python main.py
```

👉 **Xem giải thích chi tiết cơ chế Transformer, đánh giá định lượng và siêu tham số tại [main/README.md](main/README.md).**
