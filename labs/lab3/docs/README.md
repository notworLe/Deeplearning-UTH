# Tài Liệu Kỹ Thuật & Kịch Bản Thuyết Trình Lab 03 (`labs/lab3/docs/`)

Thư mục lưu trữ tập trung các tài liệu thuyết trình (`PRESENTATION`), sơ đồ luồng biểu diễn dữ liệu/tensor và bộ câu hỏi phản biện (Q&A Defense) cho từng Jupyter Notebook trong [`labs/lab3/main/`](../main/).

---

## Danh Mục Tài Liệu Thuyết Trình Theo Từng Notebook

| STT | Notebook Thực Hành (`labs/lab3/main/`) | Bản Đồng Bộ / Script CLI | Tài Liệu Thuyết Trình & Biểu Diễn (`labs/lab3/docs/`) | Trọng Tâm Kỹ Thuật |
| :-: | :--- | :--- | :--- | :--- |
| **1** | [`01_ex1_sentiment_analysis.ipynb`](../main/01_ex1_sentiment_analysis.ipynb) | [`main.ipynb`](../main/main.ipynb), [`main.py`](../main/main.py) | [`PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md`](PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md) | **Exercise 1**: So sánh High-level `pipeline` với Low-level PyTorch forward pass, WordPiece subwords (`[CLS]`, `[SEP]`, `[PAD]`), `attention_mask` và kiểm thử câu phủ định. |
| **2** | [`02_ex2_finetune_sentiment.ipynb`](../main/02_ex2_finetune_sentiment.ipynb) | [`fine-tune.ipynb`](../main/fine-tune.ipynb), [`fine-tune.py`](../main/fine-tune.py) | [`PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md`](PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md) | **Exercise 2**: Khảo sát EDA GLUE SST-2, cắt chuỗi `max_length=64`, đệm động `DataCollatorWithPadding`, fine-tuning `DistilBERT` với `Trainer` (`fp16`, `lr=2e-5`), đánh giá 5W1H và chẩn đoán phần dư (FP/FN, lỗi tự tin cao). |

---

## Liên Kết Nhanh Trong Dự Án

- 📋 Đề bài gốc Lab 03: [`labs/lab3/QUESTIOn.md`](../QUESTIOn.md)
- 🏠 Tổng quan cấu trúc Lab 03: [`labs/lab3/README.md`](../README.md)
- 📘 Hướng dẫn thực thi bản nộp (`main/`): [`labs/lab3/main/README.md`](../main/README.md)
- 📐 Quy chuẩn kỹ thuật dự án: [`.agents/rules/AI_REFERENCE.md`](../../../.agents/rules/AI_REFERENCE.md)
