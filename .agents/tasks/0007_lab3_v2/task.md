---
title: Lab 3 - Get started with Hugging Face
status: In Progress
owner: notworLe
pipeline: MODEL_PIPELINE.md
---

# Objective

Làm quen hệ sinh thái Hugging Face qua 2 bài:

- **Ex1:** dùng model có sẵn để phân tích cảm xúc (sentiment analysis).
- **Ex2:** fine-tune một pretrained model cho bài toán phân loại văn bản 2 lớp (binary text classification).

Mục tiêu không phải điểm cao, mà là hiểu luồng: **text → tokenizer → model → kết quả**, và cách `Trainer` huấn luyện model.

---

# Background

Bài thực hành thứ 3 của môn Deep Learning (Practice 3 – Get started with Hugging Face).

Bài này giúp hiểu:

- `pipeline` của `transformers` dùng để làm gì
- tokenizer biến câu chữ thành `input_ids` và `attention_mask` như thế nào
- cách load pretrained model + tokenizer từ Hub
- cách dùng `datasets`, `TrainingArguments`, `Trainer`, `evaluate`

Bài làm theo luồng `MODEL_PIPELINE.md` (bỏ qua Project Reference).

---

# Input

Thư viện

- `transformers`, `datasets`, `evaluate`

Ex1

- Model: `distilbert-base-uncased-finetuned-sst-2-english` (sentiment, có sẵn trên Hub)
- Dữ liệu: vài câu ví dụ tự viết (tiếng Anh)

Ex2

- Dataset: IMDB (review phim, 2 lớp: negative / positive), dùng **tập con nhỏ** cho nhanh
- Model: `distilbert-base-uncased` (nhẹ, phù hợp bài lab)

---

# Business Requirement

Đây là bài học, không phải hệ thống thật, nhưng vẫn làm như một dự án AI thu nhỏ.

Câu hỏi:

> Có thể phân loại cảm xúc một câu/review tự động không? Fine-tune một pretrained model trên dữ liệu nhỏ có cải thiện so với chưa fine-tune không?

---

# Problem Definition

Task: Text Classification (binary)

Input: câu / review tiếng Anh → tokenizer → `input_ids`, `attention_mask` `(B, L)`

Output: logits `(B, 2)` → nhãn negative / positive

Chia dữ liệu (Ex2)

- Train: tập con của IMDB train
- Validation: tách từ IMDB train (cố định seed)
- Test: tập con của IMDB test, **chấm một lần cuối**

Metric: Accuracy (chính), F1, loss

---

# Requirements

## Ex1: Sentiment Analysis

- [ ] Cài `transformers`
- [ ] Load model sentiment từ Hub bằng `pipeline`
- [ ] Tokenize một câu mẫu, in ra `input_ids`, `attention_mask`, và decode ngược lại để hiểu
- [ ] Chạy sentiment analysis cho vài câu (tích cực, tiêu cực, câu khó/mỉa mai)

## Ex2: Fine-tune Binary Text Classification

- [ ] Cài `transformers`, `datasets`, `evaluate`
- [ ] Load dataset, lấy tập con, tách validation
- [ ] Load model + tokenizer (`num_labels=2`)
- [ ] Tokenize dataset (`truncation`, `padding`)
- [ ] Định nghĩa `TrainingArguments`
- [ ] Tạo `Trainer` và fine-tune
- [ ] Đánh giá trên validation, sau đó chấm test một lần

## Visualization / Analysis

- Loss curve của quá trình train
- So sánh accuracy **trước** và **sau** fine-tune
- Vài ví dụ model dự đoán sai và nhận xét ngắn

---

| ID  | Task                                   | Pipeline Layer     | Output         |
| --- | -------------------------------------- | ------------------ | -------------- |
| T1  | Sentiment analysis bằng `pipeline`     | Model Engineering  | notebook (Ex1) |
| T2  | Tokenize và giải thích `input_ids`     | Data Preprocessing | notebook (Ex1) |
| T3  | Load IMDB, lấy tập con, tách val       | Data Collection    | dataset split  |
| T4  | Tokenize dataset                       | Data Preprocessing | tokenized data |
| T5  | Đo accuracy trước fine-tune (baseline) | Evaluation         | metrics        |
| T6  | Fine-tune bằng `Trainer`               | Training           | checkpoint     |
| T7  | Đánh giá val, rồi test một lần         | Evaluation         | metrics        |
| T8  | Viết report ngắn                       | Documentation      | REPORT.md      |

---

# Deliverables

- Notebook: `ex1_sentiment.ipynb`, `ex2_finetune.ipynb`
- Script (nếu cần): trong `ROOT/src/vibe_scrtips/`
- Checkpoint fine-tune và log train
- `REPORT.md` ngắn gọn

---

# Acceptance Criteria

Hoàn thành khi

- Ex1 chạy đúng, có giải thích tokenizer
- Ex2 fine-tune chạy hết, không lỗi, loss giảm
- **Accuracy trên val sau fine-tune > trước fine-tune**
- Test chỉ chấm một lần, sau khi xong val
- Report giải thích kết quả và các ca dự đoán sai

Constraints

- Full fine-tune do Kỹ sư chạy; Agent chỉ chạy smoke test (vài batch).
- Không tune theo test set.
- Không bịa số metric khi chưa chạy thật.
- Không thêm thư viện ngoài `transformers`, `datasets`, `evaluate` (và `torch`).

Assumptions (cần Kỹ sư xác nhận)

- Dùng IMDB tập con (ví dụ vài nghìn mẫu train) thay vì toàn bộ.
- Model Ex2 là `distilbert-base-uncased`; có GPU thì chạy nhanh, không thì giảm số mẫu.
- Số epoch, learning rate, batch size do Agent đề xuất trong Plan.
