# Documentation Index — `lab3_v2` (Get Started with Hugging Face: Sentiment & DistilBERT Fine-Tuning)

## Purpose

Tập hợp toàn bộ tài liệu thiết kế pipeline, đặc tả kiến trúc `DistilBERT` và **3 báo cáo đơn nhiệm SRP** (ánh xạ 1-1 với các tầng [`MODEL_PIPELINE.md`](../../../.agents/rules/dl-rule/MODEL_PIPELINE.md)) cho dự án **`lab3_v2`**, đồng bộ với mã nguồn lõi [`src/vibe_scripts/lab3_v2/`](../../../src/vibe_scripts/lab3_v2/README.md) và kết quả tại [`experiments/lab3_v2/`](../../../experiments/lab3_v2/README.md).

## Responsibilities

- Chỉ lưu tài liệu Markdown (không chứa mã nguồn thực thi).
- Mọi con số thực nghiệm trong 3 báo cáo đều trích xuất 100% từ artifacts đã xác minh tại `experiments/lab3_v2/` (tuyệt đối không bịa số trước khi Kỹ sư chạy thật ở Step 4).

## Document Map

| Tài liệu | Vai trò & Pipeline Layer | Trạng thái |
| :--- | :--- | :---: |
| [`../REPORT.md`](../REPORT.md) | Báo cáo tóm tắt điều hướng nhanh (Ex1 + Ex2) dẫn về 3 báo cáo SRP | Hoàn thiện |
| [`pipelines/modeling.md`](pipelines/modeling.md) | Đặc tả pipeline Hugging Face theo `MODEL_PIPELINE.md v1.4.0` (bỏ qua Layer 3 Project Reference), Tensor Shape Contract `(B, L) → (B, 2)` & chống Test leakage | Hoàn thiện |
| [`models/distilbert.md`](models/distilbert.md) | Đặc tả kiến trúc `DistilBERT` (`distilbert-base-uncased` & `sst-2` pretrained), cơ chế Knowledge Distillation, WordPiece Tokenizer & đầu phân loại 2 lớp | Hoàn thiện |
| [`eda_report.md`](eda_report.md) | Báo cáo dữ liệu IMDB (`4,000 / 1,000 / 1,000`, `seed = 248`), làm sạch & thống kê độ dài token tại `max_length = 256` (Layer 6) | Hoàn thiện |
| [`experiment_report.md`](experiment_report.md) | Ex1 Tokenizer & Sentiment Pipeline + Sanity Checks & Training Diagnostics của HF `Trainer` (Layer 4 & Layer 7) | Hoàn thiện |
| [`evaluation_report.md`](evaluation_report.md) | So sánh Trước vs Sau fine-tune trên Validation (`1,000`), Final Held-Out Test (`1,000`, chấm 1 lần), Error Analysis & Inference (Layer 8–10) | Hoàn thiện |

## Related Modules

- Cấu hình, CLI & 2 Notebooks (`ex1_sentiment.ipynb`, `ex2_finetune.ipynb`): [`labs/lab3_v2/README.md`](../README.md)
- Mã nguồn lõi tái sử dụng: [`src/vibe_scripts/lab3_v2/README.md`](../../../src/vibe_scripts/lab3_v2/README.md)
- Thư mục Artifacts & Biểu đồ: [`experiments/lab3_v2/README.md`](../../../experiments/lab3_v2/README.md)
