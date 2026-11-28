---
title: Knowledge Extraction — Lab 3 v2 (Hugging Face v5 & DistilBERT Fine-Tuning)
doc_type: Knowledge
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-28
updated: 2026-11-28
version: 1.0.0
status: completed
---

# Knowledge Extraction: `0007_lab3_v2`

> **Source Log:** [`0007_lab3_v2_log.md`](../logs/0007_lab3_v2_log.md) · **Standard:** [`KNOWLEDGE_EXTRACTION.md`](../rules/dl-rule/KNOWLEDGE_EXTRACTION.md)

---

## 1. Breaking API Changes in `transformers 5.x` (`5.23.0`)

- **Category:** Compatibility & Debugging
- **Description:** Nâng cấp lên `transformers v5` gỡ bỏ nhiều tham số quen thuộc của `v4`:
  1. `TrainingArguments`: dùng **`eval_strategy`** (gỡ `evaluation_strategy`) và **gỡ hoàn toàn `logging_dir`** (cấu hình thư mục TensorBoard qua biến môi trường `TENSORBOARD_LOGGING_DIR`).
  2. `Trainer`: dùng **`processing_class=tokenizer`** (gỡ tham số `tokenizer=`).
  3. `pipeline`: task chính thức trong `SUPPORTED_TASKS` là **`"text-classification"`** (không còn liệt kê `"sentiment-analysis"` trong `SUPPORTED_TASKS`).
  4. Offline unit testing: `BertTokenizer(vocab_file=...)` trong v5 không tự nạp từ file text cũ mà trả `vocab_size = 5`; cần truyền trực tiếp dict **`BertTokenizer(vocab={word: idx, ...})`** để tạo tokenizer WordPiece offline chính xác.
- **Evidence:** Kiểm chứng bằng `inspect.signature` và vượt qua `6/6` offline tests + full fine-tune trong [`ex1_sentiment.py`](../../src/vibe_scripts/lab3_v2/ex1_sentiment.py), [`ex2_finetune.py`](../../src/vibe_scripts/lab3_v2/ex2_finetune.py) và [`test_lab3_v2_ex1.py`](../../test/test_lab3_v2_ex1.py).
- **Recommendation:** Kiểm tra chữ ký `TrainingArguments` / `Trainer` ngay ở bước Planning cho mọi dự án NLP mới dùng `transformers >= 5.0` và luôn sử dụng bộ tham số chuẩn v5 ở trên.

---

## 2. `datasets 5.2.0` + `huggingface_hub` Require Namespaced Dataset IDs (`namespace/name`)

- **Category:** Data Engineering & Debugging
- **Description:** Với `datasets 5.2.0` đi kèm `huggingface_hub` mới, gọi `load_dataset("imdb")` không có namespace bị lỗi `HfUriError: Repository id must be 'namespace/name', got 'imdb'` trong `parse_hf_uri`.
- **Evidence:** Xảy ra trực tiếp khi chạy `train_ex2.py` lần đầu; sửa dứt điểm bằng cách ánh xạ sang repo ID chính thức **`"stanfordnlp/imdb"`** tại [`data.py`](../../src/vibe_scripts/lab3_v2/data.py#L52-L62).
- **Recommendation:** Luôn dùng ID có đầy đủ namespace (vd. `stanfordnlp/imdb`, `nyu-mll/glue`) khi tải dataset từ Hugging Face Hub trên `datasets >= 5.x`.

---

## 3. Filtering Post-Reload `trainer.evaluate()` Entries Before Plotting `log_history`

- **Category:** Visualization & Debugging
- **Description:** Khi `Trainer` bật `load_best_model_at_end=True`, nếu gọi thêm `trainer.evaluate()` sau `trainer.train()` để ghi nhận metric cuối của best checkpoint, `Trainer` sẽ nối thêm 1 bản ghi `eval_loss` mang `epoch = final_epoch` nằm **sau** bản ghi `"train_runtime"`. Vẽ trực tiếp toàn bộ `trainer.state.log_history` sẽ tạo một vạch rơi thẳng đứng bất thường ở cuối đường `Val Loss`.
- **Evidence:** Phát hiện ngay qua kiểm tra trực quan `monitor/loss_curve.png` (vạch tụt tại `epoch = 3.0` từ `0.3442 → 0.2764`); đã sửa sạch trong [`analysis.py`](../../src/vibe_scripts/lab3_v2/analysis.py#L22-L35) bằng cách dừng đọc vòng lặp tại mốc `"train_runtime"`.
- **Recommendation:** Mọi hàm vẽ đường học từ `Trainer.state.log_history` cần dừng quét khi gặp khóa `"train_runtime"` để chỉ vẽ đúng tiến trình trong quá trình huấn luyện.

---

## 4. Rapid Convergence & *Loss–Accuracy Divergence* When Fine-Tuning Transformers on Medium Subsets

- **Category:** Model Engineering
- **Description:** Khi fine-tune toàn bộ `66.96M` tham số của `DistilBERT` trên `4,000` review IMDB (`lr = 2e-5`), mô hình hội tụ gần như tối ưu ngay sau **Epoch 1 (`eval_loss = 0.2764`, `Val Acc = 87.80%`)**; từ Epoch 2 → 3 `train_loss` giảm sâu xuống `0.0905` nhưng `eval_loss` tăng ngược lên `0.3176 → 0.3442` dù `Val Acc` đạt đỉnh `88.80%` ở Epoch 2.
- **Evidence:** Nhật ký từng 25 steps tại [`trainer_log_history.json`](../../experiments/lab3_v2/runs/distilbert_imdb/monitor/trainer_log_history.json); checkpoint Epoch 1 được chọn theo `min(eval_loss)` đạt **`89.10%` Acc / `0.2518` Loss** trên Held-Out Test (`1,000` mẫu).
- **Recommendation:** Khi fine-tune pretrained Transformer trên tập `< 10,000` mẫu, luôn dùng `load_best_model_at_end=True` kết hợp `metric_for_best_model="eval_loss"` (hoặc bật `label_smoothing_factor` + `warmup`) thay vì mặc định lấy checkpoint epoch cuối.

---

## 5. Truncation Slice Analysis Proves `max_length = 256` Increases Error Rate by `1.71×` on Long Reviews

- **Category:** Error Analysis & Best Practice
- **Description:** Đo riêng tỷ lệ lỗi giữa nhóm văn bản bị cắt (`orig_token_len > max_length`) và không bị cắt (`≤ max_length`) biến một nhận định cảm tính thành bằng chứng định lượng rõ ràng: trên Held-Out Test IMDB, `40.8%` review dài hơn `256` tokens chịu tỷ lệ lỗi **`14.46%`** so với chỉ **`8.45%`** ở nhóm không bị cắt (`6/10` lỗi tự tin cao nhất bị cắt mất câu kết luận & điểm số ở đoạn đuôi).
- **Evidence:** [`eda_summary.json`](../../experiments/lab3_v2/eda/eda_summary.json), [`error_examples.json`](../../experiments/lab3_v2/runs/distilbert_imdb/test_eval/error_examples.json) và [`evaluation_report.md` §4](../../labs/lab3_v2/docs/evaluation_report.md).
- **Recommendation:** Với mọi bài toán phân loại văn bản dài (review, bài báo, ticket), luôn đo tỷ lệ truncation ở bước EDA và thêm **Truncation Slice** vào Error Analysis; ưu tiên `max_length = 512` hoặc chiến lược **Head + Tail truncation** để giữ nguyên câu kết luận.

---

## Suggested Documentation Updates (Chờ Kỹ sư duyệt — không tự ý sửa)

1. **`.gitignore`:** Nên bổ sung thêm `*.safetensors` và `*.bin` vào nhóm `# weights` (bên cạnh `*.pt` và `runs/`) để phòng ngừa tuyệt đối việc vô tình commit trọng số Hugging Face (`~268 MB`/checkpoint) nếu cấu trúc thư mục thay đổi.
