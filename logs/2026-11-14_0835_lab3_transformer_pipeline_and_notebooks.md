# Thiết kế Pipeline Phân tích Cảm xúc DistilBERT & Hoàn thiện Notebooks Lab 3

- **Date/Time:** 2026-11-14 08:35
- **Agent/Role:** Deep-Learning-Architect-Agent
- **Status:** Completed
- **Related Files:** 
  - `docs/lab3/README.md`
  - `docs/lab3/PIPELINE_EXPLANATION.md`
  - `labs/lab3/main/01_ex1_sentiment_analysis.ipynb`
  - `labs/lab3/main/02_ex2_finetune_sentiment.ipynb`
  - `labs/lab3/main/main.ipynb`
  - `labs/lab3/main/fine-tune.ipynb`
  - `notebooks/lab3/01_ex1_sentiment_analysis.ipynb`
  - `notebooks/lab3/02_ex2_finetune_sentiment.ipynb`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)
- **Yêu cầu ban đầu:** Người dùng yêu cầu giải quyết bài tập Lab 3 (Hugging Face Sentiment Analysis & Fine-tuning Transformer), tuân thủ nghiêm ngặt các quy tắc hiến pháp của agent (`AGENT_AI.md`, `NOTEBOOK_HEADER_CONVENTION.md`, `EXPLAIN_PIPELINE.md`, `RESULTS_REPORTING.md`, `PYTORCH_FRAMEWORK_RULES.md`), định dạng `.ipynb` (tách thành nhiều file nếu dài), và tài liệu giải thích đặc trưng từng module luồng pipeline đặt tại `docs/lab3/`.
- **Vấn đề cốt lõi cần giải quyết:** 
  1. Thiếu tài liệu phân tích kỹ thuật theo chuẩn `EXPLAIN_PIPELINE.md` tại `docs/lab3/`. Cần xây dựng bản bóc tách toàn diện 10 module theo 6 trục bắt buộc: *Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp*.
  2. Các notebook ban đầu chưa có Markdown Header Cell chuẩn 4 phần theo `NOTEBOOK_HEADER_CONVENTION.md`, một số cell gặp lỗi tham chiếu biến chưa định nghĩa (`DOCUMENTS[0]`), và chưa có cơ chế kiểm tra checkpoint để đảm bảo nguyên tắc Cell Independence.
  3. Cần tách bạch rõ ràng giữa Exercise 1 (suy luận pre-trained & mổ xẻ tokenizer/PyTorch tensors) và Exercise 2 (Fine-tuning DistilBERT trên SST-2).

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)
- **Các phương án đã cân nhắc (Alternatives Considered):**
  - *Phương án A: Gom toàn bộ vào một notebook duy nhất.*
    - Ưu điểm: Chỉ có 1 file.
    - Nhược điểm: Quá dài, vi phạm gợi ý của người dùng, pha trộn giữa pre-trained exploration và full training loop, khó theo dõi và debug.
  - *Phương án B: Tách thành 2 notebooks chuẩn mực (`01_ex1_sentiment_analysis.ipynb` và `02_ex2_finetune_sentiment.ipynb`), đồng thời đồng bộ hóa sang cả `labs/lab3/main/` và `notebooks/lab3/`.*
    - Ưu điểm: Phản ánh trực tiếp 2 bài tập trong `QUESTIOn.md`, mỗi notebook độc lập, tuân thủ `NOTEBOOK_HEADER_CONVENTION.md` và `FOLDER_STRUCTURE.md`.
- **Quyết định cuối cùng (Decision Made):** Chọn **Phương án B**.
  - Xây dựng 2 notebook độc lập với Roadmap Table và References đầy đủ liên kết clickable.
  - Viết tài liệu `docs/lab3/PIPELINE_EXPLANATION.md` chi tiết 10 module và `docs/lab3/README.md` lưu trữ bảng số liệu 5W1H (Accuracy: 90.37%, F1: 90.41% tại checkpoint-2105).
- **Giả định (Assumptions):** Checkpoint huấn luyện đã được thực hiện và lưu sẵn tại `labs/lab3/main/sst2-distilbert/checkpoint-2105`, do đó notebook Exercise 2 bổ sung cơ chế kiểm tra: nếu checkpoint đã có sẵn, ưu tiên nạp checkpoint để người dùng có thể chạy ngay bước đánh giá mà không cần chờ GPU/CPU train lại từ đầu.

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)
- **`docs/lab3/PIPELINE_EXPLANATION.md`**:
  - Triển khai sơ đồ Mermaid cho toàn bộ pipeline từ Raw Sentences $\rightarrow$ Tokenization $\rightarrow$ Dynamic Collator $\rightarrow$ DistilBERT Backbone $\rightarrow$ Classification Head $\rightarrow$ Optimizer (AdamW + FP16) $\rightarrow$ Evaluation & Inference.
  - Phân tích 10 module chi tiết đủ 6 mục theo `EXPLAIN_PIPELINE.md`.
- **`labs/lab3/main/01_ex1_sentiment_analysis.ipynb`**:
  - Step 1: Environment & device check (`torch.device`, seed 42).
  - Step 2: High-level `pipeline("sentiment-analysis")`.
  - Step 3: Mổ xẻ WordPiece tokenization, tokens, token IDs, special tokens `[CLS]`, `[SEP]`, `[PAD]`.
  - Step 4: Low-level forward pass bằng PyTorch: truyền tensor, lấy `logits`, tính xác suất bằng `F.softmax`, trích xuất nhãn và độ tin cậy.
  - Step 5: Minh họa `attention_mask` chặn tính attention lên token đệm.
  - Step 6: Kiểm thử câu phủ định và mỉa mai.
- **`labs/lab3/main/02_ex2_finetune_sentiment.ipynb`**:
  - Step 1: Thiết lập seed và kiểm tra VRAM ($\le 3.5\text{GB}$).
  - Step 2: Ingestion SST-2 từ `nyu-mll/glue`, trực quan hóa phân phối nhãn và histogram độ dài từ.
  - Step 3: Tokenization với `max_length=64` và `DataCollatorWithPadding`.
  - Step 4: Khởi tạo mô hình DistilBERT + binary classifier head.
  - Step 5: `TrainingArguments` (lr=2e-5, fp16=True, weight_decay=0.01, eval_strategy="epoch", metric_for_best_model="f1") kèm cơ chế phát hiện checkpoint.
  - Step 6: Đánh giá định lượng với Báo cáo phân loại (Classification Report) và Confusion Matrix Display chuẩn 5W1H.
  - Step 7: Phân tích lỗi (FP vs FN, top-8 lỗi tự tin cao, phân tích tỷ lệ lỗi theo nhóm độ dài câu, so sánh confidence trung bình).
  - Step 8: Phép thử câu phủ định *"It's not bad at all"*.

## 4. Khó khăn & Cách khắc phục (Challenges & Solutions)
- **Khó khăn:** Khi chạy inline script tạo notebook qua shell, dấu backtick trong Markdown bị bash hiểu lầm là command substitution.
- **Khắc phục:** Tạo script Python trung gian chuyên trách (`temp/build_lab3_notebooks.py`) và thực thi bằng Python, đảm bảo nội dung JSON của file `.ipynb` hoàn toàn nguyên vẹn và hợp lệ.

## 5. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)
- **Trạng thái hiện tại:** Bài tập Lab 3 đã hoàn thiện đầy đủ cả về mã nguồn notebook (`labs/lab3/main/`, `notebooks/lab3/`) lẫn tài liệu kiến trúc kỹ thuật (`docs/lab3/`). Checkpoint tốt nhất tại `checkpoint-2105` đã được ghi nhận.
- **Cảnh báo / Lưu ý (Gotchas/Warnings):**
  - Không tự ý xóa thư mục `sst2-distilbert/checkpoint-2105/` vì đó là checkpoint tối ưu đã đạt F1 = 90.41%.
  - Giữ nguyên cài đặt `max_length=64` và `DataCollatorWithPadding` để duy trì mức tiêu thụ VRAM an toàn dưới 3.5GB.
- **Đề xuất bước tiếp theo (Next Actions):**
  - Có thể mở notebook trên Jupyter/VS Code/Antigravity IDE và chạy tuần tự các cell để nghiệm thu.
  - Nếu cần kiểm thử đối kháng sâu hơn, có thể tích hợp kỹ thuật Data Denoising (Cleanlab) hoặc LoRA Parameter-Efficient Fine-Tuning.

