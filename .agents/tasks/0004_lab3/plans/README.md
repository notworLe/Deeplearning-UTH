# Kế Hoạch Thực Thi Lab 03 — Sửa Chữa Notebooks (`labs/lab3/main/`) & Tài Liệu Thuyết Trình (`labs/lab3/docs/`)

## Bảng Theo Dõi Tiến Độ (Progress Tracking Board)

| Task ID | Tên công việc | Phụ trách | Trạng thái | Blockers / Ghi chú |
| :---: | :--- | :--- | :---: | :--- |
| **T-01** | Kiểm toán môi trường venv `anything` & ghi nhận lỗi thiếu thư viện vào `logs/errors/` | AI Agent | `Done` | Đã ghi nhận `ERR-002` tại `logs/errors/` |
| **T-02** | Cài đặt `datasets`, `evaluate`, `accelerate>=1.1.0` vào venv `anything` | AI Agent | `Done` | Đã cài `datasets-5.2.0`, `evaluate-0.4.6`, `accelerate-1.16.0` |
| **T-03** | Chuẩn hóa & sửa lỗi Notebooks Exercise 1 (`01_ex1_sentiment_analysis.ipynb`, `main.ipynb`) | AI Agent | `Done` | Đã tách Cell 0/Cell 1, sửa link và chạy thực thi 6/6 cells |
| **T-04** | Chuẩn hóa & sửa lỗi Notebooks Exercise 2 (`02_ex2_finetune_sentiment.ipynb`, `fine-tune.ipynb`) | AI Agent | `Done` | Đã chạy thực thi 8/8 cells (3 epochs SST-2, Val F1 = 90.99%) |
| **T-05** | Viết tài liệu thuyết trình (`PRESENTATION`) cho từng Notebook tại `labs/lab3/docs/` | AI Agent | `Done` | Đã tạo `PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md`, `PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md`, `README.md` |
| **T-06** | Kiểm thử Smoke & Normal Test toàn bộ Notebooks + Audit đường dẫn & Ghi log `logs/thinking/` | AI Agent | `Done` | Đã pass `nbformat.validate` 4/4 notebooks và kiểm tra 100% link tương đối |

---

## 1. Plan (Mục tiêu & Phạm vi)
- **Goal**: Khắc phục toàn bộ lỗi môi trường và cấu hình khiến các Jupyter Notebooks trong [`labs/lab3/main/`](../../../labs/lab3/main/) không chạy được; thực thi thực tế để nhúng đầy đủ kết quả đầu ra (outputs/charts) và biên soạn bộ tài liệu thuyết trình (`PRESENTATION`) chi tiết cho từng notebook tại [`labs/lab3/docs/`](../../../labs/lab3/docs/).
- **In-Scope**:
  - Cài đặt các gói Hugging Face còn thiếu (`datasets`, `evaluate`, `accelerate>=1.1.0`) vào venv `anything`.
  - Chuẩn hóa cấu trúc Cell 0 (Header & Mục tiêu) và Cell 1 (Bảng Roadmap 4 cột + Related Documentation References với đường dẫn tương đối sống 100%) theo [`AI_REFERENCE.md`](../../../.agents/rules/AI_REFERENCE.md) cho các notebook trong `labs/lab3/main/`.
  - Sửa lỗi cấu hình `TrainingArguments` trên Windows (`dataloader_num_workers=0` trong môi trường interactive notebook để tránh lỗi multiprocessing spawn), tự động tìm/lưu checkpoint tối ưu trong `sst2-distilbert/`.
  - Chạy thực thi toàn bộ cells của `01_ex1_sentiment_analysis.ipynb` (đồng bộ `main.ipynb`) và `02_ex2_finetune_sentiment.ipynb` (đồng bộ `fine-tune.ipynb`).
  - Viết file thuyết trình riêng cho từng notebook tại `labs/lab3/docs/`.
- **Out-of-Scope**:
  - Không thay đổi mã nguồn core trong `src/`.

---

## 2. Pipeline Mapping
- **Target Notebooks**:
  - [`labs/lab3/main/01_ex1_sentiment_analysis.ipynb`](../../../labs/lab3/main/01_ex1_sentiment_analysis.ipynb) & [`labs/lab3/main/main.ipynb`](../../../labs/lab3/main/main.ipynb)
  - [`labs/lab3/main/02_ex2_finetune_sentiment.ipynb`](../../../labs/lab3/main/02_ex2_finetune_sentiment.ipynb) & [`labs/lab3/main/fine-tune.ipynb`](../../../labs/lab3/main/fine-tune.ipynb)
- **Target Documentation (`labs/lab3/docs/`)**:
  - [`labs/lab3/docs/README.md`](../../../labs/lab3/docs/README.md)
  - [`labs/lab3/docs/PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md`](../../../labs/lab3/docs/PRESENTATION_01_EX1_SENTIMENT_ANALYSIS.md)
  - [`labs/lab3/docs/PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md`](../../../labs/lab3/docs/PRESENTATION_02_EX2_FINETUNE_SENTIMENT.md)
- **Logs**:
  - [`logs/errors/README.md`](../../../logs/errors/README.md) & [`logs/errors/ERR-002_lab3_missing_hf_dependencies.md`](../../../logs/errors/ERR-002_lab3_missing_hf_dependencies.md)
  - [`logs/thinking/INDEX.md`](../../../logs/thinking/INDEX.md)

---

## 3. Detailed Execution Plan
- [x] **Step 1**: Kiểm tra phiên bản `transformers`, `torch` và phát hiện thiếu `datasets`, `evaluate`, `accelerate>=1.1.0` trong venv `anything`.
- [x] **Step 2**: Ghi nhận `ERR-002` vào `logs/errors/` và cài đặt `datasets`, `evaluate`, `accelerate>=1.1.0`.
- [x] **Step 3**: Chuẩn hóa Cell 0 & Cell 1 và thực thi `01_ex1_sentiment_analysis.ipynb` (lọc bỏ cảnh báo symlink chứa đường dẫn cá nhân Windows), đồng bộ sang `main.ipynb`.
- [x] **Step 4**: Chuẩn hóa Cell 0 & Cell 1, cập nhật `dataloader_num_workers=0`, cơ chế phát hiện checkpoint tự động trong `02_ex2_finetune_sentiment.ipynb`, chạy huấn luyện & đánh giá thực tế trên GPU, đồng bộ sang `fine-tune.ipynb` và `fine-tune.py`.
- [x] **Step 5**: Biên soạn bộ tài liệu thuyết trình chi tiết từng cell, giải thích biểu diễn tensor/pipeline và bộ câu hỏi phản biện Q&A cho từng notebook tại `labs/lab3/docs/`.
- [x] **Step 6**: Kiểm tra chất lượng (Gate 3, 4, 5) và cập nhật `logs/thinking/`.

---

## 4. Definition of Done (DoD)
- [x] Cả 4 file notebook trong `labs/lab3/main/` chạy thành công 100% cells từ đầu đến cuối, có đầy đủ output thực tế và biểu đồ trực quan hóa.
- [x] Tuân thủ chuẩn Cell 0 & Cell 1 tại `AI_REFERENCE.md` §5, không chứa sơ đồ Mermaid trong notebook, 100% link tương đối hoạt động, không rò rỉ đường dẫn cá nhân `C:\Users\...`.
- [x] Thư mục `labs/lab3/docs/` có đầy đủ file thuyết trình (`PRESENTATION`) cho từng notebook kèm mục lục điều hướng.
