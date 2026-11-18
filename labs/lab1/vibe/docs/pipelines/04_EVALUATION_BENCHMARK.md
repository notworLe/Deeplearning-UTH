---
title: "Pipeline 04: Evaluation & Benchmark — Lab 1 (Fashion-MNIST)"
doc_type: pipeline-spec
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-18
updated: 2026-11-18
last_updated_by: Antigravity
version: 0.1.0
status: draft
reviewers: []
related_docs:
  - ../README.md
  - ./README.md
  - ./03_TRAINING.md
  - ./05_INFERENCE_SERVING.md
  - ../REPORT.md
  - ../../../docs/reference/DL_PIPELINE.md
  - ../../../../../.agents/temp/06_PIPELINE_SEPARATION.md
  - ../../../../../.agents/rules/BEHAVIORAL_LAYER.md
tags: [lab1, pipeline-04, evaluation, benchmark, error-analysis]
---

# Pipeline 04: Evaluation & Benchmark (Đánh Giá Định Lượng & Phân Tích Lỗi)

Tài liệu đặc tả độc lập cho **Pipeline 4** trong hệ thống phân tách 5 pipeline của Lab 1, bám sát mục `Evaluate` và `Error analysis` tại [`labs/lab1/vibe/docs/REPORT.md`](../REPORT.md) và [`docs/reference/DL_PIPELINE.md`](../../../../../docs/reference/DL_PIPELINE.md).

---

## 1. Tài Liệu Đặc Tả Kỹ Thuật

### 1.1. Nguyên Tắc & Phạm Vi Trách Nhiệm
- **Nguyên tắc vàng đánh giá**:
  - Tập Test (10.000 mẫu) chỉ được sử dụng **đúng một lần duy nhất** ở giai đoạn cuối cùng bằng checkpoint `best.pt` đã chọn từ tập Validation.
  - Mọi phép suy luận đánh giá bắt buộc chạy dưới `model.eval()` và `torch.no_grad()`.
- **Trách nhiệm triển khai**:
  - Đo đầy đủ bộ chỉ số định lượng cho toàn bộ **6 thí nghiệm (3 mô hình × 2 cấu hình)**: Accuracy, Loss, Precision/Recall/F1 theo từng lớp (10 lớp + Macro), Confusion Matrix 10×10, số tham số, epoch hội tụ (`best_epoch`), kích thước file checkpoint (KB) và thời gian suy luận (ms).
  - Trực quan hóa đường cong học (Loss & Accuracy của Train và Validation theo epoch).
  - Phân tích theo lớp và phân tích lỗi sâu (Error Analysis): Confusion Matrix chuẩn hóa, bảng top cặp lớp nhầm lẫn lớn nhất, lưới ảnh dự đoán Đúng / Sai kèm xác suất dự đoán (đặc biệt lỗi tự tin cao).
  - Kiểm chứng toàn vẹn cơ chế Save/Load checkpoint theo yêu cầu đề bài.
  - Tổng hợp số liệu phục vụ kiểm chứng 3 giả thuyết `H1`, `H2`, `H3` và điền kết quả vào `labs/lab1/vibe/docs/REPORT.md`.

### 1.2. Hợp đồng Đầu vào / Đầu ra (Input / Output Contracts)
- **Input**:
  - Thư mục chứa kết quả 6 lần huấn luyện từ Pipeline 03: `experiments/lab1/<model>_<no_aug|aug>/{best.pt, history.json}`.
  - `test_loader` (10.000 mẫu, `shuffle=False`, không augmentation) từ Pipeline 01, sử dụng đúng `norm_stats` đã lưu trong checkpoint.
- **Output**:
  - `experiments/lab1/evaluation/benchmark_summary.json` và `benchmark_table.csv`: bảng tổng hợp 3 mô hình × 2 cấu hình.
  - `experiments/lab1/evaluation/<exp_id>/per_class_metrics.json`: Precision, Recall, F1-Score, Support cho từng lớp trong 10 lớp.
  - Thư mục hình ảnh trực quan tại `experiments/lab1/predictions/`:
    - `learning_curves_comparison.png`: đường cong Train/Val Loss và Accuracy cho cả 3 mô hình × 2 cấu hình.
    - `confusion_matrix_<exp_id>.png`: ma trận nhầm lẫn 10×10 (hiển thị cả số đếm và tỷ lệ chuẩn hóa theo hàng).
    - `per_class_f1_comparison.png`: biểu đồ cột so sánh F1-Score theo từng lớp giữa `Logistic`, `MLP`, `CNN`.
    - `predictions_gallery_<exp_id>.png`: lưới ảnh dự đoán Đúng (viền xanh) và Sai (viền đỏ) kèm nhãn thật, nhãn dự đoán và xác suất Softmax.
    - `top_errors_<exp_id>.png`: các mẫu sai có độ tự tin cao nhất để phân tích nguyên nhân hình học.

### 1.3. Cấu Trúc Bảng Tổng Hợp 3 Mô Hình × 2 Cấu Hình

| Mô hình | Cấu hình | Params | Best Epoch | File Size (KB) | Test Loss | Test Acc (%) | Macro F1 (%) | Latency (ms/batch) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Logistic` | `no_aug` | `7.850` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Logistic` | `aug` | `7.850` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `MLP` | `no_aug` | `109.386` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `MLP` | `aug` | `109.386` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `CNN` | `no_aug` | `105.866` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `CNN` | `aug` | `105.866` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 1.4. Kiểm Chứng Save/Load & Phân Tích Lỗi
1. **Giao thức kiểm tra Save/Load**:
   - Nạp lại `best.pt` vào một đối tượng mô hình khởi tạo hoàn toàn mới bằng `build_model(ckpt["model_name"])`.
   - Chạy cùng một batch cố định từ tập Test trên hai bản thể trước và sau khi lưu/nạp lại; xác nhận $\max |\text{logits}_{\text{before}} - \text{logits}_{\text{after}}| < 10^{-6}$ và `Test Accuracy` giống hệt nhau.
2. **Giao thức Phân tích lỗi (Error Analysis)**:
   - Trích xuất 5 cặp nhầm lẫn `(true_class -> pred_class)` có tần suất cao nhất trên mô hình tốt nhất và đối chiếu với `Logistic` / `MLP`.
   - Kiểm chứng giả thuyết `H3`: định lượng phần trăm tổng số lỗi rơi vào nhóm trang phục thân trên (`Shirt`, `T-shirt/top`, `Pullover`, `Coat`).
   - Trực quan hóa ảnh gốc (đảo ngược `Normalize` bằng `img * std + mean` trước khi vẽ để ảnh hiển thị đúng độ tương phản gốc!).

### 1.5. Cấu Hình Chuẩn Hóa (YAML Spec)

```yaml
evaluation:
  experiments_dir: experiments/lab1
  output_dir: experiments/lab1/evaluation
  predictions_dir: experiments/lab1/predictions
  batch_size: 32
  device: auto
  warmup_batches: 5
  gallery:
    correct_samples: 10
    incorrect_samples: 10
    top_confident_errors: 10
  save_load_tolerance: 1.0e-6
```

### 1.6. Ánh Xạ Thư Mục
- **Mã nguồn dự kiến (`SRC.md`)**: `src/vibe_scripts/lab1/evaluate.py` (không phụ thuộc vào `src/visualize.py` hay `src/objects.py` đang có lỗi ở [`logs/errors/0001.md`](../../../../../logs/errors/0001.md)).
- **Kiểm thử dự kiến**: `test/test_lab1_evaluate.py`.

---

## 2. Prompt Độc Lập Cho Pipeline 04 (Behavioral Schema)

```markdown
Name:
Lab 1 Evaluation & Benchmark Pipeline

Description:
Xây dựng pipeline đánh giá toàn diện trên tập Test (10.000 mẫu) cho ma trận 3 mô hình × 2 cấu hình, tính đầy đủ chỉ số định lượng (Accuracy, per-class Precision/Recall/F1, Confusion Matrix, Params, Best Epoch, File Size, Latency), vẽ đồ thị học và trực quan hóa phân tích lỗi.

Purpose:
Cung cấp bằng chứng định lượng và định tính khách quan để đối sánh Logistic, MLP, CNN, kiểm chứng các giả thuyết H1-H3 và xác thực tính toàn vẹn Save/Load của mô hình.

Input:
- Các thư mục kết quả huấn luyện `experiments/lab1/<exp_id>/{best.pt, history.json}`
- `test_loader` (10.000 ảnh, `shuffle=False`) từ `src/vibe_scripts/lab1/data.py`
- Kiến trúc mô hình từ `src/vibe_scripts/lab1/models.py`

Output:
- Module & CLI script `src/vibe_scripts/lab1/evaluate.py`
- Bảng số liệu tổng hợp: `experiments/lab1/evaluation/benchmark_summary.json`, `benchmark_table.csv` và bảng Rich trên console
- Biểu đồ và lưới ảnh lỗi tại `experiments/lab1/predictions/`: learning curves, confusion matrices, per-class F1, correct/incorrect prediction galleries, top confident errors
- Bộ unit test `test/test_lab1_evaluate.py`

How to do:
1. Viết hàm `evaluate_checkpoint(ckpt_path, test_loader, device)` chạy hoàn toàn trong `model.eval()` và `torch.no_grad()`, tự động khởi tạo đúng kiến trúc từ `ckpt["model_name"]`.
2. Tính Test Loss, Test Accuracy, Confusion Matrix 10×10, per-class và macro Precision/Recall/F1 (`zero_division=0`), đếm số tham số, kích thước file `best.pt` (KB) và đo thời gian suy luận sau bước warm-up (có `torch.cuda.synchronize()` khi chạy GPU).
3. Viết hàm `verify_save_load_integrity()` kiểm chứng sai số logits `< 1e-6` sau chu trình lưu/nạp lại trọng số.
4. Viết các hàm trực quan hóa (dùng `matplotlib` với `sharey=False` và khử `Normalize` trước khi vẽ ảnh) lưu độ phân giải rõ nét vào `experiments/lab1/predictions/`.
5. Xuất bảng so sánh 3×2 ra Rich Table, CSV và JSON; liệt kê top cặp lớp nhầm lẫn lớn nhất phục vụ mục Error Analysis.
6. Viết Smoke Test (chạy với checkpoint giả lập và 64 mẫu `< 5s`) và Normal Test (kiểm tra bất biến Confusion Matrix và Save/Load) tại `test/test_lab1_evaluate.py`.

Examples:
python -m src.vibe_scripts.lab1.evaluate --experiments-dir experiments/lab1
pytest test/test_lab1_evaluate.py -v
```

---

## 3. Checklist Kiểm Thử Độc Lập (Definition of Done)

- [ ] **Chế độ Đánh giá An toàn**: 100% quá trình chạy Test dùng `model.eval()` và `torch.no_grad()`; không cập nhật trọng số mô hình.
- [ ] **Bất biến Confusion Matrix**: Tổng toàn bộ phần tử ma trận 10×10 bằng đúng `10.000`; tổng từng hàng bằng đúng `1.000` (số mẫu mỗi lớp trong Test set).
- [ ] **Bất biến Chỉ số**: `Accuracy`, `Precision`, `Recall`, `F1` đều nằm trong đoạn `[0.0, 1.0]`; `Accuracy` tính từ `diag(CM).sum() / CM.sum()` khớp với `accuracy_score`.
- [ ] **Xác thực Save/Load**: Nạp lại `best.pt` vào mô hình mới cho sai số logits tối đa `< 1e-6` và nhãn dự đoán khớp 100%.
- [ ] **Đo thời gian chính xác**: Có warm-up trước khi bấm giờ và dùng `torch.cuda.synchronize()` nếu chạy trên GPU.
- [ ] **Đầy đủ Artifacts trực quan**: Xuất đủ learning curves, confusion matrix, per-class F1, lưới ảnh đúng/sai (đã khử chuẩn hóa về `[0, 1]`) vào `experiments/lab1/predictions/` không có lỗi khoảng trắng dư.
- [ ] **Bảng đối sánh 3×2**: File `benchmark_summary.json` và `benchmark_table.csv` ghi đủ 6 cấu hình kèm đầy đủ các cột theo `REPORT.md`.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo đặc tả độc lập Pipeline 04 (Evaluation & Benchmark) bám sát `REPORT.md`, `DL_PIPELINE.md` và `PIPELINE_SEPARATION.md` |
