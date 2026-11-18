---
title: "Pipeline 03: Training Loop — Lab 1 (Fashion-MNIST)"
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
  - ./01_DATA_PREPARATION.md
  - ./02_MODEL_ARCHITECTURE.md
  - ./04_EVALUATION_BENCHMARK.md
  - ../REPORT.md
  - ../../../docs/reference/DL_PIPELINE.md
  - ../../../../../.agents/temp/06_PIPELINE_SEPARATION.md
  - ../../../../../.agents/rules/BEHAVIORAL_LAYER.md
tags: [lab1, pipeline-03, training, baseline, checkpointing]
---

# Pipeline 03: Training Loop (Huấn Luyện, Baseline Sanity Check & Checkpointing)

Tài liệu đặc tả độc lập cho **Pipeline 3** trong hệ thống phân tách 5 pipeline của Lab 1, bám sát mục `Training` và `Baseline` tại [`labs/lab1/vibe/docs/REPORT.md`](../REPORT.md) và [`docs/reference/DL_PIPELINE.md`](../../../../../docs/reference/DL_PIPELINE.md).

---

## 1. Tài Liệu Đặc Tả Kỹ Thuật

### 1.1. Phạm vi & Ranh giới trách nhiệm
- **Trách nhiệm triển khai**:
  - Khóa ngẫu nhiên toàn cục (`random`, `numpy`, `torch`, `cuda`, `cudnn.deterministic`) bằng `seed = 248` trước mỗi lần chạy.
  - Điều phối vòng lặp huấn luyện chuẩn 5 bước trên `train_loader` và đánh giá trên `val_loader` sau mỗi epoch.
  - Hỗ trợ kiểm thử Baseline Sanity Check trên `LogisticClassifier` (kiểm tra loss khởi tạo $\approx \ln(10) \approx 2.3026$ và khả năng overfit 1 batch nhỏ).
  - Ghi log chi tiết từng epoch (`train_loss`, `val_loss`, `train_acc`, `val_acc`, `epoch_time_sec`) ra file `history.json` và console Rich.
  - Lưu checkpoint an toàn theo cơ chế ghi nguyên tử (atomic write `.tmp` → `replace`) cho cả `best.pt` và `last.pt`.
  - Hỗ trợ cơ chế Early Stopping tùy chọn (tắt mặc định khi chạy benchmark cố định số epoch để bảo đảm công bằng tuyệt đối).
- **Ngoài phạm vi (Out-of-Scope)**:
  - **Tuyệt đối không** đụng vào `test_loader` trong suốt quá trình huấn luyện hay chọn siêu tham số (`lr`, `best_epoch`).

### 1.2. Hợp đồng Đầu vào / Đầu ra (Input / Output Contracts)
- **Input**:
  - `model_name ∈ {"logistic", "mlp", "cnn"}` (khởi tạo từ Pipeline 02).
  - `train_loader`, `val_loader`, `norm_stats` (cung cấp từ Pipeline 01 theo `aug_enabled ∈ {False, True}`).
  - Cấu hình huấn luyện dùng chung cho cả 3 mô hình: `loss = CrossEntropyLoss`, `optimizer = Adam`, `lr`, `epochs`, `batch_size = 32`, `seed = 248`, `device`.
- **Output (lưu tách biệt theo từng mã thí nghiệm `<exp_id>`, ví dụ `cnn_aug`)**:
  - `experiments/lab1/<exp_id>/best.pt`: checkpoint tại epoch có chỉ số validation tốt nhất.
  - `experiments/lab1/<exp_id>/last.pt`: checkpoint tại epoch cuối cùng.
  - `experiments/lab1/<exp_id>/history.json`: lịch sử đầy đủ theo từng epoch (`epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `epoch_time_sec`, `best_epoch`, `total_train_time_sec`, `config`).

#### Schema bắt buộc bên trong file Checkpoint (`best.pt` / `last.pt`)
```python
{
    "model_name": str,          # "logistic" | "mlp" | "cnn"
    "aug_enabled": bool,
    "epoch": int,
    "best_metric_name": str,    # "val_loss" | "val_acc"
    "best_metric_value": float,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "norm_stats": dict,         # {"mean": float, "std": float} từ Pipeline 01
    "seed": int,                # 248
    "hyperparams": dict,        # {"lr": float, "batch_size": 32, "epochs": int}
}
```
Việc đính kèm `norm_stats` và `model_name` ngay trong checkpoint giúp Pipeline 04 và Pipeline 05 tự động tái dựng đúng kiến trúc và tham số chuẩn hóa, loại bỏ rủi ro lệch tiền xử lý khi suy luận.

### 1.3. Cơ Chế Vòng Lặp & Tính Toán An Toàn
1. **Bước Train (`model.train()`)**:
   - Đẩy `(images, labels)` lên `device` (`non_blocking=True` khi dùng `pin_memory`).
   - Thực thi đúng 5 bước: `logits = model(images)` → `loss = criterion(logits, labels)` → `optimizer.zero_grad(set_to_none=True)` → `loss.backward()` → `optimizer.step()`.
   - Tích lũy loss theo trọng số kích thước batch: `running_loss += loss.detach().item() * batch_size` (không cộng thẳng mean loss giữa các batch có kích thước cuối khác nhau và không giữ computational graph trên GPU).
2. **Bước Validation (`model.eval()` + `torch.no_grad()`)**:
   - Tính `val_loss` và `val_acc` trên toàn bộ 5.000 mẫu Validation.
   - Kiểm tra điều kiện cập nhật `best.pt` theo tiêu chí đã chốt; cập nhật `last.pt` sau mỗi epoch.
3. **Trình tự thực nghiệm chuẩn mực**:
   - Chạy **Baseline `Logistic` (`no_aug`)** đầu tiên để thiết lập mốc sàn và xác nhận toàn bộ hạ tầng huấn luyện vận hành trơn tru.
   - Chạy đủ ma trận 6 thí nghiệm (`{logistic, mlp, cnn} × {no_aug, aug}`) với cùng cấu hình để đảm bảo công bằng.

### 1.4. Cấu Hình Chuẩn Hóa (YAML Spec)

```yaml
training:
  seed: 248
  device: auto                  # cuda nếu khả dụng, ngược lại cpu
  loss: cross_entropy
  optimizer:
    name: adam
    lr: 0.001                   # [Cần kỹ sư chốt — xem lưu ý về src/config.py]
    weight_decay: 0.0
  epochs: 20                    # [Cần kỹ sư chốt — dùng chung cả 3 mô hình]
  best_checkpoint_metric: val_loss   # [Cần kỹ sư chốt: val_loss hoặc val_acc]
  early_stopping:
    enabled: false              # Tắt khi chạy benchmark công bằng cố định epoch
    patience: 5
    min_delta: 0.0005
  logging:
    backend: json_and_rich      # Không phụ thuộc cứng vào tensorboard (xem logs/errors/0001.md)
    output_dir: experiments/lab1
```

> **Điểm lưu ý quan trọng về Learning Rate & TensorBoard**:
> - `src/config.py` hiện đang đặt `LR = 1e-5` (quá nhỏ cho Adam huấn luyện từ đầu trên Fashion-MNIST trong 20–30 epoch). Cần kỹ sư xác nhận mức `lr` dùng chung (đề xuất `1e-3` hoặc chạy khảo sát nhanh lưới `{1e-3, 5e-4, 1e-4}` trên Validation cho cả 3 mô hình).
> - Môi trường hiện tại chưa cài gói `tensorboard` (đã ghi tại [`logs/errors/0001.md`](../../../../../logs/errors/0001.md)); do đó logger mặc định ghi ra `history.json` + bảng Rich và chỉ bật `SummaryWriter` tùy chọn khi `tensorboard` có sẵn.

### 1.5. Ánh Xạ Thư Mục
- **Mã nguồn dự kiến (`SRC.md`)**: `src/vibe_scripts/lab1/train.py` (có thể tái sử dụng ý tưởng atomic write từ [`src/utility.py`](../../../../../src/utility.py)).
- **Kiểm thử dự kiến**: `test/test_lab1_train.py`.

---

## 2. Prompt Độc Lập Cho Pipeline 03 (Behavioral Schema)

```markdown
Name:
Lab 1 Training Loop Pipeline

Description:
Xây dựng pipeline huấn luyện độc lập cho 3 mô hình (`logistic`, `mlp`, `cnn`) trên 2 cấu hình dữ liệu (`no_aug`, `aug`) bằng `CrossEntropyLoss` và `Adam`, tích hợp khóa seed toàn cục, sanity check, ghi log epoch và lưu checkpoint nguyên tử `best.pt`/`last.pt`.

Purpose:
Huấn luyện công bằng cả 6 cấu hình thí nghiệm theo đúng ràng buộc của `REPORT.md`, cung cấp trọng số hội tụ và lịch sử học đầy đủ cho Pipeline 04 (Evaluation) và Pipeline 05 (Inference).

Input:
- `train_loader`, `val_loader`, `norm_stats` từ `src/vibe_scripts/lab1/data.py`
- `model` từ `src/vibe_scripts/lab1/models.py`
- Tham số CLI / cấu hình: `--model {logistic,mlp,cnn}`, `--aug / --no-aug`, `--epochs`, `--lr`, `--batch-size 32`, `--seed 248`, `--output-dir experiments/lab1`

Output:
- Module & CLI script `src/vibe_scripts/lab1/train.py`
- Artifacts mỗi thí nghiệm tại `experiments/lab1/<model>_<no_aug|aug>/`: `best.pt`, `last.pt`, `history.json`
- Bộ unit test `test/test_lab1_train.py`

How to do:
1. Triển khai hàm `set_global_seed(seed=248)` khóa `random`, `numpy`, `torch`, `cuda` và `cudnn.deterministic`.
2. Triển khai hàm `train_one_epoch()` theo đúng 5 bước chuẩn (`forward -> loss -> zero_grad -> backward -> step`) và `validate_one_epoch()` dưới `model.eval()` + `torch.no_grad()`.
3. Tích lũy loss theo số mẫu thực tế (weighted by `batch_size`), luôn `.detach().item()` để tránh rò rỉ VRAM; ném lỗi ngay nếu loss là `NaN`/`Inf`.
4. Triển khai `save_checkpoint_atomic()` lưu đầy đủ `model_state_dict`, `optimizer_state_dict`, `norm_stats`, `model_name`, `epoch`, `best_metric_value`, `seed`, `hyperparams`.
5. Hỗ trợ chế độ `--sanity-check` (kiểm tra initial loss ~ ln(10) và overfit 1 batch) và tùy chọn Early Stopping có thể bật/tắt qua cờ cấu hình.
6. Ghi log `history.json` sau mỗi epoch và hỗ trợ chạy CLI cho từng thí nghiệm đơn lẻ hoặc toàn bộ ma trận 3×2.
7. Viết Smoke Test (1 epoch với subset nhỏ `< 5s`) và Normal Test tại `test/test_lab1_train.py`.

Examples:
python -m src.vibe_scripts.lab1.train --model logistic --no-aug --epochs 20 --lr 0.001 --seed 248
pytest test/test_lab1_train.py -v
```

---

## 3. Checklist Kiểm Thử Độc Lập (Definition of Done)

- [ ] **Baseline Sanity Check (Initial Loss)**: Loss khởi tạo trước khi cập nhật trọng số của `Logistic` nằm trong khoảng `[2.1, 2.5]` (gần $\ln(10) \approx 2.3026$).
- [ ] **Overfit 1 Batch**: Cả 3 mô hình có thể giảm loss về `< 0.1` và đạt accuracy `> 95%` khi huấn luyện lặp trên 1 batch cố định (32 mẫu), chứng minh luồng gradient thông suốt.
- [ ] **Bất biến cách ly Test Set**: Không có lệnh gọi hay tham chiếu `test_loader` bên trong `train.py`.
- [ ] **Checkpoint đầy đủ & Nguyên tử**: Sau khi chạy, thư mục đích chứa cả `best.pt`, `last.pt` và `history.json`; không để lại file rác `.tmp`.
- [ ] **Đầy đủ Metadata trong Checkpoint**: Nạp lại `best.pt` bằng `torch.load(..., weights_only=False)` xác nhận có đủ `model_name`, `norm_stats`, `model_state_dict`, `epoch`, `seed`.
- [ ] **Early Stopping & Logging**: Unit test xác minh Early Stopping dừng đúng sau `patience` epoch không cải thiện khi bật cờ; `history.json` có đủ `train_loss`, `val_loss`, `train_acc`, `val_acc`, `epoch_time_sec`.
- [ ] **An toàn bộ nhớ & số học**: Không rò rỉ VRAM qua các epoch (dùng `.detach().item()`); bắt và dừng ngay nếu phát sinh `NaN`/`Inf` loss.
- [ ] **Smoke Test `< 5s`**: Chạy 1 epoch trên tập con 128 mẫu pass 100% không lỗi.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo đặc tả độc lập Pipeline 03 (Training Loop) bám sát `REPORT.md`, `DL_PIPELINE.md` và `PIPELINE_SEPARATION.md` |
