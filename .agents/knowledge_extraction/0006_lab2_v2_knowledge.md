---
title: Knowledge Extraction — Lab 2 v2 (Oxford-IIIT Pets Pre-trained CNNs Pipeline)
doc_type: Knowledge
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-28
updated: 2026-11-28
version: 1.0.0
status: completed
---

# Knowledge Extraction: `0006_lab2_v2`

> **Source Log:** [`0006_lab2_v2_log.md`](../logs/0006_lab2_v2_log.md) · **Standard:** [`KNOWLEDGE_EXTRACTION.md`](../rules/dl-rule/KNOWLEDGE_EXTRACTION.md)

---

## 1. Frozen `BatchNorm2d` Must Stay in `.eval()` During Transfer Learning & Partial Fine-Tuning

- **Category:** Model Engineering
- **Description:** Trong PyTorch, gọi `model.train()` trên một pre-trained CNN đã đóng băng (`requires_grad = False`) vẫn chuyển toàn bộ `BatchNorm2d` sang chế độ training — làm chúng cập nhật `running_mean / running_var` theo mini-batch nhỏ của tập mới (`3,680` ảnh) và làm lệch thống kê đặc trưng ImageNet đã học sẵn.
- **Evidence:** Override `PetClassifier.train()` tại [`models.py`](../../src/vibe_scripts/lab2_v2/models.py#L128-L137) để giữ mọi `BatchNorm2d` có toàn bộ tham số frozen ở trạng thái `.eval()` (được kiểm chứng bằng unit test `test_frozen_bn_eval_guard_shape_validation_and_trace`), giúp mọi mô hình Transfer đạt `86%–89%` Val Acc ngay sau Epoch 1 và Save/Load Parity `0.0`.
- **Recommendation:** Mọi wrapper Transfer Learning / Partial Fine-Tuning trong `src/vibe_scripts/` cần khóa `BatchNorm2d` đóng băng ở `.eval()` trong phương thức `train(self, mode=True)` và có unit test hồi quy đi kèm.

---

## 2. Dense Feature Reuse (`DenseNet121`) Outperforms Heavier Backbones & Wide Fine-Tuning on Medium Fine-Grained Datasets

- **Category:** Model Engineering
- **Description:** Trên tập phân loại ảnh tinh cỡ vừa (`3,680` ảnh / `37` giống), chọn backbone có biểu diễn đa tầng giàu chi tiết quan trọng hơn tăng số tham số hay mở khóa diện rộng: `DenseNet121` ghép nối feature maps từ tầng thấp (họa tiết lông, màu mắt) tới tầng cao (hình khối sọ/tai) thành vector `1,024-d`, chỉ cần train `37,925` tham số (`0.54%`) đã đạt **`93.22%` Val / `91.91%` Test** — vượt cả `resnet18_finetune` (`8.41M` trainable, `90.97%` Val, `train_acc = 100%`) lẫn `vgg16_transfer` (`134.41M` params, `89.72%` Val, `1,318 MB` VRAM).
- **Evidence:** Bảng xếp hạng `9/9` runs và phân tích so sánh kiến trúc tại [`evaluation_report.md` §2.3](../../labs/lab2_v2/docs/evaluation_report.md).
- **Recommendation:** Với dữ liệu ảnh tự nhiên dưới `10k` mẫu, luôn benchmark ít nhất một kiến trúc mật độ đặc trưng cao (`DenseNet` / `ConvNeXt` / `EfficientNet`) ở chế độ Transfer sạch trước khi mở khóa fine-tune diện rộng dễ gây overfit.

---

## 3. Batch-Size-Dependent GPU Latency Inversion (`DenseNet121` vs `ResNet18` vs `VGG16`)

- **Category:** Deployment & Performance
- **Description:** Thứ hạng độ trễ suy luận trên GPU đảo ngược mạnh giữa `BS = 1` và `BS = 32` tùy nút thắt hệ thống:
  - Ở **`BS = 1` (kernel-launch & memory-bandwidth bound):** `DenseNet121` (`20.13 ms`) chậm hơn `ResNet18` (`3.17 ms`) tới `6.3×` và thậm chí chậm hơn `VGG16` (`5.94 ms`) dù ít tham số hơn `19×`, vì `121` tầng sâu với nhiều phép `torch.cat` tuần tự.
  - Ở **`BS = 32` (compute bound):** GPU bão hòa tính toán nên `DenseNet121` đạt `1.88 ms/ảnh` — **nhanh gấp `2×` `VGG16`** (`3.77 ms/ảnh`, bị nghẽn bởi `15.5 GFLOPs` dày đặc), trong khi `ResNet18` vẫn nhanh nhất (`0.55 ms/ảnh`).
- **Evidence:** Số liệu đo trực tiếp trên RTX 4060 tại [`inference_benchmark.json`](../../experiments/lab2_v2/evaluation/inference_benchmark.json) và [`evaluation_report.md` §5.2](../../labs/lab2_v2/docs/evaluation_report.md).
- **Recommendation:** Không bao giờ đánh giá tốc độ inference chỉ dựa trên số lượng tham số (params) hay chỉ ở 1 batch size; luôn benchmark đồng thời `BS = 1` (realtime latency) và `BS = 32` (batch throughput).

---

## 4. Distinguishing True Overfitting vs Benign Generalization Gap vs LR-Decay Underfitting

- **Category:** Diagnostics & Regularization
- **Description:** Khoảng cách `gap(train_acc − val_acc)` nhỏ không đồng nghĩa với mô hình tốt, và khoảng cách trung bình cũng không đồng nghĩa với overfitting hỏng mô hình:
  - `resnet18_transfer_lr1e4` có gap nhỏ nhất (`+3.50 pp`) nhưng thực chất là **underfitting** (`train_loss = 0.7677`, `val_loss = 0.8219`, `val_acc = 87.07%`) do `CosineAnnealingLR` tắt LR quá sớm trong 15 epochs.
  - `resnet18_transfer_lr5e3` không bùng phân kỳ `val_loss` ở epoch cuối nhờ cosine decay hạ LR về `5.46e-5`, nhưng vẫn dao động mạnh giữa chu kỳ và tạo gap lớn nhất nhóm pre-trained (`+10.52 pp`, `train_acc = 99.54%`).
  - Để nén khoảng tách `+5.6 pp → +10.5 pp` ở các cấu hình Transfer trên tập cỡ vừa mà không gây underfit, cần kết hợp **head `Dropout(0.3–0.5)` + `label_smoothing = 0.1` + `weight_decay = 1e-2` + `ColorJitter / RandomErasing`** thay vì giảm mù quáng learning rate.
- **Evidence:** Chẩn đoán toàn bộ 9 runs tại [`experiment_report.md` §3.1–§3.2](../../labs/lab2_v2/docs/experiment_report.md).
- **Recommendation:** Luôn đọc đồng thời 3 tín hiệu `(train_loss, val_loss trend, train−val gap)` thay vì chỉ nhìn độ hở 2 đường Accuracy; đưa `Dropout` và `label_smoothing` thành tham số mặc định có thể cấu hình cho head Transfer Learning từ các lab sau.

---

## 5. Real-World Dataset Hygiene: Hidden `RGBA` Images & Low-Resolution Fine-Grained Degradation

- **Category:** Data Engineering & Error Analysis
- **Description:** Quét 100% ảnh thật ở Layer 6.2 không phải hình thức — ngay trên bộ benchmark chuẩn `Oxford-IIIT Pets` vẫn tồn tại **`3` file `.jpg` thực chất ở mode `RGBA`** (`Egyptian_Mau_14.jpg`, `Egyptian_Mau_186.jpg`, `Abyssinian_5.jpg`) có thể gây crash channel count nếu thiếu `.convert("RGB")`. Hơn nữa, Slice Analysis theo diện tích ảnh gốc (`orig_area`) và loài (`Cat vs Dog`) phơi bày lỗi trên ảnh `Q1 LowRes` tăng gấp `2.1×` toàn cục (`13.45%` vs `6.33%`) và vọt lên **`20.41%` ở nhóm Mèo** (do chi tiết mắt/lông tinh bị mờ khi cắt/phóng và sự thiên lệch lớp chó trong ImageNet-1k).
- **Evidence:** [`cleaning_report.json`](../../experiments/lab2_v2/eda/cleaning_report.json), [`eda_report.md`](../../labs/lab2_v2/docs/eda_report.md), [`slice_analysis.json`](../../experiments/lab2_v2/evaluation/slice_analysis.json) và [`evaluation_report.md` §4.4](../../labs/lab2_v2/docs/evaluation_report.md).
- **Recommendation:** Luôn ép `.convert("RGB")` ở cả loader lẫn `preprocess_image()` inference; trong Slice Analysis cho bài toán ảnh tự nhiên, luôn thêm trục **độ phân giải gốc (`orig_area`)** và **tỷ lệ khung hình (`aspect_ratio`)** song song với phân nhóm ngữ nghĩa (như `Cat vs Dog`).

---

## 6. Checkpoint Disk Footprint Scales with `AdamW` Trainable Parameters (`optimizer_state_dict`)

- **Category:** Deployment & Storage
- **Description:** Lưu kèm `optimizer_state_dict` trong checkpoint phục vụ `--resume` làm kích thước file `.pt` tăng thêm đúng `~8 bytes/trainable_param` (`2` tensor FP32 `exp_avg` & `exp_avg_sq` của AdamW): cùng kiến trúc `ResNet18`, `resnet18_transfer/best.pt` chỉ nặng **`42.93 MB`** (`19K` trainable), trong khi `resnet18_finetune/best.pt` nặng **`106.98 MB`** (`8.41M` trainable) và `resnet18_scratch/best.pt` nặng **`128.25 MB`** (`11.20M` trainable).
- **Evidence:** Đo kích thước thực tế 5 checkpoints tại [`inference_benchmark.json`](../../experiments/lab2_v2/evaluation/inference_benchmark.json) và [`evaluation_report.md` §5.2](../../labs/lab2_v2/docs/evaluation_report.md).
- **Recommendation:** Giữ `optimizer_state_dict` trong `last.pt` để hỗ trợ `--resume`, còn với `best.pt` phục vụ đóng gói/triển khai production có thể cân nhắc chỉ lưu `model_state_dict` + metadata (hoặc có lệnh export strip optimizer state) để giảm `2.5–3×` dung lượng lưu trữ.

---

## Suggested Documentation Updates (Chờ Kỹ sư duyệt — không tự ý sửa)

1. **`.gitignore`:** Xem xét đổi `runs/` thành `/runs/` để không phải `git add -f` từng file `.json/.csv/.png` trong `experiments/*/runs/` (trong khi `*.pt` vẫn bị chặn toàn cục).
2. **`labs/lab2_v2` (Vòng nâng cấp tùy chọn v2.1):** Khi có thời gian thực nghiệm thêm, có thể bổ sung `--dropout`, `--label-smoothing` và `ColorJitter` để kiểm chứng trực tiếp giả thuyết nén Train–Val gap và fine-tune `DenseNet121` (`denseblock4 + norm5`) ở độ phân giải `288–384 px`.
