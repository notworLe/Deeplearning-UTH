---
title: "Pipeline 02: Model Architecture — Lab 1 (Fashion-MNIST)"
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
  - ./03_TRAINING.md
  - ../REPORT.md
  - ../../../../../.agents/temp/06_PIPELINE_SEPARATION.md
  - ../../../../../.agents/rules/BEHAVIORAL_LAYER.md
tags: [lab1, pipeline-02, model-architecture, logistic, mlp, cnn]
---

# Pipeline 02: Model Architecture (Định Nghĩa Kiến Trúc Mô Hình)

Tài liệu đặc tả độc lập cho **Pipeline 2** trong hệ thống phân tách 5 pipeline của Lab 1, bám sát mục `Model selection` và `Baseline` tại [`labs/lab1/vibe/docs/REPORT.md`](../REPORT.md).

---

## 1. Tài Liệu Đặc Tả Kỹ Thuật

### 1.1. Phạm vi & Ranh giới trách nhiệm
- **Trách nhiệm triển khai**:
  - Định nghĩa độc lập 3 lớp mô hình `nn.Module` tuân thủ chính xác sơ đồ tầng và số lượng tham số trong `REPORT.md`: `LogisticClassifier`, `MLPClassifier`, `CNNClassifier`.
  - Cung cấp factory function `build_model(model_name: str, num_classes: int = 10)` và hàm đếm tham số `count_parameters(model)`.
- **Nguyên tắc tách rời tuyệt đối (`PIPELINE_SEPARATION.md`)**:
  - Bên trong class Model **tuyệt đối không chứa** `CrossEntropyLoss`, `Adam`, `Softmax`, vòng lặp huấn luyện hay logic nạp dữ liệu.
  - Mọi mô hình đều nhận chung tensor `[B, 1, 28, 28]` và trả về **raw logits `[B, 10]`**.

### 1.2. Đối Chiếu Kiến Trúc & Kiểm Chứng Số Tham Số

Bảng dưới đây đối chiếu chính xác từng tầng theo `labs/lab1/vibe/docs/REPORT.md`:

| Mô hình | Vai trò & Lý do lựa chọn | Luồng biến đổi Tensor (`forward`) | Tổng tham số |
| :--- | :--- | :--- | :---: |
| **`LogisticClassifier`** | Mô hình tuyến tính đơn giản nhất, làm mốc sàn (baseline) để đo mức cải thiện của các kiến trúc sâu hơn | `[B, 1, 28, 28]` → `Flatten` `[B, 784]` → `Linear(784 → 10)` `[B, 10]` | **7.850** |
| **`MLPClassifier`** | Bổ sung 2 tầng ẩn phi tuyến ReLU vượt giới hạn tuyến tính của Logistic; vẫn flatten ảnh nên mất cấu trúc không gian 2D | `[B, 1, 28, 28]` → `Flatten` `[B, 784]` → `Linear(784 → 128)` → `ReLU` → `Linear(128 → 64)` → `ReLU` → `Linear(64 → 10)` `[B, 10]` | **109.386** |
| **`CNNClassifier`** | Kiến trúc tích chập bắt buộc theo bài học; khai thác cấu trúc không gian 2D cục bộ với số tham số cùng bậc MLP | `[B, 1, 28, 28]` → `Conv2d(1→16, 3×3, pad=1)` → `ReLU` → `MaxPool2d(2)` `[B, 16, 14, 14]` → `Conv2d(16→32, 3×3, pad=1)` → `ReLU` → `MaxPool2d(2)` `[B, 32, 7, 7]` → `Flatten` `[B, 1568]` → `Linear(1568 → 64)` → `ReLU` → `Linear(64 → 10)` `[B, 10]` | **105.866** |

#### Kiểm chứng giải tích số tham số từng tầng
1. **`LogisticClassifier`**:
   - `Linear(784, 10)`: $784 \times 10 + 10 = \mathbf{7.850}$.
2. **`MLPClassifier`**:
   - `Linear(784, 128)`: $784 \times 128 + 128 = 100.480$.
   - `Linear(128, 64)`: $128 \times 64 + 64 = 8.256$.
   - `Linear(64, 10)`: $64 \times 10 + 10 = 650$.
   - Tổng: $100.480 + 8.256 + 650 = \mathbf{109.386}$.
3. **`CNNClassifier`**:
   - `Conv2d(1, 16, kernel_size=3, padding=1)`: $1 \times 16 \times 3 \times 3 + 16 = 160$.
   - `Conv2d(16, 32, kernel_size=3, padding=1)`: $16 \times 32 \times 3 \times 3 + 32 = 4.640$.
   - `Linear(1568, 64)`: $(32 \times 7 \times 7) \times 64 + 64 = 1568 \times 64 + 64 = 100.416$.
   - `Linear(64, 10)`: $64 \times 10 + 10 = 650$.
   - Tổng: $160 + 4.640 + 100.416 + 650 = \mathbf{105.866}$.

> **Ràng buộc công bằng tham số**: Tỷ lệ tham số giữa `CNN` và `MLP` là $\frac{105.866}{109.386} \approx 0.968$ (cùng bậc độ phức tạp ~105k–109k).

### 1.3. Hợp đồng Đầu vào / Đầu ra (Input / Output Contracts)
- **Input của `forward(x)`**: `x: torch.Tensor` có shape `[B, 1, 28, 28]`, dtype `torch.float32`.
  - Fail-Fast validation: kiểm tra `x.ndim == 4` và `tuple(x.shape[1:]) == (1, 28, 28)`; ném `ValueError` rõ ràng nếu lệch shape.
- **Output của `forward(x)`**: `logits: torch.Tensor` có shape `[B, 10]`, dtype `torch.float32` trên cùng `device` với `x`.

### 1.4. Cấu Hình Chuẩn Hóa (YAML Spec)

```yaml
model:
  num_classes: 10
  input_shape: [1, 28, 28]
  candidates:
    logistic:
      type: LogisticClassifier
      expected_params: 7850
    mlp:
      type: MLPClassifier
      hidden_dims: [128, 64]
      expected_params: 109386
    cnn:
      type: CNNClassifier
      conv_channels: [16, 32]
      kernel_size: 3
      padding: 1
      fc_dim: 64
      expected_params: 105866
```

### 1.5. Ánh Xạ Thư Mục & Phân Biệt Với Mã Nguồn Cũ
- **Mã nguồn dự kiến (`SRC.md`)**: `src/vibe_scripts/lab1/models.py`.
- **Kiểm thử dự kiến**: `test/test_lab1_models.py`.
- **Lưu ý tránh nhầm lẫn**:
  - `src/model.py` hiện có lớp `FashionCNN` (32→64→128 + `BatchNorm2d` + `AdaptiveAvgPool2d` + `Dropout(0.5)`) và `labs/lab1/vibe/train.py` cũ có `BaselineMLP` (256→128 + `Dropout`). Hai kiến trúc này **KHÔNG khớp** đặc tả số tầng và số tham số trong `REPORT.md`.
  - Pipeline 02 giữ nguyên các file cũ và xây mới hoàn toàn 3 lớp đúng chuẩn `REPORT.md` tại `src/vibe_scripts/lab1/models.py`.

---

## 2. Prompt Độc Lập Cho Pipeline 02 (Behavioral Schema)

```markdown
Name:
Lab 1 Model Architecture Pipeline

Description:
Xây dựng module kiến trúc gồm 3 mô hình PyTorch độc lập (`LogisticClassifier`, `MLPClassifier`, `CNNClassifier`) khớp chính xác cấu trúc tầng và số lượng tham số quy định trong `labs/lab1/vibe/docs/REPORT.md`.

Purpose:
Cung cấp 3 ứng viên mô hình tách rời hoàn toàn khỏi training loop để so sánh công bằng giữa mô hình tuyến tính, phi tuyến phẳng (MLP) và tích chập không gian (CNN) có cùng bậc tham số.

Input:
- Tên mô hình: `model_name` thuộc `{"logistic", "mlp", "cnn"}`
- Tensor ảnh đầu vào: `x` dạng `torch.Tensor`, shape `[B, 1, 28, 28]`, dtype `torch.float32`

Output:
- Module `src/vibe_scripts/lab1/models.py` chứa `LogisticClassifier`, `MLPClassifier`, `CNNClassifier`, `build_model()`, `count_parameters()`, `summarize_models()`
- Raw logits đầu ra có shape `[B, 10]` (`float32`), không áp dụng Softmax
- Bộ unit test `test/test_lab1_models.py`

How to do:
1. Triển khai `LogisticClassifier`: `Flatten -> Linear(784, 10)` (7.850 tham số).
2. Triển khai `MLPClassifier`: `Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 10)` (109.386 tham số).
3. Triển khai `CNNClassifier`: `Conv2d(1, 16, 3, padding=1) -> ReLU -> MaxPool2d(2) -> Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(1568, 64) -> ReLU -> Linear(64, 10)` (105.866 tham số).
4. Bổ sung kiểm tra Fail-Fast shape đầu vào `(B, 1, 28, 28)` trong `forward` và viết đầy đủ PEP 484 type hints kèm Google-style docstrings ghi rõ tensor shapes.
5. Triển khai `build_model(model_name)` và `count_parameters(model)` (tổng tham số và tham số huấn luyện được).
6. Viết Smoke Test và Normal Test tại `test/test_lab1_models.py` kiểm chứng forward, backward, số tham số chính xác và trường hợp shape không hợp lệ.

Examples:
pytest test/test_lab1_models.py -v
```

---

## 3. Checklist Kiểm Thử Độc Lập (Definition of Done)

- [ ] **Khớp chính xác số tham số**:
  - `count_parameters(build_model("logistic")) == 7850`
  - `count_parameters(build_model("mlp")) == 109386`
  - `count_parameters(build_model("cnn")) == 105866`
- [ ] **Forward Pass với Dummy Input**: Với `x = torch.randn(32, 1, 28, 28)` và `B = 1` (single sample), cả 3 mô hình trả về tensor shape `[B, 10]`, dtype `torch.float32`, không `NaN`/`Inf`.
- [ ] **Đầu ra là Raw Logits**: Kiểm tra output có thể mang giá trị âm và tổng theo `dim=1` không bị ép bằng `1.0` (chứng minh không chứa `Softmax`/`LogSoftmax`).
- [ ] **Tách rời Training Loop**: Không import `torch.optim`, không khởi tạo loss hay optimizer bên trong `models.py`.
- [ ] **Lan truyền ngược Gradient**: Với dummy loss `CrossEntropyLoss()(logits, y).backward()`, mọi tham số có `requires_grad=True` đều nhận gradient khác `None` và hữu hạn.
- [ ] **Fail-Fast Validation**: Truyền tensor sai kích thước như `(4, 3, 28, 28)` hoặc `(4, 1, 32, 32)` phải ném `ValueError`.
- [ ] **Tái lập khởi tạo trọng số**: Khóa `torch.manual_seed(248)` rồi khởi tạo 2 lần cùng 1 mô hình cho `state_dict` bằng nhau tuyệt đối.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo đặc tả độc lập Pipeline 02 (Model Architecture) khớp chính xác 3 kiến trúc và số tham số trong `REPORT.md` |
