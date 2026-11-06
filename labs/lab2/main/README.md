# Lab 02 (Final): Pre-trained CNNs & Transfer Learning Pipeline

Bản mã nguồn chính thức (Final Production Code) của nhóm dùng để nghiệm thu và nộp bài Lab 02. Module triển khai giải pháp Học chuyển giao (Transfer Learning) với mạng ResNet tiền huấn luyện trên ImageNet, tối ưu hóa quá trình trích xuất đặc trưng (Feature Extraction) và tinh chỉnh tầng phân loại (Fine-tuning Classifier Head).

---

## 📌 1. Mục Tiêu Kỹ Thuật & Giả Thuyết Thực Nghiệm

- **Mục tiêu**: Khắc phục tình trạng thiếu hụt dữ liệu và thời gian huấn luyện kéo dài khi học từ đầu (from scratch) bằng cách kế thừa tri thức từ mạng nơ-ron sâu đã học trên tập dữ liệu hàng triệu ảnh ImageNet.
- **Phương pháp**:
  1. Sử dụng mô hình ResNet (ResNet-18 / ResNet-50) tiền huấn luyện từ `torchvision.models`.
  2. Đóng băng (Freeze) các tầng tích chập trích xuất đặc trưng (`conv1` đến `layer4`) để bảo toàn trọng số tối ưu.
  3. Thay thế tầng kết nối đầy đủ (Fully Connected / Classifier Head) bằng tầng tuyến tính mới phù hợp với số lượng lớp mục tiêu.
  4. Chuẩn hóa phân phối ảnh đầu vào theo không gian phân phối chuẩn của ImageNet ($mean = [0.485, 0.456, 0.406]$, $std = [0.229, 0.224, 0.225]$).

---

## 🏗️ 2. Kiến Trúc Mô Hình (`MyResNet`)

Được triển khai trong [`transfer_learning.py`](transfer_learning.py):

```python
class MyResNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # Nạp trọng số ImageNet tối ưu
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        del self.model.fc  # Loại bỏ classifier 1000 lớp ban đầu

        # Gắn classifier head mới cho bài toán hiện tại
        self.fc = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
```

---

## 📂 3. Cấu Trúc Thư Mục `main/`

```text
labs/lab2/main/
├── README.md               # Báo cáo kỹ thuật chi tiết này
├── baseline.py             # Pipeline huấn luyện mô hình đối chứng (Baseline)
├── transfer_learning.py    # Pipeline huấn luyện Transfer Learning tối ưu
├── baseline.ipynb          # Jupyter Notebook chạy thử nghiệm đối chứng
├── main.ipynb              # Jupyter Notebook chạy thực nghiệm Transfer Learning
├── result.txt              # Bảng ghi log các epoch và chỉ số độ chính xác
├── weights/                # Điểm kiểm tra (checkpoints) mô hình tốt nhất
├── runs/                   # TensorBoard logs giám sát đường cong mất mát
├── data/                   # Thư mục lưu trữ bộ dữ liệu thực nghiệm
└── result/                 # Kết quả trực quan hóa xuất ra tự động
    ├── resnet18_baseline/  # Kết quả đánh giá của mô hình baseline
    └── resnet18_transfer/  # Kết quả đánh giá của mô hình transfer learning
        └── metric/
            ├── confusion-matrix.png
            └── pre-recall-f1-acc.png
```

---

## ⚙️ 4. So Sánh Quá Trình Huấn Luyện (Baseline vs. Transfer Learning)

### 4.1. Thông số Tham số (Parameter Freezing Strategy)
- **Tổng số tham số mô hình**: $25,616,458$ parameters.
- **Trước khi đóng băng**: $25,616,458$ tham số cần tính gradient (100%).
- **Sau khi đóng băng (Freeze Backbone)**: Chỉ cập nhật $17,073,162$ tham số (đã đóng băng $8,543,296$ tham số tương đương 33.4% khối lượng trích xuất đặc trưng cơ sở).

### 4.2. Tiến Trình Hội Tụ Định Lượng (Trích xuất từ `result.txt`)
Thực nghiệm ghi nhận trên GPU NVIDIA RTX 4060:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Mức Tăng Độ Chính Xác |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | 1.1886 | 62.84% | 0.6474 | **78.01%** | Khởi tạo vượt bậc từ weights tiền huấn luyện |
| **Epoch 2** | 0.6388 | 78.57% | 0.5177 | **82.13%** | +4.12% |
| **Epoch 3** | 0.5496 | 81.37% | 0.4717 | **83.97%** | +1.84% |
| **Epoch 4** | 0.4905 | 83.42% | 0.4289 | **85.44%** | +1.47% |
| **Epoch 5** | 0.4463 | 84.79% | 0.4022 | **86.38%** | **Đạt đỉnh 86.38%** |

---

## 🚀 5. Hướng Dẫn Thực Thi

### 5.1. Chạy Thực Nghiệm Transfer Learning

```bash
python transfer_learning.py
```

### 5.2. Chạy Thực Nghiệm Đối Chứng (Baseline)

```bash
python baseline.py
```

### 5.3. Mở TensorBoard Giám Sát

```bash
tensorboard --logdir=runs
```

---

## 📊 6. Kết Quả Trực Quan Hóa

- **Ma trận nhầm lẫn**: `result/resnet18_transfer/metric/confusion-matrix.png`
- **Chỉ số chi tiết Precision / Recall / F1-Score**: `result/resnet18_transfer/metric/pre-recall-f1-acc.png`
- **Nhật ký huấn luyện chi tiết**: [`result.txt`](result.txt)

---

## 🔗 7. Liên Kết Mã Nguồn Liên Quan

- Pipeline nạp dữ liệu đa định dạng: [`src/dataset.py`](../../../src/dataset.py)
- Module huấn luyện & đánh giá: [`src/model.py`](../../../src/model.py)
- Chuẩn hóa màu & biến đổi ảnh: [`src/config.py`](../../../src/config.py)
- Quy chuẩn kỹ thuật: [`.agents/AI_REFERENCE.md`](../../../.agents/AI_REFERENCE.md)
