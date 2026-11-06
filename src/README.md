# SRC Package - Core Deep Learning Engine & Pipeline Handlers

Thư mục `src/` là gói mã nguồn Python cốt lõi (Core Package) do nhóm tự xây dựng nhằm module hóa toàn bộ các thành phần trong chu trình học sâu khép kín (**Full Loop Deep Learning Pipeline**). 

Mục tiêu thiết kế: Cung cấp các handlers dùng chung, chuẩn hóa Clean Architecture để tái sử dụng xuyên suốt các bài thực hành (`labs/`), giảm thiểu việc viết lại boilerplate code và đảm bảo tính nhất quán từ khâu nạp dữ liệu đến giám sát huấn luyện.

---

## 🗺️ Ma Trận Trạng Thái Pipeline (Pipeline Status Matrix)

Hệ thống được thiết kế theo mô hình pipeline 5 tầng tiêu chuẩn. Dưới đây là trạng thái hiện diện của từng module chức năng:

| Phase | Thành phần Kỹ thuật | File Phụ trách | Trạng thái | Mô tả & Chức năng |
| :---: | :--- | :--- | :---: | :--- |
| **01** | **Cấu hình & Môi trường** | [`config.py`](config.py) | ✅ **Đã có (Available)** | Tự động nhận diện GPU/CPU, thiết lập hằng số huấn luyện, định dạng đường dẫn gốc, chuẩn hóa tensor. |
| **02** | **Xử lý & Tăng cường Dữ liệu** | [`dataset.py`](dataset.py), [`config.py`](config.py) | ✅ **Đã có (Available)** | Pipeline nạp dataset, phân tách train/val/test, tạo `DataLoader` đa luồng (`pin_memory`), preset Augmentations. |
| **03** | **Danh mục Nhãn & Metadata** | [`class_name.py`](class_name.py) | ✅ **Đã có (Available)** | Danh mục nhãn định danh chuẩn hóa cho các tập dữ liệu thực nghiệm (Oxford-IIIT Pet 37 classes...). |
| **04** | **Kiến trúc Mạng Nơ-ron** | [`model.py`](model.py) | ✅ **Đã có (Available)** | Kiến trúc tích chập custom `FashionCNN` (3 blocks Conv + BatchNorm + MaxPool + AdaptiveAvgPool + Dropout). |
| **05** | **Vòng lặp Huấn luyện (Train Loop)** | [`model.py`](model.py) | ✅ **Đã có (Available)** | Hàm `training_phase` điều phối tiến trình tqdm, forward/backward, logging TensorBoard, tự động lưu checkpoint khi đạt Best Val. |
| **06** | **Thẩm định & Đánh giá (Eval & Test)** | [`model.py`](model.py) | ✅ **Đã có (Available)** | Hàm `evaluate_phase` và `test_phase` chạy dưới chế độ `torch.no_grad()`, đo loss và xuất confusion matrix. |
| **07** | **Đo lường Chỉ số (Metric Engine)** | [`object.py`](object.py) | ✅ **Đã có (Available)** | Lớp `MyMetric` tích lũy loss, tính Accuracy, tích hợp `scikit-learn` đo Precision, Recall, Macro-F1, Confusion Matrix. |
| **08** | **Bộ Công cụ Trực quan hóa (Visualizer)**| [`visualize.py`](visualize.py) | ✅ **Đã có (Available)** | Thư viện đồ thị trực quan: `show_batch`, phân bố nhãn `plot_class_distribution`, `plot_classification_metrics`, Confusion Matrix. |
| **09** | **Tiện ích Hệ thống & Tái lập (Utility)** | [`utility.py`](utility.py) | ✅ **Đã có (Available)** | Khóa ngẫu nhiên tập trung (`set_seed`), cơ chế lưu checkpoint an toàn qua file `.tmp` (`save_weight`), phục hồi trọng số (`resume_weight`). |
| **10** | **Chuẩn hóa Ghi Log (Console Logger)** | [`log.py`](log.py) | ✅ **Đã có (Available)** | Định dạng thông tin terminal theo độ dài cố định (`print_log`), thống kê thông số GPU, VRAM, Dataset & DataLoader. |
| **11** | **Điều phối Tốc độ Học & Dừng Sớm** | `src/scheduler.py` | ⏳ **Coming Soon** | Cơ chế Early Stopping tự động ngắt huấn luyện khi val loss chững lại; tích hợp Learning Rate Schedulers (CosineAnnealing, ReduceLROnPlateau). |
| **12** | **Huấn luyện Độ chính xác Hỗn hợp (AMP)** | `src/amp.py` | ⏳ **Coming Soon** | Tích hợp native `torch.cuda.amp.autocast()` và `GradScaler` trực tiếp vào lõi `training_phase` để tối ưu VRAM và tăng tốc x2. |
| **13** | **Decoupled CLI Pipeline Runner** | `src/pipelines/` | ⏳ **Coming Soon** | Tách hẳn runner thành các file CLI độc lập (`train.py`, `evaluate.py`) nhận config từ Hydra/YAML thay vì chạy trong lab script. |
| **14** | **Xuất Mô hình & Triển khai Production** | `src/export/` | ⏳ **Coming Soon** | Module chuyển đổi trọng số sang ONNX, TorchScript và TensorRT; profiling độ trễ suy luận (Latency) và FPS trên thiết bị biên. |
| **15** | **Huấn luyện Phân tán (Distributed DDP)** | `src/distributed.py` | ⏳ **Coming Soon** | Hỗ trợ DistributedDataParallel (DDP) phục vụ huấn luyện mô hình lớn trên nhiều GPU song song. |

---

## 📖 Chi Tiết Các Module Đã Triển Khai (Implemented Modules)

### 1. [`config.py`](config.py) - Cấu hình & Siêu tham số Tập trung
Chịu trách nhiệm khởi tạo cấu hình môi trường toàn cục:
- **Phần cứng**: Tự động gán `DEVICE = cuda` nếu khả dụng, ngược lại fallback về `cpu`.
- **Hằng số**: `BATCH_SIZE = 32`, `LR = 1e-5`, `EPOCH = 30`, `WOKER = 8`, `SEED = 248`.
- **Đường dẫn**: Quản lý đường dẫn gốc dự án `ROOT_PATH`, thư mục dữ liệu `DATA_PATH`, đường dẫn lưu trọng số `weights/`.
- **Bộ chuẩn hóa & Augmentation Presets**:
  - `DEFAULT_TRANSFORM`: Chuyển đổi tensor và chuẩn hóa ảnh đơn kênh với $mean=0.5, std=0.5$.
  - `IMAGE_NET_NORMALIZE`: Chuẩn hóa 3 kênh màu theo ImageNet ($mean=[0.485, 0.456, 0.406]$, $std=[0.229, 0.224, 0.225]$).
  - `AUGMENTED_TRANSFORM`: Kết hợp `RandomHorizontalFlip`, `RandomAffine` (xoay, dịch chuyển, phóng to/thu nhỏ) và `RandomErasing` để ép mô hình học đặc trưng phân tán.
  - `AUGMENTED_AI_VIETNGUYEN`: Pipeline tăng cường nâng cao kết hợp biến dạng hình học và biến thiên màu sắc (`ColorJitter`).

---

### 2. [`dataset.py`](dataset.py) - Tải & Phân Tách Dữ Liệu
Chịu trách nhiệm chuẩn bị dữ liệu và bọc `DataLoader`:
- `load_raw_dataset(dataset, root, p_train_transform, p_test_transform)`: Tải tập train và test từ thư viện torchvision với các pipeline transform tương ứng.
- `split_into_val_dataset(p_train_dataset, p_val_size=0.2)`: Chia ngẫu nhiên tập train thành `train` và `validation` theo tỷ lệ định sẵn bằng `random_split`.
- `create_dataloader(train_dataset, test_dataset)`: Tạo cặp DataLoader hỗ trợ đa luồng (`num_workers=8`), xáo trộn dữ liệu (`shuffle=True`), tự động bật `pin_memory=True` khi dùng GPU CUDA để tối ưu băng thông nạp batch.
- `loader_train_val_test(train_dataset, val_dataset, test_dataset)`: Khởi tạo đồng thời cả 3 bộ DataLoader (Train, Val, Test).

---

### 3. [`model.py`](model.py) - Kiến Trúc & Vòng Lặp Huấn Luyện
Chứa kiến trúc nơ-ron và các hàm điều phối chu trình huấn luyện:
- **Lớp `FashionCNN`**: Mạng tích chập 3 khối Conv:
  - Khối 1: `Conv2d(1 -> 32, k=3, p=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)`.
  - Khối 2: `Conv2d(32 -> 64, k=3, p=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)`.
  - Khối 3: `Conv2d(64 -> 128, k=3, p=1)` → `BatchNorm2d` → `ReLU`.
  - Khối gộp: `AdaptiveAvgPool2d(output_size=1)`.
  - Classifier Head: `Flatten` → `Linear(128 -> 256)` → `ReLU` → `Dropout(0.5)` → `Linear(256 -> 10)`.
- **Hàm `training_phase(...)`**:
  - Lặp qua từng epoch và batch với thanh tiến trình `tqdm`.
  - Tính loss, lan truyền ngược (`backward()`) và cập nhật trọng số (`optimizer.step()`).
  - Ghi nhật ký TensorBoard Loss và Accuracy của cả hai tập Train/Val sau mỗi epoch.
  - So sánh độ chính xác xác thực với `p_best_val`, tự động kích hoạt `save_weight` khi tìm thấy checkpoint tốt hơn.
- **Hàm `evaluate_phase(...)`**: Chạy ở chế độ `torch.no_grad()`, tính loss trung bình và vẽ ma trận nhầm lẫn cập nhật lên TensorBoard.
- **Hàm `test_phase(...)`**: Đánh giá độc lập trên tập test và đóng gói kết quả vào đối tượng `MyMetric`.

---

### 4. [`object.py`](object.py) - Đối Tượng Quản Lý Đo Lường (`MyMetric`)
Module theo dõi và tính toán các chỉ số định lượng:
- **Thuộc tính tích lũy**: Lưu tổng số mẫu `total_samples`, tổng số dự đoán đúng `total_correct`, tổng mất mát `total_loss`, danh sách nhãn thực tế `list_actual` và nhãn dự đoán `list_prediction`.
- **Properties**:
  - `average_loss`: Tính giá trị mất mát trung bình theo số mẫu.
  - `accuracy`: Tính tỷ lệ dự đoán chính xác toàn cục.
- **Tích hợp `scikit-learn`**:
  - `get_precision(*args, **kwargs)`: Tính điểm Precision (hỗ trợ average macro, micro, weighted).
  - `get_recall(*args, **kwargs)`: Tính điểm Recall.
  - `get_f1(*args, **kwargs)`: Tính điểm F1-Score.
- **Trực quan hóa Ma trận Nhầm lẫn**:
  - `cal_cf_matrix()`: Tính ma trận nhầm lẫn chuẩn hóa theo hàng.
  - `get_fig_cf_matrix(classes)`: Xuất đồ thị Heatmap Seaborn chuyên nghiệp hiển thị tỷ lệ chính xác từng lớp.

---

### 5. [`visualize.py`](visualize.py) - Bộ Công Cụ Trực Quan Hóa Đồ Họa
Hệ thống đồ thị chuyên sâu phục vụ phân tích dữ liệu và báo cáo:
- `show_batch(dataset, dataloader, save_path)`: Vẽ lưới các ảnh mẫu trong batch kèm nhãn định danh.
- `plot_class_distribution(dataset, save_path)`: Vẽ biểu đồ cột kiểm tra độ cân bằng số lượng mẫu giữa các lớp.
- `plot_classification_metrics(actual, pred, classes, save_path)`: Vẽ biểu đồ cột so sánh trực quan đồng thời 4 chỉ số (Precision, Recall, F1, Accuracy) cho từng lớp riêng biệt.
- `show_loss_acc(train_metrics, val_metrics, save_path)`: Vẽ đường cong Loss và Accuracy qua các Epoch.
- `show_predictions(model, test_dataloader, dataset, save_path)`: Hiển thị ảnh kèm nhãn thực tế và nhãn dự đoán (đánh dấu xanh lá khi đúng, đỏ khi sai).

---

### 6. [`utility.py`](utility.py) & [`log.py`](log.py) - Tiện Ích & Logging
- **`set_seed(seed)`**: Khóa nguồn sinh số ngẫu nhiên của `random`, `numpy`, `torch` và `torch.cuda` nhằm đảm bảo tính tái lập 100% của thí nghiệm.
- **`save_weight(path, epoch, model, optimizer, best_val)`**: Cơ chế lưu trọng số an toàn tuyệt đối — ghi dữ liệu ra file tạm thời `.tmp` trước khi thay thế nguyên tử (atomic rename), loại bỏ hoàn toàn rủi ro file weights bị hỏng khi tiến trình bị ngắt đột ngột (OOM hoặc crash).
- **`resume_weight(path, model, optimizer)`**: Nạp lại trọng số mô hình, trạng thái optimizer, epoch và điểm validation tốt nhất từ checkpoint để tiếp tục huấn luyện.
- **`print_log(name, value)`**: Chuẩn hóa việc in log trên terminal với độ rộng cột căn lề thống nhất.
- **`log_creation()`**: In tổng quan thông số phần cứng (GPU model, dung lượng VRAM), seed và cấu hình dataloader khi bắt đầu chạy.

---

## ⏳ Lộ Trình Triển Khai Các Phase Tiếp Theo (Coming Soon Roadmap)

Các module dưới đây đang trong lộ trình phát triển để hoàn thiện chu trình tự động hóa MLOps toàn diện:

### 1. Phase 11: Learning Rate Scheduler & Early Stopping (`src/scheduler.py`)
- **Mục tiêu**: Tự động hóa việc điều chỉnh learning rate và ngăn chặn quá khớp (overfitting).
- **Dự kiến triển khai**:
  - Lớp `EarlyStopping`: Theo dõi validation loss với tham số `patience` và `min_delta`, tự động dừng `training_phase` khi không còn cải thiện.
  - Tích hợp các bộ lập lịch học phổ biến: `torch.optim.lr_scheduler.CosineAnnealingLR` và `ReduceLROnPlateau`.

### 2. Phase 12: Native Mixed Precision Training (`src/amp.py`)
- **Mục tiêu**: Giảm thiểu tiêu thụ VRAM tới 50% và tăng tốc huấn luyện trên các dòng GPU NVIDIA RTX / Tensor Cores.
- **Dự kiến triển khai**:
  - Tích hợp bộ chuyển đổi tự động `torch.cuda.amp.autocast(dtype=torch.float16)` trong forward pass.
  - Sử dụng `torch.cuda.amp.GradScaler()` để nhân tỉ lệ gradients, ngăn chặn hiện tượng underflow khi tính toán số thực dấu phẩy động 16-bit.

### 3. Phase 13: Decoupled CLI Pipeline Runners (`src/pipelines/`)
- **Mục tiêu**: Đóng gói các quy trình thành command-line scripts độc lập theo chuẩn Clean Architecture:
  - `src/pipelines/train.py`: Nhận tham số từ file cấu hình YAML hoặc CLI arguments để huấn luyện bất kỳ mô hình/dataset nào.
  - `src/pipelines/evaluate.py`: Nạp checkpoint và xuất báo cáo định lượng JSON/Markdown độc lập.

### 4. Phase 14: Model Export & Production Engine (`src/export/`)
- **Mục tiêu**: Tối ưu hóa mô hình phục vụ triển khai thực tế với độ trễ thấp (Low-Latency Inference).
- **Dự kiến triển khai**:
  - Chuyển đổi mô hình PyTorch sang định dạng chuẩn trao đổi **ONNX** (`torch.onnx.export`).
  - Đóng gói sang **TorchScript** (Trace & Script) phục vụ chạy trên môi trường C++ hoặc mobile.
  - Công cụ đo benchmark hiệu năng: Đo lường độ trễ trung bình ($ms/sample$), độ ổn định P99 và thông lượng ($FPS$).

---

## 💻 Ví Dụ Tích Hợp Chu Trình Hoàn Chỉnh (Full Loop Pipeline Example)

Dưới đây là mẫu kết hợp các handlers có sẵn của `src/` để thực thi một chu trình huấn luyện khép kín:

```python
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST

from src import config, utility, log
from src.dataset import load_raw_dataset, create_dataloader
from src.model import FashionCNN, training_phase, test_phase
from src.visualize import plot_classification_metrics, _finish

# 1. Khởi tạo môi trường & khóa ngẫu nhiên
log.log_creation()
utility.set_seed()

# 2. Chuẩn bị dữ liệu & DataLoaders
train_ds, test_ds = load_raw_dataset(
    FashionMNIST, config.DATA_PATH,
    p_train_transform=config.AUGMENTED_TRANSFORM,
    p_test_transform=config.DEFAULT_TRANSFORM
)
train_loader, test_loader = create_dataloader(train_ds, test_ds)
log.dataloader_log(train_loader, test_loader)

# 3. Khởi tạo mô hình, hàm mất mát và optimizer
model = FashionCNN().to(config.DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LR)
writer = SummaryWriter("runs/demo_experiment")

# 4. Huấn luyện và tự động lưu checkpoint tối ưu
save_path = Path("weights/demo_model.pt")
training_phase(
    model, loss_fn, optimizer,
    train_loader, test_loader,
    path_save=save_path,
    epoch=10,
    writer=writer
)

# 5. Đánh giá kiểm thử độc lập
metric_obj = test_phase(model, test_loader, writer)
metric_obj.show_metric()

# 6. Trực quan hóa ma trận nhầm lẫn
fig_cm = metric_obj.get_fig_cf_matrix(FashionMNIST.classes)
_finish(fig_cm, save_path="result/confusion_matrix.png")

writer.close()
```
