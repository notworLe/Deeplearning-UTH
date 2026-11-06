# UTH Deep Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests: pytest](https://img.shields.io/badge/tests-3%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Clean Architecture](https://img.shields.io/badge/Architecture-Clean%20%26%20Modular-informational.svg)](.agents/PIPELINE_SEPARATION.md)

---

## 📌 Tổng Quan Dự Án (Overview)

Repository này chứa toàn bộ mã nguồn, cấu hình thực nghiệm, bài tập thực hành theo tuần và đồ án của môn **Học Sâu (Deep Learning) tại Trường Đại học Giao thông Vận tải TP.HCM (UTH)**.

Toàn bộ dự án được thiết kế theo tư duy **Clean Architecture** và bộ quy chuẩn kỹ thuật cho AI Agent & Kỹ sư tại [`.agents/`](.agents/README.md), đảm bảo tính module hóa cao, tính tái lập 100% (Reproducibility), tách rời cấu hình khỏi mã nguồn và có hệ thống kiểm thử tự động.

---

## 📋 Lộ Trình Bài Thực Hành (Labs Roadmap)

| Lab | Chủ Đề Kỹ Thuật | Trạng Thái | Pipeline Bản Final | Tài Liệu Chi Tiết | Điểm Nhấn Kỹ Thuật |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **Lab 01** | **Phân loại Thời trang Fashion-MNIST với Custom CNN** | ✅ **Hoàn thành** | [`labs/lab1/main/main.py`](labs/lab1/main/main.py) | [Tổng quan Lab](labs/lab1/README.md) · [Tài liệu Final](labs/lab1/main/README.md) | Mạng 3 block Conv + BatchNorm + MaxPool + Dropout, TensorBoard logs, Web Demo Gradio suy luận batch/folder. |
| **Lab 02** | **Pre-trained CNNs & Transfer Learning (ResNet / VGG)** | ✅ **Hoàn thành** | [`labs/lab2/main/transfer_learning.py`](labs/lab2/main/transfer_learning.py) | [Tổng quan Lab](labs/lab2/README.md) · [Tài liệu Final](labs/lab2/main/README.md) | Đóng băng tầng trích xuất đặc trưng (33.4% params), thay thế Classifier Head, chuẩn hóa ImageNet, Accuracy đạt **86.38%**. |
| **Lab 03** | **Fine-tuning Transformers (DistilBERT) cho Sentiment Analysis** | ✅ **Hoàn thành** | [`labs/lab3/main/fine-tune.py`](labs/lab3/main/fine-tune.py) | [Tổng quan Lab](labs/lab3/README.md) · [Tài liệu Final](labs/lab3/main/README.md) | Dataset GLUE SST-2, WordPiece Tokenizer, đệm padding linh hoạt `DataCollatorWithPadding`, Hugging Face Trainer `fp16`. |

---

## 👥 Quy Chuẩn Phân Định Thư Mục Trong Từng Lab (`main/` vs `submit/`)

Trong mỗi thư mục bài lab (`labs/labX/`), dự án áp dụng quy ước quản lý mã nguồn hai tầng:

| Phân Vùng | Vai Trò & Trách Nhiệm | Đối Tượng Sử Dụng |
| :--- | :--- | :--- |
| **`main/`** | **Bản Code Final Hoàn Chỉnh Nhất**: Nơi tổng hợp mã nguồn chuẩn hóa, tái cấu trúc (refactored), kiểm thử đầy đủ và là bản duy nhất dùng để nghiệm thu, báo cáo và chấm điểm đồ án. | Giảng viên, người chấm bài, triển khai production |
| **`submit/`** | **Bản Đóng Góp Của Thành Viên**: Lưu trữ mã nguồn, notebook thực nghiệm độc lập và ý tưởng cá nhân do từng thành viên trong nhóm phát triển trước khi tích hợp vào `main/`. | Lưu vết đóng góp cá nhân, đối chiếu lịch sử làm việc nhóm |

---

## ⚙️ Gói Mã Nguồn Cốt Lõi `src/` (Full-Loop Deep Learning Engine)

Gói package [`src/`](src/README.md) là bộ khung xử lý (Custom Handlers) dùng chung cho toàn bộ các bài lab và dự án, bao quát chu trình học sâu khép kín:

```text
src/
├── config.py       # Cấu hình phần cứng (GPU/CPU), seed, paths, các preset Augmentations
├── dataset.py      # Pipeline nạp dữ liệu, phân tách train/val/test, bọc DataLoader đa luồng
├── model.py        # Kiến trúc nơ-ron (FashionCNN), vòng lặp huấn luyện training_phase, evaluate & test
├── object.py       # Đối tượng MyMetric theo dõi loss, accuracy, precision, recall, F1, confusion matrix
├── visualize.py    # Thư viện vẽ đồ thị chuyên sâu: show_batch, phân bố nhãn, ma trận nhầm lẫn
├── utility.py      # Khóa seed tập trung, lưu checkpoint an toàn chống crash qua file .tmp, resume weights
├── log.py          # Chuẩn hóa in terminal log (GPU, VRAM, dataset thông số) với độ rộng cố định
└── class_name.py   # Danh mục nhãn chuẩn hóa cho các bộ dữ liệu thực nghiệm
```

👉 **Xem bảng ma trận trạng thái các phase đã hoàn thành và lộ trình sắp tới (Early Stopping, Native AMP, ONNX Export) tại [src/README.md](src/README.md).**

---

## 🗂️ Cấu Trúc Thư Mục Tổng Thể (Repository Layout)

```text
Deeplearning-UTH/ (ROOT)
├── .agents/                           # Bộ quy chuẩn vận hành nhận thức cho AI Agent & Kỹ sư
│   ├── README.md                      # Giao thức tiếp nhận và 4 cổng kiểm soát chất lượng
│   ├── AI_REFERENCE.md                # Bất biến CV/DL, quy chuẩn đặt tên, điều cấm kỵ
│   ├── PIPELINE_SEPARATION.md         # Phân tách 5 pipeline Clean Architecture
│   └── skills/                        # Bộ kỹ năng mở rộng (project-docs-analyzer, teaching-skills)
├── labs/                              # Thư mục chứa các bài thực hành học phần
│   ├── lab1/                          # Lab 01: Fashion-MNIST Custom CNN & Gradio Web App
│   │   ├── main/                      # Bản final nộp bài chính thức
│   │   └── submit/                    # Bản đóng góp của thành viên
│   ├── lab2/                          # Lab 02: Pre-trained CNNs & Transfer Learning
│   │   ├── main/                      # Bản final so sánh Baseline vs Transfer Learning
│   │   └── submit/                    # Bản đóng góp của thành viên
│   └── lab3/                          # Lab 03: Transformer Fine-Tuning với DistilBERT
│       ├── main/                      # Bản final nộp bài với Hugging Face Trainer
│       └── submit/                    # Bản đóng góp của thành viên
├── src/                               # Gói mã nguồn Python dùng chung (Full-Loop DL Engine)
│   └── README.md                      # Tài liệu kỹ thuật chi tiết của package src
├── data/                              # Dữ liệu thực nghiệm (được bỏ qua bởi Git)
├── weights/                           # Trọng số checkpoints mô hình huấn luyện (*.pt)
├── result/                            # Kết quả hình ảnh, confusion matrix, biểu đồ xuất ra
├── docs/                              # Tài liệu kỹ thuật và báo cáo chuyên sâu
├── test/                              # Bộ kiểm thử tự động với Pytest
├── pyproject.toml                     # Cấu hình metadata gói Python (pip install -e .)
├── pytest.ini                         # Cấu hình môi trường chạy kiểm thử Pytest
└── requirements.txt                   # Danh sách thư viện phụ thuộc của dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng (Quickstart)

### 1. Khởi Tạo Môi Trường

```bash
# 1. Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài đặt các thư viện phụ thuộc cơ sở
pip install -r requirements.txt

# 3. Cài đặt PyTorch phù hợp với phần cứng của bạn
# Với NVIDIA GPU (CUDA 12.1+):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Với CPU:
pip install torch torchvision

# 4. Cài đặt package `src` ở chế độ Editable
pip install -e .
```

---

### 2. Chạy Các Bài Thực Hành (Bản Final)

#### 🔹 Lab 01: Phân loại Fashion-MNIST
```bash
# Huấn luyện và đánh giá mô hình
python labs/lab1/main/main.py

# Khởi chạy giao diện Web Demo Gradio suy luận ảnh theo folder
python labs/lab1/main/inference/app.py
```

#### 🔹 Lab 02: Học Chuyển Giao (Transfer Learning)
```bash
# Chạy thực nghiệm Transfer Learning với ResNet (Classifier Fine-Tuning)
python labs/lab2/main/transfer_learning.py

# Chạy mô hình đối chứng Baseline
python labs/lab2/main/baseline.py
```

#### 🔹 Lab 03: Tinh Chỉnh Mô Hình Transformer (DistilBERT)
```bash
# Chạy quá trình fine-tuning mô hình DistilBERT trên SST-2
python labs/lab3/main/fine-tune.py

# Kiểm tra suy luận nhanh với câu mẫu
python labs/lab3/main/main.py
```

---

### 3. Giám Sát Tiến Trình Bằng TensorBoard

Mở giao diện trực quan hóa loss và accuracy theo thời gian thực:

```bash
tensorboard --logdir=labs/lab1/main/runs
# Hoặc giám sát Lab 2:
tensorboard --logdir=labs/lab2/main/runs
```
Truy cập trình duyệt tại địa chỉ: `http://localhost:6006`.

---

### 4. Kiểm Thử Tự Động (Testing)

Dự án tích hợp bộ kiểm thử đơn vị tự động với `pytest`:

```bash
pytest test/
```

Toàn bộ test cases sẽ kiểm tra tính toàn vẹn của danh mục nhãn (`test_class_name.py`) và tiện ích seed / weight checkpoint (`test_utility.py`).
