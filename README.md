# UTH deep learning

[![Python 3.10+](https://img.shields.io/badge/python-3.12%20%7C%203.12-blue.svg)](https://www.python.org/)

<!-- [![Tests: pytest](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)](https://docs.pytest.org/) -->
---

## 📌Overview

Repository này chứa toàn bộ mã nguồn, dữ liệu thực nghiệm, bài tập tuần, và dự án nhóm của **Deep learning tại Trường Đại học Giao thông Vận tải TP.HCM (UTH)**.

### 📋Labs Roadmap

|    Lab     | Chủ đề Kỹ thuật                                                                                                      | Trạng thái | Pipeline Thực thi                                      |                                     Báo cáo Kỹ thuật                                      | Unit Tests |
|:----------:|:---------------------------------------------------------------------------------------------------------------------|:----------:|:-------------------------------------------------------|:-----------------------------------------------------------------------------------------:|:----------:|
| **Lab 01** | **Xử lý ảnh cơ bản & Không gian màu** (OpenCV, Pillow, BGR/RGB/HSV/LAB, ROI, Annotations)                            | ✅ **100%** | [`lab01_pipeline.py`](src/pipelines/lab01_pipeline.py) | [Markdown](docs/labs/lab01/LAB_01_REPORT.md) · [HTML](docs/labs/lab01/LAB_01_REPORT.html) | 10 passed  |
| **Lab 02** | **Toán tử điểm ảnh & Lọc không gian** (Brightness/Contrast/Negative, Gaussian, Sobel/Prewitt, Bilateral/Median)      | ✅ **100%** | [`lab02_pipeline.py`](src/pipelines/lab02_pipeline.py) | [Markdown](docs/labs/lab02/LAB_02_REPORT.md) · [HTML](docs/labs/lab02/LAB_02_REPORT.html) | 17 passed  |
| **Lab 03** | **Canny Edge Detector Chuyên sâu** (5-stage Canny, Noise Robustness, Pratt FOM, Hough Shapes, Di Zenzo Tensor)       | ✅ **100%** | [`lab03_pipeline.py`](src/pipelines/lab03_pipeline.py) | [Markdown](docs/labs/lab03/LAB_03_REPORT.md) · [HTML](docs/labs/lab03/LAB_03_REPORT.html) | 16 passed  |
| **Lab 04** | **So sánh tương đồng ảnh dùng Wavelet Hashing** (2D DWT Haar/db4, Quantization, ROC/PR Curves, Top-k Retrieval, MIH) | ✅ **100%** | [`lab04_pipeline.py`](src/pipelines/lab04_pipeline.py) | [Markdown](docs/labs/lab04/LAB_04_REPORT.md) · [HTML](docs/labs/lab04/LAB_04_REPORT.html) |  8 passed  |

---

## 🗂️ Cấu trúc Thư mục Tổng thể (Architecture Layout)

```text
Deeplearning-UTH (ROOT)
├── .agents/                           
│   ├── WORKFLOW_OVERVIEW.md           # Tổng quan luồng thực thi khép kín
│   ├── AI_FIRST_LIFECYCLE.md          # 9 giai đoạn phát triển chuẩn
│   ├── PIPELINE_SEPARATION.md         # Phân tách 5 pipelines Clean Architecture
│   ├── AI_REFERENCE.md                # Bất biến CV/DL, chuẩn PEP 484 & docstrings
│   └── CODEBASE_AUDIT.md              # Kiểm toán mã nguồn & an toàn CV
├── labs/                              # Chứa các bài lab
│   └── lab1
│       ├── main                       # Nội dung lab1 
│       └── submit                     # Bài tham khảo của các thành viên
│   ... Lab5 
└── README.md
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng (Quickstart)

### 1. Khởi tạo Môi trường

```bash
# 1. Tạo và kích hoạt virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài đặt toàn bộ dependencies
pip install -r requirements

# 3. Cài đặt Pytorch theo máy của bạn
- Có NVIDIA GPU: lấy lệnh tại https://pytorch.org/get-started/locally
- Không GPU / Mac: pip install torch torchvision
- Colab: không cần cài, đã có sẵn
```

### 2. Chạy Các Bài Thực Hành



### 3. Kiểm thử & Kiểm toán Mã nguồn
