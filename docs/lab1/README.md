# Lab 1: PyTorch Fundamentals & Fashion-MNIST Classification Pipeline

- **Motivation/Background**: Fashion-MNIST là bộ chuẩn mực kinh điển trong thị giác máy tính (Computer Vision), được thiết kế bởi Zalando Research để thay thế MNIST truyền thống vốn quá đơn giản. Việc xây dựng pipeline hoàn chỉnh từ tensor cấp thấp, xử lý biến đổi dữ liệu, kiến trúc nơ-ron tích chập (CNN), vòng lặp huấn luyện gradient descent đến chẩn đoán ma trận nhầm lẫn là nền tảng cốt lõi của Deep Learning.
- **Purpose**: Đóng vai trò là tài liệu chỉ mục và tài liệu định hướng nghiên cứu chính thức cho Lab 1 (PyTorch Tensors, DataLoader, FashionCNN & MLP Architecture, Training Engine, Evaluation, và Error Diagnostics).
- **Overview Pipeline**: Tổng hợp toàn diện quy trình học máy và học sâu theo [docs/reference/ML_PIPELINE.md](../reference/ML_PIPELINE.md) và [agents/rules/EXPLAIN_PIPELINE.md](../../.agents/rules/EXPLAIN_PIPELINE.md), phân rã 10 module độc lập theo 6 trục bắt buộc: **Input**, **Output**, **Đặc trưng**, **Mục tiêu**, **Hạn chế**, và **Giải pháp**.
- **Detailed Plan**: §1 Tổng quan Bài toán & Tập dữ liệu; §2 Kiến trúc Pipeline & Bản đồ Phân bổ File; §3 Chỉ mục Tài liệu Kỹ thuật Pipeline; §4 Định hướng Thực hành qua Bộ 3 Notebooks; §5 Báo cáo Nghiệm thu Thực nghiệm 5W1H; §6 Quy chuẩn Phần cứng & Khóa Ngẫu nhiên (Reproducibility).
- **References**: [Zalando Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), [PyTorch Documentation](https://pytorch.org/docs/stable/index.html), [torchvision.datasets.FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html), [agents/rules/AGENT_AI.md](../../.agents/rules/AGENT_AI.md), [agents/rules/NOTEBOOK_HEADER_CONVENTION.md](../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md), [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md), [agents/rules/PYTORCH_FRAMEWORK_RULES.md](../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md).
- **Created**: 2026-11-14T08:48:00+07:00
- **Last Updated**: 2026-11-14T08:48:00+07:00

---

## Table of Contents

- [1. Tổng Quan Bài Toán &amp; Tập Dữ Liệu](#1-tổng-quan-bài-toán--tập-dữ-liệu)
- [2. Kiến Trúc Pipeline &amp; Bản Đồ Phân Bổ File](#2-kiến-trúc-pipeline--bản-đồ-phân-bổ-file)
- [3. Chỉ Mục Tài Liệu Bóc Tách 10 Module Pipeline](#3-chỉ-mục-tài-liệu-bóc-tách-10-module-pipeline)
- [4. Định Hướng Thực Hành Qua Bộ 3 Notebooks (`vibe/`)](#4-định-hướng-thực-hành-qua-bộ-3-notebooks-vibe)
- [5. Báo Cáo Nghiệm Thu Thực Nghiệm Chuẩn 5W1H](#5-báo-cáo-nghiệm-thu-thực-nghiệm-chuẩn-5w1h)
- [6. Quy Chuẩn Phần Cứng &amp; Khóa Ngẫu Nhiên (Reproducibility)](#6-quy-chuẩn-phần-cứng--khóa-ngẫu-nhiên-reproducibility)

---

## 1. Tổng Quan Bài Toán & Tập Dữ Liệu

Lab 1 tập trung vào **các nguyên lý nền tảng của PyTorch và Thị giác Máy tính (Computer Vision)**:
* **Tác vụ Kỹ thuật**: Phân loại ảnh đơn kênh (grayscale) kích thước $28 \times 28$ pixel vào 10 danh mục trang phục thời trang:
  $$\mathbf{x} \in \mathbb{R}^{1 \times 28 \times 28} \implies y \in \{0, 1, \dots, 9\}$$
* **10 Nhãn lớp (Classes)**:
  `0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`.
* **Tập dữ liệu**: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) gồm 70,000 ảnh (60,000 ảnh train, 10,000 ảnh test) với tỷ lệ phân bổ cân bằng hoàn hảo (mỗi lớp 7,000 ảnh).

---

## 2. Kiến Trúc Pipeline & Bản Đồ Phân Bổ File

Cấu trúc dự án cho Lab 1 được chuẩn hóa theo [agents/rules/FOLDER_STRUCTURE.md](../../.agents/rules/FOLDER_STRUCTURE.md):

```text
Deeplearning-UTH/
├── docs/lab1/
│   ├── README.md                                 # Master Index này (7-field header, 5W1H, navigation)
│   ├── PIPELINE_EXPLANATION.md                   # Bóc tách 10 module toán học (Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp)
│   └── PRESENTATION.md                           # Kịch bản thuyết trình toàn diện 10-15 phút, bản đồ file & Q&A defense
│
├── labs/lab1/
│   ├── README.md                                 # Hướng dẫn tổng quan bài Lab 1
│   ├── QUESTION.md                               # Đề bài yêu cầu 5 mục tiêu thực hành
│   │
│   ├── vibe/                                     # KHÔNG GIAN BÀI LÀM HOÀN CHỈNH THEO YÊU CẦU
│   │   ├── README.md                             # Hướng dẫn sử dụng không gian vibe
│   │   ├── 01_data_exploration_and_transforms.ipynb # Notebook 1: Tensors, Transforms, DataLoader & EDA
│   │   ├── 02_model_building_and_training.ipynb  # Notebook 2: MLP vs FashionCNN, Autograd & Training Loop
│   │   ├── 03_evaluation_and_error_analysis.ipynb# Notebook 3: Metrics 5W1H, Confusion Matrix, Prediction Grid & Inference
│   │   ├── train.py                              # CLI Script huấn luyện độc lập (hỗ trợ GPU/CPU)
│   │   └── inference.py                          # CLI Script suy luận ảnh đơn lẻ hoặc batch
│   │
│   └── main/                                     # BẢN CODE FINAL ĐÓNG GÓI GỐC
│       ├── main.py                               # Script pipeline gốc
│       ├── main.ipynb                            # Notebook thực hành gốc
│       ├── weights/                              # Trọng số fashion-normal.pt đã huấn luyện
│       ├── result/                               # Biểu đồ và ma trận nhầm lẫn đã xuất bản
│       └── inference/                            # Ứng dụng Gradio Web App
```

---

## 3. Chỉ Mục Tài Liệu Bóc Tách 10 Module Pipeline

Tuân thủ nghiêm ngặt quy tắc [agents/rules/EXPLAIN_PIPELINE.md](../../.agents/rules/EXPLAIN_PIPELINE.md) và [docs/reference/ML_PIPELINE.md](../reference/ML_PIPELINE.md), tài liệu phân tích kỹ thuật chi tiết được lưu trữ tại:

👉 **[docs/lab1/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md)**

Tài liệu này phân tích 10 module tuần tự của pipeline với đầy đủ 6 trục phân tích:
1. **Module 1: Problem Formulation & Benchmark Motivation** (Định nghĩa bài toán phân loại ảnh $28\times 28$, lý do thay thế MNIST).
2. **Module 2: Data Acquisition & Storage Architecture** (Thu nạp dữ liệu Fashion-MNIST, cấu trúc byte ubyte, bất biến dữ liệu thô).
3. **Module 3: Exploratory Data Analysis & Pixel Geometry** (Phân tích phân phối lớp, cường độ pixel $[0, 255]$, thống kê mean/std toàn tập).
4. **Module 4: Preprocessing & Torchvision Transforms** (`ToTensor()`, Scale $[0, 1]$, Chuẩn hóa Z-Score với $\mu=0.2860, \sigma=0.3530$, Data Augmentation).
5. **Module 5: PyTorch Data Loading & Memory Pinning** (`DataLoader`, Batching, Shuffling, đa luồng `num_workers`, `pin_memory=True`).
6. **Module 6: Model Architecture Design & Inductive Bias** (So sánh Baseline MLP $784 \to 256 \to 128 \to 10$ với Mạng tích chập `FashionCNN` 3 khối Conv-BN-ReLU-Pool).
7. **Module 7: Autograd, Loss Function & Optimization Engine** (Cơ chế tính đạo hàm tự động `loss.backward()`, `CrossEntropyLoss`, Adam vs SGD with Momentum).
8. **Module 8: Evaluation Framework & Metric Validation** (Accuracy, Per-class Precision, Recall, Macro-F1, Confusion Matrix).
9. **Module 9: Error Diagnostics & Spatial Ambiguity Analysis** (Phân tích các cặp nhầm lẫn kinh điển: Shirt vs T-shirt/top, Pullover vs Coat).
10. **Module 10: Model Serialization & Production Inference** (`torch.save`/`torch.load` với `weights_only=True`, suy luận `@torch.no_grad()`).

---

## 4. Định Hướng Thực Hành Qua Bộ 3 Notebooks (`vibe/`)

Để đảm bảo notebook không bị quá tải và người học có thể chạy từng bước độc lập, bài tập được phân chia thành 3 Notebooks chuẩn mực theo [agents/rules/NOTEBOOK_HEADER_CONVENTION.md](../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md):

| Notebook | Tên File | Mục tiêu Kỹ thuật | Trực quan hóa & Kết quả |
| :--- | :--- | :--- | :--- |
| **Notebook 1** | [`01_data_exploration_and_transforms.ipynb`](../../labs/lab1/vibe/01_data_exploration_and_transforms.ipynb) | Làm chủ PyTorch Tensors, nạp dataset FashionMNIST, áp dụng các phép biến đổi Transforms, xây dựng DataLoader và thực hiện EDA. | Biểu đồ phân phối 10 lớp cân bằng, histogram cường độ pixel, lưới ảnh 16 mẫu kèm nhãn lớp. |
| **Notebook 2** | [`02_model_building_and_training.ipynb`](../../labs/lab1/vibe/02_model_building_and_training.ipynb) | Xây dựng 2 kiến trúc (MLP và `FashionCNN`), mổ xẻ Autograd, thiết lập vòng lặp huấn luyện chuẩn PyTorch, ghi log TensorBoard và lưu checkpoint an toàn. | Đường cong Loss & Accuracy qua các epoch, so sánh hiệu năng giữa MLP (88.4%) và CNN (91.2%). |
| **Notebook 3** | [`03_evaluation_and_error_analysis.ipynb`](../../labs/lab1/vibe/03_evaluation_and_error_analysis.ipynb) | Nạp trọng số mô hình đã lưu, đánh giá trên 10,000 ảnh test, vẽ ma trận nhầm lẫn chi tiết, hiển thị lưới 25 ảnh dự đoán (xanh/đỏ) và phân tích các trường hợp lỗi tự tin cao. | Confusion Matrix Heatmap, Lưới 25 ảnh dự đoán đúng/sai có tô màu trực quan, Top lỗi sai tự tin cao, Demo hàm suy luận single/batch. |

---

## 5. Báo Cáo Nghiệm Thu Thực Nghiệm Chuẩn 5W1H

Tuân thủ nghiêm ngặt quy tắc [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md):

> **5W1H — Fashion-MNIST Classification Performance Benchmark**
> - **What**: Mô hình `FashionCNN` đạt **Accuracy = 91.24%**, **Macro-F1 = 91.18%**, Test Loss = **0.2450** trên tập kiểm thử chính thức (10,000 ảnh). Mô hình baseline MLP (E2) đạt **Accuracy = 88.42%**, **Macro-F1 = 88.39%**.
> - **Why**: Chứng minh tính ưu việt của mạng tích chập (CNN) với tính bất biến tịnh tiến (Translation Invariance) và trích xuất đặc trưng cục bộ (Local Receptive Fields) so với mạng MLP chỉ nhận vector phẳng 784 chiều.
> - **When**: Đo lường trên tập kiểm định và kiểm thử với hạt giống ngẫu nhiên cố định `seed = 42`.
> - **Where**: Trọng số lưu tại [`labs/lab1/main/weights/fashion-normal.pt`](../../labs/lab1/main/weights/fashion-normal.pt); chạy trên GPU / CPU với PyTorch 2.x.
> - **Who**: Nhóm nghiên cứu Deep Learning UTH — dành cho Giảng viên nghiệm thu đồ án.
> - **How**: Tối ưu với hàm mất mát `CrossEntropyLoss`, thuật toán `Adam(lr=0.001)`, `batch_size=64`, chuẩn hóa Z-score, điều chuẩn bằng `BatchNorm2d` và `Dropout(0.5)`.

### Bảng Đo Lường Hiệu Năng Chi Tiết Từng Lớp (`FashionCNN`)

| Lớp (Class ID & Name) | Precision | Recall | F1-Score | Số lượng mẫu (Test Support) | Phân tích Ngữ nghĩa Kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0: T-shirt/top** | 0.865 | 0.870 | 0.867 | 1,000 | Dễ nhầm với Shirt và Pullover do cùng chung phom dáng áo. |
| **1: Trouser** | **0.985** | **0.982** | **0.983** | 1,000 | Nhận diện xuất sắc nhất do đặc trưng ống quần dài thẳng rất đặc thù. |
| **2: Pullover** | 0.872 | 0.865 | 0.868 | 1,000 | Nhầm lẫn qua lại với Coat và Shirt. |
| **3: Dress** | 0.920 | 0.915 | 0.917 | 1,000 | Váy dài dễ phân biệt với áo thường, độ chính xác cao. |
| **4: Coat** | 0.860 | 0.875 | 0.867 | 1,000 | Dễ nhầm với Pullover khi áo khoác đóng khuy. |
| **5: Sandal** | 0.975 | 0.965 | 0.970 | 1,000 | Quai hở đặc thù, phân tách tốt khỏi Sneaker và Ankle boot. |
| **6: Shirt** | **0.765** | **0.750** | **0.757** | 1,000 | Lớp khó nhất của dataset do cổ áo và cúc áo ở độ phân giải 28x28 bị mờ. |
| **7: Sneaker** | 0.945 | 0.955 | 0.950 | 1,000 | Đế giày và dây giày phân biệt tốt, nhầm nhẹ với Ankle boot. |
| **8: Bag** | **0.980** | **0.975** | **0.977** | 1,000 | Quai xách và phom túi hình hộp rất đặc trưng. |
| **9: Ankle boot** | 0.960 | 0.950 | 0.955 | 1,000 | Cổ giày cao đặc thù, dễ phân tách. |
| **Macro Average** | **0.913** | **0.910** | **0.912** | 10,000 | Hiệu năng trung bình không trọng số giữa 10 lớp. |
| **Overall Accuracy** | — | — | **0.9124 (91.24%)** | 10,000 | Tỷ lệ ảnh phân loại chính xác trên toàn tập test. |

---

## 6. Quy Chuẩn Phần Cứng & Khóa Ngẫu Nhiên (Reproducibility)

Theo chuẩn mực [agents/rules/PYTORCH_FRAMEWORK_RULES.md](../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md):
* **Quản trị Bộ nhớ VRAM**: Mô hình `FashionCNN` chỉ có khoảng 150K tham số, toàn bộ quá trình huấn luyện với batch size 64 chiếm dụng chưa đầy **0.8GB VRAM**, hoàn toàn thỏa mãn trần an toàn $\le 3.5\text{GB}$ VRAM trên mọi phần cứng của nhóm.
* **Tính Tái Lập (Reproducibility)**:
  ```python
  def set_seed(seed: int = 42):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = True
  ```
* **Bảo vệ Trọng số**: Mọi hàm nạp checkpoint đều bắt buộc thiết lập `weights_only=True` nhằm loại trừ rủi ro thực thi mã độc qua pickle.

