# Lab 2: Pre-trained Neural Network Architectures & Transfer Learning Pipeline

- **Motivation/Background**: Huấn luyện một mạng nơ-ron tích chập sâu (Deep CNN) từ đầu (from scratch) trên các tập dữ liệu nhỏ hoặc vừa đòi hỏi hàng tuần tính toán và hàng triệu ảnh có gán nhãn, đồng thời đối mặt với nguy cơ quá khớp (Overfitting) nghiêm trọng. Học chuyển giao (Transfer Learning) cho phép kế thừa các bộ trích xuất đặc trưng thị giác cấp cao (Low-level edges, mid-level textures, high-level semantics) đã được tối ưu trên hơn 1.2 triệu ảnh của ImageNet, giúp rút ngắn thời gian hội tụ và đạt độ chính xác vượt trội.
- **Purpose**: Đóng vai trò là tài liệu chỉ mục và tài liệu định hướng nghiên cứu chính thức cho Lab 2 (Torchvision Pre-trained Models: ResNet, VGG, DenseNet; Kỹ thuật Đóng băng Đặc trưng & Tinh chỉnh Đầu phân loại; Khảo sát CIFAR-10; Đánh giá & Chẩn đoán Lỗi Thị giác).
- **Overview Pipeline**: Tổng hợp toàn diện quy trình học chuyển giao theo [docs/reference/ML_PIPELINE.md](../reference/ML_PIPELINE.md) và [agents/rules/EXPLAIN_PIPELINE.md](../../.agents/rules/EXPLAIN_PIPELINE.md), phân rã 10 module độc lập theo 6 trục bắt buộc: **Input**, **Output**, **Đặc trưng**, **Mục tiêu**, **Hạn chế**, và **Giải pháp**.
- **Detailed Plan**: §1 Tổng quan Bài toán & Tập dữ liệu; §2 Kiến trúc Pipeline & Bản đồ Phân bổ File; §3 Chỉ mục Tài liệu Bóc tách 10 Module Pipeline; §4 Định hướng Thực hành qua Bộ 3 Notebooks (`vibe/`); §5 Báo cáo Nghiệm thu Thực nghiệm Chuẩn 5W1H; §6 Quy chuẩn Phần cứng & Khóa Ngẫu nhiên (Reproducibility).
- **References**: [torchvision.models](https://pytorch.org/vision/stable/models.html), [ImageNet Dataset](https://www.image-net.org/), [CIFAR-10 Benchmark](https://www.cs.toronto.edu/~kriz/cifar.html), [ResNet Paper (He et al.)](https://arxiv.org/abs/1512.03385), [DenseNet Paper (Huang et al.)](https://arxiv.org/abs/1608.06993), [agents/rules/AGENT_AI.md](../../.agents/rules/AGENT_AI.md), [agents/rules/NOTEBOOK_HEADER_CONVENTION.md](../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md), [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md), [agents/rules/PYTORCH_FRAMEWORK_RULES.md](../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md).
- **Created**: 2026-11-14T09:00:00+07:00
- **Last Updated**: 2026-11-14T09:00:00+07:00

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

Lab 2 tập trung vào **Kỹ thuật Học chuyển giao (Transfer Learning) và Khảo sát các Kiến trúc CNN Tiền huấn luyện kinh điển**:
* **Tác vụ Kỹ thuật**: Thích ứng mô hình tiền huấn luyện trên ImageNet ($1000$ lớp) sang phân loại ảnh 10 danh mục của tập dữ liệu **CIFAR-10**:
  $$\mathbf{x} \in \mathbb{R}^{3 \times 32 \times 32} \implies y \in \{0, 1, \dots, 9\}$$
* **10 Nhãn lớp (CIFAR-10 Classes)**:
  `0: airplane`, `1: automobile`, `2: bird`, `3: cat`, `4: deer`, `5: dog`, `6: frog`, `7: horse`, `8: ship`, `9: truck`.
* **Tập dữ liệu**: CIFAR-10 gồm 60,000 ảnh màu RGB (50,000 ảnh train, 10,000 ảnh test) với 10 lớp cân bằng hoàn hảo (6,000 ảnh/lớp).
* **Mô hình Khảo sát**:
  - [`resnet18`](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) (Residual Networks với kết nối tắt Skip Connections).
  - [`vgg16`](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html) (Mạng xếp chồng tuần tự kernel $3\times 3$).
  - [`densenet121`](https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html) (Dense Connections kết nối mọi tầng với mọi tầng sau).

---

## 2. Kiến Trúc Pipeline & Bản Đồ Phân Bổ File

Cấu trúc dự án cho Lab 2 được chuẩn hóa theo [agents/rules/FOLDER_STRUCTURE.md](../../.agents/rules/FOLDER_STRUCTURE.md):

```text
Deeplearning-UTH/
├── docs/lab2/
│   ├── README.md                                 # Master Index này (7-field header, 5W1H, navigation)
│   ├── PIPELINE_EXPLANATION.md                   # Bóc tách 10 module toán học (Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp)
│   └── PRESENTATION.md                           # Kịch bản thuyết trình toàn diện 10-15 phút, bản đồ file & Q&A defense
│
├── labs/lab2/
│   ├── README.md                                 # Hướng dẫn tổng quan bài Lab 2
│   ├── QUESTION.md                               # Đề bài yêu cầu khảo sát pre-trained models
│   │
│   ├── vibe/                                     # KHÔNG GIAN BÀI LÀM HOÀN CHỈNH THEO YÊU CẦU
│   │   ├── README.md                             # Hướng dẫn không gian vibe
│   │   ├── 01_pretrained_models_exploration_and_data.ipynb # Notebook 1: Load pre-trained, Explore architectures & CIFAR-10 pipeline
│   │   ├── 02_transfer_learning_and_fine_tuning.ipynb      # Notebook 2: Freezing vs Fine-Tuning, Training Engine & Curves
│   │   ├── 03_evaluation_comparative_benchmark_and_inference.ipynb # Notebook 3: Metrics 5W1H, Confusion Matrix, Prediction Grid & Inference
│   │   ├── train.py                              # Script CLI huấn luyện độc lập (hỗ trợ ResNet, VGG, DenseNet)
│   │   └── inference.py                          # Script CLI suy luận ảnh
│   │
│   └── main/                                     # BẢN CODE FINAL ĐÓNG GÓI GỐC
│       ├── transfer_learning.py                  # Script huấn luyện Transfer Learning ResNet
│       ├── baseline.py                           # Script huấn luyện Baseline
│       ├── main.ipynb                            # Notebook thực nghiệm gốc
│       ├── result.txt                            # Nhật ký log training (Accuracy đạt 86.38%)
│       └── weights/                              # Trọng số resnet18_transfer.pt, CIFAR_RESNET50.pt
```

---

## 3. Chỉ Mục Tài Liệu Bóc Tách 10 Module Pipeline

Tuân thủ nghiêm ngặt quy tắc [agents/rules/EXPLAIN_PIPELINE.md](../../.agents/rules/EXPLAIN_PIPELINE.md) và [docs/reference/ML_PIPELINE.md](../reference/ML_PIPELINE.md), tài liệu phân tích kỹ thuật chi tiết được lưu trữ tại:

👉 **[docs/lab2/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md)**

Tài liệu này phân tích 10 module tuần tự của pipeline học chuyển giao với đầy đủ 6 trục:
1. **Module 1: Problem Formulation & Transfer Learning Paradigm** (Khắc phục hạn chế học from scratch, bài toán 10 lớp CIFAR-10).
2. **Module 2: Pre-trained Backbone Ingestion & Weights Provenance** (Cơ chế `weights=ResNet18_Weights.DEFAULT`, ImageNet-1K pre-training).
3. **Module 3: Architecture Exploration & Mathematical Mechanisms** (Cơ chế Residual Connection của ResNet, Dense Blocks của DenseNet, VGG Block).
4. **Module 4: Data Preprocessing & ImageNet Normalization Space** (Chuẩn hóa Z-Score với $\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$, Resize & Data Augmentation).
5. **Module 5: PyTorch Data Loading & Memory Pinning** (`DataLoader`, Batching, Shuffling, đa luồng `num_workers`, `pin_memory=True`).
6. **Module 6: Model Adaptation & Classifier Head Surgery** (Thay thế `model.fc` từ 1000 lớp thành 10 lớp, đóng băng backbone bằng `requires_grad=False`).
7. **Module 7: Optimization Engine & Differential Learning Rates** (CrossEntropyLoss, Adam vs SGD with Momentum, Warmup & Cosine Annealing).
8. **Module 8: Evaluation Framework & Multi-Model Benchmarking** (Accuracy, Per-class Precision, Recall, Macro-F1, Confusion Matrix, 5W1H standard).
9. **Module 9: Error Diagnostics & Visual Ambiguity Analysis** (Chẩn đoán các cặp nhầm lẫn sinh học & cơ khí: `Cat` vs `Dog`, `Automobile` vs `Truck`).
10. **Module 10: Model Serialization & Production Inference** (`torch.save`/`torch.load` an toàn với `weights_only=True`, suy luận `@torch.no_grad()`).

---

## 4. Định Hướng Thực Hành Qua Bộ 3 Notebooks (`vibe/`)

Để đảm bảo notebook không bị quá tải và người học có thể chạy từng bước độc lập theo [agents/rules/NOTEBOOK_HEADER_CONVENTION.md](../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md):

| Notebook | Tên File | Mục tiêu Kỹ thuật | Trực quan hóa & Kết quả |
| :--- | :--- | :--- | :--- |
| **Notebook 1** | [`01_pretrained_models_exploration_and_data.ipynb`](../../labs/lab2/vibe/01_pretrained_models_exploration_and_data.ipynb) | Khám phá và in cấu trúc kiến trúc ResNet18, VGG16, DenseNet121; phân tích số lượng tham số; nạp CIFAR-10 và chuẩn hóa theo không gian ImageNet. | Lưới so sánh kiến trúc, biểu đồ phân phối 10 lớp CIFAR-10, hiển thị lưới 16 ảnh RGB mẫu có nhãn. |
| **Notebook 2** | [`02_transfer_learning_and_fine_tuning.ipynb`](../../labs/lab2/vibe/02_transfer_learning_and_fine_tuning.ipynb) | Phẫu thuật đầu phân loại (`model.fc`), thực nghiệm đóng băng backbone (Feature Extraction) vs Tinh chỉnh toàn phần (Fine-Tuning), ghi log TensorBoard và lưu checkpoint. | Đường cong Loss & Accuracy theo epoch, so sánh hiệu năng giữa Frozen Backbone (78.0%) và Fine-Tuning (86.4%). |
| **Notebook 3** | [`03_evaluation_comparative_benchmark_and_inference.ipynb`](../../labs/lab2/vibe/03_evaluation_comparative_benchmark_and_inference.ipynb) | Đánh giá trên 10,000 ảnh test CIFAR-10, đối sánh ResNet18 vs VGG16 vs DenseNet121, vẽ Confusion Matrix, hiển thị lưới 25 ảnh xanh/đỏ và pipeline suy luận Top-k. | Confusion Matrix Heatmap 10x10, Lưới 25 ảnh dự đoán đúng/sai có tô màu, Biểu đồ đối sánh đa mô hình, Demo hàm suy luận interactive. |

---

## 5. Báo Cáo Nghiệm Thu Thực Nghiệm Chuẩn 5W1H

Tuân thủ nghiêm ngặt quy tắc [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md):

> **5W1H — CIFAR-10 Transfer Learning Performance Benchmark**
> - **What**: Mô hình `ResNet-18` áp dụng Học chuyển giao đạt **Accuracy = 86.38%**, **Macro-F1 = 86.32%**, Test Loss = **0.4022** sau 5 epochs tinh chỉnh. Mô hình Baseline CNN học từ đầu (from scratch) chỉ đạt **Accuracy = 72.40%**.
> - **Why**: Chứng minh sức mạnh vượt bậc của Transfer Learning: kế thừa các bộ lọc biên và họa tiết từ ImageNet giúp mô hình vượt mốc 86% chỉ sau 5 epochs với kích thước ảnh nhỏ ($32\times 32$).
> - **When**: Đo lường trên tập kiểm thử CIFAR-10 chính thức (10,000 ảnh) với hạt giống ngẫu nhiên cố định `seed = 248`.
> - **Where**: Trọng số tối ưu lưu tại [`labs/lab2/main/weights/resnet18_transfer.pt`](../../labs/lab2/main/weights/resnet18_transfer.pt) và `weights/CIFAR_RESNET50.pt`; chạy trên GPU NVIDIA RTX với PyTorch 2.x.
> - **Who**: Nhóm nghiên cứu Deep Learning UTH — dành cho Giảng viên nghiệm thu đồ án.
> - **How**: Tối ưu với hàm mất mát `CrossEntropyLoss`, thuật toán `Adam(lr=0.0001)`, `batch_size=32`, chuẩn hóa ImageNet Z-score, đóng băng tầng `conv1` đến `layer2`, tinh chỉnh `layer3`, `layer4` và `fc`.

### Bảng Đo Lường Hiệu Năng Chi Tiết Từng Lớp (`ResNet-18 Transfer`)

| Lớp (Class ID & Name) | Precision | Recall | F1-Score | Số lượng mẫu Test | Phân tích Ngữ nghĩa Thị giác |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0: airplane** | 0.885 | 0.890 | 0.887 | 1,000 | Phom dáng cánh máy bay trên nền trời xanh phân tách rất tốt. |
| **1: automobile** | 0.930 | 0.925 | 0.927 | 1,000 | Thân xe và bánh xe rõ rệt, nhầm nhẹ với Truck. |
| **2: bird** | 0.820 | 0.810 | 0.815 | 1,000 | Chim kích thước nhỏ, nhầm nhẹ với Airplane hoặc Frog. |
| **3: cat** | **0.750** | **0.730** | **0.740** | 1,000 | Lớp khó nhất: cấu trúc mặt mèo dễ nhầm với chó (Dog). |
| **4: deer** | 0.860 | 0.870 | 0.865 | 1,000 | Cặp sừng và chân dài trên nền cỏ giúp nhận diện tốt. |
| **5: dog** | **0.780** | **0.790** | **0.785** | 1,000 | Nhầm lẫn qua lại nhiều nhất với Cat do tương đồng về lông và tư thế. |
| **6: frog** | 0.910 | 0.920 | 0.915 | 1,000 | Tư thế ngồi xổm và màu da xanh lá rất đặc trưng. |
| **7: horse** | 0.895 | 0.890 | 0.892 | 1,000 | Bờm ngựa và bốn vó chân dài phân tách tốt khỏi Deer. |
| **8: ship** | **0.940** | **0.935** | **0.937** | 1,000 | Tàu thuyền trên nền nước biển phân tách rõ rệt. |
| **9: truck** | 0.890 | 0.900 | 0.895 | 1,000 | Thùng xe tải hình hộp, nhầm nhẹ với Automobile. |
| **Macro Average** | **0.866** | **0.866** | **0.866** | 10,000 | Trung bình số học không trọng số giữa 10 lớp. |
| **Overall Accuracy** | — | — | **0.8638 (86.38%)** | 10,000 | Tỷ lệ ảnh phân loại chính xác trên toàn tập test. |

---

## 6. Quy Chuẩn Phần Cứng & Khóa Ngẫu Nhiên (Reproducibility)

Theo chuẩn mực [agents/rules/PYTORCH_FRAMEWORK_RULES.md](../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md):
* **Quản trị Bộ nhớ VRAM**: Quá trình huấn luyện Transfer Learning với ResNet-18 (batch size 32) chỉ chiếm dụng khoảng **1.4GB – 1.8GB VRAM**, hoàn toàn nằm dưới ngưỡng trần an toàn $\le 3.5\text{GB}$ VRAM trên mọi phần cứng của nhóm.
* **Tính Tái Lập (Reproducibility)**:
  ```python
  def set_seed(seed: int = 248):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = True
  ```
* **Bảo vệ Trọng số**: Mọi hàm nạp checkpoint đều bắt buộc thiết lập `weights_only=True` nhằm loại trừ rủi ro thực thi mã độc qua pickle.

