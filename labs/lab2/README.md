# Lab 02: Pre-trained Neural Network Architectures & Transfer Learning

Dự án thực hành Deep Learning: Nghiên cứu, thực nghiệm và so sánh các kiến trúc mạng nơ-ron tích chập tiền huấn luyện (Pre-trained CNNs: ResNet, VGG, DenseNet) áp dụng kỹ thuật Học chuyển giao (Transfer Learning) và Tinh chỉnh (Fine-tuning) trên bài toán phân loại ảnh CIFAR-10.

---

## 🗂️ Cấu trúc Thư mục Lab 02

Thư mục `lab2` được phân chia thành 3 phân vùng chức năng:

```text
labs/lab2/
├── README.md                   # Tài liệu tổng quan và định hướng bài Lab 02
├── QUESTION.md                 # Đề bài gốc và yêu cầu chuyển giao bài toán
├── Lab2-learning-progress      # Nhật ký tiến độ học tập và lộ trình 5 giai đoạn
├── vibe/                       # [KHÔNG GIAN THỰC HÀNH CHUẨN VIBE VÀ NOTEBOOKS MỚI NHẤT]
│   ├── README.md               # Hướng dẫn chi tiết không gian vibe
│   ├── 01_pretrained_models_exploration_and_data.ipynb  # Notebook 1: Khảo sát 3 họ mạng & Tiền xử lý
│   ├── 02_transfer_learning_and_fine_tuning.ipynb       # Notebook 2: Head Surgery, Freezing & Training
│   ├── 03_evaluation_comparative_benchmark_and_inference.ipynb # Notebook 3: Đánh giá 5W1H, Matrix & Inference
│   ├── train.py                # Script CLI huấn luyện độc lập
│   └── inference.py            # Script CLI suy luận Top-3
├── main/                       # NƠI CHỨA CODE FINAL BẢN GỐC ĐỂ NỘP
│   ├── README.md               # Báo cáo kỹ thuật chi tiết của bản nộp chính thức
│   ├── baseline.py             # Pipeline huấn luyện mô hình Baseline
│   ├── transfer_learning.py    # Pipeline huấn luyện Transfer Learning
│   ├── baseline.ipynb          # Notebook thực nghiệm Baseline
│   ├── main.ipynb              # Notebook thực nghiệm Transfer Learning
│   ├── result.txt              # Bảng ghi nhận thông số thực nghiệm (Accuracy đạt 86.38%)
│   ├── result/                 # Biểu đồ đánh giá, Confusion Matrix, Metrics
│   ├── runs/                   # Nhật ký TensorBoard logs
│   └── weights/                # Trọng số checkpoint mô hình đã huấn luyện
└── submit/                     # NƠI LƯU BẢN CODE ĐÓNG GÓP CỦA CÁC THÀNH VIÊN TRONG NHÓM
    └── Duong/                  # Bài thực hành cá nhân của thành viên Dương
        ├── Practice2_CIFAR10_ResNet18.ipynb  # Thử nghiệm ResNet-18 trên CIFAR-10
        ├── CIFAR-10_data-master.zip         # Dữ liệu nén thử nghiệm
        └── main.py                          # Script khởi chạy cá nhân
```

---

## 📌 Phân Định Chức Năng Giữa Các Phân Vùng

| Thư mục | Định nghĩa & Trách nhiệm | Đối tượng sử dụng |
| :--- | :--- | :--- |
| **`vibe/`** | **Không Gian Thực Hành Chuẩn Mực Mới**: Bộ 3 Jupyter Notebooks phân rã theo ML Pipeline, đáp ứng `NOTEBOOK_HEADER_CONVENTION.md`, Cell Independence, script CLI và báo cáo định lượng chuẩn 5W1H. | Học tập, thuyết trình, chạy thử nghiệm tương tác |
| **`main/`** | **Bản Code Final Hoàn Chỉnh Gốc**: Nơi lưu trữ mã nguồn chuẩn hóa, module `MyResNet`, checkpoint trọng số huấn luyện sẵn (`resnet18_transfer.pt`) và kết quả thực nghiệm đạt 86.38%. | Giảng viên, người chấm bài, lưu trữ gốc |
| **`submit/`** | **Bản Đóng Góp Của Thành Viên**: Nơi lưu giữ các bài tập, notebook thực hành độc lập của các thành viên trong nhóm (như thành viên Dương). | Lưu vết đóng góp cá nhân của từng thành viên |

---

## 📚 Tài Liệu Báo Cáo Kỹ Thuật Chi Tiết (Tương ứng với `docs/lab2/`)

- 📘 [docs/lab2/README.md](../../docs/lab2/README.md): Báo cáo tổng quan, đối chuẩn 5W1H và quản trị phần cứng VRAM $\le 1.8\text{GB}$.
- 📐 [docs/lab2/PIPELINE_EXPLANATION.md](../../docs/lab2/PIPELINE_EXPLANATION.md): Bóc tách 10 module toán học theo 6 trục phân tích chi tiết.
- 🎤 [docs/lab2/PRESENTATION.md](../../docs/lab2/PRESENTATION.md): Cẩm nang thuyết trình, sơ đồ bản đồ file, kịch bản 10–15 phút và 8 câu hỏi phản biện kỹ thuật.

---

## 🚀 Hướng Dẫn Bắt Đầu Nhanh

### Chạy với Thư Mục `vibe/`:
```bash
# Huấn luyện mô hình Transfer Learning qua CLI
python labs/lab2/vibe/train.py --model resnet18 --mode finetune --epochs 5

# Suy luận nhanh Top-3 xác suất với checkpoint có sẵn
python labs/lab2/vibe/inference.py --weights labs/lab2/main/weights/resnet18_transfer.pt --samples 5
```

### Chạy với Thư Mục `main/`:
```bash
cd labs/lab2/main
python transfer_learning.py
python baseline.py
tensorboard --logdir=runs
```
