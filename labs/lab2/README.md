# Lab 02: Pre-trained Neural Network Architectures & Transfer Learning

Dự án thực hành Deep Learning: Nghiên cứu, thực nghiệm và so sánh các kiến trúc mạng nơ-ron tích chập tiền huấn luyện (Pre-trained CNNs: ResNet, VGG, DenseNet) áp dụng kỹ thuật Học chuyển giao (Transfer Learning) và Tinh chỉnh (Fine-tuning) trên bài toán phân loại ảnh.

---

## 🗂️ Cấu trúc Thư mục Lab 02

Thư mục `lab2` được phân chia thành 2 phân vùng chức năng:

```text
labs/lab2/
├── README.md                   # Tài liệu tổng quan và định hướng bài Lab 02
├── Lab2-learning-progress      # Nhật ký tiến độ học tập và lộ trình 5 giai đoạn
├── main/                       # NƠI CHỨA CODE FINAL BẢN CUỐI CÙNG ĐỂ NỘP
│   ├── README.md               # Báo cáo kỹ thuật chi tiết của bản nộp chính thức
│   ├── baseline.py             # Pipeline huấn luyện mô hình Baseline
│   ├── transfer_learning.py    # Pipeline huấn luyện Transfer Learning (Custom Classifier Head)
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

## 📌 Phân Định Chức Năng Giữa `main/` và `submit/`

| Thư mục | Định nghĩa & Trách nhiệm | Đối tượng sử dụng |
| :--- | :--- | :--- |
| **`main/`** | **Bản Code Final Hoàn Chỉnh Nhất**: Nơi tổng hợp mã nguồn chuẩn hóa, xây dựng module `MyResNet`, đóng băng tầng trích xuất đặc trưng (Feature Extractor Freezing), huấn luyện đầu phân loại mới và lưu trữ kết quả thực nghiệm kiểm thử để nộp bài. | Giảng viên, người chấm bài, nghiệm thu kỹ thuật |
| **`submit/`** | **Bản Đóng Góp Của Thành Viên**: Nơi lưu giữ các bài tập, notebook thực hành độc lập của các thành viên trong nhóm (như thành viên Dương) trong quá trình nghiên cứu và tìm hiểu kiến trúc mạng trước khi tổng hợp vào `main/`. | Lưu vết đóng góp cá nhân của từng thành viên |

---

## 🚀 Hướng Dẫn Bắt Đầu Nhanh Với Bản Final (`main/`)

Để chạy kiểm tra hoặc tái lập kết quả của bài lab, di chuyển vào thư mục `main/`:

```bash
# 1. Di chuyển vào thư mục code chính thức
cd labs/lab2/main

# 2. Chạy thử nghiệm Transfer Learning
python transfer_learning.py

# 3. Chạy thử nghiệm mô hình Baseline (để so sánh đối chứng)
python baseline.py

# 4. Xem nhật ký huấn luyện trên TensorBoard
tensorboard --logdir=runs
```

👉 **Xem phân tích so sánh chi tiết, bảng số liệu và giải thuật tại [main/README.md](main/README.md).**