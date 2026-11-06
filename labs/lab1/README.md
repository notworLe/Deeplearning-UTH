# Lab 01: Fashion-MNIST Image Classification with PyTorch

Dự án thực hành Deep Learning: Xây dựng mạng nơ-ron tích chập (Custom CNN) để phân loại ảnh thời trang trên tập dữ liệu Fashion-MNIST sử dụng PyTorch.

---

## 🗂️ Cấu trúc Thư mục Lab 01

Thư mục `lab1` được tổ chức thành 2 phân vùng chức năng rõ ràng:

```text
labs/lab1/
├── README.md               # Tài liệu định hướng và tổng quan bài Lab 01
├── main/                   # NƠI CHỨA CODE FINAL BẢN CUỐI CÙNG ĐỂ NỘP
│   ├── README.md           # Tài liệu kỹ thuật chi tiết của bản nộp chính thức
│   ├── main.py             # Script thực thi pipeline huấn luyện và đánh giá chính thức
│   ├── main.ipynb          # Notebook thực hành tương tác trực quan
│   ├── inference/          # Ứng dụng suy luận (Gradio web app & batch inference)
│   ├── weights/            # Trọng số mô hình đã huấn luyện (*.pt)
│   ├── result/             # Kết quả trực quan hóa, confusion matrix, metrics
│   ├── runs/               # Nhật ký TensorBoard logs
│   └── data/               # Dữ liệu Fashion-MNIST tải về
└── submit/                 # NƠI LƯU BẢN CODE ĐÓNG GÓP CỦA CÁC THÀNH VIÊN TRONG NHÓM
    ├── Dai-Duong-FashionMNIST/   # Bản nộp & notebook thử nghiệm của thành viên Đại Dương
    └── Git Clone/                # Bản mở rộng thử nghiệm thí nghiệm tham khảo
```

---

## 📌 Phân Định Chức Năng Giữa `main/` và `submit/`

| Thư mục | Định nghĩa & Trách nhiệm | Đối tượng sử dụng |
| :--- | :--- | :--- |
| **`main/`** | **Bản Code Final Hoàn Chỉnh Nhất**: Tổng hợp, chuẩn hóa kiến trúc Clean Architecture, tái cấu trúc (refactored), kiểm thử đầy đủ và là bản duy nhất dùng để chấm điểm, nghiệm thu và nộp đồ án. | Giảng viên, người chấm bài, triển khai production |
| **`submit/`** | **Bản Đóng Góp Của Thành Viên**: Lưu trữ các phiên bản code nháp, script thực nghiệm cá nhân, notebook ban đầu do từng thành viên trong nhóm phát triển trước khi gộp vào `main/`. | Lưu vết đóng góp cá nhân, đối chiếu lịch sử làm việc nhóm |

---

## 🚀 Hướng Dẫn Bắt Đầu Nhanh Với Bản Final (`main/`)

Để chạy kiểm tra hoặc tái lập kết quả của bài lab, di chuyển vào thư mục `main/`:

```bash
# 1. Di chuyển vào thư mục code chính thức
cd labs/lab1/main

# 2. Chạy pipeline huấn luyện & kiểm thử
python main.py

# 3. Khởi chạy giao diện Web Demo trực quan (Gradio)
python inference/app.py

# 4. Xem nhật ký huấn luyện trên TensorBoard
tensorboard --logdir=runs
```

👉 **Xem hướng dẫn kỹ thuật chi tiết, giải thích giải thuật và bảng thông số tại [main/README.md](main/README.md).**
