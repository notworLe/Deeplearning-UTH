# Lab 01 (Final): Fashion-MNIST Classification with Custom CNN

Bản mã nguồn chính thức (Final Production Code) của nhóm dùng để nghiệm thu và nộp bài Lab 01. Hệ thống triển khai mạng tích chập chuyên sâu (`FashionCNN`) kết hợp kỹ thuật tăng cường dữ liệu (Data Augmentation), giám sát TensorBoard và giao diện Web suy luận tương tác (Gradio).

---

## 📌 1. Tổng Quan Bài Toán & Tập Dữ Liệu

- **Bài toán**: Phân loại ảnh đơn kênh (grayscale) kích thước $28 \times 28$ vào 10 danh mục trang phục/phụ kiện thời trang.
- **Tập dữ liệu**: **Fashion-MNIST** (Zalando Research) gồm 70,000 ảnh (60,000 train, 10,000 test).
- **10 Lớp phân loại**:
  `0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`.

---

## 🏗️ 2. Kiến Trúc Mô Hình (`FashionCNN`)

Mô hình được định nghĩa tại [`../../../src/model.py`](../../../src/model.py), sử dụng kiến trúc phân tầng tích hợp Batch Normalization, Max Pooling và Adaptive Average Pooling để giảm thiểu rủi ro quá khớp (Overfitting):

```text
Input (Batch, 1, 28, 28)
  │
  ├── [Block 1] Conv2d(1 -> 32, k=3, p=1) ──► BatchNorm2d(32) ──► ReLU ──► MaxPool2d(2)  => (Batch, 32, 14, 14)
  │
  ├── [Block 2] Conv2d(32 -> 64, k=3, p=1) ──► BatchNorm2d(64) ──► ReLU ──► MaxPool2d(2) => (Batch, 64, 7, 7)
  │
  ├── [Block 3] Conv2d(64 -> 128, k=3, p=1) ──► BatchNorm2d(128) ──► ReLU                  => (Batch, 128, 7, 7)
  │
  ├── [Pooling] AdaptiveAvgPool2d(output_size=1)                                           => (Batch, 128, 1, 1)
  │
  └── [Classifier] Flatten ──► Linear(128 -> 256) ──► ReLU ──► Dropout(p=0.5) ──► Linear(256 -> 10) => Logits
```

---

## 📂 3. Cấu Trúc Thư Mục `main/`

```text
labs/lab1/main/
├── README.md               # Tài liệu kỹ thuật chi tiết này
├── main.py                 # Script huấn luyện & kiểm thử chính thức
├── main.ipynb              # Jupyter Notebook chạy từng bước tương tác
├── data/                   # Thư mục lưu trữ bộ dữ liệu FashionMNIST tải về
├── weights/                # Trọng số mô hình đã huấn luyện (fashion-normal.pt)
├── runs/                   # Nhật ký TensorBoard logs theo dõi training
├── inference/              # Module phục vụ suy luận thực tế
│   ├── app.py              # Ứng dụng web Gradio dự đoán ảnh theo folder
│   ├── inference.py        # Hàm suy luận đơn ảnh và batch inference (@torch.inference_mode)
│   └── input/              # Thư mục chứa ảnh đầu vào kiểm tra
└── result/                 # Kết quả trực quan hóa xuất ra tự động
    └── fashion-normal/
        ├── data-visual/    # Phân bố nhãn lớp & ảnh mẫu batch
        │   ├── clas-distribution-train.png
        │   ├── clas-distribution-test.png
        │   └── show-batch.png
        └── metric/         # Biểu đồ đánh giá định lượng
            ├── confusion-matrix.png
            └── pre-recall-f1-acc.png
```

---

## ⚙️ 4. Quy Trình Kỹ Thuật (Engineering Pipeline)

1. **Chuẩn bị Dữ liệu & Tăng Cường (Data Preparation & Augmentation)**:
   - Áp dụng các biến đổi hình học ngẫu nhiên (Random Crop, Random Horizontal Flip, Normalize) từ cấu hình chung.
   - Quản lý dữ liệu qua `DataLoader` hỗ trợ đa luồng (`num_workers`) và `pin_memory=True` khi chạy trên GPU.
2. **Khóa Ngẫu Nhiên (Reproducibility)**:
   - Sử dụng hàm `set_seed()` tại [`../../../src/utility.py`](../../../src/utility.py) để cố định seed cho Python, NumPy và PyTorch (bao gồm cả `cudnn.deterministic = True`).
3. **Huấn Luyện & Tối Ưu (Training & Optimization)**:
   - Hàm mất mát: `nn.CrossEntropyLoss()`.
   - Thuật toán tối ưu: `optim.Adam(lr=0.001)`.
   - Hỗ trợ lưu checkpoint và phục hồi trọng số (`resume_weight`).
4. **Đánh Giá Đa Chiều (Evaluation)**:
   - Tính toán đầy đủ: Accuracy, Precision, Recall, Macro-F1.
   - Vẽ Ma trận nhầm lẫn (Confusion Matrix) chi tiết giữa 10 lớp để phát hiện các cặp lớp dễ nhầm lẫn (như `Shirt` vs `T-shirt/top` hay `Coat` vs `Pullover`).

---

## 🚀 5. Hướng Dẫn Cài Đặt & Thực Thi

### 5.1. Chạy Huấn luyện & Đánh giá (CLI)

Chạy trực tiếp pipeline hoàn chỉnh từ thư mục `labs/lab1/main`:

```bash
python main.py
```

### 5.2. Chạy Giao Diện Web Suy Luận (Gradio App)

Hệ thống tích hợp sẵn ứng dụng Gradio phục vụ kiểm tra thực tế:

```bash
python inference/app.py
```

- Mở trình duyệt tại địa chỉ: `http://localhost:7860`.
- Cho phép duyệt qua các ảnh trong thư mục `inference/input/`, xem phân trang (Pagination) và hiển thị kết quả Top-k phân loại kèm xác suất phần trăm.

### 5.3. Giám sát Huấn luyện Qua TensorBoard

```bash
tensorboard --logdir=runs
```

Mở `http://localhost:6006` để quan sát đường cong Loss, Accuracy theo từng Epoch.

---

## 📊 6. Kết Quả Thực Nghiệm & Trực Quan Hóa

Toàn bộ kết quả đầu ra được tự động lưu trong thư mục `result/fashion-normal/`:
- **Ma trận nhầm lẫn**: `result/fashion-normal/metric/confusion-matrix.png`
- **Bảng chỉ số Precision, Recall, F1-Score**: `result/fashion-normal/metric/pre-recall-f1-acc.png`
- **Kiểm tra phân bố tập Train/Test**: `result/fashion-normal/data-visual/clas-distribution-train.png`
- **Lưới ảnh trực quan Batch đầu vào**: `result/fashion-normal/data-visual/show-batch.png`

---

## 🔗 7. Liên Kết Mã Nguồn Liên Quan

- Module kiến trúc mạng: [`FashionCNN` in `src/model.py`](../../../src/model.py)
- Module tiền xử lý & DataLoaders: [`src/dataset.py`](../../../src/dataset.py)
- Module trực quan hóa đồ thị: [`src/visualize.py`](../../../src/visualize.py)
- Cấu hình siêu tham số tập trung: [`src/config.py`](../../../src/config.py)
- Bộ quy chuẩn phát triển AI: [`.agents/AI_REFERENCE.md`](../../../.agents/AI_REFERENCE.md)
