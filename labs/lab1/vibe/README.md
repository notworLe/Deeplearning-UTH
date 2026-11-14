# Lab 01 (Vibe): Fashion-MNIST Classification Workspace

Không gian thực hành chuẩn mực cho **Lab 01 (Fashion-MNIST Classification with PyTorch)**, đáp ứng toàn diện các tiêu chuẩn kỹ thuật về cấu trúc notebook ([`NOTEBOOK_HEADER_CONVENTION.md`](../../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md)), quy trình ML Pipeline ([`EXPLAIN_PIPELINE.md`](../../../.agents/rules/EXPLAIN_PIPELINE.md)), báo cáo khoa học 5W1H ([`RESULTS_REPORTING.md`](../../../.agents/rules/RESULTS_REPORTING.md)) và an toàn thực thi phần cứng ([`PYTORCH_FRAMEWORK_RULES.md`](../../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md)).

---

## 🗂️ Cấu Trúc Thư Mục `vibe/`

```text
labs/lab1/vibe/
├── README.md                                 # Tài liệu định hướng không gian vibe này
├── 01_data_exploration_and_transforms.ipynb # Notebook 1: Tensors, Transforms, DataLoader & EDA
├── 02_model_building_and_training.ipynb  # Notebook 2: MLP vs FashionCNN, Autograd, Loss & Training Loop
├── 03_evaluation_and_error_analysis.ipynb# Notebook 3: Metrics 5W1H, Confusion Matrix, Prediction Grid & Inference
├── train.py                                  # Script CLI huấn luyện dòng lệnh độc lập
└── inference.py                              # Script CLI suy luận đơn ảnh và batch serving
```

---

## 📚 Bộ 3 Notebooks Thực Hành Độc Lập

Nhằm đảm bảo dung lượng hợp lý và nguyên tắc độc lập giữa các cell (**Cell Independence**), bài thực hành được chia thành 3 phần tuần tự:

1. **[`01_data_exploration_and_transforms.ipynb`](01_data_exploration_and_transforms.ipynb)**:
   - Làm chủ các thao tác với PyTorch Tensors (khởi tạo, shape, device chuyển đổi).
   - Tải dataset Fashion-MNIST và áp dụng chuẩn hóa Z-Score ($\mu=0.2860, \sigma=0.3530$).
   - Thiết lập `DataLoader` đa luồng có ghim bộ nhớ (`pin_memory=True`).
   - Khảo sát dữ liệu EDA: biểu đồ 10 lớp cân bằng (6,000 ảnh/lớp), histogram cường độ pixel, và lưới 16 ảnh mẫu có nhãn.

2. **[`02_model_building_and_training.ipynb`](02_model_building_and_training.ipynb)**:
   - Xây dựng 2 kiến trúc đối sánh: **Baseline MLP** (235K tham số) vs **Deep `FashionCNN`** (150K tham số).
   - Mổ xẻ cơ chế tính toán đạo hàm tự động của PyTorch Autograd.
   - Thiết lập hàm mất mát `CrossEntropyLoss` và bộ tối ưu `Adam` (lr=0.001).
   - Vòng lặp huấn luyện chuẩn 5 bước PyTorch, ghi nhật ký và lưu checkpoint an toàn với `weights_only=True`.

3. **[`03_evaluation_and_error_analysis.ipynb`](03_evaluation_and_error_analysis.ipynb)**:
   - Đánh giá trên 10,000 ảnh test độc lập dưới `@torch.no_grad()`.
   - Báo cáo phân loại chi tiết: `FashionCNN` đạt **Accuracy = 91.24%**, **Macro-F1 = 91.18%**.
   - Biểu đồ nhiệt Ma trận nhầm lẫn (Confusion Matrix Heatmap).
   - Lưới 25 ảnh dự đoán thực tế tô màu trực quan (Xanh: Đúng, Đỏ: Sai).
   - Chẩn đoán các cặp lỗi không gian kinh điển (`Shirt` vs `T-shirt/top`, `Coat` vs `Pullover`).
   - Hàm suy luận tương tác `predict_sample()` hiển thị phân phối xác suất Top-3.

---

## 🚀 Hướng Dẫn Sử Dụng Scripts Dòng Lệnh (CLI)

Tuân thủ quy tắc [`AGENT_AI.md`](../../../.agents/rules/AGENT_AI.md) ("All training must run via CLI Python scripts"):

### 1. Huấn Luyện Bằng Dòng Lệnh
```bash
# Huấn luyện mô hình FashionCNN mặc định
python train.py --epochs 10 --batch-size 64 --lr 0.001 --model cnn

# Huấn luyện mô hình MLP baseline để so sánh
python train.py --epochs 10 --batch-size 64 --lr 0.001 --model mlp
```

### 2. Suy Luận Bằng Dòng Lệnh
```bash
# Đánh giá suy luận trên mẫu ngẫu nhiên từ tập test
python inference.py --sample-idx 18
```

---

## 📑 Tài Liệu Bóc Tách Kỹ Thuật Liên Quan

* 👉 **[docs/lab1/PIPELINE_EXPLANATION.md](../../../docs/lab1/PIPELINE_EXPLANATION.md)**: Phân tích 10 Module toán học của Computer Vision Pipeline theo 6 trục chuẩn mực (*Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp*).
* 👉 **[docs/lab1/PRESENTATION.md](../../../docs/lab1/PRESENTATION.md)**: Kịch bản thuyết trình chi tiết 10-15 phút, bản đồ file và bộ 8 câu hỏi phản biện Q&A.
* 👉 **[docs/lab1/README.md](../../../docs/lab1/README.md)**: Master index định hướng nghiên cứu và báo cáo số liệu 5W1H.

