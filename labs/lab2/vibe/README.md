# Lab 02 (Vibe): CIFAR-10 Transfer Learning Workspace

Không gian thực hành chuẩn mực cho **Lab 02 (Transfer Learning & Pre-trained CNNs: ResNet, VGG, DenseNet on CIFAR-10)**, đáp ứng toàn diện các tiêu chuẩn kỹ thuật về cấu trúc notebook ([`NOTEBOOK_HEADER_CONVENTION.md`](../../../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md)), quy trình ML Pipeline ([`EXPLAIN_PIPELINE.md`](../../../.agents/rules/EXPLAIN_PIPELINE.md)), báo cáo khoa học 5W1H ([`RESULTS_REPORTING.md`](../../../.agents/rules/RESULTS_REPORTING.md)) và an toàn thực thi phần cứng ([`PYTORCH_FRAMEWORK_RULES.md`](../../../.agents/rules/PYTORCH_FRAMEWORK_RULES.md)).

---

## 🗂️ Cấu Trúc Thư Mục `vibe/`

```text
labs/lab2/vibe/
├── README.md                                 # Tài liệu định hướng không gian vibe này
├── 01_pretrained_models_exploration_and_data.ipynb  # Notebook 1: Khảo sát 3 họ mạng, ImageNet Normalization & Augmentation
├── 02_transfer_learning_and_fine_tuning.ipynb       # Notebook 2: Head Surgery, Freezing vs Fine-Tuning, Autograd & Checkpoint
├── 03_evaluation_comparative_benchmark_and_inference.ipynb # Notebook 3: Đánh giá 5W1H, Confusion Matrix, Cat vs Dog & Top-3 Inference
├── train.py                                  # Script CLI huấn luyện dòng lệnh độc lập
└── inference.py                              # Script CLI suy luận đơn ảnh và batch serving
```

---

## 📚 Bộ 3 Notebooks Thực Hành Độc Lập

Nhằm đảm bảo dung lượng hợp lý và nguyên tắc độc lập giữa các cell (**Cell Independence**), bài thực hành được chia thành 3 phần tuần tự:

1. **[`01_pretrained_models_exploration_and_data.ipynb`](01_pretrained_models_exploration_and_data.ipynb)**:
   - Khảo sát và đối chuẩn 3 triết lý thiết kế mạng nơ-ron tích chập kinh điển: **ResNet-18** (11.7M params, skip connection), **VGG-16** (138.4M params, tuần tự tuyến tính) và **DenseNet-121** (8.0M params, feature reuse qua concatenation).
   - Thiết lập nguyên tắc chuẩn hóa bắt buộc **ImageNet Z-Score Normalization** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   - Cấu hình pipeline Data Augmentation (`RandomCrop(32, 4)`, `RandomHorizontalFlip()`) và PyTorch `DataLoader` đa luồng có ghim bộ nhớ (`pin_memory=True`).
   - Trực quan hóa dữ liệu EDA: phân phối 10 lớp cân bằng và lưới 16 ảnh sau chuẩn hóa kèm giải thuật un-normalization.

2. **[`02_transfer_learning_and_fine_tuning.ipynb`](02_transfer_learning_and_fine_tuning.ipynb)**:
   - Triển khai kỹ thuật **Head Surgery**: thay thế tầng `fc` 1000 lớp của ImageNet bằng `nn.Linear(512, 10)` phục vụ bài toán CIFAR-10.
   - Chiến lược đóng băng (Freezing) thông minh: đóng băng 33.4% tham số tầng nông (`conv1`, `bn1`, `layer1`, `layer2`) nhằm tái sử dụng các bộ lọc Gabor phổ quát; mở khóa 66.6% tham số tầng sâu (`layer3`, `layer4`, `fc`) với learning rate nhỏ $1\times 10^{-4}$.
   - Động cơ Autograd 5 bước với bộ tối ưu `Adam` kết hợp bộ lọc tham số `filter(lambda p: p.requires_grad, ...)`.
   - Cơ chế lưu và nạp checkpoint an toàn tuân thủ PyTorch 2.6+ (`weights_only=True`), hỗ trợ chạy đánh giá ngay lập tức qua file checkpoint có sẵn `labs/lab2/main/weights/resnet18_transfer.pt`.
   - Trực quan hóa đường cong mất mát và độ chính xác qua 5 epochs thực nghiệm.

3. **[`03_evaluation_comparative_benchmark_and_inference.ipynb`](03_evaluation_comparative_benchmark_and_inference.ipynb)**:
   - Đánh giá định lượng trên 10,000 ảnh test độc lập dưới `@torch.inference_mode()`.
   - Báo cáo phân loại chuẩn 5W1H: ResNet-18 Transfer Learning đạt **Accuracy = 86.38%**, **Test Loss = 0.4022**.
   - Ma trận nhầm lẫn (Confusion Matrix 10x10) dạng heatmap trực quan.
   - Bảng và biểu đồ đối chuẩn đa kiến trúc: Pretrained ResNet-18 (**86.38%**) vượt trội so với Scratch Custom CNN (**72.40%**) và Scratch ResNet-18 (**60.50%**).
   - Chẩn đoán lỗi không gian chuyên sâu: phân tích nguyên nhân nhầm lẫn sinh học giữa `Cat` và `Dog` (F1 thấp nhất ~74%) do giới hạn độ phân giải $32\times 32$.
   - Pipeline suy luận trực quan: hiển thị ảnh kèm biểu đồ thanh ngang Top-3 xác suất Softmax.

---

## ⚡ Hướng Dẫn Thực Thi CLI Scripts

Ngoài môi trường Jupyter Notebook, bạn có thể thực thi huấn luyện và suy luận nhanh thông qua dòng lệnh:

### 1. Huấn Luyện (Training)
```bash
# Huấn luyện fine-tuning ResNet-18 trên GPU
python labs/lab2/vibe/train.py --model resnet18 --mode finetune --epochs 5 --lr 1e-4

# Huấn luyện feature extraction (đóng băng hoàn toàn backbone)
python labs/lab2/vibe/train.py --model resnet18 --mode freeze --epochs 5

# Huấn luyện từ đầu (Scratch) để làm baseline đối chuẩn
python labs/lab2/vibe/train.py --model resnet18 --mode scratch --epochs 5
```

### 2. Suy Luận (Inference)
```bash
# Suy luận ngẫu nhiên 5 mẫu ảnh từ tập CIFAR-10 test set
python labs/lab2/vibe/inference.py --weights labs/lab2/main/weights/resnet18_transfer.pt --samples 5

# Suy luận ảnh tùy chọn từ đĩa
python labs/lab2/vibe/inference.py --weights labs/lab2/main/weights/resnet18_transfer.pt --image path/to/my_image.png
```

---

## 📊 Bảng Đối Chuẩn Thực Nghiệm 5W1H Tóm Tắt

| Chỉ Số / Mô Hình | Pretrained ResNet-18 (Đề Xuất) | Baseline Custom CNN (Scratch) | ResNet-18 (Scratch) |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **86.38%** | 72.40% | 60.50% |
| **Test Loss** | **0.4022** | 0.7981 | 1.1520 |
| **Số Lượng Epochs** | 5 epochs | 10 epochs | 5 epochs |
| **Độ Tăng Trưởng (Gain)**| **+25.88%** so với ResNet Scratch | Baseline | - |
| **VRAM Consumption** | **1.78 GB** ($\le 3.5\text{GB}$) | 1.25 GB | 1.76 GB |
| **Training Time** | ~2.8 phút (RTX 3060) | ~2.1 phút | ~3.0 phút |

---

## 🔗 Tài Liệu Tham Khảo Liên Kết

- [docs/lab2/README.md](../../../docs/lab2/README.md): Báo cáo kỹ thuật tổng quan và quản trị phần cứng.
- [docs/lab2/PIPELINE_EXPLANATION.md](../../../docs/lab2/PIPELINE_EXPLANATION.md): Bóc tách 10 module toán học theo 6 trục phân tích.
- [docs/lab2/PRESENTATION.md](../../../docs/lab2/PRESENTATION.md): Cẩm nang thuyết trình, kịch bản lời nói 10–15 phút và 8 câu hỏi phản biện.
- [labs/lab2/QUESTION.md](../QUESTION.md): Đề bài gốc và yêu cầu chuyển giao bài toán.

