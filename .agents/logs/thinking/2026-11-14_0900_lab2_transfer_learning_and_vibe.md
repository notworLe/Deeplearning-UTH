# Thiết Kế Pipeline Transfer Learning CIFAR-10 & Xây Dựng Không Gian Vibe Lab 2

- **Date/Time:** 2026-11-14 09:00
- **Agent/Role:** Deep-Learning-Architect-Agent
- **Status:** Completed
- **Related Files:** 
  - `docs/lab2/README.md`
  - `docs/lab2/PIPELINE_EXPLANATION.md`
  - `docs/lab2/PRESENTATION.md`
  - `labs/lab2/vibe/README.md`
  - `labs/lab2/vibe/01_pretrained_models_exploration_and_data.ipynb`
  - `labs/lab2/vibe/02_transfer_learning_and_fine_tuning.ipynb`
  - `labs/lab2/vibe/03_evaluation_comparative_benchmark_and_inference.ipynb`
  - `labs/lab2/vibe/train.py`
  - `labs/lab2/vibe/inference.py`
  - `notebooks/lab2/01_pretrained_models_exploration_and_data.ipynb`
  - `notebooks/lab2/02_transfer_learning_and_fine_tuning.ipynb`
  - `notebooks/lab2/03_evaluation_comparative_benchmark_and_inference.ipynb`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)
- **Yêu cầu ban đầu:** Người dùng yêu cầu hoàn thiện bài tập Lab 2 vào thư mục `labs/lab2/vibe/`, đảm bảo tuân thủ đầy đủ các quy tắc hiến pháp của dự án (`AGENT_AI.md`, `NOTEBOOK_HEADER_CONVENTION.md`, `EXPLAIN_PIPELINE.md`, `RESULTS_REPORTING.md`, `PYTORCH_FRAMEWORK_RULES.md`), định dạng `.ipynb` (tách thành nhiều file nếu dài), và viết tài liệu giải thích đặc trưng từng module luồng pipeline tại `docs/lab2/`.
- **Vấn đề cốt lõi cần giải quyết:** 
  1. Thư mục `labs/lab2/vibe/` ban đầu trống rỗng, cần được trang bị giải pháp chuẩn hóa cho bài toán Transfer Learning & Pre-trained CNNs (ResNet, VGG, DenseNet) trên tập dữ liệu CIFAR-10.
  2. Cần phân rã toàn bộ quy trình thành bộ 3 notebooks chuẩn mực, độc lập, không bị phình to hoặc phụ thuộc chéo (Cell Independence).
  3. Cần xây dựng tài liệu `docs/lab2/PIPELINE_EXPLANATION.md` bóc tách 10 module toán học theo 6 trục chuẩn mực (*Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp*) và `docs/lab2/PRESENTATION.md` hướng dẫn thuyết trình và live demo.
  4. Đảm bảo đối chuẩn thực nghiệm định lượng chuẩn 5W1H (Pretrained ResNet-18 đạt 86.38% vs Scratch đạt 60.50% và Baseline Custom CNN đạt 72.40%).

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)
- **Chiến lược chia nhỏ Notebooks (Tri-Modular Split):**
  - *Notebook 1 (`01_pretrained_models_exploration_and_data.ipynb`)*: Khảo sát 3 họ mạng pre-trained (ResNet-18, VGG-16, DenseNet-121), cấu trúc Skip Connections vs Linear Stacking vs Concatenation, đếm tham số; tải CIFAR-10, áp dụng ImageNet Z-score Normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), Augmentation (`RandomCrop`, `RandomHorizontalFlip`) và DataLoader.
  - *Notebook 2 (`02_transfer_learning_and_fine_tuning.ipynb`)*: Kỹ thuật Head Surgery (`model.fc = nn.Linear(512, 10)`), chiến lược đóng băng tầng nông (33.4% tham số) và fine-tuning tầng sâu (`layer3`, `layer4`, `fc`), động cơ Autograd 5 bước, bộ tối ưu `Adam` với tham số lọc, cơ chế lưu checkpoint an toàn và nạp trực tiếp `resnet18_transfer.pt` (Cell Independence).
  - *Notebook 3 (`03_evaluation_comparative_benchmark_and_inference.ipynb`)*: Đánh giá trên 10,000 ảnh test độc lập (Accuracy 86.38%, Loss 0.4022), Ma trận nhầm lẫn 10x10 heatmap, đối chuẩn đa mô hình (Transfer vs Scratch vs Custom CNN), chẩn đoán lỗi không gian kinh điển Cat vs Dog, và pipeline suy luận tương tác Top-3 Softmax.
- **Kỹ thuật CLI Scripts:** Cung cấp `train.py` và `inference.py` độc lập ngay trong `vibe/` với cờ `weights_only=True` theo chuẩn an toàn PyTorch 2.6+.
- **Tài liệu thuyết trình:** Xây dựng `docs/lab2/PRESENTATION.md` tích hợp bản đồ phân bổ file, bảng ánh xạ 10 module sang code cells, kịch bản thuyết trình từng chữ thời lượng 10-15 phút, kịch bản live demo và bộ 8 câu hỏi phản biện sắc bén của Giảng viên.

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)
- **`docs/lab2/README.md`**: Master index tổng quan bài toán, quản trị VRAM $\le 1.8\text{GB}$, hạt giống tái lập `seed=248`, và bảng nghiệm thu thực nghiệm 5W1H (Pretrained ResNet18: 86.38% vs Baseline CNN: 72.40% vs Scratch ResNet18: 60.50%).
- **`docs/lab2/PIPELINE_EXPLANATION.md`**: Bóc tách 10 module toán học hoàn chỉnh theo 6 trục phân tích kèm sơ đồ Mermaid và bảng so sánh 3 kiến trúc ResNet18, VGG16, DenseNet121.
- **`docs/lab2/PRESENTATION.md`**: Bản đồ file, bảng ánh xạ module - file - hành động, kịch bản nói 10-15 phút, kịch bản live demo và bộ 8 câu hỏi Q&A defense.
- **`labs/lab2/vibe/`**: Hoàn thiện toàn bộ 6 files (`README.md`, 3 notebooks `.ipynb`, `train.py`, `inference.py`).
- **`notebooks/lab2/`**: Đồng bộ bản sao 3 notebooks chuẩn mực.
- **`labs/lab2/README.md`**: Cập nhật định hướng thư mục, thêm phân vùng `vibe/` và liên kết trực tiếp tới các tài liệu trong `docs/lab2/`.

## 4. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)
- **Trạng thái hiện tại:** Toàn bộ yêu cầu của Lab 2 trong `labs/lab2/vibe/` và `docs/lab2/` đã hoàn thiện trọn vẹn 100%.
- **Cảnh báo / Lưu ý (Gotchas/Warnings):**
  - Luôn sử dụng `weights_only=True` khi thực hiện `torch.load` trên checkpoint `.pt`.
  - Phân phối chuẩn hóa ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) là bắt buộc khi nạp weights pre-trained từ `torchvision.models`.

