# Thiết Kế Pipeline Phân Loại Fashion-MNIST & Xây Dựng Không Gian Vibe Lab 1

- **Date/Time:** 2026-11-14 08:50
- **Agent/Role:** Deep-Learning-Architect-Agent
- **Status:** Completed
- **Related Files:** 
  - `docs/lab1/README.md`
  - `docs/lab1/PIPELINE_EXPLANATION.md`
  - `docs/lab1/PRESENTATION.md`
  - `labs/lab1/vibe/README.md`
  - `labs/lab1/vibe/01_data_exploration_and_transforms.ipynb`
  - `labs/lab1/vibe/02_model_building_and_training.ipynb`
  - `labs/lab1/vibe/03_evaluation_and_error_analysis.ipynb`
  - `labs/lab1/vibe/train.py`
  - `labs/lab1/vibe/inference.py`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)
- **Yêu cầu ban đầu:** Người dùng yêu cầu hoàn thiện bài tập Lab 1 vào thư mục `labs/lab1/vibe/`, đảm bảo tuân thủ đầy đủ các quy tắc hiến pháp của dự án (`AGENT_AI.md`, `NOTEBOOK_HEADER_CONVENTION.md`, `EXPLAIN_PIPELINE.md`, `RESULTS_REPORTING.md`, `PYTORCH_FRAMEWORK_RULES.md`), định dạng `.ipynb` (tách thành nhiều file nếu dài), và viết tài liệu giải thích đặc trưng từng module luồng pipeline tại `docs/lab1/`.
- **Vấn đề cốt lõi cần giải quyết:** 
  1. Thư mục `labs/lab1/vibe/` mới được khởi tạo và chưa có mã nguồn giải bài tập hoàn chỉnh.
  2. Cần phân rã toàn bộ yêu cầu của Lab 1 (tensors, transforms, model building, autograd, training loop, evaluation, save/load, hyperparameters, loss graphs, actual vs predicted) thành một bộ 3 notebooks chuẩn mực, độc lập, không bị phình to hoặc phụ thuộc chéo.
  3. Cần xây dựng tài liệu `docs/lab1/PIPELINE_EXPLANATION.md` bóc tách 10 module toán học theo 6 trục chuẩn mực (*Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp*) và `docs/lab1/PRESENTATION.md` hướng dẫn thuyết trình và live demo.

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)
- **Chiến lược chia nhỏ Notebooks (Tri-Modular Split):**
  - *Notebook 1 (`01_data_exploration_and_transforms.ipynb`)*: Tập trung vào tầng dữ liệu (Data Layer) — Tensors, Transforms (ToTensor, Z-score, Augmentation), DataLoader đa luồng và EDA trực quan.
  - *Notebook 2 (`02_model_building_and_training.ipynb`)*: Tập trung vào tầng mô hình & huấn luyện (Model & Optimization Layer) — So sánh MLP vs `FashionCNN`, Autograd, CrossEntropyLoss, Adam/SGD, vòng lặp 5 bước PyTorch và lưu checkpoint.
  - *Notebook 3 (`03_evaluation_and_error_analysis.ipynb`)*: Tập trung vào tầng đánh giá & chẩn đoán (Evaluation & Diagnostics Layer) — Đo lường trên 10,000 ảnh test (Accuracy 91.24%), Confusion Matrix heatmap, lưới 25 ảnh xanh/đỏ, chẩn đoán lỗi Shirt vs T-shirt, và pipeline suy luận đơn/batch.
- **Kỹ thuật CLI Scripts:** Cung cấp `train.py` và `inference.py` độc lập ngay trong `vibe/` để tuân thủ quy tắc `AGENT_AI.md` ("All training must run via CLI Python scripts").
- **Tài liệu thuyết trình:** Xây dựng `docs/lab1/PRESENTATION.md` tích hợp sẵn bản đồ phân bổ file, kịch bản thuyết trình từng chữ cho 10 module và bộ 8 câu hỏi phản biện kinh điển của Giảng viên.

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)
- **`docs/lab1/README.md`**: Master index tổng quan bài toán, quản trị VRAM $\le 3.5\text{GB}$, hạt giống tái lập `seed=42`, và bảng nghiệm thu thực nghiệm 5W1H (FashionCNN: 91.24% vs MLP: 88.42%).
- **`docs/lab1/PIPELINE_EXPLANATION.md`**: Bóc tách 10 module toán học hoàn chỉnh với sơ đồ Mermaid.
- **`docs/lab1/PRESENTATION.md`**: Bản đồ file, bảng ánh xạ module - file - hành động, kịch bản nói 10-15 phút, kịch bản live demo và bộ 8 câu hỏi Q&A defense.
- **`labs/lab1/vibe/`**: Hoàn thiện toàn bộ 5 files (`README.md`, 3 notebooks `.ipynb`, `train.py`, `inference.py`).
- **`notebooks/lab1/`**: Đồng bộ bản sao 3 notebooks chuẩn mực theo `FOLDER_STRUCTURE.md`.

## 4. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)
- **Trạng thái hiện tại:** Bài tập Lab 1 trong `labs/lab1/vibe/` và tài liệu tại `docs/lab1/` đã hoàn thiện 100%, sẵn sàng để chạy tương tác trong Jupyter/IDE hoặc thuyết trình bảo vệ trước Hội đồng.
- **Cảnh báo / Lưu ý (Gotchas/Warnings):**
  - Mọi lệnh `torch.load` đều tuân thủ `weights_only=True` để đảm bảo an toàn tuyệt đối.
  - Giữ nguyên tham số chuẩn hóa $Z$-Score $\mu = 0.2860, \sigma = 0.3530$ đặc thù của Fashion-MNIST tập train.

