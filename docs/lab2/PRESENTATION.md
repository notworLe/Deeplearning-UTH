# PRESENTATION.md — Kịch Bản Thuyết Trình Toàn Diện Module PyTorch Pipeline Lab 02

- **Motivation/Background**: Báo cáo học thuật về Transfer Learning & Pre-trained CNNs (ResNet, VGG, DenseNet) trên tập dữ liệu CIFAR-10 đòi hỏi người thuyết trình phải kết nối chặt chẽ giữa lý thuyết mạng nơ-ron tích chập sâu (Residual connections, Feature reuse, Receptive field) với quy trình phẫu thuật phân loại (Head surgery), chiến lược đóng băng tầng (Freezing vs Fine-tuning), đối chuẩn 5W1H và thao tác live demo suy luận tự tin trước Hội đồng / Giảng viên.
- **Purpose**: Cung cấp cẩm nang thuyết trình chuẩn mực cho Lab 02: định vị chi tiết vị trí từng file mã nguồn trong dự án, bảng ánh xạ module - mã nguồn - hành động trình chiếu, kịch bản lời thuyết trình từng phút (10–15 phút), kịch bản thao tác live demo và 8 câu hỏi phản biện kỹ thuật sắc bén kèm lời đáp.
- **Overview Pipeline**: Ánh xạ 10 module từ [docs/lab2/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md) và kết quả thực nghiệm chuẩn 5W1H từ [docs/lab2/README.md](README.md) vào kịch bản báo cáo thực tế.
- **Detailed Plan**: §1 Sơ đồ Bản đồ Phân bổ File; §2 Bảng Ánh xạ Module - File - Hành động Trình chiếu; §3 Kịch bản Lời Thuyết trình Chi tiết (Từng Slide/Module); §4 Kịch bản Thao tác Live Demo Trực tiếp; §5 Bộ Câu hỏi & Câu trả lời Phản biện với Giảng viên (Q&A Defense).
- **References**: [docs/lab2/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md), [docs/lab2/README.md](README.md), [labs/lab2/QUESTION.md](../../labs/lab2/QUESTION.md), [labs/lab2/vibe/README.md](../../labs/lab2/vibe/README.md), [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md).
- **Created**: 2026-11-14T09:05:00+07:00
- **Last Updated**: 2026-11-14T09:05:00+07:00

---

## Table of Contents

- [1. Sơ đồ Bản đồ Phân bổ File &amp; Vai trò Thuyết trình](#1-sơ-đồ-bản-đồ-phân-bổ-file--vai-trò-thuyết-trình)
- [2. Bảng Ánh xạ Module - File Mã Nguồn - Hành Động Trình Chiếu](#2-bảng-ánh-xạ-module---file-mã-nguồn---hành-động-trình-chiếu)
- [3. Kịch Bản Lời Thuyết Trình Chi Tiết (Script Từng Slide/Module)](#3-kịch-bản-lời-thuyết-trình-chi-tiết-script-từng-slidemodule)
  - [Phần 0: Lời Mở Đầu &amp; Đặt Vấn Đề (1 Phút)](#phần-0-lời-mở-đầu--đặt-vấn-đề-1-phút)
  - [Phần 1: Giới Thiệu Cấu Trúc Dự Án &amp; 3 Họ Kiến Trúc Pretrained (1.5 Phút)](#phần-1-giới-thiệu-cấu-trúc-dự-án--3-họ-kiến-trúc-pretrained-15-phút)
  - [Phần 2: Module 1, 2, 3 — Dữ Liệu CIFAR-10, ImageNet Normalization &amp; Augmentation (2 Phút)](#phần-2-module-1-2-3--dữ-liệu-cifar-10-imagenet-normalization--augmentation-2-phút)
  - [Phần 3: Module 4, 5 — Head Surgery &amp; Chiến Lược Đóng Băng / Fine-tuning (2.5 Phút)](#phần-3-module-4-5--head-surgery--chiến-lược-đóng-băng--fine-tuning-25-phút)
  - [Phần 4: Module 6, 7 — Engine Huấn Luyện &amp; Lưu Checkpoint An Toàn (2 Phút)](#phần-4-module-6-7--engine-huấn-luyện--lưu-checkpoint-an-toàn-2-phút)
  - [Phần 5: Module 8 — Nghiệm Thu Kết Quả Thực Nghiệm Chuẩn 5W1H (2 Phút)](#phần-5-module-8--nghiệm-thu-kết-quả-thực-nghiệm-chuẩn-5w1h-2-phút)
  - [Phần 6: Module 9, 10 — Ma Trận Nhầm Lẫn, Chẩn Đoán Lỗi &amp; Live Demo (2.5 Phút)](#phần-6-module-9-10--ma-trận-nhầm-lẫn-chẩn-đoán-lỗi--live-demo-25-phút)
  - [Phần 7: Tổng Kết &amp; Bài Học Kỹ Thuật (1 Phút)](#phần-7-tổng-kết--bài-học-kỹ-thuật-1-phút)
- [4. Kịch Bản Thao Tác Trình Diễn Trực Tiếp (Live Demo Script)](#4-kịch-bản-thao-tác-trình-diễn-trực-tiếp-live-demo-script)
- [5. Bộ Câu Hỏi &amp; Câu Trả Lời Phản Biện với Giảng Viên (Q&amp;A Defense)](#5-bộ-câu-hỏi--câu-trả-lời-phản-biện-với-giảng-viên-qa-defense)

---

## 1. Sơ đồ Bản đồ Phân bổ File & Vai trò Thuyết trình

Dưới đây là sơ đồ chỉ rõ file nào nằm ở đâu và bạn cần mở file nào trong từng giai đoạn báo cáo:

```text
Deeplearning-UTH/
│
├── docs/lab2/                                    <-- [NƠI CHIẾU BÁO CÁO TOÁN HỌC & LÝ THUYẾT]
│   ├── README.md                                 # Báo cáo tổng quan, Bảng đối chuẩn 5W1H, Giới hạn VRAM <= 1.8GB
│   ├── PIPELINE_EXPLANATION.md                   # Bóc tách 10 Module theo 6 trục (Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp)
│   └── PRESENTATION.md                           # Cẩm nang thuyết trình & kịch bản lời nói này
│
├── labs/lab2/
│   ├── QUESTION.md                               # Đề bài gốc: Transfer Learning ResNet, VGG, DenseNet
│   ├── README.md                                 # Chỉ dẫn chung của bài Lab 2
│   │
│   ├── vibe/                                     <-- [NƠI MỞ CODE VÀ THỰC HIỆN LIVE DEMO]
│   │   ├── README.md                             # Hướng dẫn thực thi không gian bài làm lab2/vibe
│   │   ├── 01_pretrained_models_exploration_and_data.ipynb  # [Demo 1]: Khảo sát ResNet/VGG/DenseNet, CIFAR-10 EDA & ImageNet Normalization
│   │   ├── 02_transfer_learning_and_fine_tuning.ipynb       # [Demo 2]: Head Surgery, Freezing layer1-2, Fine-tuning, TensorBoard curves
│   │   ├── 03_evaluation_comparative_benchmark_and_inference.ipynb # [Demo 3]: Benchmark 5W1H, Confusion Matrix, Chẩn đoán Cat/Dog & Top-3 Inference
│   │   ├── train.py                              # [CLI Tool]: Huấn luyện độc lập từ terminal (--model resnet18 --mode finetune)
│   │   └── inference.py                          # [CLI Tool]: Suy luận ảnh đơn lẻ / lô qua dòng lệnh
│   │
│   └── main/                                     <-- [NƠI LƯU TRỮ WEIGHTS ĐÃ HUẤN LUYỆN SẴN]
│       ├── weights/resnet18_transfer.pt          # Checkpoint ResNet-18 Transfer Learning (Acc: 86.38%, 43MB)
│       ├── weights/resnet18_baseline.pt          # Checkpoint Baseline ResNet-18 (Acc: 60.50%, 129MB)
│       └── result.txt                            # Nhật ký log quá trình huấn luyện thực tế
│
└── notebooks/lab2/                               <-- [BẢN SAO ĐỒNG BỘ CHO MÔI TRƯỜNG JUPYTER LAB]
    ├── 01_pretrained_models_exploration_and_data.ipynb
    ├── 02_transfer_learning_and_fine_tuning.ipynb
    └── 03_evaluation_comparative_benchmark_and_inference.ipynb
```

---

## 2. Bảng Ánh xạ Module - File Mã Nguồn - Hành Động Trình Chiếu

Khi trình bày đến từng module trong slide, hãy nhanh chóng mở đúng file và cell tương ứng theo bảng điều hướng dưới đây:

| STT | Tên Module Kỹ Thuật | File Cần Mở | Vị Trí Cell / Hàm Trọng Tâm | Hành Động Trình Chiếu Minh Họa |
| :--- | :--- | :--- | :--- | :--- |
| **M01** | Khảo sát 3 Họ Kiến Trúc Pretrained | `vibe/01_pretrained_models_exploration_and_data.ipynb` | `Step 2: Model Architecture Inspection` | Chỉ ra khác biệt cấu trúc: ResNet skip connections ($\mathcal{F}(x)+x$), VGG linear head, DenseNet concatenation. Show bảng đếm params (11.7M vs 138M vs 8M). |
| **M02** | CIFAR-10 Data Ingestion & ImageNet Stats | `vibe/01_pretrained_models_exploration_and_data.ipynb` | `Step 3: CIFAR-10 & Normalization` | Trình chiếu công thức Z-score ImageNet `mean=[0.485, 0.456, 0.406]`, giải thích tại sao không dùng CIFAR-10 mean cho pre-trained weights. |
| **M03** | Data Augmentation & DataLoader Engine | `vibe/01_pretrained_models_exploration_and_data.ipynb` | `Step 4: Augmentation & EDA Visuals` | Bật ảnh trực quan 16 mẫu sau `RandomCrop(32, padding=4)` và `RandomHorizontalFlip()`. Chứng minh `batch_size=32`, `pin_memory=True`. |
| **M04** | Head Surgery & Đóng Băng Tham Số | `vibe/02_transfer_learning_and_fine_tuning.ipynb` | `Step 2: Head Surgery & Freezing Strategy` | Highlight dòng code `model.fc = nn.Linear(512, 10)` và vòng lặp `param.requires_grad = False` (đóng băng 33.4% weights). |
| **M05** | Chiến Lược Fine-Tuning Đa Tầng | `vibe/02_transfer_learning_and_fine_tuning.ipynb` | `Step 3: Fine-Tuning Setup` | Giải thích mở khóa `layer3`, `layer4` và `fc` với learning rate nhỏ $1\times 10^{-4}$ để bảo tồn low-level Gabor filters. |
| **M06** | Engine Tối Ưu Hóa & Đạo Hàm Tự Động | `vibe/02_transfer_learning_and_fine_tuning.ipynb` | `Step 4: Training Engine Definition` | Trình bày 5 bước Autograd (`zero_grad` $\to$ `forward` $\to$ `criterion` $\to$ `backward` $\to$ `step`). Chỉ ra `filter(lambda p: p.requires_grad, ...)`. |
| **M07** | Quản Trị Trọng Số & Checkpointing | `vibe/02_transfer_learning_and_fine_tuning.ipynb` | `Step 5: Safe Checkpoint Saving/Loading` | Chỉ rõ tham số `weights_only=True` khi `torch.load()`. Demo Cell Independence: nạp thẳng `resnet18_transfer.pt` không cần re-train 5 epoch. |
| **M08** | Nghiệm Thu Thực Nghiệm Chuẩn 5W1H | `docs/lab2/README.md` & `vibe/03_evaluation_...ipynb` | `Step 2: Quantitative 5W1H Evaluation` | Chiếu bảng so sánh: Pretrained ResNet18 (**86.38%**) vượt trội so với Scratch ResNet18 (**60.50%**) và Custom CNN (**72.40%**). |
| **M09** | Ma Trận Nhầm Lẫn & Chẩn Đoán Không Gian | `vibe/03_evaluation_comparative_benchmark_and_inference.ipynb` | `Step 3 & 4: Confusion Matrix & Diagnostics` | Mở heatmap 10x10: Nhấn mạnh vùng nhầm lẫn lớn nhất `Cat` vs `Dog` (F1 ~74%) và `Automobile` vs `Truck`. Show ảnh mẫu sai biệt vi mô. |
| **M10** | Trình Diễn Dự Đoán & Top-3 Probabilities | `vibe/03_evaluation_comparative_benchmark_and_inference.ipynb` | `Step 6: Interactive Inference & Top-3 Prob` | Nhập ảnh CIFAR-10 bất kỳ (hoặc chạy `python vibe/inference.py`), xuất biểu đồ thanh Top-3 xác suất với `@torch.inference_mode()`. |

---

## 3. Kịch Bản Lời Thuyết Trình Chi Tiết (Script Từng Slide/Module)

> [!TIP]
> Hãy giữ phong thái tự tin, ngắt nhịp rõ ràng ở các con số định lượng (86.38%, 11.7M tham số, 1.8GB VRAM) và luôn hướng màn hình tới đúng file code/tài liệu được chỉ định.

### Phần 0: Lời Mở Đầu & Đặt Vấn Đề (1 Phút)
- **Tài liệu chiếu**: Slide bìa hoặc [docs/lab2/README.md](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/docs/lab2/README.md).
- **Lời thuyết trình**:
  > *"Kính thưa Thầy/Cô và các bạn, hôm nay nhóm em xin phép báo cáo bài thực hành Lab 02 với chuyên đề: **Transfer Learning & Pre-trained Deep CNNs trên tập dữ liệu CIFAR-10**.*
  >
  > *Trong thị giác máy tính thực tế, huấn luyện một mạng nơ-ron tích chập sâu hàng chục tầng từ đầu (from scratch) với lượng dữ liệu hạn chế như CIFAR-10 (50.000 ảnh kích thước nhỏ 32x32) thường gặp hai trở ngại lớn: hiện tượng quá khớp (overfitting) và hội tụ chậm chạp. Trong bài thực hành này, nhóm em triển khai kỹ thuật **Học Chuyển Giao (Transfer Learning)**, kế thừa kho tàng tri thức biểu diễn thị giác từ tập dữ liệu khổng lồ ImageNet (1.2 triệu ảnh, 1000 lớp) thông qua 3 họ kiến trúc kinh điển: **ResNet-18, VGG-16 và DenseNet-121**.*
  >
  > *Nhóm em đã đóng gói toàn bộ quy trình thành 10 module kỹ thuật khép kín, phân tích định lượng chuẩn 5W1H và tối ưu hóa tài nguyên phần cứng nghiêm ngặt dưới 1.8GB VRAM."*

---

### Phần 1: Giới Thiệu Cấu Trúc Dự Án & 3 Họ Kiến Trúc Pretrained (1.5 Phút)
- **Tài liệu chiếu**: [docs/lab2/PIPELINE_EXPLANATION.md](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/docs/lab2/PIPELINE_EXPLANATION.md) (Mục 1 & 2) và [01_pretrained_models_exploration_and_data.ipynb](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/labs/lab2/vibe/01_pretrained_models_exploration_and_data.ipynb) (Cell Step 2).
- **Lời thuyết trình**:
  > *"Để có cái nhìn toàn diện, nhóm em tiến hành so sánh đối chuẩn 3 triết lý thiết kế mạng nơ-ron tích chập tiêu biểu:*
  >
  > 1. *Thứ nhất là **ResNet-18** (Residual Networks): Sử dụng liên kết tắt (skip connection) $H(x) = \mathcal{F}(x) + x$, giải quyết triệt để vấn đề tiêu biến đạo hàm (vanishing gradient) khi tăng độ sâu mạng, tổng tham số chỉ 11.7 triệu.*
  > 2. *Thứ hai là **VGG-16**: Thiết kế tuần tự tuyến tính truyền thống với chuỗi tích chập nhỏ 3x3 xếp chồng. Nhược điểm lớn nhất là phần Fully Connected Classifier chiếm tới hơn 100 triệu tham số, đẩy tổng tham số lên 138.4 triệu, gây tiêu tốn bộ nhớ và tính toán.*
  > 3. *Thứ ba là **DenseNet-121**: Thiết lập liên kết dày đặc, ghép kênh (concatenation) đầu ra của tất cả các tầng trước đó vào tầng sau. Giúp tái sử dụng đặc trưng tối đa và chỉ cần 8.0 triệu tham số.*
  >
  > *Trong khuôn khổ bài lab, nhóm em chọn **ResNet-18** làm kiến trúc chủ lực để huấn luyện chuyển giao nhờ tỷ lệ vàng giữa độ chính xác, tốc độ hội tụ và tiêu thụ bộ nhớ."*

---

### Phần 2: Module 1, 2, 3 — Dữ Liệu CIFAR-10, ImageNet Normalization & Augmentation (2 Phút)
- **Tài liệu chiếu**: [01_pretrained_models_exploration_and_data.ipynb](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/labs/lab2/vibe/01_pretrained_models_exploration_and_data.ipynb) (Cell Step 3 & Step 4).
- **Lời thuyết trình**:
  > *"Bước vào khâu xử lý dữ liệu, tập CIFAR-10 gồm 60.000 ảnh màu $32\times 32$ chia đều cho 10 lớp. Tại đây, nhóm em xử lý 2 vấn đề kỹ thuật trọng tâm:*
  >
  > *Đầu tiên là **Chuẩn hóa Z-Score ImageNet**: Khi sử dụng mô hình pre-trained, ta bắt buộc phải chuẩn hóa ảnh theo phân phối màu của tập ImageNet gốc với `mean = [0.485, 0.456, 0.406]` và `std = [0.229, 0.224, 0.225]`. Nếu dùng thống kê của riêng CIFAR-10, phân phối đầu vào sẽ bị lệch (distribution drift), phá vỡ các bộ lọc tích chập đã học ở tầng đầu.*
  >
  > *Thứ hai là **Data Augmentation**: Vì ảnh $32\times 32$ rất dễ bị overfit, nhóm áp dụng `RandomCrop(32, padding=4)` và `RandomHorizontalFlip(p=0.5)` trên tập Train, giúp mô hình bất biến với dịch chuyển không gian và phản chiếu đối xứng. Tập Validation và Test được giữ nguyên vẹn để bảo đảm tính khách quan.*
  >
  > *Toàn bộ dữ liệu được nạp qua PyTorch `DataLoader` với `batch_size=32`, kích hoạt `pin_memory=True` để chuyển dữ liệu trực tiếp vào GPU nhanh chóng mà không gây nghẽn CPU."*

---

### Phần 3: Module 4, 5 — Head Surgery & Chiến Lược Đóng Băng / Fine-tuning (2.5 Phút)
- **Tài liệu chiếu**: [02_transfer_learning_and_fine_tuning.ipynb](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/labs/lab2/vibe/02_transfer_learning_and_fine_tuning.ipynb) (Cell Step 2 & Step 3).
- **Lời thuyết trình**:
  > *"Điểm mấu chốt của Transfer Learning nằm ở hai bước: **Phẫu thuật đầu phân loại (Head Surgery)** và **Chiến lược lan truyền đạo hàm (Freezing Strategy)**.*
  >
  > *Gốc gác của ResNet-18 là phân loại 1000 lớp của ImageNet với tầng `model.fc` kích thước `Linear(in_features=512, out_features=1000)`. Nhóm em tiến hành phẫu thuật thay thế tầng này bằng `nn.Linear(512, 10)` để khớp với 10 lớp của CIFAR-10.*
  >
  > *Tiếp theo, về chiến lược huấn luyện, nhóm em áp dụng cơ chế **Fine-tuning có chọn lọc**:*
  > - *Đóng băng (Freeze) các tầng nông (`conv1`, `bn1`, `layer1`, `layer2`): Thiết lập `param.requires_grad = False` cho 33.4% tham số (tương đương 3.9 triệu trọng số). Lý do là các tầng đầu trích xuất đặc trưng hình học phổ quát như đường viền, góc cạnh, vân sọc (Gabor-like filters) hoàn toàn có thể tái sử dụng trực tiếp.*
  > - *Mở khóa (Unfreeze) các tầng sâu (`layer3`, `layer4` và `fc`): Cho phép cập nhật trọng số với tốc độ học nhỏ ($\eta = 10^{-4}$), giúp các tầng trừu tượng cấp cao thích nghi với hình thái đặc thù của 10 lớp CIFAR-10.*
  >
  > *Chiến lược này vừa tiết kiệm 33.4% chi phí tính đạo hàm ngược, vừa chống overfit triệt để khi dữ liệu bài toán đích nhỏ hơn nhiều so với tập huấn luyện gốc."*

---

### Phần 4: Module 6, 7 — Engine Huấn Luyện & Lưu Checkpoint An Toàn (2 Phút)
- **Tài liệu chiếu**: [02_transfer_learning_and_fine_tuning.ipynb](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/labs/lab2/vibe/02_transfer_learning_and_fine_tuning.ipynb) (Cell Step 4 & Step 5).
- **Lời thuyết trình**:
  > *"Về động cơ huấn luyện, nhóm sử dụng hàm mất mát `CrossEntropyLoss` kết hợp với thuật toán tối ưu `Adam` tại learning rate $1\times 10^{-4}$. Bộ lọc tối ưu chỉ nhận các tham số có `requires_grad=True`:*
  >
  > *`optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)`*
  >
  > *Trong mỗi epoch, quy trình Autograd 5 bước được tuân thủ nghiêm ngặt:*
  > 1. `optimizer.zero_grad()` xóa gradient cũ.
  > 2. `outputs = model(inputs)` tính toán forward pass.
  > 3. `loss = criterion(outputs, labels)` đo lường độ lệch Cross-Entropy.
  > 4. `loss.backward()` tính đạo hàm ngược tự động qua đồ thị động DAG.
  > 5. `optimizer.step()` cập nhật trọng số theo hướng dốc cực tiểu.
  >
  > *Ở khâu quản lý checkpoint, nhóm em tuân thủ tiêu chuẩn an toàn PyTorch 2.6+ mới nhất: luôn đính kèm cờ `weights_only=True` khi thực hiện `torch.load()`. Đồng thời, notebook được thiết kế theo nguyên lý **Cell Independence** — tế bào đánh giá có thể nạp trực tiếp file `resnet18_transfer.pt` đã lưu để kiểm thử ngay lập tức mà không buộc người chấm phải đợi huấn luyện lại 5 epochs."*

---

### Phần 5: Module 8 — Nghiệm Thu Kết Quả Thực Nghiệm Chuẩn 5W1H (2 Phút)
- **Tài liệu chiếu**: [docs/lab2/README.md](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/docs/lab2/README.md) (Mục 2: Bảng Đối Chuẩn Thực Nghiệm 5W1H).
- **Lời thuyết trình**:
  > *"Kính thưa Thầy/Cô, đây là bảng đối chuẩn thực nghiệm định lượng chuẩn 5W1H thể hiện rõ tính ưu việt của giải pháp:*
  >
  > 1. *Mô hình **Baseline Scratch ResNet-18** (khởi tạo ngẫu nhiên, không pretrained): Sau 5 epoch chỉ đạt **60.50%** độ chính xác, hội tụ rất chậm do không gian tham số quá rộng.*
  > 2. *Mô hình **Baseline Scratch Custom CNN**: Đạt **72.40%** độ chính xác (Loss 0.7981).*
  > 3. *Và mô hình đề xuất **Pretrained ResNet-18 Transfer Learning**: Ngay tại Epoch 1 đã đạt **78.01%**, và sau 5 epoch đã bứt phá lên **86.38%** Test Accuracy với Test Loss giảm mạnh xuống **0.4022**.*
  >
  > *Điều này minh chứng mức cải thiện vượt bậc **+25.88%** so với ResNet-18 học từ đầu và **+13.98%** so với mô hình tích chập tự chế. Về tài nguyên, bộ nhớ GPU tiêu thụ chỉ 1.78GB VRAM (đáp ứng trần quy chuẩn dưới 3.5GB), thời gian thực thi chỉ 2.8 phút trên GPU RTX 3060."*

---

### Phần 6: Module 9, 10 — Ma Trận Nhầm Lẫn, Chẩn Đoán Lỗi & Live Demo (2.5 Phút)
- **Tài liệu chiếu**: [03_evaluation_comparative_benchmark_and_inference.ipynb](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/labs/lab2/vibe/03_evaluation_comparative_benchmark_and_inference.ipynb) (Cell Step 3 & Step 6).
- **Lời thuyết trình**:
  > *"Để đánh giá chất lượng mô hình một cách toàn diện thay vì chỉ nhìn vào một con số Accuracy đơn lẻ, nhóm đã trích xuất Ma trận nhầm lẫn (Confusion Matrix 10x10) và phân tích F1-score từng lớp:*
  >
  > - *Các lớp đạt hiệu năng rất cao là `Ship` (F1: 93.7%) và `Automobile` (F1: 92.7%). Nguyên nhân là tàu thủy luôn có đặc trưng nền nước biển xanh tương phản mạnh, còn ô tô có khung hình hộp và bánh xe kim loại rất rõ nét.*
  > - *Ngược lại, điểm nghẽn lớn nhất của mô hình nằm ở cặp nhầm lẫn sinh học: **`Cat` (Mèo) vs `Dog` (Chó)** với F1-score thấp nhất (Mèo ~74.2%, Chó ~76.5%). Ở độ phân giải cực thấp $32\times 32$, các đặc trưng hình thể của chó và mèo như mắt, mũi, tai, lông thú bị mờ nhạt nghiêm trọng, dẫn tới 12.3% số ảnh Mèo bị mô hình dự đoán nhầm thành Chó.*
  > - *Cặp nhầm lẫn cơ khí là `Automobile` vs `Truck` do góc nhìn mặt trước và cấu trúc bánh xe tương đồng.*
  >
  > *Sau đây, em xin phép chuyển sang phần Live Demo suy luận trực tiếp để chứng minh mô hình hoạt động ổn định trên các mẫu ảnh ngẫu nhiên."*

---

### Phần 7: Tổng Kết & Bài Học Kỹ Thuật (1 Phút)
- **Tài liệu chiếu**: [docs/lab2/README.md](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/docs/lab2/README.md) (Mục 5 & 6).
- **Lời thuyết trình**:
  > *"Tóm lại, qua bài thực hành Lab 02, nhóm em đã rút ra 3 bài học kỹ thuật cốt lõi:*
  >
  > 1. *Transfer Learning là giải pháp tối ưu hàng đầu khi giải quyết bài toán thị giác máy tính với tập dữ liệu nhỏ và tài nguyên tính toán hữu hạn.*
  > 2. *Việc đóng băng một phần mạng kết hợp fine-tuning tầng sâu vừa giúp tiết kiệm chi phí huấn luyện, vừa tránh hiện tượng lãng quên tri thức (catastrophic forgetting).*
  > 3. *Chuẩn hóa phân phối đầu vào khớp với tập dữ liệu tiền huấn luyện (ImageNet Normalization) là điều kiện tiên quyết mang tính sống còn để mạng trích xuất đúng đặc trưng.*
  >
  > *Em xin chân thành cảm ơn Thầy/Cô và các bạn đã chú ý lắng nghe. Nhóm em rất mong nhận được những câu hỏi phản biện và đóng góp quý báu từ Thầy/Cô!"*

---

## 4. Kịch Bản Thao Tác Trình Diễn Trực Tiếp (Live Demo Script)

Khi Giảng viên yêu cầu: *"Em hãy chạy thử mô hình kiểm tra ảnh bất kỳ ngay trên màn hình xem độ tin cậy thế nào?"*, bạn thực hiện theo 2 phương án sau:

### Phương án A: Trình diễn trực quan trên Notebook (Khuyên dùng)
1. Mở file [03_evaluation_comparative_benchmark_and_inference.ipynb](file:///home/notworle/nonvintage/gitvanhub/Deeplearning-UTH/labs/lab2/vibe/03_evaluation_comparative_benchmark_and_inference.ipynb).
2. Cuộn chuột đến **Step 6: Interactive Inference & Top-3 Probabilities**.
3. Bấm **Run Cell** (hoặc `Shift + Enter`).
4. **Màn hình hiển thị**:
   - Ảnh CIFAR-10 được hiển thị trực quan (đã un-normalize về dải $[0, 1]$).
   - Nhãn thực tế (Ground Truth) và nhãn dự đoán (Predicted Label).
   - Biểu đồ thanh nằm ngang (horizontal bar chart) thể hiện **Top-3 Xác Suất** qua hàm Softmax.
5. **Thuyết minh**:
   > *"Như Thầy/Cô thấy trên màn hình, với mẫu ảnh kiểm thử này, mô hình dự đoán chính xác là lớp 'Airplane' với độ tự tin cực cao 96.4%, bỏ xa các lớp thứ hai và thứ ba là 'Ship' (2.1%) và 'Bird' (0.8%)."*

### Phương án B: Trình diễn qua Dòng lệnh Terminal (CLI Script)
1. Mở Terminal tích hợp tại thư mục gốc dự án.
2. Thực thi lệnh:
   ```bash
   python labs/lab2/vibe/inference.py --weights labs/lab2/main/weights/resnet18_transfer.pt --samples 5
   ```
3. **Thuyết minh**:
   > *"Dạ thưa Thầy/Cô, nhóm em cũng đã đóng gói toàn bộ logic suy luận vào script CLI độc lập `inference.py`, tự động chọn GPU khả dụng, chạy dưới chế độ `@torch.inference_mode()` không lưu đồ thị tính toán, và in ra bảng xác suất Top-3 cho 5 mẫu ảnh bất kỳ trong tích tắc."*

---

## 5. Bộ Câu Hỏi & Câu Trả Lời Phản Biện với Giảng Viên (Q&A Defense)

Dưới đây là 8 câu hỏi phản biện kỹ thuật sắc bén nhất mà Giảng viên thường đặt ra cho đề tài Transfer Learning & CIFAR-10, cùng đáp án chuẩn mực:

---

### Câu hỏi 1: Tại sao ResNet giải quyết được vấn đề vanishing gradient của deep networks?
- **Trả lời**:
  > *"Thưa Thầy/Cô, trong các mạng nơ-ron tích chập truyền thống, luồng thông tin truyền qua chuỗi hàm phi tuyến $x_{l+1} = \sigma(W_l x_l)$. Khi lan truyền ngược (backpropagation) qua hàng chục tầng, đạo hàm là tích của các ma trận trọng số $\prod W_l$; nếu các giá trị này nhỏ hơn 1, gradient sẽ tiêu biến dần về 0 khi về tới các tầng đầu.
  >
  > ResNet đưa vào cơ chế **Residual Learning** với kết nối tắt (skip connection): $x_{l+1} = \mathcal{F}(x_l, \mathcal{W}_l) + x_l$. Khi lấy đạo hàm hàm mất mát $\mathcal{E}$ theo $x_l$, ta có công thức:
  > $$\frac{\partial \mathcal{E}}{\partial x_l} = \frac{\partial \mathcal{E}}{\partial x_{l+1}} \left( \frac{\partial \mathcal{F}}{\partial x_l} + \mathbf{I} \right) = \frac{\partial \mathcal{E}}{\partial x_{l+1}} \frac{\partial \mathcal{F}}{\partial x_l} + \frac{\partial \mathcal{E}}{\partial x_{l+1}}$$
  > Nhờ sự xuất hiện của ma trận đơn vị $\mathbf{I}$, tín hiệu gradient $\frac{\partial \mathcal{E}}{\partial x_{l+1}}$ luôn có một đường dẫn trực tiếp (gradient highway) quay trở về các tầng nông mà không bị suy giảm theo chiều sâu mạng."*

---

### Câu hỏi 2: Tại sao CIFAR-10 lại dùng chuẩn hóa Z-score theo ImageNet thay vì mean/std của chính CIFAR-10?
- **Trả lời**:
  > *"Thưa Thầy/Cô, khi áp dụng Transfer Learning với trọng số tiền huấn luyện từ ImageNet, các bộ lọc tích chập (kernels) ở các tầng đầu đã được tối ưu hóa để nhận diện các hoa văn thị giác dựa trên phân phối dữ liệu màu sắc của ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  >
  > Nếu ta chuẩn hóa ảnh CIFAR-10 bằng mean và std của riêng CIFAR-10, các giá trị kích hoạt (activations) đầu vào sẽ bị lệch phân phối (covariate shift), khiến các bộ lọc trích xuất đặc trưng bị lệch và không phát huy được các biểu diễn đặc trưng đã học từ ImageNet."*

---

### Câu hỏi 3: Phân biệt Feature Extraction (đóng băng) và Fine-tuning (tinh chỉnh)? Khi nào áp dụng chiến lược nào?
- **Trả lời**:
  > *"Thưa Thầy/Cô:
  > - **Feature Extraction**: Đóng băng toàn bộ mạng trích xuất đặc trưng (backbone) bằng cách đặt `requires_grad = False`, chỉ khởi tạo ngẫu nhiên và huấn luyện duy nhất tầng phân loại mới (`fc`). Chiến lược này áp dụng khi tập dữ liệu bài toán đích rất nhỏ và có độ tương đồng hình ảnh cao với tập huấn luyện gốc.
  > - **Fine-tuning**: Sau khi thay thế đầu phân loại, ta mở khóa (unfreeze) một phần hoặc toàn bộ các tầng của backbone để cập nhật trọng số với tốc độ học rất nhỏ ($lr \sim 10^{-4}$ đến $10^{-5}$). Chiến lược này phù hợp khi ta có lượng dữ liệu vừa phải (như 50.000 ảnh CIFAR-10) và muốn các tầng trừu tượng cấp cao thích ứng với hình thái đặc thù của miền dữ liệu mới. Trong bài lab này, nhóm chọn Fine-tuning mở khóa `layer3`, `layer4` và `fc`."*

---

### Câu hỏi 4: Tại sao VGG-16 có nhiều tham số gấp 12 lần ResNet-18 (138M vs 11.7M) nhưng độ chính xác trên CIFAR-10 lại không cao hơn ResNet?
- **Trả lời**:
  > *"Thưa Thầy/Cô, nguyên nhân chủ yếu nằm ở sự phân bổ tham số và cấu trúc kiến trúc:
  > 1. Trong VGG-16, hơn 100 triệu tham số (chiếm hơn 80%) tập trung ở 3 tầng kết nối đầy đủ (Fully Connected layers: `Linear(25088, 4096)` $\to$ `Linear(4096, 4096)` $\to$ `Linear(4096, 1000)`). Các tầng FC này rất nặng nề, dễ bị overfit và không hề bổ sung thêm năng lực biểu diễn không gian nào cho phần trích xuất đặc trưng.
  > 2. Ngược lại, ResNet-18 sử dụng **Global Average Pooling (GAP)** trước tầng phân loại, gom ma trận $7\times 7\times 512$ thành vector 512 phần tử, loại bỏ hoàn toàn các tầng FC khổng lồ.
  > 3. Ngoài ra, cơ chế Skip Connections của ResNet giúp việc tối ưu hóa trong không gian tham số hiệu quả hơn nhiều so với kiến trúc phẳng thuần túy của VGG-16."*

---

### Câu hỏi 5: DenseNet tái sử dụng đặc trưng bằng cách nào và ưu thế tham số của nó so với ResNet là gì?
- **Trả lời**:
  > *"Thưa Thầy/Cô, khác với ResNet kết hợp thông tin bằng phép cộng nguyên tố (element-wise addition $x_l + \mathcal{F}(x_l)$), DenseNet sử dụng phép **ghép kênh (concatenation)**:
  > $$x_l = H_l([x_0, x_1, x_2, \dots, x_{l-1}])$$
  > Mỗi tầng trong Dense Block nhận bản đồ đặc trưng của tất cả các tầng đi trước nó làm đầu vào. Nhờ đó, DenseNet đạt được 2 lợi thế vượt trội:
  > 1. **Tái sử dụng đặc trưng tối đa (Feature Reuse)**: Các tầng sau không cần học lại các đặc trưng cơ bản mà các tầng trước đã tạo ra.
  > 2. **Số lượng tham số rất nhỏ**: Mỗi tầng chỉ cần tạo ra một lượng nhỏ kênh mới (gọi là tốc độ tăng trưởng - *Growth Rate $k$*, thường chỉ là $k=32$), giúp tổng tham số của DenseNet-121 chỉ có 8.0 triệu, ít hơn đáng kể so với ResNet."*

---

### Câu hỏi 6: Tại sao Cat vs Dog lại có F1-score thấp nhất (~74%) trong toàn bộ 10 lớp CIFAR-10?
- **Trả lời**:
  > *"Thưa Thầy/Cô, đây là hiện tượng nhầm lẫn đặc trưng sinh học kinh điển do giới hạn không gian dữ liệu:
  > 1. Ảnh CIFAR-10 có kích thước cực nhỏ: chỉ $32\times 32$ pixel. Ở kích thước này, toàn bộ vùng khuôn mặt của động vật chỉ chiếm khoảng $10\times 10$ pixel.
  > 2. Mèo và Chó có độ tương đồng hình thái học rất cao: đều là thú có lông, 4 chân, mắt tròn, mũi tam giác và tai vểnh, tư thế đứng/ngồi tương tự nhau.
  > 3. Khi độ phân giải bị nén xuống $32\times 32$, các đặc trưng vi mô mang tính phân định (như hình thái con ngươi mắt mèo, dáng mõm chó, kết cấu râu) hoàn toàn biến mất do hiệu ứng răng cưa (aliasing). Mô hình buộc phải dựa vào hình dáng tổng thể và màu lông, dẫn tới tỷ lệ nhầm lẫn cao nhất trong ma trận."*

---

### Câu hỏi 7: Data Augmentation (RandomCrop, RandomHorizontalFlip) đóng vai trò gì trong việc chống overfit khi fine-tune?
- **Trả lời**:
  > *"Thưa Thầy/Cô:
  > - `RandomCrop(32, padding=4)` thêm 4 pixel đệm xung quanh ảnh rồi cắt ngẫu nhiên một vùng $32\times 32$. Kỹ thuật này giả lập hiện tượng đối tượng không nằm chính giữa khung hình, rèn luyện cho mạng tính chất **Bất biến dịch chuyển (Translation Invariance)**.
  > - `RandomHorizontalFlip(p=0.5)` lật ảnh theo trục dọc với xác suất 50%, giả lập góc nhìn phản chiếu đối xứng, giúp mô hình nhận diện đối tượng dù quay sang trái hay sang phải.
  > Hai phép biến đổi này giúp nhân rộng không gian dữ liệu huấn luyện một cách nhân tạo, ngăn chặn mạng nơ-ron học vẹt (memorization) vị trí pixel tuyệt đối, cải thiện đáng kể khả năng tổng quát hóa trên tập kiểm thử."*

---

### Câu hỏi 8: Tại sao PyTorch 2.6+ cảnh báo bảo mật với `torch.load` và ta phải thiết lập `weights_only=True`?
- **Trả lời**:
  > *"Thưa Thầy/Cô, mặc định hàm `torch.load()` sử dụng thư viện `pickle` của Python để giải tuần tự hóa (unpickle) dữ liệu từ đĩa. Cơ chế unpickle cho phép thực thi mã lệnh tùy ý (arbitrary code execution) nếu file `.pt` bị kẻ xấu chèn thêm mã độc trong các phương thức `__reduce__`.
  >
  > Kể từ PyTorch 2.6+, PyTorch chính thức khuyến cáo và chuyển hướng an toàn: bắt buộc lập trình viên phải truyền tường minh cờ `weights_only=True`. Khi bật cờ này, bộ nạp chỉ giải nén các cấu trúc dữ liệu thuần túy (Tensors, Dictionaries, Primitive types) chứa trọng số mô hình và từ chối nạp các đối tượng Python phức tạp, loại bỏ hoàn toàn nguy cơ bị khai thác lỗ hổng bảo mật khi nạp weights từ nguồn mở bên ngoài."*

