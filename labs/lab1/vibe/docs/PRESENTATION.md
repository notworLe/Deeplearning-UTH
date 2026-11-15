# PRESENTATION.md — Kịch Bản Thuyết Trình Toàn Diện Module PyTorch Pipeline Lab 01

- **Motivation/Background**: Trình bày đề tài thị giác máy tính nền tảng (Fashion-MNIST Classification) đòi hỏi người thuyết trình phải kết nối liền mạch giữa các khái niệm cơ bản (Tensors, Transforms, Autograd) với kiến trúc nơ-ron thực tế (`FashionCNN`), kết quả thực nghiệm chuẩn 5W1H và thao tác live demo mượt mà trước Giảng viên.
- **Purpose**: Cung cấp tài liệu chỉ dẫn thuyết trình toàn diện cho Lab 01: định vị chính xác vị trí từng file mã nguồn trong dự án, vai trò tương ứng trong pipeline, kịch bản lời nói chi tiết (Presentation Script) cho từng module và bộ câu hỏi phản biện (Q&A Defense Guide).
- **Overview Pipeline**: Ánh xạ 10 module kỹ thuật từ [docs/lab1/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md) và kết quả thực nghiệm từ [docs/lab1/README.md](README.md) vào kịch bản thuyết trình chuẩn mực thời lượng 10–15 phút.
- **Detailed Plan**: §1 Sơ đồ Bản đồ Phân bổ File; §2 Bảng Ánh xạ Module - File - Hành động Trình chiếu; §3 Kịch bản Lời Thuyết trình Chi tiết (Từng Slide/Module); §4 Kịch bản Thao tác Live Demo Trực tiếp; §5 Bộ Câu hỏi & Câu trả lời Phản biện với Giảng viên (Q&A Defense).
- **References**: [docs/lab1/PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md), [docs/lab1/README.md](README.md), [labs/lab1/QUESTION.md](../../labs/lab1/QUESTION.md), [labs/lab1/vibe/README.md](../../labs/lab1/vibe/README.md), [agents/rules/RESULTS_REPORTING.md](../../.agents/rules/RESULTS_REPORTING.md).
- **Created**: 2026-11-14T08:48:00+07:00
- **Last Updated**: 2026-11-14T08:48:00+07:00

---

## Table of Contents

- [1. Sơ đồ Bản đồ Phân bổ File &amp; Vai trò Thuyết trình](#1-sơ-đồ-bản-đồ-phân-bổ-file--vai-trò-thuyết-trình)
- [2. Bảng Ánh xạ Module - File Mã Nguồn - Hành Động Trình Chiếu](#2-bảng-ánh-xạ-module---file-mã-nguồn---hành-động-trình-chiếu)
- [3. Kịch Bản Lời Thuyết Trình Chi Tiết (Script Từng Slide/Module)](#3-kịch-bản-lời-thuyết-trình-chi-tiết-script-từng-slidemodule)
  - [Phần 0: Lời Mở Đầu &amp; Đặt Vấn Đề (1 Phút)](#phần-0-lời-mở-đầu--đặt-vấn-đề-1-phút)
  - [Phần 1: Giới Thiệu Cấu Trúc Dự Án &amp; Khung Kiến Trúc (1.5 Phút)](#phần-1-giới-thiệu-cấu-trúc-dự-án--khung-kiến-trúc-15-phút)
  - [Phần 2: Module 1, 2, 3 — Dữ Liệu Fashion-MNIST &amp; Khảo Sát Hình Học Pixel (2 Phút)](#phần-2-module-1-2-3--dữ-liệu-fashion-mnist--khảo-sát-hình-học-pixel-2-phút)
  - [Phần 3: Module 4, 5 — Torchvision Transforms &amp; PyTorch DataLoader (2 Phút)](#phần-3-module-4-5--torchvision-transforms--pytorch-dataloader-2-phút)
  - [Phần 4: Module 6, 7 — So Sánh Kiến Trúc MLP vs FashionCNN &amp; Autograd Engine (2.5 Phút)](#phần-4-module-6-7--so-sánh-kiến-trúc-mlp-vs-fashioncnn--autograd-engine-25-phút)
  - [Phần 5: Module 8 — Nghiệm Thu Kết Quả Thực Nghiệm Chuẩn 5W1H (2 Phút)](#phần-5-module-8--nghiệm-thu-kết-quả-thực-nghiệm-chuẩn-5w1h-2-phút)
  - [Phần 6: Module 9, 10 — Chẩn Đoán Lỗi Không Gian &amp; Trình Diễn Live Demo (2.5 Phút)](#phần-6-module-9-10--chẩn-đoán-lỗi-không-gian--trình-diễn-live-demo-25-phút)
  - [Phần 7: Tổng Kết &amp; Bài Học Kỹ Thuật (1 Phút)](#phần-7-tổng-kết--bài-học-kỹ-thuật-1-phút)
- [4. Kịch Bản Thao Tác Trình Diễn Trực Tiếp (Live Demo Script)](#4-kịch-bản-thao-tác-trình-diễn-trực-tiếp-live-demo-script)
- [5. Bộ Câu Hỏi &amp; Câu Trả Lời Phản Biện với Giảng Viên (Q&amp;A Defense)](#5-bộ-câu-hỏi--câu-trả-lời-phản-biện-với-giảng-viên-qa-defense)

---

## 1. Sơ đồ Bản đồ Phân bổ File & Vai trò Thuyết trình

Dưới đây là sơ đồ chỉ rõ file nào nằm ở đâu và bạn cần mở file nào trong từng giai đoạn báo cáo:

```text
Deeplearning-UTH/
│
├── docs/lab1/                                    <-- [NƠI CHIẾU BÁO CÁO TOÁN HỌC & LÝ THUYẾT]
│   ├── README.md                                 # Tổng quan bài toán, 5W1H benchmark table, VRAM governance
│   ├── PIPELINE_EXPLANATION.md                   # Bóc tách 10 Module toán học (Input, Output, Đặc trưng, Mục tiêu, Hạn chế, Giải pháp)
│   └── PRESENTATION.md                           # File kịch bản thuyết trình này
│
├── labs/lab1/
│   ├── QUESTION.md                               # Đề bài gốc 5 yêu cầu cốt lõi
│   ├── README.md                                 # Hướng dẫn tổng quan bài Lab 1
│   │
│   ├── vibe/                                     <-- [NƠI MỞ CODE VÀ THỰC HIỆN LIVE DEMO]
│   │   ├── README.md                             # Tổng quan không gian bài làm
│   │   ├── 01_data_exploration_and_transforms.ipynb # [Demo 1]: Tensors, Transforms, DataLoader & Lưới 16 ảnh EDA
│   │   ├── 02_model_building_and_training.ipynb  # [Demo 2]: So sánh MLP vs FashionCNN, Autograd 5 bước, Loss curves
│   │   ├── 03_evaluation_and_error_analysis.ipynb# [Demo 3]: Confusion Matrix, Lưới 25 ảnh xanh/đỏ, Top lỗi sai, Single/Batch Inference
│   │   ├── train.py                              # Script CLI huấn luyện độc lập
│   │   └── inference.py                          # Script CLI suy luận ảnh
│   │
│   └── main/                                     <-- [BẢN CODE GỐC ĐÃ TRAIN & TRỌNG SỐ]
│       ├── weights/fashion-normal.pt             # Trọng số tốt nhất đã huấn luyện (FashionCNN: 91.24%)
│       ├── result/fashion-normal/                # Thư mục chứa các ảnh biểu đồ và confusion matrix đã xuất
│       └── inference/app.py                      # Ứng dụng Web Gradio phục vụ người dùng cuối
```

---

## 2. Bảng Ánh xạ Module - File Mã Nguồn - Hành Động Trình Chiếu

| Module trong Pipeline | File tương ứng cần mở | Vị trí / Đoạn code cần trỏ vào | Mục đích trình chiếu |
| :--- | :--- | :--- | :--- |
| **Tổng quan Đề tài** | [`labs/lab1/QUESTION.md`](../../labs/lab1/QUESTION.md) | Yêu cầu 1, 2, 3 | Chỉ ra đề bài yêu cầu làm gì. |
| **Kiến trúc Pipeline** | [`docs/lab1/PIPELINE_EXPLANATION.md`](PIPELINE_EXPLANATION.md) | Mục 1 (Sơ đồ Mermaid) | Chiếu toàn cảnh 5 tầng xử lý của Computer Vision Pipeline. |
| **M1: Problem & Metric** | [`docs/lab1/PIPELINE_EXPLANATION.md`](PIPELINE_EXPLANATION.md) | Mục 2 | Trình bày bài toán phân loại ảnh $28\times 28$ vào 10 lớp thời trang. |
| **M2, M3: Ingestion & EDA** | [`01_data_exploration_and_transforms.ipynb`](../../labs/lab1/vibe/01_data_exploration_and_transforms.ipynb) | Cell 4 & 6 | Chiếu biểu đồ 10 lớp cân bằng và lưới 16 ảnh mẫu có nhãn. |
| **M4: Transforms** | [`01_data_exploration_and_transforms.ipynb`](../../labs/lab1/vibe/01_data_exploration_and_transforms.ipynb) | Cell 3 (`transforms.Compose`) | Giải thích chuẩn hóa Z-score ($\mu=0.2860, \sigma=0.3530$) và lật ngang. |
| **M5: DataLoader** | [`01_data_exploration_and_transforms.ipynb`](../../labs/lab1/vibe/01_data_exploration_and_transforms.ipynb) | Cell 5 (`DataLoader`) | Giải thích `batch_size=64`, `shuffle=True`, và `pin_memory=True`. |
| **M6: Model Architecture** | [`02_model_building_and_training.ipynb`](../../labs/lab1/vibe/02_model_building_and_training.ipynb) | Cell 2 & 3 | So sánh Baseline MLP (235K params) và `FashionCNN` (150K params). |
| **M7: Autograd & Train** | [`02_model_building_and_training.ipynb`](../../labs/lab1/vibe/02_model_building_and_training.ipynb) | Cell 5 (Hàm `train_epoch`) | Chỉ rõ 5 bước cập nhật gradient kinh điển của PyTorch. |
| **M8: Đánh giá 5W1H** | [`docs/lab1/README.md`](README.md) hoặc Notebook 3 | Cell 2 (Classification Report) | Công bố số liệu chuẩn mực: FashionCNN đạt **91.24%** vs MLP đạt **88.42%**. |
| **M9: Chẩn đoán Lỗi** | [`03_evaluation_and_error_analysis.ipynb`](../../labs/lab1/vibe/03_evaluation_and_error_analysis.ipynb) | Cell 3 & 4 (Heatmap & Lưới xanh/đỏ) | Chỉ ra lỗi nhầm giữa `Shirt` vs `T-shirt`, giải thích mất thông tin không gian. |
| **M10: Live Serving Demo** | [`03_evaluation_and_error_analysis.ipynb`](../../labs/lab1/vibe/03_evaluation_and_error_analysis.ipynb) | Cell 6 (Dự đoán ảnh ngẫu nhiên) | Chạy thử nghiệm hàm `predict_sample()` ngay trước mắt Giảng viên. |

---

## 3. Kịch Bản Lời Thuyết Trình Chi Tiết (Script Từng Slide/Module)

### Phần 0: Lời Mở Đầu & Đặt Vấn Đề (1 Phút)
> *"Kính thưa Thầy/Cô và các bạn trong Hội đồng,*
> 
> *Hôm nay, em xin phép đại diện nhóm báo cáo kết quả thực hiện **Lab 01: Xây Dựng Pipeline Học Sâu Phân Loại Ảnh Thời Trang Fashion-MNIST với PyTorch**.*
> 
> *Trong thị giác máy tính, bộ dữ liệu số viết tay MNIST từng là chuẩn mực vàng, nhưng hiện nay đã trở nên quá đơn giản khi một mạng nơ-ron cơ bản cũng dễ dàng đạt trên 99% độ chính xác. Để đánh giá thực chất năng lực của các mô hình học sâu, Zalando Research đã công bố **Fashion-MNIST** — một bộ dữ liệu có cùng kích thước ảnh xám $28\times 28$ nhưng độ phức tạp thị giác cao hơn rất nhiều.*
> 
> *Mục tiêu của nhóm là làm chủ toàn diện các thành phần cốt lõi của PyTorch: từ xử lý Tensor, chuẩn hóa Transforms, nạp lô đa luồng qua DataLoader, thiết kế và so sánh giữa mạng MLP và mạng tích chập Custom CNN (`FashionCNN`), cho đến vòng lặp Autograd, tối ưu Adam và chẩn đoán ma trận nhầm lẫn. Sau đây, em xin phép đi vào chi tiết cấu trúc dự án và 10 module kỹ thuật."*

---

### Phần 1: Giới Thiệu Cấu Trúc Dự Án & Khung Kiến Trúc (1.5 Phút)
*(Hành động: Mở file [`docs/lab1/PIPELINE_EXPLANATION.md`](PIPELINE_EXPLANATION.md) tại Mục 1 — Sơ đồ Mermaid).*

> *"Thưa Thầy/Cô, toàn bộ mã nguồn của nhóm được tổ chức theo chuẩn kiến trúc Clean Architecture:*
> - *Thư mục `docs/lab1/` lưu trữ tài liệu đặc tả toán học chi tiết của 10 module độc lập theo 6 trục chuẩn mực: **Input, Output, Đặc trưng, Mục tiêu, Hạn chế, và Giải pháp**.*
> - *Thư mục `labs/lab1/vibe/` là không gian làm việc hoàn chỉnh gồm bộ 3 Jupyter Notebooks được phân chia độc lập, kèm các script dòng lệnh `train.py` và `inference.py`.*
> - *Thư mục `labs/lab1/main/weights/` lưu trữ trọng số checkpoint tối ưu nhất `fashion-normal.pt` đã được huấn luyện sẵn.*
> 
> *Như Thầy/Cô có thể quan sát trên sơ đồ Mermaid, luồng dữ liệu của nhóm gồm 5 tầng liên hoàn: **Tầng Dữ liệu** $\to$ **Tầng Tiền xử lý & Nạp lô** $\to$ **Tầng Kiến trúc Mô hình** $\to$ **Tầng Động cơ Huấn luyện** $\to$ và **Tầng Đánh giá & Phục vụ**."*

---

### Phần 2: Module 1, 2, 3 — Dữ Liệu Fashion-MNIST & Khảo Sát Hình Học Pixel (2 Phút)
*(Hành động: Mở file [`labs/lab1/vibe/01_data_exploration_and_transforms.ipynb`](../../labs/lab1/vibe/01_data_exploration_and_transforms.ipynb), cuộn đến Cell 4 và Cell 6).*

> *"Đi vào **Module 1, 2 và 3**: Bài toán và Khảo sát Dữ liệu EDA.*
> 
> *Bài toán đặt ra là phân loại ảnh đơn kênh $28 \times 28$ vào 10 danh mục trang phục thời trang. Dữ liệu gồm 60,000 ảnh train và 10,000 ảnh test.*
> *Tại Cell 4 của Notebook 1, biểu đồ cột cho thấy bộ dữ liệu có tính chất **cân bằng hoàn hảo**: đúng chính xác 6,000 ảnh cho mỗi lớp ở tập train và 1,000 ảnh mỗi lớp ở tập test. Do đó, hiện tượng thiên lệch tần suất cơ sở (Base Rate Bias) hoàn toàn bị triệt tiêu, cho phép ta sử dụng trực tiếp Accuracy và Macro-F1 làm thước đo chuẩn xác.*
> 
> *Về mặt hình học pixel (Pixel Geometry): Biểu đồ histogram cho thấy phân phối có tính chất hai đỉnh rõ rệt (Bimodal Distribution): hơn 50% diện tích ảnh là nền đen thuần túy có giá trị 0, và dải từ 50 đến 255 biểu thị chi tiết trang phục. Giá trị trung bình toàn tập là $\mu \approx 0.2860$ và độ lệch chuẩn là $\sigma \approx 0.3530$. Bộ thông số thống kê này được nhóm ghi nhận trực tiếp để làm cơ sở cho bước chuẩn hóa Z-Score."*

---

### Phần 3: Module 4, 5 — Torchvision Transforms & PyTorch DataLoader (2 Phút)
*(Hành động: Chỉ vào Cell 3 và Cell 5 của Notebook 1).*

> *"Tiếp theo là **Module 4 và 5: Biến đổi Dữ liệu và Bộ Nạp Lô DataLoader**.*
> 
> *Nếu đưa trực tiếp pixel từ 0 đến 255 vào mạng nơ-ron, dải giá trị lớn sẽ gây hiện tượng bùng nổ đạo hàm (Gradient Explosion). Nhóm áp dụng chuỗi biến đổi chuẩn mực:*
> 1. *`transforms.ToTensor()`: Ép kiểu sang Float Tensor, scale tuyến tính về khoảng $[0.0, 1.0]$, và hoán đổi trục kênh thành chuẩn PyTorch $(C, H, W) = (1, 28, 28)$.*
> 2. *`transforms.Normalize(mean=(0.2860,), std=(0.3530,))`: Chuẩn hóa Z-Score đưa phân phối về kỳ vọng 0 và phương sai 1, giúp mặt cong mất mát cân đối và Gradient Descent hội tụ nhanh hơn gấp nhiều lần.*
> 3. *Data Augmentation: Nhóm sử dụng `RandomHorizontalFlip` và `RandomCrop(28, padding=2)` trên tập train để mở rộng không gian mẫu, nhưng giữ nguyên tập test chỉ có chuẩn hóa cơ bản.*
> 
> *Tại Module 5: `DataLoader` nhóm cấu hình `batch_size = 64`, `shuffle = True` để xáo trộn mẫu ngẫu nhiên sau mỗi epoch. Đặc biệt, nhóm thiết lập `pin_memory = True` nhằm cấp phát bộ nhớ RAM dạng trang cố định, giúp tăng tốc độ truyền dữ liệu qua bus PCIe sang card đồ họa GPU."*

---

### Phần 4: Module 6, 7 — So Sánh Kiến Trúc MLP vs FashionCNN & Autograd Engine (2.5 Phút)
*(Hành động: Mở file [`labs/lab1/vibe/02_model_building_and_training.ipynb`](../../labs/lab1/vibe/02_model_building_and_training.ipynb) tại Cell 2 và Cell 3).*

> *"Sang **Module 6 và 7: Thiết kế Kiến trúc và Vòng lặp Huấn luyện**.*
> 
> *Nhóm đã thiết kế và so sánh trực tiếp hai trường phái kiến trúc kinh điển:*
> - *Mô hình 1 là **Baseline MLP**: Trải phẳng ảnh thành vector 1D 784 chiều, qua 3 tầng fully-connected $784 \to 256 \to 128 \to 10$. Nhược điểm chí mạng của MLP là làm mất hoàn toàn cấu trúc không gian 2D và có tới **235,146 tham số** (phần lớn tập trung ở tầng Linear đầu tiên $784 \times 256$).*
> - *Mô hình 2 là **Deep FashionCNN**: Gồm 3 khối tích chập chuyên sâu (Conv2d $1\to 32 \to 64 \to 128$), tích hợp `BatchNorm2d` sau mỗi lớp Conv để ổn định phân phối kích hoạt, hàm phi tuyến `ReLU`, các tầng `MaxPool2d(2)`, và tầng `AdaptiveAvgPool2d(1)`. Đầu phân loại có một lớp `Dropout(0.5)` chống quá khớp. Điểm đặc sắc là `FashionCNN` chỉ có **khoảng 150,000 tham số** (nhẹ hơn MLP 36%) nhưng lại sở hữu thiên hướng quy nạp mạnh mẽ về **tính cục bộ (Locality)** và **tính bất biến tịnh tiến (Translation Invariance)**.*
> 
> *Về vòng lặp tối ưu hóa: Nhóm áp dụng hàm mất mát `nn.CrossEntropyLoss()`, kết hợp bộ tối ưu `Adam` với tốc độ học $\alpha = 0.001$. Tại Cell 5, nhóm triển khai chuẩn mực 5 bước huấn luyện PyTorch: `zero_grad()` xóa gradient cũ $\to$ `model(inputs)` tính forward $\to$ tính `loss` $\to$ `loss.backward()` kích hoạt Autograd tính đạo hàm $\to$ và `optimizer.step()` cập nhật trọng số."*

---

### Phần 5: Module 8 — Nghiệm Thu Kết Quả Thực Nghiệm Chuẩn 5W1H (2 Phút)
*(Hành động: Mở file [`labs/lab1/vibe/03_evaluation_and_error_analysis.ipynb`](../../labs/lab1/vibe/03_evaluation_and_error_analysis.ipynb) tại Cell 2 — Báo cáo phân loại và đồ thị so sánh).*

> *"Kính thưa Hội đồng, đây là phần nghiệm thu kết quả thực nghiệm theo chuẩn khoa học 5W1H:*
> 
> *Khi đánh giá độc lập trên 10,000 ảnh kiểm thử chưa từng thấy:*
> - *Mô hình **Baseline MLP (E2)** đạt độ chính xác: **88.42%** (Macro-F1 = 88.39%).*
> - *Mô hình **FashionCNN** đạt độ chính xác vượt bậc: **91.24%** (Macro-F1 = 91.18%), với Test Loss chỉ còn **0.2450**.*
> 
> *Nhìn vào bảng hiệu năng chi tiết từng lớp của `FashionCNN`:*
> - *Các lớp đạt hiệu năng gần như tuyệt đối: `Trouser` đạt **F1 = 98.3%**, `Bag` đạt **F1 = 97.7%**, và `Sandal` đạt **F1 = 97.0%**.*
> - *Lớp thử thách nhất là `Shirt` (áo sơ mi), chỉ đạt F1 là **75.7%**.*
> 
> *Kết quả thực nghiệm này khẳng định mạnh mẽ tính ưu việt của mạng tích chập so với mạng kết nối đầy đủ truyền thống trong xử lý dữ liệu không gian ảnh."*

---

### Phần 6: Module 9, 10 — Chẩn Đoán Lỗi Không Gian & Trình Diễn Live Demo (2.5 Phút)
*(Hành động: Cuộn xuống Cell 3 và Cell 4 của Notebook 3 — hiển thị Confusion Matrix và lưới 25 ảnh xanh/đỏ).*

> *"Để hiểu rõ lý do tại sao lớp `Shirt` lại có F1 thấp hơn các lớp khác, nhóm đã thực hiện **Module 9: Chẩn đoán Lỗi Không gian (Error Diagnostics)**.*
> 
> *Quan sát Ma trận Nhầm lẫn (Confusion Matrix) tại Cell 3:*
> - *Mô hình nhầm nhiều nhất ở cặp **`Shirt (6)` và `T-shirt/top (0)`**: có tới 85 ảnh áo sơ mi bị đoán nhầm thành áo thun, và ngược lại.*
> - *Cặp nhầm lẫn thứ hai là **`Coat (4)` và `Pullover (2)`**: áo khoác kéo khóa kín có phom dáng rất giống áo len chui đầu.*
> 
> *Nguyên nhân kỹ thuật nằm ở giới hạn độ phân giải $28 \times 28$ pixel: Các chi tiết phân biệt tinh tế như hàng cúc áo và ve cổ áo sơ mi chỉ chiếm 2–3 pixel và bị nhòe sau phép pooling không gian. Tuy nhiên, ở các đặc trưng vĩ mô như ống quần dài hay quai xách của túi, mô hình nhận diện chính xác gần như 100%.*
> 
> *Tại Cell 4, lưới 25 ảnh trực quan hóa hiển thị rõ ràng: các mẫu dự đoán đúng được gán nhãn màu xanh lá, và các mẫu dự đoán sai được tô nhãn màu đỏ kèm giá trị thực tế và độ tự tin. Điều này chứng minh toàn bộ quy trình đánh giá là hoàn toàn minh bạch và kiểm chứng được."*

---

### Phần 7: Tổng Kết & Bài Học Kỹ Thuật (1 Phút)
> *"Tóm lại, qua bài thực hành Lab 01, nhóm đã hoàn thành xuất sắc toàn bộ 5 yêu cầu đề bài:*
> 1. *Làm chủ kỹ thuật PyTorch Tensors, Transforms, Dataset & DataLoader.*
> 2. *Thiết kế kiến trúc mạng nơ-ron tích chập `FashionCNN` với cơ chế điều chuẩn BatchNorm và Dropout, đạt độ chính xác 91.24%.*
> 3. *Kiểm soát tài nguyên xuất sắc: toàn bộ quá trình train chỉ chiếm chưa đầy **0.8GB VRAM**, an toàn tuyệt đối dưới ngưỡng trần 3.5GB.*
> 4. *Tuần tự hóa checkpoint an toàn với `weights_only=True` và xây dựng ứng dụng web suy luận Gradio.*
> 
> *Em xin chân thành cảm ơn Thầy/Cô và các bạn đã chú ý theo dõi. Nhóm em rất mong nhận được những câu hỏi phản biện từ Hội đồng ạ!"*

---

## 4. Kịch Bản Thao Tác Trình Diễn Trực Tiếp (Live Demo Script)

Nếu Giảng viên yêu cầu chạy thử nghiệm trực tiếp tại chỗ, bạn hãy thực hiện tuần tự:

### Thao tác 1: Demo Trực quan qua Notebook 3 (Dự đoán ảnh ngẫu nhiên)
1. Mở file [`labs/lab1/vibe/03_evaluation_and_error_analysis.ipynb`](../../labs/lab1/vibe/03_evaluation_and_error_analysis.ipynb).
2. Cuộn đến Cell 6 (Mục *Interactive Prediction Demo*).
3. Nhấn `Shift + Enter`.
* **Lời thuyết minh**:
  > *"Thưa Thầy/Cô, Cell này lấy ngẫu nhiên một ảnh từ tập test, truyền qua hàm `predict_sample()`. Mô hình trả về ảnh gốc, nhãn thực tế, nhãn dự đoán và biểu đồ xác suất Top-3 lớp. Như Thầy/Cô thấy, với ảnh chiếc giày Sneaker này, mô hình dự đoán đúng với độ tin cậy tuyệt đối 99.8%."*

### Thao tác 2: Demo Giao diện Web Gradio (Nếu có yêu cầu)
Mở terminal và gõ:
```bash
cd labs/lab1/main
python inference/app.py
```
* Mở trình duyệt tại `http://localhost:7860`, duyệt qua các ảnh áo, quần trong thư mục `inference/input/` để hiển thị xác suất phân loại Top-k tương tác trực tiếp.

---

## 5. Bộ Câu Hỏi & Câu Trả Lời Phản Biện với Giảng Viên (Q&A Defense Guide)

Dưới đây là 8 câu hỏi phản biện kinh điển của Giảng viên và câu trả lời chuẩn xác:

---

### Câu hỏi 1: *"Tại sao mô hình CNN lại đạt độ chính xác cao hơn MLP (91.2% vs 88.4%) dù CNN có ít tham số hơn (150K vs 235K)?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, sự vượt trội của CNN nằm ở hai thiên hướng quy nạp cốt lõi (Inductive Bias) mà MLP không có:
  > 1. **Tính cục bộ (Spatial Locality)**: Các kernel tích chập $3\times 3$ trích xuất đặc trưng biên, góc và họa tiết trong phạm vi lân cận của các pixel gần nhau. Ngược lại, MLP trải phẳng ảnh thành vector 784 số, phá vỡ hoàn toàn mối liên hệ không gian giữa các hàng và cột kề nhau.
  > 2. **Chia sẻ trọng số (Weight Sharing)**: Cùng một bộ lọc tích chập được trượt trên toàn bộ ảnh, mang lại **tính bất biến tịnh tiến (Translation Invariance)**: trang phục nằm lệch sang trái hay sang phải thì đặc trưng nhận diện vẫn giữ nguyên. Điều này giúp CNN học tổng quát hóa tốt hơn rất nhiều với số lượng tham số ít hơn 36%."*

---

### Câu hỏi 2: *"Hàm `optimizer.zero_grad()` có vai trò gì và điều gì sẽ xảy ra nếu ta quên gọi nó trước `loss.backward()`?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô:
  > Mặc định trong PyTorch, các gradient được tính toán bởi hàm `loss.backward()` sẽ được **cộng dồn (accumulate)** vào thuộc tính `.grad` của từng tensor tham số thay vì bị ghi đè.
  > Cơ chế này rất hữu ích khi làm Gradient Accumulation trên GPU bộ nhớ nhỏ. Nhưng trong một vòng lặp huấn luyện thông thường, nếu ta quên gọi `optimizer.zero_grad()`, gradient của batch hiện tại sẽ bị cộng dồn với gradient của tất cả các batch trước đó. Kết quả là vector gradient bị phóng đại và sai lệch phương hướng hoàn toàn, khiến quá trình cập nhật trọng số bằng `optimizer.step()` bị phân kỳ (Divergence) và loss sẽ bùng nổ thành NaN."*

---

### Câu hỏi 3: *"Tại sao nhóm lại chuẩn hóa ảnh Fashion-MNIST bằng `mean=0.2860, std=0.3530` thay vì dùng bộ số của ImageNet `mean=0.485, std=0.229`?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô:
  > 1. Bộ số của ImageNet là thống kê trên 3 kênh màu RGB của hơn 1 triệu ảnh tự nhiên (ảnh phong cảnh, đồ vật đời thực).
  > 2. Fashion-MNIST là ảnh xám đơn kênh (grayscale 1 channel) với nền đen thuần túy chiếm đa số. Thống kê thực tế qua bước EDA trên 60,000 ảnh tập train cho thấy giá trị trung bình pixel thực sự là $\approx 0.2860$ và độ lệch chuẩn là $\approx 0.3530$.
  > Việc sử dụng chính xác bộ thông số phân phối nội tại của tập train đảm bảo dữ liệu đầu vào sau phép trừ mean và chia std sẽ đạt chuẩn phân phối tắc $\mathcal{N}(0, 1)$, giúp tối ưu hóa tốt nhất cho bài toán này."*

---

### Câu hỏi 4: *"Tầng `AdaptiveAvgPool2d(output_size=1)` trong `FashionCNN` có ưu điểm gì so với việc dùng `Flatten()` trực tiếp sau các lớp Conv?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô:
  > Nếu sau lớp Conv thứ 3 ($[B, 128, 7, 7]$) mà ta dùng `Flatten()` trực tiếp, số lượng đặc trưng đầu vào cho tầng Linear tiếp theo sẽ là $128 \times 7 \times 7 = 6,272$. Khi nối vào tầng ẩn 256 nơ-ron, ma trận trọng số sẽ có kích thước $6,272 \times 256 \approx 1.6\text{ triệu}$ tham số, gây bùng nổ kích thước mô hình và rất dễ bị quá khớp (overfitting).
  > Bằng cách sử dụng `AdaptiveAvgPool2d(1)`, ta gom trung bình không gian mỗi feature map thành 1 giá trị duy nhất ($[B, 128, 1, 1]$), cắt giảm số chiều xuống đúng 128. Nhờ đó, số tham số của tầng Linear chỉ còn $128 \times 256 \approx 32,768$ tham số (giảm 98% số trọng số của tầng kết nối), vừa giúp mô hình cực kỳ nhẹ vừa chống overfit hiệu quả."*

---

### Câu hỏi 5: *"Tại sao khi đánh giá hoặc suy luận ta bắt buộc phải gọi `model.eval()` và `torch.no_grad()`?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, hai lệnh này phục vụ hai mục đích hoàn toàn khác biệt nhưng bổ trợ cho nhau:
  > 1. **`model.eval()`** chuyển đổi hành vi logic của các tầng nơ-ron: Nó tắt cơ chế Dropout ngẫu nhiên (chuyển $p=0$ để dùng toàn bộ 100% nơ-ron) và khóa cơ chế cập nhật của `BatchNorm2d` (sử dụng giá trị mean và variance tích lũy tĩnh thay vì tính trên batch hiện tại).
  > 2. **`torch.no_grad()`** (hoặc `@torch.inference_mode()`): Ra lệnh cho PyTorch vô hiệu hóa việc theo dõi lịch sử tính toán để xây dựng đồ thị đạo hàm (Computation Graph). Nhờ đó, bộ nhớ RAM/VRAM được tiết kiệm tới 50% và tốc độ tính toán suy luận tăng gấp đôi."*

---

### Câu hỏi 6: *"Tại sao lớp `Shirt (6)` lại có F1-Score thấp nhất (chỉ 75.7%) trong số 10 lớp?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, đây là điểm mù thị giác đặc trưng của bộ dữ liệu Fashion-MNIST:
  > Áo sơ mi (`Shirt`) có cấu trúc phom dáng tổng thể hình chữ T tương đồng gần như hoàn hảo với áo thun (`T-shirt/top`) và áo len (`Pullover`). Chi tiết phân biệt then chốt giữa sơ mi và áo thun là **hàng cúc áo chạy dọc ngực và ve cổ áo có nếp gấp**.
  > Tuy nhiên, ở độ phân giải siêu nhỏ $28 \times 28$ pixel và ảnh xám đơn kênh, các chi tiết cúc áo và nếp ve cổ chỉ chiếm từ 1 đến 2 pixel và bị nhòe mờ nghiêm trọng. Khi nén qua các tầng Max Pooling, thông tin biên tinh tế này bị suy giảm, dẫn đến việc mô hình nhầm lẫn tới 85 trường hợp giữa Shirt và T-shirt."*

---

### Câu hỏi 7: *"Tại sao khi lưu và tải trọng số mô hình, nhóm chỉ lưu `state_dict` thay vì lưu toàn bộ đối tượng mô hình bằng `torch.save(model)`?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, đây là quy chuẩn kỹ thuật bắt buộc của PyTorch:
  > 1. Lưu toàn bộ đối tượng `torch.save(model)` sử dụng cơ chế Pickle của Python, nó lưu kèm cả đường dẫn module mã nguồn cứng. Nếu trong tương lai ta tái cấu trúc thư mục, đổi tên file hoặc chạy trên môi trường khác, việc nạp lại sẽ bị lỗi `ModuleNotFoundError`.
  > 2. Lưu `model.state_dict()` chỉ lưu thuần túy từ điển ánh xạ giữa tên tham số và mảng Tensor số thực. Cách này hoàn toàn độc lập với cấu trúc mã nguồn, chiếm ít dung lượng lưu trữ hơn và cho phép nạp an toàn bằng cờ `weights_only=True` chống thực thi mã độc."*

---

### Câu hỏi 8: *"Quá trình chuẩn bị dữ liệu và huấn luyện của nhóm có xảy ra rò rỉ dữ liệu (Data Leakage) không?"*
* **Trả lời chuẩn**:
  > *"Dạ thưa Thầy/Cô, nhóm cam kết **100% không xảy ra rò rỉ dữ liệu**:
  > 1. Giá trị chuẩn hóa $\mu = 0.2860$ và $\sigma = 0.3530$ được tính toán thuần túy trên 60,000 ảnh của tập huấn luyện. Tập kiểm thử chỉ áp dụng lại bộ tham số này mà không hề tham gia vào bước tính toán thống kê.
  > 2. Các phép tăng cường dữ liệu ngẫu nhiên (Random Horizontal Flip, Random Crop) chỉ được áp dụng trên tập train. Tập test giữ nguyên bản gốc để đảm bảo tính khách quan tuyệt đối.
  > 3. Tập test chỉ được nạp vào ở bước `test_phase()` sau khi mô hình đã hoàn thành toàn bộ quá trình cập nhật trọng số."*

