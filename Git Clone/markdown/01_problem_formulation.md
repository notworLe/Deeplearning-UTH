# 01 — Problem Formulation: FashionMNIST Classification

> **Practice 1 — PyTorch FashionMNIST Classification**  
> **Giai đoạn:** 1/6 — Xác định vấn đề, chưa huấn luyện  
> **Từ khóa:** WHAT · WHY · Input/Output · Dataset · Mathematical Formulation · Evaluation

## 1.1. WHAT — Bài toán đang nói về điều gì?

### 1.1.1. Phát biểu bài toán (Problem Statement)

FashionMNIST Classification là bài toán **phân loại ảnh có giám sát** (*supervised image classification*). Từ một ảnh xám 28 × 28 pixel thể hiện trang phục/phụ kiện, một mô hình mạng nơ-ron học cách dự đoán ảnh đó thuộc **một trong 10 lớp**.

Một mẫu dữ liệu gồm:

- **Đầu vào (input):** 784 giá trị độ sáng của ảnh xám, được lưu ở các cột `pixel1`–`pixel784`.
- **Nhãn đúng (ground truth):** một số nguyên `label` từ 0 đến 9.
- **Đầu ra (output):** vector điểm số (*logits*) cho 10 lớp; lớp dự đoán là vị trí có logit lớn nhất.

**Câu hỏi cốt lõi:** Làm thế nào thiết kế và huấn luyện một mạng nơ-ron PyTorch để học quan hệ giữa pixel ảnh và loại trang phục, sau đó kiểm tra độ chính xác trên những ảnh chưa được dùng để cập nhật trọng số?

### 1.1.2. Các yêu cầu từ đề bài

| Yêu cầu | Ý nghĩa cần giải quyết | Minh chứng cuối cùng |
|---|---|---|
| Load FashionMNIST | Đọc dữ liệu, tạo Dataset/DataLoader | Kích thước tập, batch Tensor |
| Apply transforms | Đưa pixel về dạng/thang giá trị phù hợp | Shape và phạm vi dữ liệu |
| Build neural network | Tạo mô hình phân loại 10 lớp | Kiến trúc MLP và shape logits |
| Training loop | Forward → Loss → Backward → Optimize | Loss theo epoch |
| Evaluate accuracy | Đo khả năng phân loại | Accuracy trên test |
| Save and load model | Tái sử dụng trọng số đã train | File `.pth`, kiểm tra nạp lại |
| Experiment | Thay đổi network/hyperparameters | Bảng so sánh có kiểm soát |
| Visualize | Loss graph, predicted vs actual | Các file `.png` |
| Deliver | Code + brief report + hình minh họa | File `.py`, `.md`, bảng/ảnh |

## 1.2. WHY — Tại sao giải quyết bài toán này?

### 1.2.1. Động cơ và ý nghĩa

Ảnh có hàng trăm giá trị pixel nhưng chúng không trực tiếp cho biết tên trang phục. Bài toán đặt ra yêu cầu mô hình **học đặc trưng từ ví dụ có nhãn** thay vì sử dụng luật viết tay để nhận diện từng loại ảnh. FashionMNIST đủ nhỏ để huấn luyện và kiểm tra, nhưng vẫn có những lớp dễ nhầm, chẳng hạn Shirt/Coat/Pullover.

### 1.2.2. Vì sao chọn PyTorch?

PyTorch cho phép mô tả toàn bộ quy trình học sâu thông qua:

- **Tensor:** biểu diễn batch ảnh và nhãn.
- **Dataset/DataLoader:** đọc mẫu, chia batch, shuffle tập train.
- **Transforms/normalization:** chuyển pixel thành dạng số phù hợp.
- **`nn.Module`:** định nghĩa cấu trúc mạng.
- **Autograd (`loss.backward()`):** tự tính gradient bằng đạo hàm lan truyền ngược.
- **Optimizer (`optimizer.step()`):** cập nhật trọng số để giảm loss.
- **`state_dict`:** lưu/tải tham số mô hình.

### 1.2.3. Các câu hỏi nghiên cứu/thực hành

1. Mạng MLP có thể học phân biệt 10 loại trang phục từ pixel ảnh 28 × 28 ở mức nào?
2. Training loss và validation loss thay đổi ra sao qua các epoch?
3. Thay đổi số neuron/layer hoặc learning rate có ảnh hưởng gì đến chất lượng dự đoán?
4. Mô hình thường nhầm những nhãn nào, và giới hạn của cách tiếp cận MLP là gì?
5. Làm thế nào xác nhận mô hình tổng quát hóa tốt mà không để dữ liệu test ảnh hưởng quá trình chọn mô hình?

## 1.3. Dataset Understanding — Hiểu dữ liệu trước khi giải bài toán

### 1.3.1. Nguồn và đặc điểm

- **Tên bộ dữ liệu:** Fashion-MNIST.
- **Đơn vị phát triển:** Zalando Research.
- **Nguồn CSV:** <https://www.kaggle.com/datasets/zalando-research/fashionmnist>.
- **Tệp của dự án:** `data/fashion_mnist_full.csv` (gộp hai CSV train/test gốc, thêm cột `split`).
- **Tổng số dòng:** 70.000; **tổng số cột:** 786.
- **Ảnh:** một kênh xám, 28 × 28 = 784 pixels; pixel ban đầu trong thang 0–255.

### 1.3.2. Cấu trúc CSV đã xác minh

| Cột | Kiểu/giá trị mong đợi | Vai trò |
|---|---|---|
| `label` | Số nguyên 0–9 | Nhãn đúng của ảnh |
| `pixel1` … `pixel784` | Giá trị pixel 0–255 | Đặc trưng đầu vào |
| `split` | `train`, `validation`, `test` | Xác định tập được phép sử dụng ở từng giai đoạn |

Từ file đã kiểm tra:

| Tập | Số mẫu | Mục đích | Số mẫu mỗi lớp |
|---|---:|---|---:|
| Train | 54.000 | Cập nhật trọng số | 5.400 |
| Validation | 6.000 | Theo dõi/chọn hyperparameters và mô hình | 600 |
| Test | 10.000 | Đánh giá cuối cùng sau khi chọn mô hình | 1.000 |
| **Tổng** | **70.000** | | **7.000** |

Tập train và validation được chia có phân tầng (*stratified*) từ 60.000 ảnh train gốc; tập 10.000 ảnh test gốc được giữ lại. Cột `split` đã lưu vai trò các mẫu trong một file duy nhất; các chương trình tiếp theo sẽ **lọc theo cột này** chứ không tự chia lại.

### 1.3.3. Ý nghĩa 10 nhãn

| Label | Category | Diễn giải |
|---:|---|---|
| 0 | T-shirt/top | Áo thun/áo mặc trên |
| 1 | Trouser | Quần dài |
| 2 | Pullover | Áo chui đầu |
| 3 | Dress | Váy liền |
| 4 | Coat | Áo khoác |
| 5 | Sandal | Dép/sandal |
| 6 | Shirt | Áo sơ mi |
| 7 | Sneaker | Giày thể thao |
| 8 | Bag | Túi xách |
| 9 | Ankle boot | Bốt cổ thấp |

> **Phạm vi kiểm chứng hiện tại:** Đã đối chiếu tên cột, tổng số dòng, các giá trị `split` và phân bố nhãn trong CSV. Kiểm tra toàn bộ pixel, dữ liệu thiếu/trùng, thống kê ảnh và hiển thị ảnh mẫu thuộc **Giai đoạn 2 (EDA/Preprocessing)**.

## 1.4. Mathematical Problem Formulation — Mô hình hóa toán học

### 1.4.1. Biểu diễn mẫu và dữ liệu

Gọi tập dữ liệu có nhãn là:

$$
\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^{N},\qquad N=70{,}000.
$$

Trong đó:

- $\mathbf{x}_i\in\{0,1,\ldots,255\}^{784}$ là vector gồm 784 pixel gốc.
- $y_i\in\{0,1,\ldots,9\}$ là nhãn đúng.
- $\mathcal{D}_{\text{train}}$, $\mathcal{D}_{\text{val}}$ và $\mathcal{D}_{\text{test}}$ là ba tập **không giao nhau**, có kích thước lần lượt 54.000, 6.000 và 10.000.

Nếu chỉ scale cố định pixel về [0, 1]:

$$
\widetilde{\mathbf{x}}_i=\frac{\mathbf{x}_i}{255}.
$$

Phương án có chuẩn hóa theo mean/std sẽ được quyết định ở Giai đoạn 2. Nếu ước lượng $\mu$, $\sigma$ từ dữ liệu thì **chỉ dùng tập train**:

$$
\widehat{\mathbf{x}}_i=\frac{\widetilde{\mathbf{x}}_i-\mu_{\text{train}}}{\sigma_{\text{train}}}.
$$

### 1.4.2. Mô hình và vector đầu ra

Với mạng nơ-ron có tham số $\theta$, đầu ra thô (*logits*) là:

$$
\mathbf{z}_i=f_\theta(\widetilde{\mathbf{x}}_i)\in\mathbb{R}^{10}.
$$

Xác suất dự đoán lớp $k$ theo **softmax**:

$$
P_\theta(y=k\mid \mathbf{x}_i)
=\frac{e^{z_{i,k}}}{\sum_{j=0}^{9}e^{z_{i,j}}}.
$$

Lớp được dự đoán:

$$
\widehat{y}_i=\arg\max_{k\in\{0,\ldots,9\}} z_{i,k}.
$$

**Lưu ý triển khai:** `torch.nn.CrossEntropyLoss` nhận **logits**, vì vậy không thêm `Softmax` vào cuối mô hình trước khi tính loss.

### 1.4.3. Hàm mất mát Cross-Entropy

Với một mẫu, loss phân loại là:

$$
\ell_i(\theta)=-\log P_\theta(y=y_i\mid\mathbf{x}_i).
$$

Mục tiêu học trên **tập train**:

$$
\theta^*=\arg\min_\theta\;\mathcal{L}_{\text{train}}(\theta),
\qquad
\mathcal{L}_{\text{train}}(\theta)
=\frac{1}{N_{\text{train}}}
\sum_{i\in I_{\text{train}}}\ell_i(\theta),\quad\text{với }I_{\text{train}}\text{ là tập chỉ số huấn luyện}.
$$

Các gradient $\nabla_\theta\mathcal{L}$ được tính qua backpropagation (PyTorch Autograd); optimizer cập nhật $\theta$ theo từng batch để tìm tham số có loss thấp hơn.

### 1.4.4. Accuracy và phương pháp đánh giá

Với tập $S$ có $|S|$ mẫu:

$$
\operatorname{Accuracy}(S)
=\frac{1}{|S|}
\sum_{i\in S}\mathbf{1}(\widehat y_i=y_i)\times 100\%.
$$

$\mathbf{1}(\cdot)$ bằng 1 khi dự đoán đúng và 0 khi sai. Ví dụ **minh họa công thức, không phải kết quả chạy**: nếu đúng 8.500 trên 10.000 ảnh thì accuracy = 85%.

Loss thể hiện mức sai lệch của dự đoán; Accuracy thể hiện tỷ lệ ảnh đúng. Kết hợp cả hai giúp nhận diện dấu hiệu **overfitting** (train loss tiếp tục giảm nhưng validation loss tăng hoặc validation accuracy không cải thiện).

## 1.5. Nguyên tắc tách tập và chống Data Leakage

**Thứ tự logic bắt buộc:**

1. Đọc CSV và xác định đúng vai trò `train`/`validation`/`test` bằng cột `split`.
2. Chỉ sử dụng train để fit các phép tiền xử lý có tham số học từ dữ liệu (nếu có).
3. Áp dụng cùng quy tắc tiền xử lý cho ba tập; giữ nguyên nhãn và không làm đảo lẫn các tập.
4. Chỉ cập nhật trọng số mạng bằng train. Dùng validation để theo dõi, chọn hyperparameters/chọn mô hình và quyết định dừng.
5. Đánh giá trên test **sau khi** hoàn tất lựa chọn mô hình; không dùng test để sửa cấu hình hay chọn nghiệm tốt nhất.

**Phân biệt quan trọng:** Chia cho 255 là biến đổi cố định, không tự gây data leakage; tính mean/std từ toàn bộ 70.000 ảnh **trước khi tách tập** mới là bước cần tránh.

## 1.6. Objectives & Success Criteria — Mục tiêu và điều kiện hoàn thành

### Mục tiêu kỹ thuật

- Nạp thành công **một** CSV và bảo toàn 54k/6k/10k theo `split`.
- Có Tensor/batch đúng kích thước, ảnh đầu vào sử dụng 784 đặc trưng và output gồm 10 logits.
- Huấn luyện bằng loss và optimizer hợp lệ, lưu lại lịch sử theo epoch.
- Chỉ chọn cấu hình bằng validation; cung cấp **test accuracy cuối cùng**.
- Lưu và nạp lại mô hình; đối chiếu dự đoán trước/sau nạp khi thích hợp.
- Có ít nhất một phép so sánh cấu hình/hyperparameters với điều kiện so sánh rõ ràng.
- Xuất loss curves, ảnh dự đoán so với nhãn thật, và báo cáo giải thích kết quả.

### Thước đo và đầu ra cần báo cáo

| Nội dung | Cách xác minh | Đầu ra dự kiến |
|---|---|---|
| Dữ liệu | Count, label distribution, ảnh ví dụ | `dataset_samples.png` |
| Khả năng học | Training/validation loss theo epoch | `metrics.csv`, `loss_curve.png` |
| Chất lượng mô hình | Accuracy (val để chọn; test để chốt) | Bảng metrics, accuracy curve |
| Khả năng phân biệt lớp | Predicted vs actual, nhầm lẫn | `predictions.png`, `confusion_matrix.png` |
| Tác động hyperparameter | So sánh E1/E2/E3 | `experiment_comparison.csv/.png` |
| Tái sử dụng mô hình | Lưu và load tham số | `best_model.pth` |
| Tài liệu | Giải thích phương pháp và kết quả thực tế | `03_experiments.md`, `04_results_conclusion.md` |

**Không đặt trước kết quả accuracy hoặc tự điền kết quả chưa chạy**. Chỉ báo cáo số đo thu được trên dữ liệu thật.

## 1.7. Scope & Assumptions — Phạm vi ban đầu

- Triển khai baseline **MLP** đơn giản, không khẳng định MLP tốt hơn CNN.
- Ảnh được xem là mẫu phân loại độc lập; không phải bài toán phát hiện vị trí đối tượng.
- Cấu hình khởi điểm để thử nghiệm: batch size 64, 5 epochs, Adam, learning rate 0.001, CrossEntropyLoss và ReLU. Những tham số này có thể thay đổi trong Giai đoạn 5.
- Cần cố định random seed, phương pháp chia tập và điều kiện thí nghiệm để tăng khả năng tái lập.
- Không sử dụng thêm dữ liệu bên ngoài FashionMNIST.

## 1.8. Liên kết với 6 bước triển khai (HOW — Roadmap)

| Bước | Nhiệm vụ | Module dự kiến |
|---:|---|---|
| 1 | Load Dataset, lọc train/validation/test | `src/data_loader.py` |
| 2 | Preprocessing, normalization, DataLoader | `src/data_loader.py` |
| 3 | Build Neural Network | `src/model.py` |
| 4 | Training Loop | `src/train.py` |
| 5 | Evaluate & Save/Load | `src/evaluate.py`, `main.py` |
| 6 | Visualization & Report | `src/visualize.py`, `markdown/03_*.md`, `markdown/04_*.md` |

Chi tiết **WHEN / WHERE / WHICH / HOW** đã được trình bày trong `02_methodology.md`; thực nghiệm và kết quả thực tế lần lượt có trong `03_experiments.md` và `04_results_conclusion.md`.

## 1.9. Checklist duyệt Giai đoạn 1

- [x] Xác định đúng bài toán classification và 10 lớp đầu ra.
- [x] Định nghĩa input, output, train/validation/test.
- [x] Kiểm tra CSV có 70.000 mẫu, 786 cột, phân bố nhãn/split như thiết kế.
- [x] Mô hình hóa dữ liệu, logits, softmax, Cross-Entropy, Accuracy.
- [x] Nêu quy tắc tránh data leakage và thứ tự train/validation/test.
- [x] Ánh xạ yêu cầu của đề bài với từng đầu ra cần kiểm chứng.
- [ ] Duyệt nội dung tài liệu với người thực hiện trước khi sang Giai đoạn 2.

## Tham khảo

1. Fashion-MNIST (Zalando Research), Kaggle: <https://www.kaggle.com/datasets/zalando-research/fashionmnist>.
2. Fashion-MNIST project repository: <https://github.com/zalandoresearch/fashion-mnist>.
3. PyTorch Documentation: <https://docs.pytorch.org/docs/stable/index.html>.
4. CrossEntropyLoss — PyTorch: <https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>.
