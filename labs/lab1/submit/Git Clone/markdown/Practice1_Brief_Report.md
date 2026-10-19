# Practice 1 — PyTorch FashionMNIST Classification

**Brief Final Report** · FashionMNIST · PyTorch · Một file CSV nguồn

## 1. Problem & Motivation (WHAT / WHY)

Bài toán yêu cầu xây dựng mạng nơ-ron bằng PyTorch để phân loại ảnh xám **28×28 pixel** thành **10 lớp trang phục**. Bài thực hành nhằm hiểu quy trình từ chuẩn bị dữ liệu đến train, validation, experiment, model saving/loading, evaluation và visualization.

## 2. Dataset & Method (WHEN / WHERE / WHICH / HOW)

Nguồn: [FashionMNIST của Zalando Research trên Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist). Toàn bộ dữ liệu được gộp thành **một file** `fashion_mnist_full.csv` gồm **70.000 ảnh** (`label`, 784 pixel, `split`). Phân chia gốc: 54.000 train, 6.000 validation, 10.000 test. Loại 19 ảnh train giống hệt pixel với validation/test, còn **53.981 train hiệu lực**, giữ nguyên CSV gốc. Scale pixel và chuẩn hóa bằng mean/std **chỉ của tập train**.

Mô hình tốt nhất trong ba cấu hình là **E2 MLP 784→256→128→10**, ReLU, Adam (lr=0,001), batch size 64, 5 epochs, seed 42 và **235.146 tham số**. Training sử dụng CrossEntropyLoss và `zero_grad → forward → loss → backward → optimizer.step`. Checkpoint được chọn theo **minimum validation loss**, không sử dụng test để chọn mô hình.

## 3. Experiments & Results

| Experiment | Kiến trúc | Learning rate | Best validation loss | Validation accuracy |
|---|---|---:|---:|---:|
| E1 | 784→128→10 | 0.0010 | 0.3333 | 87.78% |
| **E2** | **784→256→128→10** | **0.0010** | **0.3130** | **88.23%** |
| E3 | 784→256→128→10 | 0.0001 | 0.3559 | 87.22% |

**Final E2 test:** **8.842/10.000 ảnh đúng (88,42%)**, cross-entropy loss **0,3089**, macro-F1 **0,8839**. E2 được chọn theo minimum validation loss ở **epoch 4**, trước khi tính các chỉ số test.

### Training and Validation Loss

![Training and validation loss](figures/loss_curve.png)

### Comparison of Experiments

![Experimental validation comparison](experiments/figures/experiment_comparison.png)

### Predicted vs Actual

![20 real test examples, correctly and incorrectly classified](figures/predictions.png)

### Confusion Matrix

![E2 confusion matrix, 10,000 test samples](experiments/figures/test_confusion_matrix.png)

## 4. Interpretation & Conclusion

Mô hình học được các đặc điểm phân loại hữu ích với test accuracy **88,42%**. Các lỗi chủ yếu nằm ở những loại áo gần giống nhau. **Shirt** có recall thấp nhất (**69,9%**), trong khi **Trouser 97,9%**, **Bag 96,8%** và **Sandal 96,5%** dễ nhận diện hơn. MLP dùng ảnh được flatten nên khó nắm bắt cấu trúc không gian như CNN. Mới có ba thí nghiệm và một seed, không thể kết luận mô hình tối ưu toàn cục.

**Kết luận:** Đã đáp ứng yêu cầu bài thực hành: Python code, mô hình train và save/load, đánh giá accuracy, thử nghiệm hyperparameters, loss graphs, hình actual vs predicted và báo cáo ngắn. Chi tiết phương pháp và toán học xem `../markdown/01_problem_formulation.md`, `02_methodology.md`, `03_experiments.md`, `04_results_conclusion.md`.
