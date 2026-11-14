# TÀI LIỆU MÔ TẢ QUÁ TRÌNH HUẤN LUYỆN

## 1. Giới thiệu

Phần này trình bày quá trình huấn luyện các mô hình ResNet18, VGG16 và DenseNet121 trên bộ dữ liệu CIFAR-10.

Mục tiêu là giúp mô hình nhận biết và phân loại chính xác 10 loại hình ảnh.

## 2. Chuẩn bị dữ liệu

Trước khi huấn luyện, dữ liệu được thực hiện các bước:

- Tải bộ dữ liệu CIFAR-10.
- Chuyển đổi hình ảnh thành Tensor.
- Thay đổi kích thước hình ảnh thành 224 x 224 pixel.
- Chuẩn hóa dữ liệu theo thông số ImageNet.
- Chia dữ liệu thành tập huấn luyện và tập xác thực.

## 3. Phương pháp huấn luyện

Bài thực hành sử dụng hai phương pháp:

- **Transfer Learning:** Sử dụng trọng số đã được huấn luyện trước và thay đổi lớp phân loại cuối cùng.
- **Fine-tuning:** Tiếp tục huấn luyện một số lớp của mô hình để cải thiện khả năng phân loại trên CIFAR-10.

## 4. Các thông số huấn luyện

Các thông số dự kiến sử dụng:

| Thông số | Giá trị |
|---|---|
| Bộ dữ liệu | CIFAR-10 |
| Số lớp phân loại | 10 |
| Kích thước ảnh | 224 x 224 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Hàm mất mát | CrossEntropyLoss |
| Số Epoch | 5 |

Các thông số có thể được điều chỉnh trong quá trình thực nghiệm.

## 5. Theo dõi quá trình huấn luyện

Sử dụng TensorBoard để theo dõi:

- Training Loss: Giá trị mất mát trên tập huấn luyện.
- Validation Loss: Giá trị mất mát trên tập xác thực.
- Accuracy: Độ chính xác của mô hình.

Các thông tin được ghi lại để thuận tiện cho việc quan sát và so sánh.

## 6. Thử nghiệm thông số

Thực hiện thay đổi một số thông số như:

- Learning Rate: 0.001 và 0.0001.
- Batch Size: 32 và 64.

Sau đó so sánh kết quả để đánh giá ảnh hưởng của từng thông số đến quá trình huấn luyện.

## 7. Kết luận

Quá trình huấn luyện giúp các mô hình học cách phân loại hình ảnh CIFAR-10. Việc theo dõi Loss và Accuracy giúp đánh giá quá trình học và lựa chọn thông số phù hợp.