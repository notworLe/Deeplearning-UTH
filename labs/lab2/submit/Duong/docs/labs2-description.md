# BÀI THỰC HÀNH 2: SỬ DỤNG CÁC KIẾN TRÚC MẠNG NƠ-RON ĐÃ ĐƯỢC HUẤN LUYỆN TRƯỚC

## 1. Giới thiệu

Bài thực hành này tập trung vào việc sử dụng các mô hình mạng nơ-ron tích chập (CNN) đã được huấn luyện trước để thực hiện bài toán phân loại hình ảnh.

Mô hình được huấn luyện trước là những mô hình đã học từ một tập dữ liệu lớn. Chúng ta có thể sử dụng lại kiến thức của các mô hình này cho một bài toán mới thông qua hai phương pháp là học chuyển giao (Transfer Learning) và tinh chỉnh mô hình (Fine-tuning).

Trong bài thực hành này, chúng ta sử dụng bộ dữ liệu CIFAR-10 để thực hiện phân loại hình ảnh và đánh giá hiệu quả của các mô hình.

## 2. Mục tiêu

Các mục tiêu chính của bài thực hành:

- Tìm hiểu kiến trúc của các mô hình mạng nơ-ron đã được huấn luyện trước.
- Biết cách tải và sử dụng mô hình có sẵn.
- Thay đổi lớp phân loại cuối cùng để phù hợp với bộ dữ liệu mới.
- Thực hiện học chuyển giao và tinh chỉnh mô hình.
- Thử nghiệm các thông số huấn luyện khác nhau.
- Theo dõi quá trình huấn luyện bằng TensorBoard.
- Đánh giá và so sánh kết quả của các mô hình.

## 3. Bộ dữ liệu sử dụng

Bài thực hành sử dụng bộ dữ liệu CIFAR-10.

CIFAR-10 gồm 60.000 hình ảnh màu, được chia thành 10 nhóm:

1. Máy bay.
2. Ô tô.
3. Chim.
4. Mèo.
5. Hươu.
6. Chó.
7. Ếch.
8. Ngựa.
9. Tàu thủy.
10. Xe tải.

## 4. Các mô hình sử dụng

Bài thực hành lựa chọn ba kiến trúc mạng nơ-ron:

- ResNet18.
- VGG16.
- DenseNet121.

Các mô hình được tải thông qua thư viện Torchvision với trọng số đã được huấn luyện trên bộ dữ liệu ImageNet.

## 5. Công cụ và công nghệ

- Python: Ngôn ngữ lập trình chính.
- PyTorch: Thư viện xây dựng và huấn luyện mô hình.
- Torchvision: Cung cấp dữ liệu và các kiến trúc mô hình.
- NumPy: Hỗ trợ xử lý dữ liệu.
- Matplotlib: Hiển thị hình ảnh và biểu đồ.
- TensorBoard: Theo dõi quá trình huấn luyện.
- Jupyter Notebook: Môi trường viết và chạy chương trình.

## 6. Kết quả mong đợi

Sau khi hoàn thành bài thực hành, chúng ta mong muốn:

- Hiểu được cách hoạt động của mô hình đã được huấn luyện trước.
- Áp dụng được học chuyển giao cho bộ dữ liệu CIFAR-10.
- So sánh hiệu quả giữa các kiến trúc CNN.
- Hiểu ảnh hưởng của việc thay đổi thông số huấn luyện.
- Biết cách đánh giá kết quả thông qua các chỉ số và biểu đồ.