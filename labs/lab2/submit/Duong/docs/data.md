# TÀI LIỆU MÔ TẢ BỘ DỮ LIỆU

## 1. Giới thiệu bộ dữ liệu

Bài thực hành sử dụng bộ dữ liệu CIFAR-10 để thực hiện bài toán phân loại hình ảnh bằng các mô hình học sâu.

CIFAR-10 là bộ dữ liệu gồm 60.000 hình ảnh màu, có kích thước ban đầu là 32 x 32 pixel.

## 2. Cấu trúc bộ dữ liệu

Bộ dữ liệu được chia thành hai phần:

- Tập huấn luyện (Training Set): 50.000 hình ảnh.
- Tập kiểm tra (Testing Set): 10.000 hình ảnh.

Bộ dữ liệu có tổng cộng 10 lớp, mỗi lớp đại diện cho một loại đối tượng.

## 3. Danh sách 10 lớp dữ liệu

| STT | Tên lớp | Ý nghĩa |
|---|---|---|
| 1 | Airplane | Máy bay |
| 2 | Automobile | Ô tô |
| 3 | Bird | Chim |
| 4 | Cat | Mèo |
| 5 | Deer | Hươu |
| 6 | Dog | Chó |
| 7 | Frog | Ếch |
| 8 | Horse | Ngựa |
| 9 | Ship | Tàu thủy |
| 10 | Truck | Xe tải |

## 4. Tiền xử lý dữ liệu

Để phù hợp với các mô hình đã được huấn luyện trước trên ImageNet, dữ liệu được xử lý như sau:

- Thay đổi kích thước hình ảnh từ 32 x 32 thành 224 x 224 pixel.
- Chuyển hình ảnh thành dạng Tensor.
- Chuẩn hóa giá trị hình ảnh theo thông số của ImageNet.
- Chia dữ liệu huấn luyện thành tập huấn luyện và tập xác thực.

## 5. Cách sử dụng dữ liệu

Dữ liệu CIFAR-10 được tải thông qua thư viện Torchvision.

Các hình ảnh sau khi tiền xử lý được đưa vào mô hình ResNet18, VGG16 và DenseNet121 để thực hiện phân loại.

## 6. Kết luận

CIFAR-10 là bộ dữ liệu phù hợp để thực hành phân loại hình ảnh. Bộ dữ liệu giúp đánh giá khả năng học và phân loại của các mô hình mạng nơ-ron đã được huấn luyện trước.