# TÀI LIỆU ĐÁNH GIÁ KẾT QUẢ MÔ HÌNH

## 1. Mục đích đánh giá

Phần này dùng để đánh giá khả năng phân loại hình ảnh của ba mô hình ResNet18, VGG16 và DenseNet121 trên bộ dữ liệu CIFAR-10.

## 2. Các chỉ số đánh giá

Sử dụng các chỉ số sau:

- **Accuracy (Độ chính xác):** Tỷ lệ hình ảnh được mô hình dự đoán đúng.
- **Loss (Giá trị mất mát):** Đánh giá mức độ sai lệch giữa kết quả dự đoán và nhãn thực tế.
- **Confusion Matrix (Ma trận nhầm lẫn):** Cho biết mô hình dự đoán đúng hoặc nhầm giữa các lớp như thế nào.

## 3. Phương pháp đánh giá

Sau khi huấn luyện, các mô hình được kiểm tra trên tập dữ liệu kiểm tra CIFAR-10.

Quá trình đánh giá gồm:

1. Đưa hình ảnh vào mô hình.
2. Nhận kết quả dự đoán.
3. So sánh kết quả dự đoán với nhãn thực tế.
4. Tính độ chính xác và giá trị mất mát.
5. Tạo ma trận nhầm lẫn để quan sát kết quả.

## 4. So sánh các mô hình

Kết quả của ba mô hình sẽ được tổng hợp theo bảng sau:

| Mô hình | Độ chính xác | Thời gian huấn luyện |
|---|---|---|
| ResNet18 | Cập nhật sau khi chạy | Cập nhật sau khi chạy |
| VGG16 | Cập nhật sau khi chạy | Cập nhật sau khi chạy |
| DenseNet121 | Cập nhật sau khi chạy | Cập nhật sau khi chạy |

## 5. Hình ảnh trực quan

Sử dụng Matplotlib để hiển thị:

- Biểu đồ Loss trong quá trình huấn luyện.
- Biểu đồ Accuracy.
- Ma trận nhầm lẫn.
- Một số hình ảnh thực tế và kết quả dự đoán của mô hình.

## 6. Kết luận

Thông qua các chỉ số đánh giá và biểu đồ trực quan, chúng ta có thể so sánh khả năng phân loại hình ảnh của từng mô hình và lựa chọn mô hình phù hợp.