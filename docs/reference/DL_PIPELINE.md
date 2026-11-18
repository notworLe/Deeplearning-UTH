# Training
## Loss
- CrossEntropyLoss (nhận logits, đã gồm log_softmax)

## Optimizer
- Loại: Adam (hoặc SGD + momentum), lr, weight decay

## Cấu hình huấn luyện
- Batch size 32, số epoch, seed 248, device (CPU/GPU)

## Vòng lặp
- forward → loss → zero_grad → backward → step
- Mỗi epoch: đánh giá trên validation (model.eval(), no_grad)

## Theo dõi & lưu
- Log: train/val loss, accuracy, thời gian/epoch
- Checkpoint: lưu model có val loss/accuracy tốt nhất


# Model selection
## Tiêu chí
## Ứng viên (bảng: vai trò, kiến trúc, số tham số)
## Phương án đã loại và lý do
## Giả thuyết (H1-H3)
## Cách chọn (validation, lưới lr, tiêu chí checkpoint)

# Baseline
## Mục đích
## Mốc tham chiếu (mốc 0, Logistic)
## Kiến trúc
## Cấu hình (kế thừa Training; có/không augmentation)
## Sanity check
## Kết quả (bảng, loss curve, confusion matrix, ảnh đúng/sai)
## Phân tích
## Kết luận

# Baseline
## Mục đích
## Mốc tham chiếu (mốc 0, Logistic)
## Kiến trúc
## Cấu hình (kế thừa Training; có/không augmentation)
## Sanity check
## Kết quả (bảng, loss curve, confusion matrix, ảnh đúng/sai)
## Phân tích
## Kết luận


# Evaluate
## Nguyên tắc
## Metric
## Bảng kết quả (3 mô hình × 2 cấu hình)
## Đường cong học
## Phân tích theo lớp
## Phân tích lỗi (confusion matrix, ảnh đúng/sai)
## Kiểm tra giả thuyết H1-H3
## Kiểm tra save/load
## Hạn chế và kết luận