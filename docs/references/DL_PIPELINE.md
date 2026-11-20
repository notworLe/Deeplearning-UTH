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

# Errors analysis

## Example

nhầm phải viết báo cáo trên REPORT.md thêm lan t-08 chứ
nhưng bổ sung evaltue và thêm về đồ thị thêm ảnh learning curve của từng riêng model nữa

Bản chất

Accuracy/loss chỉ cho bạn một con số trung bình. Error analysis là mở từng mẫu sai ra xem, để biết nên sửa cái gì. Không có nó, bạn chỉ đoán mò kiểu “train thêm epoch”, “đổi optimizer”.

Ví dụ: FashionMNIST, accuracy 88%

88% không cho bạn biết gì để cải thiện. Bạn làm 3 bước:

1. Chia lỗi theo class (confusion matrix)

python
model.eval()
preds, labels, imgs = [], [], []
with torch.no_grad():
for x, y in test_loader: # x: [B,1,28,28]
p = model(x.to(device)).argmax(1).cpu() # logits [B,10] -> [B]
preds.append(p); labels.append(y); imgs.append(x)
preds, labels, imgs = map(torch.cat, (preds, labels, imgs))

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(labels, preds, target_names=class_names))
cm = confusion_matrix(labels, preds)

Kết quả thường thấy:

Class Accuracy
Sneaker, Bag, Trouser ~96-98%
Pullover, Coat ~80%
Shirt ~60%

→ Model không “yếu đều”. Nó chủ yếu nhầm Shirt ↔ T-shirt ↔ Pullover ↔ Coat. Vấn đề đã thu hẹp từ “cải thiện 88%” xuống “phân biệt các loại áo”.

2. Xem trực tiếp các ảnh bị sai

python
wrong = (preds != labels).nonzero().squeeze()

# plot ~25 ảnh trong wrong, ghi nhãn thật vs nhãn dự đoán

Bạn sẽ thấy: ảnh 28x28 mờ, Shirt và T-shirt gần như giống nhau, một số label vốn mơ hồ.

3. Từ nguyên nhân → hành động

Quan sát Nguyên nhân Hướng sửa
Train acc cao, test thấp, sai ở class khó Overfit Augmentation, dropout, weight decay
Cả train lẫn test đều thấp ở các class áo Underfit, model nhỏ CNN sâu hơn, thêm conv layer
Ảnh người cũng khó phân biệt Label noise / giới hạn dữ liệu Chấp nhận trần accuracy, đừng cố ép

Cách này tiết kiệm hơn nhiều so với thử ngẫu nhiên.

Vòng lặp cần nhớ

Train → phân tích lỗi → đặt giả thuyết → sửa 1 thứ → train lại → so với baseline.

Chỉ sửa một thứ mỗi lần, nếu không bạn không biết cái gì có tác dụng.

Checklist ngắn khi làm
Per-class metrics (precision/recall), không chỉ accuracy
Confusion matrix: cặp class nào nhầm nhau nhất
Xem ~20-50 mẫu sai bằng mắt
So train error vs val error để phân biệt overfit/underfit
