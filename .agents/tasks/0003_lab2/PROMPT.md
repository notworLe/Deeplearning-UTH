Name: Lab 2 – Đánh giá fine-tune cho bài Oxford-IIIT Pets (Baseline / Transfer / Fine-tune)

Description: So sánh ba run ResNet18 trên Oxford-IIIT Pets để trả lời câu hỏi:
fine-tune có hợp lý cho case này không. Đi đủ vòng đời: data collection -> cleaning ->
EDA -> preprocessing -> split -> model -> train -> evaluate -> error analysis.

Purpose: Hiểu khác biệt giữa train từ đầu, transfer learning (freeze backbone) và
fine-tune (mở khóa một phần backbone). Kết luận fine-tune có đáng làm hay không dựa trên
số liệu thật của bài, không dựa vào giả định.

Input:

- Dataset: torchvision.datasets.OxfordIIITPet, target_types="category"
  - 7.349 ảnh, 37 giống (25 chó, 12 mèo), khoảng 200 ảnh/lớp
  - Split chính thức: trainval 3.680 / test 3.669
  - Split thực nghiệm: Train dùng toàn bộ trainval (3.680 ảnh); Val tách 10% từ tập test gốc (367 ảnh, stratified theo giống, seed cố định, ~10 ảnh/lớp nên không diễn giải kết quả từng lớp trên val); Test cuối cùng là 90% còn lại (3.302 ảnh, disjoint với Val, chỉ chạm một lần ở cuối)
  - Ảnh không đồng nhất kích thước, đổi về 3x224x224 (Resize(256) + CenterCrop(224)),
    normalize theo ImageNet
- Model: ResNet18 (torchvision)

Thí nghiệm (cùng cấu hình: 224x224, val 10%, 30 epoch, Adam, batch 32, cùng seed,
cùng augmentation; lr của head giữ như config baseline hiện tại):

1. resnet18_baseline (ĐÃ CÓ): weights=None, train toàn bộ, fc 512->37.
2. resnet18_transfer: weights=IMAGENET1K_V1, thay fc thành 512->37 (18.981 tham số,
   khởi tạo ngẫu nhiên), requires_grad=False cho toàn bộ backbone, backbone luôn ở
   eval() để BatchNorm không cập nhật; optimizer chỉ cho fc.
3. resnet18_finetune: thay fc 512->37 trước, rồi nạp checkpoint resnet18_transfer
   (head đã train sẵn), unfreeze layer4 + fc (phạm vi unfreeze ghi rõ trong báo cáo),
   lr backbone nhỏ hơn lr head (param groups), các BN thuộc phần khóa để eval().

Output: REPORT.md cấu trúc như notebook, mỗi mục kèm kết quả và phân tích kỹ thuật ngắn:
REPORT.md NGẮN GỌN, XÚC TÍCH, Ưu tiên bảng và số liệu; không giải thích lại lý thuyết, không lặp ý

1. Abstract
2. Problem statement (input/output, metric, giả thuyết, tiêu chí kết luận đặt TRƯỚC khi xem kết quả)
3. Data collection & cleaning (ảnh lỗi, chế độ màu, ảnh trùng)

- Lý do chọn dataset

4. EDA (số ảnh theo giống ở trainval/test, phân bố kích thước và tỉ lệ cạnh, ảnh mẫu,
   tỉ lệ chó/mèo, kiểm tra nhãn loài qua tên file: mèo viết hoa chữ đầu, chó viết thường)
5. Preprocessing & split
6. Model & thí nghiệm (số tham số trainable của từng run)
7. Training setup
8. Results (bảng accuracy, macro-F1 trên test, epoch tới val tốt nhất, thời gian train,
   tham số trainable; learning curve train/val của cả ba run)
9. Error analysis (confusion matrix, top cặp giống nhầm, ảnh dự đoán sai)
10. Conclusion (fine-tune có hợp lý không, kèm số liệu)

How to do:

- Tuân thủ AI delegation loop trong .agents (plan -> implement -> verify -> report).
- Thứ tự code: thay fc TRƯỚC, nạp checkpoint SAU (checkpoint transfer đã có fc 37 lớp).
- Mỗi run chỉ đổi một biến so với run liền trước (baseline -> transfer: pretrained +
  freeze; transfer -> fine-tune: mở khóa layer4). Chọn checkpoint theo val, test chỉ
  chạm một lần ở cuối.
- Câu hỏi chính được trả lời bằng so sánh transfer và fine-tune. Baseline so với transfer
  chỉ cho biết pretrained có giá trị không; không dùng so sánh baseline và fine-tune
  để kết luận về fine-tune vì gộp hai biến.
- Tiêu chí "hợp lý" (đặt trước): mức tăng test macro-F1 và accuracy của fine-tune so với
  transfer, đối chiếu chi phí (tham số trainable, thời gian train). Test cuối cùng có 3.302 ảnh nên
  chênh lệch dưới khoảng 1 điểm phần trăm gần như nằm trong nhiễu một lần chạy; nếu rơi
  vào vùng này thì chạy thêm 2-3 seed trước khi kết luận.
- Đọc cả train và val: nếu fine-tune tăng train mà val không tăng, ghi nhận là overfit,
  không tính là lợi ích.
- Kiểm chứng giả thuyết (chưa được đo): lỗi dồn vào giống mèo nhiều hơn giống chó vì
  ImageNet chỉ có 5 lớp mèo. Nêu rõ đúng hay sai theo số liệu.
- Báo cáo trung thực kể cả kết quả bất ngờ, kể cả khi fine-tune không thắng.

Examples:

- Pets (3x224x224, 37 giống) -> ResNet18 -> transfer train fc (~19k tham số) vs fine-tune
  layer4 + fc -> fine-tune tăng X điểm macro-F1 so với transfer với chi phí Y -> kết luận.

Reference:

- ROOT/labs/lab2/vibe/docs/report
- ROOT/.agents/rules/AI_FIRST_LIFECYCLE.md
- ROOT/.agents/rules/PLANNING.md
