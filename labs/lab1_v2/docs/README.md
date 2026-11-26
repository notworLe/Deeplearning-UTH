# Documentation Index — `lab1_v2` (FashionMNIST Classification)

## Purpose

Tập hợp toàn bộ tài liệu thiết kế, đặc tả mô hình và báo cáo thực nghiệm của dự án **`lab1_v2`**, đồng bộ 1-1 với mã nguồn tại [`src/vibe_scripts/lab1_v2/`](../../../src/vibe_scripts/lab1_v2/README.md) và kết quả tại [`experiments/lab1_v2/`](../../../experiments/lab1_v2/README.md).

## Responsibilities

- Chỉ lưu tài liệu Markdown (không chứa mã nguồn thực thi).
- Mọi kết quả số liệu trong báo cáo phải được trích xuất từ artifact đã xác minh trong `experiments/lab1_v2/`.

## Document Map

| Tài liệu | Vai trò | Trạng thái |
| :--- | :--- | :---: |
| [`pipelines/modeling.md`](pipelines/modeling.md) | Đặc tả trọn bộ 10 lớp pipeline, bảng biến đổi Tensor Shape, quy ước chống leakage | Hoàn thiện |
| [`models/logistic_regression.md`](models/logistic_regression.md) | Đặc tả Baseline Logistic Regression (`7,850` params), toán học & giới hạn tuyến tính | Hoàn thiện |
| [`models/mlp.md`](models/mlp.md) | Đặc tả MLP (`[128]`, `[256, 128]`, `[512, 256, 128]`), trade-offs & giới hạn so với CNN | Hoàn thiện |
| [`eda_report.md`](eda_report.md) | Báo cáo số liệu thật 6.1 Collection → 6.2 Cleaning → 6.3 EDA kèm 5 biểu đồ (Layer 6) | Hoàn thiện |
| [`experiment_report.md`](experiment_report.md) | Thiết kế chuỗi `12+1` runs, Sanity Checks & Training Diagnostics từng cấu hình (Layer 4, 7) | Hoàn thiện |
| [`evaluation_report.md`](evaluation_report.md) | Đánh giá & Chọn mô hình (Val 5k), Final Test (`10k`), Error Analysis & Inference (Layer 8, 9, 10) | Hoàn thiện |

## Related Modules

- Module thực thi & CLI: [`labs/lab1_v2/README.md`](../README.md)
- Thư mục Artifacts & Biểu đồ: [`experiments/lab1_v2/README.md`](../../../experiments/lab1_v2/README.md)
