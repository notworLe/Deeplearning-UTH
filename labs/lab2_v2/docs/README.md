# Documentation Index — `lab2_v2` (Oxford-IIIT Pets Pre-trained CNNs)

## Purpose

Tập hợp toàn bộ tài liệu thiết kế pipeline, đặc tả 3 kiến trúc pre-trained (`ResNet18`, `DenseNet121`, `VGG16`) và 3 báo cáo đơn nhiệm (SRP 1-1 theo các tầng `MODEL_PIPELINE.md`) của dự án **`lab2_v2`**, đồng bộ với mã nguồn lõi tại [`src/vibe_scripts/lab2_v2/`](../../../src/vibe_scripts/lab2_v2/README.md) và kết quả tại [`experiments/lab2_v2/`](../../../experiments/lab2_v2/README.md).

## Responsibilities

- Chỉ lưu tài liệu Markdown (không chứa mã nguồn thực thi).
- Mọi kết quả số liệu trong báo cáo phải được trích xuất từ artifact đã xác minh trong `experiments/lab2_v2/` (không suy đoán trước khi chạy thật).

## Document Map

| Tài liệu | Vai trò & Pipeline Layer | Trạng thái |
| :--- | :--- | :---: |
| [`pipelines/modeling.md`](pipelines/modeling.md) | Đặc tả trọn bộ 10 tầng pipeline, Tensor Shape Contract, quy tắc chọn mô hình §8.1 & chống leakage | Hoàn thiện |
| [`models/resnet18.md`](models/resnet18.md) | Đặc tả `ResNet18` (3 chế độ: `scratch`, `transfer`, `finetune layer4`), Residual block & differential LR | Hoàn thiện |
| [`models/densenet121.md`](models/densenet121.md) | Đặc tả `DenseNet121` (`transfer`), Dense Connectivity & hiệu quả tham số | Hoàn thiện |
| [`models/vgg16.md`](models/vgg16.md) | Đặc tả `VGG16` (`transfer`), cấu trúc conv `3×3` cổ điển & nút thắt `134.4M` params | Hoàn thiện |
| [`eda_report.md`](eda_report.md) | Báo cáo số liệu thật 6.1 Collection → 6.2 Cleaning → 6.3 EDA kèm 5 biểu đồ (Layer 6) | Hoàn thiện |
| [`experiment_report.md`](experiment_report.md) | Thiết kế 9 runs, Sanity Checks, Training Diagnostics & TensorBoard monitoring (Layer 4, 7) | Hoàn thiện |
| [`evaluation_report.md`](evaluation_report.md) | Đánh giá & Chọn mô hình trên Val, Final Test 1 lần, Error Analysis 4 bước & Inference (Layer 8–10) | Hoàn thiện |

## Related Modules

- Module thực thi & CLI: [`labs/lab2_v2/README.md`](../README.md)
- Thư mục Artifacts & Biểu đồ: [`experiments/lab2_v2/README.md`](../../../experiments/lab2_v2/README.md)
