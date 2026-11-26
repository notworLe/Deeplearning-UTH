# `src/vibe_scripts/lab1_v2/` — Reusable Core ML Engine (`lab1_v2`)

## Purpose

Gói mã nguồn lõi tái sử dụng cho bài toán phân loại ảnh (**FashionMNIST** và các bài toán ảnh xám/đa lớp tương tự), thiết kế theo nguyên tắc **Single Responsibility**, **Config-first** (không hardcode đường dẫn hay siêu tham số) và **Fail-Fast Tensor Contracts**.

## Responsibilities

Mỗi module chỉ phụ trách đúng 1 tầng trong `MODEL_PIPELINE.md`:

| Module | Trách nhiệm duy nhất | Không được làm |
| :--- | :--- | :--- |
| `transforms.py` | Tính `NormStats` từ Train, `validate_image_tensor`, `normalize_batch`, `denormalize_tensor`, `preprocess_image` | Không nạp đĩa, không train |
| `data.py` | 6.1 `run_data_collection_audit`, 6.2 `run_data_cleaning_audit`, 6.3 `export_eda_artifacts`, Stratified split `55k/5k/10k` (`prepare_fashion_mnist`) | Không train hoặc evaluate model |
| `logistic.py` | Kiến trúc `LogisticRegressionModel` (`7,850` params), `trace_shapes()`, `count_parameters()` | Không chứa loss, optimizer hay Softmax |
| `mlp.py` | Kiến trúc `MLPModel` cấu hình động (`hidden_dims`, `ReLU`, `Dropout`), factory `build_model()` | Không nạp dataset hay chạy train loop |
| `trainer.py` | `set_global_seed`, `run_sanity_checks`, `train_single_run` (`tqdm` monitoring → `historys/history.{json,csv}` + `best.pt`/`last.pt`), `run_experiment_suite` (per-run eval & early-stop) | Không đánh giá tập Test cuối |
| `evaluator.py` | `evaluate_model_on_loader`, `verify_save_load_parity` (`< 1e-6`), `evaluate_run_dir` (`eval/` vs `test_eval/`), `evaluate_final_test` | Không cập nhật trọng số mô hình |
| `inference.py` | `load_checkpoint`, `FashionPredictor` (`predict_batch`, `predict_image`, `benchmark_latency` dưới `@torch.inference_mode()`) | Không cập nhật trọng số mô hình |

## Public API & Tensor Contracts

- **Input Contract:** `(B, 1, 28, 28)` dtype `torch.float32`, hữu hạn (kiểm tra Fail-Fast bằng `validate_image_tensor`, sai shape ném `ValueError`).
- **Model Output Contract:** raw logits `(B, 10)` dtype `torch.float32` (chưa qua Softmax).
- **Inference Output Contract:** xác suất Softmax `(B, 10)` có tổng `= 1.0 ± 1e-5`, Top-K giảm dần, `requires_grad == False`.
- Toàn bộ public symbols được re-export thống nhất tại `src/vibe_scripts/lab1_v2/__init__.py`.

## Dependencies

- Thư viện: `torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`.
- Không phụ thuộc cứng vào `labs/lab1_v2/config.py` (mọi cấu hình được truyền qua tham số hàm).

## Usage

```python
from pathlib import Path
from src.vibe_scripts.lab1_v2 import (
    FashionPredictor,
    build_model,
    prepare_fashion_mnist,
    run_sanity_checks,
)

bundle = prepare_fashion_mnist(data_root=Path("data"), batch_size=64, val_size=5000, seed=248)
model = build_model("mlp", hidden_dims=(256, 128), dropout=0.2)
```

- CLI Entrypoints: [`labs/lab1_v2/README.md`](../../../labs/lab1_v2/README.md)
- Artifacts Output: [`experiments/lab1_v2/README.md`](../../../experiments/lab1_v2/README.md)
- Đặc tả Pipeline: [`labs/lab1_v2/docs/pipelines/modeling.md`](../../../labs/lab1_v2/docs/pipelines/modeling.md)
