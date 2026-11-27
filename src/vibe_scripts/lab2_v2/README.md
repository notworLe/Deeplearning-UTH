# `src/vibe_scripts/lab2_v2/` — Reusable Core Engine for Oxford-IIIT Pets Pre-trained CNNs

## Purpose

Chứa toàn bộ mã nguồn lõi tái sử dụng (data engineering, transforms, pre-trained model factory, training & TensorBoard monitoring, 4-step error analysis và inference) cho mini-project **`lab2_v2`**, tách biệt hoàn toàn với cấu hình & CLI tại [`labs/lab2_v2/`](../../../labs/lab2_v2/README.md) và cô lập khỏi bản `lab2/` v1.

## Responsibilities

- **Không hardcode** đường dẫn tuyệt đối hay siêu tham số thí nghiệm (nhận tham số từ `labs/lab2_v2/config.py`).
- **Bảo toàn Tensor Contract:** `(B, 3, 224, 224) float32 → (B, 37)` raw logits; Fail-Fast `ValueError` khi sai shape.
- **Chống Data/Test Leakage:** split cố định lưu tại `split_indices.json`, tách rõ `historys/`, `eval/` (Val) và `test_eval/` (Test 1 lần cuối).

## Public API & Modules

| Module | Vai trò |
| :--- | :--- |
| `transforms.py` | `ImageNetNormStats`, `build_train_transform`, `build_eval_transform`, `denormalize_tensor`, `preprocess_image`, `validate_image_tensor` |
| `data.py` | 6.1 Collection Audit, 6.2 Cleaning Audit, 6.3 EDA (5 PNGs), 6.4 Stratified Split (`trainval` + `35%/65% test`) & `prepare_oxford_pets()` |
| `models.py` | `build_model(arch, mode, pretrained)`, `get_param_groups()`, `count_parameters()`, `trace_shapes()` cho `ResNet18`, `DenseNet121`, `VGG16` |
| `trainer.py` | `set_global_seed`, `run_sanity_checks`, `train_single_run` (`--amp`, `--resume`, `tqdm`, TensorBoard `SummaryWriter`), `run_experiment_suite` |
| `evaluator.py` | `evaluate_run_dir`, `evaluate_final_test`, `run_deep_error_analysis` (4 bước + Cat-vs-Dog + 60 audit sheets + Slice Analysis), `plot_error_taxonomy_breakdown` |
| `inference.py` | `load_checkpoint`, `PetPredictor` (Top-5 Softmax, species prediction, latency benchmark) |

## Dependencies

- Nội bộ: không phụ thuộc ngược vào `labs/lab2_v2/config.py`.
- Thư viện: `torch`, `torchvision`, `numpy`, `Pillow`, `matplotlib`, `scikit-learn`, `tqdm` (và `tensorboard` tùy chọn).
