# `labs/lab2_v2/` — Oxford-IIIT Pets Pre-trained CNNs Mini-Project (`ResNet18`, `DenseNet121`, `VGG16`)

## Purpose

Điểm vào thực thi (cấu hình, thin CLI scripts, executable report notebook và tài liệu kỹ thuật) của mini-project **`lab2_v2`** — phân loại 37 giống chó & mèo trên **Oxford-IIIT Pets** bằng Transfer Learning & Fine-Tuning theo đủ 10 tầng [`MODEL_PIPELINE.md`](../../.agents/rules/dl-rule/MODEL_PIPELINE.md).

## Responsibilities

- **Cấu hình (`config.py`):** Kế thừa `src/config.py` qua `VibeConfig`, định nghĩa split (`trainval` + `35%/65% test`), chuẩn hóa ImageNet và chuỗi 9 runs có kiểm soát.
- **Thin CLI Entrypoints:** Điều phối Data Engineering (`prepare_data.py`), Training & TensorBoard Monitoring (`train.py`), Evaluation & 4-Step Error Analysis (`evaluate.py`) và Inference (`inference.py`), gọi lõi tại [`src/vibe_scripts/lab2_v2/`](../../src/vibe_scripts/lab2_v2/README.md).
- **Executable Report & Docs:** Trình bày trực quan kết quả từ [`experiments/lab2_v2/`](../../experiments/lab2_v2/README.md) qua `experiment.ipynb` và bộ tài liệu SRP tại [`docs/README.md`](docs/README.md).

## Public API & Entrypoints

| File | Vai trò |
| :--- | :--- |
| `config.py` | Xuất `CONFIG`, `EXPERIMENT_SPECS` (9 runs), đường dẫn `experiments/lab2_v2/*`, `get_run_spec(run_id)` |
| `prepare_data.py` | CLI chạy **6.1 Collection → 6.2 Cleaning → 6.3 EDA** + xuất `split_indices.json` |
| `train.py` | CLI chạy **Sanity Check → Training (`--amp`, `--resume`, `tqdm`, TensorBoard) → Per-Run Val Eval → Final Test** |
| `evaluate.py` | CLI đánh giá 1 run, chạy lại Final Test (1 lần) hoặc chạy **4-Step Error Analysis** (`--error-analysis`) |
| `inference.py` | CLI nạp checkpoint `.pt`, dự đoán Top-5 Softmax (kèm loài Cat/Dog) & đo latency |
| `experiment.ipynb` | Executable report notebook đọc artifacts & kiểm chứng nhẹ (không train lại) |

## Usage

```powershell
# 1. Data Engineering (6.1 Collection -> 6.2 Cleaning -> 6.3 EDA + split_indices.json)
python labs/lab2_v2/prepare_data.py

# (Tùy chọn) Cài TensorBoard để xem biểu đồ trực tiếp khi train:
pip install tensorboard

# 2. Kỹ sư chạy Training & Monitoring (có tqdm bar log, AMP FP16, TensorBoard & per-run Val eval)
python labs/lab2_v2/train.py --run-all --amp

# (Tùy chọn) Chạy từng run riêng / resume thêm epochs / xem TensorBoard:
python labs/lab2_v2/train.py --run-id resnet18_transfer --amp
python labs/lab2_v2/train.py --run-id resnet18_finetune --resume experiments/lab2_v2/runs/resnet18_finetune/last.pt --extra-epochs 10 --output-run-id resnet18_finetune_ep25 --skip-sanity
tensorboard --logdir experiments/lab2_v2/tb_logs

# 3. Đánh giá Final Test (1 lần) & Error Analysis 4 bước định lượng
python labs/lab2_v2/evaluate.py
python labs/lab2_v2/evaluate.py --error-analysis

# 4. Suy luận Top-5 & đo độ trễ trên checkpoint tốt nhất
python labs/lab2_v2/inference.py --sample-idx 0 --benchmark

# 5. Kiểm thử tự động offline (Smoke -> Normal)
pytest -m smoke test/test_lab2_v2_data.py test/test_lab2_v2_models.py test/test_lab2_v2_pipeline.py -v
pytest test/test_lab2_v2_data.py test/test_lab2_v2_models.py test/test_lab2_v2_pipeline.py -v
```
