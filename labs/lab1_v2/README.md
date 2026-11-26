# `labs/lab1_v2/` — FashionMNIST Classification Mini-Project (`Logistic Regression` vs `MLP`)

## Purpose

Điểm vào thực thi (CLI entrypoints, cấu hình, executable report notebook và tài liệu kỹ thuật) của mini-project **`lab1_v2`** — phân loại 10 lớp trang phục **FashionMNIST** theo đầy đủ 10 lớp của `MODEL_PIPELINE.md`.

## Responsibilities

- **Cấu hình (`config.py`):** Kế thừa `src/config.py` qua `VibeConfig` (`strict=True`), định nghĩa toàn bộ siêu tham số, ngưỡng dừng sớm và chuỗi 12 runs có kiểm soát.
- **Thin CLI Entrypoints:** Điều phối luồng Data Engineering (`prepare_data.py`), Training & Monitoring (`train.py`), Evaluation (`evaluate.py`) và Inference (`inference.py`) bằng cách gọi mã nguồn lõi tại [`src/vibe_scripts/lab1_v2/`](../../src/vibe_scripts/lab1_v2/README.md).
- **Executable Report & Docs:** Trình bày trực quan kết quả từ [`experiments/lab1_v2/`](../../experiments/lab1_v2/README.md) qua `experiment.ipynb` và bộ tài liệu tại [`docs/README.md`](docs/README.md).
- **Không chứa:** logic thuật toán lõi tái sử dụng (đặt tại `src/vibe_scripts/lab1_v2/`).

## Public API & Entrypoints

| File | Vai trò |
| :--- | :--- |
| `config.py` | Xuất `CONFIG`, `EXPERIMENT_SPECS` (12 runs), đường dẫn `experiments/lab1_v2/*`, hàm `get_run_spec(run_id)` |
| `prepare_data.py` | CLI chạy **6.1 Collection → 6.2 Cleaning → 6.3 EDA** vào `experiments/lab1_v2/eda/` |
| `train.py` | CLI chạy **Sanity Check → Training + `tqdm` Monitoring (`historys/history.*`) → Per-Run Val Eval (`eval/`) → Early-Stop → Final Test (10k)** |
| `evaluate.py` | CLI đánh giá 1 run cụ thể hoặc chạy lại so sánh Final Test (`Baseline` vs `Best MLP`) |
| `inference.py` | CLI nạp checkpoint `.pt`, dự đoán Top-K Softmax cho ảnh mẫu/ảnh ngoài và đo latency |
| `experiment.ipynb` | Executable report notebook đọc artifacts & chạy kiểm chứng nhẹ (không train lại) |

## Dependencies

- Nội bộ: [`src/vibe_scripts/lab1_v2/`](../../src/vibe_scripts/lab1_v2/README.md), `src/config.py`, `src/vibe_scripts/config_override.py`, đề bài [`labs/lab1/QUESTION.md`](../lab1/QUESTION.md).
- Thư viện: `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`, `Pillow`, `nbformat`, `pytest`.

## Usage

```powershell
# 1. Data Engineering (6.1 Collection -> 6.2 Cleaning -> 6.3 EDA)
python labs/lab1_v2/prepare_data.py

# 2. Kỹ sư chạy Training & Monitoring (có tqdm bar log, lưu historys/history.* & eval/)
python labs/lab1_v2/train.py --run-all

# (Tùy chọn) Chạy đủ 12 runs không dừng sớm / bật batch bar / chạy riêng 1 run / resume:
python labs/lab1_v2/train.py --run-all --no-stop-on-target
python labs/lab1_v2/train.py --run-all --batch-bar
python labs/lab1_v2/train.py --run-id mlp_default
python labs/lab1_v2/train.py --run-id mlp_opt_sgd --resume experiments/lab1_v2/runs/mlp_opt_sgd/last.pt --extra-epochs 20 --output-run-id mlp_opt_sgd_ep40 --skip-sanity

# 3. Đánh giá Final Test (10k) cho Baseline vs Winning MLP
python labs/lab1_v2/evaluate.py

# 4. Suy luận Top-3 & đo độ trễ trên checkpoint tốt nhất
python labs/lab1_v2/inference.py --sample-idx 0 --benchmark

# 5. Kiểm thử tự động (Dual Testing: Smoke -> Normal)
pytest -m smoke test/test_lab1_v2_*.py -v
pytest test/test_lab1_v2_*.py -v
```
