---
trigger: always_on
---

# PROJECT CONTEXT ANCHOR
If this file is still a template. Stop and notice the engineer write or link the project context if they didn't change

## 1. Task & Metrics
- **Task:** Làm bài tập môn hhojc
- **Primary Metric:** cụ thể ghi ở pipelines
- **Deploy Target:** Không có

## 2. Hardware & Environment
- **Train Compute:** rtx 4060
- **Precision:** Không rõ
- **Env:** có GPU

## 3. Tech Stack
- **Core:** Không cụ thể, cứ dev đi
- **Config & Tracking:** Không cụ thể, cứ dev đi
- **Export/Serving:** Không cụ thể, cứ dev đi

## 4. Directory Structure
```text
my-project/
├── README.md
├── pyproject.toml            # dependency, ruff, pytest config
├── uv.lock
├── Makefile                  # make train / eval / test / serve
├── .env.example              # không commit .env thật
├── .gitignore
├── .pre-commit-config.yaml
├── dvc.yaml                  # pipeline: prepare → train → eval (giai đoạn 2)
│
├── configs/                  # TOÀN BỘ hyperparameter nằm ở đây
│   ├── config.yaml
│   ├── data/                 # path, batch size, augmentation
│   ├── model/                # resnet18.yaml, yolo.yaml...
│   ├── trainer/              # lr, epochs, optimizer, scheduler
│   └── experiment/           # combo config cho từng thí nghiệm
│
├── data/                     # gitignore, quản bằng DVC
│   ├── raw/                  # bất biến, không sửa tay
│   ├── interim/
│   └── processed/
│
├── src/
│   ├── data/                 # Dataset, DataLoader, transforms, split
│   ├── models/               # backbone, head, loss, model factory
│   ├── training/             # train loop, callbacks, early stop, checkpoint
│   ├── evaluation/           # metrics, confusion matrix, error analysis
│   ├── inference/            # predictor, preprocess/postprocess
│   ├── optimization/         # quantize, prune, export ONNX, benchmark
│   ├── serving/              # FastAPI app, schemas
│   └── utils/                # seed, logger, io
│
├── scripts/                  # entrypoint mỏng: train.py, evaluate.py, export.py
├── notebooks/                # chỉ để EDA/thử nghiệm, đặt tên 01_eda.ipynb
├── tests/                    # test data shape, forward pass, overfit-1-batch
├── models/                   # checkpoint, .onnx (gitignore/DVC)
├── reports/                  # figures, kết quả, báo cáo
├── docker/                   # Dockerfile.train, Dockerfile.serve
├── docs/
└── .github/workflows/        # CI: ruff + pytest
```
## 5. Data & Tensor Contract
- **Image:** RGB, float32, normalized ImageNet, shape `(B, 3, H, W)` với `H=W=640`
- **Target/Mask:** [VD: `(B, H, W)` dtype `torch.long`, ignore_index=255] hoặc [VD: BBox format `xyxy` normalized 0..1]
- **Model Output:** [VD: Logits shape `(B, num_classes, H, W)` chưa qua Softmax/Sigmoid]

## 6. Engineering Guardrails
- KHÔNG hardcode path, batch_size, lr trong `src/` — mọi thứ lấy từ `cfg` (Hydra).
- Mọi thay đổi ở `models/` hoặc `datasets/` phải kèm assert kiểm tra tensor shape.
- Giữ code tối giản, ưu tiên PyTorch thuần, không tự thêm thư viện ngoài danh sách mục 3.
- Luôn tách logic `train_one_epoch` và `evaluate` (`torch.inference_mode()`).

## 7. Current State (Cập nhật thường xuyên)
- **Done:** Data pipeline + Baseline ResNet50 (val mIoU = 0.61).
- **Current Focus:** Chuyển backbone sang ConvNeXt-Small, tối ưu DataLoader đang nghẽn CPU.
- **Known Issue:** Batch size 32 bị OOM, hiện tạm chạy batch 16 + grad_accum 2.

## Writing readme
Khởi tạo file "README.md" cho mỗi 1 module phức tạp, nhầm nhiệm vụ cụ thể
nhầm mô tả chung nhất về module.

