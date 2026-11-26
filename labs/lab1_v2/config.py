"""Cấu hình trung tâm cho Lab 1 v2 (FashionMNIST: Logistic Regression vs MLP).

Kế thừa hằng số nền từ `src/config.py` qua `VibeConfig` (chế độ `strict=True`)
và khai báo toàn bộ siêu tham số, đường dẫn artifact và chuỗi 12 runs ablation.
Không hardcode đường dẫn tuyệt đối hay siêu tham số trong `src/vibe_scripts/lab1_v2/`.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import src.config as base_config
from src.vibe_scripts.config_override import VibeConfig

_ROOT_PATH: Path = base_config.ROOT_PATH
_EXPERIMENT_DIR: Path = _ROOT_PATH / "experiments" / "lab1_v2"

_DEFAULT_HIDDEN_DIMS: Tuple[int, ...] = (256, 128)
_DEFAULT_DROPOUT: float = 0.2
_DEFAULT_LR: float = 1e-3
_DEFAULT_BATCH_SIZE: int = 64
_DEFAULT_EPOCHS: int = 20

_EXPERIMENT_SPECS: Tuple[Dict[str, Any], ...] = (
    # Run 1: Baseline (luôn chạy đầu tiên để làm mốc đối chứng)
    {
        "run_id": "baseline_logistic",
        "group": "Baseline",
        "axis": "baseline",
        "model_type": "logistic",
        "hidden_dims": (),
        "dropout": 0.0,
        "optimizer": "adam",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": True,
    },
    # Run 2: MLP Anchor (cấu hình chuẩn làm tâm cho 5 trục ablation)
    {
        "run_id": "mlp_default",
        "group": "MLP Anchor",
        "axis": "anchor",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    # Trục 1: hidden_dims
    {
        "run_id": "mlp_h128",
        "group": "Ablation: Hidden Dims",
        "axis": "hidden_dims",
        "model_type": "mlp",
        "hidden_dims": (128,),
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    {
        "run_id": "mlp_h512_256_128",
        "group": "Ablation: Hidden Dims",
        "axis": "hidden_dims",
        "model_type": "mlp",
        "hidden_dims": (512, 256, 128),
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    # Trục 2: learning_rate
    {
        "run_id": "mlp_lr1e4",
        "group": "Ablation: Learning Rate",
        "axis": "learning_rate",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": 1e-4,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    {
        "run_id": "mlp_lr1e2",
        "group": "Ablation: Learning Rate",
        "axis": "learning_rate",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": 1e-2,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    # Trục 3: optimizer
    {
        "run_id": "mlp_opt_sgd",
        "group": "Ablation: Optimizer",
        "axis": "optimizer",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "sgd",
        "lr": _DEFAULT_LR,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    {
        "run_id": "mlp_opt_adamw",
        "group": "Ablation: Optimizer",
        "axis": "optimizer",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adamw",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 1e-2,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    # Trục 4: batch_size
    {
        "run_id": "mlp_bs32",
        "group": "Ablation: Batch Size",
        "axis": "batch_size",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": 32,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    {
        "run_id": "mlp_bs256",
        "group": "Ablation: Batch Size",
        "axis": "batch_size",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": 256,
        "epochs": _DEFAULT_EPOCHS,
        "is_baseline": False,
    },
    # Trục 5: epoch
    {
        "run_id": "mlp_ep10",
        "group": "Ablation: Epoch",
        "axis": "epoch",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": 10,
        "is_baseline": False,
    },
    {
        "run_id": "mlp_ep30",
        "group": "Ablation: Epoch",
        "axis": "epoch",
        "model_type": "mlp",
        "hidden_dims": _DEFAULT_HIDDEN_DIMS,
        "dropout": _DEFAULT_DROPOUT,
        "optimizer": "adam",
        "lr": _DEFAULT_LR,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": 30,
        "is_baseline": False,
    },
)

CONFIG = VibeConfig(
    base_module=base_config,
    strict=True,
    BATCH_SIZE=_DEFAULT_BATCH_SIZE,
    LR=_DEFAULT_LR,
    EPOCH=_DEFAULT_EPOCHS,
    WOKER=0,
    extras={
        "IMAGE_SHAPE": (1, 28, 28),
        "INPUT_DIM": 784,
        "NUM_CLASSES": 10,
        "CLASS_NAMES": (
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ),
        "TRAIN_SIZE": 55_000,
        "VAL_SIZE": 5_000,
        "TEST_SIZE": 10_000,
        "BEST_METRIC": "val_loss",
        "DEFAULT_HIDDEN_DIMS": _DEFAULT_HIDDEN_DIMS,
        "DEFAULT_DROPOUT": _DEFAULT_DROPOUT,
        "DEFAULT_OPTIMIZER": "adam",
        "DEFAULT_WEIGHT_DECAY": 0.0,
        "TARGET_MLP_VAL_ACC": 0.89,
        "TARGET_MLP_VAL_F1": 0.89,
        "STOP_ON_TARGET_DEFAULT": True,
        "SANITY_BATCH_SIZE": 32,
        "SANITY_OVERFIT_ACC": 0.95,
        "SANITY_MAX_STEPS": 200,
        "SAVE_LOAD_TOLERANCE": 1e-6,
        "LAB_DIR": _ROOT_PATH / "labs" / "lab1_v2",
        "EXPERIMENT_DIR": _EXPERIMENT_DIR,
        "EDA_DIR": _EXPERIMENT_DIR / "eda",
        "RUNS_DIR": _EXPERIMENT_DIR / "runs",
        "ABLATION_DIR": _EXPERIMENT_DIR / "ablation",
        "EVALUATION_DIR": _EXPERIMENT_DIR / "evaluation",
        "FIGURES_DIR": _EXPERIMENT_DIR / "figures",
        "EXPERIMENT_SPECS": _EXPERIMENT_SPECS,
    },
)

CONFIG.export_to_namespace(globals())


def get_run_spec(run_id: str) -> Dict[str, Any]:
    """Trả về bản sao cấu hình của một run theo `run_id`."""
    for spec in _EXPERIMENT_SPECS:
        if spec["run_id"] == run_id:
            return dict(spec)
    valid = ", ".join(spec["run_id"] for spec in _EXPERIMENT_SPECS)
    raise KeyError(f"Không tìm thấy run_id '{run_id}'. Danh sách hợp lệ: [{valid}]")
