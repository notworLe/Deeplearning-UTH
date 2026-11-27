"""Cấu hình trung tâm cho Lab 2 v2 (Oxford-IIIT Pets: Pre-trained CNN Transfer & Fine-Tuning).

Kế thừa hằng số nền từ `src/config.py` qua `VibeConfig` (chế độ `strict=True`)
và khai báo toàn bộ siêu tham số, đường dẫn artifact và chuỗi 9 runs có kiểm soát.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import src.config as base_config
from src.vibe_scripts.config_override import VibeConfig

_ROOT_PATH: Path = base_config.ROOT_PATH
_RAW_DATA_ROOT: Path = _ROOT_PATH / "data" / "raw"
_EXPERIMENT_DIR: Path = _ROOT_PATH / "experiments" / "lab2_v2"

_DEFAULT_LR: float = 1e-3
_DEFAULT_LR_BACKBONE: float = 1e-4
_DEFAULT_WEIGHT_DECAY: float = 1e-4
_DEFAULT_BATCH_SIZE: int = 32
_DEFAULT_TRANSFER_EPOCHS: int = 15
_DEFAULT_SCRATCH_EPOCHS: int = 20

_EXPERIMENT_SPECS: Tuple[Dict[str, Any], ...] = (
    # Run 1: Baseline (ResNet18 train từ đầu — làm mốc đối chứng)
    {
        "run_id": "resnet18_scratch",
        "group": "Baseline (Scratch)",
        "axis": "baseline",
        "arch": "resnet18",
        "mode": "scratch",
        "pretrained": False,
        "optimizer": "adamw",
        "lr": _DEFAULT_LR,
        "lr_backbone": _DEFAULT_LR,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_SCRATCH_EPOCHS,
        "is_baseline": True,
    },
    # Run 2: Transfer Anchor (ResNet18 freeze backbone, train fc)
    {
        "run_id": "resnet18_transfer",
        "group": "Transfer Anchor",
        "axis": "transfer_mode",
        "arch": "resnet18",
        "mode": "transfer",
        "pretrained": True,
        "optimizer": "adamw",
        "lr": _DEFAULT_LR,
        "lr_backbone": 0.0,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_TRANSFER_EPOCHS,
        "is_baseline": False,
    },
    # Run 3: Fine-Tune (ResNet18 unfreeze layer4 + fc, differential LR)
    {
        "run_id": "resnet18_finetune",
        "group": "Fine-Tune (layer4)",
        "axis": "transfer_mode",
        "arch": "resnet18",
        "mode": "finetune",
        "pretrained": True,
        "optimizer": "adamw",
        "lr": _DEFAULT_LR,
        "lr_backbone": _DEFAULT_LR_BACKBONE,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_TRANSFER_EPOCHS,
        "is_baseline": False,
    },
    # Run 4: Architecture Comparison — DenseNet121 Transfer
    {
        "run_id": "densenet121_transfer",
        "group": "Architecture Comparison",
        "axis": "architecture",
        "arch": "densenet121",
        "mode": "transfer",
        "pretrained": True,
        "optimizer": "adamw",
        "lr": _DEFAULT_LR,
        "lr_backbone": 0.0,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_TRANSFER_EPOCHS,
        "is_baseline": False,
    },
    # Run 5: Architecture Comparison — VGG16 Transfer
    {
        "run_id": "vgg16_transfer",
        "group": "Architecture Comparison",
        "axis": "architecture",
        "arch": "vgg16",
        "mode": "transfer",
        "pretrained": True,
        "optimizer": "adamw",
        "lr": _DEFAULT_LR,
        "lr_backbone": 0.0,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_TRANSFER_EPOCHS,
        "is_baseline": False,
    },
    # Run 6–7: Ablation Learning Rate (quanh resnet18_transfer)
    {
        "run_id": "resnet18_transfer_lr1e4",
        "group": "Ablation: Learning Rate",
        "axis": "learning_rate",
        "arch": "resnet18",
        "mode": "transfer",
        "pretrained": True,
        "optimizer": "adamw",
        "lr": 1e-4,
        "lr_backbone": 0.0,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_TRANSFER_EPOCHS,
        "is_baseline": False,
    },
    {
        "run_id": "resnet18_transfer_lr5e3",
        "group": "Ablation: Learning Rate",
        "axis": "learning_rate",
        "arch": "resnet18",
        "mode": "transfer",
        "pretrained": True,
        "optimizer": "adamw",
        "lr": 5e-3,
        "lr_backbone": 0.0,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "epochs": _DEFAULT_TRANSFER_EPOCHS,
        "is_baseline": False,
    },
    # Run 8–9: Ablation Batch Size (quanh resnet18_transfer)
    {
        "run_id": "resnet18_transfer_bs16",
        "group": "Ablation: Batch Size",
        "axis": "batch_size",
        "arch": "resnet18",
        "mode": "transfer",
        "pretrained": True,
        "optimizer": "adamw",
        "lr": _DEFAULT_LR,
        "lr_backbone": 0.0,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": 16,
        "epochs": _DEFAULT_TRANSFER_EPOCHS,
        "is_baseline": False,
    },
    {
        "run_id": "resnet18_transfer_bs64",
        "group": "Ablation: Batch Size",
        "axis": "batch_size",
        "arch": "resnet18",
        "mode": "transfer",
        "pretrained": True,
        "optimizer": "adamw",
        "lr": _DEFAULT_LR,
        "lr_backbone": 0.0,
        "weight_decay": _DEFAULT_WEIGHT_DECAY,
        "batch_size": 64,
        "epochs": _DEFAULT_TRANSFER_EPOCHS,
        "is_baseline": False,
    },
)

CONFIG = VibeConfig(
    SEED=248,
    BATCH_SIZE=_DEFAULT_BATCH_SIZE,
    LR=_DEFAULT_LR,
    EPOCH=_DEFAULT_TRANSFER_EPOCHS,
    WOKER=0,  # An toàn với Windows multiprocessing
    extras={
        "PROJECT_NAME": "lab2_v2",
        "DATASET_NAME": "OxfordIIITPet",
        "RAW_DATA_ROOT": _RAW_DATA_ROOT,
        "RAW_PET_DIR": _RAW_DATA_ROOT / "oxford-iiit-pet",
        "NUM_CLASSES": 37,
        "IMAGE_SIZE": 224,
        "RESIZE_SIZE": 256,
        "INPUT_SHAPE": (3, 224, 224),
        "IMAGENET_MEAN": (0.485, 0.456, 0.406),
        "IMAGENET_STD": (0.229, 0.224, 0.225),
        "CROP_SCALE": (0.75, 1.0),
        "HFLIP_PROB": 0.5,
        "VAL_RATIO_OF_TEST": 0.35,  # Train=trainval, Val=35% test, Test=65% test (stratified)
        "DEFAULT_WEIGHT_DECAY": _DEFAULT_WEIGHT_DECAY,
        "DEFAULT_LR_BACKBONE": _DEFAULT_LR_BACKBONE,
        "SAVE_LOAD_TOLERANCE": 1e-6,
        "USE_AMP_DEFAULT": True,
        "EXPERIMENT_SPECS": _EXPERIMENT_SPECS,
        "LAB_DIR": _ROOT_PATH / "labs" / "lab2_v2",
        "EXPERIMENT_DIR": _EXPERIMENT_DIR,
        "EDA_DIR": _EXPERIMENT_DIR / "eda",
        "SPLITS_DIR": _EXPERIMENT_DIR / "splits",
        "SPLIT_FILE": _EXPERIMENT_DIR / "splits" / "split_indices.json",
        "RUNS_DIR": _EXPERIMENT_DIR / "runs",
        "TB_LOGS_DIR": _EXPERIMENT_DIR / "tb_logs",
        "ABLATION_DIR": _EXPERIMENT_DIR / "ablation",
        "EVALUATION_DIR": _EXPERIMENT_DIR / "evaluation",
        "FIGURES_DIR": _EXPERIMENT_DIR / "figures",
    },
)

CONFIG.export_to_namespace(globals())


def get_run_spec(run_id: str) -> Dict[str, Any]:
    """Lấy bản sao cấu hình của 1 run theo `run_id`; ném `KeyError` nếu không tồn tại."""
    for spec in _EXPERIMENT_SPECS:
        if spec["run_id"] == run_id:
            return dict(spec)
    valid = [spec["run_id"] for spec in _EXPERIMENT_SPECS]
    raise KeyError(f"Không tìm thấy run_id='{run_id}'. Các run_id hợp lệ: {valid}")
