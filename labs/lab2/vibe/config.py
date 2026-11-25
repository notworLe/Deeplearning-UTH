"""Cấu hình riêng cho Lab 2 (Oxford-IIIT Pets — Baseline / Transfer / Fine-tune), kế thừa `src/config.py`."""

from src.vibe_scripts.config_override import VibeConfig

cfg = VibeConfig(
    # Override đúng tên biến tồn tại trong src/config.py
    LR=1e-3,
    EPOCH=30,
    BATCH_SIZE=32,
    SEED=248,
    # Tham số mở rộng riêng cho thực nghiệm Lab 2
    extras={
        "DATASET_NAME": "OxfordIIITPet",
        "NUM_CLASSES": 37,
        "IMAGE_SIZE": 224,
        "RESIZE_SIZE": 256,
        "VAL_RATIO": 0.1,
        "BEST_METRIC": "val_loss",
        "LR_HEAD": 1e-3,
        "LR_BACKBONE": 1e-4,
        "MULTI_SEEDS": (248, 42, 2026),
        "NOISE_THRESHOLD_PCT": 1.0,
        "AUG_CONFIG": {
            "crop_scale": (0.7, 1.0),
            "horizontal_flip_p": 0.5,
        },
        "RUN_NAMES": ("resnet18_baseline", "resnet18_transfer", "resnet18_finetune"),
    },
)

cfg.add_extras(
    LAB2_DIR=cfg.ROOT_PATH / "labs" / "lab2" / "vibe",
    EXPERIMENT_DIR=cfg.ROOT_PATH / "experiments" / "lab2",
    EDA_DIR=cfg.ROOT_PATH / "experiments" / "lab2" / "eda",
    EVALUATION_DIR=cfg.ROOT_PATH / "experiments" / "lab2" / "evaluation",
    PREDICTIONS_DIR=cfg.ROOT_PATH / "experiments" / "lab2" / "predictions",
)

# Export toàn bộ hằng số ra cấp module (`config.LR`, `config.SEED`, ...)
cfg.export_to_namespace(globals())
