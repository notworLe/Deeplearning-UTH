"""Cấu hình riêng cho Lab 1 (Fashion-MNIST), kế thừa và override `src/config.py`."""

from src.vibe_scripts.config_override import VibeConfig

cfg = VibeConfig(
    # Override đúng tên biến tồn tại trong src/config.py
    LR=1e-3,
    EPOCH=20,
    # Tham số mở rộng riêng cho thực nghiệm Lab 1
    extras={
        "WEIGHT_DECAY": 0.0,
        "BEST_METRIC": "val_loss",
        "DATASET_NAME": "fashion_mnist",
        "NUM_CLASSES": 10,
        "INPUT_SHAPE": (1, 28, 28),
        "TRAIN_SIZE": 55000,
        "VAL_SIZE": 5000,
        "TEST_SIZE": 10000,
        "AUG_CONFIG": {
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 10.0,
            "translate": (0.08, 0.08),
        },
        "MODEL_NAMES": ("logistic", "mlp", "cnn"),
        "AUG_MODES": (False, True),
    },
)

cfg.add_extras(
    MANIFEST_PATH=cfg.DATA_PATH / "FashionMNIST" / "raw" / "manifest.json",
    LAB1_DIR=cfg.ROOT_PATH / "labs" / "lab1" / "vibe",
    EXPERIMENT_DIR=cfg.ROOT_PATH / "experiments" / "lab1",
    EVALUATION_DIR=cfg.ROOT_PATH / "experiments" / "lab1" / "evaluation",
    PREDICTIONS_DIR=cfg.ROOT_PATH / "experiments" / "lab1" / "predictions",
)

# Export toàn bộ hằng số ra cấp module để các pipeline import trực tiếp (`config.LR`, `config.SEED`, ...)
cfg.export_to_namespace(globals())
