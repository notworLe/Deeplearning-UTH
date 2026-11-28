"""Cấu hình trung tâm cho Lab 3 v2 (Hugging Face Sentiment Pipeline & DistilBERT Fine-Tuning on IMDB).

Kế thừa hằng số nền từ `src/config.py` qua `VibeConfig` và định nghĩa toàn bộ siêu tham số,
đường dẫn artifact theo chuẩn `MODEL_PIPELINE.md v1.4.0`.
"""

from pathlib import Path
from typing import Tuple

import src.config as base_config
from src.vibe_scripts.config_override import VibeConfig

_ROOT_PATH: Path = base_config.ROOT_PATH
_EXPERIMENT_DIR: Path = _ROOT_PATH / "experiments" / "lab3_v2"
_RUN_NAME: str = "distilbert_imdb"
_RUN_DIR: Path = _EXPERIMENT_DIR / "runs" / _RUN_NAME

_DEFAULT_LR: float = 2e-5
_DEFAULT_TRAIN_BATCH_SIZE: int = 16
_DEFAULT_EVAL_BATCH_SIZE: int = 32
_DEFAULT_EPOCHS: int = 3
_DEFAULT_MAX_LENGTH: int = 256
_DEFAULT_WEIGHT_DECAY: float = 0.01

CONFIG = VibeConfig(
    SEED=248,
    BATCH_SIZE=_DEFAULT_TRAIN_BATCH_SIZE,
    LR=_DEFAULT_LR,
    EPOCH=_DEFAULT_EPOCHS,
    WOKER=0,  # An toàn trên Windows
    MAX_LENGTH=_DEFAULT_MAX_LENGTH,
    extras={
        "PROJECT_NAME": "lab3_v2",
        "EX1_MODEL_NAME": "distilbert-base-uncased-finetuned-sst-2-english",
        "EX2_MODEL_NAME": "distilbert-base-uncased",
        "DATASET_NAME": "stanfordnlp/imdb",
        "NUM_LABELS": 2,
        "LABEL_NAMES": ("Negative", "Positive"),
        "ID2LABEL": {0: "NEGATIVE", 1: "POSITIVE"},
        "LABEL2ID": {"NEGATIVE": 0, "POSITIVE": 1},
        # Stratified subset 50/50 đã Kỹ sư chốt (A1)
        "TRAIN_SUBSET_SIZE": 4000,
        "VAL_SUBSET_SIZE": 1000,
        "TEST_SUBSET_SIZE": 1000,
        "EVAL_BATCH_SIZE": _DEFAULT_EVAL_BATCH_SIZE,
        "WEIGHT_DECAY": _DEFAULT_WEIGHT_DECAY,
        "USE_FP16_DEFAULT": True,
        "RUN_NAME": _RUN_NAME,
        "LAB_DIR": _ROOT_PATH / "labs" / "lab3_v2",
        "EXPERIMENT_DIR": _EXPERIMENT_DIR,
        "EX1_DIR": _EXPERIMENT_DIR / "ex1",
        "EDA_DIR": _EXPERIMENT_DIR / "eda",
        "SPLITS_DIR": _EXPERIMENT_DIR / "splits",
        "SPLIT_FILE": _EXPERIMENT_DIR / "splits" / "split_indices.json",
        "RUNS_DIR": _EXPERIMENT_DIR / "runs",
        "RUN_DIR": _RUN_DIR,
        "MONITOR_DIR": _RUN_DIR / "monitor",
        "TB_LOGS_DIR": _RUN_DIR / "monitor" / "tb_logs",
        "VAL_EVAL_DIR": _RUN_DIR / "val_eval",
        "TEST_EVAL_DIR": _RUN_DIR / "test_eval",
        "WEIGHTS_DIR": _RUN_DIR / "weights",
        "BEST_MODEL_DIR": _RUN_DIR / "weights" / "best_model",
    },
)

CONFIG.export_to_namespace(globals())

__all__: Tuple[str, ...] = tuple(k for k in globals() if k.isupper())
