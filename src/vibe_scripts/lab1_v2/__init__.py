"""Các module lõi tái sử dụng cho Lab 1 v2 (FashionMNIST: Logistic Regression vs MLP)."""

from src.vibe_scripts.lab1_v2.data import (
    DEFAULT_CLASS_NAMES,
    FashionMNISTBundle,
    TensorImageDataset,
    build_dataloaders_from_tensors,
    compute_eda_summary,
    export_eda_artifacts,
    prepare_fashion_mnist,
    run_data_cleaning_audit,
    run_data_collection_audit,
    stratified_train_val_indices,
    verify_fashion_mnist_manifest,
)
from src.vibe_scripts.lab1_v2.evaluator import (
    evaluate_final_test,
    evaluate_model_on_loader,
    evaluate_run_dir,
    verify_save_load_parity,
)
from src.vibe_scripts.lab1_v2.inference import FashionPredictor, load_checkpoint
from src.vibe_scripts.lab1_v2.logistic import LogisticRegressionModel, count_parameters
from src.vibe_scripts.lab1_v2.mlp import MLPModel, build_model
from src.vibe_scripts.lab1_v2.trainer import (
    atomic_torch_save,
    build_optimizer,
    run_experiment_suite,
    run_sanity_checks,
    set_global_seed,
    train_one_epoch,
    train_single_run,
    validate_one_epoch,
)
from src.vibe_scripts.lab1_v2.transforms import (
    NormStats,
    build_eval_transform,
    build_train_transform,
    compute_train_norm_stats,
    denormalize_tensor,
    normalize_batch,
    preprocess_image,
    validate_image_tensor,
)

__all__ = [
    "DEFAULT_CLASS_NAMES",
    "FashionMNISTBundle",
    "FashionPredictor",
    "LogisticRegressionModel",
    "MLPModel",
    "NormStats",
    "TensorImageDataset",
    "atomic_torch_save",
    "build_dataloaders_from_tensors",
    "build_eval_transform",
    "build_model",
    "build_optimizer",
    "build_train_transform",
    "compute_eda_summary",
    "compute_train_norm_stats",
    "count_parameters",
    "denormalize_tensor",
    "evaluate_final_test",
    "evaluate_model_on_loader",
    "evaluate_run_dir",
    "export_eda_artifacts",
    "load_checkpoint",
    "normalize_batch",
    "prepare_fashion_mnist",
    "preprocess_image",
    "run_data_cleaning_audit",
    "run_data_collection_audit",
    "run_experiment_suite",
    "run_sanity_checks",
    "set_global_seed",
    "stratified_train_val_indices",
    "train_one_epoch",
    "train_single_run",
    "validate_image_tensor",
    "validate_one_epoch",
    "verify_fashion_mnist_manifest",
    "verify_save_load_parity",
]
