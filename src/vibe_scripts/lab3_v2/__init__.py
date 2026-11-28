"""Reusable Hugging Face core modules cho Lab 3 v2 (Sentiment Pipeline & DistilBERT Fine-Tuning on IMDB)."""

from src.vibe_scripts.lab3_v2.analysis import (
    plot_before_after_comparison,
    plot_binary_confusion_matrix,
    plot_loss_curve,
)
from src.vibe_scripts.lab3_v2.data import (
    LABEL_NAMES,
    build_stratified_subset_indices,
    load_and_split_imdb,
    run_imdb_data_audit_and_eda,
    tokenize_dataset,
)
from src.vibe_scripts.lab3_v2.ex1_sentiment import (
    DEFAULT_EX1_SAMPLES,
    build_sentiment_pipeline,
    inspect_batch_padding,
    inspect_tokenization,
    run_ex1_demo,
)
from src.vibe_scripts.lab3_v2.ex2_finetune import (
    build_compute_metrics,
    build_training_arguments,
    count_parameters,
    predict_texts,
    run_ex2_final_test,
    run_ex2_training,
    validate_classifier_batch,
)

__all__ = [
    "DEFAULT_EX1_SAMPLES",
    "LABEL_NAMES",
    "build_compute_metrics",
    "build_sentiment_pipeline",
    "build_stratified_subset_indices",
    "build_training_arguments",
    "count_parameters",
    "inspect_batch_padding",
    "inspect_tokenization",
    "load_and_split_imdb",
    "plot_before_after_comparison",
    "plot_binary_confusion_matrix",
    "plot_loss_curve",
    "predict_texts",
    "run_ex1_demo",
    "run_ex2_final_test",
    "run_ex2_training",
    "run_imdb_data_audit_and_eda",
    "tokenize_dataset",
    "validate_classifier_batch",
]
