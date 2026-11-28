"""Offline Smoke & Normal Tests cho Lab 3 v2 — Exercise 2 (IMDB Split, EDA, HF Trainer v5, Test & Inference)."""

import json
from pathlib import Path

import nbformat
import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

import labs.lab3_v2.config as cfg
from src.vibe_scripts.lab3_v2.data import (
    load_and_split_imdb,
    run_imdb_data_audit_and_eda,
    tokenize_dataset,
)
from src.vibe_scripts.lab3_v2.ex2_finetune import (
    count_parameters,
    predict_texts,
    run_ex2_final_test,
    run_ex2_training,
    validate_classifier_batch,
)


def _build_offline_assets(tmp_path: Path):
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "good",
        "great",
        "awesome",
        "movie",
        "story",
        "bad",
        "boring",
        "awful",
        "terrible",
        "waste",
        "film",
        "very",
        "really",
        "br",
        "<",
        ">",
        "/",
    ]
    tokenizer = BertTokenizer(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True)

    tiny_cfg = DistilBertConfig(
        vocab_size=len(vocab),
        max_position_embeddings=64,
        n_layers=1,
        n_heads=2,
        dim=32,
        hidden_dim=64,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )
    model = DistilBertForSequenceClassification(tiny_cfg)

    train_texts, train_labels = [], []
    for i in range(40):
        if i % 2 == 0:
            # Negative (một số review dài > 16 token để kiểm tra truncation slice)
            extra = " bad boring awful" * (6 if i % 4 == 0 else 1)
            train_texts.append(f"bad boring film terrible waste <br />{extra}")
            train_labels.append(0)
        else:
            extra = " good great awesome" * (6 if i % 4 == 1 else 1)
            train_texts.append(f"good great movie awesome story{extra}")
            train_labels.append(1)

    test_texts, test_labels = [], []
    for i in range(20):
        if i % 2 == 0:
            extra = " terrible waste" * (7 if i < 10 else 1)
            test_texts.append(f"awful boring film{extra}")
            test_labels.append(0)
        else:
            extra = " great awesome" * (7 if i < 10 else 1)
            test_texts.append(f"very good movie story{extra}")
            test_labels.append(1)

    raw_ds = DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
            "test": Dataset.from_dict({"text": test_texts, "label": test_labels}),
        }
    )
    return tokenizer, model, raw_ds


@pytest.mark.smoke
def test_ex2_tiny_trainer_smoke(tmp_path: Path) -> None:
    """Smoke test: chạy 1 epoch HF Trainer chuẩn v5 (vài batch offline) + lưu đủ artifacts Val/Monitor/Weights."""
    tokenizer, model, raw_ds = _build_offline_assets(tmp_path)
    counts = count_parameters(model)
    assert counts["total"] > 0 and counts["trainable"] == counts["total"]

    splits, _ = load_and_split_imdb(
        train_size=16,
        val_size=8,
        test_size=10,
        seed=248,
        split_file=tmp_path / "split_indices.json",
        raw_dataset=raw_ds,
    )
    run_dir = tmp_path / "runs" / "tiny_distilbert"
    summary = run_ex2_training(
        dataset_dict=splits,
        tokenizer=tokenizer,
        model=model,
        run_dir=run_dir,
        max_length=24,
        epochs=1,
        lr=1e-3,
        train_bs=4,
        eval_bs=4,
        seed=248,
        use_fp16=False,
        enable_tensorboard=False,
        use_hf_evaluate=False,
    )
    assert summary["train_samples"] == 16
    assert summary["val_samples"] == 8
    assert (run_dir / "monitor" / "trainer_log_history.json").exists()
    assert (run_dir / "monitor" / "train_summary.json").exists()
    assert (run_dir / "monitor" / "loss_curve.png").exists()
    assert (run_dir / "val_eval" / "baseline_before_finetune.json").exists()
    assert (run_dir / "val_eval" / "after_finetune_val.json").exists()
    assert (run_dir / "val_eval" / "val_comparison.png").exists()
    assert (run_dir / "weights" / "best_model").exists()
    # Anti-leakage: training tuyệt đối không tạo test_eval/
    assert not (run_dir / "test_eval").exists()


def test_stratified_split_disjointness_determinism_and_eda(tmp_path: Path) -> None:
    """Normal test: kiểm chứng split cân bằng 50/50 không giao nhau, tái lập và EDA/Cleaning audit."""
    tokenizer, _, raw_ds = _build_offline_assets(tmp_path)
    split_file = tmp_path / "splits" / "split_indices.json"

    splits1, meta1 = load_and_split_imdb(
        train_size=24, val_size=10, test_size=12, seed=248, split_file=split_file, raw_dataset=raw_ds
    )
    splits2, meta2 = load_and_split_imdb(
        train_size=24, val_size=10, test_size=12, seed=248, split_file=split_file, raw_dataset=raw_ds
    )
    assert meta1["train_indices"] == meta2["train_indices"]
    assert meta1["val_indices"] == meta2["val_indices"]
    assert set(meta1["train_indices"]).isdisjoint(set(meta1["val_indices"]))
    assert sum(splits1["train"]["label"]) == 12
    assert sum(splits1["validation"]["label"]) == 5
    assert sum(splits1["test"]["label"]) == 6

    eda_dir = tmp_path / "eda"
    eda_out = run_imdb_data_audit_and_eda(
        dataset_dict=splits1,
        tokenizer=tokenizer,
        eda_dir=eda_dir,
        max_length=16,
        split_metadata=meta1,
    )
    assert (eda_dir / "collection_report.json").exists()
    assert (eda_dir / "cleaning_report.json").exists()
    assert (eda_dir / "eda_summary.json").exists()
    assert (eda_dir / "eda_class_distribution.png").exists()
    assert (eda_dir / "eda_token_length.png").exists()
    assert eda_out["cleaning_report"]["reviews_with_html_br_tag"]["train"] > 0
    assert eda_out["eda_summary"]["truncated_pct_above_max_length"]["train"] > 0.0

    tok = tokenize_dataset(splits1, tokenizer=tokenizer, max_length=16)
    assert all(len(row["input_ids"]) <= 16 for row in tok["train"])


def test_validate_classifier_batch_fail_fast() -> None:
    """Normal test: kiểm chứng Fail-Fast Tensor Contract `(B, L)` int64."""
    good_ids = torch.ones((2, 8), dtype=torch.int64)
    good_mask = torch.ones((2, 8), dtype=torch.int64)
    validate_classifier_batch(good_ids, good_mask, max_length=16)

    with pytest.raises(ValueError):
        validate_classifier_batch(torch.ones((8,), dtype=torch.int64), good_mask, max_length=16)
    with pytest.raises(ValueError):
        validate_classifier_batch(good_ids, torch.ones((2, 6), dtype=torch.int64), max_length=16)
    with pytest.raises(ValueError):
        validate_classifier_batch(good_ids.float(), good_mask, max_length=16)
    with pytest.raises(ValueError):
        validate_classifier_batch(torch.ones((2, 20), dtype=torch.int64), torch.ones((2, 20), dtype=torch.int64), max_length=16)


def test_final_test_error_analysis_live_inference_and_notebook(tmp_path: Path) -> None:
    """Normal test: end-to-end Train -> Final Test 1 lần -> Error Analysis -> Inference + notebook schema."""
    tokenizer, model, raw_ds = _build_offline_assets(tmp_path)
    splits, _ = load_and_split_imdb(
        train_size=16, val_size=8, test_size=12, seed=248, split_file=tmp_path / "split.json", raw_dataset=raw_ds
    )
    run_dir = tmp_path / "runs" / "tiny_e2e"
    run_ex2_training(
        dataset_dict=splits,
        tokenizer=tokenizer,
        model=model,
        run_dir=run_dir,
        max_length=16,
        epochs=1,
        lr=1e-3,
        train_bs=4,
        eval_bs=4,
        seed=248,
        use_fp16=False,
        enable_tensorboard=False,
        use_hf_evaluate=False,
    )
    best_dir = run_dir / "weights" / "best_model"

    test_metrics = run_ex2_final_test(
        test_dataset=splits["test"],
        model_dir=best_dir,
        run_dir=run_dir,
        max_length=16,
        eval_bs=4,
        device="cpu",
    )
    assert test_metrics["num_samples"] == 12
    assert (run_dir / "test_eval" / "final_test_metrics.json").exists()
    assert (run_dir / "test_eval" / "confusion_matrix.png").exists()
    assert (run_dir / "test_eval" / "error_examples.json").exists()
    err_data = json.loads((run_dir / "test_eval" / "error_examples.json").read_text(encoding="utf-8"))
    assert err_data["truncation_slice"]["truncated_samples"] > 0

    preds = predict_texts(["good great movie", "bad boring waste"], model_dir=best_dir, max_length=16, device="cpu")
    assert len(preds) == 2
    assert all(p["requires_grad"] is False for p in preds)
    assert all(p["predicted_label"] in {"Negative", "Positive"} for p in preds)

    nb_path = cfg.LAB_DIR / "ex2_finetune.ipynb"
    assert nb_path.exists()
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 8
