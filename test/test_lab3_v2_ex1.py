"""Offline Smoke & Normal Tests cho Lab 3 v2 — Exercise 1 (Tokenizer Anatomy & Sentiment Pipeline)."""

import json
from pathlib import Path

import nbformat
import pytest
from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

import labs.lab3_v2.config as cfg
from src.vibe_scripts.lab3_v2.ex1_sentiment import (
    inspect_batch_padding,
    inspect_tokenization,
    run_ex1_demo,
)


def _build_offline_tokenizer_and_model(tmp_path: Path):
    """Tạo WordPiece tokenizer cục bộ và tiny DistilBERT 2 lớp hoàn toàn offline (không gọi mạng)."""
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "i",
        "is",
        "love",
        "loved",
        "this",
        "movie",
        "great",
        "terrible",
        "waste",
        "not",
        "bad",
        "film",
        "acting",
        "boring",
        "un",
        "##bel",
        "##ie",
        "##vable",
        "awesome",
        "!",
        ".",
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
    return tokenizer, model


@pytest.mark.smoke
def test_config_and_ex1_tokenizer_and_pipeline_smoke(tmp_path: Path) -> None:
    """Smoke test: config, WordPiece tokenizer anatomy, batch padding & HF text-classification pipeline offline."""
    assert cfg.TRAIN_SUBSET_SIZE == 4000
    assert cfg.VAL_SUBSET_SIZE == 1000
    assert cfg.TEST_SUBSET_SIZE == 1000
    assert cfg.MAX_LENGTH == 256
    assert cfg.NUM_LABELS == 2
    assert cfg.SEED == 248

    tokenizer, model = _build_offline_tokenizer_and_model(tmp_path)

    tok_info = inspect_tokenization("This movie is unbelievable!", tokenizer=tokenizer, max_length=32)
    assert tok_info["num_tokens"] == len(tok_info["input_ids"]) == len(tok_info["attention_mask"])
    assert tok_info["tokens"][0] == "[CLS]"
    assert tok_info["tokens"][-1] == "[SEP]"
    assert "##bel" in tok_info["subword_pieces"]
    assert "unbelievable" in tok_info["decoded_clean"]

    batch_info = inspect_batch_padding(["Great movie!", "I loved this awesome movie."], tokenizer=tokenizer)
    assert batch_info["input_ids_shape"][0] == 2
    assert batch_info["input_ids_shape"] == batch_info["attention_mask_shape"]
    assert batch_info["rows"][0]["pad_tokens"] > 0

    out_dir = tmp_path / "ex1"
    samples = [
        {"category": "pos", "expected": "POSITIVE", "text": "I loved this great movie!"},
        {"category": "neg", "expected": "NEGATIVE", "text": "This boring film was a terrible waste."},
    ]
    result = run_ex1_demo(
        model_name="offline-tiny-distilbert",
        output_dir=out_dir,
        samples=samples,
        sample_sentence="This movie is unbelievable!",
        device=-1,
        model=model,
        tokenizer=tokenizer,
    )
    assert result["num_samples"] == 2
    assert (out_dir / "ex1_results.json").exists()
    persisted = json.loads((out_dir / "ex1_results.json").read_text(encoding="utf-8"))
    assert len(persisted["predictions"]) == 2
    assert all(p["predicted_label"] in {"NEGATIVE", "POSITIVE"} for p in persisted["predictions"])
    assert all(0.0 <= p["score"] <= 1.0 for p in persisted["predictions"])


def test_ex1_notebook_schema_valid() -> None:
    """Normal test: xác thực schema nbformat v4 của ex1_sentiment.ipynb."""
    nb_path = cfg.LAB_DIR / "ex1_sentiment.ipynb"
    assert nb_path.exists()
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 6
