"""Exercise 2: Fine-Tuning DistilBERT bằng Hugging Face `Trainer` (chuẩn `transformers v5`), Val/Test Eval & Inference."""

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.vibe_scripts.lab3_v2.analysis import (
    plot_before_after_comparison,
    plot_binary_confusion_matrix,
    plot_loss_curve,
)
from src.vibe_scripts.lab3_v2.data import LABEL_NAMES, tokenize_dataset


def count_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    """Đếm tổng tham số và tham số trainable của mô hình."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "trainable_pct": round((trainable / max(total, 1)) * 100.0, 2),
    }


def validate_classifier_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int = 256,
) -> None:
    """Kiểm chứng Fail-Fast Tensor Contract `(B, L)` int64 với `1 <= L <= max_length`."""
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors (B, L), got input_ids={tuple(input_ids.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"Shape mismatch: input_ids={tuple(input_ids.shape)} vs attention_mask={tuple(attention_mask.shape)}"
        )
    if input_ids.dtype != torch.int64:
        raise ValueError(f"input_ids phải có dtype=torch.int64, nhận {input_ids.dtype}")
    seq_len = int(input_ids.shape[1])
    if not (1 <= seq_len <= max_length):
        raise ValueError(f"Sequence length L={seq_len} nằm ngoài khoảng hợp lệ [1, {max_length}]")


def build_compute_metrics(use_hf_evaluate: bool = True) -> Callable[[Any], Dict[str, float]]:
    """Tạo hàm `compute_metrics` cho HF `Trainer` (ưu tiên `evaluate`, tự fallback `sklearn` khi offline)."""
    acc_metric = None
    f1_metric = None
    if use_hf_evaluate:
        try:
            import evaluate

            acc_metric = evaluate.load("accuracy")
            f1_metric = evaluate.load("f1")
        except Exception:
            acc_metric = None
            f1_metric = None

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        if acc_metric is not None and f1_metric is not None:
            acc = float(acc_metric.compute(predictions=preds, references=labels)["accuracy"])
            f1 = float(f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"])
        else:
            acc = float(accuracy_score(labels, preds))
            f1 = float(f1_score(labels, preds, average="binary", zero_division=0))
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        return {
            "accuracy": acc,
            "f1": f1,
            "macro_f1": macro_f1,
        }

    return compute_metrics


def build_training_arguments(
    output_dir: Path,
    epochs: int = 3,
    lr: float = 2e-5,
    train_bs: int = 16,
    eval_bs: int = 32,
    weight_decay: float = 0.01,
    seed: int = 248,
    use_fp16: bool = True,
    report_to: Optional[Sequence[str]] = None,
    logging_steps: int = 25,
) -> TrainingArguments:
    """Tạo `TrainingArguments` chuẩn `transformers v5` (`eval_strategy`, không dùng `logging_dir`)."""
    active_report_to = list(report_to) if report_to is not None else ["none"]
    fp16_enabled = bool(use_fp16 and torch.cuda.is_available())
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(epochs),
        learning_rate=float(lr),
        per_device_train_batch_size=int(train_bs),
        per_device_eval_batch_size=int(eval_bs),
        weight_decay=float(weight_decay),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=max(1, int(logging_steps)),
        fp16=fp16_enabled,
        seed=int(seed),
        data_seed=int(seed),
        report_to=active_report_to,
        dataloader_num_workers=0,
    )


def run_ex2_training(
    dataset_dict: DatasetDict,
    tokenizer: Any,
    model: torch.nn.Module,
    run_dir: Path,
    max_length: int = 256,
    epochs: int = 3,
    lr: float = 2e-5,
    train_bs: int = 16,
    eval_bs: int = 32,
    weight_decay: float = 0.01,
    seed: int = 248,
    use_fp16: bool = True,
    enable_tensorboard: bool = True,
    use_hf_evaluate: bool = True,
) -> Dict[str, Any]:
    """Chạy trọn bộ Ex2 (Baseline Val trước fine-tune → `Trainer.train()` → Val sau fine-tune).

    Tuyệt đối KHÔNG đánh giá tập `test` trong hàm này (anti-leakage theo MODEL_PIPELINE.md v1.4.0).
    """
    set_seed(seed)
    run_path = Path(run_dir)
    monitor_dir = run_path / "monitor"
    val_eval_dir = run_path / "val_eval"
    weights_dir = run_path / "weights"
    best_model_dir = weights_dir / "best_model"
    tb_dir = monitor_dir / "tb_logs"
    for d in (monitor_dir, val_eval_dir, weights_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Chỉ tokenize train & validation trong giai đoạn training
    train_val_ds = DatasetDict(
        {
            "train": dataset_dict["train"],
            "validation": dataset_dict["validation"],
        }
    )
    tokenized = tokenize_dataset(train_val_ds, tokenizer=tokenizer, max_length=max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Kiểm chứng Tensor Contract trên 1 mini-batch trước khi train
    sample_batch = data_collator([tokenized["train"][i] for i in range(min(4, len(tokenized["train"])))])
    validate_classifier_batch(
        sample_batch["input_ids"], sample_batch["attention_mask"], max_length=max_length
    )

    tb_active = False
    report_to: List[str] = ["none"]
    if enable_tensorboard:
        try:
            import tensorboard  # noqa: F401

            tb_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)
            report_to = ["tensorboard"]
            tb_active = True
        except Exception:
            report_to = ["none"]
            tb_active = False

    steps_per_ep = max(1, len(tokenized["train"]) // max(1, train_bs))
    log_steps = max(1, min(25, steps_per_ep // 4 or 1))
    training_args = build_training_arguments(
        output_dir=weights_dir,
        epochs=epochs,
        lr=lr,
        train_bs=train_bs,
        eval_bs=eval_bs,
        weight_decay=weight_decay,
        seed=seed,
        use_fp16=use_fp16,
        report_to=report_to,
        logging_steps=log_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(use_hf_evaluate=use_hf_evaluate),
    )

    # 1. Đo Baseline TRƯỚC fine-tune trên Validation
    baseline_val = trainer.evaluate(eval_dataset=tokenized["validation"])
    baseline_clean = {k: float(v) for k, v in baseline_val.items() if isinstance(v, (int, float))}
    (val_eval_dir / "baseline_before_finetune.json").write_text(
        json.dumps(baseline_clean, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Reset log_history trước khi train để loss curve sạch (không lẫn bước baseline eval)
    trainer.state.log_history.clear()

    # 2. Chạy Fine-Tune với HF Trainer
    t0 = time.perf_counter()
    train_output = trainer.train()
    train_time_sec = round(time.perf_counter() - t0, 2)

    # 3. Đo Validation SAU fine-tune (đã nạp best checkpoint theo min eval_loss)
    after_val = trainer.evaluate(eval_dataset=tokenized["validation"])
    after_clean = {k: float(v) for k, v in after_val.items() if isinstance(v, (int, float))}
    (val_eval_dir / "after_finetune_val.json").write_text(
        json.dumps(after_clean, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Lưu best model + tokenizer để eval_ex2.py & inference sử dụng độc lập
    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    log_history = list(trainer.state.log_history)
    (monitor_dir / "trainer_log_history.json").write_text(
        json.dumps(log_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    loss_plot = plot_loss_curve(log_history, output_path=monitor_dir / "loss_curve.png")
    cmp_plot = plot_before_after_comparison(
        baseline_clean, after_clean, output_path=val_eval_dir / "val_comparison.png"
    )

    param_counts = count_parameters(trainer.model)
    summary = {
        "seed": int(seed),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "fp16": bool(training_args.fp16),
        "tensorboard_active": tb_active,
        "train_samples": len(tokenized["train"]),
        "val_samples": len(tokenized["validation"]),
        "max_length": int(max_length),
        "epochs": float(epochs),
        "learning_rate": float(lr),
        "train_batch_size": int(train_bs),
        "eval_batch_size": int(eval_bs),
        "weight_decay": float(weight_decay),
        "param_counts": param_counts,
        "train_time_sec": train_time_sec,
        "global_step": int(train_output.global_step),
        "train_loss_final": float(train_output.training_loss),
        "best_metric_eval_loss": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
        "best_model_checkpoint": str(trainer.state.best_model_checkpoint),
        "best_model_dir": str(best_model_dir),
        "baseline_before_finetune": baseline_clean,
        "after_finetune_val": after_clean,
        "val_accuracy_gain_pp": round(
            (after_clean.get("eval_accuracy", 0.0) - baseline_clean.get("eval_accuracy", 0.0)) * 100.0, 2
        ),
        "val_f1_gain_pp": round(
            (after_clean.get("eval_f1", 0.0) - baseline_clean.get("eval_f1", 0.0)) * 100.0, 2
        ),
        "artifacts": {
            "loss_curve": str(loss_plot),
            "val_comparison": str(cmp_plot),
            "log_history": str(monitor_dir / "trainer_log_history.json"),
        },
    }
    (monitor_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


@torch.inference_mode()
def run_ex2_final_test(
    test_dataset: Dataset,
    model_dir: Path,
    run_dir: Path,
    max_length: int = 256,
    eval_bs: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Đánh giá tập Test khóa kín duy nhất 1 lần cuối, vẽ Confusion Matrix & trích xuất mẫu dự đoán sai."""
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_dir)
    test_eval_dir = Path(run_dir) / "test_eval"
    test_eval_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(dev)
    model.eval()

    texts = list(test_dataset["text"])
    y_true = np.asarray(test_dataset["label"], dtype=np.int64)

    # Đo độ dài token gốc để phân tích slice truncated vs non-truncated
    orig_tok_lens: List[int] = []
    for start in range(0, len(texts), 256):
        enc_full = tokenizer(
            texts[start : start + 256],
            truncation=False,
            padding=False,
            add_special_tokens=True,
            verbose=False,
        )
        orig_tok_lens.extend(len(ids) for ids in enc_full["input_ids"])

    tok_test = test_dataset.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing held-out Test",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(tok_test, batch_size=eval_bs, shuffle=False, collate_fn=collator)

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for batch in loader:
        labels_b = batch.pop("labels").to(dev)
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        validate_classifier_batch(input_ids, attention_mask, max_length=max_length)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        total_loss += float(criterion(logits, labels_b).item())
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    n_total = len(y_true)
    avg_loss = total_loss / max(n_total, 1)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1_per_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    binary_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    cm_path = plot_binary_confusion_matrix(
        cm=cm,
        class_names=LABEL_NAMES,
        title=f"DistilBERT IMDB — Held-Out Test Confusion Matrix (Acc={acc*100:.2f}%)",
        output_path=test_eval_dir / "confusion_matrix.png",
    )

    # Phân tích lỗi & Slice theo truncation (> max_length)
    wrong_mask = y_true != y_pred
    trunc_mask = np.asarray([n > max_length for n in orig_tok_lens], dtype=bool)
    trunc_slice = {
        "max_length": int(max_length),
        "truncated_samples": int(trunc_mask.sum()),
        "truncated_errors": int((wrong_mask & trunc_mask).sum()),
        "truncated_error_rate_pct": round(
            (int((wrong_mask & trunc_mask).sum()) / max(int(trunc_mask.sum()), 1)) * 100.0, 2
        ),
        "non_truncated_samples": int((~trunc_mask).sum()),
        "non_truncated_errors": int((wrong_mask & (~trunc_mask)).sum()),
        "non_truncated_error_rate_pct": round(
            (int((wrong_mask & (~trunc_mask)).sum()) / max(int((~trunc_mask).sum()), 1)) * 100.0, 2
        ),
    }

    wrong_indices = np.flatnonzero(wrong_mask)
    error_records: List[Dict[str, Any]] = []
    for idx in wrong_indices:
        i = int(idx)
        t_lbl = int(y_true[i])
        p_lbl = int(y_pred[i])
        error_records.append(
            {
                "test_idx": i,
                "true_label": LABEL_NAMES[t_lbl],
                "pred_label": LABEL_NAMES[p_lbl],
                "pred_confidence": round(float(y_probs[i, p_lbl]), 4),
                "true_probability": round(float(y_probs[i, t_lbl]), 4),
                "orig_token_len": int(orig_tok_lens[i]),
                "was_truncated": bool(orig_tok_lens[i] > max_length),
                "text_snippet": str(texts[i])[:420].replace("\n", " "),
                "text_tail": str(texts[i])[-220:].replace("\n", " ") if len(str(texts[i])) > 420 else "",
            }
        )

    hi_conf_errors = sorted(error_records, key=lambda r: r["pred_confidence"], reverse=True)[:10]
    lo_conf_errors = sorted(error_records, key=lambda r: r["pred_confidence"])[:10]

    error_payload = {
        "total_test_samples": n_total,
        "total_errors": int(len(wrong_indices)),
        "error_rate_pct": round((len(wrong_indices) / max(n_total, 1)) * 100.0, 2),
        "false_positives_neg_as_pos": int(cm[0, 1]),
        "false_negatives_pos_as_neg": int(cm[1, 0]),
        "truncation_slice": trunc_slice,
        "high_confidence_errors": hi_conf_errors,
        "low_confidence_boundary_errors": lo_conf_errors,
    }
    (test_eval_dir / "error_examples.json").write_text(
        json.dumps(error_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics = {
        "split": "test",
        "num_samples": n_total,
        "test_loss": round(float(avg_loss), 4),
        "test_accuracy": round(acc, 4),
        "test_f1": round(binary_f1, 4),
        "test_macro_f1": round(macro_f1, 4),
        "per_class": {
            LABEL_NAMES[c]: {
                "precision": round(float(prec[c]), 4),
                "recall": round(float(rec[c]), 4),
                "f1": round(float(f1_per_cls[c]), 4),
                "support": int(support[c]),
            }
            for c in (0, 1)
        },
        "confusion_matrix": cm.tolist(),
        "truncation_slice": trunc_slice,
        "artifacts": {
            "confusion_matrix": str(cm_path),
            "error_examples": str(test_eval_dir / "error_examples.json"),
        },
    }
    (test_eval_dir / "final_test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


@torch.inference_mode()
def predict_texts(
    texts: Sequence[str],
    model_dir: Path,
    max_length: int = 256,
    device: Optional[Union[str, torch.device]] = None,
) -> List[Dict[str, Any]]:
    """Suy luận sạch trên các câu mới bằng checkpoint đã fine-tune (`@torch.inference_mode()`)."""
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(dev)
    model.eval()

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(dev)
    attention_mask = encoded["attention_mask"].to(dev)
    validate_classifier_batch(input_ids, attention_mask, max_length=max_length)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = logits.argmax(dim=-1).cpu().numpy()

    results: List[Dict[str, Any]] = []
    for i, text in enumerate(texts):
        p_id = int(preds[i])
        results.append(
            {
                "text": text,
                "predicted_id": p_id,
                "predicted_label": LABEL_NAMES[p_id],
                "confidence": round(float(probs[i, p_id]), 4),
                "probabilities": {
                    "Negative": round(float(probs[i, 0]), 4),
                    "Positive": round(float(probs[i, 1]), 4),
                },
                "token_count": int(attention_mask[i].sum().item()),
                "requires_grad": bool(logits.requires_grad),
            }
        )
    return results
