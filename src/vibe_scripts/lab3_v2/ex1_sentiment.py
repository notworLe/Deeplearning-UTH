"""Exercise 1: Giải phẫu WordPiece Tokenizer & Sentiment Analysis bằng Hugging Face `pipeline`."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

DEFAULT_SAMPLE_SENTENCE: str = (
    "Hugging Face transformers make fine-tuning unbelievable and surprisingly enjoyable!"
)

DEFAULT_EX1_SAMPLES: Tuple[Dict[str, str], ...] = (
    {
        "category": "positive_clear",
        "expected": "POSITIVE",
        "text": "I absolutely loved this movie! The acting was brilliant and the story kept me hooked from start to finish.",
    },
    {
        "category": "positive_clear",
        "expected": "POSITIVE",
        "text": "This library makes working with pretrained transformer models surprisingly intuitive and fast.",
    },
    {
        "category": "negative_clear",
        "expected": "NEGATIVE",
        "text": "What a complete waste of two hours. The plot made zero sense and the dialogue was awful.",
    },
    {
        "category": "negative_clear",
        "expected": "NEGATIVE",
        "text": "The service was painfully slow, the staff were rude, and the food arrived completely cold.",
    },
    {
        "category": "negation_contrast",
        "expected": "POSITIVE",
        "text": "I expected this film to be terrible, yet it turned out not bad at all and genuinely entertaining.",
    },
    {
        "category": "mixed_contrast",
        "expected": "NEGATIVE",
        "text": "The cinematography is breathtaking, although the pacing drags terribly and the ending ruins everything.",
    },
    {
        "category": "sarcasm_hard",
        "expected": "NEGATIVE",
        "text": "Oh great, another three-hour sequel with absolutely nothing new to say. Just what I wanted.",
    },
    {
        "category": "sarcasm_hard",
        "expected": "NEGATIVE",
        "text": "Yeah, waiting four hours at a delayed airport gate was definitely the highlight of my vacation.",
    },
)


def inspect_tokenization(
    text: str,
    tokenizer: Any,
    max_length: int = 64,
) -> Dict[str, Any]:
    """Biến 1 câu thành `tokens`, `input_ids`, `attention_mask` và `decode` ngược lại để minh họa."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    input_ids = [int(x) for x in encoded["input_ids"]]
    attention_mask = [int(x) for x in encoded["attention_mask"]]
    tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
    subwords = [tok for tok in tokens if tok.startswith("##")]

    return {
        "original_text": text,
        "num_tokens": len(input_ids),
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "subword_pieces": subwords,
        "special_tokens_map": {
            "cls_token": getattr(tokenizer, "cls_token", None),
            "cls_token_id": getattr(tokenizer, "cls_token_id", None),
            "sep_token": getattr(tokenizer, "sep_token", None),
            "sep_token_id": getattr(tokenizer, "sep_token_id", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        },
        "decoded_with_special": tokenizer.decode(input_ids, skip_special_tokens=False),
        "decoded_clean": tokenizer.decode(input_ids, skip_special_tokens=True),
    }


def inspect_batch_padding(
    texts: Sequence[str],
    tokenizer: Any,
    max_length: int = 64,
) -> Dict[str, Any]:
    """Minh họa cách `padding=True` gom nhiều câu khác độ dài thành tensor `(B, L)` cùng `attention_mask`."""
    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    return {
        "texts": list(texts),
        "input_ids_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "dtype": str(input_ids.dtype),
        "rows": [
            {
                "text": txt,
                "tokens": list(tokenizer.convert_ids_to_tokens(input_ids[i].tolist())),
                "input_ids": input_ids[i].tolist(),
                "attention_mask": attention_mask[i].tolist(),
                "real_tokens": int(attention_mask[i].sum().item()),
                "pad_tokens": int((attention_mask[i] == 0).sum().item()),
            }
            for i, txt in enumerate(texts)
        ],
    }


def build_sentiment_pipeline(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: Optional[int] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Any:
    """Tạo Hugging Face `pipeline('text-classification')` chuẩn `transformers v5` (hỗ trợ cả offline model)."""
    resolved_device = device if device is not None else (0 if torch.cuda.is_available() else -1)
    if model is not None and tokenizer is not None:
        return pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
        )
    loaded_tok = AutoTokenizer.from_pretrained(model_name)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        task="text-classification",
        model=loaded_model,
        tokenizer=loaded_tok,
        device=resolved_device,
    )


def run_ex1_demo(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    output_dir: Optional[Path] = None,
    samples: Optional[Sequence[Dict[str, str]]] = None,
    sample_sentence: str = DEFAULT_SAMPLE_SENTENCE,
    device: Optional[int] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Chạy trọn bộ Ex1 (Tokenizer anatomy + Batch padding demo + Sentiment pipeline) và lưu JSON."""
    clf = build_sentiment_pipeline(
        model_name=model_name,
        device=device,
        model=model,
        tokenizer=tokenizer,
    )
    active_tokenizer = clf.tokenizer

    tok_demo = inspect_tokenization(sample_sentence, tokenizer=active_tokenizer, max_length=64)
    batch_demo = inspect_batch_padding(
        [
            "Great movie!",
            "This movie was surprisingly thoughtful and well directed.",
        ],
        tokenizer=active_tokenizer,
        max_length=64,
    )

    sample_list = list(samples) if samples is not None else list(DEFAULT_EX1_SAMPLES)
    texts = [item["text"] for item in sample_list]
    raw_preds = clf(texts, truncation=True, max_length=256)

    predictions: List[Dict[str, Any]] = []
    matched = 0
    for item, pred in zip(sample_list, raw_preds):
        pred_label = str(pred["label"]).upper()
        expected = str(item.get("expected", "")).upper()
        is_match = (pred_label == expected) if expected else None
        if is_match:
            matched += 1
        predictions.append(
            {
                "category": item.get("category", "custom"),
                "text": item["text"],
                "expected": expected or None,
                "predicted_label": pred_label,
                "score": round(float(pred["score"]), 4),
                "matches_expected": is_match,
            }
        )

    result: Dict[str, Any] = {
        "model_name": model_name,
        "task": "text-classification (sentiment-analysis)",
        "device": "cuda" if (device == 0 or (device is None and torch.cuda.is_available())) else "cpu",
        "tokenizer_demo": tok_demo,
        "batch_padding_demo": batch_demo,
        "num_samples": len(predictions),
        "matched_expected": matched,
        "predictions": predictions,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ex1_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return result
