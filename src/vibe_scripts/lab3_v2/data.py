"""Data Engineering Layer (6.1–6.4) cho Lab 3 v2: Tải IMDB, Stratified Split, Audit/EDA & Tokenization."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from datasets import DatasetDict, load_dataset

from src.vibe_scripts.lab3_v2.analysis import _ensure_headless_matplotlib

LABEL_NAMES: Tuple[str, str] = ("Negative", "Positive")
_HTML_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def build_stratified_subset_indices(
    labels: Sequence[int],
    n_samples: int,
    seed: int = 248,
    exclude_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Lấy chỉ số stratified cân bằng 2 lớp (không trùng với `exclude_indices`) theo `seed` cố định."""
    if n_samples <= 0:
        raise ValueError(f"n_samples phải > 0, nhận {n_samples}")
    excluded = set(int(i) for i in (exclude_indices or ()))
    arr = np.asarray(labels, dtype=np.int64)
    unique_classes = sorted(int(c) for c in np.unique(arr))
    if len(unique_classes) < 2:
        raise ValueError(f"Cần ít nhất 2 lớp để stratified split, nhận {unique_classes}")

    rng = np.random.default_rng(seed)
    base_per_class = n_samples // len(unique_classes)
    remainder = n_samples % len(unique_classes)

    chosen: List[int] = []
    for pos, cls_id in enumerate(unique_classes):
        need = base_per_class + (1 if pos < remainder else 0)
        pool = [int(i) for i in np.flatnonzero(arr == cls_id) if int(i) not in excluded]
        if len(pool) < need:
            raise ValueError(
                f"Không đủ mẫu lớp {cls_id}: cần {need}, chỉ còn {len(pool)} mẫu khả dụng."
            )
        perm = rng.permutation(pool)
        chosen.extend(int(i) for i in perm[:need])

    rng.shuffle(chosen)
    return chosen


def load_and_split_imdb(
    train_size: int = 4000,
    val_size: int = 1000,
    test_size: int = 1000,
    seed: int = 248,
    split_file: Optional[Path] = None,
    raw_dataset: Optional[DatasetDict] = None,
    dataset_name: str = "stanfordnlp/imdb",
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """Tải IMDB (hoặc dùng `raw_dataset` offline), lấy stratified subset & khóa `split_indices.json`."""
    resolved_name = "stanfordnlp/imdb" if dataset_name == "imdb" else dataset_name
    raw = raw_dataset if raw_dataset is not None else load_dataset(resolved_name)
    raw_train = raw["train"]
    raw_test = raw["test"]

    split_meta: Optional[Dict[str, Any]] = None
    if split_file is not None and Path(split_file).exists():
        candidate = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if (
            int(candidate.get("seed", -1)) == int(seed)
            and int(candidate.get("train_size", -1)) == int(train_size)
            and int(candidate.get("val_size", -1)) == int(val_size)
            and int(candidate.get("test_size", -1)) == int(test_size)
            and max(candidate["train_indices"], default=-1) < len(raw_train)
            and max(candidate["val_indices"], default=-1) < len(raw_train)
            and max(candidate["test_indices"], default=-1) < len(raw_test)
        ):
            split_meta = candidate

    if split_meta is None:
        train_labels = list(raw_train["label"])
        test_labels = list(raw_test["label"])

        train_idx = build_stratified_subset_indices(train_labels, n_samples=train_size, seed=seed)
        val_idx = build_stratified_subset_indices(
            train_labels,
            n_samples=val_size,
            seed=seed + 1,
            exclude_indices=train_idx,
        )
        test_idx = build_stratified_subset_indices(test_labels, n_samples=test_size, seed=seed + 2)

        assert set(train_idx).isdisjoint(val_idx), "Phát hiện rò rỉ chỉ số giữa Train và Validation!"

        split_meta = {
            "dataset": "imdb",
            "seed": int(seed),
            "raw_train_total": len(raw_train),
            "raw_test_total": len(raw_test),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        }
        if split_file is not None:
            sp = Path(split_file)
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(json.dumps(split_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    splits = DatasetDict(
        {
            "train": raw_train.select(split_meta["train_indices"]),
            "validation": raw_train.select(split_meta["val_indices"]),
            "test": raw_test.select(split_meta["test_indices"]),
        }
    )
    return splits, split_meta


def _text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.strip().lower().split()).encode("utf-8")).hexdigest()


def _length_stats(values: Sequence[int]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": int(arr.min()),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "median": round(float(np.percentile(arr, 50)), 1),
        "mean": round(float(arr.mean()), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "max": int(arr.max()),
    }


def run_imdb_data_audit_and_eda(
    dataset_dict: DatasetDict,
    tokenizer: Any,
    eda_dir: Path,
    max_length: int = 256,
    split_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Chạy 6.1 Collection, 6.2 Cleaning Audit và 6.3 EDA (kèm 2 biểu đồ) cho tập con IMDB."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    out_dir = Path(eda_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "validation", "test")

    # 6.1 Collection Report
    collection_report = {
        "dataset": "imdb",
        "label_names": list(LABEL_NAMES),
        "raw_train_total": int(split_metadata["raw_train_total"]) if split_metadata else None,
        "raw_test_total": int(split_metadata["raw_test_total"]) if split_metadata else None,
        "seed": int(split_metadata["seed"]) if split_metadata else None,
        "split_sizes": {s: len(dataset_dict[s]) for s in split_names},
    }
    (out_dir / "collection_report.json").write_text(
        json.dumps(collection_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6.2 Cleaning Audit (empty texts, HTML <br />, internal & cross-split exact duplicates)
    hashes_by_split: Dict[str, List[str]] = {}
    empty_counts: Dict[str, int] = {}
    html_br_counts: Dict[str, int] = {}
    internal_dups: Dict[str, int] = {}
    for s in split_names:
        texts = list(dataset_dict[s]["text"])
        empty_counts[s] = sum(1 for t in texts if not str(t).strip())
        html_br_counts[s] = sum(1 for t in texts if _HTML_BR_RE.search(str(t)))
        hs = [_text_hash(str(t)) for t in texts]
        hashes_by_split[s] = hs
        internal_dups[s] = len(hs) - len(set(hs))

    hash_sets = {s: set(hashes_by_split[s]) for s in split_names}
    cross_split_dups = {
        "train_val": len(hash_sets["train"] & hash_sets["validation"]),
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "val_test": len(hash_sets["validation"] & hash_sets["test"]),
    }
    cleaning_report = {
        "empty_or_whitespace_texts": empty_counts,
        "reviews_with_html_br_tag": html_br_counts,
        "internal_exact_duplicates": internal_dups,
        "cross_split_exact_duplicates": cross_split_dups,
        "note": (
            "Giữ nguyên văn bản gốc IMDB (không xóa thủ công `<br />`) để bảo toàn tính tái lập "
            "của benchmark chuẩn và kiểm chứng khả năng xử lý nhiễu nhẹ của DistilBERT."
        ),
    }
    (out_dir / "cleaning_report.json").write_text(
        json.dumps(cleaning_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6.3 EDA Summary (class balance, word counts, WordPiece token lengths & truncation rate)
    class_counts: Dict[str, Dict[str, int]] = {}
    word_stats: Dict[str, Dict[str, float]] = {}
    token_stats: Dict[str, Dict[str, float]] = {}
    truncation_pct: Dict[str, float] = {}
    all_token_lens: Dict[str, List[int]] = {}

    for s in split_names:
        texts = list(dataset_dict[s]["text"])
        labels = list(dataset_dict[s]["label"])
        neg = sum(1 for y in labels if int(y) == 0)
        pos = sum(1 for y in labels if int(y) == 1)
        class_counts[s] = {"Negative": neg, "Positive": pos}

        w_lens = [len(str(t).split()) for t in texts]
        word_stats[s] = _length_stats(w_lens)

        # Đếm độ dài WordPiece thực (không cắt) theo từng lô để đo chính xác tỷ lệ truncation tại max_length
        tok_lens: List[int] = []
        batch_step = 256
        for start in range(0, len(texts), batch_step):
            enc = tokenizer(
                texts[start : start + batch_step],
                truncation=False,
                padding=False,
                add_special_tokens=True,
                verbose=False,
            )
            tok_lens.extend(len(ids) for ids in enc["input_ids"])
        all_token_lens[s] = tok_lens
        token_stats[s] = _length_stats(tok_lens)
        over = sum(1 for n in tok_lens if n > max_length)
        truncation_pct[s] = round((over / max(len(tok_lens), 1)) * 100.0, 2)

    # Plot 1: Class Distribution across Train / Val / Test
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    x = np.arange(len(split_names))
    w = 0.34
    neg_vals = [class_counts[s]["Negative"] for s in split_names]
    pos_vals = [class_counts[s]["Positive"] for s in split_names]
    b1 = ax.bar(x - w / 2, neg_vals, w, label="Negative (0)", color="#c53030")
    b2 = ax.bar(x + w / 2, pos_vals, w, label="Positive (1)", color="#2b6cb0")
    y_max = max(neg_vals + pos_vals, default=1)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.02, f"{int(h):,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "Validation", "Test"])
    ax.set_ylabel("Number of Reviews")
    ax.set_ylim(0, y_max * 1.18)
    ax.set_title("IMDB Stratified Subset — Class Distribution (50/50 Balanced)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    class_plot_path = out_dir / "eda_class_distribution.png"
    fig.savefig(class_plot_path, dpi=150)
    plt.close(fig)

    # Plot 2: WordPiece Token Length Distribution + max_length cutoff
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    capped_train = np.clip(all_token_lens["train"], 0, 800)
    capped_val = np.clip(all_token_lens["validation"], 0, 800)
    ax.hist(capped_train, bins=40, alpha=0.65, density=True, label=f"Train (n={len(capped_train):,})", color="#2b6cb0")
    ax.hist(capped_val, bins=40, alpha=0.50, density=True, label=f"Validation (n={len(capped_val):,})", color="#dd6b20")
    ax.axvline(
        max_length,
        color="#c53030",
        linestyle="--",
        linewidth=2.0,
        label=f"max_length = {max_length} (Train truncated: {truncation_pct['train']:.1f}%)",
    )
    ax.set_xlabel("WordPiece Token Length (clipped at 800 for display)")
    ax.set_ylabel("Density")
    ax.set_title("IMDB Review Token Length Distribution vs Truncation Threshold")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    len_plot_path = out_dir / "eda_token_length.png"
    fig.savefig(len_plot_path, dpi=150)
    plt.close(fig)

    sample_previews = [
        {
            "label": LABEL_NAMES[int(dataset_dict["train"][i]["label"])],
            "token_len": int(all_token_lens["train"][i]),
            "snippet": str(dataset_dict["train"][i]["text"])[:180].replace("\n", " "),
        }
        for i in range(min(4, len(dataset_dict["train"])))
    ]

    eda_summary = {
        "max_length": int(max_length),
        "class_counts": class_counts,
        "word_length_stats": word_stats,
        "token_length_stats": token_stats,
        "truncated_pct_above_max_length": truncation_pct,
        "sample_previews": sample_previews,
        "figures": {
            "class_distribution": str(class_plot_path),
            "token_length": str(len_plot_path),
        },
    }
    (out_dir / "eda_summary.json").write_text(
        json.dumps(eda_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "collection_report": collection_report,
        "cleaning_report": cleaning_report,
        "eda_summary": eda_summary,
    }


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: Any,
    max_length: int = 256,
) -> DatasetDict:
    """Tokenize các split với `truncation=True` (padding động do `DataCollatorWithPadding` xử lý)."""

    def _tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset_dict.map(
        _tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing IMDB splits",
    )
    # Xác thực Tensor Contract trên mẫu đầu tiên
    sample = tokenized["train"][0]
    if "input_ids" not in sample or "attention_mask" not in sample:
        raise ValueError("Tokenized dataset thiếu cột 'input_ids' hoặc 'attention_mask'.")
    if not (1 <= len(sample["input_ids"]) <= max_length):
        raise ValueError(
            f"Độ dài token vượt giới hạn [1, {max_length}], nhận {len(sample['input_ids'])}."
        )
    return tokenized
