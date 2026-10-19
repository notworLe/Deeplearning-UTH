"""Practice 1 / Stage 6 — visualize *saved* final test predictions.

No model training, checkpoint selection, or repeat test inference happens here.
The numeric results come exclusively from Stage-5 test_predictions.csv and
per_class_metrics.csv. Pixel values are drawn from the same 70k-row source CSV,
matching rows by the original test-split order and verifying their labels.

Run in a terminal from the project root:
    python src/visualize.py
or click VS Code's 'Run Python File' button on this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def resolve_path(path: str) -> Path:
    """Interpret relative paths relative to the project root, not VS Code cwd."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def validate_evidence(pred: pd.DataFrame, per_class: pd.DataFrame, metrics: dict) -> None:
    columns = {"index_in_test", "actual", "predicted", "confidence", "correct"}
    if not columns.issubset(pred.columns):
        raise ValueError(f"Missing prediction columns: {sorted(columns - set(pred.columns))}")
    expected_n = int(metrics["test_samples"])
    if expected_n != 10000 or len(pred) != expected_n:
        raise ValueError(f"Prediction count mismatch with test_metrics.json: {len(pred)} != {expected_n}")
    if pred["index_in_test"].tolist() != list(range(expected_n)):
        raise ValueError("test_predictions.csv must have consecutive test-split indexes in original order.")
    if not pred["actual"].between(0, 9).all() or not pred["predicted"].between(0, 9).all():
        raise ValueError("Class ids must be in [0,9].")
    if not pred["confidence"].between(0, 1).all():
        raise ValueError("Confidence must be between 0 and 1.")
    truth = pred["actual"].to_numpy() == pred["predicted"].to_numpy()
    if not np.array_equal(truth.astype(int), pred["correct"].to_numpy()):
        raise ValueError("Prediction correctness flags do not match actual/predicted classes.")
    if int(truth.sum()) != int(metrics["test_correct"]):
        raise ValueError("Prediction accuracy does not match test_metrics.json.")
    if len(per_class) != 10 or set(per_class["class_id"]) != set(range(10)):
        raise ValueError("Expected exactly ten per-class metric rows.")
    for row in per_class.itertuples(index=False):
        n = int((pred["actual"] == int(row.class_id)).sum())
        hits = int(((pred["actual"] == int(row.class_id)) & (pred["correct"] == 1)).sum())
        if n != int(row.support) or abs(hits / n - float(row.recall)) > 1e-6:
            raise ValueError(f"Per-class counts disagree for {row.class_name}.")


def choose_indices(pred: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    """One correct and one error per actual class; high-confidence errors separately."""
    correct, incorrect = [], []
    for class_id in range(10):
        subset = pred.loc[pred["actual"] == class_id]
        right = subset.loc[subset["correct"] == 1]
        wrong = subset.loc[subset["correct"] == 0]
        if right.empty or wrong.empty:
            raise ValueError(f"Need a correct and an incorrect example for class {class_id}.")
        correct.append(int(right.iloc[0]["index_in_test"]))
        incorrect.append(int(wrong.iloc[0]["index_in_test"]))
    top_errors = pred.loc[pred["correct"] == 0].sort_values(
        ["confidence", "index_in_test"], ascending=[False, True]
    ).head(12)["index_in_test"].astype(int).tolist()
    return correct, incorrect, top_errors


def load_selected_pixels(csv_path: Path, wanted: set[int], pred: pd.DataFrame) -> dict[int, np.ndarray]:
    """Scan one source CSV; materialize only selected images, not all 70,000 rows.

    Fast byte-wise iteration avoids a large 70,000 × 784 pixel allocation.
    Confirms each selected row belongs to the `test` split and has the
    label stored in Stage-5 prediction evidence.
    """
    images = {}
    test_idx = 0
    with csv_path.open("rb") as handle:
        header = handle.readline().decode("utf-8-sig").strip().split(",")
        expected = ["label"] + [f"pixel{i}" for i in range(1, 785)] + ["split"]
        if header != expected:
            raise ValueError("CSV schema mismatch; expected label,pixel1..pixel784,split.")
        for line in handle:
            line = line.strip()
            if not line:
                continue
            first_part, delimiter, split = line.rpartition(b",")
            if not delimiter or split != b"test":
                continue
            if test_idx in wanted:
                raw = np.fromstring(first_part.decode("ascii"), sep=",", dtype=np.int16)
                if raw.shape != (785,):
                    raise ValueError(f"Malformed CSV test row {test_idx}: {raw.shape}.")
                actual = int(pred.iloc[test_idx]["actual"])
                if int(raw[0]) != actual:
                    raise ValueError(f"CSV/test_predictions mismatch at test row {test_idx}: {raw[0]} vs {actual}")
                if (raw[1:] < 0).any() or (raw[1:] > 255).any():
                    raise ValueError(f"Pixel out of 0–255 at test row {test_idx}.")
                images[test_idx] = raw[1:].reshape(28, 28).astype(np.uint8)
            test_idx += 1
    if test_idx != len(pred):
        raise ValueError(f"CSV test split has {test_idx} rows, but saved predictions have {len(pred)}.")
    if set(images) != wanted:
        raise ValueError(f"Could not retrieve selected CSV test images: {sorted(wanted - set(images))}")
    return images


def draw_grid(ids: list[int], pred: pd.DataFrame, images: dict[int, np.ndarray], output: Path,
              title: str, columns: int = 4) -> None:
    rows = int(np.ceil(len(ids) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.5, rows * 3.55))
    axes = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes, ids):
        record = pred.iloc[idx]
        y, yhat, probability = int(record.actual), int(record.predicted), float(record.confidence)
        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"Actual: {LABEL_NAMES[y]}\nPred: {LABEL_NAMES[yhat]}  ({probability:.0%})",
                     fontsize=9.4, color="darkgreen" if y == yhat else "firebrick", pad=8)
        ax.axis("off")
    for ax in axes[len(ids):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.997)
    fig.text(.5, .006, "Images: original grayscale test CSV | Confidence: saved Stage-5 prediction",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .015, 1, .976), pad=1.6)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_recall(per_class: pd.DataFrame, output: Path) -> None:
    df = per_class.sort_values("class_id")
    fig, ax = plt.subplots(figsize=(11, 6.4))
    y = np.arange(10)
    recall = df["recall"].to_numpy(dtype=float) * 100
    bars = ax.barh(y, recall, color=plt.get_cmap("tab20c")(np.linspace(.12, .8, 10)))
    ax.set_yticks(y, labels=df["class_name"])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Test recall (%) — correct among 1,000 actual samples per class")
    ax.set_title("Final E2 test: recall by FashionMNIST class")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, recall):
        ax.text(value + 1.2, bar.get_y() + bar.get_height()/2, f"{value:.1f}%",
                va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 — visualize final saved test predictions; NO training/retest.")
    parser.add_argument("--csv", default="data/fashion_mnist_full.csv")
    parser.add_argument("--predictions", default="outputs/experiments/test_predictions.csv")
    parser.add_argument("--test-metrics", default="outputs/experiments/test_metrics.json")
    parser.add_argument("--per-class", default="outputs/experiments/per_class_metrics.csv")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()
    csv_path = resolve_path(args.csv)
    predictions_path = resolve_path(args.predictions)
    metrics_path = resolve_path(args.test_metrics)
    per_class_path = resolve_path(args.per_class)
    out_dir = resolve_path(args.out_dir)
    for p in (csv_path, predictions_path, metrics_path, per_class_path):
        if not p.is_file():
            parser.error(f"Missing required Stage-5 input file: {p}")
    pred = pd.read_csv(predictions_path)
    per_class = pd.read_csv(per_class_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_evidence(pred, per_class, metrics)
    correct, incorrect, high_conf_errors = choose_indices(pred)
    selected = correct + incorrect
    pixels = load_selected_pixels(csv_path, set(selected + high_conf_errors), pred)
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_grid(selected, pred, pixels, out_dir / "predictions.png",
              "Predicted vs Actual — 10 correct + 10 incorrect test samples")
    draw_grid(high_conf_errors, pred, pixels, out_dir / "misclassified_examples.png",
              "12 high-confidence misclassifications (purposefully selected)")
    draw_recall(per_class, out_dir / "per_class_recall.png")
    summary = {
        "stage": 6,
        "method": "visualization from saved Stage-5 predictions; no training or model inference",
        "selected_experiment": metrics["selected_experiment"],
        "test_samples": metrics["test_samples"],
        "test_correct": metrics["test_correct"],
        "test_accuracy_percent": metrics["test_accuracy_percent"],
        "correct_examples_test_indices": correct,
        "wrong_examples_test_indices": incorrect,
        "high_confidence_error_test_indices": high_conf_errors,
        "selection_disclosure": "Per-class examples are the first matching samples; high-confidence errors are intentionally selected, not representative/random.",
        "figures": ["figures/predictions.png", "figures/misclassified_examples.png", "figures/per_class_recall.png"],
        "predictions_vs_csv_labels_verified": True,
    }
    (out_dir.parent / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Stage 6: E2 test={metrics['test_correct']:,}/{metrics['test_samples']:,} "
          f"({metrics['test_accuracy_percent']:.2f}%).", flush=True)
    for item in summary["figures"]:
        print(f"Saved: {(out_dir.parent / item).resolve()}", flush=True)
    print("No training, new test inference, or hyperparameter selection performed.", flush=True)


if __name__ == "__main__":
    main()
