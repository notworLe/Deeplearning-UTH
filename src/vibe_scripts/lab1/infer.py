"""Pipeline 05: Inference & Serving (Suy luận đơn ảnh, batch & đo độ trễ) cho Lab 1."""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table
import torch
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

from labs.lab1.vibe import config as lab1_cfg
from src.vibe_scripts import eda_helpers
from src.vibe_scripts.lab1.models import build_model

console = Console()
FASHION_CLASSES = eda_helpers.FASHION_CLASSES


def load_predictor(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Nạp checkpoint `best.pt`, tự khôi phục kiến trúc và tham số chuẩn hóa `norm_stats`."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {ckpt_path}")

    target_device = device if device is not None else lab1_cfg.DEVICE
    ckpt = torch.load(ckpt_path, map_location=target_device, weights_only=False)
    model_name = str(ckpt["model_name"])
    model = build_model(model_name, num_classes=lab1_cfg.NUM_CLASSES).to(target_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "model_name": model_name,
        "norm_stats": ckpt.get("norm_stats", {"mean": 0.0, "std": 1.0}),
        "checkpoint_size_kb": float(ckpt_path.stat().st_size / 1024.0),
        "device": target_device,
    }


def preprocess_image(
    image_or_path: Union[Path, str, Image.Image, np.ndarray, torch.Tensor],
    norm_stats: Dict[str, Any],
) -> torch.Tensor:
    """
    Chuyển ảnh từ file / PIL / NumPy / Tensor bất kỳ về tensor `[1, 1, 28, 28]` chuẩn hóa.
    """
    mean_v = float(norm_stats.get("mean", 0.0))
    std_v = float(norm_stats.get("std", 1.0))

    if isinstance(image_or_path, torch.Tensor):
        t = image_or_path.to(torch.float32)
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        elif t.ndim != 4:
            raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")
        if tuple(t.shape[1:]) != (1, 28, 28):
            raise ValueError(f"Expected (B, 1, 28, 28), got {tuple(t.shape)}")
        return t

    if isinstance(image_or_path, (Path, str)):
        pil_img = Image.open(image_or_path).convert("L")
    elif isinstance(image_or_path, Image.Image):
        pil_img = image_or_path.convert("L")
    elif isinstance(image_or_path, np.ndarray):
        arr = image_or_path
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr).convert("L")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_or_path)}")

    tf = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((mean_v,), (std_v,)),
        ]
    )
    return tf(pil_img).unsqueeze(0)


@torch.inference_mode()
def predict_single(
    predictor: Dict[str, Any],
    image_or_path: Union[Path, str, Image.Image, np.ndarray, torch.Tensor],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Suy luận 1 ảnh dưới `@torch.inference_mode()`, trả về Top-1, Top-3 Softmax và latency."""
    model = predictor["model"]
    device: torch.device = predictor["device"]
    x = preprocess_image(image_or_path, predictor["norm_stats"]).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_indices = probs.topk(min(top_k, lab1_cfg.NUM_CLASSES))

    top3: List[Dict[str, Any]] = [
        {
            "class_id": int(idx.item()),
            "label": FASHION_CLASSES[int(idx.item())],
            "prob": float(p.item()),
        }
        for p, idx in zip(top_probs, top_indices)
    ]

    return {
        "model_name": predictor["model_name"],
        "checkpoint_size_kb": predictor["checkpoint_size_kb"],
        "predicted_class_id": top3[0]["class_id"],
        "predicted_label": top3[0]["label"],
        "confidence": top3[0]["prob"],
        "top3": top3,
        "latency_ms": float(latency_ms),
        "probabilities": probs.cpu(),
    }


@torch.inference_mode()
def predict_batch(
    predictor: Dict[str, Any],
    batch_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Suy luận theo lô `[B, 1, 28, 28]`, trả về `(preds [B], probs [B, 10])`."""
    model = predictor["model"]
    device: torch.device = predictor["device"]
    x = batch_tensor.to(device=device, dtype=torch.float32)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    return preds.cpu(), probs.cpu()


@torch.inference_mode()
def benchmark_latency(
    predictor: Dict[str, Any],
    batch_sizes: Tuple[int, ...] = (1, 32),
    warmup_runs: int = 10,
    measure_runs: int = 50,
) -> Dict[int, Dict[str, float]]:
    """Đo độ trễ suy luận (mean, p95, throughput) sau warm-up cho các batch size."""
    model = predictor["model"]
    device: torch.device = predictor["device"]
    results: Dict[int, Dict[str, float]] = {}

    for bs in batch_sizes:
        dummy = torch.randn(bs, 1, 28, 28, device=device)
        for _ in range(warmup_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings_ms: List[float] = []
        for _ in range(measure_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(timings_ms, dtype=np.float64)
        mean_ms = float(np.mean(arr))
        p95_ms = float(np.percentile(arr, 95))
        results[bs] = {
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "samples_per_sec": float((bs * 1000.0) / max(mean_ms, 1e-7)),
            "checkpoint_size_kb": predictor["checkpoint_size_kb"],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 1 Inference & Serving CLI")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-idx", type=int, default=18)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    predictor = load_predictor(args.checkpoint)
    if args.benchmark:
        bench = benchmark_latency(predictor)
        console.print(bench)
        return

    if args.image_path is not None:
        res = predict_single(predictor, args.image_path)
        true_label = "N/A (External Image)"
    else:
        raw_test = FashionMNIST(root=lab1_cfg.DATA_PATH, train=False, download=False)
        pil_img, label_id = raw_test[args.sample_idx]
        res = predict_single(predictor, pil_img)
        true_label = f"{FASHION_CLASSES[label_id]} ({label_id})"

    table = Table(title=f"INFERENCE RESULT ({res['model_name'].upper()})", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Ground Truth", true_label)
    table.add_row("Top-1 Prediction", f"{res['predicted_label']} ({res['confidence']*100:.2f}%)")
    table.add_row("Top-3", ", ".join(f"{r['label']} ({r['prob']*100:.1f}%)" for r in res["top3"]))
    table.add_row("Latency", f"{res['latency_ms']:.3f} ms")
    table.add_row("Checkpoint Size", f"{res['checkpoint_size_kb']:.2f} KB")
    console.print(table)


if __name__ == "__main__":
    main()
