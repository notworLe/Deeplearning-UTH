"""
Pipeline 05: Suy luận độc lập (Single Image / Batch) và đo độ trễ cho Lab 2 (Oxford-IIIT Pets).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from rich.console import Console
from rich.table import Table
from torchvision.datasets import OxfordIIITPet

from labs.lab2.vibe import config as lab2_cfg
from src import class_name
from src.vibe_scripts.lab2.data import build_transforms
from src.vibe_scripts.lab2.models import build_resnet18_pet

console = Console()


def load_predictor(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    classes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Nạp độc lập checkpoint `best.pt` vào ResNet-18 ở chế độ `eval()`."""
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt}")

    dev = device if device is not None else lab2_cfg.DEVICE
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    mode = payload.get("mode", "baseline")
    num_classes = int(payload.get("num_classes", lab2_cfg.NUM_CLASSES))

    model = build_resnet18_pet(mode=mode, num_classes=num_classes, use_pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(dev).eval()

    _, eval_tf = build_transforms()
    label_list = list(classes) if classes is not None else list(class_name.OXFORD_III_PET)

    return {
        "model": model,
        "mode": mode,
        "exp_id": payload.get("exp_id", ckpt.parent.name),
        "device": dev,
        "transform": eval_tf,
        "classes": label_list,
        "checkpoint_size_kb": ckpt.stat().st_size / 1024.0,
    }


def _prepare_single_tensor(image: Union[Path, str, Image.Image, torch.Tensor], transform: Any) -> torch.Tensor:
    """Chuyển đổi ảnh ngoài (Path / PIL / Tensor `[3, H, W]`) thành batch `[1, 3, 224, 224]`."""
    if isinstance(image, (Path, str)):
        with Image.open(image) as im:
            tensor = transform(im.convert("RGB"))
    elif isinstance(image, Image.Image):
        tensor = transform(image.convert("RGB"))
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3:
            tensor = image.float()
        elif image.ndim == 4 and image.shape[0] == 1:
            return image.float()
        else:
            raise ValueError(f"Expected 3D (3, H, W) or 4D (1, 3, H, W) tensor, got {tuple(image.shape)}")
    else:
        raise TypeError(f"Kiểu ảnh không được hỗ trợ: {type(image)}")
    return tensor.unsqueeze(0)


@torch.inference_mode()
def predict_single(
    predictor: Dict[str, Any],
    image: Union[Path, str, Image.Image, torch.Tensor],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Dự đoán 1 ảnh đơn dưới `@torch.inference_mode()`, trả về Top-k Softmax."""
    dev = predictor["device"]
    x = _prepare_single_tensor(image, predictor["transform"]).to(dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = predictor["model"](x)
    probs = torch.softmax(logits, dim=1)[0]
    if dev.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    k = min(top_k, probs.numel())
    top_vals, top_ids = torch.topk(probs, k=k)
    classes = predictor["classes"]
    top_k_list = [
        {
            "class_id": int(cid.item()),
            "label": classes[int(cid.item())] if int(cid.item()) < len(classes) else str(int(cid.item())),
            "prob": float(val.item()),
        }
        for val, cid in zip(top_vals, top_ids)
    ]

    return {
        "exp_id": predictor["exp_id"],
        "mode": predictor["mode"],
        "predicted_id": top_k_list[0]["class_id"],
        "predicted_label": top_k_list[0]["label"],
        "confidence": top_k_list[0]["prob"],
        "top3": top_k_list,
        "probabilities": probs.cpu(),
        "latency_ms": latency_ms,
        "checkpoint_size_kb": predictor["checkpoint_size_kb"],
    }


@torch.inference_mode()
def predict_batch(predictor: Dict[str, Any], batch_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Suy luận một batch tensor `(B, 3, 224, 224)`, trả về `(pred_ids, probs)`."""
    dev = predictor["device"]
    logits = predictor["model"](batch_tensor.to(dev))
    probs = torch.softmax(logits, dim=1).cpu()
    return probs.argmax(dim=1), probs


@torch.inference_mode()
def benchmark_latency(
    predictor: Dict[str, Any],
    batch_sizes: Sequence[int] = (1, 32),
    warmup_runs: int = 5,
    measure_runs: int = 20,
) -> Dict[int, Dict[str, float]]:
    """Đo độ trễ suy luận (ms) và thông lượng (img/s) cho từng kích thước batch."""
    dev = predictor["device"]
    model = predictor["model"]
    out: Dict[int, Dict[str, float]] = {}

    for bs in batch_sizes:
        dummy = torch.randn(bs, 3, lab2_cfg.IMAGE_SIZE, lab2_cfg.IMAGE_SIZE, device=dev)
        for _ in range(warmup_runs):
            _ = model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        for _ in range(measure_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(sum(times) / max(len(times), 1))
        out[bs] = {
            "mean_ms": mean_ms,
            "samples_per_sec": (bs * 1000.0) / max(mean_ms, 1e-7),
            "checkpoint_size_kb": predictor["checkpoint_size_kb"],
        }
    return out


def main() -> None:
    """Điểm vào CLI suy luận đơn ảnh hoặc benchmark tốc độ cho Lab 2."""
    parser = argparse.ArgumentParser(description="Lab 2 Inference CLI")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    predictor = load_predictor(args.checkpoint)
    if args.benchmark:
        console.print(benchmark_latency(predictor))
        return

    if args.image_path is not None:
        res = predict_single(predictor, args.image_path)
        true_lbl = "External Image"
    else:
        ds = OxfordIIITPet(root=lab2_cfg.DATA_PATH, split="test", target_types="category", download=False)
        predictor["classes"] = list(ds.classes)
        pil_img, y = ds[args.sample_idx]
        res = predict_single(predictor, pil_img)
        true_lbl = f"{ds.classes[y]} ({y})"

    table = Table(title=f"INFERENCE RESULT ({res['exp_id']})", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Ground Truth", true_lbl)
    table.add_row("Top-1", f"{res['predicted_label']} ({res['confidence']*100:.2f}%)")
    table.add_row("Top-3", ", ".join(f"{r['label']} ({r['prob']*100:.1f}%)" for r in res["top3"]))
    table.add_row("Latency", f"{res['latency_ms']:.2f} ms")
    console.print(table)


if __name__ == "__main__":
    main()
