"""Pipeline suy luận (Inference) độc lập: nạp checkpoint, tái dựng kiến trúc và dự đoán Top-K."""

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.vibe_scripts.lab1_v2.data import DEFAULT_CLASS_NAMES
from src.vibe_scripts.lab1_v2.mlp import build_model
from src.vibe_scripts.lab1_v2.transforms import NormStats, preprocess_image, validate_image_tensor


def load_checkpoint(
    checkpoint_path: Path,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Nạp checkpoint `.pt` và tự động tái dựng kiến trúc cùng `norm_stats`."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {ckpt_path}")

    device_obj = torch.device(device)
    payload: Dict[str, Any] = torch.load(ckpt_path, map_location=device_obj, weights_only=False)
    model_cfg = payload["model_config"]

    model = build_model(
        model_type=model_cfg["model_type"],
        input_shape=tuple(model_cfg["input_shape"]),
        num_classes=int(model_cfg["num_classes"]),
        hidden_dims=tuple(model_cfg.get("hidden_dims", ())),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return model, payload


class FashionPredictor:
    """Bộ suy luận gọn nhẹ chạy dưới `@torch.inference_mode()` (không bao giờ cập nhật trọng số)."""

    def __init__(
        self,
        model: nn.Module,
        norm_stats: NormStats,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        device: Union[str, torch.device] = "cpu",
        checkpoint_path: Union[str, Path, None] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.norm_stats = norm_stats
        self.class_names = tuple(class_names)
        self.input_shape = tuple(input_shape)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: Union[str, torch.device] = "cpu",
    ) -> "FashionPredictor":
        """Khởi tạo Predictor trực tiếp từ file `best.pt` / `last.pt`."""
        model, payload = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
        norm_stats = NormStats.from_dict(payload["norm_stats"])
        class_names = tuple(payload.get("class_names", DEFAULT_CLASS_NAMES))
        input_shape = tuple(payload["model_config"]["input_shape"])
        return cls(
            model=model,
            norm_stats=norm_stats,
            class_names=class_names,
            input_shape=input_shape,
            device=device,
            checkpoint_path=checkpoint_path,
        )

    @torch.inference_mode()
    def predict_batch(self, batch: torch.Tensor, top_k: int = 3) -> List[Dict[str, Any]]:
        """Dự đoán trên một batch tensor đã chuẩn hóa `(B, 1, 28, 28)`."""
        validate_image_tensor(batch, expected_shape=self.input_shape)
        k = max(1, min(int(top_k), len(self.class_names)))
        logits = self.model(batch.to(self.device))
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=k, dim=1)

        results: List[Dict[str, Any]] = []
        probs_cpu = probs.cpu()
        top_probs_cpu = top_probs.cpu()
        top_indices_cpu = top_indices.cpu()

        for i in range(batch.shape[0]):
            top_list = [
                {
                    "class_id": int(top_indices_cpu[i, j].item()),
                    "class_name": self.class_names[int(top_indices_cpu[i, j].item())],
                    "probability": float(top_probs_cpu[i, j].item()),
                }
                for j in range(k)
            ]
            results.append(
                {
                    "predicted_id": top_list[0]["class_id"],
                    "predicted_class": top_list[0]["class_name"],
                    "confidence": top_list[0]["probability"],
                    "top_k": top_list,
                    "probabilities": [float(v) for v in probs_cpu[i].tolist()],
                    "requires_grad": bool(probs.requires_grad),
                }
            )
        return results

    @torch.inference_mode()
    def predict_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Dự đoán cho 1 ảnh (đường dẫn file ảnh, PIL Image, NumPy array hoặc Tensor)."""
        if isinstance(image, (str, Path)):
            img_obj: Union[Image.Image, np.ndarray, torch.Tensor] = Image.open(image)
        else:
            img_obj = image
        tensor = preprocess_image(img_obj, norm_stats=self.norm_stats, expected_shape=self.input_shape)
        return self.predict_batch(tensor, top_k=top_k)[0]

    @torch.inference_mode()
    def benchmark_latency(
        self,
        batch_size: int = 1,
        warmup_runs: int = 10,
        timed_runs: int = 50,
    ) -> Dict[str, float]:
        """Đo độ trễ suy luận (ms/batch và ms/sample) kèm warm-up và đồng bộ CUDA."""
        dummy = torch.randn((batch_size, *self.input_shape), device=self.device)
        for _ in range(warmup_runs):
            _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        latencies_ms: List[float] = []
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.asarray(latencies_ms, dtype=np.float64)
        ckpt_kb = (
            float(self.checkpoint_path.stat().st_size / 1024.0)
            if self.checkpoint_path and self.checkpoint_path.exists()
            else 0.0
        )
        return {
            "batch_size": float(batch_size),
            "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "ms_per_sample": float(arr.mean() / batch_size),
            "checkpoint_size_kb": ckpt_kb,
        }
