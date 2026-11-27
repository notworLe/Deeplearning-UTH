"""Inference Engine cho Lab 2 v2 (Oxford-IIIT Pets — nạp checkpoint, Top-5 Softmax & đo độ trễ)."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.vibe_scripts.lab2_v2.models import PetClassifier, build_model
from src.vibe_scripts.lab2_v2.transforms import (
    ImageNetNormStats,
    preprocess_image,
    validate_image_tensor,
)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[PetClassifier, Dict[str, Any]]:
    """Nạp checkpoint `.pt`, tự tái dựng kiến trúc (`pretrained=False`, không cần mạng) và nạp trọng số."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    device_obj = torch.device(device)
    payload = torch.load(ckpt_path, map_location=device_obj, weights_only=False)
    m_cfg = payload["model_config"]
    model = build_model(
        arch=str(m_cfg["arch"]),
        mode=str(m_cfg.get("mode", "transfer")),
        num_classes=int(m_cfg["num_classes"]),
        pretrained=False,
        image_size=int(m_cfg.get("image_size", 224)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return model, payload


class PetPredictor:
    """Bộ suy luận tự khép kín cho Oxford-IIIT Pets (37 giống chó & mèo)."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.payload = load_checkpoint(self.checkpoint_path, device=self.device)
        self.class_names: Tuple[str, ...] = tuple(str(c) for c in self.payload["class_names"])
        self.species_by_class: Dict[str, str] = dict(self.payload.get("species_by_class", {}))
        self.norm_stats = ImageNetNormStats.from_dict(self.payload["norm_stats"])
        self.image_size = int(self.payload["model_config"].get("image_size", 224))
        self.resize_size = int(self.payload.get("resize_size", 256))

    @torch.inference_mode()
    def predict_tensor(
        self,
        tensor: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Dự đoán trên tensor đã chuẩn hóa `(3, 224, 224)` hoặc `(B, 3, 224, 224)` dưới `inference_mode`."""
        x = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        validate_image_tensor(x, expected_image_size=self.image_size, allow_unbatched=False)
        x_dev = x.to(self.device)
        logits = self.model(x_dev)
        probs = F.softmax(logits, dim=1)

        k = min(int(top_k), len(self.class_names))
        top_probs, top_ids = torch.topk(probs, k=k, dim=1)
        results: List[Dict[str, Any]] = []
        for row_i in range(x.shape[0]):
            pred_id = int(top_ids[row_i, 0].item())
            pred_name = self.class_names[pred_id]
            top_list = []
            for rank in range(k):
                cid = int(top_ids[row_i, rank].item())
                cname = self.class_names[cid]
                top_list.append(
                    {
                        "class_id": cid,
                        "class_name": cname,
                        "species": self.species_by_class.get(cname, "Unknown"),
                        "probability": float(top_probs[row_i, rank].item()),
                    }
                )
            results.append(
                {
                    "predicted_id": pred_id,
                    "predicted_class": pred_name,
                    "species": self.species_by_class.get(pred_name, "Unknown"),
                    "confidence": float(top_probs[row_i, 0].item()),
                    "top_k": top_list,
                    "requires_grad": bool(logits.requires_grad),
                }
            )
        return results

    def predict_image(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Nạp 1 ảnh từ đường dẫn/PIL, tiền xử lý nhất quán với eval và trả về kết quả Top-K."""
        tensor = preprocess_image(
            image=image,
            image_size=self.image_size,
            resize_size=self.resize_size,
            norm_stats=self.norm_stats,
        )
        return self.predict_tensor(tensor, top_k=top_k)[0]

    @torch.inference_mode()
    def benchmark_latency(
        self,
        batch_sizes: Sequence[int] = (1, 32),
        warmup: int = 8,
        repeats: int = 40,
    ) -> Dict[str, Any]:
        """Đo độ trễ suy luận (mean, p95, ms/sample) và kích thước checkpoint trên thiết bị hiện tại."""
        size_mb = round(self.checkpoint_path.stat().st_size / (1024.0 * 1024.0), 2)
        bench: Dict[str, Any] = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_size_mb": size_mb,
            "device": str(self.device),
            "arch": self.payload["model_config"]["arch"],
            "mode": self.payload["model_config"]["mode"],
            "batches": {},
        }
        for bs in batch_sizes:
            dummy = torch.randn((int(bs), 3, self.image_size, self.image_size), device=self.device)
            for _ in range(warmup):
                _ = self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            timings_ms: List[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = self.model(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                timings_ms.append((time.perf_counter() - t0) * 1000.0)

            arr = np.asarray(timings_ms, dtype=np.float64)
            bench["batches"][f"bs_{int(bs)}"] = {
                "batch_size": int(bs),
                "mean_batch_ms": round(float(arr.mean()), 3),
                "p95_batch_ms": round(float(np.percentile(arr, 95)), 3),
                "ms_per_sample": round(float(arr.mean() / int(bs)), 4),
            }
        return bench
