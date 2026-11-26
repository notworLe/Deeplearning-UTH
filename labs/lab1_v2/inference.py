"""Thin CLI entrypoint suy luận đơn ảnh / mẫu Test và đo latency từ checkpoint Lab 1 v2."""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab1_v2.config as cfg
from src.vibe_scripts.lab1_v2.data import prepare_fashion_mnist
from src.vibe_scripts.lab1_v2.inference import FashionPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 1 v2 Inference CLI — Dự đoán Top-3 & đo độ trễ từ checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Đường dẫn tới file best.pt (nếu bỏ trống sẽ tự chọn winning_mlp trong ablation_summary.json).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Đường dẫn ảnh ngoài (PNG/JPG). Nếu bỏ trống sẽ lấy ảnh từ tập Test theo --sample-idx.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Chỉ số ảnh mẫu trong tập Test FashionMNIST (0..9999).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Số lớp có xác suất cao nhất cần hiển thị (mặc định: 3).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Đo thêm độ trễ suy luận (batch_size=1 và batch_size=32).",
    )
    return parser.parse_args()


def _resolve_default_checkpoint() -> Path:
    summary_path = cfg.ABLATION_DIR / "ablation_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        winner = data.get("winning_mlp_run_id")
        if winner:
            return cfg.RUNS_DIR / winner / "best.pt"
    return cfg.RUNS_DIR / "mlp_default" / "best.pt"


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint or _resolve_default_checkpoint()
    predictor = FashionPredictor.from_checkpoint(checkpoint_path=ckpt_path, device=cfg.DEVICE)

    if args.image is not None:
        pred = predictor.predict_image(args.image, top_k=args.top_k)
        output = {"source": str(args.image), "prediction": pred}
    else:
        bundle = prepare_fashion_mnist(
            data_root=cfg.DATA_PATH,
            batch_size=cfg.BATCH_SIZE,
            val_size=cfg.VAL_SIZE,
            seed=cfg.SEED,
            num_workers=cfg.WOKER,
            class_names=cfg.CLASS_NAMES,
            verify_manifest=True,
            download=False,
        )
        img_tensor, true_label = bundle.test_dataset[args.sample_idx]
        pred = predictor.predict_batch(img_tensor.unsqueeze(0), top_k=args.top_k)[0]
        true_id = int(true_label.item())
        output = {
            "source": f"FashionMNIST Test[{args.sample_idx}]",
            "true_id": true_id,
            "true_class": cfg.CLASS_NAMES[true_id],
            "correct": bool(pred["predicted_id"] == true_id),
            "prediction": {k: v for k, v in pred.items() if k != "probabilities"},
        }

    if args.benchmark:
        output["latency"] = {
            "bs1": predictor.benchmark_latency(batch_size=1),
            "bs32": predictor.benchmark_latency(batch_size=32),
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
