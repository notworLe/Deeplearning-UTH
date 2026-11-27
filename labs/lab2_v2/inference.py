"""Thin CLI entrypoint suy luận Top-5 Softmax & đo độ trễ cho Lab 2 v2 (Oxford-IIIT Pets)."""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab2_v2.config as cfg
from src.vibe_scripts.lab2_v2.data import prepare_oxford_pets
from src.vibe_scripts.lab2_v2.inference import PetPredictor


def _resolve_default_checkpoint() -> Path:
    summary_path = cfg.ABLATION_DIR / "ablation_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        winner = data.get("winning_model_run_id")
        if winner:
            return cfg.RUNS_DIR / str(winner) / "best.pt"
    return cfg.RUNS_DIR / "resnet18_finetune" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 2 v2 Inference CLI — Top-5 Pet Breed Prediction & Latency Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint `.pt` (mặc định: winning model).")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn ảnh PNG/JPG bên ngoài để dự đoán.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Chỉ số ảnh mẫu trong tập Test (khi không truyền --image).")
    parser.add_argument("--top-k", type=int, default=5, help="Số lớp xác suất cao nhất cần in (mặc định: 5).")
    parser.add_argument("--benchmark", action="store_true", help="Chạy đo độ trễ suy luận (BS=1 & BS=32).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _resolve_default_checkpoint()
    predictor = PetPredictor(checkpoint_path=ckpt_path, device=cfg.DEVICE)

    output = {}
    if args.image:
        output["prediction"] = predictor.predict_image(args.image, top_k=args.top_k)
        output["source"] = str(args.image)
    else:
        bundle = prepare_oxford_pets(
            raw_data_root=cfg.RAW_DATA_ROOT,
            batch_size=1,
            val_ratio_of_test=cfg.VAL_RATIO_OF_TEST,
            seed=cfg.SEED,
            image_size=cfg.IMAGE_SIZE,
            resize_size=cfg.RESIZE_SIZE,
            num_workers=0,
            split_file=cfg.SPLIT_FILE,
            download=False,
        )
        idx = int(args.sample_idx)
        img_path = bundle.image_paths["test"][idx]
        true_id = int(bundle.labels["test"][idx])
        true_name = bundle.class_names[true_id]
        pred = predictor.predict_image(img_path, top_k=args.top_k)
        output["source"] = f"test[{idx}] ({img_path.name})"
        output["ground_truth"] = {
            "class_id": true_id,
            "class_name": true_name,
            "species": bundle.species_by_class.get(true_name, "Unknown"),
        }
        output["prediction"] = pred
        output["is_correct"] = pred["predicted_id"] == true_id

    if args.benchmark:
        output["benchmark"] = predictor.benchmark_latency(batch_sizes=(1, 32))

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
