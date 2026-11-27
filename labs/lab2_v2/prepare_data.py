"""Thin CLI entrypoint thực thi Data Engineering Layer (6.1 Collection -> 6.2 Cleaning -> 6.3 EDA) cho Lab 2 v2."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab2_v2.config as cfg
from src.vibe_scripts.lab2_v2.data import run_data_engineering_pipeline


def main() -> None:
    result = run_data_engineering_pipeline(
        raw_data_root=cfg.RAW_DATA_ROOT,
        eda_dir=cfg.EDA_DIR,
        split_file=cfg.SPLIT_FILE,
        val_ratio_of_test=cfg.VAL_RATIO_OF_TEST,
        seed=cfg.SEED,
        download=False,
    )
    print(
        json.dumps(
            {
                "collection": {
                    "total_samples": result["collection"]["total_samples"],
                    "manifest_mismatches": result["collection"]["manifest_mismatches"],
                    "num_classes": result["collection"]["num_classes"],
                    "cats_vs_dogs_breeds": [
                        result["collection"]["num_cat_breeds"],
                        result["collection"]["num_dog_breeds"],
                    ],
                },
                "cleaning": {
                    "total_scanned": result["cleaning"]["total_scanned"],
                    "corrupt_count": result["cleaning"]["corrupt_count"],
                    "color_mode_counts": result["cleaning"]["color_mode_counts"],
                    "internal_exact_duplicates": result["cleaning"]["internal_exact_duplicates"],
                    "cross_split_exact_duplicates": result["cleaning"]["cross_split_exact_duplicates"],
                    "conflicting_label_duplicates": result["cleaning"]["conflicting_label_duplicates"],
                    "dimensions": result["cleaning"]["dimensions"],
                },
                "eda": {
                    "split_sizes": result["eda"]["split_sizes"],
                    "train_imbalance_ratio": result["eda"]["train_imbalance_ratio"],
                    "species_counts": result["eda"]["species_counts"],
                    "train_rgb_mean": result["eda"]["train_rgb_mean"],
                    "train_rgb_std": result["eda"]["train_rgb_std"],
                },
                "split_file": result["split_file"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
