"""Thin CLI entrypoint cho Data Engineering Layer (6.1 Collection -> 6.2 Cleaning -> 6.3 EDA).

Script này độc lập hoàn toàn với `train.py` (không chạy huấn luyện mô hình).
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab1_v2.config as cfg
from src.vibe_scripts.lab1_v2.data import (
    export_eda_artifacts,
    run_data_cleaning_audit,
    run_data_collection_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 1 v2 Data Engineering CLI — 6.1 Collection -> 6.2 Cleaning -> 6.3 EDA"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.EDA_DIR,
        help=f"Thư mục lưu báo cáo & biểu đồ dữ liệu (mặc định: {cfg.EDA_DIR}).",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Bỏ qua bước kiểm tra SHA-256 manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    # 6.1 Data Collection Audit
    collection = run_data_collection_audit(
        data_root=cfg.DATA_PATH,
        output_dir=out_dir,
        verify_manifest=not args.skip_manifest,
    )
    print(
        f"[6.1 COLLECTION] total={collection['records']['total']:,} "
        f"(train={collection['records']['raw_train']:,}, test={collection['records']['raw_test']:,}) | "
        f"manifest_verified={collection['manifest_integrity']['verified']} "
        f"({collection['manifest_integrity']['checked_files']} files)"
    )

    # 6.2 Data Cleaning Audit
    cleaning = run_data_cleaning_audit(
        data_root=cfg.DATA_PATH,
        output_dir=out_dir,
        val_size=cfg.VAL_SIZE,
        seed=cfg.SEED,
        class_names=cfg.CLASS_NAMES,
    )
    dup = cleaning["duplicate_audit"]
    print(
        f"[6.2 CLEANING]   corrupt_or_invalid={cleaning['clean_status']['corrupt_or_invalid_total']} | "
        f"intra_dups(tr/va/te)={dup['train_intra_duplicates']}/{dup['val_intra_duplicates']}/{dup['test_intra_duplicates']} | "
        f"cross_dups(tr<->va={dup['cross_train_val_val_samples_affected']}, "
        f"tr_raw<->te={dup['cross_trainraw_test_test_samples_affected']}) | "
        f"label_conflicts={dup['conflicting_label_duplicate_groups']}"
    )

    # 6.3 Exploratory Data Analysis
    eda = export_eda_artifacts(
        data_root=cfg.DATA_PATH,
        output_dir=out_dir,
        val_size=cfg.VAL_SIZE,
        seed=cfg.SEED,
        class_names=cfg.CLASS_NAMES,
        verify_manifest=not args.skip_manifest,
    )
    px = eda["train_pixel_stats"]
    out = eda["outlier_analysis_iqr"]
    top_pair = eda["prototype_cosine_similarity"]["top_similar_pairs"][0]
    print(
        f"[6.3 EDA]        mean={px['mean']:.4f}, std={px['std']:.4f}, "
        f"sparsity={px['zero_pixel_ratio']*100:.2f}% | "
        f"outliers(low/high)={out['low_intensity_outliers']}/{out['high_intensity_outliers']} | "
        f"top_sim={top_pair['class_a']} <-> {top_pair['class_b']} ({top_pair['cosine_similarity']:.4f})"
    )
    print(f"[DONE] Đã xuất đầy đủ Collection + Cleaning + EDA artifacts vào: {out_dir}")


if __name__ == "__main__":
    main()
