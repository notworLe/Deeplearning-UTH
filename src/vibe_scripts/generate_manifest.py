"""
Script tự động sinh và kiểm toán manifest.json theo chuẩn ML_PIPELINE.md mục 3.2.
Tạo manifest.json cho data/fashion/raw và data/FashionMNIST/raw.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any


def hash_file(path: Path) -> str:
    """Tính mã băm SHA-256 cho tệp dữ liệu thô."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifests() -> None:
    tz = timezone(timedelta(hours=7))
    collected_at = datetime(2026, 11, 15, 19, 13, 57, tzinfo=tz).isoformat()

    core_artifacts = [
        {
            "file_name": "train-images-idx3-ubyte.gz",
            "dataset_name": "fashion_mnist_train_images",
            "source_uri": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "total_records": 60000,
            "description": "Fashion-MNIST training set images (compressed ubyte)",
        },
        {
            "file_name": "train-labels-idx1-ubyte.gz",
            "dataset_name": "fashion_mnist_train_labels",
            "source_uri": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "total_records": 60000,
            "description": "Fashion-MNIST training set labels (compressed ubyte)",
        },
        {
            "file_name": "t10k-images-idx3-ubyte.gz",
            "dataset_name": "fashion_mnist_test_images",
            "source_uri": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "total_records": 10000,
            "description": "Fashion-MNIST test set images (compressed ubyte)",
        },
        {
            "file_name": "t10k-labels-idx1-ubyte.gz",
            "dataset_name": "fashion_mnist_test_labels",
            "source_uri": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "total_records": 10000,
            "description": "Fashion-MNIST test set labels (compressed ubyte)",
        },
    ]

    target_dirs = [Path("data/fashion/raw"), Path("data/FashionMNIST/raw")]

    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_artifacts: List[Dict[str, Any]] = []

        for art in core_artifacts:
            fpath = target_dir / art["file_name"]
            if not fpath.exists():
                fpath = Path("data/FashionMNIST/raw") / art["file_name"]

            sha = hash_file(fpath)
            sz = fpath.stat().st_size

            art_manifest = {
                "dataset_name": art["dataset_name"],
                "raw_file": f"{target_dir.as_posix()}/{art['file_name']}",
                "source_uri": art["source_uri"],
                "collected_at": collected_at,
                "sha256_checksum": sha,
                "total_records": art["total_records"],
                "byte_size": sz,
                "collection_agent_version": "torchvision_v0.29.2+cu132",
            }
            manifest_artifacts.append(art_manifest)

            # Paired manifest cho từng artifact riêng biệt
            paired_file = target_dir / f"{art['file_name']}.manifest.json"
            with open(paired_file, "w", encoding="utf-8") as pf:
                json.dump(art_manifest, pf, indent=2, ensure_ascii=False)

        # Extracted artifacts (uncompressed ubyte files)
        extracted_artifacts = []
        for art in core_artifacts:
            uncompressed_name = art["file_name"].replace(".gz", "")
            ufpath = target_dir / uncompressed_name
            if not ufpath.exists():
                ufpath = Path("data/FashionMNIST/raw") / uncompressed_name
            if ufpath.exists():
                usha = hash_file(ufpath)
                usz = ufpath.stat().st_size
                extracted_artifacts.append({
                    "file_name": uncompressed_name,
                    "raw_file": f"{target_dir.as_posix()}/{uncompressed_name}",
                    "sha256_checksum": usha,
                    "byte_size": usz,
                    "total_records": art["total_records"]
                })

        # Master manifest.json bao quát toàn bộ dataset theo đúng format §3.2
        master_manifest = {
            "dataset_name": "fashion_mnist",
            "raw_file": f"{target_dir.as_posix()}/train-images-idx3-ubyte.gz",
            "source_uri": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
            "collected_at": collected_at,
            "sha256_checksum": manifest_artifacts[0]["sha256_checksum"],
            "total_records": 70000,
            "collection_agent_version": "torchvision_v0.29.2+cu132",
            "artifacts": manifest_artifacts,
            "extracted_artifacts": extracted_artifacts,
        }

        master_file = target_dir / "manifest.json"
        with open(master_file, "w", encoding="utf-8") as mf:
            json.dump(master_manifest, mf, indent=2, ensure_ascii=False)

        print(f"Đã tạo thành công manifest.json và paired manifests tại: {target_dir}")


if __name__ == "__main__":
    generate_manifests()
