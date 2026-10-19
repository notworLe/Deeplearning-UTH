"""Reassemble the original 70,000-row FashionMNIST CSV from two ZIP files.
Put this script in the same folder as both ZIP files, then run:
    python restore_fashion_mnist.py
"""
from pathlib import Path
from zipfile import ZipFile
import hashlib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fashion_mnist_full.csv"
EXPECTED_SHA256 = "c8085b78437bd582211c2d3e54b8331bfe3d3b537ad77b8cf87a9250aa49fab3"

def main():
    digest = hashlib.sha256()
    with OUT.open("wb") as target:
        original_header = None
        for i in (1, 2):
            filename = f"fashion_mnist_full_part{i}.csv"
            with ZipFile(ROOT / f"FashionMNIST_part{i}.zip") as z:
                with z.open(filename) as source:
                    header = source.readline()
                    if original_header is None:
                        original_header = header
                        target.write(header)
                        digest.update(header)
                    elif header != original_header:
                        raise RuntimeError("The CSV headers in the ZIP parts do not match")
                    for line in source:
                        target.write(line)
                        digest.update(line)
    checksum = digest.hexdigest()
    if checksum != EXPECTED_SHA256:
        OUT.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch: {checksum}")
    print(f"Restored: {OUT.name} ({OUT.stat().st_size:,} bytes)")
    print("SHA256 verified: original file restored exactly")

if __name__ == "__main__":
    main()
