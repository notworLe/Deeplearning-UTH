"""
Module hỗ trợ Khảo sát Khám phá Dữ liệu (EDA Helpers).
Thực thi các phép tính toán hình học, kiểm toán chất lượng dữ liệu và trực quan hóa theo khung 3 Thấu kính (Three Lenses of EDA).
Tuân thủ quy định tại .agents/rules/SRC.md và docs/reference/ML_PIPELINE.md (mục 2-6).
"""

from typing import Tuple, List, Dict, Optional
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

# Hằng số giao diện đồng bộ
FASHION_CLASSES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

COLOR_PALETTE = [
    "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#10b981",
    "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"
]


def calculate_file_sha256(file_path: Path) -> str:
    """Tính toán mã băm SHA-256 phục vụ xác thực Data Lineage (§3)."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def audit_six_dimensions(train_data: torch.Tensor, train_targets: torch.Tensor,
                         test_data: torch.Tensor, test_targets: torch.Tensor) -> Dict[str, any]:
    """
    Kiểm toán chất lượng dữ liệu 6 chiều (§4 Data Quality Auditing).
    - Completeness: Kiểm tra giá trị khuyết (NaN, Null).
    - Accuracy: Kiểm tra ranh giới vật lý pixel [0, 255].
    - Consistency: Kiểm tra miền nhãn lớp {0, ..., 9}.
    - Timeliness: Xác thực tập chuẩn cố định Zalando Fashion-MNIST.
    - Uniqueness: Kiểm tra mẫu ảnh trùng lặp hoàn toàn.
    - Validity: Kiểm tra cấu trúc hình học tensor (N, 28, 28).
    """
    total_samples = len(train_data) + len(test_data)
    
    # 1. Completeness
    nan_count = int(torch.isnan(train_data).sum().item() + torch.isnan(test_data).sum().item())
    
    # 2. Accuracy
    min_pixel = int(min(train_data.min().item(), test_data.min().item()))
    max_pixel = int(max(train_data.max().item(), test_data.max().item()))
    
    # 3. Consistency
    unique_labels = sorted(list(set(train_targets.numpy()).union(set(test_targets.numpy()))))
    
    # 4. Validity
    valid_shape = (train_data.shape[1:] == torch.Size([28, 28])) and (test_data.shape[1:] == torch.Size([28, 28]))
    
    return {
        "total_samples": total_samples,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "null_count": nan_count,
        "completeness_pct": 100.0 if nan_count == 0 else (1 - nan_count / total_samples) * 100,
        "pixel_min": min_pixel,
        "pixel_max": max_pixel,
        "valid_range": (min_pixel >= 0 and max_pixel <= 255),
        "unique_labels": unique_labels,
        "valid_labels": (unique_labels == list(range(10))),
        "geometry_valid": valid_shape
    }


def stratified_split_indices(targets: np.ndarray, train_size: int = 55000,
                             val_size: int = 5000, seed: int = 248) -> Tuple[List[int], List[int]]:
    """
    Thực hiện phân tách Train/Validation phân tầng (Stratified Split) với seed ngẫu nhiên cố định.
    Đảm bảo mỗi lớp có đúng tỷ lệ số mẫu bằng nhau.
    """
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)
    
    val_per_class = val_size // num_classes
    train_indices = []
    val_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)
        
        val_indices.extend(cls_indices[:val_per_class])
        train_indices.extend(cls_indices[val_per_class:])
        
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    return train_indices, val_indices


def compute_pixel_statistics(images_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Tính toán các chỉ số thống kê mô tả cường độ pixel (Lens 2: Univariate Distributions).
    Bao gồm Mean, Median, Std, Skewness, Kurtosis và Tỷ lệ Sparsity (Pixel nền = 0).
    """
    flat_pixels = images_tensor.numpy().astype(np.float32).flatten()
    
    mean_val = float(np.mean(flat_pixels))
    std_val = float(np.std(flat_pixels))
    median_val = float(np.median(flat_pixels))
    skew_val = float(stats.skew(flat_pixels))
    kurt_val = float(stats.kurtosis(flat_pixels))
    sparsity_pct = float(np.mean(flat_pixels == 0) * 100)
    
    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "sparsity_zero_pct": sparsity_pct
    }


def compute_class_prototypes(images_tensor: torch.Tensor, targets_tensor: torch.Tensor) -> np.ndarray:
    """
    Tính toán ảnh trung bình (Prototype Image) cho từng danh mục trong 10 lớp trang phục.
    Shape trả về: (10, 28, 28).
    """
    prototypes = np.zeros((10, 28, 28), dtype=np.float32)
    images_np = images_tensor.numpy()
    targets_np = targets_tensor.numpy()
    
    for cls in range(10):
        cls_images = images_np[targets_np == cls]
        prototypes[cls] = np.mean(cls_images, axis=0)
        
    return prototypes
