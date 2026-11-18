---
title: "Pipeline 01: Data Preparation — Lab 1 (Fashion-MNIST)"
doc_type: pipeline-spec
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-18
updated: 2026-11-18
last_updated_by: Antigravity
version: 0.1.0
status: draft
reviewers: []
related_docs:
  - ../README.md
  - ./README.md
  - ./02_MODEL_ARCHITECTURE.md
  - ../REPORT.md
  - ../../lab1.ipynb
  - ../../../../../.agents/temp/06_PIPELINE_SEPARATION.md
  - ../../../../../.agents/rules/BEHAVIORAL_LAYER.md
tags: [lab1, pipeline-01, data-preparation, fashion-mnist, dataloader]
---

# Pipeline 01: Data Preparation (Chuẩn Bị Dữ Liệu & DataLoader)

Tài liệu đặc tả độc lập cho **Pipeline 1** trong hệ thống phân tách 5 pipeline của Lab 1, bám sát mục `Dataset` và `Data processing` tại [`labs/lab1/vibe/docs/REPORT.md`](../REPORT.md).

---

## 1. Tài Liệu Đặc Tả Kỹ Thuật

### 1.1. Phạm vi & Ranh giới trách nhiệm
- **Đã hoàn thành ở bước trước ([`labs/lab1/vibe/lab1.ipynb`](../../lab1.ipynb))**:
  - Thu nạp bất biến tại `data/FashionMNIST/raw/` kèm xác thực SHA-256 từ `manifest.json` (§3.2).
  - Kiểm toán chất lượng 6 chiều → xác nhận bộ dữ liệu sạch, bỏ qua bước Data Cleaning.
  - Khảo sát EDA 3 Thấu kính.
- **Trách nhiệm triển khai của Pipeline 01**:
  - Xác thực toàn vẹn `manifest.json` trước khi đọc dữ liệu (Fail-Fast).
  - Phân chia phân tầng (stratified split) tập Train gốc 60.000 mẫu thành **Train 55.000** và **Validation 5.000** với `seed = 248`; giữ nguyên **Test 10.000**.
  - Tính `mean` và `std` trên đúng tập **Train 55.000** ở thang `[0, 1]` để loại trừ hoàn toàn rò rỉ dữ liệu từ Validation và Test.
  - Xây dựng đúng 2 nhánh biến đổi dữ liệu phục vụ ma trận thí nghiệm: `no_aug` và `aug` (Augmentation chỉ áp dụng lên tập Train).
  - Khởi tạo `train_loader`, `val_loader`, `test_loader` tái lập hoàn toàn với `seed = 248`, dùng chung cho cả 3 mô hình (`Logistic`, `MLP`, `CNN`).
- **Ngoài phạm vi (Out-of-Scope)**: Không chứa định nghĩa mô hình, loss, optimizer hay vòng lặp huấn luyện.

### 1.2. Hợp đồng Đầu vào / Đầu ra (Input / Output Contracts)
- **Input**:
  - `data_dir: Path` — thư mục chứa `FashionMNIST/raw/` và `manifest.json` (mặc định `data/`).
  - `aug_enabled: bool` — `False` cho nhánh `no_aug`, `True` cho nhánh `aug`.
  - `batch_size: int = 32`, `num_workers: int = 8`, `seed: int = 248`.
- **Output**:
  - `train_loader`, `val_loader`, `test_loader`: PyTorch `DataLoader`.
  - Mỗi batch trả về `(images, labels)` với:
    - `images`: `torch.Tensor`, shape `[B, 1, 28, 28]`, dtype `torch.float32`, đã chuẩn hóa Z-Score.
    - `labels`: `torch.Tensor`, shape `[B]`, dtype `torch.int64`, giá trị trong `{0, ..., 9}`.
  - `norm_stats: dict` gồm `{"mean": float, "std": float, "train_size": 55000, "val_size": 5000, "test_size": 10000, "seed": 248}`.

### 1.3. Luồng Biến Đổi Dữ Liệu (Transform Order & Leakage Boundary)

Bắt buộc giữ đúng thứ tự trong `REPORT.md`: **`Augmentation → ToTensor → Normalize`**.

1. **Tách tập trước khi gắn Transform (Golden Leakage Boundary)**:
   - Không dùng `random_split` trên một đối tượng `Dataset` đã gắn sẵn `train_transform` (vì sẽ khiến `val_subset` vô tình bị áp dụng augmentation!).
   - Khởi tạo 2 đối tượng `FashionMNIST(train=True, download=False)` riêng biệt — một gắn `train_transform` cho `Subset(..., train_idx)` và một gắn `eval_transform` cho `Subset(..., val_idx)` — dùng chung bộ chỉ số `train_idx`, `val_idx` từ `eda_helpers.stratified_split_indices(..., seed=248)`.
2. **Nhánh A (`no_aug`) — Áp dụng cho Train (`no_aug`), Validation và Test**:
   - `ToTensor()`: chuyển pixel `0–255` sang `[0.0, 1.0]`, shape `[1, 28, 28]`.
   - `Normalize(mean=(train_mean,), std=(train_std,))`: đưa trung bình về `0`, độ lệch chuẩn về `1`.
3. **Nhánh B (`aug`) — Chỉ áp dụng cho Train (`aug`)**:
   - Lật ngang: `RandomHorizontalFlip(p=0.5)` (bảo toàn ngữ nghĩa trang phục thời trang).
   - Crop / dịch nhẹ và xoay nhỏ: `RandomAffine(degrees=10, translate=(0.08, 0.08))` hoặc `RandomCrop(28, padding=2)` (tham số góc xoay và biên dịch chờ kỹ sư chốt trước khi code).
   - `ToTensor()` → `Normalize(mean=(train_mean,), std=(train_std,))`.
   - *Lưu ý*: Không dùng `RandomVerticalFlip` hay xoay góc lớn vì quần áo/giày dép Fashion-MNIST có hướng đứng cố định theo trọng lực.

### 1.4. Cấu Hình Chuẩn Hóa (YAML Spec)

```yaml
dataset:
  name: fashion_mnist
  root: data
  manifest_path: data/FashionMNIST/raw/manifest.json
  verify_manifest: true
  split:
    strategy: stratified
    train_size: 55000
    val_size: 5000
    test_size: 10000
    seed: 248
  normalization:
    fit_on: train_only
  augmentation:
    enabled: false          # false -> no_aug | true -> aug
    horizontal_flip_p: 0.5
    rotation_degrees: 10    # [Cần kỹ sư chốt]
    translate_ratio: 0.08   # [Cần kỹ sư chốt]
    crop_padding: 2         # [Cần kỹ sư chốt]
  dataloader:
    batch_size: 32
    num_workers: 8
    shuffle_train: true
    drop_last_train: false
    pin_memory: auto        # True khi device == cuda
```

### 1.5. Ánh Xạ Thư Mục & Rủi Ro Kỹ Thuật
- **Mã nguồn dự kiến (`SRC.md`)**: `src/vibe_scripts/lab1/data.py` (kế thừa `load_and_verify_manifest` và `stratified_split_indices` từ [`src/vibe_scripts/eda_helpers.py`](../../../../../src/vibe_scripts/eda_helpers.py)).
- **Kiểm thử dự kiến**: `test/test_lab1_data.py`.
- **Rủi ro & Biện pháp phòng ngừa**:
  - *Leakage khi gắn transform*: Giải quyết bằng 2 bản thể `FashionMNIST` tách rời cho `train_subset` và `val_subset`.
  - *Lỗi `src/datasets.py` cũ*: Giữ nguyên `src/datasets.py` (có hàm `load_train_val_test` chưa hoàn thiện và `drop_last=(DEVICE == "cuda")` sai logic ở test loader), triển khai sạch tại `src/vibe_scripts/lab1/data.py`.
  - *Multiprocessing trên Windows (`num_workers=8`)*: Truyền `worker_init_fn` khóa seed cho từng worker, bảo vệ điểm vào bằng `if __name__ == "__main__":`, và đặt `persistent_workers=(num_workers > 0)` để giảm chi phí khởi tạo lại worker mỗi epoch.

---

## 2. Prompt Độc Lập Cho Pipeline 01 (Behavioral Schema)

```markdown
Name:
Lab 1 Data Preparation Pipeline

Description:
Xây dựng module chuẩn bị dữ liệu Fashion-MNIST bao gồm xác thực manifest SHA-256, chia tầng 55.000/5.000/10.000, tính tham số chuẩn hóa trên tập Train, dựng 2 nhánh transform (no_aug và aug) và khởi tạo 3 PyTorch DataLoader tái lập theo seed 248.

Purpose:
Cung cấp luồng dữ liệu sạch, không data leakage, bảo đảm cả 3 mô hình (Logistic, MLP, CNN) nhận cùng phân bố đầu vào chuẩn hóa và cùng thứ tự batch tái lập.

Input:
- Thư mục dữ liệu thô bất biến: `data/FashionMNIST/raw/` và `data/FashionMNIST/raw/manifest.json`
- Cấu hình phân chia: Train 55.000, Val 5.000 (stratified split), Test 10.000, `seed = 248`
- Cờ cấu hình: `aug_enabled` (bool), `batch_size = 32`, `num_workers = 8`

Output:
- Module `src/vibe_scripts/lab1/data.py` cung cấp hàm `build_dataloaders(...)`
- Trả về `(train_loader, val_loader, test_loader, norm_stats)` với batch shape `[B, 1, 28, 28]` (`float32`) và nhãn `[B]` (`int64`)
- Bộ unit test `test/test_lab1_data.py`

How to do:
1. Gọi `eda_helpers.load_and_verify_manifest()` xác thực SHA-256 trên `data/FashionMNIST/raw/manifest.json`; dừng ngay (Fail-Fast) nếu có file không khớp.
2. Gọi `eda_helpers.stratified_split_indices()` trên 60.000 nhãn train với `seed = 248` để thu bộ chỉ số `train_idx` (55.000) và `val_idx` (5.000).
3. Tính `train_mean`, `train_std` bằng tensor vector hóa trực tiếp trên đúng `train_idx` (chia 255.0), không đụng tới `val_idx` và `test`.
4. Dựng `eval_transform = Compose([ToTensor(), Normalize((train_mean,), (train_std,))])` và `train_transform` theo thứ tự `Augmentation -> ToTensor -> Normalize` khi `aug_enabled=True` (ngược lại dùng `eval_transform`).
5. Khởi tạo 2 đối tượng `FashionMNIST(train=True, download=False)` riêng biệt gắn `train_transform` và `eval_transform`, bọc bằng `Subset` để đảm bảo `val_loader` tuyệt đối không dính augmentation.
6. Tạo `train_loader` (`shuffle=True` với `torch.Generator().manual_seed(248)`), `val_loader` và `test_loader` (`shuffle=False`, `drop_last=False`), có `worker_init_fn`.
7. Viết Smoke Test và Normal Test tại `test/test_lab1_data.py` kiểm chứng toàn bộ checklist.

Examples:
pytest test/test_lab1_data.py -v
```

---

## 3. Checklist Kiểm Thử Độc Lập (Definition of Done)

- [ ] **Toàn vẹn dữ liệu**: `manifest.json` kiểm toán SHA-256 đạt `[PASS]` 8/8 tệp; hàm báo lỗi tường minh nếu giả lập sai hash.
- [ ] **Không Data Leakage**: `set(train_idx).isdisjoint(set(val_idx))` là `True`; `len(train_idx) == 55000`, `len(val_idx) == 5000`, `len(test) == 10000`.
- [ ] **Cân bằng phân tầng**: Mỗi lớp trong 10 lớp có đúng `5.500` mẫu ở Train, `500` ở Val và `1.000` ở Test.
- [ ] **Bất biến Scaling**: `train_mean` và `train_std` tính duy nhất trên 55.000 ảnh Train; sau chuẩn hóa, trung bình toàn tập Train `no_aug` xấp xỉ `0.0` (sai số `< 1e-3`) và độ lệch chuẩn xấp xỉ `1.0` (sai số `< 1e-3`).
- [ ] **Tách biệt Augmentation**: Với `aug_enabled=True`, gọi lặp lại cùng chỉ số trên `val_loader` và `test_loader` cho tensor giống hệt nhau (`torch.equal == True`), trong khi trên `train_loader` tạo biến thể hình học hợp lệ.
- [ ] **Hợp đồng Tensor**: Batch sinh ra có `images.shape == (32, 1, 28, 28)`, `images.dtype == torch.float32`, `labels.shape == (32,)`, `labels.dtype == torch.int64`, không chứa `NaN`/`Inf`.
- [ ] **Tái lập ngẫu nhiên**: Hai lần gọi `build_dataloaders(..., seed=248, aug_enabled=False)` sinh ra đúng cùng thứ tự nhãn ở batch đầu tiên của `train_loader`.
- [ ] **Hiệu năng nạp dữ liệu**: Smoke test đọc 10 batch hoàn tất `< 5s`, không deadlock trên Windows với `num_workers`.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.1.0 | 2026-11-18 | notworLe | Antigravity | Khởi tạo đặc tả độc lập Pipeline 01 (Data Preparation) theo `REPORT.md` và `PIPELINE_SEPARATION.md` |
