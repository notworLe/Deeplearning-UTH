# Thinking Process: Tích Hợp Manifest.json §3.2 & Hoàn Thiện Lab1 Notebook (Define đến EDA)

- **Date / Time**: 2026-11-16 12:25 (UTC+7)
- **Task Slug**: `lab1_manifest_and_eda_refinement`
- **Agent Role**: Principal Deep-Learning Engineer & Data Architect
- **Status**: `Completed`
- **Related Specs / Rules**:
  - `docs/reference/ML_PIPELINE.md` (§3.1 Immutable Raw Zone, §3.2 Data Provenance & Cryptographic Hashing)
  - `labs/lab1/vibe/docs/REPORT.md` (Kim chỉ nam kỹ thuật)
  - `.agents/rules/` (`SRC.md`, `LOG_AI_THINKING.md`, `LOG_PLANS.md`, `ERROR_LOG.md`)

---

## 1. Bối cảnh & Yêu cầu Người Dùng

1. Người dùng đã đổi tên notebook từ `labs/lab1/vibe/01_dataset.ipynb` thành `labs/lab1/vibe/lab1.ipynb`.
2. Phát hiện vi phạm quy chuẩn quản trị dữ liệu: Tại thư mục dữ liệu thô `data/FashionMNIST/raw/`, thiếu tệp kê khai thu nạp `manifest.json` theo đúng đặc tả tại §3.2 của `docs/reference/ML_PIPELINE.md`.
3. Yêu cầu:
   - Sinh tệp `manifest.json` chuẩn hóa và các manifest ghép cặp cho từng artifact dữ liệu thô tại `data/FashionMNIST/raw/`.
   - Cập nhật notebook `lab1.ipynb` để tải, kiểm toán toàn vẹn mã băm SHA-256 từ `manifest.json` trong Mục 3 (Data Collection & Governance).
   - Thực thi lại notebook để nhúng toàn bộ kết quả bảng biểu Rich và đồ thị trực quan, đảm bảo không có lỗi hiển thị khoảng trắng (Error 0001 đã khắc phục trước đó).
   - Giữ nghiêm ngặt phạm vi kỹ thuật: Chỉ làm từ **Define đến EDA**, tuyệt đối **không làm processing**.

---

## 2. Thiết Kế & Giải Pháp Kỹ Thuật

### 2.1. Cấu trúc Ingestion Manifest (§3.2 Data Provenance)
Mỗi artifact dữ liệu thô lưu tại `data/FashionMNIST/raw/` được ánh xạ và xác thực toàn vẹn bằng mã băm mật mã học (SHA-256):
- `train-images-idx3-ubyte.gz`: `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` (26.42 MB, 60.000 records)
- `train-labels-idx1-ubyte.gz`: `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` (29.5 KB, 60.000 records)
- `t10k-images-idx3-ubyte.gz`: `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` (4.42 MB, 10.000 records)
- `t10k-labels-idx1-ubyte.gz`: `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` (5.1 KB, 10.000 records)

Tạo `data/FashionMNIST/raw/manifest.json` theo đúng schema quy định tại §3.2:
```json
{
  "dataset_name": "fashion_mnist",
  "raw_file": "data/FashionMNIST/raw/train-images-idx3-ubyte.gz",
  "source_uri": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
  "collected_at": "2026-11-15T19:13:57+07:00",
  "sha256_checksum": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
  "total_records": 70000,
  "collection_agent_version": "torchvision_v0.29.2+cu132",
  "artifacts": [...],
  "extracted_artifacts": [...]
}
```
Đồng thời sinh các manifest ghép cặp `<file>.manifest.json` cho từng artifact nén.

### 2.2. Nâng cấp Module `src/vibe_scripts/eda_helpers.py`
- Bổ sung hàm `load_and_verify_manifest(manifest_path: Path)`: Tự động kiểm tra sự tồn tại của tệp, tính toán lại mã băm SHA-256 thực tế trên ổ đĩa và so sánh với giá trị kỳ vọng trong `manifest.json`.
- Bổ sung hàm `print_manifest_table(manifest_data, verify_results)`: Trình bày metadata xuất xứ và kết quả xác thực mật mã học dưới dạng Rich Table trang nhã với trạng thái `[PASS]`.

### 2.3. Cập nhật & Thực thi `labs/lab1/vibe/lab1.ipynb`
- Cập nhật Bảng Lộ Trình (Cell 1): Ghi nhận Step 3 là `Immutable Raw Ingestion & Manifest`.
- Cập nhật Mục 3 Markdown (Cell 5): Bổ sung luận cứ kỹ thuật về §3.1 Immutable Raw Zone và §3.2 Data Provenance & Cryptographic Hashing.
- Cập nhật Mục 3 Code (Cell 6): Tải và gọi hàm kiểm toán `manifest.json`.
- Sử dụng `src/vibe_scripts/execute_lab1_notebook.py` để thực thi toàn bộ các code cell trong môi trường cách ly, thu thập output Rich (Text/HTML) và toàn bộ biểu đồ trực quan base64 sạch sẽ.

---

## 3. Kết Quả Bàn Giao

1. **Tệp Manifest Chuẩn Hóa**:
   - `data/FashionMNIST/raw/manifest.json`
   - `data/FashionMNIST/raw/train-images-idx3-ubyte.gz.manifest.json`
   - `data/FashionMNIST/raw/train-labels-idx1-ubyte.gz.manifest.json`
   - `data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz.manifest.json`
   - `data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz.manifest.json`
2. **Notebook Thực Hành Đã Hoàn Thiện**:
   - `labs/lab1/vibe/lab1.ipynb`: Chạy thành công 100%, tất cả các cell có output trực quan thực tế, mã băm SHA-256 `[PASS]`, biểu đồ cân bằng lớp tỷ lệ chuẩn (1590 x 475), không có khoảng trắng thừa.
3. **Mã Nguồn Hỗ Trợ Tuân Thủ `SRC.md`**:
   - `src/vibe_scripts/eda_helpers.py`
   - `src/vibe_scripts/generate_manifest.py`
   - `src/vibe_scripts/execute_lab1_notebook.py`
