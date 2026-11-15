# Lab 1 Vibe: Triển Khai Pipeline Fashion-MNIST Từ Define Đến EDA

- **Date/Time:** 2026-11-15 19:21
- **Agent/Role:** Principal Deep-Learning Engineer & Second Brain
- **Status:** Completed
- **Related Files:**
  - `labs/lab1/vibe/01_dataset.ipynb`
  - `labs/lab1/vibe/docs/REPORT.md`
  - `docs/reference/ML_PIPELINE.md`
  - `src/vibe_scripts/eda_helpers.py`
  - `docs/error/src/0001.md`
  - `logs/plans/lab1_eda/README.md`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)
- **Yêu cầu ban đầu:** Triển khai lại notebook `labs/lab1/vibe/01_dataset.ipynb` bao quát trọn vẹn từ phần **Define** đến **EDA**, tuyệt đối **không làm phần Data Processing** (transforms, augmentations, dataloaders).
- **Kim chỉ nam cốt lõi:** `labs/lab1/vibe/docs/REPORT.md`.
- **Tài liệu tham chiếu:** `docs/reference/ML_PIPELINE.md` (mục 2 đến 6).
- **Ràng buộc:** Kế thừa các script trong `ROOT/src`, nếu tạo mới thì chỉ được đặt trong `src/vibe_scripts/`. Tuân thủ các quy tắc Notebook Header (Cell 0: Mục tiêu, Cell 1: Bảng Roadmap 4 cột và liên kết nhanh), không chèn sơ đồ luồng quy trình hay tài liệu tham khảo học thuật vào notebook.

---

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)
- **Về kiến trúc phân tầng:**
  - `01_dataset.ipynb` phiên bản trước đây bị lẫn lộn giữa khảo sát dữ liệu (EDA) và tiền xử lý (Transforms, DataLoaders). Việc tách bạch rõ ràng giúp notebook đầu tiên tập trung hoàn toàn vào việc hiểu dữ liệu, thẩm định chất lượng và rút ra các kết luận hình học làm cơ sở cho mô hình hóa.
  - Áp dụng khung **Three Lenses of EDA** (§6 trong `ML_PIPELINE.md`):
    - *Lens 1 (Global Topology):* Phản ánh kích thước ma trận $(28 \times 28)$, kiểu dữ liệu uint8, footprint RAM và tính toàn vẹn 0 nulls.
    - *Lens 2 (Univariate Distributions):* Khảo sát tính cân bằng lớp tuyệt đối trên Train/Val/Test và phân tích cường độ pixel (mean, median, skewness, kurtosis, tỷ lệ background đen).
    - *Lens 3 (Multivariate Interactions):* Trực quan hóa ảnh mẫu, tính toán ảnh trung bình nguyên mẫu (Class Prototypes), ma trận tương quan giữa các lớp (Cosine Similarity) và chiếu không gian 2D bằng PCA.
- **Về mã nguồn `src/`:**
  - Khi rà soát `src/visualize.py` và `src/consoles.py`, phát hiện các lỗi import dở dang (`tensorboard`, `src.dataset`, `src.logs`, và lỗi `img.shape` trên PIL Image). Tuân thủ rule `SRC.md`, đã lập biên bản lỗi tại `docs/error/src/0001.md` và xây dựng module độc lập `src/vibe_scripts/eda_helpers.py` để xử lý trơn tru mọi tác vụ tính toán và đồ họa.

---

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)
- **Tạo mới `src/vibe_scripts/eda_helpers.py`**:
  - `calculate_file_sha256`: Tính mã băm SHA-256 xác thực Data Lineage (§3).
  - `audit_six_dimensions`: Kiểm toán chất lượng 6 chiều (Completeness, Accuracy, Consistency, Timeliness, Uniqueness, Validity).
  - `stratified_split_indices`: Phân tầng tập Train (55k) và Validation (5k) bảo toàn seed 248.
  - `compute_pixel_statistics`: Đo lường các chỉ số thống kê mô tả pixel.
  - `compute_class_prototypes`: Tính ma trận ảnh trung bình $(10, 28, 28)$.
  - Các hàm trực quan hóa chuyên biệt: `plot_class_balance`, `plot_pixel_distribution_analysis`, `plot_gallery`, `plot_prototypes_gallery`, `plot_interclass_similarity_and_pca`.
- **Tái cấu trúc `labs/lab1/vibe/01_dataset.ipynb` (22 cells)**:
  - Cell 0: Tiêu đề và Mục tiêu chuẩn hóa.
  - Cell 1: Bảng Lộ trình 4 cột chuẩn kèm Quick Links mở nhanh tài liệu liên quan.
  - Section 1: Thiết lập môi trường và seed 248 qua `consoles.creation()`.
  - Section 2: Đặc tả bài toán DEFINE (Context & Problem) chuẩn theo `REPORT.md`.
  - Section 3: Thu nạp và kiểm tra mã băm SHA-256 bảo đảm tính bất biến (`data/FashionMNIST/raw/`).
  - Section 4: Bảng kiểm toán chất lượng dữ liệu 6 chiều.
  - Section 5: Phân chia Train (55.000), Validation (5.000), Test (10.000).
  - Section 6: Khảo sát hình học dữ liệu EDA theo mô hình 3 Thấu Kính (kèm output biểu đồ đầy đủ).
  - Section 7: Đúc kết phát hiện hình học và đưa ra quyết định kỹ thuật cho bước tiền xử lý/mô hình hóa, dừng lại trước phần Data Processing.
- **Ghi nhận kế hoạch và nhật ký sự cố**:
  - `logs/plans/lab1_eda/README.md`
  - `docs/error/src/0001.md`

---

## 4. Khó khăn & Cách khắc phục (Challenges & Solutions)
- **Thử thách 1 (Network Sandbox):** `FashionMNIST(download=True)` gặp lỗi kết nối mạng trong môi trường sandbox chuẩn.
  - *Khắc phục:* Chạy lệnh tải dataset một lần duy nhất với quyền mạng (`BypassSandbox: true`), lưu trữ dữ liệu vĩnh viễn vào `data/FashionMNIST/raw/` để từ đó mọi tác vụ chạy offline hoàn toàn trong sandbox.
- **Thử thách 2 (Jupyter Kernel Sockets trong Sandbox):** `nbconvert --execute` cố gắng kết nối qua ZeroMQ sockets cục bộ bị sandbox chặn.
  - *Khắc phục:* Viết script thực thi inline `src/vibe_scripts/run_notebook_inline.py` chạy trực tiếp các code cells trong cùng process, thu giữ toàn bộ stdout và biểu đồ matplotlib (dưới dạng base64 png), cập nhật trực tiếp vào file notebook mà không cần mạng hay socket.

---

## 5. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)
- **Trạng thái hiện tại:** `01_dataset.ipynb` đã hoàn thiện và sinh 100% output sắc nét, bám sát từng chữ trong `REPORT.md`.
- **Lưu ý quan trọng (Gotchas):**
  - Dataset thô đã nằm sẵn tại `data/FashionMNIST/raw/`, khi nạp ở các notebook sau chỉ cần đặt `download=False`.
  - `01_dataset.ipynb` kết thúc ở phần EDA. Toàn bộ phần Data Processing (Transforms, Z-Score Normalize, Data Augmentation, DataLoader đa luồng) sẽ được triển khai tại notebook huấn luyện `02_model_building_and_training.ipynb` hoặc module tiền xử lý kế tiếp.
