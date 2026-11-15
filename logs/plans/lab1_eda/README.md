# Kế Hoạch Triển Khai: Lab 1 Vibe Dataset Pipeline (Define đến EDA)

Tài liệu kế hoạch theo quy định tại `.agents/rules/logs/LOG_PLANS.md`.

## 1. Mục tiêu & Phạm vi
- **Mục tiêu**: Xây dựng lại notebook `labs/lab1/vibe/01_dataset.ipynb` bao phủ trọn vẹn từ phần **Define** đến **EDA**, tuyệt đối không thực hiện phần **Data processing** (transforms, augmentations, dataloaders).
- **Kim chỉ nam cốt lõi**: `labs/lab1/vibe/docs/REPORT.md`.
- **Tài liệu tham chiếu phương pháp luận**: `docs/reference/ML_PIPELINE.md` (mục 2 đến 6: Data Understanding, Acquisition, Lineage & Governance, Quality Audit, Integration, và EDA 3 thấu kính).
- **Phạm vi mã nguồn**: Kế thừa và sử dụng các module tại `ROOT/src`; các module bổ trợ riêng của AI được đặt tại `src/vibe_scripts/`. Ghi nhận lỗi của `src/` vào `docs/error/src/0001.md`.

---

## 2. Các Giai đoạn Thực hiện (Phased Execution Plan)

### Giai đoạn 1: Ghi nhận Lỗi của `src/` & Xây dựng Module Bổ trợ tại `src/vibe_scripts/`
- Ghi nhận lỗi thiếu package tensorboard / import sai tên `src.dataset` / thiếu `src.logs` trong `src/visualize.py` vào `docs/error/src/0001.md`.
- Xây dựng `src/vibe_scripts/eda_helpers.py` chứa các hàm chuyên trách phục vụ EDA:
  - Kiểm tra tính toàn vẹn (Integrity & SHA-256 Checksum).
  - Khảo sát Thấu kính 1 (Global Topology: shapes, memory, null count).
  - Khảo sát Thấu kính 2 (Univariate Distributions: phân bố lớp, thống kê pixel skewness, kurtosis, sparsity).
  - Khảo sát Thấu kính 3 (Multivariate Interactions: ảnh trung bình từng lớp, ma trận tương quan giữa các lớp, trích xuất đặc trưng PCA 2D/3D).

### Giai đoạn 2: Thiết kế Cấu trúc Notebook `labs/lab1/vibe/01_dataset.ipynb`
- **Cell 0**: Header & Mục tiêu thực nghiệm (bám sát `REPORT.md`).
- **Cell 1**: Bảng Lộ trình Thực thi 4 cột chuẩn (`Step`, `Description`, `What it does`, `Import path` là clickable link tương đối) kèm Danh mục Liên kết Nhanh Tài liệu Kỹ thuật. Tuyệt đối không chèn sơ đồ workflow hay references học thuật vào notebook.
- **Section 1: Định nghĩa Bài toán & Bối cảnh (Define Context & Problem)**:
  - Bối cảnh bài toán phân loại Fashion-MNIST 10 lớp, mục tiêu nghiên cứu, ràng buộc công bằng thí nghiệm, baselines và metrics.
- **Section 2: Thu Nạp & Quản Trị Dữ Liệu Bất Biến (Data Collection & Governance - §2, §3)**:
  - Tải dữ liệu Fashion-MNIST qua torchvision vào thư mục bất biến `data/`.
  - Khảo sát metadata (70.000 ảnh: 60k train, 10k test, kích thước 28x28, thang đo tỉ lệ 0-255).
  - Kiểm tra tính toàn vẹn và lineage bằng mã băm SHA-256.
- **Section 3: Kiểm Toán Chất Lượng Dữ Liệu 6 Chiều & Làm Sạch (Data Quality Audit - §4, §5)**:
  - Đánh giá 6 chiều chất lượng dữ liệu: Completeness, Accuracy, Consistency, Timeliness, Uniqueness, Validity.
  - Xác thực bộ dữ liệu sạch chuẩn mực $\rightarrow$ Kết luận bỏ qua data cleaning.
- **Section 4: Phân Chia Dữ Liệu Thực Nghiệm Chuẩn Mực (Dataset Splitting)**:
  - Thực hiện phân chia Train (55.000), Validation (5.000), Test (10.000) bảo toàn seed 248 và cân bằng lớp (stratified split).
- **Section 5: Khảo Sát Hình Học Dữ Liệu Toàn Diện (EDA - Three Lenses Framework - §6)**:
  - **Lens 1: Cấu trúc Hình học Tổng thể (Global Topology)**: Shape, dtypes, memory footprint, null count.
  - **Lens 2: Phân bố Đơn biến (Univariate Distributions)**: Cân bằng lớp trên 3 tập (Train/Val/Test), phân bố cường độ pixel (mean, median, skewness, kurtosis, tỷ lệ background đen).
  - **Lens 3: Tương tác Đa biến & Khả năng Phân tách Lớp (Multivariate & Visual Separability)**: Thư viện ảnh mẫu 10 lớp, ảnh đại diện trung bình (Class Prototypes), bản đồ tương quan giữa các lớp, trực quan hóa không gian PCA 2D.
- **Section 6: Kết Luận Kỹ Thuật & Quyết Định Cho Giai Đoạn Tiếp Theo**:
  - Đưa ra quyết định cho việc scaling và chuẩn hóa ($z$-score), lựa chọn augmentation phù hợp, lý do cần mô hình trích xuất đặc trưng không gian (CNN) so với baseline (Logistic/MLP).
  - **Dừng lại tại EDA, không thực hiện Data Processing (transforms, augmentation, dataloader)** theo đúng yêu cầu đề bài.

### Giai đoạn 3: Kiểm Thử Chạy Thực Tế (Execution & Sanity Verification)
- Chạy toàn bộ notebook `01_dataset.ipynb` qua Jupyter client/nbconvert để bảo đảm mọi cell sinh output thực tế, biểu đồ trực quan sắc nét và không có bất kỳ ngoại lệ (zero crash).

### Giai đoạn 4: Hoàn Tất Báo Cáo Luồng Tư Duy (Thinking Log & Summary)
- Ghi log tư duy vào `logs/thinking/` và cập nhật `logs/thinking/INDEX.md`.
