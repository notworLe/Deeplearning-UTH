# AI REFERENCE - BỘ QUY CHUẨN THAM CHIẾU BẮT BUỘC CHO AI AGENT

AI Agent phải tuyệt đối tuân thủ toàn bộ quy chuẩn trong tài liệu này. Bất kỳ sai lệch nào sẽ bị từ chối ở bước Codebase Audit.

---

## 1. Quy chuẩn Đặt tên (Naming Conventions)

| Đối tượng                       | Quy chuẩn                                                    | Ví dụ Chuẩn (Good)                                   | Ví dụ Sai (Bad)                          |
| :------------------------------ | :----------------------------------------------------------- | :--------------------------------------------------- | :--------------------------------------- |
| **Tên Thư mục**                 | `snake_case` hoặc kebab-case ngắn                            | `models/`, `data_loader/`, `pre_trained/`            | `Models/`, `dataLoader/`, `MyData/`      |
| **Tên File Markdown**           | VIẾT HOA TOÀN BỘ (`UPPER_SNAKE_CASE.md` hoặc `UPPERCASE.md`) | `README.md`, `REPORT.md`, `TEMPLATE.md`              | `readme.md`, `report.md`, `template.md`  |
| **Tên File Python**             | `snake_case.py`                                              | `datamodule.py`, `metrics.py`, `train.py`            | `DataModule.py`, `my_Train.py`           |
| **Tên File Cấu hình**           | `snake_case.yaml`                                            | `default.yaml`, `template.yaml`                      | `Default.YAML`, `my-config.yml`          |
| **Tên Class**                   | `PascalCase`                                                 | `StandardCVDataset`, `CVDataModule`, `CustomCNN`     | `standard_cv_dataset`, `cvDataModule`    |
| **Tên Function / Method**       | `snake_case` (bắt đầu bằng động từ)                          | `calculate_iou()`, `apply_filter()`, `build_model()` | `CalculateIoU()`, `iou()`, `DoProcess()` |
| **Tên Biến thông thường**       | `snake_case`                                                 | `image_tensor`, `batch_size`, `learning_rate`        | `ImageTensor`, `bs`, `lrVal`             |
| **Tên Hằng số**                 | `UPPER_SNAKE_CASE`                                           | `DEFAULT_SEED`, `IMAGE_SIZE`, `NUM_CLASSES`          | `default_seed`, `ImageSize`              |
| **Tên Biến Riêng tư (Private)** | `_leading_snake_case`                                        | `_init_weights()`, `_cached_features`                | `__init_weights()`, `privateFeatures`    |

---

## 2. Quy chuẩn Cấu trúc Thư mục (Folder Structure Invariants)

AI không được tự ý tạo thêm thư mục gốc mới ngoài danh mục đã được phê duyệt:

```text
<project-root>/
├── configs/          # CHỈ chứa file .yaml cấu hình (Hydra)
├── data/             # CHỈ chứa dữ liệu thô, trung gian và processed (ignored by Git)
├── docs/             # CHỈ chứa tài liệu hướng dẫn và báo cáo kỹ thuật markdown (Tên file VIẾT HOA)
├── experiments/      # CHỈ chứa output sinh ra tự động: checkpoints, logs, predictions
├── notebooks/        # CHỈ chứa notebook tương tác .ipynb
├── requirements/     # CHỈ chứa danh sách dependencies theo môi trường .txt
├── scripts/          # CHỈ chứa shell scripts tự động hóa .sh
├── src/              # CHỈ chứa mã nguồn Python chính thức của package
└── tests/            # CHỈ chứa unit tests và fixtures phục vụ pytest
└── plans/
└── prompts/
└── logs/

```

---

## 3. Quy chuẩn Conventional Commits

Mọi commit do AI đề xuất hoặc kỹ sư thực hiện phải tuân thủ nghiêm ngặt cú pháp:

```text
<type>(<scope>): <subject>

[optional body: mô tả nguyên nhân và giải pháp kỹ thuật]

[optional footer: liên kết issue hoặc breaking changes]
```

### Danh mục Types cho Computer Vision:

- `feat`: Thêm tính năng mới (thuật toán lọc, kiến trúc model, metric mới, datamodule).
- `fix`: Sửa lỗi (shape mismatch, zero-division, NaN loss, off-by-one index).
- `perf`: Tối ưu hóa hiệu năng (vectorization, DataLoader prefetching, GPU memory).
- `refactor`: Tái cấu trúc mã nguồn mà không làm thay đổi tính năng hay kết quả.
- `test`: Thêm hoặc cập nhật unit test, smoke test, benchmark fixtures.
- `docs`: Cập nhật tài liệu kỹ thuật, docstring, báo cáo thí nghiệm.
- `chore`: Thay đổi cấu hình build, pre-commit, package dependencies.

### Danh mục Scopes chuẩn:

`data`, `models`, `pipelines`, `metrics`, `visualizer`, `configs`, `tests`, `scripts`

### Quy tắc commit:

1. Viết subject ở thể mệnh lệnh hiện tại (`implement`, `add`, `fix`, không dùng `implemented`, `fixing`).
2. Subject không viết hoa chữ đầu, không kết thúc bằng dấu chấm, tối đa 72 ký tự.
3. Commit nguyên tử (Atomic Commit): Một commit chỉ giải quyết trọn vẹn một tác vụ độc lập.

---

## 4. Bất biến Kỹ thuật Ngành Thị giác Máy tính (CV/DL Domain Invariants)

Mọi dòng code xử lý hình ảnh và mô hình học sâu phải tuân thủ nghiêm ngặt 5 tiêu chuẩn kỹ thuật:

### 4.1. Quy chuẩn Tensor & Kích thước Chiều (Shape Conventions)

- **OpenCV / NumPy**: Luôn ở định dạng `(H, W, C)` hoặc `(H, W)` (grayscale), kiểu dữ liệu `uint8` [0, 255] hoặc `float32` [0.0, 1.0].
- **PyTorch**: Luôn ở định dạng `(B, C, H, W)` (batch) hoặc `(C, H, W)` (đơn mẫu), kiểu dữ liệu `float32`.
- **Docstrings**: Bắt buộc ghi rõ shape tensor đầu vào và đầu ra trong mọi hàm xử lý ảnh.

### 4.2. Nhận diện Không gian Màu (Color Space Awareness)

- **OpenCV**: Đọc và ghi mặc định ở hệ màu `BGR`.
- **PIL / Matplotlib / Torchvision**: Sử dụng mặc định hệ màu `RGB`.
- **Quy tắc**: Phải chuyển đổi tường minh qua `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` khi trao đổi dữ liệu giữa OpenCV và Torchvision / Matplotlib. Cấm để lệch màu ẩn.

### 4.3. Định dạng Tọa độ Hộp bao (Bounding Box Formats)

Không bao giờ ngầm định định dạng hộp bao. Phải ghi rõ 1 trong các chuẩn sau:

- `XYXY` / Pascal VOC: `[x1, y1, x2, y2]` (pixel tuyệt đối).
- `XYWH` / COCO: `[x_min, y_min, width, height]` (pixel tuyệt đối).
- `CXCYWH` / YOLO: `[cx, cy, w, h]` (chuẩn hóa về khoảng [0.0, 1.0]).

### 4.4. Chuẩn hóa Tính Tái lập Thí nghiệm (Reproducibility Mandate)

Mọi thí nghiệm phải sử dụng hàm seed tập trung để khóa ngẫu nhiên trên toàn bộ môi trường:

- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`

### 4.5. Quản lý Tài nguyên GPU & Tối ưu Băng thông (GPU Hygiene)

- Không hardcode `.cuda()` hoặc `'cuda:0'`, luôn dùng `device = torch.device(...)` nhận từ file cấu hình.
- DataLoader trên GPU phải bật `pin_memory=True` và cấu hình `num_workers > 0` phù hợp với số CPU cores.
- Sử dụng `torch.cuda.amp.autocast()` khi huấn luyện để tối ưu tốc độ và dung lượng VRAM.

---

## 5. Quy chuẩn Bắt buộc cho Jupyter Notebooks (Notebook Standard Blueprint)

Mọi Jupyter Notebook thực hành (`notebooks/labs/*.ipynb` hoặc `notebooks/exploratory/*.ipynb`) BẮT BUỘC phải tuân thủ kiến trúc phân tầng chuẩn:

```text
[ Cell 0: Header & Mục tiêu ] ──► [ Cell 1: Bảng Roadmap (4 cột) + Links Tài liệu liên quan ] ──► [ Các Cells Thực hành Kỹ thuật ]
```

### 5.1. Cell 1: Bảng Lộ trình Thực thi & Liên kết Tài liệu (Roadmap & Documentation References)

- Bắt buộc nằm ở Cell Markdown độc lập ngay sau Cell Tiêu đề (Cell 1).
- Gồm 2 phần chính:
  1. **Bảng Lộ trình Thực thi 4 cột chuẩn bắt buộc**:

| Step | Description           | What it does                                   | Import path                      |
| :--: | :-------------------- | :--------------------------------------------- | :------------------------------- |
| `1`  | `[Tên Bước Kỹ Thuật]` | `[Mô tả chi tiết hành vi, giải thuật, chỉ số]` | `—`                              |
| `2`  | `...`                 | `...`                                          | ``[`src/...py`](../../src/...py)`` |

     - **Yêu cầu đối với 4 cột**:
       1. `Step`: Đánh số thứ tự bước thực thi (1, 2, 3...).
       2. `Description`: Tên ngắn gọn của bước kỹ thuật.
       3. `What it does`: Mô tả chính xác việc mà bước này thực hiện.
       4. `Import path`: **BẮT BUỘC là LINK BẤM ĐƯỢC TRỰC TIẾP** trỏ thẳng tới file mã nguồn phụ trách (`[src/...](../../src/...)`). Nếu là cấu hình môi trường / thư viện gốc, ghi `—`.

2. **Mục Liên kết Nhanh Tài Liệu & Báo Cáo Kỹ Thuật (Related Documentation References)**:
   - Nằm ngay dưới bảng Roadmap sau đường phân cách `---`.
   - Cung cấp các liên kết Markdown trực tiếp bấm mở nhanh các tài liệu liên quan của module / thí nghiệm:
     - 📄 Báo Cáo Kỹ Thuật Chi Tiết: ``[`docs/.../REPORT.md`](../../docs/.../REPORT.md)`` _(chứa toàn bộ Sơ đồ luồng xử lý toàn diện và phân tích học thuật)_.
     - 🌐 Báo Cáo Trực Quan HTML: ``[`docs/.../REPORT.html`](../../docs/.../REPORT.html)`` _(mở trên trình duyệt xem toàn bộ visual gallery & artifacts)_.
     - 📋 Đặc Tả Yêu Cầu Kỹ Thuật: ``[`docs/.../PLANNING.md`](../../docs/.../PLANNING.md)`` _(nội dung bài toán và tiêu chí đánh giá)_.
     - 🏠 Tài Liệu Hướng Dẫn Dự Án: ``[`README.md`](../../README.md)`` _(kiến trúc mã nguồn và hướng dẫn chạy tự động)_.

### 5.2. Sơ đồ Luồng Công việc (Workflow Diagram) - DÀNH RIÊNG CHO BÁO CÁO (REPORTS ONLY)

- **Tuyệt đối KHÔNG đưa sơ đồ quy trình vào Jupyter Notebook**: Giúp notebook luôn tinh gọn, tập trung hoàn toàn vào việc chạy mã nguồn thực thi, tương tác kết quả và tránh lỗi hiển thị mã Mermaid thô.
- **Bắt buộc lưu trữ trong Báo cáo**: Sơ đồ luồng công việc toàn diện (Input Layer, Clean Arch `src/`, Core Processing Stages, Artifacts Output, QA Verification) BẮT BUỘC phải nằm trong:
  1. File Markdown báo cáo (`docs/**/REPORT.md`) (khối mã Mermaid + liên kết ảnh/vector).
  2. File HTML báo cáo (`docs/**/REPORT.html`) (giao diện đồ họa vector SVG / infographic tương tác).

---

## 6. Quy chuẩn Mục Tài Liệu Tham Khảo (References Standard - Reports)

Mọi báo cáo thực nghiệm chuyên sâu (`docs/**/*.md`, `docs/**/*.html`) BẮT BUỘC phải có **Mục Tài Liệu Tham Khảo (References & Technical Documentation)** ở cuối tài liệu với 4 phân nhóm quy chuẩn (trong khi Notebook chỉ chứa các link liên kết nhanh tài liệu nội bộ ở Cell 1):

1. **Thư viện Cốt lõi & Tài liệu API Chính thức (Official Library Documentation)**:
   - Các API cụ thể được sử dụng trong bài toán (ví dụ: OpenCV, Pillow, PyTorch, NumPy, Matplotlib, Albumentations, v.v.).
   - Đường dẫn liên kết trực tiếp tới trang tài liệu chính thức của thư viện.
2. **Tiêu chuẩn Kỹ thuật & Không gian Màu (Technical Standards & Specifications)**:
   - Các tiêu chuẩn quốc tế, bài báo khoa học hoặc giải thuật nền tảng liên quan trực tiếp đến nghiệp vụ của bài toán.
3. **Quy ước Thị giác Máy tính & Định dạng Dữ liệu (CV Conventions & Formats)**:
   - Quy ước thứ tự kênh màu bộ nhớ (Memory Layout: BGR vs RGB), dtypes và tensor shapes.
   - Quy ước định dạng bounding box, mask phân đoạn hoặc keypoints tương ứng.
4. **Mã Nguồn Module Tái Sử Dụng Nội Bộ (Internal Project Architecture)**:
   - Danh sách link bấm được trực tiếp tới các module mã nguồn trong package `src/` phụ trách nghiệp vụ.
