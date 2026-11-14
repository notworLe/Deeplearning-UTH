# AI Agent Rule: Error Tracking & Bug Knowledge Base Logging

## 1. Mục đích (Objective)

Quy tắc này là **kim chỉ nam bắt buộc** cho mọi AI Agent khi gặp lỗi trong quá trình thực thi (Runtime Error, Build/Compile Error, Logic Bug, Dependency Conflict, v.v.) — bất kể lỗi đó do **Người dùng (User) gửi lên** hay do **AI Agent tự gặp phải** khi chạy lệnh/viết code.

Mục tiêu cốt lõi:
* **Xây dựng Kho tàng Sửa lỗi (Error Knowledge Base):** Tạo nguồn tài liệu tra cứu lỗi khổng lồ, thực tế và sát với dự án nhất dành cho User.
* **Tiết kiệm Thời gian & Token cho AI Agent:** Giúp các Agent về sau khi gặp lại mã lỗi tương tự có thể tra cứu và áp dụng ngay giải pháp trong vòng vài giây mà không cần suy luận hay thử sai (trial-and-error) lại từ đầu.
* **Không bỏ sót bất kỳ lỗi nào:** Ghi nhận mọi lỗi ngay khi phát sinh (kể cả khi chưa giải quyết được) kèm trạng thái (`Status`) rõ ràng để theo dõi.

---

## 2. Quy trình Bắt buộc Khi Gặp Lỗi (Mandatory Error Workflow)

### Bước 1: Tra cứu trước khi tự suy nghĩ (Search Before Thinking)
Ngay khi User báo lỗi hoặc Agent gặp lỗi từ Terminal/IDE:
1. **Đọc ngay file `/logs/ERROR_INDEX.md`** (hoặc tìm kiếm từ khóa mã lỗi trong thư mục `/logs/errors/`).
2. **Nếu lỗi đã từng xuất hiện và có trạng thái `RESOLVED`:** Áp dụng ngay giải pháp đã ghi nhận, không tự ý thử các cách vòng vo khác.
3. **Nếu lỗi đang ở trạng thái `UNRESOLVED` hoặc `INVESTIGATING`:** Đọc phần *"Các cách đã thử nhưng thất bại"* trong file log của lỗi đó để **KHÔNG lặp lại các cách sửa vô ích**.

### Bước 2: Ghi nhận lỗi ngay lập tức (Immediate Error Logging)
* Nếu là lỗi mới hoàn toàn, Agent **BẮT BUỘC** phải tạo ngay mục ghi nhận lỗi (dù lúc đó đã sửa được hay chưa).
* Nếu chưa sửa được ngay, để trạng thái là `UNRESOLVED` hoặc `WORKAROUND` (tạm thời lách lỗi) và ghi lại những gì đã thử.
* Ngay khi sửa thành công, cập nhật trạng thái thành `RESOLVED` và điền **Giải pháp ngắn gọn (Quick Fix / Root Cause)**.

---

## 3. Cấu trúc Thư mục Lưu trữ Lỗi (Directory Structure)

```
root/
├── rules/
│   └── agent_error_logging_rules.md       <-- File kim chỉ nam này
└── logs/
    ├── ERROR_INDEX.md                     <-- Bảng tra cứu lỗi nhanh (Bắt buộc cập nhật)
    └── errors/
        └── ERR-XXX_[short-error-slug].md  <-- File chi tiết từng lỗi (hoặc nhóm lỗi)
```

### Quy chuẩn bảng `/logs/ERROR_INDEX.md`:

Đây là trang mục lục giúp bạn và AI Agent quét mắt tìm lỗi chỉ trong 3 giây. Mỗi lỗi là 1 dòng:

| Error ID | Nguồn phát hiện | Thông báo lỗi chính (Error Signature) | Ngữ cảnh / Thư viện | Trạng thái (`Status`) | Giải pháp cực ngắn (TL;DR Fix) | File Chi tiết |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ERR-001` | **Agent** | `RuntimeError: CUDA out of memory` | PyTorch / Training | `RESOLVED` | Giảm `batch_size` từ 64 xuống 16 và thêm `torch.cuda.empty_cache()` | `errors/ERR-001_cuda-oom.md` |
| `ERR-002` | **User** | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Model Head / Linear | `RESOLVED` | Đổi `in_features` của tầng `fc` thành `2048` khớp với ResNet50 | `errors/ERR-002_shape-mismatch.md` |
| `ERR-003` | **Agent** | `ConnectionResetError: [Errno 104]` khi tải dataset | Torchvision Dataset | `WORKAROUND` | Link gốc lỗi, tạm tải file `.tar` thủ công bỏ vào `/data` | `errors/ERR-003_dataset-download.md` |
| `ERR-004` | **User** | Loss trả về `NaN` sau epoch thứ 3 | Fine-tuning | `UNRESOLVED` | Đang nghi ngờ do Learning Rate quá lớn, chờ kiểm tra dữ liệu | `errors/ERR-004_nan-loss.md` |

---

## 4. Định nghĩa các Trạng thái Lỗi (Error Status Definitions)

Agent chỉ được phép sử dụng 1 trong 4 trạng thái chuẩn sau:
* `RESOLVED` (Đã giải quyết triệt để): Đã tìm ra nguyên nhân gốc rễ và có cách sửa dứt điểm, chạy lại thành công.
* `WORKAROUND` (Giải pháp tạm thời): Chưa sửa được tận gốc nhưng đã có cách lách qua để hệ thống chạy tiếp được.
* `INVESTIGATING` (Đang điều tra): Lỗi đang trong quá trình debug, đã loại trừ được một số nguyên nhân.
* `UNRESOLVED` (Chưa giải quyết được): Bế tắc hoặc tạm gác lại để làm việc khác, cần User hoặc Agent khác hỗ trợ sau.

---

## 5. Khuôn mẫu File Chi tiết Lỗi (Error Log Entry Template)

Mỗi khi tạo file trong `root/logs/errors/ERR-XXX_[slug].md`, Agent **BẮT BUỘC** viết ngắn gọn, súc tích theo đúng khuôn mẫu sau:

```markdown
# [ERR-XXX] Tên tóm tắt của lỗi

- **Status:** [RESOLVED | WORKAROUND | INVESTIGATING | UNRESOLVED]
- **First Seen:** YYYY-MM-DD HH:MM
- **Reported By:** [User gửi lỗi | Agent tự gặp khi chạy lệnh]
- **Affected Files / Modules:** `path/to/file.py` (dòng X)
- **Environment / Stack:** [VD: Python 3.10, PyTorch 2.1, Windows/Linux]

---

## 1. Dấu hiệu & Thông báo Lỗi (Error Traceback / Symptom)
> *Dán chính xác dòng báo lỗi quan trọng nhất (không dán log rác quá dài) để dễ tìm kiếm bằng Ctrl+F.*

```text
[Dán đoạn Traceback hoặc câu thông báo lỗi cốt lõi vào đây]
```

**Bối cảnh xảy ra lỗi:**
- Lỗi xuất hiện khi thực hiện hành động gì? (VD: Khi truyền batch ảnh qua tầng `model(inputs)` ở bước validation).

---

## 2. Nguyên nhân Gốc rễ (Root Cause)
- **Tại sao bị lỗi?** [Giải thích ngắn gọn 1-2 câu về bản chất kỹ thuật. Nếu chưa rõ thì ghi: *"Chưa xác định rõ, nghi ngờ do..."*].

---

## 3. Giải pháp Ngắn gọn (Quick Fix / Solution)
> *Nếu `Status` là `RESOLVED` hoặc `WORKAROUND`, bắt buộc ghi rõ cách sửa ngắn gọn nhất tại đây.*

- **Cách sửa tóm tắt:** [1-2 câu chỉ rõ cần sửa dòng nào, đổi tham số gì, hoặc chạy lệnh gì].
- **Code / Lệnh khắc phục:**
  ```python
  # Trước khi sửa (Code lỗi):
  ...
  # Sau khi sửa (Code chạy đúng):
  ...
  ```

---

## 4. Các cách đã thử nhưng THẤT BẠI (Failed Attempts - Rất quan trọng)
> *Bắt buộc ghi lại nếu quá trình sửa lỗi phải thử nhiều cách, hoặc khi lỗi đang `UNRESOLVED`. Giúp Agent sau không đạp lại vết xe đổ.*

1. **Thử nghiệm 1:** [Đã thử làm X] $\rightarrow$ **Kết quả:** Vẫn lỗi hoặc sinh ra lỗi Y.
2. **Thử nghiệm 2:** [Đã thử làm Z] $\rightarrow$ **Kết quả:** Không hiệu quả vì...
```

---

## 6. Nguyên tắc Vàng cho AI Agent (Core Error Logging Principles)

1. **Ghi ngay cả khi chưa sửa xong:** Nghiêm cấm thói quen "đợi sửa xong mới ghi". Nếu thử 2 lần chưa sửa được lỗi, phải lập tức ghi nhận vào `/logs/errors/` với trạng thái `INVESTIGATING` hoặc `UNRESOLVED` để lưu lại các cách đã thử thất bại.
2. **Giải pháp phải cực kỳ ngắn gọn và đi thẳng vào vấn đề:** Không viết lý thuyết dài dòng. Chỉ rõ: *Lỗi gì $\rightarrow$ Do đâu $\rightarrow$ Sửa dòng code/lệnh nào*.
3. **Không trùng lặp mã lỗi:** Nếu gặp lại một lỗi đã có trong `ERROR_INDEX.md` nhưng ở một file khác hoặc có biến thể mới, hãy **cập nhật bổ sung** vào file `ERR-XXX` sẵn có thay vì tạo file mới gây loãng tài liệu.