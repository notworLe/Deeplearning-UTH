---
trigger: always_on
---

# AI Agent Rule: Vibe Scripts & Feature Enhancement Logging

## 1. Mục đích (Objective)

Quy tắc này bắt buộc AI Agent phải ghi nhận đầy đủ mọi **script tự tạo trong `ROOT/src/vibe_scripts/`** và mọi **chức năng cải thiện theo yêu cầu của Kỹ sư**, nhằm:

- **Minh bạch nguồn gốc mã nguồn**: Phân biệt rạch ròi mã nguồn thủ công trong `src/` với các module/script do AI mở rộng trong `src/vibe_scripts/`.
- **Tra cứu tức thì**: Giúp Kỹ sư và Agent kế nhiệm biết ngay chức năng nào đã có sẵn để tái sử dụng, tránh viết trùng lặp.
- **Truy vết yêu cầu cải thiện**: Liên kết thẳng mỗi cải tiến/script về đúng Prompt nguồn (`ROOT/prompts/000x_.../`) và file test kiểm chứng.

---

## 2. Quy trình Bắt buộc (Mandatory Workflow)

1. **Trước khi tạo script mới trong `src/vibe_scripts/`**: Đọc [`ROOT/logs/vibe_scripts/README.md`](../../../logs/vibe_scripts/README.md) để kiểm tra xem chức năng tương tự đã tồn tại chưa.
2. **Ngay sau khi tạo/cập nhật script hoặc thêm tính năng cải thiện**: Agent **BẮT BUỘC** cập nhật bảng đăng ký tại `ROOT/logs/vibe_scripts/README.md` trước khi báo cáo hoàn thành hoặc commit.

---

## 3. Cấu trúc Lưu trữ & Bảng Chuẩn

```text
root/
├── .agents/rules/logs/
│   └── LOG_VIBE_SCRIPTS.md      <-- File quy tắc này
└── logs/vibe_scripts/
    └── README.md                <-- Bảng tra cứu tất cả script AI & cải tiến
```

### Quy chuẩn bảng `ROOT/logs/vibe_scripts/README.md`:

| Script / Module | Prompt nguồn | Chức năng & Cải thiện chính | Phụ thuộc `src/` | Test kiểm chứng | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `src/vibe_scripts/<file>.py` | `prompts/000x_.../prompt.md` | Mô tả ngắn gọn công dụng, lý do tạo | `src/config.py` | `test/test_<...>.py` | `Active \| Deprecated` |

---

## 4. Nguyên tắc Viết Log

1. **Ngắn gọn, không sao chép code**: Chỉ ghi mục đích, hàm/class chính, file phụ thuộc và file test.
2. **Không xóa lịch sử**: Nếu script bị thay thế, đổi trạng thái sang `Deprecated` và ghi chú module thay thế.
3. **100% đường dẫn tương đối**: Mọi liên kết trong bảng phải bấm mở trực tiếp được (`Ctrl + Click`).
