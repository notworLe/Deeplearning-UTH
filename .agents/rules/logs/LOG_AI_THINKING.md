# AI Agent Rule: Thinking & Context History Logging

## 1. Mục đích (Objective)
Quy tắc này là **kim chỉ nam bắt buộc** đối với mọi AI Agent tham gia vào dự án. Việc ghi log lại quá trình suy nghĩ (Thinking Process) và quyết định kỹ thuật (Decision Making) sau khi hoàn thành một task hoặc chức năng nhằm:
- **Duy trì Context History:** Giúp chính Agent hoặc phiên làm việc (session) sau nắm bắt ngay lập tức bối cảnh mà không cần đọc lại toàn bộ mã nguồn.
- **Đồng bộ hóa Đa Agent (Multi-Agent Alignment):** Giúp các Agent khác (Coding Agent, Review Agent, Testing Agent, Docs Agent) hiểu rõ lý do đằng sau mỗi thay đổi, tại sao chọn giải pháp A thay vì B.
- **Truy vết & Tham khảo (Traceability):** Làm tài liệu lịch sử để con người (Developer/Mentor) hoặc Agent kiểm tra lại luồng tư duy khi cần debug hoặc mở rộng hệ thống.

---

## 2. Quy trình Bắt buộc (Mandatory Workflow)

### Bước 1: Trước khi bắt đầu Task (Pre-Task Context Loading)
1. Kiểm tra thư mục `/logs/` ở root dự án.
2. Đọc file `/logs/INDEX.md` (nếu có) hoặc **2-3 file log gần nhất** liên quan đến module đang làm việc để hiểu bối cảnh, kiến trúc hiện tại và các ràng buộc trước đó.

### Bước 2: Sau khi hoàn thành Task / Chức năng (Post-Task Logging)
Ngay sau khi hoàn thành code, sửa lỗi hoặc thiết kế xong một chức năng, Agent **BẮT BUỘC** phải tạo hoặc cập nhật file log trước khi báo cáo kết quả cuối cùng cho người dùng.

---

## 3. Cấu trúc Thư mục & Quy tắc Đặt tên (Directory & Naming Convention)

Toàn bộ log được lưu tại thư mục gốc của dự án theo cấu trúc:

```text
root/
├── rules/logs
│   └── LOG_AI_THINKING.md   <-- File kim chỉ nam này
└── logs/
    ├── INDEX.md                 <-- Danh mục tóm tắt các log (Agent phải cập nhật dòng mới vào đây)
    └── YYYY-MM-DD_HHMM_[task-slug].md
```

- **Đường dẫn lưu file:** `logs/YYYY-MM-DD_HHMM_[ten-task-ngan-gon].md`
  - *Ví dụ:* `logs/2026-11-14_0115_setup-fashion-mnist-dataloader.md`
- **File `logs/INDEX.md`:** Chứa bảng danh sách các file log theo thứ tự thời gian (mới nhất ở dưới hoặc trên cùng) kèm 1 câu tóm tắt để các Agent sau tra cứu nhanh.

---

## 4. Khuôn mẫu Nội dung File Log (Log Template)

Mỗi file log trong `logs/` **phải** tuân thủ cấu trúc Markdown dưới đây:

```markdown
# [Tên Task / Chức năng]

- **Date/Time:** YYYY-MM-DD HH:MM
- **Agent/Role:** [Tên hoặc vai trò của Agent, VD: DL-Engineer-Agent / Refactor-Agent]
- **Status:** [Completed | Partial | Blocked]
- **Related Files:** 
  - `path/to/file1.py`
  - `path/to/file2.md`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)
- **Yêu cầu ban đầu:** Tóm tắt ngắn gọn yêu cầu của User hoặc hệ thống.
- **Vấn đề cốt lõi cần giải quyết:** Tại sao phải thực hiện task này?

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)
- **Các phương án đã cân nhắc (Alternatives Considered):**
  - *Phương án A:* [Mô tả] -> Ưu/Nhược điểm.
  - *Phương án B:* [Mô tả] -> Ưu/Nhược điểm.
- **Quyết định cuối cùng (Decision Made):** Chọn phương án nào? **Tại sao?** (Giải thích rõ bản chất kỹ thuật/toán học/kiến trúc).
- **Giả định (Assumptions):** Những giả định nào đã được đặt ra trong quá trình làm?

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)
- Tóm tắt những thay đổi quan trọng về mặt logic, cấu trúc dữ liệu, luồng dữ liệu (VD: tensor shape biến đổi ra sao, kiến trúc thay đổi tầng nào, API nào được thêm).
- *Lưu ý: Không copy-paste toàn bộ code vào đây, chỉ mô tả bản chất và logic.*

## 4. Khó khăn & Cách khắc phục (Challenges & Solutions)
- Các lỗi (bugs), giới hạn bộ nhớ, xung đột thư viện hoặc edge-cases gặp phải trong lúc làm và cách Agent đã xử lý.

## 5. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)
- **Trạng thái hiện tại:** Hệ thống đang ở trạng thái nào?
- **Cảnh báo / Lưu ý (Gotchas/Warnings):** Các Agent sau KHÔNG NÊN chạm vào phần nào hoặc cần chú ý điều gì?
- **Đề xuất bước tiếp theo (Next Actions):** Gợi ý các việc cần làm tiếp theo.
```

---

## 5. Nguyên tắc Viết Log cho AI Agent (Best Practices)
1. **Giải thích "Tại sao" (Why) thay vì chỉ "Làm gì" (What):** Code đã thể hiện "Làm gì", file log phải thể hiện được **tư duy đằng sau code đó**.
2. **Ngắn gọn nhưng Đầy đủ ngữ cảnh (High Signal-to-Noise):** Viết súc tích bằng bullet points, dễ đọc cho cả LLM (tối ưu token) và con người.
3. **Trung thực về giới hạn (Honesty on Limitations):** Nếu giải pháp hiện tại chỉ là tạm thời (workaround) hoặc còn nợ kỹ thuật (technical debt), phải ghi rõ ở mục 5 để Agent sau biết và xử lý.
4. **Không ghi đè lịch sử:** Mỗi chức năng/task lớn là một file riêng biệt hoặc một mục nối tiếp rõ ràng, tuyệt đối không xóa suy nghĩ cũ của Agent trước trừ khi được yêu cầu dọn dẹp.