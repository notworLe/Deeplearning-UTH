---
title: Log Directory README
doc_type: readme
project: retail-system
layer: all
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-08
updated: 2026-11-08
last_updated_by: Antigravity
version: 0.1.0
status: draft
reviewers: []
related_docs:
  - ../.agents/rule/LOG.md
  - DECISION_LOG.md
tags: [log, readme, governance]
---

# Log

Thư mục này lưu lại **bộ nhớ dự án** — các quyết định và hành động quan trọng theo từng phiên làm việc.

---

## Nội dung

| File | Mô tả | Ai ghi |
| :--- | :--- | :--- |
| [`AI_THINKING.md`](AI_THINKING.md) | **Mental Model & Handoff**: Luồng suy nghĩ của AI, bẫy ngầm kỹ thuật, bài học và ngữ cảnh bàn giao cho AI Agent kế nhiệm | AI Agent |
| [`DECISION_LOG.md`](DECISION_LOG.md) | **Quyết định kỹ thuật (ADR)**: Kiến trúc, thư viện, quy ước (D-xxx) có giá trị ràng buộc | Human + Agent |
| [`ERROR.md`](ERROR.md) | **Sự cố & Bài học sửa lỗi**: Ghi lại các lỗi build/runtime nghiêm trọng và cách giải quyết triệt để | AI Agent |

---

## Quy tắc ghi log

Xem chi tiết tại [`.agents/rule/LOG.md`](../.agents/rule/LOG.md). Tóm tắt:

**Ghi khi quyết định thỏa ít nhất một điều:**
- Khó đảo ngược (xóa file, đổi schema, đổi cấu trúc thư mục)
- Ràng buộc công việc sau (chọn thư viện, quy ước đặt tên, kiến trúc)
- Đã thử rồi bỏ — ghi lý do để khỏi thử lại
- Người dùng sửa lỗi lặp lại của agent

**Không ghi:** lựa chọn nhỏ, sửa typo, nội dung git commit đã ghi sẵn.

---

## Lịch sử thay đổi

| Version | Ngày | Người sửa | Agent | Mô tả thay đổi |
| :--- | :--- | :--- | :--- | :--- |
| 0.2.0 | 2026-11-10 | notworLe | Antigravity | Bổ sung AI_THINKING.md (luồng tư duy bàn giao) và ERROR.md (nhật ký sự cố) vào bảng danh mục |
| 0.1.0 | 2026-11-08 | notworLe | Antigravity | Khởi tạo README mô tả thư mục log |
