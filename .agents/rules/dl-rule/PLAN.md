---
title: Planning
doc_type: Template
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-23
updated: 2026-11-23
version: 0.1.0
status: approved
---

# Plan

AGENT writes and submits plans at `.agents/plans/`.

## Objective

Task cần giải quyết (tham chiếu tới task tương ứng trong `.agents/tasks/`).

---

## Understanding

Agent hiểu task như thế nào.

Nếu có assumption phải ghi rõ để Kỹ sư xác nhận.

---

## Scope

- **In Scope:**
- **Out of Scope:**

---

## Work Breakdown

Chia nhỏ thành từng bước độc lập, có thể tự kiểm toán (Self Audit) ở bước [`EXECUTION.md`](EXECUTION.md):

1. Step 1
2. Step 2
3. Step 3

---

## Risks

Có thể ảnh hưởng gì tới các module hoặc pipeline khác.

---

## Validation

Làm sao biết từng bước hoàn thành (Smoke Test, Normal Test, Acceptance Criteria).

---

## Deliverables

- Code
- Document
- Benchmark

---

## Estimated Impact

- Module / file sẽ bị sửa:
- Module / file không đụng tới:

---

## Approval Gate (Bắt buộc)

Sau khi giao nộp Plan:

- **STOP** — không tự ý viết code hoặc chỉnh sửa file trước khi Kỹ sư duyệt Plan.
- Nếu Kỹ sư yêu cầu chỉnh sửa → cập nhật Plan và giao nộp lại.
- Chỉ chuyển sang [`EXECUTION.md`](EXECUTION.md) sau khi Kỹ sư xác nhận duyệt.
