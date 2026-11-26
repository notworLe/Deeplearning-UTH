---
title: Task Definition
doc_type: Template
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-23
updated: 2026-11-26
version: 0.1.0
status: approved
---

# Task

ENGINEER and AGENT write, track at `.agents/tasks/<task_name>/task.md`.

## Metadata

- **Name:**
- **Version:**
- **Priority:**

---

## Objective

Mục tiêu của task.

---

## Background

Bối cảnh của vấn đề.

Tại sao phải làm?

---

## Inputs

AI được cung cấp:

- source code
- dataset
- document
- model

---

## Requirements

Những việc AI phải thực hiện (có thể là checklist).

---

## Constraints

Không được làm gì.

Ví dụ:

- Không đổi API
- Không thêm package
- Không sửa module khác

---

## Deliverables

AI phải giao nộp:

- code
- report
- benchmark
- plan
- document

---

## Acceptance Criteria

Điều kiện hoàn thành.

Ví dụ:

- Test pass
- Accuracy > 95%
- Không warning

---

## References

AI phải đọc:

- `PROJECT_RULE.md`
- `MODEL_PIPELINE.md` (hoặc `DEPLOY_MODEL.md` nếu là task deploy)
- `README.md`

---

## Assumptions

Nếu thiếu thông tin, AI phải liệt kê rõ:

- assumption 1
- assumption 2

và yêu cầu kỹ sư xác nhận.

---

## Validation Checklist

- [ ] Đã đọc Project Rules (`PROJECT_RULE.md`)
- [ ] Đã hiểu Objective
- [ ] Input đầy đủ
- [ ] Acceptance Criteria rõ
- [ ] Không có ambiguity

Nếu còn mục chưa check được:

```text
=> STOP
=> hỏi kỹ sư xác nhận trước khi sang bước Planning.
```
