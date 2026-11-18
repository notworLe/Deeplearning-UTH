---
trigger: always_on
---

# WORKFLOW OVERVIEW - TỔNG QUAN TOÀN BỘ QUY TRÌNH HỆ THỐNG

Toàn cảnh luồng thực thi khép kín từ Requirement đến Complete.
Khung làm việc với AI Agent

---

## End-to-End Execution Flowchart

```text
Requirement (Đề bài Lab / Yêu cầu Kỹ thuật)
        │
        ▼
LLM Context Management (CONTEXT_MANAGEMENT.md)                                 |
[Context Relevance │ Faithfulness │ Hallucination Control │ Knowledge Store]   |
        │                                                                      |
        ▼                                                                      |
Behavior Layer - Prompting (BEHAVIORAL_LAYER.md)                               | Các file khởi tạo vai
[Markdown Schema: Name → Description → Purpose → Input → Output → Workflow]    | trò, hành vi, cung cấp
        │                                                                      | ngữ cảnh .
        ▼                                                                      |
AI Reference Standards (AI_REFERENCE.md)                                       |
[Naming Conventions │ Folder Structure Hierarchy]                              |
        │                                                                      |
        ▼                                                                      |
Progress Tracking (PROGRESS_TRACKING.md)                                       |
[Kanban Statuses: To Do → In Progress → Done → On Hold → Cancelled]            |
        │
        ▼
Planning (PLANNING.md)
[Plan → Pipeline → Detailed Execution Plan]
        │
        ▼
AI Generate Code (AGENT_INTERACTION_LOOP.md)
[Code Generation strictly within scoped contract]
        │
        ▼
Codebase Audit (CODEBASE_AUDIT.md)
[Audit Architecture → Audit Detailed Plan → Audit Conventions]
        │
        ▼
Unit Test Authoring (TESTING_STRATEGY.md)
[Viết test cases đối chiếu với Detail Plan & Invariants]
        │
        ▼
Smoke Test (TESTING_STRATEGY.md)
[Kiểm tra nhanh: cú pháp, import, instantiate, zero-crash]
        │
        ▼
Normal Test (TESTING_STRATEGY.md)
[Kiểm tra đầy đủ: logic, shapes, numerical invariants, performance]
        │
        ▼
Complete & Commit (AI_FIRST_LIFECYCLE.md & AI_REFERENCE.md)
[Conventional Commits & Technical Documentation]
```

---

## Tồn tại 2 vòng lặp làm việc chính

### 1. Macro Loop (Vòng lặp Kỹ thuật Cấp cao)

Quy trình phát triển toàn diện của kỹ sư:
`Define → Design → Breakdown → Build MVP → Implement → Test → Refactor → Commit → Document`

### 2. Micro Loop (Chu trình Tương tác Cấp thấp với Agent)

Áp dụng cho từng đầu việc nhỏ (Task) trong giai đoạn `Implement`:
`Describe → Generate → Review → Run → Test → Refactor`

---

## Cổng Kiểm soát Chất lượng (Quality Gates)

Trước khi chuyển từ giai đoạn này sang giai đoạn tiếp theo, hệ thống bắt buộc phải vượt qua các cổng kiểm soát:

1. **Gate 1 (Context Gate)**: AI đã nạp đủ tài liệu tham chiếu từ [`AI_REFERENCE.md`](AI_REFERENCE.md) chưa?
2. **Gate 2 (Design Gate)**: Đã có Detailed Plan và Definition of Done chưa?
3. **Gate 3 (Audit Gate)**: Code sinh ra có bám sát kiến trúc và không có code thừa/tự ý bịa đặt không?
4. **Gate 4 (Test Gate)**: Đã pass cả Smoke Test và Normal Test chưa?
5. **Gate 5 (Document & Delivery Gate)**:
   - **Báo cáo chuyên sâu (`docs/`)**: Đã có đủ cặp tệp Markdown báo cáo và HTML trực quan tương tác, tích hợp đầy đủ Sơ đồ Luồng toàn diện và Mục Tài liệu Tham khảo (References) 4 nhóm chuẩn chưa?
   - **Jupyter Notebook (`notebooks/**/*.ipynb`)**: Đã tinh gọn theo chuẩn (Cell 0: Mục tiêu, Cell 1: Bảng Roadmap 4 cột kèm Quick Links tới các tài liệu liên quan), tuyệt đối KHÔNG chứa sơ đồ workflow diagram hay references học thuật chưa?
   - **Bảo mật & Điều hướng (`Ctrl + Click`)**: 100% liên kết nội bộ là đường dẫn tương đối (`../../src/...`), tuyệt đối KHÔNG chứa đường dẫn tuyệt đối chứa thông tin người dùng (`file:///home/...`) chưa?

## Index

| Reference                                                | Tiêu đề                   | Ý nghĩa                                                |
| :------------------------------------------------------- | :------------------------ | :----------------------------------------------------- |
| [`WORKFLOW_OVERVIEW.md`](WORKFLOW_OVERVIEW.md)           | Tổng quan luồng làm việc  | Toàn cảnh luồng thực thi khép kín và 5 Quality Gates   |
| [`CONTEXT_MANAGEMENT.md`](CONTEXT_MANAGEMENT.md)         | Quản lý ngữ cảnh          | Cung cấp đúng ngữ cảnh, chống ảo giác cho Agent        |
| [`BEHAVIORAL_LAYER.md`](BEHAVIORAL_LAYER.md)             | Đặc tả hành vi            | Prompting schema 7 trường điều khiển hành vi Agent     |
| [`AI_REFERENCE.md`](AI_REFERENCE.md)                     | Quy chuẩn tham chiếu      | Naming, Folder Invariants, Commits và 5 bất biến CV/DL |
| [`AI_PROBLEMS.md`](AI_PROBLEMS.md)                       | Lỗi văn phong cần tránh   | Ràng buộc ngắn gọn, tránh chú thích thừa               |
| [`WRITE_README.md`](WRITE_README.md)                     | Quy tắc viết README       | Khởi tạo `README.md` mô tả cho từng module phức tạp    |
| [`PROGRESS_TRACKING.md`](PROGRESS_TRACKING.md)           | Theo dõi tiến độ          | Bộ chuẩn trạng thái Kanban và luật Single-WIP          |
| [`PLANNING.md`](PLANNING.md)                             | Lên kế hoạch              | Phân rã Plan → Pipeline → Detailed Execution Plan      |
| [`AI_FIRST_LIFECYCLE.md`](AI_FIRST_LIFECYCLE.md)         | Vòng đời AI-First         | 9 giai đoạn phát triển từ Define đến Document          |
| [`AGENT_INTERACTION_LOOP.md`](AGENT_INTERACTION_LOOP.md) | Chu trình tương tác vi mô | Nhịp làm việc 6 bước giữa Kỹ sư và Agent               |
| [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md)                 | Kiểm toán mã nguồn        | Rà soát 3 tầng kiến trúc, plan và quy ước CV/DL        |
| [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)             | Chiến lược kiểm thử       | Kiểm thử 2 chế độ Smoke Test và Normal Test            |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)               | Xử lý sự cố (RCA)         | Phân tích nguyên nhân gốc rễ 5 bước khi gặp lỗi        |
| [`logs/LOG_AI_THINKING.md`](logs/LOG_AI_THINKING.md)     | Nhật ký tư duy            | Lưu vết suy nghĩ & bàn giao tại `logs/thinking/`       |
| [`logs/LOG_ERROR.md`](logs/LOG_ERROR.md)                 | Nhật ký lỗi               | Tra cứu và ghi nhận sự cố tại `logs/errors/`           |
