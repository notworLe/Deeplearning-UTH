# AI delegation loop

Hệ thống quy trình kỹ thuật chuẩn hóa, vòng đời phát triển và quy chuẩn chất lượng cao nhất cho AI Agent & Kỹ sư trong các dự án Thị giác Máy tính (Computer Vision) & MLOps.

Thư mục này đóng vai trò là **Bộ não Thứ hai (Second Brain)** của Kỹ sư. Mỗi khi bắt đầu một dự án mới, chỉ cần đưa thư mục `.agents/` vào repository và yêu cầu: _"Đọc thư mục `.agents/rules/` và bắt đầu làm việc"_. Agent sẽ lập tức tiếp nhận 100% tư duy thiết kế, ranh giới Clean Architecture, chu trình phát triển 9 giai đoạn và các luật bất biến kỹ thuật để triển khai chính xác tuyệt đối mọi yêu cầu.

---

## Define

Quy trình phối hợp tổng quát giữa Kỹ sư và AI Agent:

```mermaid
flowchart TD
    A["1. Phân vai trò & Bối cảnh dự án"] --> B["2. Agent học quy trình làm việc"]
    B --> C["3. Kỹ sư đưa vấn đề / yêu cầu cho Agent"]
    C --> D["4. Agent lập & giao nộp Plan"]
    D --> E{"Kỹ sư duyệt Plan?"}
    E -- "Chưa đạt" --> B
    E -- "Duyệt" --> F["5. Thực thi Plan (Implement & Audit)"]
    F --> G{"Xong Plan?"}
    G -- "Chưa xong" --> B
    G -- "Xong" --> H["6. TEST (Smoke & Normal)"]
    H --> I{"Pass?"}
    I -- "Fail (RCA)" --> F
    I -- "Pass" --> J["7. LOG các sự kiện"]
```

## 1. Giao thức Tiếp nhận Nhận thức cho AI Agent (Agent Cognitive Protocol)

Khi được yêu cầu đọc thư mục `.agents/`, AI Agent phải tự động thực thi quy trình tiếp nhận 4 bước:

```mermaid
flowchart TD
    S1["Bước 1: Role Alignment — Thiết lập nhân cách"] --> S2["Bước 2: Context Anchoring — Quét cấu trúc repository & ranh giới module"]
    S2 --> S3["Bước 3: Task Routing — Phân luồng Macro Lifecycle hoặc Micro Loop"]
    S3 --> S4["Bước 4: Contract Output — Xuất hợp đồng kỹ thuật trước khi code"]
```

### Bước 1: Thiết lập Tư thế Nhận thức (Role Alignment)

- Đọc vai trò vận hành tại [`AI_ENG_SHORT_RESPONSE.md`](../role_alignment/AI_ENG_SHORT_RESPONSE.md).

### Bước 2: Khóa Ranh giới & Neo Ngữ cảnh (Context Anchoring)

- Quét cấu trúc repository hiện tại: cụ thể [`README.md`](../../README.md).

### Bước 3: Phân luồng Nhiệm vụ (Task Routing)

- Nếu Kỹ sư đưa ra bài toán/đề tài mới: Tham chiếu [`AI_FIRST_LIFECYCLE.md`](AI_FIRST_LIFECYCLE.md) để lập bản đặc tả DEFINE (7 trụ cột) và DESIGN (7 thành phần).
- Nếu Kỹ sư giao một task cụ thể: Tham chiếu [`PLANNING.md`](PLANNING.md) và chạy chu trình vi mô [`AGENT_INTERACTION_LOOP.md`](AGENT_INTERACTION_LOOP.md).

### Bước 4: Định dạng Phản hồi Bắt buộc (Contract-First Response Schema)

Trước khi sinh hoặc sửa bất kỳ dòng mã nguồn nào, Agent **bắt buộc** cấu trúc phản hồi theo đúng 4 phần:

1. **Scope & Tensor Contracts**: File mục tiêu, Tensor shapes `(B, C, H, W)` hoặc `(H, W, C)`, dtypes, device (`to(device)`), thư viện sử dụng.
2. **Detailed Execution Plan & DoD**: Từng bước code cụ thể từ Interface → Validation → Core Math → Corner Cases kèm tiêu chuẩn hoàn thành.
3. **Production Code**: Mã nguồn chuẩn PEP 484 Type Hints, Google-style docstrings, 100% vector hóa, không dùng `print()`, không magic numbers.
4. **Self-Audit & Verification Status**: Báo cáo kiểm toán 3 tầng ([`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md)) và trạng thái kiểm thử Smoke/Normal test ([`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)).

---

## 2. Bản đồ Phản xạ Nhận thức (Cognitive Dispatch Matrix)

Khi tiếp nhận bất kỳ mệnh lệnh nào từ Kỹ sư, Agent tự động tra cứu bảng phản xạ sau để đọc đúng tài liệu và hành động chuẩn mực:

| Ý định của Kỹ sư                        | Tài liệu Quy trình Tham chiếu                                                                      | Hành động Chuẩn mực của Agent                                                                      |
| :-------------------------------------- | :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| Bắt đầu bài lab mới / Lên ý tưởng dự án | [`AI_FIRST_LIFECYCLE.md`](AI_FIRST_LIFECYCLE.md)                                                   | Phân tích bài toán, xuất bản đặc tả DEFINE (7 trụ cột) và DESIGN (7 thành phần).                   |
| Giao task / Lên kế hoạch tính năng      | [`PLANNING.md`](PLANNING.md), [`PROGRESS_TRACKING.md`](PROGRESS_TRACKING.md)                       | Phân rã Plan → Pipeline → Detailed Plan → DoD, cập nhật trạng thái Kanban.                         |
| Yêu cầu sinh mã nguồn (Code Generation) | [`AI_REFERENCE.md`](AI_REFERENCE.md), [`AGENT_INTERACTION_LOOP.md`](AGENT_INTERACTION_LOOP.md)     | Sinh code chuẩn PEP 484, docstring tensor shapes, 100% vector hóa, không `print()`.                |
| Kiểm tra / Review mã nguồn              | [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md)                                                           | Thẩm định 3 tầng (Kiến trúc, Plan, Conventions) + 5 Bất biến CV/DL (Shapes, Color spaces, Memory). |
| Viết kiểm thử / Chạy test               | [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)                                                       | Viết Smoke Test (@pytest.mark.smoke, < 5s) và Normal Test (shapes, numerical invariants).          |
| Gặp lỗi crash / bug runtime / OOM / NaN | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)                                                         | Phân tích nguyên nhân gốc rễ 5 bước (RCA: Reproduce, Trace, Invariants, Memory, Root Fix).         |
| Tối ưu hóa / Refactor mã nguồn          | [`AI_FIRST_LIFECYCLE.md`](AI_FIRST_LIFECYCLE.md) (Pha 7), [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md) | Vector hóa triệt để, tách hằng số vào `configs/`, kiểm tra rò rỉ GPU memory.                       |
| Lưu vết phiên bản / Commit              | [`AI_REFERENCE.md`](AI_REFERENCE.md) (Mục 3)                                                       | Tạo Conventional Commit message chuẩn format `<type>(<scope>): <subject>`.                         |
| Viết báo cáo bài lab / Tài liệu hóa     | [`AI_FIRST_LIFECYCLE.md`](AI_FIRST_LIFECYCLE.md) (Pha 9)                                           | Tổng hợp bảng số liệu định lượng, phân tích định tính, đính kèm seed tái lập.                      |

---

## 3. Mẫu Phản hồi Xác nhận Tiếp nhận (Cold-Start Acknowledgement)

Khi Kỹ sư yêu cầu: _"Đọc thư mục `.agents/rules/` và chuẩn bị làm việc"_, Agent đọc xong và lập tức phản hồi theo mẫu chuẩn sau:

```text
Tôi đã tiếp nhận toàn bộ hệ thống quy trình kỹ thuật từ thư mục .agents/rules/.

Trạng thái sẵn sàng hoạt động (Second Brain Active):
1. Nhân cách vận hành:
2. Neo ngữ cảnh:
3. Cam kết bất biến:
4. Quy trình sẵn sàng:
Xin mời Kỹ sư đưa ra bài toán DEFINE/DESIGN hoặc chỉ định Task cần thực hiện!
```

---

## 4. Danh mục Quy trình Kỹ thuật Hoàn chỉnh (Workflow Directory)

Hệ thống bao gồm 16 tài liệu quy trình chuyên sâu:

| File Quy trình                                           | Trụ cột Nghiệp vụ        | Mục đích & Trọng tâm Kỹ thuật                                                                                                                                 |
| :------------------------------------------------------- | :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`WORKFLOW_OVERVIEW.md`](WORKFLOW_OVERVIEW.md)           | **Tổng quan**            | Toàn cảnh luồng thực thi khép kín từ Requirement → Complete kết nối Macro Loop và Micro Loop.                                                                 |
| [`AI_FIRST_LIFECYCLE.md`](AI_FIRST_LIFECYCLE.md)         | **Vòng đời Vĩ mô**       | 9 giai đoạn phát triển: DEFINE (7 trụ cột), DESIGN (7 thành phần), BREAKDOWN, MVP, IMPLEMENT, TEST, REFACTOR, COMMIT, DOCUMENT.                               |
| [`AGENT_INTERACTION_LOOP.md`](AGENT_INTERACTION_LOOP.md) | **Chu trình Vi mô**      | Chu trình pair-programming 6 bước: Describe → Generate → Review → Run → Test → Refactor.                                                                      |
| [`CONTEXT_MANAGEMENT.md`](CONTEXT_MANAGEMENT.md)         | **Quản trị Ngữ cảnh**    | 4 trụ cột kiểm soát LLM: Context Relevance, Faithfulness, Hallucination Control, Knowledge Store.                                                             |
| [`BEHAVIORAL_LAYER.md`](BEHAVIORAL_LAYER.md)             | **Đặc tả Hành vi**       | Markdown Schema 7 trường chuẩn hóa hành vi và luồng công việc của Agent.                                                                                      |
| [`PLANNING.md`](PLANNING.md)                             | **Đặc tả Tác vụ**        | Quy trình phân cấp: Plan → Pipeline → Detailed Execution Plan → Definition of Done (DoD).                                                                     |
| [`PROGRESS_TRACKING.md`](PROGRESS_TRACKING.md)           | **Quản lý Tiến độ**      | Quản trị trạng thái Kanban (`To Do`, `In Progress`, `Done`, `On Hold`, `Cancelled`) và luật Single-WIP.                                                       |
| [`AI_REFERENCE.md`](AI_REFERENCE.md)                     | **Quy chuẩn & Bất biến** | Naming Conventions, Folder Invariants, Conventional Commits và 5 Bất biến CV/DL (Shapes, Color spaces, Box formats, Seed, GPU Hygiene).                       |
| [`AI_PROBLEMS.md`](AI_PROBLEMS.md)                       | **Quy chuẩn Phản hồi**   | Danh sách lỗi văn phong/trình bày cấm AI lặp lại (không chú thích thừa, trả lời ngắn gọn).                                                                    |
| [`WRITE_README.md`](WRITE_README.md)                     | **Tài liệu Module**      | Quy định khởi tạo `README.md` mô tả tổng quan cho từng module phức tạp.                                                                                       |
| [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md)                 | **Kiểm toán Mã nguồn**   | Kiểm toán 3 tầng (Kiến trúc, Plan, Conventions) + 5 tiêu chuẩn an toàn chuyên sâu CV/DL.                                                                      |
| [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)             | **Chiến lược Kiểm thử**  | Kiểm thử 2 chế độ: Smoke Test (< 5s) vs Normal Test (Đầy đủ) & Pytest patterns.                                                                               |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)               | **Xử lý Sự cố & RCA**    | Quy trình phân tích nguyên nhân gốc rễ 5 bước (RCA) và cẩm nang xử lý 6 sự cố kinh điển (OOM, NaN, Shape, Deadlock, BGR/RGB, Leakage).                        |
| [`logs/LOG_AI_THINKING.md`](logs/LOG_AI_THINKING.md)     | **Nhật ký Tư duy**       | Quy chuẩn ghi lịch sử suy nghĩ, quyết định kỹ thuật và bàn giao ngữ cảnh tại `logs/thinking/`.                                                                |
| [`logs/LOG_ERROR.md`](logs/LOG_ERROR.md)                 | **Nhật ký Sự cố**        | Quy chuẩn tra cứu và ghi nhận lỗi phát sinh tại `logs/errors/` (tránh thử sai lặp lại).                                                                       |
| [`logs/LOG_VIBE_SCRIPTS.md`](logs/LOG_VIBE_SCRIPTS.md)   | **Nhật ký Vibe Scripts** | Quy chuẩn đăng ký các script AI tự viết trong `src/vibe_scripts/` và tính năng cải thiện tại `logs/vibe_scripts/`.                                            |

---

## 5. Sơ đồ Kiến trúc Vận hành Nhận thức

```mermaid
flowchart TD
    subgraph Ingestion ["0. Cognitive Operating Blueprint"]
        RM["README.md<br/>(Agent Ingestion Protocol & Dispatch Matrix)"]
    end

    subgraph Macro ["1. Macro Engineering Lifecycle"]
        OV["WORKFLOW_OVERVIEW.md"]
        LC["AI_FIRST_LIFECYCLE.md<br/>(Define → Design → Breakdown → MVP)"]
        TD["PLANNING.md<br/>(Plan → Pipeline → Detailed Plan → DoD)"]
        PT["PROGRESS_TRACKING.md<br/>(Kanban Statuses)"]
    end

    subgraph Micro ["2. Micro Pair Programming & Quality Loop"]
        LP["AGENT_INTERACTION_LOOP.md<br/>(Describe → Generate → Review → Run → Test → Refactor)"]
        CA["CODEBASE_AUDIT.md<br/>(Architecture, Plan, Convention, CV Invariants)"]
        TS["TESTING_STRATEGY.md<br/>(Smoke Test & Normal Test)"]
        TB["TROUBLESHOOTING.md<br/>(5-Step RCA & Root Fix)"]
    end

    subgraph Standards ["3. Governance, Standards & Logs"]
        CM["CONTEXT_MANAGEMENT.md<br/>(Context Relevance & Faithfulness)"]
        BL["BEHAVIORAL_LAYER.md<br/>(Prompting Schema)"]
        RF["AI_REFERENCE.md & AI_PROBLEMS.md"]
        LG["logs/LOG_AI_THINKING.md & logs/LOG_ERROR.md"]
    end

    Ingestion --> Macro
    Ingestion --> Standards
    Macro --> Micro
    Standards --> Micro
    TS -. Fail .-> TB
    TB -. Fix .-> TS
```

---

## 6. 5 Cổng Kiểm soát Chất lượng Tuyệt đối (The 5 Quality Gates)

Trước khi chấp nhận bất kỳ mã nguồn hoặc tài liệu nào vào dự án:

1. **Gate 1 - Context Gate**: Agent không tự suy diễn, bám sát cấu hình và tài liệu tham chiếu ([`CONTEXT_MANAGEMENT.md`](CONTEXT_MANAGEMENT.md)).
2. **Gate 2 - Design Gate**: Luôn có hợp đồng Input/Output shapes, Plan và DoD trước khi sinh code ([`PLANNING.md`](PLANNING.md)).
3. **Gate 3 - Audit Gate**: Vượt qua thẩm định 3 tầng kiến trúc, quy chuẩn và 5 bất biến CV/DL ([`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md)).
4. **Gate 4 - Test Gate**: Pass 100% Smoke Test (< 5s) và Normal Test với Pytest trước khi xác nhận Done ([`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)). Nếu test fail, kích hoạt ngay quy trình phân tích nguyên nhân gốc rễ tại [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md), tuyệt đối không dùng `try-except pass` che giấu lỗi.
5. **Gate 5 - Document & Delivery Gate**: Báo cáo chuyên sâu (`docs/`) có đủ sơ đồ luồng và References 4 nhóm; Notebook tinh gọn (Cell 0 + Cell 1 Roadmap, không sơ đồ/references học thuật); 100% đường dẫn tương đối và đã ghi log sự kiện ([`logs/LOG_AI_THINKING.md`](logs/LOG_AI_THINKING.md)).
