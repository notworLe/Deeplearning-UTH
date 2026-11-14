# Soạn Thảo Kịch Bản Thuyết Trình Lab 3 & Bản Đồ Phân Bổ File

- **Date/Time:** 2026-11-14 08:45
- **Agent/Role:** Presentation-Coach-Agent
- **Status:** Completed
- **Related Files:** 
  - `docs/lab3/PRESENTATION.md`
  - `docs/lab3/README.md`
  - `docs/lab3/PIPELINE_EXPLANATION.md`
  - `labs/lab3/main/01_ex1_sentiment_analysis.ipynb`
  - `labs/lab3/main/02_ex2_finetune_sentiment.ipynb`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)
- **Yêu cầu ban đầu:** Người dùng yêu cầu tạo tài liệu thuyết trình cho các module tại `docs/lab3/PRESENTATION.md`, chỉ rõ vị trí file nào để ở đâu và viết sẵn lời thuyết trình chi tiết.
- **Vấn đề cốt lõi cần giải quyết:** Giúp người dùng tự tin báo cáo đồ án trước Giảng viên và Hội đồng bằng cách cung cấp một tài liệu "all-in-one" tích hợp: bản đồ file, bảng hướng dẫn mở file khi nói, kịch bản thuyết trình từng chữ cho 10 module, kịch bản live demo và bộ câu hỏi phản biện kèm câu trả lời đắt giá.

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)
- **Bản đồ phân bổ file (File Map):** Cần phân định rõ ràng giữa tầng tài liệu (`docs/lab3/` - nơi trình chiếu slide/báo cáo), tầng mã nguồn (`labs/lab3/main/` - nơi mở code và live demo), và tầng trọng số (`sst2-distilbert/` - bằng chứng thực nghiệm).
- **Kịch bản thuyết trình (Script):** Thiết kế theo luồng tự nhiên (10-15 phút), chia thành 8 phần từ Mở đầu, Kiến trúc, Khảo sát dữ liệu, Tokenization, Huấn luyện, Nghiệm thu 5W1H, Phân tích lỗi đến Kết luận. Mỗi phần đều kèm hướng dẫn hành động cụ thể (mở file nào, chỉ vào cell nào).
- **Bộ câu hỏi phản biện (Q&A Defense):** Tổng hợp 8 câu hỏi phổ biến và hóc búa nhất mà Giảng viên thường hỏi (tại sao chọn DistilBERT, dynamic padding khác static padding thế nào, tại sao `max_length=64`, tại sao F1 đạt đỉnh ở epoch 1, rò rỉ dữ liệu, token `[CLS]`).

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)
- Tạo mới `docs/lab3/PRESENTATION.md` với đầy đủ 7 trường header theo `MD_CONVENTION.md` và 5 phần nội dung toàn diện.
- Cập nhật mục lục và liên kết trong `docs/lab3/README.md`.
- Cập nhật `logs/INDEX.md`.

## 4. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)
- Tài liệu thuyết trình đã sẵn sàng. Người dùng có thể in ra hoặc mở song song trên màn hình phụ để thuyết trình và trả lời chất vấn của Giảng viên.

