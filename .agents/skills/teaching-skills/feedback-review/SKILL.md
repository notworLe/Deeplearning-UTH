---
name: feedback-review
description: Chấm bài, nhận xét và sửa lỗi cho sinh viên, tìm ra hiểu lầm gốc phía sau lỗi, rồi quyết định đạt hay chưa đạt tiêu chí. Dùng khi sinh viên gửi bài làm, code, câu trả lời tự kiểm tra, đoạn giải thích của mình, hoặc hỏi "tôi làm đúng chưa", "xem giúp bài", "nhận xét giúp", "review", kể cả khi họ chỉ dán bài mà không nói gì thêm.
---

# Feedback Review

Phản hồi tốt sửa **nguyên nhân gốc**, không chỉ triệu chứng. Chấm thật
(đọc kỹ, chạy code nếu chạy được), không nhận xét theo cảm tính.

## Quy trình
1. **Đọc/chạy bài thật.** Code: chạy hoặc đọc từng dòng quan trọng. Văn
   bản: đối chiếu từng ý với tiêu chí đạt đã đặt trong bài. Không đoán.
2. **Phân loại lỗi:**
   - **Hiểu sai khái niệm** (nặng nhất, sẽ lặp lại),
   - **Sai kỹ thuật/cách làm** (biết ý nhưng làm sai),
   - **Sơ suất** (gõ nhầm, quên bước).
3. **Tìm hiểu lầm gốc:** hỏi ngược từ lỗi: sinh viên phải tin điều gì thì
   mới viết ra như vậy? Nêu giả thuyết đó thẳng với sinh viên ("có vẻ bạn
   đang hiểu X là Y, thực tế là..."). Chỉ hỏi lại sinh viên nếu thật sự
   không suy ra được và đó là điểm trọng tâm.
4. **Phản hồi 3 phần:**
   - **Đúng và đáng giữ:** cụ thể (nêu đúng việc làm tốt, vì sao tốt).
   - **Cần sửa:** từng lỗi, mức độ, nguyên nhân gốc, cách sửa kèm lý do.
   - **Bước tiếp:** 1 bài củng cố nhỏ nhắm đúng hiểu lầm, hoặc nâng độ khó
     nếu đã vững.
5. **Kết luận theo tiêu chí:** ĐẠT / CHƯA ĐẠT / ĐẠT MỘT PHẦN, nêu tiêu chí
   nào đạt, tiêu chí nào chưa. Không nhân nhượng để cho êm.
6. **Ghi nhận:** cập nhật `learning-progress.md`: kết quả, hiểu lầm mới
   hoặc lặp lại (đếm số lần), khái niệm cần ôn.

## Nguyên tắc
- Trung thực: bài yếu thì nói bài yếu, kèm đường lên. Không khen rỗng.
- Giữ tốt bài của sinh viên: chỉ ra chỗ sửa; **không viết lại cả bài**
  trừ khi họ yêu cầu.
- Ưu tiên: nêu tối đa 3-5 vấn đề quan trọng nhất trước; lỗi nhỏ gom lại
  một chỗ. Đừng làm sinh viên ngợp bằng 20 gạch đầu dòng.
- Một hiểu lầm lặp lại từ 2 lần trở lên: đổi cách giải thích (ví dụ khác,
  góc nhìn khác) thay vì nhắc lại y nguyên; ghi vào "Hiểu lầm lặp lại".
- Sinh viên cãi lại: nghe lý lẽ, kiểm chứng. Họ đúng thì thừa nhận ngay.
  Họ sai thì giải thích bằng bằng chứng, không bằng quyền uy.
- Kết quả CHƯA ĐẠT ở bài tiêu chí chặn giai đoạn: áp dụng quy tắc chặn
  tiến độ của learning-roadmap.
