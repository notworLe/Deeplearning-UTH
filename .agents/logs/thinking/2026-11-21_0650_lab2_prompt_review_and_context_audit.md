# Lab 2 — Kiểm Tra Logic `prompts/0003_lab2/PROMPT.md` (Bản Cập Nhật) & Audit Hiện Trạng Workspace

- **Date/Time:** 2026-11-21 06:50
- **Agent/Role:** Principal Deep-Learning Engineer (Antigravity)
- **Status:** Partial *(đã hoàn tất nạp context & rà soát logic prompt; đang chờ kỹ sư chốt 4 điểm trước khi lập Plan & code)*
- **Related Files:**
  - `prompts/0003_lab2/PROMPT.md`
  - `labs/lab2/QUESTION.md`
  - `labs/lab2/main/baseline.py`
  - `labs/lab2/main/transfer_learning.py`
  - `src/config.py`
  - `logs/errors/0001.md`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)

- **Yêu cầu ban đầu:** Thực hiện `prompts/0003_lab2/PROMPT.md` bản vừa cập nhật (so sánh 3 run ResNet-18: `resnet18_baseline`, `resnet18_transfer`, `resnet18_finetune` trên **Oxford-IIIT Pets** — 37 giống, trả lời câu hỏi fine-tune có hợp lý không dựa trên số liệu thật; đồng bộ `REPORT.md` ngắn gọn xúc tích với notebook) và ghi log cập nhật.
- **Vấn đề cốt lõi cần giải quyết:** Thực hiện bước Gate 1 & Special Requirement trong `WORKFLOW_OVERVIEW.md` — kiểm tra logic prompt đối chiếu với thực tế workspace trước khi lập Plan và viết code.

---

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)

- **So sánh thay đổi giữa bản `PROMPT.md` cũ và bản mới (2026-11-21):**
  - Chốt cứng 3 run (`baseline` → `transfer` → `finetune`) thay vì để `finetune` tùy chọn.
  - Cô lập chặt biến giữa các cặp so sánh: `baseline → transfer` kiểm tra giá trị pretrained; `transfer → finetune` (unfreeze `layer4`, nạp checkpoint transfer sau khi thay `fc 512→37`) trả lời câu hỏi chính.
  - Đặt tiêu chí định lượng trước khi xem kết quả: nếu chênh lệch Test Accuracy/Macro-F1 `< ~1` điểm % trên 3.669 mẫu test → chạy thêm 2–3 seeds trước khi kết luận; nếu tăng train mà val không tăng → ghi nhận overfit.
  - Bổ sung kiểm chứng giả thuyết lỗi dồn vào giống mèo > giống chó (nhãn loài qua quy ước viết hoa/thường tên file) và yêu cầu `REPORT.md` ngắn gọn, ưu tiên bảng/số liệu.
- **Các điểm bất nhất / chặn kỹ thuật phát hiện trên đĩa:**
  1. **Lỗi logic tách Val (dòng 16 `PROMPT.md`):** ghi *"tách 10% từ test (khoảng 368 ảnh)"* — mâu thuẫn trực tiếp với nguyên tắc *"test chỉ chạm một lần ở cuối"* (dòng 58) và con số *"Test 3.669 ảnh"* (dòng 63). `10% × 3.680 (trainval) = 368` → khả năng cao gõ nhầm `trainval` thành `test`.
  2. **Thiếu dữ liệu & checkpoint trên đĩa:** `data/` chỉ có `FashionMNIST` (chưa có `oxford-iiit-pet`), `weights/` và `~/.cache/torch/hub/checkpoints/` trống → cần tải dữ liệu + weights ngoài sandbox và train sạch cả 3 run.
  3. **Rủi ro `LR = 1e-5` & bug của `labs/lab2/main/baseline.py`:** `src/config.py` đặt `LR = 1e-5` (quá nhỏ cho Adam 30 epoch với `transfer` linear head và `finetune` differential LR `1e-6`), đồng thời `baseline.py` truyền nhầm `v_train_transform` vào `v_test`, làm rò rỉ augmentation sang `v_val` và không stratified; môi trường cũng thiếu `tensorboard` (`logs/errors/0001.md`).
  4. **File CIFAR-10 cũ trong `labs/lab2/vibe/`:** đang chứa 3 notebook + 2 script cũ về CIFAR-10, cần xác nhận thay thế theo cấu trúc chuẩn Lab 1 (`config.py`, `lab2.ipynb`, `docs/REPORT.md`, `docs/pipelines/`).

---

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)

- Chưa chỉnh sửa mã nguồn hay tạo Plan; hoàn tất kiểm toán hiện trạng môi trường (GPU `RTX 4060 Laptop` khả dụng, thiếu `tensorboard`, chưa có dataset `OxfordIIITPet` và pretrained weights trong cache).
- Ghi nhận log cập nhật phiên review prompt này vào `logs/thinking/` và `logs/thinking/INDEX.md`.

---

## 4. Khó khăn & Cách khắc phục (Challenges & Solutions)

- Mâu thuẫn dòng 16 `PROMPT.md` (`Val tách 10% từ test` vs `Test 3.669 ảnh chỉ chạm 1 lần`) có thể gây data leakage nghiêm trọng nếu làm máy móc → Dừng trước khi lập Plan để hỏi xác nhận kỹ sư theo đúng quy trình `WORKFLOW_OVERVIEW.md`.

---

## 5. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)

- **Trạng thái hiện tại:** Đang chờ kỹ sư chốt 4 điểm: (1) Val tách 10% từ `trainval` (không phải `test`), (2) cho phép tải `OxfordIIITPet` + `IMAGENET1K_V1` và train sạch cả 3 run, (3) chốt `LR` head / backbone (`1e-3` & `1e-4` hay giữ `1e-5`) + augmentation train, (4) thay thế file CIFAR-10 cũ trong `labs/lab2/vibe/`.
- **Cảnh báo / Lưu ý (Gotchas/Warnings):** Không sửa các file thủ công trong `src/`; mọi pipeline Lab 2 đặt tại `src/vibe_scripts/lab2/` và cấu hình override qua `VibeConfig` tại `labs/lab2/vibe/config.py`; luôn dùng `history.json` thay vì `tensorboard`.
- **Đề xuất bước tiếp theo (Next Actions):** Ngay khi kỹ sư chốt 4 điểm trên → lập `prompts/0003_lab2/plans/README.md` (T-01 → T-09) → tải dữ liệu + sinh `manifest.json` → code 5 pipeline + Dual Testing → chạy thực nghiệm 3 run (và đa seed nếu chênh `< 1` điểm %) → viết `REPORT.md` & `lab2.ipynb`.
