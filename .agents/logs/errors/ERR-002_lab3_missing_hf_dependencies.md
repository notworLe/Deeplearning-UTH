# [ERR-002] Thiếu Thư Viện Hugging Face (`datasets`, `evaluate`, `accelerate>=1.1.0`) Khi Chạy Notebooks Lab 3

- **Status:** RESOLVED
- **First Seen:** 2026-11-21 07:45
- **Reported By:** User yêu cầu sửa Notebooks Lab 3 & Agent phát hiện khi kiểm tra venv `anything`
- **Affected Files / Modules:** `labs/lab3/main/02_ex2_finetune_sentiment.ipynb`, `labs/lab3/main/fine-tune.ipynb`, `labs/lab3/main/fine-tune.py`
- **Environment / Stack:** Python 3.13.14, PyTorch 2.15.0+cu132, Transformers 5.23.0, Windows

---

## 1. Dấu hiệu & Thông báo Lỗi (Error Traceback / Symptom)

Khi chạy Cell 1 của `02_ex2_finetune_sentiment.ipynb`:
```text
ModuleNotFoundError: No module named 'datasets'
```
Và khi khởi tạo `TrainingArguments` trong `transformers 5.23.0`:
```text
ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=1.1.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=1.1.0'`
```

**Bối cảnh xảy ra lỗi:**
- Xảy ra khi import thư viện `datasets`, `evaluate` và khởi tạo `TrainingArguments` / `Trainer` trong môi trường ảo `anything` cho Exercise 2 của Lab 3.
- Ngoài ra, cấu hình `dataloader_num_workers=2` (hoặc `8` trong `fine-tune.py`) trên Windows trong Jupyter Notebook dễ gây treo worker (`spawn` multiprocessing) và việc hardcode `checkpoint-2105` khi thư mục `sst2-distilbert/` chưa tồn tại cần được xử lý linh hoạt.

---

## 2. Nguyên nhân Gốc rễ (Root Cause)

- Môi trường ảo `anything` đã cài `transformers 5.23.0` và `torch 2.15.0+cu132` nhưng chưa cài 3 gói phụ thuộc bắt buộc cho pipeline Fine-tuning của Hugging Face: `datasets`, `evaluate`, và `accelerate>=1.1.0`.

---

## 3. Giải pháp Ngắn gọn (Quick Fix / Solution)

- **Cách sửa tóm tắt:**
  1. Cài đặt bổ sung `datasets`, `evaluate`, `accelerate>=1.1.0` vào venv `anything`.
  2. Đặt `dataloader_num_workers=0` trên Windows trong notebook để tránh lỗi multiprocessing spawn và tự động dò tìm/lưu checkpoint tối ưu trong `sst2-distilbert/`.
- **Lệnh khắc phục:**
  ```powershell
  & "d:\notworLe\Desktop\IT\Programing Language\Python\virtual_environment\anything\Scripts\python.exe" -m pip install datasets evaluate "accelerate>=1.1.0"
  ```

---

## 4. Các cách đã thử nhưng THẤT BẠI (Failed Attempts)

- Không có (xác định trực tiếp nguyên nhân qua kiểm tra `pip list` và test import trong venv).
