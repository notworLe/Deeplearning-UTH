# Error Knowledge Base — Bảng Tra Cứu Lỗi Nhanh

Nơi agent ghi các lỗi dính phải trong lúc làm việc hay suy nghĩ của người dùng cũng như của agent theo quy định tại [`.agents/rules/logs/LOG_ERROR.md`](../../.agents/rules/logs/LOG_ERROR.md).

---

| Error ID | Nguồn phát hiện | Thông báo lỗi chính (Error Signature) | Ngữ cảnh / Thư viện | Trạng thái (`Status`) | Giải pháp cực ngắn (TL;DR Fix) | File Chi tiết |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ERR-001` | **Agent** | `ModuleNotFoundError: No module named 'tensorboard'` & lệch import `src.dataset` | `src/visualize.py`, `src/datasets.py` | `INVESTIGATING` | Giữ nguyên `src/`, tách helper độc lập vào `src/vibe_scripts/` không phụ thuộc `tensorboard` | [`0001.md`](0001.md) |
| `ERR-002` | **User / Agent** | `ModuleNotFoundError: No module named 'datasets'` & `ImportError: requires accelerate>=1.1.0` | Hugging Face / Lab 3 Notebooks (`anything` venv) | `RESOLVED` | Cài `datasets`, `evaluate`, `accelerate>=1.1.0` vào venv `anything` và đặt `dataloader_num_workers=0` trên Windows | [`ERR-002_lab3_missing_hf_dependencies.md`](ERR-002_lab3_missing_hf_dependencies.md) |
