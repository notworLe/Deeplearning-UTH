---
title: "Test Suite — Kiểm Thử Tự Động (Smoke & Normal Tests)"
doc_type: readme
author: notworLe
ai_assisted: true
ai_agent: Antigravity
created: 2026-11-20
updated: 2026-11-20
last_updated_by: Antigravity
version: 0.1.0
status: active
related_docs:
  - ../.agents/rules/TESTING_STRATEGY.md
  - ../pytest.ini
  - ../src/README.md
  - ../prompts/0001_lab1/plans/README.md
tags: [test, pytest, smoke-test, unit-test, lab1]
---

# Thư Mục Kiểm Thử (`test/`)

Chứa bộ unit test chạy bằng `pytest` cho các module cốt lõi trong [`src/`](../src/README.md) và 5 pipeline Lab 1 tại [`src/vibe_scripts/lab1/`](../logs/vibe_scripts/README.md), áp dụng chiến lược kiểm thử 2 chế độ (**Smoke Test** + **Normal Test**) theo [`TESTING_STRATEGY.md`](../.agents/rules/TESTING_STRATEGY.md) và cấu hình tại [`pytest.ini`](../pytest.ini).

---

## 1. Cách Chạy Kiểm Thử

Chạy từ thư mục gốc dự án:

```powershell
# Chế độ 1: Smoke Test nhanh (sanity check, 9 test items)
pytest -m smoke -v

# Chế độ 2: Toàn bộ test suite (55 test items)
pytest test/ -v

# Chạy riêng bộ test Lab 1 (33 test items)
pytest test/test_lab1_*.py -v

# Chạy một file cụ thể
pytest test/test_lab1_models.py -v
```

---

## 2. Danh Mục File Kiểm Thử

### A. Module Core (`src/`) & Config Override

| File | Đối tượng kiểm thử | Items (Smoke) | Phạm vi kiểm tra chính |
| :--- | :--- | :---: | :--- |
| [`test_config_override.py`](./test_config_override.py) | [`config_override.py`](../src/vibe_scripts/config_override.py), [`labs/lab1/vibe/config.py`](../labs/lab1/vibe/config.py) | `7` (`2`) | Kế thừa hằng số từ `src/config.py`, không mutate gốc, fail-fast khi override sai key, tách `extras`, `to_dict` |
| [`test_consoles.py`](./test_consoles.py) | [`src/consoles.py`](../src/consoles.py) | `12` (`0`) | Render bảng `df_to_table`: cấu trúc cột, căn lề số, hiển thị `NaN`, định dạng float, escape markup, cắt gọn đầu/cuối, DataFrame rỗng |
| [`test_utility.py`](./test_utility.py) | [`src/utility.py`](../src/utility.py) | `2` (`0`) | Ghép 1 hoặc nhiều transform qua `add_transform` |
| [`test_class_name.py`](./test_class_name.py) | [`src/class_name.py`](../src/class_name.py) | `1` (`0`) | Xác thực đủ 37 nhãn lớp `OXFORD_III_PET` |
| [`test_dataset.py`](./test_dataset.py) | [`src/datasets.py`](../src/datasets.py) | `0` (`0`) | Placeholder rỗng (chưa triển khai) |
| [`test_object.py`](./test_object.py) | [`src/objects.py`](../src/objects.py) | `0` (`0`) | Placeholder rỗng (chưa triển khai) |

### B. 5 Pipeline Lab 1 (`src/vibe_scripts/lab1/`)

| File | Pipeline phụ trách | Items (Smoke) | Phạm vi kiểm tra chính |
| :--- | :--- | :---: | :--- |
| [`test_lab1_data.py`](./test_lab1_data.py) | Pipe 01 — [`data.py`](../src/vibe_scripts/lab1/data.py) | `7` (`1`) | Shape `(B, 1, 28, 28)`, tách tầng `55k/5k/10k` không rò rỉ, cân bằng lớp, chuẩn hóa Train (`mean≈0, std≈1`), cô lập augmentation trên Train, tái lập `seed=248`, phát hiện hỏng SHA-256 `manifest.json` |
| [`test_lab1_models.py`](./test_lab1_models.py) | Pipe 02 — [`models.py`](../src/vibe_scripts/lab1/models.py) | `13` (`3`) | Smoke forward 3 mô hình, khớp chính xác số tham số (`7.850` / `109.386` / `105.866`), raw logits & gradient hữu hạn, fail-fast khi shape sai, tái lập trọng số & tách rời loss/optimizer |
| [`test_lab1_train.py`](./test_lab1_train.py) | Pipe 03 — [`train.py`](../src/vibe_scripts/lab1/train.py) | `5` (`1`) | Smoke 1 epoch sinh `best.pt`/`last.pt`/`history.json`, sanity check baseline (initial loss $\approx \ln 10$, overfit 1 batch), schema checkpoint không tham chiếu `test_loader`, early stopping, bắt `NaN` loss |
| [`test_lab1_evaluate.py`](./test_lab1_evaluate.py) | Pipe 04 — [`evaluate.py`](../src/vibe_scripts/lab1/evaluate.py) | `4` (`1`) | Smoke đánh giá 1 checkpoint, bất biến Confusion Matrix & metric trong `[0, 1]`, kiểm chứng Save/Load `< 1e-6`, `run_benchmark` sinh đủ JSON/CSV/PNG (gồm `learning_curve_<exp_id>.png`) |
| [`test_lab1_infer.py`](./test_lab1_infer.py) | Pipe 05 — [`infer.py`](../src/vibe_scripts/lab1/infer.py) | `4` (`1`) | Smoke suy luận 1 tensor, tổng xác suất Softmax `= 1.0` & `requires_grad == False`, xử lý ảnh RGB ngoài / suy luận batch, `benchmark_latency` |

**Tổng cộng:** `55` test items (`9` smoke + `46` normal).

---

## 3. Quy Ước Bổ Sung Test Mới

1. Đặt tên file `test_<module>.py`, hàm kiểm thử `test_<hành_vi>()`.
2. Tách rõ **Smoke Test** (gắn `@pytest.mark.smoke`, chạy nhanh kiểm tra import/khởi tạo/dummy forward) và **Normal Test** (logic toán học, tensor shape, bất biến số học, edge cases).
3. Không sửa dữ liệu thật hay ghi đè `experiments/` — luôn dùng fixture `tmp_path` của `pytest` cho artifact tạm.
4. Không xóa test case để ép pass; khi test fail cần truy nguyên nhân gốc theo [`TROUBLESHOOTING.md`](../.agents/rules/TROUBLESHOOTING.md).

---

## Lịch sử thay đổi

| Version | Ngày       | Người sửa | Agent       | Mô tả thay đổi                                                        |
| :------ | :--------- | :-------- | :---------- | :-------------------------------------------------------------------- |
| 0.1.0   | 2026-11-20 | notworLe  | Antigravity | Khởi tạo `test/README.md` mô tả danh mục 11 file kiểm thử & cách chạy |
