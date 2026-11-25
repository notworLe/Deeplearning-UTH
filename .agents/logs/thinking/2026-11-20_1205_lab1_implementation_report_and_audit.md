# Lab 1 — Hoàn Thiện 5 Pipeline, Thực Nghiệm 3×2, Báo Cáo Phản Biện & Audit Cuối (T-08, T-09)

- **Date/Time:** 2026-11-20 12:05
- **Agent/Role:** Principal Deep-Learning Engineer & Audit Agent (Antigravity)
- **Status:** Completed
- **Related Files:**
  - `prompts/0001_lab1/plans/README.md`
  - `labs/lab1/vibe/docs/REPORT.md`
  - `labs/lab1/vibe/docs/pipelines/04_EVALUATION_BENCHMARK.md`
  - `labs/lab1/vibe/lab1.ipynb`
  - `src/vibe_scripts/lab1/data.py`
  - `src/vibe_scripts/lab1/models.py`
  - `src/vibe_scripts/lab1/train.py`
  - `src/vibe_scripts/lab1/evaluate.py`
  - `src/vibe_scripts/lab1/infer.py`
  - `test/test_lab1_data.py`
  - `test/test_lab1_models.py`
  - `test/test_lab1_train.py`
  - `test/test_lab1_evaluate.py`
  - `test/test_lab1_infer.py`
  - `logs/vibe_scripts/README.md`
  - `experiments/lab1/`

---

## 1. Bối cảnh & Mục tiêu (Context & Goal)

- **Yêu cầu ban đầu:** Hoàn tất T-08 (thực nghiệm ma trận 3 mô hình × 2 cấu hình, bổ sung phân tích lỗi theo `docs/references/DL_PIPELINE.md`, nhúng đầy đủ biểu đồ EDA/Evaluation vào `REPORT.md`) và T-09 (Codebase Audit, Dual Testing Smoke + Normal, cập nhật registry & Thinking Log theo 5 Quality Gates của `WORKFLOW_OVERVIEW.md`).
- **Vấn đề cốt lõi cần giải quyết:**
  1. Đảm bảo báo cáo kỹ thuật phản ánh trung thực 100% số liệu & hình ảnh thực tế, không suy diễn post-hoc hay ngộ nhận khái niệm (overfit gap âm, đo sai checkpoint, bịa mô tả gallery khi chưa mở ảnh).
  2. Khắc phục lỗi hiển thị biểu đồ rò rỉ trong notebook khi import `evaluate.py`.
  3. Đạt đủ 5 Quality Gates trước khi bàn giao commit.

---

## 2. Luồng suy nghĩ & Phân tích (Chain of Thought & Reasoning)

- **Các phương án đã cân nhắc (Alternatives Considered):**
  - *Báo cáo overfit gap:*
    - Phương án A: Dùng train/val ở epoch 20 (cuối) để so với Test Acc từ `best.pt` → **Sai về điểm so sánh** (so 2 checkpoint khác nhau) và trộn đơn vị loss vs accuracy.
    - Phương án B: Đo toàn bộ train/val/gap **tại `best_epoch`** (cùng checkpoint với Test) và quy ước đồng chiều `gap_loss = val − train`, `gap_acc = train − val` → **Đã chọn** vì đảm bảo tính nhất quán checkpoint.
  - *Cách diễn giải nhánh `aug` có `val_loss < train_loss`:*
    - Phương án A: Coi gap âm là "hết overfit" hoặc "underfit" → **Sai khái niệm** vì train loss đo trên ảnh augmented (khó hơn), val đo trên ảnh sạch.
    - Phương án B: Ghi rõ gap âm là **artifact** của augmentation, đánh giá tác động `aug` bằng Test metrics trên cùng tập sạch và phân tách giả thuyết theo từng kiến trúc (`Logistic aug` đã phẳng từ epoch 2, còn `MLP/CNN aug` vẫn giảm chậm đến epoch 20) → **Đã chọn**.
  - *Xuất hình EDA cho `REPORT.md`:*
    - Phương án A: Chỉ giữ con số văn bản (`mean`, `std`, `cosine > 0.85`) → Thiếu bằng chứng trực quan cho nhận định PCA 2D và class overlap.
    - Phương án B: Xuất 3 ảnh trực tiếp từ `eda_helpers.py` trên 55.000 mẫu Train (tránh leakage) vào `experiments/lab1/eda/`, kiểm tra trực tiếp cả 3 ảnh + ghi rõ cosine đo giữa 10 class prototype (`0.90–0.99`) và explained variance PCA 2D (`47.19%`) → **Đã chọn**.
- **Quyết định cuối cùng (Decision Made):**
  - Phát biểu giả thuyết rõ ràng ở `Define → Hypotheses` (ghi trung thực H1 pre-hoc, tiêu chí định lượng H2/H3 post-hoc), kiểm chứng bằng bảng 4 cột ở `Evaluate` (**H1 Đúng, H2 Bác bỏ, H3 Đúng**).
  - Kiểm tra trực tiếp bằng mắt tất cả 4 ảnh `predictions_gallery_*.png` và 3 ảnh EDA trước khi viết phân tích, sửa dứt điểm lỗi mô tả sai thực tế trước đó.
- **Giả định (Assumptions):**
  - Toàn bộ kết quả dựa trên 1 seed cố định (`seed = 248`) và ngân sách `20 epochs`; chênh lệch nhỏ (`0.39` điểm % giữa `CNN no_aug` và `CNN aug`) chưa đủ kết luận mạnh nếu không chạy đa seed.

---

## 3. Chi tiết Thay đổi Chính (Key Implementation Details)

- **Code & Test (`src/vibe_scripts/lab1/`, `test/`):**
  - Bổ sung `plot_individual_learning_curves()` vào `evaluate.py` sinh 6 biểu đồ `learning_curve_<exp_id>.png` (2 panel Loss/Acc, đường dọc Best Epoch, vùng tô Overfit Gap) và thêm assertion vào `test/test_lab1_evaluate.py`.
  - Bổ sung guard Matplotlib backend trong `evaluate.py` (`if "inline" not in matplotlib.get_backend().lower(): matplotlib.use("Agg")`) để không phá `%matplotlib inline` trong `lab1.ipynb`.
  - Bổ sung docstring cho `main()` ở `train.py`, `evaluate.py`, `infer.py` qua bước Audit.
- **Báo cáo & Artifacts (`labs/lab1/vibe/docs/REPORT.md` v0.13.0, `experiments/lab1/eda/`):**
  - Nhúng 19 ảnh PNG bằng 100% đường dẫn tương đối (3 ảnh EDA + 1 learning curve tổng hợp + 6 learning curve riêng + 1 per-class F1 + 4 confusion matrix + 4 predictions gallery).
  - Bổ sung nhận xét đọc bảng Metric tổng hợp, bảng kiểm chứng giả thuyết H1–H3, quy trình Error Analysis 3 bước chuẩn `DL_PIPELINE.md`, và sửa toàn bộ lỗi kỹ thuật được phản biện qua các bản v0.4.0 → v0.13.0.
- **Đồng bộ Tài liệu & Notebook:**
  - Cập nhật `04_EVALUATION_BENCHMARK.md` (v0.4.0), `logs/vibe_scripts/README.md` (v0.2.0), `prompts/0001_lab1/plans/README.md` (v0.4.0, đánh dấu `T-08`, `T-09` Done và 5/5 DoD).
  - Sửa 2 broken links phát hiện khi audit: `logs/thinking/INDEX.md` (`../.agents/` → `../../.agents/`) và `labs/lab1/vibe/lab1.ipynb` Cell 11 (`docs/reference/` → `docs/references/`).

---

## 4. Khó khăn & Cách khắc phục (Challenges & Solutions)

- **Rò rỉ hình Matplotlib trong Notebook:** `matplotlib.use("Agg")` chạy ở mức module trong `evaluate.py` ghi đè backend inline khi cell sau import, gây lệch hiển thị ở cell EDA → Khắc phục bằng guard kiểm tra `"inline"` trước khi gọi `use("Agg")`.
- **Sai lệch mô tả ảnh Gallery:** Mô tả ban đầu suy đoán "top confident errors chủ yếu là `Shirt → T-shirt`" mà chưa mở ảnh xem → Khi mở trực tiếp 4 ảnh PNG phát hiện 5/5 lỗi tự tin nhất của cả 4 mô hình đều đạt softmax 100% và chủ yếu là ảnh dị biệt (`Shirt → Bag`, `Ankle boot → Sandal`, `Trouser → Dress`, `Dress → Trouser`) → Đã viết lại đúng 100% thực tế ảnh và ghi rõ giới hạn phương pháp.
- **Nhiễu đo Inference Latency:** Cùng kiến trúc nhưng `aug`/`no_aug` lệch 24–46% dù augmentation không ảnh hưởng forward pass → Ghi rõ cột latency chỉ tham khảo tại mục Hạn chế #4 thay vì suy diễn trade-off sai.

---

## 5. Ghi chú bàn giao cho Agent tiếp theo (Handover & Next Steps)

- **Trạng thái hiện tại:**
  - Dual Testing PASS 100%: `pytest -m smoke -v` (9 passed), `pytest test/ -v` (55 passed).
  - Audit 3 tầng & 5 Quality Gates đạt chuẩn (0 broken relative links, 0 absolute paths/username, 0 vi phạm 5 bất biến CV/DL).
  - Tất cả Task T-01 → T-09 trong `prompts/0001_lab1/plans/README.md` đã `Done`.
- **Cảnh báo / Lưu ý (Gotchas/Warnings):**
  - Không sửa `src/config.py` hoặc `labs/lab1/vibe/config.py` ngoài phạm vi.
  - Không tự chạy `labs/lab1/vibe/lab1.ipynb`; kỹ sư tự chạy *Restart Kernel → Run All* để làm sạch output cell trước khi commit nếu cần.
  - Trong `git status` có sẵn staged delete cho `labs/lab1/vibe/docs/pipelines/*.md` — phải chạy `git reset HEAD` trước khi stage lại các atomic commits.
- **Đề xuất bước tiếp theo (Next Actions):**
  - Thực hiện 3–4 atomic commits theo bộ lệnh đã chuẩn bị (bao gồm các file cập nhật mới nhất ở `plans/`, `logs/thinking/`, `logs/vibe_scripts/`, `04_EVALUATION_BENCHMARK.md`, `experiments/lab1/eda/`).
  - Nếu mở rộng Lab 1 (v2): chạy đa seed (≥3 seeds), thử regularization (`Dropout`, `Weight Decay`, `BatchNorm`), tăng epoch cho nhánh `aug`, và lấy mẫu ngẫu nhiên 20–50 lỗi thuộc cặp `Shirt ↔ T-shirt/top` để phân tích sâu hơn.
