# `labs/lab3_v2/` — Get Started with Hugging Face (`Sentiment Pipeline` & `DistilBERT IMDB Fine-Tuning`)

## Purpose

Điểm vào thực thi (cấu hình, thin CLI scripts, 2 executable notebooks và bộ tài liệu SRP) của mini-project **`lab3_v2`** (`0007_lab3_v2`) — làm chủ luồng **text → tokenizer → pretrained model → Trainer → đánh giá Val/Test & Error Analysis** theo [`MODEL_PIPELINE.md`](../../.agents/rules/dl-rule/MODEL_PIPELINE.md) (`v1.4.0`), cô lập hoàn toàn khỏi `labs/lab3/` cũ.

## Public API & Entrypoints

| File | Vai trò |
| :--- | :--- |
| `config.py` | Cấu hình kế thừa `VibeConfig`: IMDB subset `4,000 / 1,000 / 1,000` (`seed=248`), `max_length=256`, `3 ep`, `lr=2e-5`, `BS=16`, `fp16` |
| `run_ex1.py` | CLI chạy **Exercise 1**: giải phẫu WordPiece tokenizer + sentiment analysis bằng HF `pipeline` |
| `train_ex2.py` | CLI chạy **Exercise 2**: IMDB Data Audit/EDA → Baseline Val trước fine-tune → `Trainer.train()` → Val sau fine-tune |
| `eval_ex2.py` | CLI chấm **Held-Out Test (`1,000` mẫu) đúng 1 lần cuối**, xuất Confusion Matrix + error examples & chạy live inference (`--predict`) |
| `ex1_sentiment.ipynb` | Executable notebook trình bày trực quan **Ex1** (`tokens`, `input_ids`, `attention_mask`, `decode`, sentiment predictions) |
| `ex2_finetune.ipynb` | Executable notebook trình bày trực quan **Ex2** (EDA, Before vs After Val, loss curve, Final Test, error analysis & live inference) |
| `REPORT.md` | Báo cáo tóm tắt điều hướng nhanh dẫn trực tiếp về 3 báo cáo SRP trong [`docs/README.md`](docs/README.md) |

## Usage

```powershell
# 1. Exercise 1 — Giải phẫu Tokenizer & Sentiment Pipeline (SST-2)
python labs/lab3_v2/run_ex1.py

# 2. Exercise 2 — IMDB Audit/EDA, Baseline Val & Fine-tune DistilBERT (4,000 train / 1,000 val, 3 epochs, fp16)
python labs/lab3_v2/train_ex2.py

# (Tùy chọn) Xem TensorBoard trực tiếp:
tensorboard --logdir experiments/lab3_v2/runs/distilbert_imdb/monitor/tb_logs

# 3. Đánh giá Held-Out Test (1,000 mẫu — chỉ chạy 1 lần cuối sau khi Val đạt)
python labs/lab3_v2/eval_ex2.py

# 4. Suy luận thử trên câu review tự viết
python labs/lab3_v2/eval_ex2.py --predict "Brilliant direction and deeply moving performances!" "Boring, predictable, and painfully slow."

# 5. Chạy bộ kiểm thử offline (Smoke -> Normal)
pytest -m smoke test/test_lab3_v2_ex1.py test/test_lab3_v2_ex2.py -v
pytest test/test_lab3_v2_ex1.py test/test_lab3_v2_ex2.py -v
```
