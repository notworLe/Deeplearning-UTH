"""
Script nối tiếp phần Processing -> Model -> Train -> Evaluate -> Infer vào `labs/lab1/vibe/lab1.ipynb`,
thực thi toàn bộ notebook và nhúng đầy đủ output Rich + biểu đồ PNG.
"""

import base64
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console


def _new_md_cell(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def _new_code_cell(lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def append_pipeline_cells(nb: dict) -> dict:
    """Cập nhật Cell 0/1 và giữ nguyên Cell 2..21 (Define -> EDA), nối tiếp Section 8..12."""
    # Cập nhật Cell 0 (phạm vi toàn luồng)
    nb["cells"][0]["source"] = [
        "# Thực Hành 1: Phân Loại Thời Trang Fashion-MNIST với PyTorch\n",
        "## Quy Trình Toàn Diện: Define → EDA → Data Processing → Model → Training → Evaluation → Inference\n",
        "\n",
        "Notebook này triển khai khép kín bài thực hành so sánh **Logistic Regression (7.850)**, **MLP (109.386)** và **CNN (105.866)** trên **Fashion-MNIST** theo kim chỉ nam [`docs/REPORT.md`](docs/REPORT.md), sử dụng cấu hình override tại [`config.py`](config.py) và 5 pipeline độc lập trong `src/vibe_scripts/lab1/`."
    ]

    # Cập nhật Cell 1 (Bảng Roadmap 4 cột chuẩn, thêm Step 10..14)
    nb["cells"][1]["source"] = [
        "## Bảng Lộ Trình Thực Thi & Liên Kết Tài Liệu Kỹ Thuật\n",
        "\n",
        "| Step | Description | What it does | Import path |\n",
        "| :---: | :--- | :--- | :---: |\n",
        "| `1` | Environment & Seed Setup | Thiết lập cấu hình thiết bị động, seed ngẫu nhiên 248 và console giám sát | [`src/config.py`](../../../src/config.py) |\n",
        "| `2` | Define Context & Problem | Bản đặc tả mục tiêu, ràng buộc công bằng thí nghiệm, baselines và metrics | — |\n",
        "| `3` | Immutable Raw Ingestion & Manifest | Tải Fashion-MNIST, kiểm toán `manifest.json` và xác thực mã băm SHA-256 (§3.2) | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `4` | 6-Dimension Quality Audit | Kiểm toán 6 chiều chất lượng dữ liệu (Completeness, Accuracy, Consistency, Timeliness, Uniqueness, Validity) | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `5` | Dataset Splitting | Phân chia Train (55.000), Validation (5.000), Test (10.000) bảo toàn phân tầng và seed 248 | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `6` | EDA Lens 1: Global Topology | Khảo sát cấu trúc hình học tổng thể, tensor shapes, kiểu dữ liệu và dung lượng bộ nhớ | — |\n",
        "| `7` | EDA Lens 2: Univariate Distribution | Đánh giá cân bằng lớp trên 3 tập và thống kê cường độ pixel | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `8` | EDA Lens 3: Visual & Separability | Thư viện ảnh mẫu 10 lớp, Class Prototypes, tương quan và PCA 2D | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `9` | Technical Decisions (EDA) | Đúc kết phát hiện hình học làm căn cứ cho Data Processing & Modeling | — |\n",
        "| `10` | Pipeline 01: Data Processing | Chuẩn hóa Z-Score (fit trên Train), tách 2 nhánh `no_aug` / `aug` chống leakage, dựng DataLoader | [`src/vibe_scripts/lab1/data.py`](../../../src/vibe_scripts/lab1/data.py) |\n",
        "| `11` | Pipeline 02: Model Architecture | Khởi tạo 3 kiến trúc `Logistic` (7.850), `MLP` (109.386), `CNN` (105.866) | [`src/vibe_scripts/lab1/models.py`](../../../src/vibe_scripts/lab1/models.py) |\n",
        "| `12` | Pipeline 03: Baseline & Training | Sanity check Baseline và huấn luyện công bằng 6 cấu hình (`LR=1e-3`, `20 epochs`, `seed=248`) | [`src/vibe_scripts/lab1/train.py`](../../../src/vibe_scripts/lab1/train.py) |\n",
        "| `13` | Pipeline 04: Evaluation & Error Analysis | Chạy Test 1 lần, lập bảng 3×2, kiểm chứng Save/Load, Learning Curves, Confusion Matrix & Error Gallery | [`src/vibe_scripts/lab1/evaluate.py`](../../../src/vibe_scripts/lab1/evaluate.py) |\n",
        "| `14` | Pipeline 05: Inference & Latency | Suy luận mẫu Top-3 Softmax và benchmark tốc độ (`batch_size=1, 32`) | [`src/vibe_scripts/lab1/infer.py`](../../../src/vibe_scripts/lab1/infer.py) |\n",
        "\n",
        "---\n",
        "### Liên Kết Nhanh Tài Liệu & Báo Cáo Kỹ Thuật\n",
        "- 📄 Báo Cáo Chính (Kim Chỉ Nam): [`docs/REPORT.md`](docs/REPORT.md)\n",
        "- 📋 Đặc Tả Yêu Cầu Kỹ Thuật: [`docs/QUESTION.md`](docs/QUESTION.md)\n",
        "- 🧩 Chỉ Mục 5 Pipeline Độc Lập: [`docs/pipelines/README.md`](docs/pipelines/README.md)\n",
        "- ⚙️ Cấu Hình Override Lab 1: [`config.py`](config.py)\n",
        "- 📖 Tài Liệu Hướng Dẫn Dự Án: [`README.md`](../../../README.md)\n"
    ]

    # Giữ nguyên 22 cells đầu (Cell 0..21), sửa 4 dấu cách thừa ở dòng 1 Cell 3 nếu có
    base_cells = nb["cells"][:22]
    if base_cells[3]["source"] and base_cells[3]["source"][0].startswith("    import os"):
        base_cells[3]["source"][0] = "import os\n"

    added_cells = [
        _new_md_cell([
            "---\n",
            "## 8. Pipeline 01: Data Processing & DataLoader\n",
            "\n",
            "Sử dụng cấu hình override từ `labs/lab1/vibe/config.py` (`VibeConfig` kế thừa `src/config.py`): `BATCH_SIZE = 32`, `LR = 1e-3`, `EPOCH = 20`, `SEED = 248`.\n",
            "- Tính `mean`, `std` **duy nhất trên 55.000 mẫu Train** để loại bỏ rò rỉ dữ liệu.\n",
            "- Tách 2 đối tượng `FashionMNIST(train=True)` độc lập cho Train (`no_aug` / `aug`) và Validation (`eval_transform`), giữ đúng thứ tự `Augmentation → ToTensor → Normalize`."
        ]),
        _new_code_cell([
            "import sys\n",
            "if str(config.ROOT_PATH) not in sys.path:\n",
            "    sys.path.insert(0, str(config.ROOT_PATH))\n",
            "from PIL import Image\n",
            "from labs.lab1.vibe import config as lab1_cfg\n",
            "from src.vibe_scripts.lab1 import data as lab1_data\n",
            "from src.vibe_scripts.lab1 import models as lab1_models\n",
            "from src.vibe_scripts.lab1 import train as lab1_train\n",
            "from src.vibe_scripts.lab1 import evaluate as lab1_eval\n",
            "from src.vibe_scripts.lab1 import infer as lab1_infer\n",
            "\n",
            "# Dùng num_workers=0 trong notebook để tránh overhead spawn multiprocessing trên Windows\n",
            "train_loader_no_aug, val_loader, test_loader, norm_stats = lab1_data.build_dataloaders(\n",
            "    aug_enabled=False, num_workers=0\n",
            ")\n",
            "train_loader_aug, _, _, _ = lab1_data.build_dataloaders(\n",
            "    aug_enabled=True, num_workers=0, verify_manifest=False\n",
            ")\n",
            "\n",
            "df_proc = pd.DataFrame(\n",
            "    [\n",
            "        (\"Mean chuẩn hóa (fit trên Train 55k)\", f\"{norm_stats['mean']:.6f}\"),\n",
            "        (\"Std chuẩn hóa (fit trên Train 55k)\",  f\"{norm_stats['std']:.6f}\"),\n",
            "        (\"Batch Size\", str(lab1_cfg.BATCH_SIZE)),\n",
            "        (\"Learning Rate (Adam override)\", str(lab1_cfg.LR)),\n",
            "        (\"Epochs\", str(lab1_cfg.EPOCH)),\n",
            "        (\"Best Checkpoint Metric\", str(lab1_cfg.BEST_METRIC)),\n",
            "        (\"Augmentation (chỉ Train aug)\", f\"HFlip(p={lab1_cfg.AUG_CONFIG['horizontal_flip_p']}), Affine(deg={lab1_cfg.AUG_CONFIG['rotation_degrees']}, trans={lab1_cfg.AUG_CONFIG['translate']})\"),\n",
            "    ],\n",
            "    columns=[\"Cấu hình\", \"Giá trị\"],\n",
            ")\n",
            "v_objConsole.print(consoles.df_to_table(df_proc, title=\"PIPELINE 01: DATA PROCESSING & CONFIG OVERRIDE\"))"
        ]),
        _new_md_cell([
            "## 9. Pipeline 02: Kiến Trúc 3 Mô Hình Đối Sánh\n",
            "\n",
            "Khởi tạo độc lập 3 mô hình khớp chính xác đặc tả tại `docs/REPORT.md`, nhận chung tensor `[B, 1, 28, 28]` và trả về raw logits `[B, 10]`."
        ]),
        _new_code_cell([
            "dummy_x = torch.randn(4, 1, 28, 28)\n",
            "model_rows = []\n",
            "for m_name in lab1_cfg.MODEL_NAMES:\n",
            "    m = lab1_models.build_model(m_name)\n",
            "    out = m(dummy_x)\n",
            "    model_rows.append((m_name.upper(), f\"{lab1_models.count_parameters(m):,}\", str(tuple(out.shape))))\n",
            "\n",
            "df_models = pd.DataFrame(model_rows, columns=[\"Mô hình\", \"Số tham số\", \"Output Shape (Dummy B=4)\"])\n",
            "v_objConsole.print(consoles.df_to_table(df_models, title=\"PIPELINE 02: MODEL ARCHITECTURES SUMMARY\"))"
        ]),
        _new_md_cell([
            "## 10. Pipeline 03: Baseline Sanity Check & Huấn Luyện Ma Trận 3 × 2\n",
            "\n",
            "1. Kiểm tra nhanh Baseline `Logistic` (initial loss $\\approx \\ln(10) \\approx 2.3026$ và khả năng overfit 1 batch)."
        ]),
        _new_code_cell([
            "sanity = lab1_train.run_sanity_check(\"logistic\", train_loader_no_aug, device=lab1_cfg.DEVICE, lr=0.05, steps=80)\n",
            "df_sanity = pd.DataFrame(\n",
            "    [\n",
            "        (\"Initial Loss (kỳ vọng ~2.3026)\", f\"{sanity['initial_loss']:.4f}\"),\n",
            "        (\"Overfit 1 Batch Final Loss\", f\"{sanity['overfit_final_loss']:.4f}\"),\n",
            "        (\"Overfit 1 Batch Final Accuracy\", f\"{sanity['overfit_final_acc']*100:.2f}%\"),\n",
            "    ],\n",
            "    columns=[\"Kiểm tra Sanity (Logistic)\", \"Kết quả\"],\n",
            ")\n",
            "v_objConsole.print(consoles.df_to_table(df_sanity, title=\"BASELINE SANITY CHECK\"))"
        ]),
        _new_md_cell([
            "2. Huấn luyện công bằng toàn bộ 6 cấu hình (`{logistic, mlp, cnn} × {no_aug, aug}`, `20 epochs`, `Adam lr=1e-3`, `seed=248`), lưu `best.pt`, `last.pt` và `history.json` vào `experiments/lab1/`."
        ]),
        _new_code_cell([
            "from tqdm.auto import tqdm\n",
            "train_summaries = []\n",
            "runs = [(m, a) for m in lab1_cfg.MODEL_NAMES for a in lab1_cfg.AUG_MODES]\n",
            "for m_name, aug_flag in tqdm(runs, desc=\"All Experiments (3x2)\"):\n",
            "    tr_loader = train_loader_aug if aug_flag else train_loader_no_aug\n",
            "    res = lab1_train.train_experiment(\n",
            "        model_name=m_name,\n",
            "        aug_enabled=aug_flag,\n",
            "        epochs=lab1_cfg.EPOCH,\n",
            "        lr=lab1_cfg.LR,\n",
            "        seed=lab1_cfg.SEED,\n",
            "        best_metric=lab1_cfg.BEST_METRIC,\n",
            "        train_loader=tr_loader,\n",
            "        val_loader=val_loader,\n",
            "        norm_stats=norm_stats,\n",
            "        show_progress=True,\n",
            "    )\n",
            "    last_ep = res[\"history\"][-1]\n",
            "    train_summaries.append((\n",
            "        res[\"exp_id\"],\n",
            "        res[\"best_epoch\"],\n",
            "        f\"{res['best_metric_value']:.4f}\",\n",
            "        f\"{last_ep['train_acc']*100:.2f}%\",\n",
            "        f\"{last_ep['val_acc']*100:.2f}%\",\n",
            "        f\"{res['total_train_time_sec']:.1f}s\",\n",
            "    ))\n",
            "\n",
            "df_train = pd.DataFrame(\n",
            "    train_summaries,\n",
            "    columns=[\"Exp ID\", \"Best Epoch\", \"Best Val Loss\", \"Last Train Acc\", \"Last Val Acc\", \"Total Time\"],\n",
            ")\n",
            "v_objConsole.print(consoles.df_to_table(df_train, title=\"PIPELINE 03: TRAINING SUMMARY (6 CONFIGURATIONS)\"))"
        ]),
        _new_md_cell([
            "## 11. Pipeline 04: Đánh Giá Trên Tập Test (10.000 Mẫu) & Phân Tích Lỗi\n",
            "\n",
            "Đánh giá **một lần duy nhất** trên tập Test với checkpoint `best.pt` của cả 6 cấu hình, kiểm chứng Save/Load (`max_diff < 1e-6`), trực quan hóa đường cong học, F1 theo lớp, Confusion Matrix và lưới ảnh dự đoán."
        ]),
        _new_code_cell([
            "benchmark_out = lab1_eval.run_benchmark(\n",
            "    experiments_dir=lab1_cfg.EXPERIMENT_DIR,\n",
            "    test_loader=test_loader,\n",
            "    device=lab1_cfg.DEVICE,\n",
            ")\n",
            "\n",
            "df_bench = pd.DataFrame(benchmark_out[\"benchmark_rows\"])\n",
            "v_objConsole.print(consoles.df_to_table(df_bench, title=\"PIPELINE 04: TEST BENCHMARK (3 MODELS x 2 CONFIGS)\"))\n",
            "\n",
            "# Hiển thị Top-5 cặp nhầm lẫn của mô hình CNN (aug & no_aug)\n",
            "for eid in [\"cnn_no_aug\", \"cnn_aug\"]:\n",
            "    confs = benchmark_out[\"experiments\"][eid][\"top_confusions\"]\n",
            "    df_conf = pd.DataFrame(confs)\n",
            "    v_objConsole.print(consoles.df_to_table(df_conf, title=f\"TOP CONFUSIONS — {eid.upper()}\"))\n",
            "\n",
            "# Hiển thị các biểu đồ đã xuất trong experiments/lab1/predictions/\n",
            "for img_name in [\n",
            "    \"learning_curves_comparison.png\",\n",
            "    \"per_class_f1_comparison.png\",\n",
            "    \"confusion_matrix_cnn_aug.png\",\n",
            "    \"predictions_gallery_cnn_aug.png\",\n",
            "]:\n",
            "    p = lab1_cfg.PREDICTIONS_DIR / img_name\n",
            "    if p.exists():\n",
            "        fig, ax = plt.subplots(figsize=(12, 6))\n",
            "        ax.imshow(Image.open(p))\n",
            "        ax.axis(\"off\")\n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "\n",
            "# Hiển thị learning curve riêng từng thí nghiệm\n",
            "print(\"\\n--- Learning Curves Riêng Từng Thí Nghiệm ---\")\n",
            "for exp_sub in sorted(lab1_cfg.EXPERIMENT_DIR.iterdir()):\n",
            "    lc_path = lab1_cfg.PREDICTIONS_DIR / f\"learning_curve_{exp_sub.name}.png\"\n",
            "    if lc_path.exists():\n",
            "        fig, ax = plt.subplots(figsize=(12, 4.5))\n",
            "        ax.imshow(Image.open(lc_path))\n",
            "        ax.set_title(f\"Learning Curve — {exp_sub.name}\", fontsize=12, fontweight=\"bold\")\n",
            "        ax.axis(\"off\")\n",
            "        plt.tight_layout()\n",
            "        plt.show()"
        ]),
        _new_md_cell([
            "## 12. Pipeline 05: Suy Luận Mẫu & Benchmark Độ Trễ\n",
            "\n",
            "Nạp độc lập checkpoint `best.pt` dưới `@torch.inference_mode()`, xuất phân phối xác suất Top-3 Softmax và đo tốc độ suy luận chuẩn (`batch_size = 1` & `32`)."
        ]),
        _new_code_cell([
            "bench_rows = []\n",
            "for m_name in lab1_cfg.MODEL_NAMES:\n",
            "    ckpt_p = lab1_cfg.EXPERIMENT_DIR / f\"{m_name}_no_aug\" / \"best.pt\"\n",
            "    pred = lab1_infer.load_predictor(ckpt_p, device=lab1_cfg.DEVICE)\n",
            "    lat = lab1_infer.benchmark_latency(pred, batch_sizes=(1, 32), warmup_runs=10, measure_runs=50)\n",
            "    bench_rows.append((\n",
            "        m_name.upper(),\n",
            "        f\"{lat[1]['checkpoint_size_kb']:.2f} KB\",\n",
            "        f\"{lat[1]['mean_ms']:.3f} ms\",\n",
            "        f\"{lat[32]['mean_ms']:.3f} ms\",\n",
            "        f\"{lat[32]['samples_per_sec']:,.0f} img/s\",\n",
            "    ))\n",
            "\n",
            "df_lat = pd.DataFrame(bench_rows, columns=[\"Mô hình\", \"Checkpoint Size\", \"Latency (BS=1)\", \"Latency (BS=32)\", \"Throughput (BS=32)\"])\n",
            "v_objConsole.print(consoles.df_to_table(df_lat, title=\"PIPELINE 05: INFERENCE LATENCY BENCHMARK\"))\n",
            "\n",
            "# Suy luận thử 1 mẫu Test trên cnn_aug\n",
            "cnn_pred = lab1_infer.load_predictor(lab1_cfg.EXPERIMENT_DIR / \"cnn_aug\" / \"best.pt\", device=lab1_cfg.DEVICE)\n",
            "sample_pil, sample_y = FashionMNIST(root=lab1_cfg.DATA_PATH, train=False, download=False)[18]\n",
            "inf_res = lab1_infer.predict_single(cnn_pred, sample_pil)\n",
            "df_inf = pd.DataFrame(\n",
            "    [(r[\"class_id\"], r[\"label\"], f\"{r['prob']*100:.2f}%\") for r in inf_res[\"top3\"]],\n",
            "    columns=[\"Class ID\", \"Predicted Label\", \"Probability\"],\n",
            ")\n",
            "v_objConsole.print(consoles.df_to_table(df_inf, title=f\"SAMPLE #18 TOP-3 (TRUE: {eda_helpers.FASHION_CLASSES[sample_y]})\"))"
        ]),
    ]

    nb["cells"] = base_cells + added_cells
    return nb


def execute_notebook() -> None:
    nb_path = Path("labs/lab1/vibe/lab1.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb = append_pipeline_cells(nb)

    shared_console = Console(record=True, width=115)
    from src import consoles
    from src.vibe_scripts import eda_helpers

    consoles.v_objConsole = shared_console
    eda_helpers.console = shared_console

    current_cell_figures = []

    def custom_show(*args, **kwargs):
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        buf.seek(0)
        current_cell_figures.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close(fig)

    plt.show = custom_show
    exec_globals = {"__name__": "__main__", "plt": plt}
    exec_count = 1

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_str = "".join(cell["source"])
        print(f"--- Executing Cell {idx} ---")
        shared_console.clear()
        current_cell_figures.clear()

        old_stdout = sys.stdout
        buf_out = io.StringIO()
        sys.stdout = buf_out
        cell_outputs = []
        try:
            exec(code_str, exec_globals)
        finally:
            sys.stdout = old_stdout

        text_out = buf_out.getvalue()
        if text_out.strip():
            cell_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in text_out.splitlines()],
            })

        rich_text = shared_console.export_text(clear=False)
        rich_html = shared_console.export_html(clear=True, inline_styles=True)
        if rich_text.strip():
            cell_outputs.append({
                "output_type": "display_data",
                "data": {
                    "text/plain": [line + "\n" for line in rich_text.splitlines()],
                    "text/html": [rich_html],
                },
                "metadata": {},
            })

        for fnum in plt.get_fignums():
            fig = plt.figure(fnum)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            buf.seek(0)
            current_cell_figures.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close(fig)

        for b64_img in current_cell_figures:
            cell_outputs.append({
                "output_type": "display_data",
                "data": {"text/plain": ["<Figure>"], "image/png": b64_img},
                "metadata": {},
            })

        cell["outputs"] = cell_outputs
        cell["execution_count"] = exec_count
        exec_count += 1

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Executed and saved {nb_path} successfully.")


if __name__ == "__main__":
    execute_notebook()
