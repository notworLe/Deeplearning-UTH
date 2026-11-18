"""
Script thực thi notebook lab1.ipynb và nhúng đầy đủ outputs/visualizations chuẩn mực.
"""

import json
import io
import sys
import base64
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console


def execute_notebook():
    nb_path = Path("labs/lab1/vibe/lab1.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Cập nhật Cell 1 (Roadmap table)
    cell_1_source = [
        "## Bảng Lộ Trình Thực Thi & Liên Kết Tài Liệu Kỹ Thuật\n",
        "\n",
        "| Step | Description | What it does | Import path |\n",
        "| :---: | :--- | :--- | :---: |\n",
        "| `1` | Environment & Seed Setup | Thiết lập cấu hình thiết bị động, seed ngẫu nhiên 248 và console giám sát | [`src/config.py`](../../../src/config.py) |\n",
        "| `2` | Define Context & Problem | Bản đặc tả mục tiêu, ràng buộc công bằng thí nghiệm, baselines và metrics | — |\n",
        "| `3` | Immutable Raw Ingestion & Manifest | Tải Fashion-MNIST từ Zalando, kiểm toán manifest.json và xác thực mã băm SHA-256 (§3.2) | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `4` | 6-Dimension Quality Audit | Kiểm toán 6 chiều chất lượng dữ liệu: Completeness, Accuracy, Consistency, Timeliness, Uniqueness, Validity | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `5` | Dataset Splitting | Phân chia tập dữ liệu: Train (55.000), Validation (5.000), Test (10.000) bảo toàn phân tầng và seed 248 | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `6` | EDA Lens 1: Global Topology | Khảo sát cấu trúc hình học tổng thể, tensor shapes, kiểu dữ liệu và dung lượng bộ nhớ | — |\n",
        "| `7` | EDA Lens 2: Univariate Distribution | Đánh giá cân bằng lớp trên 3 tập và thống kê cường độ pixel (Histogram, KDE, Skewness, Kurtosis) | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `8` | EDA Lens 3: Visual & Separability | Thư viện ảnh mẫu 10 lớp, ảnh đại diện trung bình (Class Prototypes), tương quan và không gian PCA 2D | [`src/vibe_scripts/eda_helpers.py`](../../../src/vibe_scripts/eda_helpers.py) |\n",
        "| `9` | Technical Decisions | Đúc kết phát hiện hình học để làm căn cứ thiết kế pipeline tiền xử lý và mô hình ở giai đoạn sau | — |\n",
        "\n",
        "---\n",
        "### Liên Kết Nhanh Tài Liệu & Báo Cáo Kỹ Thuật\n",
        "- 📄 Báo Cáo Kỹ Thuật Toàn Diện: [`docs/REPORT.md`](docs/REPORT.md)\n",
        "- 🌐 Báo Cáo Trực Quan HTML: [`docs/REPORT.html`](docs/REPORT.html)\n",
        "- 📋 Đặc Tả Yêu Cầu Kỹ Thuật Bài Thực Hành: [`docs/QUESTION.md`](docs/QUESTION.md)\n",
        "- 📖 Phương Pháp Luận Pipeline Chuẩn Hóa: [`docs/reference/ML_PIPELINE.md`](../../../docs/reference/ML_PIPELINE.md)\n",
        "- 🏠 Tài Liệu Hướng Dẫn Dự Án: [`README.md`](../../../README.md)\n"
    ]
    nb["cells"][1]["source"] = cell_1_source

    # 2. Cập nhật Cell 5 (Markdown Section 3)
    cell_5_source = [
        "## 3. Thu Nạp & Quản Trị Dữ Liệu Bất Biến (Data Collection & Governance - §2, §3)\n",
        "\n",
        "Áp dụng nguyên tắc **Vùng Dữ Liệu Bất Biến (Immutable Raw Zone - §3.1)**:\n",
        "- Dữ liệu thô tải từ Zalando Research qua `torchvision.datasets.FashionMNIST` được lưu trữ bất biến tại `data/FashionMNIST/raw/`.\n",
        "- Toàn bộ các file nhị phân gốc là chỉ đọc (read-only), tuyệt đối không sửa đổi hay làm sạch trực tiếp trên vùng lưu trữ này.\n",
        "\n",
        "Áp dụng quy định **Nguồn Gốc Dữ Liệu, Bản Kê Khai Thu Nạp & Mã Băm Mật Mã Học (Data Provenance & Manifests - §3.2)**:\n",
        "- Mọi artifact dữ liệu thô trong vùng lưu trữ bắt buộc phải được ghép cặp với bản kê khai thu nạp (`manifest.json`) và manifest tương ứng (`<file>.manifest.json`).\n",
        "- Cấu trúc bản kê khai chuẩn hóa bao gồm: `dataset_name`, `raw_file`, `source_uri`, `collected_at`, `sha256_checksum`, `total_records`, `collection_agent_version`.\n",
        "- Việc tính toán và xác thực mã băm mật mã học (SHA-256) bảo đảm tính toàn vẹn 100% của tập dữ liệu trên mọi môi trường phát triển, ngăn chặn hoàn toàn nguy cơ sai lệch dữ liệu ngầm (silent data corruption)."
    ]
    nb["cells"][5]["source"] = cell_5_source

    # 3. Cập nhật Cell 6 (Code Section 3)
    cell_6_source = [
        "# Tải và khởi tạo dataset thô (dữ liệu pixel gốc 0-255)\n",
        "raw_train = FashionMNIST(root=config.DATA_PATH, train=True, download=False)\n",
        "raw_test  = FashionMNIST(root=config.DATA_PATH, train=False, download=False)\n",
        "\n",
        "# ---------- Bảng 1: thông tin dữ liệu thô ----------\n",
        "df_raw = pd.DataFrame(\n",
        "    [\n",
        "        (\"Số lượng mẫu Train thô\", f\"{len(raw_train):,}\"),\n",
        "        (\"Số lượng mẫu Test thô\",  f\"{len(raw_test):,}\"),\n",
        "        (\"Kích thước ảnh mẫu\",     f\"{tuple(raw_train.data[0].shape)} (H x W)\"),\n",
        "        (\"Kiểu dữ liệu gốc\",       str(raw_train.data.dtype)),\n",
        "        (\"Số lớp\",                 str(len(eda_helpers.FASHION_CLASSES))),\n",
        "        (\"Danh sách nhãn lớp\",     \", \".join(eda_helpers.FASHION_CLASSES)),\n",
        "    ],\n",
        "    columns=[\"metric\", \"value\"],\n",
        ")\n",
        "v_objConsole.print(consoles.df_to_table(df_raw, title=\"RAW DATASET OVERVIEW\", show_header=False))\n",
        "\n",
        "# ---------- Bảng 2 & 3: Ingestion Manifest & Checksum Verification (§3.2) ----------\n",
        "raw_dir = config.DATA_PATH / \"FashionMNIST\" / \"raw\"\n",
        "manifest_path = raw_dir / \"manifest.json\"\n",
        "\n",
        "manifest_data, verify_results = eda_helpers.load_and_verify_manifest(manifest_path)\n",
        "eda_helpers.print_manifest_table(manifest_data, verify_results)\n"
    ]
    nb["cells"][6]["source"] = cell_6_source

    # Tạo captured rich console
    shared_console = Console(record=True, width=110)

    from src import consoles
    from src.vibe_scripts import eda_helpers
    consoles.v_objConsole = shared_console
    eda_helpers.console = shared_console

    current_cell_figures = []
    def custom_show(*args, **kwargs):
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        b64_str = base64.b64encode(buf.read()).decode("utf-8")
        current_cell_figures.append(b64_str)
        plt.close(fig)

    plt.show = custom_show

    exec_globals = {
        "__name__": "__main__",
        "shared_console": shared_console,
        "plt": plt
    }
    execution_counter = 1

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        code_str = "".join(cell["source"])
        print(f"--- Executing Cell {idx} ---")

        shared_console.clear()
        current_cell_figures.clear()

        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        cell_outputs = []

        try:
            exec(code_str, exec_globals)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"Error executing cell {idx}: {e}")
            import traceback
            traceback.print_exc()
            raise e
        finally:
            sys.stdout = old_stdout

        text_out = redirected_output.getvalue()
        if text_out.strip():
            cell_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in text_out.splitlines()]
            })

        # Rich console captured output
        rich_text = shared_console.export_text(clear=False)
        rich_html = shared_console.export_html(clear=True, inline_styles=True)
        if rich_text.strip():
            cell_outputs.append({
                "output_type": "display_data",
                "data": {
                    "text/plain": [line + "\n" for line in rich_text.splitlines()],
                    "text/html": [rich_html]
                },
                "metadata": {}
            })

        # Captured figures
        # Remaining figures not closed
        fig_after = plt.get_fignums()
        for fnum in fig_after:
            fig = plt.figure(fnum)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64_str = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            current_cell_figures.append(b64_str)

        for b64_img in current_cell_figures:
            cell_outputs.append({
                "output_type": "display_data",
                "data": {
                    "text/plain": ["<Figure size ...>"],
                    "image/png": b64_img
                },
                "metadata": {}
            })

        cell["outputs"] = cell_outputs
        cell["execution_count"] = execution_counter
        execution_counter += 1

    # Lưu lại notebook đã thực thi
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Đã hoàn thành thực thi và lưu notebook thành công: {nb_path}")


if __name__ == "__main__":
    execute_notebook()
