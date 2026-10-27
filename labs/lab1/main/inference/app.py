# app.py
import math
from pathlib import Path
import gradio as gr
from PIL import Image
from inference import predict_batch

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "input"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAGE_SIZE = 12

def resolve_path(s: str) -> Path:
    p = Path(s.strip().strip('"'))
    return p if p.is_absolute() else INPUT_DIR / p

def scan(folder: str, top_k: int):
    d = resolve_path(folder)
    if not d.is_dir():
        raise gr.Error(f"Không phải folder: {d}")
    files = sorted(f for f in d.iterdir() if f.suffix.lower() in EXTS)
    if not files:
        raise gr.Error("Folder không có ảnh.")

    imgs = [Image.open(f).convert("RGB") for f in files]   # ảnh nhiều thì xem lưu ý bên dưới
    preds = predict_batch(imgs, int(top_k))
    results = [(str(f), " | ".join(f"{c} {p:.0%}" for c, p in pr))
               for f, pr in zip(files, preds)]
    return (results, 0, *render(results, 0))

def render(results, page):
    n_pages = max(1, math.ceil(len(results) / PAGE_SIZE))
    page = min(max(page, 0), n_pages - 1)
    items = results[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    return items, f"Trang {page + 1}/{n_pages}  ({len(results)} ảnh)"

def go(results, page, delta):
    n_pages = max(1, math.ceil(len(results) / PAGE_SIZE))
    page = min(max(page + delta, 0), n_pages - 1)
    gallery, label = render(results, page)
    return page, gallery, label

with gr.Blocks() as demo:
    results_st = gr.State([])
    page_st = gr.State(0)

    folder = gr.Textbox(label="Folder ảnh (tương đối từ input/ hoặc tuyệt đối)")
    top_k = gr.Slider(1, 5, value=1, step=1, label="Top-k")
    run_btn = gr.Button("Predict folder", variant="primary")

    gallery = gr.Gallery(columns=4, height="auto", label="Kết quả")
    with gr.Row():
        prev_btn = gr.Button("◀ Trước")
        page_lbl = gr.Markdown("Trang 0/0")
        next_btn = gr.Button("Sau ▶")

    run_btn.click(scan, [folder, top_k], [results_st, page_st, gallery, page_lbl])
    prev_btn.click(lambda r, p: go(r, p, -1), [results_st, page_st], [page_st, gallery, page_lbl])
    next_btn.click(lambda r, p: go(r, p, +1), [results_st, page_st], [page_st, gallery, page_lbl])

demo.launch()