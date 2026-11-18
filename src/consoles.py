import torch
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.table import Table
import pandas as pd

import src.config as config

v_objConsole = Console()


def print_log(name, value, max_length=config.MAX_LENGTH):
    """Log dạng 'tên    : giá trị', tên có màu, canh lề thống nhất."""
    if len(name) > max_length:
        name = name[:max_length - 3] + "..."
    # escape: tránh rich hiểu nhầm giá trị dạng [..] là markup
    v_objConsole.print(f"[cyan]{name:<{max_length}}[/] : {escape(str(value))}")


def creation():
    """Bảng cấu hình chung."""
    v_objTable = Table(title="Configuration", show_header=False)
    v_objTable.add_column(style="cyan")
    v_objTable.add_column()

    v_objTable.add_row("Root", str(config.ROOT_PATH))
    v_objTable.add_row("Version", str(torch.version.__version__))
    v_objTable.add_row("Device", str(config.DEVICE))
    if config.DEVICE.type == "cuda":
        v_objTable.add_row("GPU", torch.cuda.get_device_name(config.DEVICE))
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        v_objTable.add_row("VRAM", f"{vram:.1f} GB")
    v_objTable.add_row("Seed", str(config.SEED))
    v_objTable.add_row("Workers", str(config.WOKER))
    v_objTable.add_row("Batch size", str(config.BATCH_SIZE))
    v_objTable.add_row("Learning rate", str(config.LR))
    v_objTable.add_row("Epoch", str(config.EPOCH))

    v_objConsole.print(v_objTable)


def dataset_log(dataset, train_dataset, test_dataset, idx=0):
    v_objConsole.print(Rule("Dataset information", style="green"))

    print_log("Dataset name", str(dataset), 20)
    print_log("Training dataset size", len(train_dataset), 20)
    print_log("Test dataset size", len(test_dataset), 20)

    img, label = test_dataset[idx]
    print_log("Label index", label, 20)
    print_log("Label name", dataset.classes[label], 20)
    print_log("Image shape", img.shape, 20)
    print_log("Min - max image", f"{img.min()}, {img.max()}", 20)


def dataset_log_v2(class_name: dict, test_dataset, idx=0, **kwargs):
    v_objConsole.print(Rule("Dataset information", style="green"))

    for key, val in kwargs.items():
        print_log(f"{key} size", len(val), 20)
    print_log("Test size", len(test_dataset), 20)

    img, label = test_dataset[idx]
    print_log("Num of class", len(test_dataset.classes), 20)
    print_log("Label index", label, 20)
    print_log("Label name", class_name[label], 20)
    print_log("Image shape", img.shape, 20)
    print_log("Min - max image", f"{img.min()}, {img.max()}", 20)


def dataloader_log(train_dataloader, test_dataloader, max_length=20):
    v_objConsole.print(Rule("Dataloader", style="green"))
    print_log("Train dataloader", f"{len(train_dataloader)} batch", max_length)
    print_log("Test dataloader", f"{len(test_dataloader)} batch", max_length)


def dataloader_log_v2(max_length=20, **kwargs):
    v_objConsole.print(Rule("Dataloader", style="green"))
    for name, loader in kwargs.items():
        print_log(f"{name} dataloader", f"{len(loader)} batch", max_length)

def df_to_table(df, title=None, max_rows=10, show_index=False, float_fmt=".3f", show_header=True):
    """DataFrame -> rich Table. Cột số canh phải, NaN hiện '-', dữ liệu dài thì rút gọn."""
    v_objConsole = Table(title=title, caption=f"{df.shape[0]} rows × {df.shape[1]} cols", show_header=show_header)

    if show_index:
        v_objConsole.add_column(str(df.index.name or ""), style="dim")
    for col in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[col])
        v_objConsole.add_column(
            str(col),
            justify="right" if is_num else "left",
            style="cyan" if is_num else None,
        )

    def fmt(x):
        if pd.isna(x):
            return "[dim]-[/]"
        if isinstance(x, float):
            return format(x, float_fmt)
        return escape(str(x))          # tránh [..] bị hiểu nhầm là markup

    def add(idx, row):
        cells = [fmt(x) for x in row]
        v_objConsole.add_row(*([escape(str(idx))] + cells if show_index else cells))

    if len(df) <= max_rows:
        for idx, row in zip(df.index, df.itertuples(index=False)):
            add(idx, row)
    else:                               # lấy đầu + cuối, giữa là dòng "..."
        half = max_rows // 2
        head, tail = df.head(half), df.tail(half)
        for idx, row in zip(head.index, head.itertuples(index=False)):
            add(idx, row)
        v_objConsole.add_row(*(["..."] * (df.shape[1] + int(show_index))))
        for idx, row in zip(tail.index, tail.itertuples(index=False)):
            add(idx, row)

    return v_objConsole


if __name__ == "__main__":
    creation()