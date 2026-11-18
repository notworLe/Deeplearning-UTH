# tests/test_df_to_table.py
import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from src.consoles import df_to_table


def render(table, width=120) -> str:
    """Table -> chuỗi thuần (không màu, không phụ thuộc terminal)."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=False,
            color_system=None, legacy_windows=False).print(table)
    return buf.getvalue()


@pytest.fixture
def df():
    return pd.DataFrame({
        "model": ["resnet18", "resnet50", "vit"],
        "acc": [0.912, 0.934, np.nan],
        "params_m": [11.7, 25.6, 86.0],
    })


# ---------- cấu trúc ----------
def test_structure(df):
    t = df_to_table(df)
    assert len(t.columns) == 3
    assert t.row_count == 3


def test_numeric_columns_right_aligned(df):
    t = df_to_table(df)
    assert [c.justify for c in t.columns] == ["left", "right", "right"]


# ---------- nội dung hiển thị ----------
def test_nan_shown_as_dash(df):
    out = render(df_to_table(df))
    assert "nan" not in out.lower()
    assert "-" in out


def test_float_format(df):
    out = render(df_to_table(df, float_fmt=".2f"))
    assert "0.91" in out
    assert "0.912" not in out


def test_markup_is_escaped():
    out = render(df_to_table(pd.DataFrame({"x": ["[red]hello[/red]"]})))
    assert "[red]hello[/red]" in out        # in nguyên văn, không bị ăn thẻ


# ---------- tùy chọn ----------
def test_show_header_false(df):
    out = render(df_to_table(df, show_header=False))
    assert "params_m" not in out            # tên cột không hiện
    assert "resnet18" in out                # dữ liệu vẫn hiện


def test_show_index():
    d = pd.DataFrame({"a": [1, 2]}, index=["r1", "r2"])
    out = render(df_to_table(d, show_index=True))
    assert "r1" in out and "r2" in out


# ---------- rút gọn dòng ----------
def test_truncation_keeps_head_and_tail():
    d = pd.DataFrame({"v": [f"item{i:03d}" for i in range(100)]})
    t = df_to_table(d, max_rows=10)
    out = render(t)
    assert t.row_count == 11                # 5 đầu + "..." + 5 cuối
    assert "item000" in out and "item004" in out
    assert "item095" in out and "item099" in out
    assert "item050" not in out             # đoạn giữa bị bỏ
    assert "..." in out


def test_no_truncation_when_small(df):
    assert "..." not in render(df_to_table(df, max_rows=10))


def test_empty_dataframe():
    t = df_to_table(pd.DataFrame({"a": [], "b": []}))
    assert t.row_count == 0
    render(t)                               # không được raise


# ---------- bảng phân chia tập ----------
@pytest.mark.parametrize("sizes", [(100, 20, 30), (0, 0, 0)])
def test_split_table(sizes):
    d = pd.DataFrame(
        list(zip(["Train", "Val", "Test"], sizes)), columns=["split", "samples"]
    )
    out = render(df_to_table(d, show_header=False))
    for name in ["Train", "Val", "Test"]:
        assert name in out