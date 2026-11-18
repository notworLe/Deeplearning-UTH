"""Cơ chế kế thừa và override an toàn cho cấu hình mặc định từ `src/config.py`."""

from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Set

import src.config as default_base_config


def _extract_upper_constants(module: ModuleType) -> Dict[str, Any]:
    """Trích xuất toàn bộ hằng số viết HOA hợp lệ từ module cấu hình gốc."""
    return {
        name: getattr(module, name)
        for name in dir(module)
        if name.isupper() and not name.startswith("_")
    }


class VibeConfig:
    """
    Quản lý cấu hình kế thừa từ `src/config.py` và chỉ cho phép override đúng tên biến gốc.

    Khi `strict=True`, truyền sai tên biến không tồn tại trong module gốc sẽ báo `KeyError` ngay.
    Các tham số mở rộng riêng cho từng lab được khai báo tường minh qua `extras`.
    """

    def __init__(
        self,
        base_module: ModuleType = default_base_config,
        strict: bool = True,
        extras: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> None:
        base_values = _extract_upper_constants(base_module)
        self._base_keys: Set[str] = set(base_values.keys())
        self._values: Dict[str, Any] = dict(base_values)
        self._overridden: Dict[str, Any] = {}
        self._extras: Set[str] = set()

        self.override(strict=strict, **overrides)
        if extras:
            self.add_extras(**extras)

    def override(self, strict: bool = True, **overrides: Any) -> "VibeConfig":
        """Ghi đè các biến cấu hình đã tồn tại trong module gốc."""
        for key, value in overrides.items():
            if strict and key not in self._base_keys:
                available = ", ".join(sorted(self._base_keys))
                raise KeyError(
                    f"Biến '{key}' không tồn tại trong cấu hình gốc. "
                    f"Các biến hợp lệ: [{available}]"
                )
            self._values[key] = value
            self._overridden[key] = value
        return self

    def add_extras(self, **extras: Any) -> "VibeConfig":
        """Đăng ký thêm tham số mở rộng riêng cho bài lab (không có trong `src/config.py`)."""
        for key, value in extras.items():
            if key in self._base_keys:
                raise ValueError(
                    f"Biến '{key}' đã tồn tại trong cấu hình gốc; "
                    "hãy truyền trực tiếp vào override thay vì `extras`."
                )
            self._values[key] = value
            self._extras.add(key)
        return self

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(f"'{type(self).__name__}' không có thuộc tính '{name}'")

    def to_dict(self, keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Xuất cấu hình dạng dictionary (hữu ích khi lưu metadata checkpoint/history)."""
        if keys is None:
            return dict(self._values)
        return {k: self._values[k] for k in keys}

    def overridden(self) -> Dict[str, Any]:
        """Trả về dictionary chỉ gồm các biến đã override từ cấu hình gốc."""
        return dict(self._overridden)

    def export_to_namespace(self, namespace: Dict[str, Any]) -> None:
        """Đưa toàn bộ hằng số đã hợp nhất ra namespace cấp module."""
        namespace.update(self._values)
