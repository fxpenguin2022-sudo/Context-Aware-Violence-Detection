from __future__ import annotations

from collections import OrderedDict
from typing import Any


class RamCache:
    def __init__(self, capacity: int = 512) -> None:
        self.capacity = max(1, capacity)
        self._data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)


class NullCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any) -> None:
        return None
