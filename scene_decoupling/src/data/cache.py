from __future__ import annotations

from collections import OrderedDict
from typing import Any


class RamCache:
    def __init__(self, capacity: int = 128) -> None:
        self.capacity = max(1, int(capacity))
        self.data: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self.data:
            return None
        val = self.data.pop(key)
        self.data[key] = val
        return val

    def set(self, key: str, value: Any) -> None:
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)


class NullCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any) -> None:
        return None
