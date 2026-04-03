from __future__ import annotations

import csv
from pathlib import Path


class CsvWriter:
    def __init__(self, path: str, headers: list[str]) -> None:
        self.path = Path(path)
        self.headers = headers
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def write(self, row: dict) -> None:
        with self.path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)
