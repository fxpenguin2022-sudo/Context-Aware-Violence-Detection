from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def inspect_npz(npz_path: str, key_name: str) -> dict[str, int]:
    arr = np.load(npz_path)[key_name]
    if arr.ndim != 4:
        raise ValueError(f"Expected [T, M, K, C], got {arr.shape} for {npz_path}")
    t, m, k, c = arr.shape
    return {"frames": int(t), "persons": int(m), "joints": int(k), "channels": int(c)}


def build_pose_index(
    pose_root: str,
    class_to_label: dict[str, int],
    split_names: dict[str, str],
    key_name: str,
    out_file: str,
) -> list[dict[str, Any]]:
    root = Path(pose_root)
    rows: list[dict[str, Any]] = []

    all_files: list[tuple[str, int, Path, str]] = []
    for logical_split, real_split in split_names.items():
        for class_name, label in class_to_label.items():
            class_dir = root / real_split / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.glob("*.npz")):
                all_files.append((logical_split, int(label), path, class_name))

    iterator = all_files
    if tqdm is not None:
        iterator = tqdm(all_files, total=len(all_files), desc="BuildIndex", leave=False)

    for logical_split, label, path, class_name in iterator:
        meta = inspect_npz(str(path), key_name)
        rows.append(
            {
                "video_id": path.stem,
                "split": logical_split,
                "class_name": class_name,
                "label": int(label),
                "pose_path": str(path.resolve()),
                **meta,
            }
        )

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return rows


def load_index(index_file: str, split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(index_file).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if split is None or row["split"] == split:
                rows.append(row)
    return rows
