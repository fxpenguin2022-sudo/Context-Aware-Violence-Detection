from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def inspect_pose(path: str, key_name: str) -> dict[str, int]:
    arr = np.load(path)[key_name]
    if arr.ndim != 4:
        raise ValueError(f'Invalid pose shape {arr.shape} in {path}')
    t, m, k, c = arr.shape
    return {'pose_frames': int(t), 'pose_persons': int(m), 'pose_joints': int(k), 'pose_channels': int(c)}


def build_video_pose_index(
    video_root: str,
    pose_root: str,
    class_to_label: dict[str, int],
    split_names: dict[str, str],
    key_name: str,
    out_file: str,
) -> list[dict[str, Any]]:
    vroot = Path(video_root)
    proot = Path(pose_root)
    rows: list[dict[str, Any]] = []

    files: list[tuple[str, str, int, Path, Path]] = []
    for logical_split, split_name in split_names.items():
        for cls, label in class_to_label.items():
            vdir = vroot / split_name / cls
            pdir = proot / split_name / cls
            if not vdir.exists() or not pdir.exists():
                continue
            for vpath in sorted(vdir.glob('*.avi')):
                ppath = pdir / f'{vpath.stem}.npz'
                if ppath.exists():
                    files.append((logical_split, cls, int(label), vpath, ppath))

    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, total=len(files), desc='IndexJoint', leave=False)

    for split, cls, label, vpath, ppath in iterator:
        meta = inspect_pose(str(ppath), key_name)
        rows.append(
            {
                'sample_id': f'{split}/{cls}/{vpath.stem}',
                'video_id': vpath.stem,
                'split': split,
                'class_name': cls,
                'label': label,
                'video_path': str(vpath.resolve()),
                'pose_path': str(ppath.resolve()),
                **meta,
            }
        )

    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')

    return rows


def load_index(path: str, split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if split is None or row['split'] == split:
                rows.append(row)
    return rows
