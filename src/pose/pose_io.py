from __future__ import annotations

from pathlib import Path

import numpy as np


def save_pose_npz(path: str, keypoints: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), keypoints=keypoints.astype(np.float32))


def load_pose_npz(path: str, key_name: str = "keypoints") -> np.ndarray:
    arr = np.load(path)[key_name]
    return arr.astype(np.float32)
