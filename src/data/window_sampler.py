from __future__ import annotations

import numpy as np


def _build_window_indices(start: int, window: int, total: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(start, start + window, dtype=np.int64)
    valid = idx < total
    if total > 0:
        idx = np.clip(idx, 0, total - 1)
    else:
        idx = np.zeros_like(idx)
        valid[:] = False
    return idx, valid


def sample_train_windows(
    total_frames: int,
    window_size: int,
    num_windows: int,
    temporal_jitter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    max_start = max(total_frames - window_size, 0)
    starts = rng.integers(0, max_start + 1, size=num_windows, endpoint=False) if max_start > 0 else np.zeros(num_windows, dtype=np.int64)

    if temporal_jitter > 0:
        jitter = rng.integers(-temporal_jitter, temporal_jitter + 1, size=num_windows)
        starts = np.clip(starts + jitter, 0, max_start)

    win_idx = []
    win_mask = []
    for s in starts:
        idx, mask = _build_window_indices(int(s), window_size, total_frames)
        win_idx.append(idx)
        win_mask.append(mask)
    return np.stack(win_idx), np.stack(win_mask)


def sample_eval_windows(
    total_frames: int,
    window_size: int,
    num_windows: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_start = max(total_frames - window_size, 0)

    starts = list(range(0, max_start + 1, max(1, stride)))
    if not starts:
        starts = [0]

    if len(starts) >= num_windows:
        idxs = np.linspace(0, len(starts) - 1, num_windows, dtype=np.int64)
        starts = [starts[i] for i in idxs]
    else:
        starts = np.linspace(0, max_start, num_windows, dtype=np.int64).tolist()

    win_idx = []
    win_mask = []
    for s in starts:
        idx, mask = _build_window_indices(int(s), window_size, total_frames)
        win_idx.append(idx)
        win_mask.append(mask)
    return np.stack(win_idx), np.stack(win_mask)
