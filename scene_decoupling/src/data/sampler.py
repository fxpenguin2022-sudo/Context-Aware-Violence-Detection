from __future__ import annotations

import numpy as np


def _clip_indices(start: int, clip_len: int, total: int) -> np.ndarray:
    idx = np.arange(start, start + clip_len, dtype=np.int64)
    idx = np.clip(idx, 0, max(total - 1, 0))
    return idx


def _subsample_starts(starts: np.ndarray, max_clips: int, train_mode: bool, rng: np.random.Generator) -> np.ndarray:
    if max_clips <= 0 or starts.shape[0] <= max_clips:
        return starts

    if train_mode:
        # Randomly cover the full sequence instead of taking one contiguous
        # chunk, which reduces temporal bias while keeping compute bounded.
        keep_idx = np.sort(rng.choice(starts.shape[0], size=max_clips, replace=False))
        return starts[keep_idx]

    # Deterministic uniform sampling for eval.
    keep_idx = np.linspace(0, starts.shape[0] - 1, max_clips).round().astype(np.int64)
    return starts[keep_idx]


def build_stream_clip_indices(
    total: int,
    clip_len: int,
    clip_step: int,
    train_mode: bool,
    rng: np.random.Generator,
    full_sequence: bool = True,
    max_clips: int = 0,
) -> np.ndarray:
    if total <= 0:
        return np.zeros((1, clip_len), dtype=np.int64)

    step = max(1, int(clip_step))

    if full_sequence:
        offset = int(rng.integers(0, step)) if train_mode else 0
        starts = np.arange(offset, total, step, dtype=np.int64)
    else:
        max_start = max(total - clip_len, 0)
        nclips = max(1, int(max_clips) if int(max_clips) > 0 else 8)
        if nclips == 1:
            starts = np.array([int(rng.integers(0, max_start + 1))], dtype=np.int64)
        else:
            anchors = np.linspace(0, max_start, nclips)
            if train_mode:
                jitter = rng.integers(-max(1, clip_len // 2), max(1, clip_len // 2) + 1, size=nclips)
            else:
                jitter = np.zeros(nclips, dtype=np.int64)
            starts = np.clip(np.round(anchors).astype(np.int64) + jitter, 0, max_start)

    starts = _subsample_starts(starts, max_clips=max_clips, train_mode=train_mode, rng=rng)
    return np.stack([_clip_indices(int(s), clip_len, total) for s in starts], axis=0)


def map_video_to_pose_indices(video_idx: np.ndarray, t_video: int, t_pose: int) -> np.ndarray:
    if t_video <= 1 or t_pose <= 1:
        return np.zeros_like(video_idx)
    scale = (t_pose - 1) / float(t_video - 1)
    pose_idx = np.rint(video_idx * scale).astype(np.int64)
    return np.clip(pose_idx, 0, t_pose - 1)
