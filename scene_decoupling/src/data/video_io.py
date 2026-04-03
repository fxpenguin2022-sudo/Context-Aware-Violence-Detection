from __future__ import annotations

import time

import cv2
import numpy as np


def decode_video_rgb(
    path: str,
    resize_to: tuple[int, int] | None = None,
    return_orig_hw: bool = False,
    read_timeout_ms: int = 0,
    max_decode_seconds: float = 0.0,
    max_frames: int = 0,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')

    # Best-effort backend timeout knobs (may be ignored by some OpenCV builds).
    if read_timeout_ms > 0:
        if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
            cap.set(getattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'), float(read_timeout_ms))
        if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
            cap.set(getattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'), float(read_timeout_ms))

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resize_wh: tuple[int, int] | None = None
    if resize_to is not None:
        # resize_to is (H, W), cv2 expects (W, H)
        resize_wh = (int(resize_to[1]), int(resize_to[0]))

    # Hard stop to avoid decode loops on corrupted streams.
    hard_cap = int(max_frames)
    if hard_cap <= 0 and expected_frames > 0:
        hard_cap = int(expected_frames * 2 + 32)

    frames = []
    start_ts = time.monotonic()
    try:
        while True:
            if max_decode_seconds > 0 and (time.monotonic() - start_ts) > max_decode_seconds:
                raise RuntimeError(
                    f'Decode timeout ({max_decode_seconds:.1f}s): {path}'
                )

            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                continue

            if resize_wh is not None:
                frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if hard_cap > 0 and len(frames) >= hard_cap:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f'No frames decoded: {path}')
    if orig_h <= 0 or orig_w <= 0:
        orig_h, orig_w = int(frames[0].shape[0]), int(frames[0].shape[1])

    stacked = np.stack(frames, axis=0)  # [Tv, H, W, 3]
    if return_orig_hw:
        return stacked, (orig_h, orig_w)
    return stacked


def resize_frames(frames: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    out = [cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in frames]
    return np.stack(out, axis=0)


def maybe_color_jitter(frames: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return frames
    alpha = 1.0 + rng.uniform(-strength, strength)
    beta = rng.uniform(-strength, strength) * 255.0
    x = frames.astype(np.float32) * alpha + beta
    return np.clip(x, 0, 255).astype(np.uint8)
