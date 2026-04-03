from __future__ import annotations

from typing import Any

import torch


def rwf_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    video_ids = [x["video_id"] for x in batch]
    pose_paths = [x["pose_path"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)

    windows = torch.stack([x["windows"] for x in batch], dim=0).float()
    window_valid = torch.stack([x["window_valid"] for x in batch], dim=0).bool()
    frame_count = torch.tensor([x["frame_count"] for x in batch], dtype=torch.int64)

    return {
        "video_id": video_ids,
        "pose_path": pose_paths,
        "label": labels,
        "windows": windows,
        "window_valid": window_valid,
        "frame_count": frame_count,
    }
