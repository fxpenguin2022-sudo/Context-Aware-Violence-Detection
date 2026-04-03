from __future__ import annotations

from typing import Any

import torch


def collate_joint(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bsz = len(batch)
    pose_windows = torch.stack([x['pose_windows'] for x in batch], dim=0).float()
    pose_window_valid = torch.stack([x['pose_window_valid'] for x in batch], dim=0).bool()

    max_nc = max(int(x['video_clips'].shape[0]) for x in batch)
    clip_len = int(batch[0]['video_clips'].shape[1])
    h = int(batch[0]['video_clips'].shape[3])
    w = int(batch[0]['video_clips'].shape[4])
    max_persons = int(batch[0]['video_poses'].shape[2])
    num_keypoints = int(batch[0]['video_poses'].shape[3])

    video_clips = torch.zeros((bsz, max_nc, clip_len, 3, h, w), dtype=torch.float32)
    video_poses = torch.zeros((bsz, max_nc, clip_len, max_persons, num_keypoints, 3), dtype=torch.float32)
    clip_valid_mask = torch.zeros((bsz, max_nc), dtype=torch.bool)

    for i, item in enumerate(batch):
        nc = int(item['video_clips'].shape[0])
        video_clips[i, :nc] = item['video_clips']
        video_poses[i, :nc] = item['video_poses']
        clip_valid_mask[i, :nc] = True

    return {
        'sample_id': [x['sample_id'] for x in batch],
        'video_id': [x['video_id'] for x in batch],
        'video_path': [x['video_path'] for x in batch],
        'pose_path': [x['pose_path'] for x in batch],
        'label': torch.tensor([x['label'] for x in batch], dtype=torch.float32),
        'pose_windows': pose_windows,
        'pose_window_valid': pose_window_valid,
        'video_clips': video_clips,
        'video_poses': video_poses,
        'clip_valid_mask': clip_valid_mask,
    }
