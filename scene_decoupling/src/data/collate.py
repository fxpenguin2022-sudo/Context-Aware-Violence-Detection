from __future__ import annotations

from typing import Any

import torch


def collate_video_pose(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bsz = len(batch)
    max_nc = max(int(x['clips'].shape[0]) for x in batch)
    clip_len = int(batch[0]['clips'].shape[1])
    h = int(batch[0]['clips'].shape[3])
    w = int(batch[0]['clips'].shape[4])
    max_persons = int(batch[0]['poses'].shape[2])
    num_keypoints = int(batch[0]['poses'].shape[3])

    clips = torch.zeros((bsz, max_nc, clip_len, 3, h, w), dtype=torch.float32)
    poses = torch.zeros((bsz, max_nc, clip_len, max_persons, num_keypoints, 3), dtype=torch.float32)
    clip_valid_mask = torch.zeros((bsz, max_nc), dtype=torch.bool)

    for i, item in enumerate(batch):
        nc = int(item['clips'].shape[0])
        clips[i, :nc] = item['clips']
        poses[i, :nc] = item['poses']
        clip_valid_mask[i, :nc] = True

    return {
        'sample_id': [x['sample_id'] for x in batch],
        'video_id': [x['video_id'] for x in batch],
        'video_path': [x['video_path'] for x in batch],
        'pose_path': [x['pose_path'] for x in batch],
        'label': torch.tensor([x['label'] for x in batch], dtype=torch.float32),
        'clips': clips,
        'poses': poses,
        'clip_valid_mask': clip_valid_mask,
    }
