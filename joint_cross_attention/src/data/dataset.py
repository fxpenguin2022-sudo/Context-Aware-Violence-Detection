from __future__ import annotations

from typing import Any

from scene_decoupling.src.data.dataset import VideoPoseContextDataset
from src.data.rwf_dataset import RWFPoseDataset


class JointRWF2000Dataset:
    def __init__(self, records: list[dict[str, Any]], cfg_data: dict[str, Any], split: str, seed: int) -> None:
        self.records = records
        self.cfg_data = cfg_data
        self.split = split
        self.seed = int(seed)
        self.current_epoch = 0

        self.pose_cfg = self._build_pose_cfg(cfg_data)
        self.context_cfg = self._build_context_cfg(cfg_data)

        pose_records = [
            {
                'video_id': rec['video_id'],
                'pose_path': rec['pose_path'],
                'label': int(rec['label']),
                'frames': int(rec.get('pose_frames', 0)),
            }
            for rec in records
        ]

        self.pose_ds = RWFPoseDataset(pose_records, self.pose_cfg, split=split, seed=self.seed)
        self.context_ds = VideoPoseContextDataset(records, self.context_cfg, split=split, seed=self.seed)

        if len(self.pose_ds) != len(self.context_ds):
            raise ValueError('Pose/context dataset lengths do not match')

    @staticmethod
    def _build_pose_cfg(cfg_data: dict[str, Any]) -> dict[str, Any]:
        branch = dict(cfg_data['pose_branch'])
        branch['key_name'] = cfg_data['key_name']
        branch['max_persons'] = int(branch['max_persons'])
        branch['num_keypoints'] = int(branch['num_keypoints'])
        return branch

    @staticmethod
    def _build_context_cfg(cfg_data: dict[str, Any]) -> dict[str, Any]:
        branch = dict(cfg_data['context_branch'])
        branch['key_name'] = cfg_data['key_name']
        branch['max_persons'] = int(branch['max_persons'])
        branch['num_keypoints'] = int(branch['num_keypoints'])
        return branch

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self.pose_ds.seed = int(self.seed + self.current_epoch * 1000003)
        if hasattr(self.context_ds, 'set_epoch'):
            self.context_ds.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pose_item = self.pose_ds[index]
        context_item = self.context_ds[index]
        if pose_item['video_id'] != context_item['video_id']:
            raise ValueError(f"Mismatched video_id at index {index}: {pose_item['video_id']} vs {context_item['video_id']}")
        return {
            'sample_id': context_item['sample_id'],
            'video_id': context_item['video_id'],
            'video_path': context_item['video_path'],
            'pose_path': context_item['pose_path'],
            'label': int(context_item['label']),
            'pose_windows': pose_item['windows'],
            'pose_window_valid': pose_item['window_valid'],
            'video_clips': context_item['clips'],
            'video_poses': context_item['poses'],
        }
