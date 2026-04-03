from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import NullCache, RamCache
from .transforms import add_velocity_channel, normalize_keypoints, random_augment
from .window_sampler import sample_eval_windows, sample_train_windows


@dataclass
class AugCfg:
    jitter_std: float
    drop_joint_prob: float
    drop_person_prob: float
    temporal_jitter: int


class RWFPoseDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        data_cfg: dict[str, Any],
        split: str,
        seed: int,
    ) -> None:
        self.records = records
        self.split = split
        self.data_cfg = data_cfg
        self.seed = seed
        self.train_mode = split == "train"

        self.key_name = data_cfg["key_name"]
        self.max_persons = int(data_cfg["max_persons"])
        self.num_keypoints = int(data_cfg["num_keypoints"])
        self.window_size = int(data_cfg["window_size"])
        self.num_windows = int(data_cfg["train_num_windows"] if self.train_mode else data_cfg["eval_num_windows"])
        self.eval_stride = int(data_cfg["eval_window_stride"])
        self.include_velocity = bool(data_cfg.get("include_velocity", True))
        self.min_valid_conf = float(data_cfg.get("min_valid_conf", 0.0))
        self.frame_valid_mean_conf = float(data_cfg.get("frame_valid_mean_conf", 0.0))
        self.frame_valid_active_ratio = float(data_cfg.get("frame_valid_active_ratio", 0.0))

        aug_dict = data_cfg["augment"]["train" if self.train_mode else "eval"]
        self.aug_cfg = AugCfg(
            jitter_std=float(aug_dict["jitter_std"]),
            drop_joint_prob=float(aug_dict["drop_joint_prob"]),
            drop_person_prob=float(aug_dict["drop_person_prob"]),
            temporal_jitter=int(aug_dict["temporal_jitter"]),
        )

        cache_mode = data_cfg.get("cache_mode", "none")
        if cache_mode == "ram":
            self.cache = RamCache(capacity=int(data_cfg.get("cache_size", 512)))
        else:
            self.cache = NullCache()

        norm_cfg = data_cfg.get("normalize", {})
        self.norm_enabled = bool(norm_cfg.get("enabled", False))
        self.center_joint = int(norm_cfg.get("center_joint", 11))

    def __len__(self) -> int:
        return len(self.records)

    def _load_pose(self, path: str) -> np.ndarray:
        cached = self.cache.get(path)
        if cached is not None:
            return cached

        arr = np.load(path)[self.key_name].astype(np.float32)

        if arr.ndim != 4:
            raise ValueError(f"Unexpected pose shape {arr.shape} from {path}")

        t, m, k, c = arr.shape
        if k != self.num_keypoints:
            raise ValueError(f"Unexpected keypoints count {k}, expected {self.num_keypoints} for {path}")
        if c < 3:
            raise ValueError(f"Expected at least 3 channels, got {c} for {path}")

        arr = arr[:, :, :, :3]

        if self.min_valid_conf > 0:
            valid = arr[..., 2] >= self.min_valid_conf
            arr[..., :2] = np.where(valid[..., None], arr[..., :2], 0.0)
            arr[..., 2] = np.where(valid, arr[..., 2], 0.0)

        # Keep top persons by mean confidence and pad/truncate to max_persons.
        conf_score = arr[..., 2].mean(axis=(0, 2))
        order = np.argsort(-conf_score)
        arr = arr[:, order]

        if m >= self.max_persons:
            arr = arr[:, : self.max_persons]
        else:
            pad = np.zeros((t, self.max_persons - m, k, 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)

        self.cache.set(path, arr)
        return arr

    def _sample_windows(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        t = x.shape[0]
        if self.train_mode:
            return sample_train_windows(
                total_frames=t,
                window_size=self.window_size,
                num_windows=self.num_windows,
                temporal_jitter=self.aug_cfg.temporal_jitter,
                rng=rng,
            )
        return sample_eval_windows(
            total_frames=t,
            window_size=self.window_size,
            num_windows=self.num_windows,
            stride=self.eval_stride,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        pose = self._load_pose(rec["pose_path"])  # [T, M, K, 3]

        rng = np.random.default_rng(self.seed + index * 17)

        pose_conf = pose[..., 2].copy()

        if self.norm_enabled:
            pose = normalize_keypoints(pose, center_joint=self.center_joint)

        if self.train_mode:
            pose = random_augment(
                pose,
                jitter_std=self.aug_cfg.jitter_std,
                drop_joint_prob=self.aug_cfg.drop_joint_prob,
                drop_person_prob=self.aug_cfg.drop_person_prob,
                rng=rng,
            )

        if self.include_velocity:
            pose = add_velocity_channel(pose)

        window_idx, window_mask = self._sample_windows(pose, rng)
        base_window_mask = window_mask.copy()

        if self.frame_valid_mean_conf > 0 or self.frame_valid_active_ratio > 0:
            frame_mean_conf = pose_conf.mean(axis=(1, 2))
            frame_active_ratio = (pose_conf > 0).mean(axis=(1, 2))
            sampled_mean_conf = frame_mean_conf[window_idx]
            sampled_active_ratio = frame_active_ratio[window_idx]
            quality_mask = np.ones_like(window_mask, dtype=bool)
            if self.frame_valid_mean_conf > 0:
                quality_mask &= sampled_mean_conf >= self.frame_valid_mean_conf
            if self.frame_valid_active_ratio > 0:
                quality_mask &= sampled_active_ratio >= self.frame_valid_active_ratio
            window_mask = window_mask & quality_mask
            if not window_mask.any():
                window_mask = base_window_mask

        windows = pose[window_idx]  # [W, L, M, K, C]

        return {
            "video_id": rec["video_id"],
            "pose_path": rec["pose_path"],
            "label": int(rec["label"]),
            "windows": torch.from_numpy(windows),
            "window_valid": torch.from_numpy(window_mask),
            "frame_count": int(rec["frames"]),
        }
