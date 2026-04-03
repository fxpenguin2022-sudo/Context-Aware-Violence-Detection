from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .cache import NullCache, RamCache
from .sampler import build_stream_clip_indices, map_video_to_pose_indices
from .video_io import decode_video_rgb, maybe_color_jitter


@dataclass
class AugCfg:
    horizontal_flip_prob: float
    color_jitter: float


class VideoPoseContextDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], cfg_data: dict[str, Any], split: str, seed: int) -> None:
        self.records = records
        self.cfg = cfg_data
        self.split = split
        self.train_mode = split == 'train'
        self.seed = int(seed)

        self.key_name = cfg_data['key_name']
        self.clip_len = int(cfg_data['clip_len'])
        self.clip_step = int(cfg_data.get('clip_step', max(1, self.clip_len // 2)))
        self.full_sequence = bool(cfg_data.get('full_sequence', True))
        self.max_clips = int(
            cfg_data.get(
                'max_clips_train' if self.train_mode else 'max_clips_eval',
                cfg_data.get('num_clips_train' if self.train_mode else 'num_clips_eval', 0),
            )
        )

        self.frame_size = tuple(int(x) for x in cfg_data['frame_size'])
        self.max_persons = int(cfg_data['max_persons'])
        self.num_keypoints = int(cfg_data['num_keypoints'])
        self.decode_timeout_ms = int(cfg_data.get('decode_timeout_ms', 15000))
        self.decode_max_seconds = float(cfg_data.get('decode_max_seconds', 20.0))
        self.decode_max_frames = int(cfg_data.get('decode_max_frames', 3000))
        self.max_resample_tries = max(1, int(cfg_data.get('max_resample_tries', 8)))

        aug = cfg_data['augment']['train' if self.train_mode else 'eval']
        self.aug = AugCfg(float(aug['horizontal_flip_prob']), float(aug['color_jitter']))

        cache_mode = cfg_data.get('cache_mode', 'none')
        if cache_mode == 'ram':
            self.video_cache = RamCache(capacity=int(cfg_data.get('cache_size_video', 64)))
            self.pose_cache = RamCache(capacity=int(cfg_data.get('cache_size_pose', 512)))
        else:
            self.video_cache = NullCache()
            self.pose_cache = NullCache()

        mean = np.array(cfg_data['normalize']['mean'], dtype=np.float32)
        std = np.array(cfg_data['normalize']['std'], dtype=np.float32)
        self.mean = mean[None, None, None, :]
        self.std = std[None, None, None, :]
        self.current_epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _load_video(self, path: str) -> tuple[np.ndarray, tuple[int, int]]:
        x = self.video_cache.get(path)
        if x is not None:
            return x
        # Decode directly at training resolution to keep memory bounded in DDP workers.
        frames, orig_hw = decode_video_rgb(
            path,
            resize_to=self.frame_size,
            return_orig_hw=True,
            read_timeout_ms=self.decode_timeout_ms,
            max_decode_seconds=self.decode_max_seconds,
            max_frames=self.decode_max_frames,
        )
        payload = (frames, orig_hw)
        self.video_cache.set(path, payload)
        return payload

    def _load_pose(self, path: str) -> np.ndarray:
        x = self.pose_cache.get(path)
        if x is not None:
            return x

        arr = np.load(path)[self.key_name].astype(np.float32)  # [Tp, M, K, 3]
        t, m, k, c = arr.shape
        if c < 3:
            raise ValueError(f'Pose channels must be >=3, got {arr.shape} in {path}')
        arr = arr[:, :, :, :3]

        conf = arr[..., 2].mean(axis=(0, 2))
        order = np.argsort(-conf)
        arr = arr[:, order]

        if m >= self.max_persons:
            arr = arr[:, : self.max_persons]
        else:
            pad = np.zeros((t, self.max_persons - m, k, 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)

        if k != self.num_keypoints:
            if k > self.num_keypoints:
                arr = arr[:, :, : self.num_keypoints]
            else:
                pad = np.zeros((t, self.max_persons, self.num_keypoints - k, 3), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=2)

        self.pose_cache.set(path, arr)
        return arr

    def _sample(self, total_video_frames: int, rng: np.random.Generator) -> np.ndarray:
        return build_stream_clip_indices(
            total=total_video_frames,
            clip_len=self.clip_len,
            clip_step=self.clip_step,
            train_mode=self.train_mode,
            rng=rng,
            full_sequence=self.full_sequence,
            max_clips=self.max_clips,
        )

    def _rng_for_index(self, index: int) -> np.random.Generator:
        worker = get_worker_info()
        worker_seed = int(worker.seed if worker is not None else torch.initial_seed())
        # Mix in epoch explicitly so stochastic augmentation changes even with
        # persistent workers.
        mixed = (
            self.seed * 1000003
            + worker_seed * 9176
            + int(index) * 1009
            + int(self.current_epoch) * 15485863
        ) & ((1 << 63) - 1)
        return np.random.default_rng(mixed)

    def _build_item(self, rec: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        video, orig_hw = self._load_video(rec['video_path'])  # [Tv, H, W, 3], (H0, W0)
        pose = self._load_pose(rec['pose_path'])  # [Tp, M, K, 3]

        clip_vid_idx = self._sample(video.shape[0], rng)  # [Nc, L]
        clip_pose_idx = map_video_to_pose_indices(clip_vid_idx, video.shape[0], pose.shape[0])
        nc = int(clip_vid_idx.shape[0])

        clips = video[clip_vid_idx.reshape(-1)]

        if self.train_mode and self.aug.color_jitter > 0:
            clips = maybe_color_jitter(clips, self.aug.color_jitter, rng)

        clips = clips.reshape(nc, self.clip_len, self.frame_size[0], self.frame_size[1], 3)

        poses = pose[clip_pose_idx.reshape(-1)]
        # Pose coordinates are stored in original video scale. Align them to
        # resized frame space before mask construction.
        sy = float(self.frame_size[0]) / float(max(int(orig_hw[0]), 1))
        sx = float(self.frame_size[1]) / float(max(int(orig_hw[1]), 1))
        poses[..., 0] = np.clip(poses[..., 0] * sx, 0.0, float(self.frame_size[1] - 1))
        poses[..., 1] = np.clip(poses[..., 1] * sy, 0.0, float(self.frame_size[0] - 1))
        poses = poses.reshape(nc, self.clip_len, self.max_persons, self.num_keypoints, 3)

        if self.train_mode and self.aug.horizontal_flip_prob > 0 and rng.random() < self.aug.horizontal_flip_prob:
            clips = clips[:, :, :, ::-1, :].copy()
            poses[..., 0] = self.frame_size[1] - 1 - poses[..., 0]

        clips = clips.astype(np.float32) / 255.0
        clips = (clips - self.mean) / self.std
        clips = np.transpose(clips, (0, 1, 4, 2, 3))  # [Nc, L, 3, H, W]

        return {
            'sample_id': rec['sample_id'],
            'video_id': rec['video_id'],
            'video_path': rec['video_path'],
            'pose_path': rec['pose_path'],
            'label': int(rec['label']),
            'clips': torch.from_numpy(clips).float(),
            'poses': torch.from_numpy(poses).float(),
            'num_clips': int(nc),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = int(index)
        n = len(self.records)
        last_err: Exception | None = None

        for attempt in range(self.max_resample_tries):
            cur_index = (base_index + attempt) % n
            rec = self.records[cur_index]
            rng = self._rng_for_index(base_index + attempt * 1000003)
            try:
                return self._build_item(rec, rng)
            except Exception as e:  # pragma: no cover - corruption path is data-dependent
                last_err = e
                warnings.warn(
                    (
                        f'[{self.split}] skip broken sample '
                        f'idx={cur_index} video={rec.get("video_path", "?")} '
                        f'pose={rec.get("pose_path", "?")} err={type(e).__name__}: {e}'
                    ),
                    stacklevel=2,
                )

        raise RuntimeError(
            f'Failed to fetch sample after {self.max_resample_tries} attempts at base index {base_index}'
        ) from last_err
