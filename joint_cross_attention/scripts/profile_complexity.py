#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter
from statistics import mean, median
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from scene_decoupling.src.common.config import load_config, resolve_paths
from scene_decoupling.src.data.sampler import build_stream_clip_indices
from joint_cross_attention.src.models.joint_model import JointViolenceModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Profile joint-model params and FLOPs')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--output', default='')
    return p.parse_args()


def _count_params(model: torch.nn.Module, trainable_only: bool) -> int:
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return sum(int(p.numel()) for p in params)


def _load_val_clip_counts(index_file: Path, split: str, clip_len: int, clip_step: int, full_sequence: bool, max_clips_eval: int) -> list[int]:
    counts: list[int] = []
    with index_file.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if str(rec.get('split')) != split:
                continue
            total = int(rec.get('pose_frames', 0))
            idx = build_stream_clip_indices(
                total=total,
                clip_len=clip_len,
                clip_step=clip_step,
                train_mode=False,
                rng=None,  # ignored in eval/full-sequence path
                full_sequence=full_sequence,
                max_clips=max_clips_eval,
            )
            counts.append(int(idx.shape[0]))
    if not counts:
        raise ValueError(f'No records found for split={split} in {index_file}')
    return counts


def _profile_flops(model: torch.nn.Module, pose_shape: tuple[int, ...], clip_shape: tuple[int, ...], pose_eval_dtype: torch.dtype = torch.float32) -> int:
    pose_windows = torch.randn(*pose_shape, dtype=pose_eval_dtype)
    pose_window_valid = torch.ones((pose_shape[0], pose_shape[1], pose_shape[2]), dtype=torch.bool)
    video_clips = torch.randn(*clip_shape, dtype=torch.float32)
    video_poses = torch.randn((clip_shape[0], clip_shape[1], clip_shape[2], pose_shape[3], pose_shape[4], 3), dtype=torch.float32)
    clip_valid_mask = torch.ones((clip_shape[0], clip_shape[1]), dtype=torch.bool)

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False, with_flops=True) as prof:
            out = model(pose_windows, pose_window_valid, video_clips, video_poses, clip_valid_mask)
            _ = out['video_logit']
    return int(sum(int(evt.flops) for evt in prof.key_averages()))


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])

    model = JointViolenceModel(cfg['model'], cfg['data']).eval()
    total_params = _count_params(model, trainable_only=False)
    trainable_params = _count_params(model, trainable_only=True)

    pose_cfg = cfg['data']['pose_branch']
    ctx_cfg = cfg['data']['context_branch']
    index_file = Path(cfg['paths']['index_file'])

    eval_num_windows = int(pose_cfg['eval_num_windows'])
    window_size = int(pose_cfg['window_size'])
    max_persons = int(pose_cfg['max_persons'])
    num_keypoints = int(pose_cfg['num_keypoints'])
    pose_channels = 5 if bool(pose_cfg.get('include_velocity', True)) else 3

    clip_len = int(ctx_cfg['clip_len'])
    clip_step = int(ctx_cfg['clip_step'])
    frame_h, frame_w = int(ctx_cfg['frame_size'][0]), int(ctx_cfg['frame_size'][1])
    full_sequence = bool(ctx_cfg.get('full_sequence', True))
    max_clips_eval = int(ctx_cfg.get('max_clips_eval', 0))
    val_split = str(cfg['data']['val_split'])

    clip_counts = _load_val_clip_counts(
        index_file=index_file,
        split=val_split,
        clip_len=clip_len,
        clip_step=clip_step,
        full_sequence=full_sequence,
        max_clips_eval=max_clips_eval,
    )
    avg_num_clips = float(mean(clip_counts))
    med_num_clips = float(median(clip_counts))
    max_num_clips = int(max(clip_counts))
    if len(set(clip_counts)) == 1:
        profile_num_clips = clip_counts[0]
        profile_num_clips_mode = 'fixed_eval_clips'
    else:
        profile_num_clips = int(round(avg_num_clips))
        profile_num_clips_mode = 'rounded_mean_eval_clips'

    per_view_flops = _profile_flops(
        model=model,
        pose_shape=(1, 1, window_size, max_persons, num_keypoints, pose_channels),
        clip_shape=(1, 1, clip_len, 3, frame_h, frame_w),
    )
    per_video_flops = _profile_flops(
        model=model,
        pose_shape=(1, eval_num_windows, window_size, max_persons, num_keypoints, pose_channels),
        clip_shape=(1, profile_num_clips, clip_len, 3, frame_h, frame_w),
    )

    payload: dict[str, Any] = {
        'params': {
            'total': total_params,
            'total_m': total_params / 1e6,
            'trainable': trainable_params,
            'trainable_m': trainable_params / 1e6,
        },
        'flops': {
            'definition': 'PyTorch profiler forward FLOPs on CPU with batch_size=1',
            'per_view': {
                'pose_windows': 1,
                'context_clips': 1,
                'flops': per_view_flops,
                'gflops': per_view_flops / 1e9,
            },
            'per_video_eval': {
                'pose_windows': eval_num_windows,
                'context_clips_profiled': profile_num_clips,
                'context_clips_profiled_mode': profile_num_clips_mode,
                'flops': per_video_flops,
                'gflops': per_video_flops / 1e9,
            },
        },
        'eval_clip_stats': {
            'count_videos': len(clip_counts),
            'min': min(clip_counts),
            'mean': avg_num_clips,
            'median': med_num_clips,
            'max': max_num_clips,
            'histogram': dict(sorted(Counter(clip_counts).items())),
        },
        'input_signature': {
            'pose_window_size': window_size,
            'eval_num_windows': eval_num_windows,
            'clip_len': clip_len,
            'clip_step': clip_step,
            'frame_size': [frame_h, frame_w],
            'max_persons': max_persons,
            'num_keypoints': num_keypoints,
            'pose_channels': pose_channels,
        },
    }

    text = json.dumps(payload, ensure_ascii=True, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
