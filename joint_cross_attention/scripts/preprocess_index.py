#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter

from scene_decoupling.src.common.config import load_config, resolve_paths
from joint_cross_attention.src.data.index_builder import build_video_pose_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build the index for the joint model')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--stats-output', default='')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])
    rows = build_video_pose_index(
        video_root=cfg['paths']['video_root'],
        pose_root=cfg['paths']['pose_root'],
        class_to_label=cfg['data']['class_to_label'],
        split_names={'train': cfg['data']['train_split'], 'val': cfg['data']['val_split']},
        key_name=cfg['data']['key_name'],
        out_file=cfg['paths']['index_file'],
    )
    shape_cnt = Counter((x['pose_frames'], x['pose_persons'], x['pose_joints'], x['pose_channels']) for x in rows)
    split_cls_cnt = Counter((x['split'], x['class_name']) for x in rows)
    payload = {
        'index_file': cfg['paths']['index_file'],
        'num_samples': len(rows),
        'shape_distribution': {str(k): int(v) for k, v in shape_cnt.items()},
        'split_class_distribution': {f'{k[0]}/{k[1]}': int(v) for k, v in split_cls_cnt.items()},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    out = args.stats_output or str(Path(cfg['paths']['cache_root']) / 'dataset_stats.json')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


if __name__ == '__main__':
    main()
