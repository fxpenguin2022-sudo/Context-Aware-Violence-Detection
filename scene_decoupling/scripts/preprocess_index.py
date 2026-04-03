#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter

from scene_decoupling.src.common.config import load_config, resolve_paths
from scene_decoupling.src.data.index_builder import build_video_pose_index, load_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build the scene-decoupling video+pose index')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--output', default='')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(load_config(args.config, args.override), project_root=Path(__file__).resolve().parents[2])

    if bool(cfg['data'].get('use_prebuilt_index', False)) and Path(cfg['paths']['index_file']).exists():
        rows = load_index(cfg['paths']['index_file'], split=None)
    else:
        rows = build_video_pose_index(
            video_root=cfg['paths']['video_root'],
            pose_root=cfg['paths']['pose_root'],
            class_to_label=cfg['data']['class_to_label'],
            split_names={'train': cfg['data']['train_split'], 'val': cfg['data']['val_split']},
            key_name=cfg['data']['key_name'],
            out_file=cfg['paths']['index_file'],
        )

    split_cls = Counter((r['split'], r['class_name']) for r in rows)
    payload = {
        'index_file': cfg['paths']['index_file'],
        'num_samples': len(rows),
        'split_class_distribution': {f'{k[0]}/{k[1]}': int(v) for k, v in split_cls.items()},
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))

    out = args.output or str(Path(cfg['paths']['cache_root']) / 'scene_decoupling_index_stats.json')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
