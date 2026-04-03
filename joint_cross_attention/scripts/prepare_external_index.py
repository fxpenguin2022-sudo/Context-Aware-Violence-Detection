#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build a joint-model index for external datasets')
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--override', nargs='*', default=[])
    p.add_argument('--stats-output', default='')
    return p.parse_args()


def _load_cfg(config_paths: list[str], overrides: list[str]) -> dict[str, Any]:
    from scene_decoupling.src.common.config import load_config, resolve_paths
    return resolve_paths(load_config(config_paths, overrides), project_root=Path(__file__).resolve().parents[2])


def _inspect_pose(npz_path: Path, key_name: str) -> dict[str, int]:
    arr = np.load(npz_path)[key_name]
    if arr.ndim != 4:
        raise ValueError(f'Expected 4D pose in {npz_path}, got {arr.shape}')
    t, m, k, c = arr.shape
    return {
        'pose_frames': int(t),
        'pose_persons': int(m),
        'pose_joints': int(k),
        'pose_channels': int(c),
    }


def _build_violent_flow_index(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    video_root = Path(cfg['paths']['video_root'])
    pose_root = Path(cfg['paths']['pose_root'])
    key_name = cfg['data']['key_name']
    val_fold = int(cfg['data'].get('split_protocol', {}).get('val_fold', 5))
    rows: list[dict[str, Any]] = []
    class_map = {'Violence': ('Violence', 1), 'NonViolence': ('NonViolence', 0)}

    for fold_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        fold_name = fold_dir.name
        split = 'val' if int(fold_name) == val_fold else 'train'
        for raw_class, (class_name, label) in class_map.items():
            video_dir = video_root / fold_name / raw_class
            pose_dir = pose_root / fold_name / raw_class
            if not video_dir.exists() or not pose_dir.exists():
                continue
            for video_path in sorted(video_dir.glob('*.avi')):
                pose_path = pose_dir / f'{video_path.stem}.npz'
                if not pose_path.exists():
                    continue
                meta = _inspect_pose(pose_path, key_name)
                rows.append(
                    {
                        'sample_id': f'{split}/{class_name}/f{fold_name}_{video_path.stem}',
                        'video_id': f'f{fold_name}_{video_path.stem}',
                        'split': split,
                        'class_name': class_name,
                        'label': int(label),
                        'video_path': str(video_path.resolve()),
                        'pose_path': str(pose_path.resolve()),
                        **meta,
                    }
                )
    return rows


def _build_hockey_fight_index(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    video_root = Path(cfg['paths']['video_root'])
    pose_root = Path(cfg['paths']['pose_root'])
    key_name = cfg['data']['key_name']
    split_cfg = cfg['data'].get('split_protocol', {})
    val_ratio = float(split_cfg.get('val_ratio', 0.2))
    seed = int(split_cfg.get('seed', cfg['project']['seed']))
    rng = np.random.default_rng(seed)

    by_class: dict[str, list[tuple[Path, Path]]] = {'Violence': [], 'NonViolence': []}
    for video_path in sorted(video_root.glob('*.avi')):
        stem = video_path.stem.lower()
        if stem.startswith('fi'):
            class_name = 'Violence'
        elif stem.startswith('no'):
            class_name = 'NonViolence'
        else:
            continue
        pose_path = pose_root / f'{video_path.stem}.npz'
        if pose_path.exists():
            by_class[class_name].append((video_path, pose_path))

    rows: list[dict[str, Any]] = []
    for class_name, label in [('Violence', 1), ('NonViolence', 0)]:
        files = list(by_class[class_name])
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * val_ratio))) if files else 0
        val_keys = {str(v.resolve()) for v, _ in files[:n_val]}
        for video_path, pose_path in sorted(files):
            split = 'val' if str(video_path.resolve()) in val_keys else 'train'
            meta = _inspect_pose(pose_path, key_name)
            rows.append(
                {
                    'sample_id': f'{split}/{class_name}/{video_path.stem}',
                    'video_id': video_path.stem,
                    'split': split,
                    'class_name': class_name,
                    'label': int(label),
                    'video_path': str(video_path.resolve()),
                    'pose_path': str(pose_path.resolve()),
                    **meta,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config, args.override)
    dataset = str(cfg['data'].get('dataset_name', '')).lower()
    if dataset == 'violent_flow':
        rows = _build_violent_flow_index(cfg)
    elif dataset == 'hockey_fight':
        rows = _build_hockey_fight_index(cfg)
    else:
        raise ValueError(f'Unsupported dataset_name: {dataset}')

    out_file = Path(cfg['paths']['index_file'])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')

    shape_cnt = Counter((x['pose_frames'], x['pose_persons'], x['pose_joints'], x['pose_channels']) for x in rows)
    split_cls_cnt = Counter((x['split'], x['class_name']) for x in rows)
    payload = {
        'dataset': dataset,
        'index_file': str(out_file.resolve()),
        'num_samples': len(rows),
        'shape_distribution': {str(k): int(v) for k, v in shape_cnt.items()},
        'split_class_distribution': {f'{k[0]}/{k[1]}': int(v) for k, v in split_cls_cnt.items()},
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))

    stats_out = args.stats_output or str(Path(cfg['paths']['cache_root']) / f'{dataset}_joint_dataset_stats.json')
    Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_out).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
