from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from joint_cross_attention.src.data.collate import collate_joint
from joint_cross_attention.src.data.dataset import JointRWF2000Dataset
from joint_cross_attention.src.data.index_builder import build_video_pose_index, load_index


def ensure_index(cfg: dict[str, Any]) -> str:
    index_file = cfg['paths']['index_file']
    use_prebuilt = bool(cfg['data'].get('use_prebuilt_index', False))

    if Path(index_file).exists():
        return index_file

    if use_prebuilt:
        raise FileNotFoundError(f'Configured prebuilt index not found: {index_file}')

    build_video_pose_index(
        video_root=cfg['paths']['video_root'],
        pose_root=cfg['paths']['pose_root'],
        class_to_label=cfg['data']['class_to_label'],
        split_names={'train': cfg['data']['train_split'], 'val': cfg['data']['val_split']},
        key_name=cfg['data']['key_name'],
        out_file=index_file,
    )
    return index_file


def build_datasets(cfg: dict[str, Any]):
    index_file = ensure_index(cfg)
    train_rows = load_index(index_file, split='train')
    val_rows = load_index(index_file, split='val')

    train_ds = JointRWF2000Dataset(train_rows, cfg['data'], split='train', seed=int(cfg['project']['seed']))
    val_ds = JointRWF2000Dataset(val_rows, cfg['data'], split='val', seed=int(cfg['project']['seed']))
    return train_ds, val_ds


def build_dataloaders(cfg: dict[str, Any], train_ds, val_ds, distributed: bool):
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    workers = int(cfg['train']['num_workers'])
    persistent = bool(cfg['train']['persistent_workers']) and workers > 0
    prefetch = int(cfg['train']['prefetch_factor']) if workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg['train']['batch_size']),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=bool(cfg['train']['pin_memory']),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_joint,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg['train']['eval_batch_size']),
        shuffle=False,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=bool(cfg['train']['pin_memory']),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_joint,
        drop_last=False,
    )
    return train_loader, val_loader
