from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.collate import rwf_collate
from src.data.index_builder import build_pose_index, load_index
from src.data.rwf_dataset import RWFPoseDataset


def ensure_index(cfg: dict[str, Any]) -> str:
    index_file = cfg["paths"]["index_file"]
    if Path(index_file).exists():
        return index_file

    build_pose_index(
        pose_root=cfg["paths"]["pose_root"],
        class_to_label=cfg["data"]["class_to_label"],
        split_names={
            "train": cfg["data"]["train_split"],
            "val": cfg["data"]["val_split"],
        },
        key_name=cfg["data"]["key_name"],
        out_file=index_file,
    )
    return index_file


def build_datasets(cfg: dict[str, Any]) -> tuple[RWFPoseDataset, RWFPoseDataset]:
    index_file = ensure_index(cfg)
    train_records = load_index(index_file, split="train")
    val_records = load_index(index_file, split="val")

    train_ds = RWFPoseDataset(train_records, cfg["data"], split="train", seed=int(cfg["project"]["seed"]))
    val_ds = RWFPoseDataset(val_records, cfg["data"], split="val", seed=int(cfg["project"]["seed"]))
    return train_ds, val_ds


def build_dataloaders(cfg: dict[str, Any], train_ds, val_ds, distributed: bool):
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    num_workers = int(cfg["train"]["num_workers"])
    persistent_workers = bool(cfg["train"]["persistent_workers"]) and num_workers > 0
    prefetch_factor = int(cfg["train"]["prefetch_factor"])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg["train"]["pin_memory"]),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=rwf_collate,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["eval_batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg["train"]["pin_memory"]),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=rwf_collate,
        drop_last=False,
    )

    return train_loader, val_loader
