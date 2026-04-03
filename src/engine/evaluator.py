from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.common.distributed import gather_objects, is_main_process

from .metrics import binary_metrics
from .threshold import scan_thresholds

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class EvalResult:
    summary: dict[str, Any]
    predictions: list[dict[str, Any]]
    threshold_records: list[dict[str, float]]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    fixed_threshold: float,
    scan_cfg: dict[str, Any],
    max_batches: int = 0,
    show_progress: bool = False,
) -> EvalResult:
    model.eval()
    preds: list[dict[str, Any]] = []

    iterator = dataloader
    if show_progress and is_main_process() and tqdm is not None:
        total = min(len(dataloader), max_batches) if max_batches > 0 else len(dataloader)
        iterator = tqdm(dataloader, total=total, desc="Eval", leave=False)

    for batch_idx, batch in enumerate(iterator):
        windows = batch["windows"].to(device, non_blocking=True)
        window_valid = batch["window_valid"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
            out = model(windows, window_valid)

        probs = out["video_prob"].detach().float().cpu().numpy()
        y = labels.detach().float().cpu().numpy()

        for i, vid in enumerate(batch["video_id"]):
            preds.append(
                {
                    "video_id": vid,
                    "pose_path": batch["pose_path"][i],
                    "label": int(y[i] > 0.5),
                    "prob": float(probs[i]),
                }
            )

        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    gathered = gather_objects(preds)
    merged: list[dict[str, Any]] = []
    for g in gathered:
        merged.extend(g)

    # Deduplicate in DDP case where sampler can produce overlap at the tail.
    # Use pose_path as unique key; video_id can collide across classes in RWF-2000.
    uniq = {}
    for rec in merged:
        uniq[rec["pose_path"]] = rec
    merged = list(uniq.values())

    y_true = np.array([x["label"] for x in merged], dtype=np.int64)
    y_prob = np.array([x["prob"] for x in merged], dtype=np.float32)

    fixed = binary_metrics(y_true, y_prob, fixed_threshold)
    best, records = scan_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        t_min=float(scan_cfg["scan_min"]),
        t_max=float(scan_cfg["scan_max"]),
        steps=int(scan_cfg["scan_steps"]),
    )

    summary = {
        "num_videos": int(len(merged)),
        "fixed": fixed,
        "best": best,
    }

    return EvalResult(summary=summary, predictions=merged, threshold_records=records)
