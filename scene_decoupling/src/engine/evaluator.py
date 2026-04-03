from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from scene_decoupling.src.common.distributed import all_reduce_sum, gather_objects, is_main_process

from .metrics import binary_metrics
from .threshold import scan_thresholds

try:
    from tqdm import tqdm
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
    criterion: torch.nn.Module | None = None,
    loss_weight_scale: float = 1.0,
    max_batches: int = 0,
    show_progress: bool = False,
) -> EvalResult:
    model.eval()
    preds: list[dict[str, Any]] = []

    running_loss = 0.0
    running_count = 0

    iterator = dataloader
    if show_progress and is_main_process() and tqdm is not None:
        total = min(len(dataloader), max_batches) if max_batches > 0 else len(dataloader)
        iterator = tqdm(
            dataloader,
            total=total,
            desc='Eval-Scene',
            leave=False,
            dynamic_ncols=True,
            miniters=1,
            file=sys.stdout,
        )

    for bidx, batch in enumerate(iterator):
        clips = batch['clips'].to(device, non_blocking=True)
        poses = batch['poses'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        clip_valid_mask = batch['clip_valid_mask'].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
            out = model(clips, poses, clip_valid_mask=clip_valid_mask)
            if criterion is not None:
                try:
                    loss_dict = criterion(
                        out['video_logit'],
                        labels,
                        f_evol=out.get('f_evol'),
                        f_scene=out.get('f_scene'),
                        mask_overlap=out.get('mask_overlap'),
                        fg_ratio=out.get('fg_ratio'),
                        weight_scale=loss_weight_scale,
                    )
                except TypeError:
                    loss_dict = criterion(out['video_logit'], labels)
                loss = loss_dict['loss']
                running_loss += float(loss.detach().item()) * int(labels.shape[0])
                running_count += int(labels.shape[0])

        probs = out['video_prob'].detach().float().cpu().numpy()
        y = labels.detach().float().cpu().numpy()
        fg = out['fg_ratio'].detach().float().cpu().numpy()
        overlap_t = out.get('mask_overlap')
        if overlap_t is None:
            overlap = np.zeros_like(fg, dtype=np.float32)
        else:
            overlap = overlap_t.detach().float().cpu().numpy()

        for i in range(len(probs)):
            preds.append(
                {
                    'sample_id': batch['sample_id'][i],
                    'video_id': batch['video_id'][i],
                    'video_path': batch['video_path'][i],
                    'pose_path': batch['pose_path'][i],
                    'label': int(y[i] > 0.5),
                    'prob': float(probs[i]),
                    'fg_ratio': float(fg[i]),
                    'mask_overlap': float(overlap[i]),
                }
            )

        if max_batches > 0 and (bidx + 1) >= max_batches:
            break

    merged = []
    for part in gather_objects(preds):
        merged.extend(part)

    uniq = {}
    for x in merged:
        uniq[x['sample_id']] = x
    merged = list(uniq.values())

    y_true = np.array([x['label'] for x in merged], dtype=np.int64)
    y_prob = np.array([x['prob'] for x in merged], dtype=np.float32)

    fixed = binary_metrics(y_true, y_prob, fixed_threshold)
    best_f1, records = scan_thresholds(
        y_true,
        y_prob,
        t_min=float(scan_cfg['scan_min']),
        t_max=float(scan_cfg['scan_max']),
        steps=int(scan_cfg['scan_steps']),
    )
    best_acc = max(records, key=lambda x: (x['acc'], x['f1']))
    best_balanced_acc = max(records, key=lambda x: (x['balanced_acc'], x['acc'], x['f1']))

    if criterion is not None:
        stat = torch.tensor([running_loss, running_count], dtype=torch.float64, device=device)
        stat = all_reduce_sum(stat)
        val_loss = float(stat[0].item() / max(1.0, stat[1].item()))
    else:
        val_loss = 0.0

    summary = {
        'num_videos': int(len(merged)),
        'val_loss': float(val_loss),
        'fixed': fixed,
        # Backward-compatible alias: `best` keeps the best-F1 record.
        'best': best_f1,
        'best_f1': best_f1,
        'best_acc': best_acc,
        'best_balanced_acc': best_balanced_acc,
        'fg_ratio_mean': float(np.mean([x['fg_ratio'] for x in merged])) if merged else 0.0,
        'mask_overlap_mean': float(np.mean([x['mask_overlap'] for x in merged])) if merged else 0.0,
    }
    return EvalResult(summary=summary, predictions=merged, threshold_records=records)
