from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_metric': best_metric,
        'extra': extra or {},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f'.tmp.{os.getpid()}')
    torch.save(payload, str(tmp))
    os.replace(tmp, out)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = 'cpu',
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload['model'], strict=True)
    if optimizer is not None and payload.get('optimizer') is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None and payload.get('scheduler') is not None:
        scheduler.load_state_dict(payload['scheduler'])
    if scaler is not None and payload.get('scaler') is not None:
        scaler.load_state_dict(payload['scaler'])
    return payload
