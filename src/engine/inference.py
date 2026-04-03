from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def infer_batch(
    model: torch.nn.Module,
    windows: torch.Tensor,
    window_valid: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    windows = windows.to(device)
    window_valid = window_valid.to(device)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
        out = model(windows, window_valid)

    return {
        "video_prob": out["video_prob"].detach().cpu(),
        "window_probs": out["window_probs"].detach().cpu(),
        "attn_weight": out["attn_weight"].detach().cpu(),
    }
