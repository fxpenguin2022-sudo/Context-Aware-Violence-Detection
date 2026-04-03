from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def infer_batch(
    model: torch.nn.Module,
    clips: torch.Tensor,
    poses: torch.Tensor,
    clip_valid_mask: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    clips = clips.to(device)
    poses = poses.to(device)
    clip_valid_mask = clip_valid_mask.to(device)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
        out = model(clips, poses, clip_valid_mask=clip_valid_mask)

    return {
        'video_prob': out['video_prob'].detach().cpu(),
        'f_context': out['f_context'].detach().cpu(),
        'fg_ratio': out['fg_ratio'].detach().cpu(),
    }
